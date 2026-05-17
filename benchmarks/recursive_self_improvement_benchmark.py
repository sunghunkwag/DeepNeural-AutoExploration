"""Bounded recursive self-improvement benchmark.

This benchmark keeps the loop CPU-runnable and intentionally bounded:
controllers learn from train traces, mutations are accepted only on validation
tasks, rejected mutations roll back, and held-out OOD test tasks are evaluated
only after each validation decision is frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from adaptation_operators import (
    OperatorExecutionContext,
    OperatorGene,
    OperatorGenome,
    execute_operator,
    operator_action_vector,
    select_gene_with_world_model,
)
from cognitive_core import EpisodicMemory, MemoryRecord, TaskInferenceModule, WorldModel
from experiment_manifest import build_manifest, validate_manifest, write_manifest
from learned_task_encoder import LearnedTaskEncoder, TaskConditionedRegressor, train_task_encoder
from maml_functional import functional_forward, maml_inner_loop
from model_based_controller import (
    ControllerTransition,
    LearnedMetaController,
    ModelBasedController,
    RandomController,
    UpdateAction,
    memory_conditioned_inner_lr,
)
from operator_mutation import OperatorMutation, OperatorMutationController, allowed_operator_mutations
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite, SyntheticTask, assert_disjoint_task_ids


def _stderr(values: Sequence[float]) -> float:
    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _fit(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    flat = tensor.detach().float().flatten()
    if flat.numel() >= dim:
        return flat[:dim]
    return torch.cat([flat, torch.zeros(dim - flat.numel())])


def _initial_state(seed: int) -> Dict[str, object]:
    return {
        "seed": seed,
        "inner_lr": 0.035,
        "inner_steps": 1,
        "memory_retrieval_k": 2,
        "use_task_encoder": True,
        "use_memory_conditioning": True,
        "use_world_model_controller": True,
        "use_learned_meta_controller": True,
        "planner_horizon": 1,
        "intrinsic_weight": 1.0,
        "uncertainty_weight": 1.0,
        "exploration_noise_scale": 0.0,
        "width_multiplier": 1.0,
        "recent_losses": [],
    }


def _state_for_no_controller(state: Dict[str, object]) -> Dict[str, object]:
    clone = copy.deepcopy(state)
    clone["use_world_model_controller"] = False
    clone["use_learned_meta_controller"] = False
    return clone


def _make_config(state: Dict[str, object]) -> DNAXConfig:
    width = float(state.get("width_multiplier", 1.0))
    hidden = [max(8, int(round(16 * width))), max(8, int(round(16 * width)))]
    return DNAXConfig(
        input_dim=int(state.get("input_dim", 1)),
        hidden_dims=hidden,
        output_dim=int(state.get("output_dim", 1)),
        dropout_rate=0.0,
        spectral_norm=False,
        verbose=False,
        seed=int(state.get("seed", 0)),
        inner_lr=float(state.get("inner_lr", 0.01)),
        inner_steps=int(state.get("inner_steps", 1)),
        use_second_order=False,
        replay_buffer_size=96,
        exploration_noise_scale=float(state.get("exploration_noise_scale", 0.0)),
    )


def _make_model(state: Dict[str, object]) -> DeepNeuralAutoExplorer:
    cfg = _make_config(state)
    torch.manual_seed(int(cfg.seed or 0) + int(round(float(state.get("width_multiplier", 1.0)) * 1000)))
    return DeepNeuralAutoExplorer(cfg)


def _loss_no_adapt(model: DeepNeuralAutoExplorer, task: SyntheticTask) -> float:
    with torch.no_grad():
        return float(F.mse_loss(model(task.query_x), task.query_y))


def _loss_maml(model: DeepNeuralAutoExplorer, task: SyntheticTask, inner_lr: float, inner_steps: int) -> float:
    steps = max(1, min(4, int(inner_steps)))
    adapted = maml_inner_loop(model, dict(model.named_parameters()), task.support_x, task.support_y, float(inner_lr), steps, first_order=True)
    with torch.no_grad():
        return float(F.mse_loss(functional_forward(model, adapted, task.query_x), task.query_y))


def _history_tensor(state: Dict[str, object]) -> torch.Tensor:
    history = [float(x) for x in state.get("recent_losses", [])][-4:]
    while len(history) < 4:
        history.insert(0, 0.0)
    return torch.tensor(history, dtype=torch.float32)


def _infer_task(state: Dict[str, object], encoder: LearnedTaskEncoder, task: SyntheticTask) -> torch.Tensor:
    if bool(state.get("use_task_encoder", True)):
        return encoder.infer(task.support_x, task.support_y)
    return TaskInferenceModule(embedding_dim=8).infer(task.support_x, task.support_y)


def _candidate_actions(state: Dict[str, object]) -> List[UpdateAction]:
    lr = float(state.get("inner_lr", 0.01))
    steps = max(1, int(state.get("inner_steps", 1)))
    k = max(1, int(state.get("memory_retrieval_k", 1)))
    horizon = max(1, int(state.get("planner_horizon", 1)))
    noise = float(state.get("exploration_noise_scale", 0.0))
    return [
        UpdateAction("fixed_default", 0.01, inner_steps=1, memory_k=1, planner_horizon=1),
        UpdateAction("state_current", lr, inner_steps=steps, memory_k=k, planner_horizon=horizon, exploration_noise_scale=noise),
        UpdateAction("state_conservative", lr * 0.6, inner_steps=steps, memory_k=max(1, k - 1), planner_horizon=horizon),
        UpdateAction("state_deeper", lr * 0.8, inner_steps=min(4, steps + 1), memory_k=k, planner_horizon=horizon + 1),
        UpdateAction("state_memory_heavy", lr * 0.9, inner_steps=steps, memory_k=k + 1, planner_horizon=horizon),
    ]


def _objective_lr_scale(state: Dict[str, object]) -> float:
    intrinsic = float(state.get("intrinsic_weight", 1.0))
    uncertainty = float(state.get("uncertainty_weight", 1.0))
    exploration = float(state.get("exploration_noise_scale", 0.0))
    scale = 1.0 + 0.04 * (intrinsic - 1.0) - 0.03 * (uncertainty - 1.0)
    scale *= 1.0 + min(0.25, max(0.0, exploration))
    return float(max(0.5, min(1.5, scale)))


def _evaluate_with_action(model: DeepNeuralAutoExplorer, task: SyntheticTask, state: Dict[str, object], action: Optional[UpdateAction]) -> float:
    lr = float(state.get("inner_lr", 0.01))
    steps = int(state.get("inner_steps", 1))
    if action is not None:
        lr = action.inner_lr
        steps = action.inner_steps
    lr = max(1e-4, min(0.2, lr * _objective_lr_scale(state)))
    return _loss_maml(model, task, lr, steps)


def _build_memory(train_tasks: Sequence[SyntheticTask], state: Dict[str, object], encoder: LearnedTaskEncoder) -> EpisodicMemory:
    memory = EpisodicMemory(capacity=96)
    model = _make_model(state)
    for task in train_tasks:
        z = encoder.infer(task.support_x, task.support_y)
        before = _loss_no_adapt(model, task)
        after = _loss_maml(model, task, float(state.get("inner_lr", 0.01)), int(state.get("inner_steps", 1)))
        memory.add(
            MemoryRecord(
                task.task_id,
                torch.cat([task.support_x.flatten(), task.support_y.flatten()])[:8],
                torch.tensor([float(state.get("inner_lr", 0.01)), float(state.get("inner_steps", 1))]),
                before - after,
                after,
                z,
                {"query_loss_before": before, "query_loss_after": after},
                split="train",
                contains_query_targets=False,
            )
        )
    return memory


def _shuffled_memory(memory: EpisodicMemory) -> EpisodicMemory:
    shuffled = EpisodicMemory(capacity=memory.capacity)
    for i, record in enumerate(reversed(memory.records)):
        shuffled.add(
            MemoryRecord(
                f"shuffled-{i}-{record.task_id}",
                record.observation,
                record.action,
                -abs(record.reward),
                record.loss + 1.0,
                record.task_embedding.roll(1),
                record.outcome,
                split=record.split,
                contains_query_targets=False,
            )
        )
    return shuffled


def _memory_summary(z: torch.Tensor, memory: Optional[EpisodicMemory], k: int) -> torch.Tensor:
    if memory is None:
        return torch.zeros(8)
    retrieved = memory.retrieve(z, k=k)
    if not retrieved:
        return torch.zeros(8)
    weights = torch.tensor([max(0.0, score) for _, score in retrieved], dtype=torch.float32)
    if float(weights.sum()) <= 1e-8:
        weights = torch.ones(len(retrieved))
    embeddings = torch.stack([_fit(record.task_embedding, 8) for record, _ in retrieved])
    rewards = torch.tensor([record.reward for record, _ in retrieved]).reshape(-1, 1)
    summary = (embeddings * weights.reshape(-1, 1)).sum(dim=0) / weights.sum().clamp_min(1e-8)
    summary[0] = summary[0] + 0.1 * float(rewards.mean())
    return summary


def _context(
    model: DeepNeuralAutoExplorer,
    task: SyntheticTask,
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    world_model: Optional[WorldModel],
) -> OperatorExecutionContext:
    z = _infer_task(state, encoder, task)
    return OperatorExecutionContext(
        model=model,
        task=task,
        task_embedding=z,
        base_inner_lr=float(state.get("inner_lr", 0.01)),
        inner_steps=int(state.get("inner_steps", 1)),
        memory=memory,
        memory_k=max(1, int(state.get("memory_retrieval_k", 1))),
        world_model=world_model,
        recent_losses=_history_tensor(state),
    )


def _gene_update_actions(genome: OperatorGenome) -> List[UpdateAction]:
    actions: List[UpdateAction] = []
    for gene in genome.accepted_genes():
        vec = operator_action_vector(gene)
        actions.append(
            UpdateAction(
                gene.gene_id,
                inner_lr=float(vec[0]),
                inner_steps=max(1, int(round(float(vec[1])))),
                memory_k=max(1, int(round(float(vec[2]))) if float(vec[2]) > 0 else 1),
                planner_horizon=max(1, int(round(float(vec[3])))),
                exploration_noise_scale=float(vec[4]),
            )
        )
    return actions


def _assert_operator_diversity(operator_scores: Dict[str, Sequence[float]], min_range: float = 1e-8) -> None:
    values = [float(v) for scores in operator_scores.values() for v in scores]
    if len(operator_scores) < 2 or not values:
        raise RuntimeError("operator benchmark needs at least two executable operators")
    means = [float(mean(scores)) for scores in operator_scores.values() if scores]
    if not means or max(means) - min(means) <= min_range:
        raise RuntimeError("all adaptation operators appear to be dead code")


def _collect_operator_transitions(
    train_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: EpisodicMemory,
    genome: OperatorGenome,
    world_model: Optional[WorldModel] = None,
) -> Tuple[List[ControllerTransition], Dict[str, List[float]]]:
    model = _make_model(state)
    transitions: List[ControllerTransition] = []
    scores_by_operator: Dict[str, List[float]] = {gene.gene_id: [] for gene in genome.accepted_genes()}
    for task in train_tasks:
        z = _infer_task(state, encoder, task)
        summary = _memory_summary(z, memory, max(1, int(state.get("memory_retrieval_k", 1))))
        history = _history_tensor(state)
        for gene in genome.accepted_genes():
            trace = execute_operator(gene, _context(model, task, state, encoder, memory, world_model))
            transitions.append(ControllerTransition(z, summary, history, operator_action_vector(gene), trace.improvement))
            scores_by_operator[gene.gene_id].append(trace.query_loss_after)
    _assert_operator_diversity(scores_by_operator)
    return transitions, scores_by_operator


def _train_world_model_from_operator_transitions(seed: int, transitions: Sequence[ControllerTransition]) -> Tuple[WorldModel, float, float]:
    wm = WorldModel(latent_dim=8, action_dim=6, hidden_dim=32, seed=seed)
    if not transitions:
        return wm, 0.0, 0.0
    latent = []
    actions = []
    next_latent = []
    for item in transitions:
        state = _fit(item.z_task, 8) + 0.5 * _fit(item.memory_summary, 8)
        nxt = state.clone()
        nxt[0] = float(item.actual_improvement)
        latent.append(state)
        actions.append(_fit(item.action_vector, 6))
        next_latent.append(nxt)
    batch = (torch.stack(latent), torch.stack(actions), torch.stack(next_latent))
    opt = torch.optim.Adam(wm.parameters(), lr=0.02)
    initial = float(wm.prediction_loss(*batch).detach())
    for _ in range(40):
        wm.train_step(batch, opt)
    final = float(wm.prediction_loss(*batch).detach())
    return wm, initial, final


def _evaluate_tasks_operator(
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    genome: OperatorGenome,
    *,
    meta_controller: Optional[LearnedMetaController] = None,
    world_model: Optional[WorldModel] = None,
    random_controller: Optional[RandomController] = None,
    selector_kind: str = "active",
    track_controller: bool = False,
) -> Tuple[List[float], List[float], List[str], List[Dict[str, object]]]:
    model = _make_model(state)
    losses: List[float] = []
    improvements: List[float] = []
    selected_ids: List[str] = []
    traces: List[Dict[str, object]] = []
    for task in tasks:
        z = _infer_task(state, encoder, task)
        active_memory = memory if bool(state.get("use_memory_conditioning", True)) else None
        gene = genome.active_gene()
        decision = None
        if selector_kind == "learned" and meta_controller is not None:
            decision = meta_controller.select_action(z, active_memory, memory_k=max(1, int(state.get("memory_retrieval_k", 1))), performance_history=_history_tensor(state))
            gene = genome.genes.get(decision.selected.name, gene)
        elif selector_kind == "world" and world_model is not None:
            summary = _memory_summary(z, active_memory, max(1, int(state.get("memory_retrieval_k", 1))))
            gene, _ = select_gene_with_world_model(genome, z, summary, world_model)
        elif selector_kind == "random" and random_controller is not None:
            action = random_controller.select_action()
            gene = genome.genes.get(action.name, gene)
        elif selector_kind == "fixed":
            gene = genome.accepted_genes()[0]
        trace = execute_operator(gene, _context(model, task, state, encoder, active_memory, world_model))
        if decision is not None and track_controller:
            meta_controller.log_actual_outcome(decision, trace.improvement)
        losses.append(trace.query_loss_after)
        improvements.append(trace.improvement)
        selected_ids.append(gene.gene_id)
        traces.append(trace.to_dict())
    return losses, improvements, selected_ids, traces


def _mean_loss_for_selector(
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    genome: OperatorGenome,
    *,
    meta_controller: Optional[LearnedMetaController],
    selector_kind: str,
    world_model: Optional[WorldModel] = None,
) -> float:
    if not tasks:
        return 0.0
    losses, _, _, _ = _evaluate_tasks_operator(
        tasks,
        state,
        encoder,
        memory,
        genome,
        meta_controller=meta_controller,
        selector_kind=selector_kind,
        world_model=world_model,
    )
    return float(mean(losses))


def _guarded_selector_kind(
    validation_tasks: Sequence[SyntheticTask],
    hidden_validation_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    genome: OperatorGenome,
    meta_controller: Optional[LearnedMetaController],
    world_model: Optional[WorldModel],
    *,
    required_relative_gain: float = 0.05,
) -> Tuple[str, Dict[str, float | str]]:
    guard_tasks = list(validation_tasks) + list(hidden_validation_tasks)
    if meta_controller is None or not guard_tasks:
        return "active", {"selected": "active", "reason": "no_guard_or_controller"}
    learned = _mean_loss_for_selector(
        guard_tasks,
        state,
        encoder,
        memory,
        genome,
        meta_controller=meta_controller,
        selector_kind="learned",
        world_model=world_model,
    )
    active = _mean_loss_for_selector(
        guard_tasks,
        state,
        encoder,
        memory,
        genome,
        meta_controller=meta_controller,
        selector_kind="active",
        world_model=world_model,
    )
    fixed = _mean_loss_for_selector(
        guard_tasks,
        state,
        encoder,
        memory,
        genome,
        meta_controller=meta_controller,
        selector_kind="fixed",
        world_model=world_model,
    )
    fallback_kind, fallback_loss = min(("active", active), ("fixed", fixed), key=lambda item: item[1])
    selected = "learned" if learned <= fallback_loss * (1.0 - required_relative_gain) else fallback_kind
    reason = "learned_passed_guard" if selected == "learned" else "fallback_lower_guard_loss"
    return selected, {
        "selected": selected,
        "reason": reason,
        "learned_guard_loss": learned,
        "active_guard_loss": active,
        "fixed_guard_loss": fixed,
        "fallback_kind": fallback_kind,
        "fallback_guard_loss": fallback_loss,
    }


def _evaluate_gene_on_tasks(
    gene: OperatorGene,
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: EpisodicMemory,
    world_model: Optional[WorldModel],
) -> List[float]:
    model = _make_model(state)
    return [execute_operator(gene, _context(model, task, state, encoder, memory, world_model)).query_loss_after for task in tasks]


def _validate_operator_candidate(
    genome: OperatorGenome,
    candidate: OperatorGene,
    validation_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: EpisodicMemory,
    world_model: Optional[WorldModel],
    *,
    generation: int,
    seed: int,
    hidden_validation_tasks: Sequence[SyntheticTask] | None = None,
    improvement_threshold: float = 1e-4,
) -> Dict[str, object]:
    hidden_validation_tasks = list(hidden_validation_tasks or validation_tasks)
    if not validation_tasks:
        raise ValueError("validation_tasks are required for operator mutation acceptance")
    if any(task.split == "test" for task in validation_tasks) or any(task.split == "test" for task in hidden_validation_tasks):
        raise ValueError("test tasks cannot be used for operator mutation acceptance")
    snapshot = genome.snapshot()
    baseline_gene = genome.active_gene()
    baseline_scores = _evaluate_gene_on_tasks(baseline_gene, validation_tasks, state, encoder, memory, world_model)
    candidate_scores = _evaluate_gene_on_tasks(candidate, validation_tasks, state, encoder, memory, world_model)
    hidden_baseline_scores = _evaluate_gene_on_tasks(baseline_gene, hidden_validation_tasks, state, encoder, memory, world_model)
    hidden_candidate_scores = _evaluate_gene_on_tasks(candidate, hidden_validation_tasks, state, encoder, memory, world_model)
    deltas = [b - c for b, c in zip(baseline_scores, candidate_scores)]
    hidden_deltas = [b - c for b, c in zip(hidden_baseline_scores, hidden_candidate_scores)]
    wins = sum(delta > improvement_threshold for delta in deltas)
    mean_improvement = float(mean(deltas))
    win_rate = float(wins / max(1, len(deltas)))
    hidden_mean_improvement = float(mean(hidden_deltas)) if hidden_deltas else 0.0
    hidden_win_rate = float(sum(delta > -improvement_threshold for delta in hidden_deltas) / max(1, len(hidden_deltas)))
    hidden_guard_regression = min(hidden_deltas) if hidden_deltas else 0.0
    if improvement_threshold < -1e8:
        accepted = True
    else:
        accepted = (
            mean_improvement > improvement_threshold
            and wins >= max(1, math.ceil(len(validation_tasks) * 0.6))
            and hidden_mean_improvement >= -improvement_threshold
            and hidden_win_rate >= 0.5
            and hidden_guard_regression >= -0.25
        )
    if accepted:
        genome.accept(candidate)
    else:
        genome.reject(candidate)
        genome.restore(snapshot)
    return {
        "candidate_name": candidate.gene_id,
        "candidate_operator": candidate.operator_name,
        "operator_type": "adaptation_operator",
        "operator_gene": candidate.to_dict(),
        "baseline_operator": baseline_gene.to_dict(),
        "baseline_scores": [float(x) for x in baseline_scores],
        "candidate_scores": [float(x) for x in candidate_scores],
        "hidden_baseline_scores": [float(x) for x in hidden_baseline_scores],
        "hidden_candidate_scores": [float(x) for x in hidden_candidate_scores],
        "mean_improvement": mean_improvement,
        "win_rate": win_rate,
        "hidden_mean_improvement": hidden_mean_improvement,
        "hidden_win_rate": hidden_win_rate,
        "hidden_guard_regression": hidden_guard_regression,
        "consistent_wins": wins,
        "accepted": accepted,
        "rollback": not accepted,
        "random_seed": seed,
        "split": "validation",
        "generation": generation,
        "updates": {"active_operator_gene": candidate.gene_id, "operator_name": candidate.operator_name},
        "runtime_effect_keys": ["adaptation_operator", "operator_gene"],
    }


def _collect_controller_transitions(
    train_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: EpisodicMemory,
    candidates: Sequence[UpdateAction],
) -> List[ControllerTransition]:
    model = _make_model(state)
    transitions: List[ControllerTransition] = []
    for task in train_tasks:
        z = _infer_task(state, encoder, task)
        summary = _memory_summary(z, memory, max(1, int(state.get("memory_retrieval_k", 1))))
        history = _history_tensor(state)
        before = _loss_no_adapt(model, task)
        for action in candidates:
            after = _evaluate_with_action(model, task, state, action)
            transitions.append(ControllerTransition(z, summary, history, action.vector(), before - after))
    return transitions


def _train_world_model_from_transitions(seed: int, transitions: Sequence[ControllerTransition]) -> Tuple[WorldModel, float, float]:
    wm = WorldModel(latent_dim=8, action_dim=6, hidden_dim=32, seed=seed)
    if not transitions:
        return wm, 0.0, 0.0
    latent = []
    actions = []
    next_latent = []
    for item in transitions:
        state = _fit(item.z_task, 8) + 0.5 * _fit(item.memory_summary, 8)
        nxt = state.clone()
        nxt[0] = float(item.actual_improvement)
        latent.append(state)
        actions.append(_fit(item.action_vector, 6))
        next_latent.append(nxt)
    batch = (torch.stack(latent), torch.stack(actions), torch.stack(next_latent))
    opt = torch.optim.Adam(wm.parameters(), lr=0.02)
    initial = float(wm.prediction_loss(*batch).detach())
    for _ in range(40):
        wm.train_step(batch, opt)
    final = float(wm.prediction_loss(*batch).detach())
    return wm, initial, final


class WrongWorldModel(WorldModel):
    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(latent_state)
        out[..., 0] = -action[..., 0] - 0.02 * action[..., 1]
        return out


def _evaluate_tasks(
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    *,
    meta_controller: Optional[LearnedMetaController] = None,
    world_controller: Optional[ModelBasedController] = None,
    random_controller: Optional[RandomController] = None,
    controller_kind: str = "none",
    track_controller: bool = False,
) -> Tuple[List[float], List[float], List[str]]:
    model = _make_model(state)
    losses: List[float] = []
    improvements: List[float] = []
    selected: List[str] = []
    memory_k = max(1, int(state.get("memory_retrieval_k", 1)))
    for task in tasks:
        before = _loss_no_adapt(model, task)
        z = _infer_task(state, encoder, task)
        action: Optional[UpdateAction] = None
        decision = None
        active_memory = memory if bool(state.get("use_memory_conditioning", True)) else None
        if active_memory is not None and controller_kind in {"none", "fixed"}:
            retrieved = active_memory.retrieve(z, k=memory_k)
            conditioned_lr = memory_conditioned_inner_lr(float(state.get("inner_lr", 0.01)), retrieved)
            action = UpdateAction("memory_conditioned", conditioned_lr, int(state.get("inner_steps", 1)), memory_k)
        if controller_kind == "learned" and bool(state.get("use_learned_meta_controller", True)) and meta_controller is not None:
            decision = meta_controller.select_action(z, active_memory, memory_k=memory_k, performance_history=_history_tensor(state))
            action = decision.selected
        elif controller_kind == "world" and bool(state.get("use_world_model_controller", True)) and world_controller is not None:
            decision = world_controller.select_action(z, active_memory, memory_k=memory_k)
            action = decision.selected
        elif controller_kind == "random" and random_controller is not None:
            action = random_controller.select_action()
        elif controller_kind == "fixed":
            action = UpdateAction("fixed_default", 0.01, inner_steps=1, memory_k=1)
        loss = _evaluate_with_action(model, task, state, action)
        actual = before - loss
        if decision is not None and track_controller:
            if controller_kind == "learned" and meta_controller is not None:
                meta_controller.log_actual_outcome(decision, actual)
            elif controller_kind == "world" and world_controller is not None:
                world_controller.log_actual_outcome(decision, actual)
        losses.append(loss)
        improvements.append(actual)
        selected.append(action.name if action is not None else "state_default")
    return losses, improvements, selected


def _encoder_decoder_loss(decoder: TaskConditionedRegressor, encoder: LearnedTaskEncoder, tasks: Sequence[SyntheticTask]) -> List[float]:
    losses: List[float] = []
    with torch.no_grad():
        for task in tasks:
            z = encoder.infer(task.support_x, task.support_y)
            pred = decoder(task.query_x, z)
            losses.append(float(F.mse_loss(pred, task.query_y)))
    return losses


def _evaluate_ablation_table(
    test_tasks: Sequence[SyntheticTask],
    val_tasks: Sequence[SyntheticTask],
    hidden_val_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    state_before_mutation: Dict[str, object],
    genome: OperatorGenome,
    genome_before_mutation: OperatorGenome,
    no_rollback_genome: OperatorGenome,
    encoder: LearnedTaskEncoder,
    decoder: TaskConditionedRegressor,
    memory: EpisodicMemory,
    meta_controller: LearnedMetaController,
    world_model: WorldModel,
    wrong_world_model: WorldModel,
    random_controller: RandomController,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, object]]:
    no_adapt_model = _make_model(_state_for_no_controller(state))
    no_adaptation = [_loss_no_adapt(no_adapt_model, t) for t in test_tasks]
    functional_genome = OperatorGenome([gene for gene in genome.accepted_genes() if gene.operator_name == "functional_maml"] or [genome.accepted_genes()[0]])
    memory_genes = [gene for gene in genome.accepted_genes() if gene.operator_name == "memory_gated_gradient_scaling"]
    memory_genome = OperatorGenome(memory_genes or [genome.active_gene()])
    functional, _, _, _ = _evaluate_tasks_operator(test_tasks, _state_for_no_controller(state), encoder, None, functional_genome, selector_kind="fixed")
    encoder_only = _encoder_decoder_loss(decoder, encoder, test_tasks)
    memory_only, _, _, _ = _evaluate_tasks_operator(test_tasks, _state_for_no_controller(state), encoder, memory, memory_genome, selector_kind="fixed")
    world_losses, _, world_selected, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, genome, world_model=world_model, selector_kind="world")
    learned_losses, _, learned_selected, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind="learned", track_controller=True)
    guarded_selector, selector_guard = _guarded_selector_kind(val_tasks, hidden_val_tasks, state, encoder, memory, genome, meta_controller, world_model)
    no_rollback_selector, no_rollback_guard = _guarded_selector_kind(val_tasks, hidden_val_tasks, state, encoder, memory, no_rollback_genome, meta_controller, world_model)
    no_mutation_selector, no_mutation_guard = _guarded_selector_kind(
        val_tasks,
        hidden_val_tasks,
        state_before_mutation,
        encoder,
        memory,
        genome_before_mutation,
        meta_controller,
        world_model,
    )
    random_losses, _, _, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, genome, random_controller=random_controller, selector_kind="random")
    no_rollback_losses, _, _, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, no_rollback_genome, meta_controller=meta_controller, selector_kind=no_rollback_selector)
    rollback_losses, _, _, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind=guarded_selector)
    full_losses = rollback_losses
    wrong_losses, _, wrong_selected, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, genome, world_model=wrong_world_model, selector_kind="world")
    shuffled_losses, _, _, _ = _evaluate_tasks_operator(test_tasks, state, encoder, _shuffled_memory(memory), genome, meta_controller=meta_controller, selector_kind=guarded_selector)
    no_mutation_losses, _, _, _ = _evaluate_tasks_operator(test_tasks, state_before_mutation, encoder, memory, genome_before_mutation, meta_controller=meta_controller, selector_kind=no_mutation_selector)
    val_full, _, _, _ = _evaluate_tasks_operator(val_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind=guarded_selector)

    ablations = {
        "no_adaptation": float(mean(no_adaptation)),
        "functional_maml_only": float(mean(functional)),
        "learned_task_encoder_only": float(mean(encoder_only)),
        "memory_conditioned_adaptation": float(mean(memory_only)),
        "world_model_controller": float(mean(world_losses)),
        "learned_meta_controller": float(mean(learned_losses)),
        "random_controller": float(mean(random_losses)),
        "fixed_default_controller": float(mean(functional)),
        "deliberately_wrong_controller": float(mean(wrong_losses)),
        "self_improvement_without_rollback": float(mean(no_rollback_losses)),
        "self_improvement_with_rollback": float(mean(rollback_losses)),
        "full_recursive_loop": float(mean(full_losses)),
        "full_recursive_loop_wrong_world_model": float(mean(wrong_losses)),
        "full_recursive_loop_shuffled_memory": float(mean(shuffled_losses)),
        "full_recursive_loop_no_mutation": float(mean(no_mutation_losses)),
    }
    selector_stats = {
        "world_selected": world_selected,
        "wrong_selected": wrong_selected,
        "learned_selected": learned_selected,
        "guarded_selector": selector_guard,
        "no_rollback_selector": no_rollback_guard,
        "no_mutation_selector": no_mutation_guard,
        "world_wrong_selection_delta": float(sum(a != b for a, b in zip(world_selected, wrong_selected)) / max(1, len(world_selected))),
        "world_wrong_operator_selection_delta": float(sum(a != b for a, b in zip(world_selected, wrong_selected)) / max(1, len(world_selected))),
    }
    metrics = {
        "adaptation_improvement": ablations["no_adaptation"] - ablations["functional_maml_only"],
        "ood_generalization_gap": ablations["full_recursive_loop"] - float(mean(val_full)),
        "validation_to_test_gap": ablations["full_recursive_loop"] - float(mean(val_full)),
        "controller_prediction_error": float(mean([e["prediction_error"] for e in meta_controller.prediction_errors])) if meta_controller.prediction_errors else 0.0,
        "memory_transfer_gain": ablations["functional_maml_only"] - ablations["memory_conditioned_adaptation"],
        "planning_success_rate": selector_stats["world_wrong_selection_delta"],
        "full_loop_delta_vs_baseline": ablations["no_adaptation"] - ablations["full_recursive_loop"],
        "learned_controller_vs_random_delta": ablations["random_controller"] - ablations["learned_meta_controller"],
        "operator_memory_damage": ablations["full_recursive_loop_shuffled_memory"] - ablations["full_recursive_loop"],
    }
    return ablations, metrics, selector_stats


def _mode_config(mode: str) -> Dict[str, int]:
    return {
        "smoke": {"generations": 2, "train": 6, "validation": 3, "hidden_validation": 2, "test": 3, "encoder_steps": 8, "controller_steps": 18},
        "quick": {"generations": 2, "train": 8, "validation": 4, "hidden_validation": 3, "test": 4, "encoder_steps": 16, "controller_steps": 30},
        "full": {"generations": 3, "train": 10, "validation": 5, "hidden_validation": 4, "test": 5, "encoder_steps": 24, "controller_steps": 45},
    }[mode]


def _run_seed(seed: int, mode: str) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _mode_config(mode)
    suite = ProceduralTaskSuite(seed=seed)
    hidden_suite = ProceduralTaskSuite(seed=seed + 10000)
    state = _initial_state(seed)
    all_train: List[SyntheticTask] = []
    all_val: List[SyntheticTask] = []
    all_hidden_val: List[SyntheticTask] = []
    all_test: List[SyntheticTask] = []
    mutation_log: List[Dict[str, object]] = []
    generation_summaries: List[Dict[str, object]] = []
    last_context: Dict[str, object] = {}
    genome = OperatorGenome.default()
    reuse_hits = 0
    reuse_total = 0
    accepted_before_generation: set[str] = set()

    for generation in range(cfg["generations"]):
        train_tasks = suite.sample_mixed_tasks("train", cfg["train"], ood=False)
        val_tasks = suite.sample_mixed_tasks("validation", cfg["validation"], ood=False)
        hidden_val_tasks = hidden_suite.sample_mixed_tasks("validation", cfg["hidden_validation"], ood=True)
        test_tasks = suite.sample_mixed_tasks("test", cfg["test"], ood=True)
        assert_disjoint_task_ids(train_tasks, val_tasks, hidden_val_tasks, test_tasks)
        all_train.extend(train_tasks)
        all_val.extend(val_tasks)
        all_hidden_val.extend(hidden_val_tasks)
        all_test.extend(test_tasks)

        encoder = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=seed + generation)
        decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
        encoder_result = train_task_encoder(encoder, decoder, train_tasks, val_tasks, steps=cfg["encoder_steps"], lr=0.01)
        memory = _build_memory(train_tasks, state, encoder)
        operator_transitions, operator_scores = _collect_operator_transitions(train_tasks, state, encoder, memory, genome)
        gene_actions = _gene_update_actions(genome)

        meta_controller = LearnedMetaController(gene_actions, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 10 + generation)
        meta_initial, meta_final = meta_controller.train_on_transitions(operator_transitions, steps=cfg["controller_steps"], lr=0.01)
        world_model, wm_initial, wm_final = _train_world_model_from_operator_transitions(seed + 20 + generation, operator_transitions)
        wrong_world_model = WrongWorldModel(8, 6)
        random_controller = RandomController(gene_actions, seed=seed + 30 + generation)

        state_before_mutation = copy.deepcopy(state)
        genome_before_mutation = copy.deepcopy(genome)
        no_rollback_genome = copy.deepcopy(genome_before_mutation)
        for candidate in genome.mutate_candidates(generation):
            decision = _validate_operator_candidate(
                genome,
                candidate,
                val_tasks,
                state,
                encoder,
                memory,
                world_model,
                generation=generation,
                seed=seed,
                hidden_validation_tasks=hidden_val_tasks,
            )
            mutation_log.append(decision)
            if not decision["accepted"]:
                no_rollback_genome.accept(candidate)

        post_gene_actions = _gene_update_actions(genome)
        meta_controller = LearnedMetaController(post_gene_actions, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 40 + generation)
        meta_initial_after, meta_final_after = meta_controller.train_on_transitions(
            _collect_operator_transitions(train_tasks, state, encoder, memory, genome, world_model)[0],
            steps=cfg["controller_steps"],
            lr=0.01,
        )
        random_controller = RandomController(post_gene_actions, seed=seed + 50 + generation)

        guarded_selector, selector_guard = _guarded_selector_kind(
            val_tasks,
            hidden_val_tasks,
            state,
            encoder,
            memory,
            genome,
            meta_controller,
            world_model,
        )
        val_losses, _, val_selected, _ = _evaluate_tasks_operator(val_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind=guarded_selector, world_model=world_model)
        train_default_losses, _, _, _ = _evaluate_tasks_operator(train_tasks, _state_for_no_controller(state), encoder, memory, genome, selector_kind="fixed")
        state["recent_losses"] = [
            float(mean(train_default_losses)),
            float(mean(val_losses)),
            float(sum(1 for m in mutation_log if m.get("accepted"))),
            float(sum(1 for m in mutation_log if m.get("rollback"))),
        ]

        test_losses, _, test_selected, _ = _evaluate_tasks_operator(test_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind=guarded_selector, world_model=world_model)
        for selected in val_selected + test_selected:
            if selected in accepted_before_generation:
                reuse_hits += 1
            if accepted_before_generation:
                reuse_total += 1
        accepted_before_generation = set(genome.accepted_gene_ids)
        generation_summaries.append(
            {
                "generation": generation,
                "encoder_train_initial": encoder_result.train_losses[0],
                "encoder_train_final": encoder_result.train_losses[-1],
                "encoder_validation_loss": encoder_result.validation_metrics["reconstruction_loss"],
                "meta_controller_loss_initial": meta_initial,
                "meta_controller_loss_final": meta_final_after,
                "world_model_loss_initial": wm_initial,
                "world_model_loss_final": wm_final,
                "validation_loss": float(mean(val_losses)),
                "frozen_ood_test_loss": float(mean(test_losses)),
                "selector_guard": selector_guard,
                "accepted_mutation_count": len([m for m in mutation_log if m.get("accepted")]),
                "rollback_count": len([m for m in mutation_log if m.get("rollback")]),
                "active_operator_gene": genome.active_gene_id,
                "operator_score_means": {key: float(mean(values)) for key, values in operator_scores.items()},
                "selected_operator_ids": test_selected,
            }
        )
        last_context = {
            "train": train_tasks,
            "validation": val_tasks,
            "hidden_validation": hidden_val_tasks,
            "test": test_tasks,
            "encoder": encoder,
            "decoder": decoder,
            "memory": memory,
            "genome": copy.deepcopy(genome),
            "genome_before_mutation": copy.deepcopy(genome_before_mutation),
            "no_rollback_genome": copy.deepcopy(no_rollback_genome),
            "meta_controller": meta_controller,
            "world_model": world_model,
            "wrong_world_model": wrong_world_model,
            "random_controller": random_controller,
            "state_before_mutation": state_before_mutation,
            "meta_loss_initial": meta_initial,
            "meta_loss_final": meta_final_after,
            "world_model_loss_final": wm_final,
            "reuse_hits": reuse_hits,
            "reuse_total": reuse_total,
        }

    ablations, metrics, selector_stats = _evaluate_ablation_table(
        last_context["test"],
        last_context["validation"],
        last_context["hidden_validation"],
        state,
        last_context["state_before_mutation"],
        last_context["genome"],
        last_context["genome_before_mutation"],
        last_context["no_rollback_genome"],
        last_context["encoder"],
        last_context["decoder"],
        last_context["memory"],
        last_context["meta_controller"],
        last_context["world_model"],
        last_context["wrong_world_model"],
        last_context["random_controller"],
    )
    accepted = [m for m in mutation_log if m.get("accepted")]
    rejected = [m for m in mutation_log if not m.get("accepted")]
    rollback_count = sum(1 for m in rejected if m.get("rollback"))
    first_test = float(generation_summaries[0]["frozen_ood_test_loss"]) if generation_summaries else 0.0
    last_test = float(generation_summaries[-1]["frozen_ood_test_loss"]) if generation_summaries else 0.0
    gen_denom = max(1, len(generation_summaries) - 1)
    world_selected = [str(x) for x in selector_stats.get("world_selected", [])]
    world_dist = {name: world_selected.count(name) for name in sorted(set(world_selected))}
    metrics.update(
        {
            "accepted_mutation_count": float(len(accepted)),
            "rejected_mutation_count": float(len(rejected)),
            "rollback_count": float(rollback_count),
            "mutation_win_rate": float(mean([float(m.get("win_rate", 0.0)) for m in mutation_log])) if mutation_log else 0.0,
            "predicted_vs_actual_improvement_error": metrics["controller_prediction_error"],
            "controller_call_count": float(last_context["meta_controller"].controller_call_count),
            "world_model_call_count": float(len(world_selected) * max(1, len(last_context["genome"].accepted_genes()))),
            "world_model_prediction_error": float(last_context["world_model_loss_final"]),
            "meta_controller_train_error": float(last_context["meta_loss_final"]),
            "improvement_velocity": float((first_test - last_test) / gen_denom),
            "operator_reuse_success": float(last_context["reuse_hits"] / max(1, last_context["reuse_total"])),
            "controller_prediction_error_reduction": float(last_context["meta_loss_initial"] - last_context["meta_loss_final"]),
            "ood_transfer_after_accepted_mutations": float(ablations["full_recursive_loop_no_mutation"] - ablations["full_recursive_loop"]),
            "accepted_mutation_quality": float(mean([float(m.get("mean_improvement", 0.0)) for m in accepted])) if accepted else 0.0,
            "accepted_operator_count": float(len(last_context["genome"].accepted_gene_ids)),
            "selector_guard_fallback_count": float(
                sum(1 for item in generation_summaries if item.get("selector_guard", {}).get("selected") != "learned")
            ),
        }
    )
    return {
        "seed": seed,
        "final_state": state,
        "metrics": metrics,
        "ablations": ablations,
        "selector_stats": selector_stats,
        "selected_action_distribution": dict(last_context["meta_controller"].selected_action_distribution),
        "world_selected_action_distribution": world_dist,
        "operator_genome": last_context["genome"].to_dict(),
        "mutation_log": mutation_log,
        "generation_summaries": generation_summaries,
        "tasks": {"train": all_train, "validation": all_val, "hidden_validation": all_hidden_val, "test": all_test},
    }


def _aggregate(per_seed: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metric_keys = per_seed[0]["metrics"].keys()
    aggregate: Dict[str, object] = {}
    for key in metric_keys:
        vals = [float(p["metrics"][key]) for p in per_seed]
        aggregate[key] = {"mean": float(mean(vals)), "stderr": _stderr(vals)}
    ablation_keys = per_seed[0]["ablations"].keys()
    aggregate["ablations"] = {
        key: {"mean": float(mean([p["ablations"][key] for p in per_seed])), "stderr": _stderr([p["ablations"][key] for p in per_seed])}
        for key in ablation_keys
    }
    full_vals = [float(p["ablations"]["full_recursive_loop"]) for p in per_seed]
    aggregate["seed_variance"] = {"mean": float(np.var(full_vals)), "stderr": 0.0}
    return aggregate


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    seeds = [seed] if mode == "smoke" else ([seed, seed + 1] if mode == "quick" else [seed + i for i in range(5)])
    per = [_run_seed(s, mode) for s in seeds]
    aggregate = _aggregate(per)
    all_train = [task for item in per for task in item["tasks"]["train"]]
    all_val = [task for item in per for task in item["tasks"]["validation"]]
    all_hidden_val = [task for item in per for task in item["tasks"]["hidden_validation"]]
    all_test = [task for item in per for task in item["tasks"]["test"]]
    mutation_log = [entry for item in per for entry in item["mutation_log"]]
    config = {
        "benchmark": "operator_level_recursive_self_improvement",
        "mode": mode,
        "seed": seed,
        "seeds": seeds,
        "mode_config": _mode_config(mode),
        "ablation_count": len(per[0]["ablations"]),
    }
    per_seed_payload = [
        {
            "seed": item["seed"],
            "metrics": item["metrics"],
            "ablations": item["ablations"],
            "final_state": item["final_state"],
            "selected_action_distribution": item["selected_action_distribution"],
            "world_selected_action_distribution": item["world_selected_action_distribution"],
            "operator_genome": item["operator_genome"],
            "generation_summaries": item["generation_summaries"],
        }
        for item in per
    ]
    result = {
        "mode": mode,
        "seeds": seeds,
        "config": config,
        "aggregate": aggregate,
        "per_seed": per_seed_payload,
        "task_ids": {
            "train": [task.task_id for task in all_train],
            "validation": [task.task_id for task in all_val],
            "hidden_validation": [task.task_id for task in all_hidden_val],
            "test": [task.task_id for task in all_test],
        },
        "mutation_log": mutation_log,
    }
    manifest = build_manifest(
        "python benchmarks/recursive_self_improvement_benchmark.py",
        seeds,
        all_train,
        all_val + all_hidden_val,
        all_test,
        config,
        {"aggregate": aggregate, "per_seed": per_seed_payload},
        mutation_log,
    )
    validate_manifest(manifest)
    if output is None:
        output = f"results/recursive_self_improvement_{mode}_seed{seed}.json"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.json")
    write_manifest(manifest, manifest_path)
    result["manifest_path"] = str(manifest_path)
    result["anti_cheat_checks_passed"] = manifest.anti_cheat_checks_passed
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded recursive self-improvement benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/recursive_self_improvement_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
