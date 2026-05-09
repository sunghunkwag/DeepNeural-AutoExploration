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
        input_dim=1,
        hidden_dims=hidden,
        output_dim=1,
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
    state: Dict[str, object],
    state_before_mutation: Dict[str, object],
    no_rollback_state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    decoder: TaskConditionedRegressor,
    memory: EpisodicMemory,
    meta_controller: LearnedMetaController,
    world_controller: ModelBasedController,
    wrong_world_controller: ModelBasedController,
    random_controller: RandomController,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, object]]:
    no_adapt_model = _make_model(_state_for_no_controller(state))
    no_adaptation = [_loss_no_adapt(no_adapt_model, t) for t in test_tasks]
    functional, _, _ = _evaluate_tasks(test_tasks, _state_for_no_controller(state), encoder, None, controller_kind="fixed")
    encoder_only = _encoder_decoder_loss(decoder, encoder, test_tasks)
    memory_only, _, _ = _evaluate_tasks(test_tasks, _state_for_no_controller(state), encoder, memory, controller_kind="none")
    world_losses, _, world_selected = _evaluate_tasks(test_tasks, state, encoder, memory, world_controller=world_controller, controller_kind="world", track_controller=True)
    learned_losses, _, learned_selected = _evaluate_tasks(test_tasks, state, encoder, memory, meta_controller=meta_controller, controller_kind="learned", track_controller=True)
    random_losses, _, _ = _evaluate_tasks(test_tasks, state, encoder, memory, random_controller=random_controller, controller_kind="random")
    no_rollback_losses, _, _ = _evaluate_tasks(test_tasks, no_rollback_state, encoder, memory, meta_controller=meta_controller, controller_kind="learned")
    rollback_losses, _, _ = _evaluate_tasks(test_tasks, state, encoder, memory, meta_controller=meta_controller, controller_kind="learned")
    full_losses = rollback_losses
    wrong_losses, _, wrong_selected = _evaluate_tasks(test_tasks, state, encoder, memory, world_controller=wrong_world_controller, controller_kind="world")
    shuffled_losses, _, _ = _evaluate_tasks(test_tasks, state, encoder, _shuffled_memory(memory), meta_controller=meta_controller, controller_kind="learned")
    no_mutation_losses, _, _ = _evaluate_tasks(test_tasks, state_before_mutation, encoder, memory, meta_controller=meta_controller, controller_kind="learned")
    val_full, _, _ = _evaluate_tasks(val_tasks, state, encoder, memory, meta_controller=meta_controller, controller_kind="learned")

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
        "world_wrong_selection_delta": float(sum(a != b for a, b in zip(world_selected, wrong_selected)) / max(1, len(world_selected))),
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
    }
    return ablations, metrics, selector_stats


def _mode_config(mode: str) -> Dict[str, int]:
    return {
        "smoke": {"generations": 2, "train": 6, "validation": 3, "test": 3, "encoder_steps": 8, "controller_steps": 18},
        "quick": {"generations": 2, "train": 8, "validation": 4, "test": 4, "encoder_steps": 16, "controller_steps": 30},
        "full": {"generations": 3, "train": 10, "validation": 5, "test": 5, "encoder_steps": 24, "controller_steps": 45},
    }[mode]


def _run_seed(seed: int, mode: str) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _mode_config(mode)
    suite = ProceduralTaskSuite(seed=seed)
    state = _initial_state(seed)
    all_train: List[SyntheticTask] = []
    all_val: List[SyntheticTask] = []
    all_test: List[SyntheticTask] = []
    mutation_log: List[Dict[str, object]] = []
    generation_summaries: List[Dict[str, object]] = []
    last_context: Dict[str, object] = {}

    for generation in range(cfg["generations"]):
        train_tasks = suite.sample_mixed_tasks("train", cfg["train"], ood=False)
        val_tasks = suite.sample_mixed_tasks("validation", cfg["validation"], ood=False)
        test_tasks = suite.sample_mixed_tasks("test", cfg["test"], ood=True)
        assert_disjoint_task_ids(train_tasks, val_tasks, test_tasks)
        all_train.extend(train_tasks)
        all_val.extend(val_tasks)
        all_test.extend(test_tasks)

        encoder = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=seed + generation)
        decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
        encoder_result = train_task_encoder(encoder, decoder, train_tasks, val_tasks, steps=cfg["encoder_steps"], lr=0.01)
        memory = _build_memory(train_tasks, state, encoder)
        candidates = _candidate_actions(state)
        transitions = _collect_controller_transitions(train_tasks, state, encoder, memory, candidates)

        meta_controller = LearnedMetaController(candidates, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 10 + generation)
        meta_initial, meta_final = meta_controller.train_on_transitions(transitions, steps=cfg["controller_steps"], lr=0.01)
        world_model, wm_initial, wm_final = _train_world_model_from_transitions(seed + 20 + generation, transitions)
        world_controller = ModelBasedController(world_model, candidates, latent_dim=8)
        wrong_world_controller = ModelBasedController(WrongWorldModel(8, 6), candidates, latent_dim=8)
        random_controller = RandomController(candidates, seed=seed + 30 + generation)

        state_before_mutation = copy.deepcopy(state)
        mutation_controller = OperatorMutationController(
            state,
            improvement_threshold=1e-4,
            min_consistent_wins=max(1, math.ceil(len(val_tasks) * 0.6)),
            random_seed=seed,
        )

        def evaluator(candidate_state: Dict[str, object], tasks: Sequence[SyntheticTask]) -> List[float]:
            losses, _, _ = _evaluate_tasks(
                tasks,
                candidate_state,
                encoder,
                memory,
                meta_controller=meta_controller,
                world_controller=world_controller,
                controller_kind="learned" if bool(candidate_state.get("use_learned_meta_controller", True)) else "world",
            )
            return losses

        proposed = allowed_operator_mutations(mutation_controller.state)
        no_rollback_state = copy.deepcopy(state_before_mutation)
        for mutation in proposed:
            decision = mutation_controller.evaluate(mutation, val_tasks, evaluator, split="validation")
            mutation_log.append(decision)
            if not decision["accepted"]:
                no_rollback_state.update(copy.deepcopy(mutation.updates))
        state = mutation_controller.state

        val_losses = evaluator(state, val_tasks)
        train_default_losses, _, _ = _evaluate_tasks(train_tasks, _state_for_no_controller(state), encoder, memory, controller_kind="fixed")
        state["recent_losses"] = [
            float(mean(train_default_losses)),
            float(mean(val_losses)),
            float(mutation_controller.accepted_mutation_count),
            float(mutation_controller.rollback_count),
        ]

        test_losses, _, _ = _evaluate_tasks(test_tasks, state, encoder, memory, meta_controller=meta_controller, controller_kind="learned")
        generation_summaries.append(
            {
                "generation": generation,
                "encoder_train_initial": encoder_result.train_losses[0],
                "encoder_train_final": encoder_result.train_losses[-1],
                "encoder_validation_loss": encoder_result.validation_metrics["reconstruction_loss"],
                "meta_controller_loss_initial": meta_initial,
                "meta_controller_loss_final": meta_final,
                "world_model_loss_initial": wm_initial,
                "world_model_loss_final": wm_final,
                "validation_loss": float(mean(val_losses)),
                "frozen_ood_test_loss": float(mean(test_losses)),
                "accepted_mutation_count": mutation_controller.accepted_mutation_count,
                "rollback_count": mutation_controller.rollback_count,
            }
        )
        last_context = {
            "train": train_tasks,
            "validation": val_tasks,
            "test": test_tasks,
            "encoder": encoder,
            "decoder": decoder,
            "memory": memory,
            "meta_controller": meta_controller,
            "world_controller": world_controller,
            "wrong_world_controller": wrong_world_controller,
            "random_controller": random_controller,
            "state_before_mutation": state_before_mutation,
            "no_rollback_state": no_rollback_state,
            "meta_loss_final": meta_final,
            "world_model_loss_final": wm_final,
        }

    ablations, metrics, selector_stats = _evaluate_ablation_table(
        last_context["test"],
        last_context["validation"],
        state,
        last_context["state_before_mutation"],
        last_context["no_rollback_state"],
        last_context["encoder"],
        last_context["decoder"],
        last_context["memory"],
        last_context["meta_controller"],
        last_context["world_controller"],
        last_context["wrong_world_controller"],
        last_context["random_controller"],
    )
    accepted = [m for m in mutation_log if m.get("accepted")]
    rejected = [m for m in mutation_log if not m.get("accepted")]
    rollback_count = sum(1 for m in rejected if m.get("rollback"))
    metrics.update(
        {
            "accepted_mutation_count": float(len(accepted)),
            "rejected_mutation_count": float(len(rejected)),
            "rollback_count": float(rollback_count),
            "mutation_win_rate": float(mean([float(m.get("win_rate", 0.0)) for m in mutation_log])) if mutation_log else 0.0,
            "predicted_vs_actual_improvement_error": metrics["controller_prediction_error"],
            "controller_call_count": float(last_context["meta_controller"].controller_call_count),
            "world_model_call_count": float(last_context["world_controller"].world_model_call_count),
            "world_model_prediction_error": float(last_context["world_model_loss_final"]),
            "meta_controller_train_error": float(last_context["meta_loss_final"]),
        }
    )
    return {
        "seed": seed,
        "final_state": state,
        "metrics": metrics,
        "ablations": ablations,
        "selector_stats": selector_stats,
        "selected_action_distribution": dict(last_context["meta_controller"].selected_action_distribution),
        "world_selected_action_distribution": dict(last_context["world_controller"].selected_action_distribution),
        "mutation_log": mutation_log,
        "generation_summaries": generation_summaries,
        "tasks": {"train": all_train, "validation": all_val, "test": all_test},
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
    all_test = [task for item in per for task in item["tasks"]["test"]]
    mutation_log = [entry for item in per for entry in item["mutation_log"]]
    config = {
        "benchmark": "recursive_self_improvement",
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
            "test": [task.task_id for task in all_test],
        },
        "mutation_log": mutation_log,
    }
    manifest = build_manifest(
        "python benchmarks/recursive_self_improvement_benchmark.py",
        seeds,
        all_train,
        all_val,
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
