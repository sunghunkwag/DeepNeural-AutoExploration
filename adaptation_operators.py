"""Executable adaptation operators and operator genomes.

The operators here are bounded and CPU-runnable.  They adapt only from support
data, optional non-leaking memory summaries, and optional world-model
predictions.  Query/test labels are used only to score an already executed
operator in benchmark evaluation code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel
from maml_functional import functional_forward, maml_inner_loop


OPERATOR_NAMES = (
    "functional_maml",
    "reptile_update",
    "memory_gated_gradient_scaling",
    "world_model_lr_schedule",
)


@dataclass(frozen=True)
class OperatorGene:
    gene_id: str
    operator_name: str
    params: Dict[str, float | int | bool] = field(default_factory=dict)
    generation_created: int = 0
    parent_id: Optional[str] = None
    accepted: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "gene_id": self.gene_id,
            "operator_name": self.operator_name,
            "params": dict(self.params),
            "generation_created": self.generation_created,
            "parent_id": self.parent_id,
            "accepted": self.accepted,
        }


@dataclass
class OperatorTrace:
    gene_id: str
    operator_name: str
    query_loss_before: float
    query_loss_after: float
    improvement: float
    selected_lrs: List[float]
    memory_retrieval_count: int = 0
    memory_reward_signal: float = 0.0
    world_model_score: float = 0.0
    used_memory: bool = False
    used_world_model: bool = False
    params: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "gene_id": self.gene_id,
            "operator_name": self.operator_name,
            "query_loss_before": self.query_loss_before,
            "query_loss_after": self.query_loss_after,
            "improvement": self.improvement,
            "selected_lrs": list(self.selected_lrs),
            "memory_retrieval_count": self.memory_retrieval_count,
            "memory_reward_signal": self.memory_reward_signal,
            "world_model_score": self.world_model_score,
            "used_memory": self.used_memory,
            "used_world_model": self.used_world_model,
            "params": dict(self.params),
        }


@dataclass
class OperatorExecutionContext:
    model: nn.Module
    task: object
    task_embedding: torch.Tensor
    base_inner_lr: float
    inner_steps: int
    memory: Optional[EpisodicMemory] = None
    memory_k: int = 1
    world_model: Optional[WorldModel] = None
    recent_losses: torch.Tensor = field(default_factory=lambda: torch.zeros(4))


class AdaptationOperator:
    name: str

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        raise NotImplementedError

    def _before_loss(self, context: OperatorExecutionContext) -> float:
        with torch.no_grad():
            return float(F.mse_loss(context.model(context.task.query_x), context.task.query_y))

    def _trace(
        self,
        gene: OperatorGene,
        context: OperatorExecutionContext,
        adapted_params: Dict[str, torch.Tensor],
        selected_lrs: Sequence[float],
        *,
        memory_retrieval_count: int = 0,
        memory_reward_signal: float = 0.0,
        world_model_score: float = 0.0,
        used_memory: bool = False,
        used_world_model: bool = False,
    ) -> OperatorTrace:
        before = self._before_loss(context)
        with torch.no_grad():
            pred = functional_forward(context.model, adapted_params, context.task.query_x)
            after = float(F.mse_loss(pred, context.task.query_y))
        return OperatorTrace(
            gene_id=gene.gene_id,
            operator_name=gene.operator_name,
            query_loss_before=before,
            query_loss_after=after,
            improvement=before - after,
            selected_lrs=[float(x) for x in selected_lrs],
            memory_retrieval_count=memory_retrieval_count,
            memory_reward_signal=float(memory_reward_signal),
            world_model_score=float(world_model_score),
            used_memory=used_memory,
            used_world_model=used_world_model,
            params=dict(gene.params),
        )


def _clamp_lr(lr: float) -> float:
    return float(max(1e-4, min(0.2, lr)))


def _steps(gene: OperatorGene, context: OperatorExecutionContext) -> int:
    extra = int(gene.params.get("inner_steps_delta", 0))
    explicit = gene.params.get("inner_steps", None)
    if explicit is not None:
        return max(1, min(4, int(explicit)))
    return max(1, min(4, int(context.inner_steps) + extra))


def _base_lr(gene: OperatorGene, context: OperatorExecutionContext) -> float:
    return _clamp_lr(float(context.base_inner_lr) * float(gene.params.get("inner_lr_scale", 1.0)))


def variable_lr_inner_loop(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    lr_schedule: Sequence[float],
    *,
    first_order: bool = True,
) -> Dict[str, torch.Tensor]:
    adapted = {key: value.clone() for key, value in dict(model.named_parameters()).items()}
    for lr in lr_schedule:
        pred = functional_forward(model, adapted, support_x)
        loss = F.mse_loss(pred, support_y)
        grads = torch.autograd.grad(
            loss,
            list(adapted.values()),
            create_graph=not first_order,
            retain_graph=not first_order,
            allow_unused=True,
        )
        adapted = {
            key: value - float(lr) * (grad.detach() if first_order and grad is not None else (grad if grad is not None else torch.zeros_like(value)))
            for (key, value), grad in zip(adapted.items(), grads)
        }
    return adapted


class FunctionalMAMLOperator(AdaptationOperator):
    name = "functional_maml"

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        lr = _base_lr(gene, context)
        steps = _steps(gene, context)
        adapted = maml_inner_loop(context.model, dict(context.model.named_parameters()), context.task.support_x, context.task.support_y, lr, steps, first_order=True)
        return self._trace(gene, context, adapted, [lr] * steps)


class ReptileStyleOperator(AdaptationOperator):
    name = "reptile_update"

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        lr = _base_lr(gene, context)
        steps = _steps(gene, context)
        fast = maml_inner_loop(context.model, dict(context.model.named_parameters()), context.task.support_x, context.task.support_y, lr, steps, first_order=True)
        mix = float(gene.params.get("reptile_mix", 0.5))
        mix = max(0.0, min(1.0, mix))
        base = dict(context.model.named_parameters())
        reptile = {key: base[key] + mix * (fast[key] - base[key]) for key in base}
        return self._trace(gene, context, reptile, [lr] * steps)


def memory_reward_signal(retrieved: Sequence[Tuple[MemoryRecord, float]]) -> float:
    weighted = 0.0
    denom = 0.0
    for record, score in retrieved:
        if record.split == "test":
            continue
        weight = max(0.0, float(score))
        weighted += weight * float(record.reward)
        denom += weight
    if denom <= 1e-8:
        return 0.0
    return float(max(-1.0, min(1.0, weighted / denom)))


class MemoryGatedGradientScalingOperator(AdaptationOperator):
    name = "memory_gated_gradient_scaling"

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        base = _base_lr(gene, context)
        steps = _steps(gene, context)
        retrieved = context.memory.retrieve(context.task_embedding, k=context.memory_k) if context.memory is not None else []
        signal = memory_reward_signal(retrieved)
        strength = float(gene.params.get("memory_gate_strength", 0.6))
        gate = max(0.25, min(1.75, 1.0 + strength * signal))
        lr = _clamp_lr(base * gate)
        adapted = variable_lr_inner_loop(context.model, context.task.support_x, context.task.support_y, [lr] * steps)
        return self._trace(
            gene,
            context,
            adapted,
            [lr] * steps,
            memory_retrieval_count=len(retrieved),
            memory_reward_signal=signal,
            used_memory=True,
        )


def _memory_summary(embedding: torch.Tensor, memory: Optional[EpisodicMemory], k: int) -> torch.Tensor:
    if memory is None:
        return torch.zeros_like(embedding.float().flatten())
    retrieved = memory.retrieve(embedding, k=k)
    if not retrieved:
        return torch.zeros_like(embedding.float().flatten())
    weights = torch.tensor([max(0.0, score) for _, score in retrieved], dtype=torch.float32)
    if float(weights.sum()) <= 1e-8:
        weights = torch.ones(len(retrieved))
    embeddings = torch.stack([record.task_embedding.float().flatten()[: embedding.numel()] for record, _ in retrieved])
    return (embeddings * weights.reshape(-1, 1)).sum(dim=0) / weights.sum().clamp_min(1e-8)


class WorldModelPredictedLRScheduleOperator(AdaptationOperator):
    name = "world_model_lr_schedule"

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        base = _base_lr(gene, context)
        steps = _steps(gene, context)
        scales = [0.5, 0.8, 1.0, 1.3, 1.6]
        state = context.task_embedding.float().flatten()
        if context.world_model is not None:
            summary = _memory_summary(context.task_embedding, context.memory, context.memory_k)
            state = state + 0.5 * summary[: state.numel()]
        schedule: List[float] = []
        score_total = 0.0
        for step in range(steps):
            if context.world_model is None:
                chosen_scale = float(gene.params.get("schedule_scale", 1.0))
                score = 0.0
            else:
                scored: List[Tuple[float, float]] = []
                for scale in scales:
                    action = torch.zeros(context.world_model.action_dim)
                    action[0] = base * scale
                    if action.numel() > 1:
                        action[1] = float(step + 1)
                    if action.numel() > 2:
                        action[2] = float(gene.params.get("schedule_scale", 1.0))
                    with torch.no_grad():
                        pred = context.world_model.forward(state.reshape(1, -1), action.reshape(1, -1)).reshape(-1)
                    scored.append((float(pred[0]), scale))
                score, chosen_scale = max(scored, key=lambda item: item[0])
            score_total += float(score)
            schedule.append(_clamp_lr(base * chosen_scale))
        adapted = variable_lr_inner_loop(context.model, context.task.support_x, context.task.support_y, schedule)
        return self._trace(
            gene,
            context,
            adapted,
            schedule,
            world_model_score=score_total,
            used_world_model=context.world_model is not None,
        )


def default_operator_registry() -> Dict[str, AdaptationOperator]:
    return {
        "functional_maml": FunctionalMAMLOperator(),
        "reptile_update": ReptileStyleOperator(),
        "memory_gated_gradient_scaling": MemoryGatedGradientScalingOperator(),
        "world_model_lr_schedule": WorldModelPredictedLRScheduleOperator(),
    }


def operator_action_vector(gene: OperatorGene) -> torch.Tensor:
    type_index = OPERATOR_NAMES.index(gene.operator_name) if gene.operator_name in OPERATOR_NAMES else 0
    return torch.tensor(
        [
            float(gene.params.get("inner_lr_scale", 1.0)),
            float(gene.params.get("inner_steps", 1)) + float(gene.params.get("inner_steps_delta", 0)),
            float(gene.params.get("memory_gate_strength", 0.0)),
            float(gene.params.get("schedule_scale", 1.0)),
            float(type_index) / max(1, len(OPERATOR_NAMES) - 1),
            float(gene.params.get("reptile_mix", 0.0)),
        ],
        dtype=torch.float32,
    )


class OperatorGenome:
    def __init__(self, genes: Sequence[OperatorGene], active_gene_id: Optional[str] = None):
        if not genes:
            raise ValueError("operator genome requires at least one gene")
        self.genes: Dict[str, OperatorGene] = {gene.gene_id: gene for gene in genes}
        self.active_gene_id = active_gene_id or genes[0].gene_id
        self.accepted_gene_ids = {gene.gene_id for gene in genes if gene.accepted}
        self.rejected_genes: List[OperatorGene] = []

    @classmethod
    def default(cls) -> "OperatorGenome":
        genes = [
            OperatorGene("op0_functional_maml", "functional_maml", {"inner_lr_scale": 1.0, "inner_steps": 1}),
            OperatorGene("op0_reptile", "reptile_update", {"inner_lr_scale": 1.0, "inner_steps": 1, "reptile_mix": 0.5}),
            OperatorGene("op0_memory_gate", "memory_gated_gradient_scaling", {"inner_lr_scale": 1.0, "inner_steps": 1, "memory_gate_strength": 0.7}),
            OperatorGene("op0_world_schedule", "world_model_lr_schedule", {"inner_lr_scale": 1.0, "inner_steps": 1, "schedule_scale": 1.0}),
        ]
        return cls(genes, active_gene_id=genes[0].gene_id)

    def snapshot(self) -> Dict[str, object]:
        return {
            "genes": copy.deepcopy(self.genes),
            "active_gene_id": self.active_gene_id,
            "accepted_gene_ids": set(self.accepted_gene_ids),
            "rejected_genes": copy.deepcopy(self.rejected_genes),
        }

    def restore(self, snapshot: Dict[str, object]) -> None:
        self.genes = copy.deepcopy(snapshot["genes"])
        self.active_gene_id = str(snapshot["active_gene_id"])
        self.accepted_gene_ids = set(snapshot["accepted_gene_ids"])
        self.rejected_genes = copy.deepcopy(snapshot["rejected_genes"])

    def active_gene(self) -> OperatorGene:
        return self.genes[self.active_gene_id]

    def accepted_genes(self) -> List[OperatorGene]:
        return [self.genes[gene_id] for gene_id in sorted(self.accepted_gene_ids) if gene_id in self.genes]

    def accept(self, gene: OperatorGene) -> None:
        accepted = OperatorGene(gene.gene_id, gene.operator_name, dict(gene.params), gene.generation_created, gene.parent_id, True)
        self.genes[accepted.gene_id] = accepted
        self.accepted_gene_ids.add(accepted.gene_id)
        self.active_gene_id = accepted.gene_id

    def reject(self, gene: OperatorGene) -> None:
        rejected = OperatorGene(gene.gene_id, gene.operator_name, dict(gene.params), gene.generation_created, gene.parent_id, False)
        self.rejected_genes.append(rejected)

    def mutate_candidates(self, generation: int) -> List[OperatorGene]:
        parent = self.active_gene()
        base_scale = float(parent.params.get("inner_lr_scale", 1.0))
        base_steps = int(parent.params.get("inner_steps", 1))
        return [
            OperatorGene(f"g{generation}_functional_maml_lr", "functional_maml", {"inner_lr_scale": base_scale * 0.8, "inner_steps": base_steps}, generation, parent.gene_id, False),
            OperatorGene(f"g{generation}_reptile", "reptile_update", {"inner_lr_scale": base_scale, "inner_steps": base_steps, "reptile_mix": 0.65}, generation, parent.gene_id, False),
            OperatorGene(f"g{generation}_memory_gate", "memory_gated_gradient_scaling", {"inner_lr_scale": base_scale, "inner_steps": base_steps, "memory_gate_strength": 0.9}, generation, parent.gene_id, False),
            OperatorGene(f"g{generation}_world_schedule", "world_model_lr_schedule", {"inner_lr_scale": base_scale, "inner_steps": min(4, base_steps + 1), "schedule_scale": 1.2}, generation, parent.gene_id, False),
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "active_gene_id": self.active_gene_id,
            "accepted_gene_ids": sorted(self.accepted_gene_ids),
            "genes": [gene.to_dict() for gene in self.genes.values()],
            "rejected_genes": [gene.to_dict() for gene in self.rejected_genes],
        }


def execute_operator(gene: OperatorGene, context: OperatorExecutionContext, registry: Optional[Dict[str, AdaptationOperator]] = None) -> OperatorTrace:
    registry = registry or default_operator_registry()
    if gene.operator_name not in registry:
        raise ValueError(f"unknown adaptation operator: {gene.operator_name}")
    return registry[gene.operator_name].apply(gene, context)


def select_gene_with_world_model(genome: OperatorGenome, embedding: torch.Tensor, memory_summary: torch.Tensor, world_model: WorldModel) -> Tuple[OperatorGene, Dict[str, float]]:
    state = embedding.float().flatten() + 0.5 * memory_summary.float().flatten()[: embedding.numel()]
    scores: Dict[str, float] = {}
    for gene in genome.accepted_genes():
        action = operator_action_vector(gene)
        padded = torch.zeros(world_model.action_dim)
        padded[: min(padded.numel(), action.numel())] = action[: padded.numel()]
        with torch.no_grad():
            pred = world_model.forward(state.reshape(1, -1), padded.reshape(1, -1)).reshape(-1)
        scores[gene.gene_id] = float(pred[0])
    selected = max(genome.accepted_genes(), key=lambda gene: scores[gene.gene_id])
    return selected, scores
