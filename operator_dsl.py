"""Bounded operator-program DSL for code-level RSI experiments.

The DSL in this module is intentionally small and inspectable.  It does not
execute arbitrary generated Python.  Instead, it materializes bounded operator
programs as executable operator graphs composed from named primitives.  The
compiled operator is compatible with the existing AdaptationOperator interface
and adapts only from support/context data, optional non-test memory, and an
optional world model.  Query labels are used only by the evaluation trace after
adaptation has finished.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from adaptation_operators import AdaptationOperator, OperatorExecutionContext, OperatorGene, OperatorTrace
from arc_hdc_signature import HDC_DIMENSION, grid_pair_similarity
from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel
from maml_functional import functional_forward


GRADIENT_PRIMITIVES = (
    "maml_step",
    "support_proxy_step",
    "reptile_interpolation",
    "variable_lr_step",
    "gradient_norm_clipping",
    "gradient_noise_injection",
    "support_loss_weighting",
)
MEMORY_PRIMITIVES = (
    "memory_reward_gate",
    "memory_similarity_gate",
    "memory_retrieval_k",
    "memory_shuffled_control",
    "memory_disabled_control",
)
WORLD_MODEL_PRIMITIVES = (
    "world_model_lr_score",
    "world_model_operator_score",
    "predicted_improvement_gate",
    "wrong_world_model_control",
)
OBJECTIVE_PRIMITIVES = (
    "support_loss",
    "validation_proxy_loss",
    "prediction_error_reduction",
    "uncertainty_penalty",
    "novelty_bonus",
    "support_color_mapping",
    "support_local_pattern_mapping",
    "support_feature_lookup_mapping",
    "support_geometric_transform_mapping",
    "support_nested_ring_reversal_mapping",
    "support_translation_mapping",
    "support_exact_repair_search",
)
SCHEDULE_PRIMITIVES = (
    "constant_lr",
    "decayed_lr",
    "increasing_lr",
    "world_model_selected_lr",
    "memory_gated_lr",
)
ALL_PRIMITIVES = (
    GRADIENT_PRIMITIVES
    + MEMORY_PRIMITIVES
    + WORLD_MODEL_PRIMITIVES
    + OBJECTIVE_PRIMITIVES
    + SCHEDULE_PRIMITIVES
)


def _stable_hash(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("operator program produced a non-finite scalar")
    return float(max(lo, min(hi, value)))


def _fit(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    flat = tensor.detach().float().flatten()
    if flat.numel() >= dim:
        return flat[:dim]
    return torch.cat([flat, torch.zeros(dim - flat.numel())])


def supervised_task_loss(pred: torch.Tensor, target: torch.Tensor, metadata: Optional[Dict[str, object]] = None) -> torch.Tensor:
    """Return the task loss used by executable operator programs."""

    metadata = metadata or {}
    if metadata.get("target_type") == "classification":
        labels = target.reshape(-1).long()
        if pred.ndim != 2:
            raise ValueError("classification operator programs require [N, C] logits")
        return F.cross_entropy(pred, labels)
    return F.mse_loss(pred, target.float())


@dataclass(frozen=True)
class PrimitiveStep:
    name: str
    args: Dict[str, float | int | bool | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ALL_PRIMITIVES:
            raise ValueError(f"unknown DSL primitive: {self.name}")

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass(frozen=True)
class OperatorProgram:
    program_id: str
    primitive_sequence: Tuple[PrimitiveStep, ...]
    parameters: Dict[str, float | int | bool | str] = field(default_factory=dict)
    parent_program_id: Optional[str] = None
    generation_created: int = 0
    source_hash: str = ""
    accepted: bool = False
    status: str = "candidate"

    def __post_init__(self) -> None:
        if not self.primitive_sequence:
            raise ValueError("operator program requires at least one primitive")
        payload = {
            "program_id": self.program_id,
            "primitive_sequence": [step.to_dict() for step in self.primitive_sequence],
            "parameters": dict(self.parameters),
            "parent_program_id": self.parent_program_id,
            "generation_created": self.generation_created,
        }
        computed = _stable_hash(payload)
        if not self.source_hash:
            object.__setattr__(self, "source_hash", computed)

    def to_dict(self) -> Dict[str, object]:
        return {
            "program_id": self.program_id,
            "primitive_sequence": [step.to_dict() for step in self.primitive_sequence],
            "parameters": dict(self.parameters),
            "parent_program_id": self.parent_program_id,
            "generation_created": self.generation_created,
            "source_hash": self.source_hash,
            "accepted": self.accepted,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OperatorProgram":
        return cls(
            program_id=str(data["program_id"]),
            primitive_sequence=tuple(PrimitiveStep(str(step["name"]), dict(step.get("args", {}))) for step in data["primitive_sequence"]),  # type: ignore[index]
            parameters=dict(data.get("parameters", {})),  # type: ignore[arg-type]
            parent_program_id=data.get("parent_program_id") if data.get("parent_program_id") is None else str(data.get("parent_program_id")),
            generation_created=int(data.get("generation_created", 0)),
            source_hash=str(data.get("source_hash", "")),
            accepted=bool(data.get("accepted", False)),
            status=str(data.get("status", "candidate")),
        )

    def with_status(self, *, accepted: bool, status: str) -> "OperatorProgram":
        return OperatorProgram(
            self.program_id,
            self.primitive_sequence,
            dict(self.parameters),
            self.parent_program_id,
            self.generation_created,
            self.source_hash,
            accepted,
            status,
        )


@dataclass
class ProgramExecutionDetails:
    selected_lrs: List[float] = field(default_factory=list)
    memory_retrieval_count: int = 0
    memory_reward_signal: float = 0.0
    memory_similarity_signal: float = 0.0
    world_model_score: float = 0.0
    used_memory: bool = False
    used_world_model: bool = False
    support_proxy_loss: float = 0.0
    objective_signal: float = 0.0
    behavior_signature: List[float] = field(default_factory=list)
    primitive_effects: Dict[str, float] = field(default_factory=dict)
    prediction_override: Optional[torch.Tensor] = None


class ProgramGenome:
    """Accepted executable operator programs plus rollback-safe snapshots."""

    def __init__(self, programs: Sequence[OperatorProgram], active_program_id: Optional[str] = None):
        if not programs:
            raise ValueError("program genome requires at least one operator program")
        accepted = [program.with_status(accepted=True, status="accepted") for program in programs]
        self.programs: Dict[str, OperatorProgram] = {program.program_id: program for program in accepted}
        self.accepted_program_ids = {program.program_id for program in accepted}
        self.active_program_id = active_program_id or accepted[0].program_id

    @classmethod
    def default(cls) -> "ProgramGenome":
        return cls(default_operator_programs())

    def snapshot(self) -> Dict[str, object]:
        return {
            "programs": copy.deepcopy(self.programs),
            "accepted_program_ids": set(self.accepted_program_ids),
            "active_program_id": self.active_program_id,
        }

    def restore(self, snapshot: Dict[str, object]) -> None:
        self.programs = copy.deepcopy(snapshot["programs"])
        self.accepted_program_ids = set(snapshot["accepted_program_ids"])
        self.active_program_id = str(snapshot["active_program_id"])

    def active_program(self) -> OperatorProgram:
        return self.programs[self.active_program_id]

    def accepted_programs(self) -> List[OperatorProgram]:
        return [self.programs[program_id] for program_id in sorted(self.accepted_program_ids)]

    def accept(self, program: OperatorProgram) -> None:
        accepted = program.with_status(accepted=True, status="accepted")
        self.programs[accepted.program_id] = accepted
        self.accepted_program_ids.add(accepted.program_id)
        self.active_program_id = accepted.program_id

    def to_dict(self) -> Dict[str, object]:
        return {
            "active_program_id": self.active_program_id,
            "accepted_program_ids": sorted(self.accepted_program_ids),
            "programs": [program.to_dict() for program in self.programs.values()],
        }


class ProgramAdaptationOperator(AdaptationOperator):
    """Compiled executable form of an OperatorProgram."""

    def __init__(self, program: OperatorProgram):
        self.program = program
        self.name = f"dsl_program:{program.program_id}"

    def adapt_params(self, context: OperatorExecutionContext) -> Tuple[Dict[str, torch.Tensor], ProgramExecutionDetails]:
        state = _ProgramState(self.program, context)
        state.run()
        return state.adapted_params, state.details

    def _before_loss(self, context: OperatorExecutionContext) -> float:
        with torch.no_grad():
            pred = context.model(context.task.query_x)
            return float(supervised_task_loss(pred, context.task.query_y, getattr(context.task, "metadata", {})))

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        adapted, details = self.adapt_params(context)
        before = self._before_loss(context)
        with torch.no_grad():
            pred = details.prediction_override if details.prediction_override is not None else functional_forward(context.model, adapted, context.task.query_x)
            after = float(supervised_task_loss(pred, context.task.query_y, getattr(context.task, "metadata", {})))
        if not math.isfinite(after):
            raise FloatingPointError("operator program produced a non-finite query loss")
        return OperatorTrace(
            gene_id=gene.gene_id,
            operator_name=self.name,
            query_loss_before=before,
            query_loss_after=after,
            improvement=before - after,
            selected_lrs=details.selected_lrs,
            memory_retrieval_count=details.memory_retrieval_count,
            memory_reward_signal=details.memory_reward_signal,
            world_model_score=details.world_model_score,
            used_memory=details.used_memory,
            used_world_model=details.used_world_model,
            params={
                "program_id": self.program.program_id,
                "source_hash": self.program.source_hash,
                "primitive_sequence": [step.name for step in self.program.primitive_sequence],
                "support_proxy_loss": details.support_proxy_loss,
                "objective_signal": details.objective_signal,
                "memory_similarity_signal": details.memory_similarity_signal,
                "behavior_signature": list(details.behavior_signature),
                "primitive_effects": dict(details.primitive_effects),
            },
        )


class _ProgramState:
    def __init__(self, program: OperatorProgram, context: OperatorExecutionContext):
        self.program = program
        self.context = context
        self.base_params = {key: value for key, value in context.model.named_parameters()}
        self.adapted_params = {key: value.clone() for key, value in self.base_params.items()}
        self.base_lr = _clamp(float(context.base_inner_lr) * float(program.parameters.get("lr_scale", 1.0)), 1e-5, 0.3)
        self.current_lr = self.base_lr
        self.step_index = 0
        self.loss_weight = float(program.parameters.get("support_weight", 1.0))
        self.clip_norm: Optional[float] = None
        self.noise_scale = 0.0
        self.reptile_mix = float(program.parameters.get("reptile_mix", 0.5))
        self.memory_k = max(1, min(5, int(program.parameters.get("memory_k", context.memory_k))))
        self.memory_disabled = False
        self.memory_shuffled = False
        self.memory_reward_gate = 0.0
        self.memory_similarity_gate = 0.0
        self.world_score = 0.0
        self.wrong_world_model = False
        self.details = ProgramExecutionDetails()
        self.generator = torch.Generator().manual_seed(int(program.parameters.get("noise_seed", program.generation_created + 17)))

    def run(self) -> None:
        for step in self.program.primitive_sequence:
            getattr(self, f"_op_{step.name}")(step)
        if not self.details.selected_lrs and self.details.prediction_override is None:
            self._gradient_step(self.current_lr)
        self.details.behavior_signature = [
            float(sum(self.details.selected_lrs)),
            float(self.details.memory_reward_signal),
            float(self.details.memory_similarity_signal),
            float(self.details.world_model_score),
            float(self.details.support_proxy_loss),
            float(self.details.objective_signal),
            1.0 if self.details.prediction_override is not None else 0.0,
        ]

    def _retrieved_memory(self) -> List[Tuple[MemoryRecord, float]]:
        if self.memory_disabled or self.context.memory is None:
            return []
        retrieved = self.context.memory.retrieve(self.context.task_embedding, k=self.memory_k)
        if self.memory_shuffled:
            retrieved = [(record, -abs(score)) for record, score in reversed(retrieved)]
        return [(record, score) for record, score in retrieved if record.split != "test"]

    def _support_loss(self) -> torch.Tensor:
        pred = functional_forward(self.context.model, self.adapted_params, self.context.task.support_x)
        return supervised_task_loss(pred, self.context.task.support_y, getattr(self.context.task, "metadata", {})) * float(self.loss_weight)

    def _support_proxy_loss(self) -> torch.Tensor:
        sx = self.context.task.support_x
        sy = self.context.task.support_y
        if sx.shape[0] < 4:
            return self._support_loss()
        split = max(1, int(sx.shape[0] * 0.65))
        pred = functional_forward(self.context.model, self.adapted_params, sx[split:])
        return supervised_task_loss(pred, sy[split:], getattr(self.context.task, "metadata", {}))

    def _gradient_step(self, lr: float, *, proxy: bool = False) -> None:
        lr = _clamp(lr, 1e-5, 0.3)
        loss = self._support_proxy_loss() if proxy else self._support_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError("operator program produced non-finite support loss")
        grads = torch.autograd.grad(loss, list(self.adapted_params.values()), create_graph=False, retain_graph=False, allow_unused=True)
        clean_grads: List[torch.Tensor] = []
        for param, grad in zip(self.adapted_params.values(), grads):
            clean_grads.append(torch.zeros_like(param) if grad is None else grad.detach())
        if self.clip_norm is not None:
            total_norm = torch.sqrt(sum((g.float().pow(2).sum() for g in clean_grads))).clamp_min(1e-8)
            scale = min(1.0, float(self.clip_norm) / float(total_norm))
            clean_grads = [g * scale for g in clean_grads]
            self.details.primitive_effects["gradient_norm_clipping"] = float(scale)
        if self.noise_scale > 0.0:
            noisy: List[torch.Tensor] = []
            for grad in clean_grads:
                noise = torch.randn(grad.shape, generator=self.generator, dtype=grad.dtype, device=grad.device) * self.noise_scale
                noisy.append(grad + noise)
            clean_grads = noisy
        self.adapted_params = {key: value - lr * grad for (key, value), grad in zip(self.adapted_params.items(), clean_grads)}
        self.step_index += 1
        self.details.selected_lrs.append(float(lr))
        self.details.support_proxy_loss = float(loss.detach())

    def _op_maml_step(self, step: PrimitiveStep) -> None:
        self._gradient_step(self.current_lr)

    def _op_support_proxy_step(self, step: PrimitiveStep) -> None:
        self._gradient_step(self.current_lr, proxy=True)

    def _op_variable_lr_step(self, step: PrimitiveStep) -> None:
        decay = float(step.args.get("decay", self.program.parameters.get("decay", 0.75)))
        lr = self.current_lr * (decay ** self.step_index)
        self._gradient_step(lr)

    def _op_reptile_interpolation(self, step: PrimitiveStep) -> None:
        mix = _clamp(float(step.args.get("mix", self.reptile_mix)), 0.0, 1.0)
        self.adapted_params = {key: self.base_params[key] + mix * (self.adapted_params[key] - self.base_params[key]) for key in self.base_params}
        self.details.primitive_effects["reptile_interpolation"] = mix

    def _op_gradient_norm_clipping(self, step: PrimitiveStep) -> None:
        self.clip_norm = _clamp(float(step.args.get("max_norm", self.program.parameters.get("clip_norm", 2.0))), 0.05, 20.0)

    def _op_gradient_noise_injection(self, step: PrimitiveStep) -> None:
        self.noise_scale = _clamp(float(step.args.get("scale", self.program.parameters.get("noise_scale", 0.001))), 0.0, 0.05)
        self.details.primitive_effects["gradient_noise_injection"] = self.noise_scale

    def _op_support_loss_weighting(self, step: PrimitiveStep) -> None:
        self.loss_weight = _clamp(float(step.args.get("weight", self.program.parameters.get("support_weight", 1.0))), 0.2, 2.0)
        self.details.primitive_effects["support_loss_weighting"] = self.loss_weight

    def _op_memory_retrieval_k(self, step: PrimitiveStep) -> None:
        self.memory_k = max(1, min(5, int(step.args.get("k", self.program.parameters.get("memory_k", self.memory_k)))))
        self.details.primitive_effects["memory_retrieval_k"] = float(self.memory_k)

    def _op_memory_reward_gate(self, step: PrimitiveStep) -> None:
        retrieved = self._retrieved_memory()
        self.details.memory_retrieval_count = len(retrieved)
        self.details.used_memory = bool(retrieved)
        if not retrieved:
            self.memory_reward_gate = 0.0
            return
        denom = sum(max(0.0, float(score)) for _, score in retrieved)
        if denom <= 1e-8:
            self.memory_reward_gate = 0.0
        else:
            self.memory_reward_gate = sum(max(0.0, float(score)) * float(record.reward) for record, score in retrieved) / denom
        self.memory_reward_gate = _clamp(self.memory_reward_gate, -1.0, 1.0)
        self.details.memory_reward_signal = float(self.memory_reward_gate)

    def _op_memory_similarity_gate(self, step: PrimitiveStep) -> None:
        retrieved = self._retrieved_memory()
        self.details.memory_retrieval_count = max(self.details.memory_retrieval_count, len(retrieved))
        self.details.used_memory = bool(retrieved)
        self.memory_similarity_gate = 0.0 if not retrieved else _clamp(sum(float(score) for _, score in retrieved) / len(retrieved), -1.0, 1.0)
        self.details.memory_similarity_signal = float(self.memory_similarity_gate)

    def _op_memory_shuffled_control(self, step: PrimitiveStep) -> None:
        self.memory_shuffled = True
        self.details.primitive_effects["memory_shuffled_control"] = 1.0

    def _op_memory_disabled_control(self, step: PrimitiveStep) -> None:
        self.memory_disabled = True
        self.details.primitive_effects["memory_disabled_control"] = 1.0

    def _op_world_model_lr_score(self, step: PrimitiveStep) -> None:
        self.world_score = self._score_lr(self.current_lr)
        self.details.world_model_score = float(self.world_score)
        self.details.used_world_model = self.context.world_model is not None

    def _op_world_model_operator_score(self, step: PrimitiveStep) -> None:
        score = self._score_lr(self.current_lr * (1.0 + 0.1 * len(self.program.primitive_sequence)))
        self.world_score = 0.5 * self.world_score + 0.5 * score
        self.details.world_model_score = float(self.world_score)
        self.details.used_world_model = self.context.world_model is not None

    def _op_predicted_improvement_gate(self, step: PrimitiveStep) -> None:
        gate = 1.0 + _clamp(self.world_score, -0.5, 0.5)
        self.current_lr = _clamp(self.current_lr * gate, 1e-5, 0.3)
        self.details.primitive_effects["predicted_improvement_gate"] = float(gate)

    def _op_wrong_world_model_control(self, step: PrimitiveStep) -> None:
        self.wrong_world_model = True
        self.details.primitive_effects["wrong_world_model_control"] = 1.0

    def _op_support_loss(self, step: PrimitiveStep) -> None:
        loss = self._support_loss()
        self.details.support_proxy_loss = float(loss.detach())
        self.details.objective_signal += -float(loss.detach())

    def _op_validation_proxy_loss(self, step: PrimitiveStep) -> None:
        loss = self._support_proxy_loss()
        self.details.support_proxy_loss = float(loss.detach())
        self.details.objective_signal += -0.5 * float(loss.detach())

    def _op_prediction_error_reduction(self, step: PrimitiveStep) -> None:
        if self.context.world_model is None:
            return
        state = _fit(self.context.task_embedding, self.context.world_model.latent_dim)
        action = torch.zeros(self.context.world_model.action_dim)
        action[0] = self.current_lr
        with torch.no_grad():
            pred = self.context.world_model.forward(state.reshape(1, -1), action.reshape(1, -1)).reshape(-1)
        signal = -float((pred - state).abs().mean())
        self.details.objective_signal += signal
        self.details.used_world_model = True

    def _op_uncertainty_penalty(self, step: PrimitiveStep) -> None:
        pred = functional_forward(self.context.model, self.adapted_params, self.context.task.support_x)
        penalty = float(torch.var(pred.detach(), unbiased=False))
        self.current_lr = _clamp(self.current_lr / (1.0 + 0.05 * penalty), 1e-5, 0.3)
        self.details.objective_signal -= penalty
        self.details.primitive_effects["uncertainty_penalty"] = penalty

    def _op_novelty_bonus(self, step: PrimitiveStep) -> None:
        retrieved = self._retrieved_memory()
        novelty = 1.0 if not retrieved else 1.0 - max(-1.0, min(1.0, sum(float(score) for _, score in retrieved) / len(retrieved)))
        self.current_lr = _clamp(self.current_lr * (1.0 + 0.03 * novelty), 1e-5, 0.3)
        self.details.objective_signal += float(novelty)
        self.details.primitive_effects["novelty_bonus"] = float(novelty)

    def _op_support_color_mapping(self, step: PrimitiveStep) -> None:
        if getattr(self.context.task, "metadata", {}).get("target_type") != "classification":
            self.details.primitive_effects["support_color_mapping"] = 0.0
            return
        support_x = self.context.task.support_x.float().reshape(self.context.task.support_x.shape[0], -1)
        query_x = self.context.task.query_x.float().reshape(self.context.task.query_x.shape[0], -1)
        support_y = self.context.task.support_y.reshape(-1).long()
        if support_x.shape[1] < 5 or query_x.shape[1] < 5 or support_y.numel() == 0:
            self.details.primitive_effects["support_color_mapping"] = 0.0
            return
        input_colors = torch.clamp(torch.round(support_x[:, 4] * 9.0), 0, 9).long()
        query_colors = torch.clamp(torch.round(query_x[:, 4] * 9.0), 0, 9).long()
        global_counts = torch.bincount(torch.clamp(support_y, 0, 9), minlength=10).float()
        global_default = int(torch.argmax(global_counts).item())
        mapping: Dict[int, int] = {}
        confidence: Dict[int, float] = {}
        for color in range(10):
            mask = input_colors == color
            if bool(mask.any()):
                counts = torch.bincount(torch.clamp(support_y[mask], 0, 9), minlength=10).float()
                mapped = int(torch.argmax(counts).item())
                mapping[color] = mapped
                confidence[color] = float(counts.max().item() / counts.sum().clamp_min(1.0).item())
        predictions = [mapping.get(int(color), int(color) if 0 <= int(color) <= 9 else global_default) for color in query_colors]
        logits = torch.full((len(predictions), 10), -6.0, dtype=torch.float32, device=self.context.task.query_x.device)
        for row, label in enumerate(predictions):
            logits[row, int(label)] = 6.0 * confidence.get(int(query_colors[row]), 1.0)
        self.details.prediction_override = logits
        self.details.objective_signal += float(sum(confidence.values()) / max(1, len(confidence)))
        self.details.primitive_effects["support_color_mapping"] = float(len(mapping))

    def _op_support_local_pattern_mapping(self, step: PrimitiveStep) -> None:
        self._support_lookup_mapping("local_pattern")

    def _op_support_feature_lookup_mapping(self, step: PrimitiveStep) -> None:
        self._support_lookup_mapping(str(step.args.get("mode", "feature")))

    def _op_support_geometric_transform_mapping(self, step: PrimitiveStep) -> None:
        if getattr(self.context.task, "metadata", {}).get("target_type") != "classification":
            self.details.primitive_effects["support_geometric_transform_mapping"] = 0.0
            return

        metadata = getattr(self.context.task, "metadata", {})
        support_grids = self._feature_grids(
            self.context.task.support_x,
            metadata.get("support_shapes", []),
            metadata.get("support_offsets", []),
        )
        output_grids = metadata.get("support_expected_outputs", [])
        query_grids = self._feature_grids(
            self.context.task.query_x,
            metadata.get("query_shapes", []),
            metadata.get("query_offsets", []),
        )
        if not support_grids or len(support_grids) != len(output_grids) or not query_grids:
            self.details.primitive_effects["support_geometric_transform_mapping"] = 0.0
            return

        global_counts = torch.bincount(torch.clamp(self.context.task.support_y.reshape(-1).long(), 0, 9), minlength=10)
        global_default = int(torch.argmax(global_counts).item()) if int(global_counts.sum().item()) > 0 else 0
        candidates = []
        for transform_name, transform_fn in self._grid_transforms().items():
            votes: Dict[int, Dict[int, int]] = {}
            valid = True
            for input_grid, expected_grid in zip(support_grids, output_grids):
                transformed = transform_fn(input_grid)
                if self._grid_shape(transformed) != self._grid_shape(expected_grid):
                    valid = False
                    break
                for in_row, out_row in zip(transformed, expected_grid):
                    for in_color, out_color in zip(in_row, out_row):
                        src = int(max(0, min(9, int(in_color))))
                        dst = int(max(0, min(9, int(out_color))))
                        votes.setdefault(src, {})
                        votes[src][dst] = votes[src].get(dst, 0) + 1
            if not valid:
                continue
            mapping: Dict[int, int] = {}
            confidence: Dict[int, float] = {}
            for color, color_votes in votes.items():
                total = max(1, sum(color_votes.values()))
                label, count = max(color_votes.items(), key=lambda item: (item[1], -item[0]))
                mapping[color] = int(label)
                confidence[color] = float(count / total)

            cell_total = 0
            cell_correct = 0
            exact_total = 0
            exact_correct = 0
            for input_grid, expected_grid in zip(support_grids, output_grids):
                transformed = transform_fn(input_grid)
                predicted = self._apply_color_mapping(transformed, mapping, global_default)
                exact_total += 1
                exact_correct += 1 if predicted == expected_grid else 0
                for pred_row, exp_row in zip(predicted, expected_grid):
                    for pred_color, exp_color in zip(pred_row, exp_row):
                        cell_total += 1
                        cell_correct += 1 if int(pred_color) == int(exp_color) else 0
            cell_accuracy = cell_correct / max(1, cell_total)
            exact_accuracy = exact_correct / max(1, exact_total)
            mean_confidence = sum(confidence.values()) / max(1, len(confidence))
            candidates.append(
                (
                    exact_accuracy,
                    cell_accuracy,
                    mean_confidence,
                    -len(mapping),
                    transform_name,
                    transform_fn,
                    mapping,
                )
            )

        if not candidates:
            self.details.primitive_effects["support_geometric_transform_mapping"] = 0.0
            return

        best_exact, best_cell = max(candidates, key=lambda item: item[:2])[:2]
        tied = [item for item in candidates if item[0] == best_exact and item[1] == best_cell]
        hdc_ranked = []
        for exact_accuracy, cell_accuracy, mean_confidence, neg_mapping_size, transform_name, transform_fn, mapping in tied:
            hdc_scores = []
            for input_grid, expected_grid in zip(support_grids, output_grids):
                predicted = self._apply_color_mapping(transform_fn(input_grid), mapping, global_default)
                hdc_scores.append(grid_pair_similarity(predicted, expected_grid))
            hdc_similarity = sum(hdc_scores) / max(1, len(hdc_scores))
            hdc_ranked.append(
                (
                    exact_accuracy,
                    cell_accuracy,
                    hdc_similarity,
                    mean_confidence,
                    neg_mapping_size,
                    transform_name,
                    transform_fn,
                    mapping,
                )
            )
        exact_accuracy, cell_accuracy, hdc_similarity, mean_confidence, _, transform_name, transform_fn, mapping = max(hdc_ranked, key=lambda item: item[:6])
        predicted_query = [
            self._apply_color_mapping(transform_fn(grid), mapping, global_default)
            for grid in query_grids
        ]
        logits = self._logits_from_grids(predicted_query, self.context.task.query_x.device)
        self.details.prediction_override = logits
        self.details.objective_signal += float(exact_accuracy + cell_accuracy + hdc_similarity)
        self.details.primitive_effects["support_geometric_transform_mapping"] = float(cell_accuracy)
        self.details.primitive_effects["support_geometric_exact"] = float(exact_accuracy)
        self.details.primitive_effects["support_geometric_hdc_similarity"] = float(hdc_similarity)
        self.details.primitive_effects["support_geometric_hdc_dimension"] = float(HDC_DIMENSION)
        self.details.primitive_effects[f"support_geometric_transform:{transform_name}"] = float(mean_confidence)

    def _op_support_nested_ring_reversal_mapping(self, step: PrimitiveStep) -> None:
        if getattr(self.context.task, "metadata", {}).get("target_type") != "classification":
            self.details.primitive_effects["support_nested_ring_reversal_mapping"] = 0.0
            return

        metadata = getattr(self.context.task, "metadata", {})
        support_grids = self._feature_grids(
            self.context.task.support_x,
            metadata.get("support_shapes", []),
            metadata.get("support_offsets", []),
        )
        output_grids = metadata.get("support_expected_outputs", [])
        query_grids = self._feature_grids(
            self.context.task.query_x,
            metadata.get("query_shapes", []),
            metadata.get("query_offsets", []),
        )
        if not support_grids or len(support_grids) != len(output_grids) or not query_grids:
            self.details.primitive_effects["support_nested_ring_reversal_mapping"] = 0.0
            return

        cell_total = 0
        cell_correct = 0
        exact_total = 0
        exact_correct = 0
        hdc_scores = []
        for input_grid, expected_grid in zip(support_grids, output_grids):
            predicted = self._reverse_nested_ring_colors(input_grid)
            if self._grid_shape(predicted) != self._grid_shape(expected_grid):
                self.details.primitive_effects["support_nested_ring_reversal_mapping"] = 0.0
                return
            exact_total += 1
            exact_correct += 1 if predicted == expected_grid else 0
            hdc_scores.append(grid_pair_similarity(predicted, expected_grid))
            for pred_row, exp_row in zip(predicted, expected_grid):
                for pred_color, exp_color in zip(pred_row, exp_row):
                    cell_total += 1
                    cell_correct += 1 if int(pred_color) == int(exp_color) else 0

        predicted_query = [self._reverse_nested_ring_colors(grid) for grid in query_grids]
        logits = self._logits_from_grids(predicted_query, self.context.task.query_x.device)
        exact_accuracy = exact_correct / max(1, exact_total)
        cell_accuracy = cell_correct / max(1, cell_total)
        hdc_similarity = sum(hdc_scores) / max(1, len(hdc_scores))
        self.details.prediction_override = logits
        self.details.objective_signal += float(exact_accuracy + cell_accuracy + hdc_similarity)
        self.details.primitive_effects["support_nested_ring_reversal_mapping"] = float(cell_accuracy)
        self.details.primitive_effects["support_nested_ring_exact"] = float(exact_accuracy)
        self.details.primitive_effects["support_nested_ring_hdc_similarity"] = float(hdc_similarity)
        self.details.primitive_effects["support_nested_ring_hdc_dimension"] = float(HDC_DIMENSION)

    def _op_support_translation_mapping(self, step: PrimitiveStep) -> None:
        if getattr(self.context.task, "metadata", {}).get("target_type") != "classification":
            self.details.primitive_effects["support_translation_mapping"] = 0.0
            return

        metadata = getattr(self.context.task, "metadata", {})
        support_grids = self._feature_grids(
            self.context.task.support_x,
            metadata.get("support_shapes", []),
            metadata.get("support_offsets", []),
        )
        output_grids = metadata.get("support_expected_outputs", [])
        query_grids = self._feature_grids(
            self.context.task.query_x,
            metadata.get("query_shapes", []),
            metadata.get("query_offsets", []),
        )
        if not support_grids or len(support_grids) != len(output_grids) or not query_grids:
            self.details.primitive_effects["support_translation_mapping"] = 0.0
            return

        candidates = []
        for mode in ("drop", "clamp"):
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    cell_total = 0
                    cell_correct = 0
                    exact_total = 0
                    exact_correct = 0
                    for input_grid, expected_grid in zip(support_grids, output_grids):
                        background = self._most_common_color(input_grid)
                        predicted = self._translate_non_background(input_grid, dr, dc, background, mode=mode)
                        if self._grid_shape(predicted) != self._grid_shape(expected_grid):
                            continue
                        exact_total += 1
                        exact_correct += 1 if predicted == expected_grid else 0
                        for pred_row, exp_row in zip(predicted, expected_grid):
                            for pred_color, exp_color in zip(pred_row, exp_row):
                                cell_total += 1
                                cell_correct += 1 if int(pred_color) == int(exp_color) else 0
                    if cell_total == 0:
                        continue
                    candidates.append((exact_correct / max(1, exact_total), cell_correct / cell_total, dr, dc, mode))

        if not candidates:
            self.details.primitive_effects["support_translation_mapping"] = 0.0
            return

        best_exact, best_cell = max(candidates, key=lambda item: item[:2])[:2]
        tied = [item for item in candidates if item[0] == best_exact and item[1] == best_cell]
        hdc_ranked = []
        for exact_accuracy, cell_accuracy, dr, dc, mode in tied:
            hdc_scores = []
            for input_grid, expected_grid in zip(support_grids, output_grids):
                background = self._most_common_color(input_grid)
                predicted = self._translate_non_background(input_grid, dr, dc, background, mode=mode)
                hdc_scores.append(grid_pair_similarity(predicted, expected_grid))
            hdc_ranked.append((exact_accuracy, cell_accuracy, sum(hdc_scores) / max(1, len(hdc_scores)), dr, dc, mode))
        exact_accuracy, cell_accuracy, hdc_similarity, dr, dc, mode = max(hdc_ranked, key=lambda item: item[:3])

        predicted_query = [
            self._translate_non_background(grid, dr, dc, self._most_common_color(grid), mode=mode)
            for grid in query_grids
        ]
        logits = self._logits_from_grids(predicted_query, self.context.task.query_x.device)
        self.details.prediction_override = logits
        self.details.objective_signal += float(exact_accuracy + cell_accuracy + hdc_similarity)
        self.details.primitive_effects["support_translation_mapping"] = float(cell_accuracy)
        self.details.primitive_effects["support_translation_exact"] = float(exact_accuracy)
        self.details.primitive_effects["support_translation_hdc_similarity"] = float(hdc_similarity)
        self.details.primitive_effects["support_translation_hdc_dimension"] = float(HDC_DIMENSION)
        self.details.primitive_effects["support_translation_dr"] = float(dr)
        self.details.primitive_effects["support_translation_dc"] = float(dc)
        self.details.primitive_effects[f"support_translation_mode:{mode}"] = 1.0

    def _op_support_exact_repair_search(self, step: PrimitiveStep) -> None:
        if getattr(self.context.task, "metadata", {}).get("target_type") != "classification":
            self.details.primitive_effects["support_exact_repair_search"] = 0.0
            return

        metadata = getattr(self.context.task, "metadata", {})
        support_grids = self._feature_grids(
            self.context.task.support_x,
            metadata.get("support_shapes", []),
            metadata.get("support_offsets", []),
        )
        output_grids = metadata.get("support_expected_outputs", [])
        query_grids = self._feature_grids(
            self.context.task.query_x,
            metadata.get("query_shapes", []),
            metadata.get("query_offsets", []),
        )
        if not support_grids or len(support_grids) != len(output_grids) or not query_grids:
            self.details.primitive_effects["support_exact_repair_search"] = 0.0
            return

        candidates = self._exact_repair_candidates(support_grids, output_grids, query_grids)
        if not candidates:
            self.details.primitive_effects["support_exact_repair_search"] = 0.0
            return

        scored = []
        for name, support_predicted, query_predicted in candidates:
            exact_accuracy, cell_accuracy = self._grid_prediction_scores(support_predicted, output_grids)
            scored.append((exact_accuracy, cell_accuracy, name, support_predicted, query_predicted))

        best_exact, best_cell = max(scored, key=lambda item: item[:2])[:2]
        tied = [item for item in scored if item[0] == best_exact and item[1] == best_cell]
        ranked = []
        for exact_accuracy, cell_accuracy, name, support_predicted, query_predicted in tied:
            hdc_similarity = sum(
                grid_pair_similarity(predicted, expected)
                for predicted, expected in zip(support_predicted, output_grids)
            ) / max(1, len(output_grids))
            ranked.append((exact_accuracy, cell_accuracy, hdc_similarity, name, query_predicted))

        exact_accuracy, cell_accuracy, hdc_similarity, name, predicted_query = max(ranked, key=lambda item: item[:4])
        logits = self._logits_from_grids(predicted_query, self.context.task.query_x.device)
        self.details.prediction_override = logits
        self.details.objective_signal += float(2.0 * exact_accuracy + cell_accuracy + hdc_similarity)
        self.details.primitive_effects["support_exact_repair_search"] = float(cell_accuracy)
        self.details.primitive_effects["support_exact_repair_exact"] = float(exact_accuracy)
        self.details.primitive_effects["support_exact_repair_hdc_similarity"] = float(hdc_similarity)
        self.details.primitive_effects["support_exact_repair_hdc_dimension"] = float(HDC_DIMENSION)
        self.details.primitive_effects[f"support_exact_repair_method:{name}"] = 1.0

    def _support_lookup_mapping(self, mode: str) -> None:
        if getattr(self.context.task, "metadata", {}).get("target_type") != "classification":
            self.details.primitive_effects[f"support_lookup_{mode}"] = 0.0
            return
        support_x = self.context.task.support_x.float().reshape(self.context.task.support_x.shape[0], -1)
        query_x = self.context.task.query_x.float().reshape(self.context.task.query_x.shape[0], -1)
        support_y = self.context.task.support_y.reshape(-1).long()
        if support_x.shape[1] < 10 or query_x.shape[1] < 10 or support_y.numel() == 0:
            self.details.primitive_effects[f"support_lookup_{mode}"] = 0.0
            return

        def _color(row: torch.Tensor, index: int) -> int:
            return int(max(0, min(9, round(float(row[index] * 9.0)))))

        def _key(row: torch.Tensor) -> Tuple[object, ...]:
            color = _color(row, 4)
            if mode == "local_pattern":
                return tuple(_color(row, index) for index in (4, 6, 7, 8, 9))
            if mode == "position_color":
                return (round(float(row[0]), 4), round(float(row[1]), 4), color)
            return tuple(round(float(value), 4) for value in row.tolist())

        lookup_votes: Dict[Tuple[object, ...], Dict[int, int]] = {}
        color_votes: Dict[int, Dict[int, int]] = {}
        for row, target in zip(support_x, support_y):
            label = int(max(0, min(9, int(target))))
            key = _key(row)
            color = _color(row, 4)
            lookup_votes.setdefault(key, {})
            lookup_votes[key][label] = lookup_votes[key].get(label, 0) + 1
            color_votes.setdefault(color, {})
            color_votes[color][label] = color_votes[color].get(label, 0) + 1

        def _winner(votes: Dict[int, int]) -> Tuple[int, float]:
            total = max(1, sum(votes.values()))
            label, count = max(votes.items(), key=lambda item: (item[1], -item[0]))
            return int(label), float(count / total)

        predictions: List[int] = []
        confidences: List[float] = []
        hits = 0
        for row in query_x:
            key = _key(row)
            color = _color(row, 4)
            if key in lookup_votes:
                label, confidence = _winner(lookup_votes[key])
                hits += 1
            elif color in color_votes:
                label, confidence = _winner(color_votes[color])
            else:
                label, confidence = color, 0.5
            predictions.append(label)
            confidences.append(confidence)

        logits = torch.full((len(predictions), 10), -6.0, dtype=torch.float32, device=self.context.task.query_x.device)
        for row_index, (label, confidence) in enumerate(zip(predictions, confidences)):
            logits[row_index, label] = 6.0 * max(0.5, min(1.0, confidence))
        self.details.prediction_override = logits
        self.details.objective_signal += float(sum(confidences) / max(1, len(confidences)))
        self.details.primitive_effects[f"support_lookup_{mode}"] = float(hits / max(1, len(predictions)))

    def _exact_repair_candidates(
        self,
        support_grids: Sequence[Sequence[Sequence[int]]],
        output_grids: Sequence[Sequence[Sequence[int]]],
        query_grids: Sequence[Sequence[Sequence[int]]],
    ) -> List[Tuple[str, List[List[List[int]]], List[List[List[int]]]]]:
        candidates: List[Tuple[str, List[List[List[int]]], List[List[List[int]]]]] = []

        def add_candidate(name: str, fn: Callable[[Sequence[Sequence[int]]], List[List[int]]]) -> None:
            support_predicted = [fn(grid) for grid in support_grids]
            query_predicted = [fn(grid) for grid in query_grids]
            if all(self._grid_shape(predicted) == self._grid_shape(expected) for predicted, expected in zip(support_predicted, output_grids)):
                candidates.append((name, support_predicted, query_predicted))

        output_background = self._background_color_in_grids(output_grids)

        for keep_color in range(10):
            for replacement_name, replacement_fn in self._binary_replacement_policies(keep_color, output_grids):
                add_candidate(
                    f"binary_mask:{keep_color}:{replacement_name}",
                    lambda grid, keep_color=keep_color, replacement_fn=replacement_fn: self._binary_mask_recolor(
                        grid,
                        keep_color,
                        replacement_fn(grid),
                        output_background,
                    ),
                )

        for mode in ("top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"):
            add_candidate(f"edge_reflection:{mode}", lambda grid, mode=mode: self._edge_reflection_copy(grid, mode))

        marker_colors = sorted({int(color) for grid in support_grids for row in grid for color in row if int(color) != 0})
        fill_colors = self._support_added_colors(support_grids, output_grids, output_background)
        if not fill_colors:
            fill_colors = sorted({int(color) for grid in output_grids for row in grid for color in row if int(color) != output_background})
        for marker_color in marker_colors:
            for fill_color in fill_colors:
                if marker_color == fill_color:
                    continue
                add_candidate(
                    f"marker_rectangle_fill:{marker_color}:{fill_color}",
                    lambda grid, marker_color=marker_color, fill_color=fill_color: self._marker_rectangle_fill(grid, marker_color, fill_color),
                )

        add_candidate("corner_projection_2x2", self._corner_projection_2x2)

        for mode in ("drop", "clamp"):
            for dc in range(-4, 5):
                if dc == 0:
                    continue
                add_candidate(
                    f"rowwise_terminal_shift:{dc}:{mode}",
                    lambda grid, dc=dc, mode=mode: self._rowwise_terminal_shift(grid, dc, mode),
                )

        for block_color in marker_colors:
            add_candidate(
                f"project_markers_to_block:{block_color}",
                lambda grid, block_color=block_color: self._project_markers_to_block(grid, block_color),
            )

        add_candidate("move_objects_up_by_height", self._move_objects_up_by_height)

        return candidates

    @staticmethod
    def _grid_prediction_scores(
        predicted_grids: Sequence[Sequence[Sequence[int]]],
        expected_grids: Sequence[Sequence[Sequence[int]]],
    ) -> Tuple[float, float]:
        exact_total = 0
        exact_correct = 0
        cell_total = 0
        cell_correct = 0
        for predicted, expected in zip(predicted_grids, expected_grids):
            exact_total += 1
            exact_correct += 1 if predicted == expected else 0
            for pred_row, exp_row in zip(predicted, expected):
                for pred_color, exp_color in zip(pred_row, exp_row):
                    cell_total += 1
                    cell_correct += 1 if int(pred_color) == int(exp_color) else 0
        return exact_correct / max(1, exact_total), cell_correct / max(1, cell_total)

    @staticmethod
    def _most_common_color_in_grids(grids: Sequence[Sequence[Sequence[int]]]) -> int:
        counts: Dict[int, int] = {}
        for grid in grids:
            for row in grid:
                for color in row:
                    value = int(color)
                    counts[value] = counts.get(value, 0) + 1
        if not counts:
            return 0
        return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]

    @classmethod
    def _background_color_in_grids(cls, grids: Sequence[Sequence[Sequence[int]]]) -> int:
        if any(int(color) == 0 for grid in grids for row in grid for color in row):
            return 0
        return cls._most_common_color_in_grids(grids)

    @classmethod
    def _support_added_colors(
        cls,
        support_grids: Sequence[Sequence[Sequence[int]]],
        output_grids: Sequence[Sequence[Sequence[int]]],
        output_background: int,
    ) -> List[int]:
        added = set()
        for input_grid, expected_grid in zip(support_grids, output_grids):
            input_background = cls._background_color(input_grid)
            for input_row, expected_row in zip(input_grid, expected_grid):
                for input_color, expected_color in zip(input_row, expected_row):
                    if int(input_color) == int(input_background) and int(expected_color) != int(output_background):
                        added.add(int(expected_color))
        return sorted(added)

    def _binary_replacement_policies(
        self,
        keep_color: int,
        output_grids: Sequence[Sequence[Sequence[int]]],
    ) -> List[Tuple[str, Callable[[Sequence[Sequence[int]]], int]]]:
        policies: List[Tuple[str, Callable[[Sequence[Sequence[int]]], int]]] = []
        output_colors = sorted({int(color) for grid in output_grids for row in grid for color in row})
        for color in output_colors:
            policies.append((f"constant_{color}", lambda grid, color=color: int(color)))
        policies.append((f"other_positive_than_{keep_color}", lambda grid, keep_color=keep_color: self._other_positive_color(grid, keep_color)))
        policies.append((f"least_positive_than_{keep_color}", lambda grid, keep_color=keep_color: self._ranked_positive_color(grid, keep_color, least=True)))
        policies.append((f"most_positive_than_{keep_color}", lambda grid, keep_color=keep_color: self._ranked_positive_color(grid, keep_color, least=False)))
        return policies

    @staticmethod
    def _positive_color_counts(grid: Sequence[Sequence[int]]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for row in grid:
            for color in row:
                value = int(color)
                if value > 0:
                    counts[value] = counts.get(value, 0) + 1
        return counts

    def _other_positive_color(self, grid: Sequence[Sequence[int]], keep_color: int) -> int:
        counts = self._positive_color_counts(grid)
        others = {color: count for color, count in counts.items() if color != int(keep_color)}
        if not others:
            return int(keep_color)
        return max(others.items(), key=lambda item: (item[1], -item[0]))[0]

    def _ranked_positive_color(self, grid: Sequence[Sequence[int]], keep_color: int, *, least: bool) -> int:
        counts = self._positive_color_counts(grid)
        others = {color: count for color, count in counts.items() if color != int(keep_color)}
        if not others:
            return int(keep_color)
        key = (lambda item: (item[1], -item[0])) if not least else (lambda item: (-item[1], -item[0]))
        return max(others.items(), key=key)[0]

    @staticmethod
    def _binary_mask_recolor(grid: Sequence[Sequence[int]], keep_color: int, replacement_color: int, background: int) -> List[List[int]]:
        return [
            [int(replacement_color) if int(color) == int(keep_color) else int(background) for color in row]
            for row in grid
        ]

    def _edge_reflection_copy(self, grid: Sequence[Sequence[int]], mode: str) -> List[List[int]]:
        rows, cols = self._grid_shape(grid)
        if rows == 0 or cols == 0:
            return []
        background = self._background_color(grid)
        out = [list(map(int, row)) for row in grid]

        if mode in {"top_to_bottom", "bottom_to_top"}:
            row_indices = range(rows) if mode == "top_to_bottom" else range(rows - 1, -1, -1)
            band: List[List[int]] = []
            for row_index in row_indices:
                row = [int(color) for color in grid[row_index]]
                if all(color == background for color in row):
                    break
                band.append(row)
            if not band or len(band) >= rows:
                return out
            target_rows = range(rows - len(band), rows) if mode == "top_to_bottom" else range(len(band) - 1, -1, -1)
            for target_row, source_row in zip(target_rows, reversed(band)):
                out[target_row] = list(source_row)
            return out

        col_indices = range(cols) if mode == "left_to_right" else range(cols - 1, -1, -1)
        band_cols: List[List[int]] = []
        for col_index in col_indices:
            col = [int(grid[row_index][col_index]) for row_index in range(rows)]
            if all(color == background for color in col):
                break
            band_cols.append(col)
        if not band_cols or len(band_cols) >= cols:
            return out
        target_cols = range(cols - len(band_cols), cols) if mode == "left_to_right" else range(len(band_cols) - 1, -1, -1)
        for target_col, source_col in zip(target_cols, reversed(band_cols)):
            for row_index, color in enumerate(source_col):
                out[row_index][target_col] = int(color)
        return out

    def _marker_rectangle_fill(self, grid: Sequence[Sequence[int]], marker_color: int, fill_color: int) -> List[List[int]]:
        rows, cols = self._grid_shape(grid)
        if rows == 0 or cols == 0:
            return []
        background = self._background_color(grid)
        out = [list(map(int, row)) for row in grid]
        row_to_cols: Dict[int, set[int]] = {}
        for row_index, row in enumerate(grid):
            cols_with_marker = {col_index for col_index, color in enumerate(row) if int(color) == int(marker_color)}
            if len(cols_with_marker) >= 2:
                row_to_cols[row_index] = cols_with_marker
        row_indices = sorted(row_to_cols)
        for top_index, top in enumerate(row_indices):
            for bottom in row_indices[top_index + 1 :]:
                if bottom - top < 2:
                    continue
                common_cols = sorted(row_to_cols[top] & row_to_cols[bottom])
                for left_index, left in enumerate(common_cols):
                    for right in common_cols[left_index + 1 :]:
                        if right - left < 2:
                            continue
                        for row_index in range(top + 1, bottom):
                            for col_index in range(left + 1, right):
                                if int(out[row_index][col_index]) == int(background):
                                    out[row_index][col_index] = int(fill_color)
        return out

    def _corner_projection_2x2(self, grid: Sequence[Sequence[int]]) -> List[List[int]]:
        rows, cols = self._grid_shape(grid)
        if rows == 0 or cols == 0:
            return []
        background = self._background_color(grid)
        blocks: List[Tuple[int, int]] = []
        for row_index in range(rows - 1):
            for col_index in range(cols - 1):
                cells = [
                    int(grid[row_index][col_index]),
                    int(grid[row_index][col_index + 1]),
                    int(grid[row_index + 1][col_index]),
                    int(grid[row_index + 1][col_index + 1]),
                ]
                if all(color != background for color in cells):
                    blocks.append((row_index, col_index))
        if len(blocks) != 1:
            return [list(map(int, row)) for row in grid]

        row_index, col_index = blocks[0]
        top_left = int(grid[row_index][col_index])
        top_right = int(grid[row_index][col_index + 1])
        bottom_left = int(grid[row_index + 1][col_index])
        bottom_right = int(grid[row_index + 1][col_index + 1])
        out = [list(map(int, row)) for row in grid]
        left_width = min(2, cols)
        right_start = max(0, cols - 2)
        for row in range(max(0, row_index - 2), row_index):
            for col in range(left_width):
                out[row][col] = bottom_right
            for col in range(right_start, cols):
                out[row][col] = bottom_left
        for row in range(row_index + 2, min(rows, row_index + 4)):
            for col in range(left_width):
                out[row][col] = top_right
            for col in range(right_start, cols):
                out[row][col] = top_left
        return out

    def _rowwise_terminal_shift(self, grid: Sequence[Sequence[int]], dc: int, mode: str) -> List[List[int]]:
        rows, cols = self._grid_shape(grid)
        if rows == 0 or cols == 0:
            return []
        background = self._background_color(grid)
        out: List[List[int]] = []
        for row_index, row in enumerate(grid):
            values = [int(color) for color in row]
            if all(color == background for color in values):
                out.append(list(values))
                continue
            next_row_is_background = row_index + 1 >= rows or all(int(color) == background for color in grid[row_index + 1])
            if next_row_is_background and self._longest_non_background_run(values, background) >= 3:
                out.append(list(values))
                continue
            shifted = [int(background) for _ in range(cols)]
            for col_index, color in enumerate(values):
                if int(color) == int(background):
                    continue
                nc = col_index + int(dc)
                if mode == "clamp":
                    nc = max(0, min(cols - 1, nc))
                if 0 <= nc < cols:
                    shifted[nc] = int(color)
            out.append(shifted)
        return out

    @staticmethod
    def _longest_non_background_run(row: Sequence[int], background: int) -> int:
        longest = 0
        current = 0
        for color in row:
            if int(color) == int(background):
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    def _project_markers_to_block(self, grid: Sequence[Sequence[int]], block_color: int) -> List[List[int]]:
        rows, cols = self._grid_shape(grid)
        if rows == 0 or cols == 0:
            return []
        background = self._background_color(grid)
        cells = [
            (row_index, col_index)
            for row_index, row in enumerate(grid)
            for col_index, color in enumerate(row)
            if int(color) == int(block_color)
        ]
        if len(cells) < 4:
            return [list(map(int, row)) for row in grid]
        min_row = min(row for row, _ in cells)
        max_row = max(row for row, _ in cells)
        min_col = min(col for _, col in cells)
        max_col = max(col for _, col in cells)
        if min_row == max_row or min_col == max_col:
            return [list(map(int, row)) for row in grid]
        out = [list(map(int, row)) for row in grid]
        for row_index, row in enumerate(grid):
            for col_index, color in enumerate(row):
                value = int(color)
                if value in {int(background), int(block_color)}:
                    continue
                inside_block_box = min_row <= row_index <= max_row and min_col <= col_index <= max_col
                if inside_block_box:
                    continue
                if min_row <= row_index <= max_row:
                    target_col = min_col if col_index < min_col else max_col if col_index > max_col else None
                    if target_col is not None:
                        out[row_index][target_col] = value
                if min_col <= col_index <= max_col:
                    target_row = min_row if row_index < min_row else max_row if row_index > max_row else None
                    if target_row is not None:
                        out[target_row][col_index] = value
        return out

    def _move_objects_up_by_height(self, grid: Sequence[Sequence[int]]) -> List[List[int]]:
        rows, cols = self._grid_shape(grid)
        if rows == 0 or cols == 0:
            return []
        background = self._background_color(grid)
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        components: List[Tuple[int, List[Tuple[int, int]]]] = []
        for start_row in range(rows):
            for start_col in range(cols):
                color = int(grid[start_row][start_col])
                if color == int(background) or visited[start_row][start_col]:
                    continue
                stack = [(start_row, start_col)]
                visited[start_row][start_col] = True
                cells: List[Tuple[int, int]] = []
                while stack:
                    row_index, col_index = stack.pop()
                    cells.append((row_index, col_index))
                    for nr, nc in (
                        (row_index - 1, col_index),
                        (row_index + 1, col_index),
                        (row_index, col_index - 1),
                        (row_index, col_index + 1),
                    ):
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and int(grid[nr][nc]) == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((color, cells))

        out = [[int(background) for _ in range(cols)] for _ in range(rows)]
        for color, cells in sorted(components, key=lambda item: (min(row for row, _ in item[1]), min(col for _, col in item[1]))):
            min_row = min(row for row, _ in cells)
            max_row = max(row for row, _ in cells)
            height = max_row - min_row + 1
            for row_index, col_index in cells:
                target_row = row_index - height
                if 0 <= target_row < rows:
                    out[target_row][col_index] = int(color)
        return out

    @staticmethod
    def _grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
        return len(grid), len(grid[0]) if grid else 0

    @staticmethod
    def _feature_grids(features: torch.Tensor, shapes: object, offsets: object) -> List[List[List[int]]]:
        if not isinstance(shapes, Sequence) or not isinstance(offsets, Sequence):
            return []
        if len(offsets) != len(shapes) + 1:
            return []
        flat_features = features.detach().float().reshape(features.shape[0], -1)
        if flat_features.shape[1] < 5:
            return []
        grids: List[List[List[int]]] = []
        for index, shape in enumerate(shapes):
            if not isinstance(shape, Sequence) or len(shape) != 2:
                return []
            start = int(offsets[index])
            end = int(offsets[index + 1])
            rows, cols = int(shape[0]), int(shape[1])
            if start < 0 or end > flat_features.shape[0] or end - start != rows * cols:
                return []
            colors = [
                int(max(0, min(9, round(float(value * 9.0)))))
                for value in flat_features[start:end, 4].tolist()
            ]
            grids.append([colors[row * cols : (row + 1) * cols] for row in range(rows)])
        return grids

    @staticmethod
    def _grid_transforms():
        return {
            "identity": lambda grid: [list(row) for row in grid],
            "hflip": lambda grid: [list(reversed(row)) for row in grid],
            "vflip": lambda grid: [list(row) for row in reversed(grid)],
            "rot180": lambda grid: [list(reversed(row)) for row in reversed(grid)],
            "transpose": lambda grid: [list(row) for row in zip(*grid)] if len(grid) == len(grid[0]) else [],
            "rot90": lambda grid: [list(row) for row in zip(*reversed(grid))] if len(grid) == len(grid[0]) else [],
            "rot270": lambda grid: [list(row) for row in reversed(list(zip(*grid)))] if len(grid) == len(grid[0]) else [],
        }

    @staticmethod
    def _apply_color_mapping(grid: Sequence[Sequence[int]], mapping: Dict[int, int], default_color: int) -> List[List[int]]:
        return [
            [int(mapping.get(int(color), int(color) if 0 <= int(color) <= 9 else default_color)) for color in row]
            for row in grid
        ]

    @staticmethod
    def _reverse_nested_ring_colors(grid: Sequence[Sequence[int]]) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0]) if grid else 0
        if rows == 0 or cols == 0:
            return []
        ring_votes: Dict[int, Dict[int, int]] = {}
        for row_index, row in enumerate(grid):
            for col_index, color in enumerate(row):
                ring = min(row_index, col_index, rows - 1 - row_index, cols - 1 - col_index)
                value = int(color)
                ring_votes.setdefault(ring, {})
                ring_votes[ring][value] = ring_votes[ring].get(value, 0) + 1
        ring_ids = sorted(ring_votes)
        ring_colors = [
            max(ring_votes[ring].items(), key=lambda item: (item[1], -item[0]))[0]
            for ring in ring_ids
        ]
        reversed_colors = list(reversed(ring_colors))
        ring_map = {ring: reversed_colors[index] for index, ring in enumerate(ring_ids)}
        return [
            [
                int(ring_map[min(row_index, col_index, rows - 1 - row_index, cols - 1 - col_index)])
                for col_index in range(cols)
            ]
            for row_index in range(rows)
        ]

    @staticmethod
    def _most_common_color(grid: Sequence[Sequence[int]]) -> int:
        counts: Dict[int, int] = {}
        for row in grid:
            for color in row:
                value = int(color)
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return 0
        return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]

    @classmethod
    def _background_color(cls, grid: Sequence[Sequence[int]]) -> int:
        if any(int(color) == 0 for row in grid for color in row):
            return 0
        return cls._most_common_color(grid)

    @staticmethod
    def _translate_non_background(grid: Sequence[Sequence[int]], dr: int, dc: int, background: int, *, mode: str = "drop") -> List[List[int]]:
        rows, cols = len(grid), len(grid[0]) if grid else 0
        out = [[int(background) for _ in range(cols)] for _ in range(rows)]
        for row_index, row in enumerate(grid):
            for col_index, color in enumerate(row):
                value = int(color)
                if value == int(background):
                    continue
                nr, nc = row_index + int(dr), col_index + int(dc)
                if mode == "clamp":
                    nr = max(0, min(rows - 1, nr))
                    nc = max(0, min(cols - 1, nc))
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr][nc] = value
        return out

    @staticmethod
    def _logits_from_grids(grids: Sequence[Sequence[Sequence[int]]], device: torch.device) -> torch.Tensor:
        labels = [int(max(0, min(9, int(color)))) for grid in grids for row in grid for color in row]
        logits = torch.full((len(labels), 10), -6.0, dtype=torch.float32, device=device)
        for row_index, label in enumerate(labels):
            logits[row_index, label] = 6.0
        return logits

    def _op_constant_lr(self, step: PrimitiveStep) -> None:
        self.current_lr = _clamp(float(step.args.get("lr", self.base_lr)), 1e-5, 0.3)

    def _op_decayed_lr(self, step: PrimitiveStep) -> None:
        decay = float(step.args.get("decay", self.program.parameters.get("decay", 0.75)))
        self.current_lr = _clamp(self.current_lr * decay, 1e-5, 0.3)
        self.details.primitive_effects["decayed_lr"] = float(decay)

    def _op_increasing_lr(self, step: PrimitiveStep) -> None:
        growth = float(step.args.get("growth", self.program.parameters.get("growth", 1.15)))
        self.current_lr = _clamp(self.current_lr * growth, 1e-5, 0.3)
        self.details.primitive_effects["increasing_lr"] = float(growth)

    def _op_world_model_selected_lr(self, step: PrimitiveStep) -> None:
        scales = [0.5, 0.8, 1.0, 1.25, 1.5]
        scored = [(self._score_lr(self.base_lr * scale), scale) for scale in scales]
        score, scale = max(scored, key=lambda item: item[0])
        self.current_lr = _clamp(self.base_lr * scale, 1e-5, 0.3)
        self.world_score = score
        self.details.world_model_score = float(score)
        self.details.used_world_model = self.context.world_model is not None
        self.details.primitive_effects["world_model_selected_lr"] = float(scale)

    def _op_memory_gated_lr(self, step: PrimitiveStep) -> None:
        strength = float(step.args.get("strength", self.program.parameters.get("memory_gate_strength", 0.5)))
        gate = 1.0 + strength * (0.65 * self.memory_reward_gate + 0.35 * self.memory_similarity_gate)
        self.current_lr = _clamp(self.current_lr * gate, 1e-5, 0.3)
        self.details.primitive_effects["memory_gated_lr"] = float(gate)

    def _score_lr(self, lr: float) -> float:
        if self.context.world_model is None:
            return -abs(float(lr) - self.base_lr)
        state = _fit(self.context.task_embedding, self.context.world_model.latent_dim)
        action = torch.zeros(self.context.world_model.action_dim)
        action[0] = float(lr)
        if action.numel() > 1:
            action[1] = float(self.step_index + 1)
        if action.numel() > 2:
            action[2] = float(self.memory_reward_gate)
        with torch.no_grad():
            pred = self.context.world_model.forward(state.reshape(1, -1), action.reshape(1, -1)).reshape(-1)
        score = float(pred[0])
        return -score if self.wrong_world_model else score


def compile_operator_program(program: OperatorProgram) -> ProgramAdaptationOperator:
    for step in program.primitive_sequence:
        if step.name not in ALL_PRIMITIVES:
            raise ValueError(f"unknown DSL primitive: {step.name}")
    return ProgramAdaptationOperator(program)


def execute_operator_program(program: OperatorProgram, context: OperatorExecutionContext) -> OperatorTrace:
    operator = compile_operator_program(program)
    gene = OperatorGene(program.program_id, operator.name, {"source_hash": program.source_hash, "program": program.to_dict()}, program.generation_created, program.parent_program_id, program.accepted)
    return operator.apply(gene, context)


def program_action_vector(program: OperatorProgram) -> torch.Tensor:
    names = {step.name for step in program.primitive_sequence}
    lr_scale = float(program.parameters.get("lr_scale", 1.0))
    step_count = sum(1 for step in program.primitive_sequence if step.name in {"maml_step", "support_proxy_step", "variable_lr_step"})
    memory_flag = 1.0 if names & set(MEMORY_PRIMITIVES) else 0.0
    world_flag = 1.0 if names & set(WORLD_MODEL_PRIMITIVES) else 0.0
    objective_flag = 1.0 if names & set(OBJECTIVE_PRIMITIVES) else 0.0
    complexity = len(program.primitive_sequence) / 12.0
    return torch.tensor([lr_scale, float(max(1, step_count)), memory_flag, world_flag, objective_flag, complexity], dtype=torch.float32)


def program_to_update_action(program: OperatorProgram):
    from model_based_controller import UpdateAction

    vector = program_action_vector(program)
    return UpdateAction(
        program.program_id,
        inner_lr=float(vector[0]),
        inner_steps=max(1, int(round(float(vector[1])))),
        memory_k=max(1, int(round(1.0 + 2.0 * float(vector[2])))),
        planner_horizon=max(1, int(round(1.0 + 2.0 * float(vector[3])))),
        exploration_noise_scale=float(vector[5]),
    )


def default_operator_programs() -> List[OperatorProgram]:
    return [
        OperatorProgram(
            "dsl0_maml",
            (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step")),
            {"lr_scale": 1.0},
            generation_created=0,
            accepted=True,
            status="accepted",
        ),
        OperatorProgram(
            "dsl0_reptile",
            (PrimitiveStep("constant_lr"), PrimitiveStep("maml_step"), PrimitiveStep("reptile_interpolation", {"mix": 0.55})),
            {"lr_scale": 1.0, "reptile_mix": 0.55},
            generation_created=0,
            accepted=True,
            status="accepted",
        ),
        OperatorProgram(
            "dsl0_memory_gate",
            (
                PrimitiveStep("memory_retrieval_k", {"k": 2}),
                PrimitiveStep("memory_reward_gate"),
                PrimitiveStep("memory_similarity_gate"),
                PrimitiveStep("memory_gated_lr", {"strength": 0.6}),
                PrimitiveStep("maml_step"),
            ),
            {"lr_scale": 1.0, "memory_k": 2, "memory_gate_strength": 0.6},
            generation_created=0,
            accepted=True,
            status="accepted",
        ),
        OperatorProgram(
            "dsl0_world_lr",
            (
                PrimitiveStep("world_model_lr_score"),
                PrimitiveStep("world_model_selected_lr"),
                PrimitiveStep("variable_lr_step", {"decay": 0.85}),
            ),
            {"lr_scale": 1.0, "decay": 0.85},
            generation_created=0,
            accepted=True,
            status="accepted",
        ),
    ]


def primitive_catalog() -> Dict[str, List[str]]:
    return {
        "gradient": list(GRADIENT_PRIMITIVES),
        "memory": list(MEMORY_PRIMITIVES),
        "world_model": list(WORLD_MODEL_PRIMITIVES),
        "objective": list(OBJECTIVE_PRIMITIVES),
        "schedule": list(SCHEDULE_PRIMITIVES),
    }


def assert_no_task_family_branching(source: str) -> None:
    subject = "task" + "." + "family"
    banned = ("if " + subject, "elif " + subject, "== " + '"sinusoid"', "== " + "'sinusoid'")
    for token in banned:
        if token in source:
            raise AssertionError(f"hardcoded task-family branch detected: {token}")


def assert_program_diversity(program_scores: Dict[str, Sequence[float]], min_range: float = 1e-8) -> None:
    if len(program_scores) < 2:
        raise RuntimeError("code-level RSI needs at least two executable programs")
    means = []
    for scores in program_scores.values():
        if scores:
            means.append(sum(float(x) for x in scores) / len(scores))
    if not means or max(means) - min(means) <= min_range:
        raise RuntimeError("all synthesized operator programs appear to be dead code")
