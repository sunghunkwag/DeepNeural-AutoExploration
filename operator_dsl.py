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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from adaptation_operators import AdaptationOperator, OperatorExecutionContext, OperatorGene, OperatorTrace
from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel
from maml_functional import functional_forward


GRADIENT_PRIMITIVES = (
    "maml_step",
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

    def apply(self, gene: OperatorGene, context: OperatorExecutionContext) -> OperatorTrace:
        adapted, details = self.adapt_params(context)
        before = self._before_loss(context)
        with torch.no_grad():
            pred = functional_forward(context.model, adapted, context.task.query_x)
            after = float(F.mse_loss(pred, context.task.query_y))
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
        if not self.details.selected_lrs:
            self._gradient_step(self.current_lr)
        self.details.behavior_signature = [
            float(sum(self.details.selected_lrs)),
            float(self.details.memory_reward_signal),
            float(self.details.memory_similarity_signal),
            float(self.details.world_model_score),
            float(self.details.support_proxy_loss),
            float(self.details.objective_signal),
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
        return F.mse_loss(pred, self.context.task.support_y) * float(self.loss_weight)

    def _support_proxy_loss(self) -> torch.Tensor:
        sx = self.context.task.support_x
        sy = self.context.task.support_y
        if sx.shape[0] < 4:
            return self._support_loss()
        split = max(1, int(sx.shape[0] * 0.65))
        pred = functional_forward(self.context.model, self.adapted_params, sx[split:])
        return F.mse_loss(pred, sy[split:])

    def _gradient_step(self, lr: float) -> None:
        lr = _clamp(lr, 1e-5, 0.3)
        loss = self._support_loss()
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
    step_count = sum(1 for step in program.primitive_sequence if step.name in {"maml_step", "variable_lr_step"})
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
