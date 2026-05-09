"""Failure grammar for rejected code-level RSI candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from operator_dsl import OperatorProgram, PrimitiveStep
from rsi_candidate_generator import CandidateGenerationRecord


FAILURE_RULE_TYPES = (
    "validation_only_overfit",
    "ood_collapse",
    "memory_gate_brittleness",
    "wrong_world_model_sensitivity",
    "no_runtime_behavior_difference",
    "exploding_or_nan_loss",
    "deterministic_replay_failure",
    "dead_code_candidate",
    "high_validation_gain_poor_transfer",
)


@dataclass(frozen=True)
class FailureRule:
    rule_id: str
    rule_type: str
    reason: str
    trigger_count: int
    penalty: float
    avoid_primitives: List[str] = field(default_factory=list)
    require_primitives: List[str] = field(default_factory=list)
    max_lr_scale: float | None = None
    created_generation: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "reason": self.reason,
            "trigger_count": self.trigger_count,
            "penalty": float(self.penalty),
            "avoid_primitives": list(self.avoid_primitives),
            "require_primitives": list(self.require_primitives),
            "max_lr_scale": self.max_lr_scale,
            "created_generation": self.created_generation,
        }


class FailureGrammar:
    """Compress rejected candidate decisions into reusable generation rules."""

    def __init__(self):
        self.rules: Dict[str, FailureRule] = {}
        self.reuse_count = 0
        self.rewrite_log: List[Dict[str, object]] = []

    def update_from_decisions(self, decisions: Sequence[Dict[str, object]], generation: int) -> List[FailureRule]:
        added: List[FailureRule] = []
        for decision in decisions:
            if decision.get("accepted"):
                continue
            rule_type = classify_failure(decision)
            program = decision.get("candidate_program", {})
            primitives = [str(step.get("name")) for step in program.get("primitive_sequence", [])] if isinstance(program, dict) else []
            rule_id = f"{rule_type}:{','.join(sorted(set(primitives))) or 'generic'}"
            previous = self.rules.get(rule_id)
            count = 1 if previous is None else previous.trigger_count + 1
            avoid, require, max_lr = _rule_actions(rule_type, primitives)
            rule = FailureRule(
                rule_id=rule_id,
                rule_type=rule_type,
                reason=str(decision.get("rejection_reason", rule_type)),
                trigger_count=count,
                penalty=min(2.0, 0.25 * count),
                avoid_primitives=avoid,
                require_primitives=require,
                max_lr_scale=max_lr,
                created_generation=generation if previous is None else previous.created_generation,
            )
            self.rules[rule_id] = rule
            if previous is None:
                added.append(rule)
        return added

    def apply_to_candidates(self, records: Sequence[CandidateGenerationRecord], generation: int) -> List[CandidateGenerationRecord]:
        rewritten: List[CandidateGenerationRecord] = []
        active_rules = list(self.rules.values())
        for record in records:
            candidate = record.candidate
            sequence = list(candidate.primitive_sequence)
            params = dict(candidate.parameters)
            applied: List[str] = []
            for rule in active_rules:
                names = [step.name for step in sequence]
                if rule.max_lr_scale is not None:
                    old = float(params.get("lr_scale", 1.0))
                    params["lr_scale"] = min(old, rule.max_lr_scale)
                    if params["lr_scale"] != old:
                        applied.append(rule.rule_id)
                for primitive in rule.avoid_primitives:
                    if primitive in names:
                        sequence = [step for step in sequence if step.name != primitive]
                        applied.append(rule.rule_id)
                for primitive in rule.require_primitives:
                    if primitive not in [step.name for step in sequence]:
                        sequence.insert(0, _default_step(primitive))
                        applied.append(rule.rule_id)
            if not any(step.name in {"maml_step", "variable_lr_step"} for step in sequence):
                sequence.append(PrimitiveStep("maml_step"))
            if applied:
                self.reuse_count += 1
                new_program = OperatorProgram(
                    f"{candidate.program_id}_fg{len(applied)}",
                    tuple(sequence[:9]),
                    params,
                    candidate.parent_program_id,
                    candidate.generation_created,
                )
                rewritten.append(
                    CandidateGenerationRecord(
                        new_program,
                        f"{record.method}+failure_grammar",
                        record.reason + f"; rewritten by failure grammar rules {sorted(set(applied))}",
                        list(record.parent_program_ids),
                        record.generation,
                        used_test_results=False,
                    )
                )
                self.rewrite_log.append(
                    {
                        "generation": generation,
                        "original_program_id": candidate.program_id,
                        "rewritten_program_id": new_program.program_id,
                        "applied_rules": sorted(set(applied)),
                    }
                )
            else:
                rewritten.append(record)
        return rewritten

    def penalty_for_program(self, program: OperatorProgram) -> float:
        names = {step.name for step in program.primitive_sequence}
        penalty = 0.0
        for rule in self.rules.values():
            if names.intersection(rule.avoid_primitives):
                penalty += rule.penalty
            lr = float(program.parameters.get("lr_scale", 1.0))
            if rule.max_lr_scale is not None and lr > rule.max_lr_scale:
                penalty += rule.penalty
        return float(penalty)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rules": [rule.to_dict() for rule in self.rules.values()],
            "rule_count": len(self.rules),
            "reuse_count": self.reuse_count,
            "rewrite_log": list(self.rewrite_log),
        }


def classify_failure(decision: Dict[str, object]) -> str:
    reason = str(decision.get("rejection_reason", ""))
    validation_gain = float(decision.get("mean_improvement", 0.0))
    hidden_guard = float(decision.get("hidden_guard_regression", 0.0))
    behavior = float(decision.get("runtime_behavior_difference", 0.0))
    program = decision.get("candidate_program", {})
    primitive_names = [str(step.get("name")) for step in program.get("primitive_sequence", [])] if isinstance(program, dict) else []
    if reason in {"non_finite_loss", "exploding_loss"} or "nan" in reason:
        return "exploding_or_nan_loss"
    if reason == "deterministic_replay_failed":
        return "deterministic_replay_failure"
    if behavior <= 1e-9:
        return "no_runtime_behavior_difference"
    if validation_gain > 0.0 and hidden_guard <= -0.5:
        return "high_validation_gain_poor_transfer"
    if reason == "hidden_validation_guard_regression":
        return "validation_only_overfit"
    if any("memory" in name for name in primitive_names) and hidden_guard < 0.0:
        return "memory_gate_brittleness"
    if any("world_model" in name or "predicted" in name for name in primitive_names) and hidden_guard < 0.0:
        return "wrong_world_model_sensitivity"
    if reason == "no_runtime_behavior_difference":
        return "dead_code_candidate"
    if hidden_guard <= -1.0:
        return "ood_collapse"
    return "validation_only_overfit"


def _rule_actions(rule_type: str, primitives: Sequence[str]) -> tuple[List[str], List[str], float | None]:
    if rule_type in {"exploding_or_nan_loss", "ood_collapse", "validation_only_overfit", "high_validation_gain_poor_transfer"}:
        return [], ["gradient_norm_clipping"], 0.9
    if rule_type == "memory_gate_brittleness":
        return [], ["memory_similarity_gate", "gradient_norm_clipping"], 0.95
    if rule_type == "wrong_world_model_sensitivity":
        return ["wrong_world_model_control"], ["validation_proxy_loss"], 1.0
    if rule_type == "no_runtime_behavior_difference" or rule_type == "dead_code_candidate":
        return [], ["variable_lr_step"], None
    if rule_type == "deterministic_replay_failure":
        return ["gradient_noise_injection"], ["gradient_norm_clipping"], 0.8
    return [], ["gradient_norm_clipping"], 1.0


def _default_step(name: str) -> PrimitiveStep:
    if name == "gradient_norm_clipping":
        return PrimitiveStep(name, {"max_norm": 1.5})
    if name == "variable_lr_step":
        return PrimitiveStep(name, {"decay": 0.8})
    if name == "memory_similarity_gate":
        return PrimitiveStep(name)
    if name == "validation_proxy_loss":
        return PrimitiveStep(name)
    return PrimitiveStep(name)

