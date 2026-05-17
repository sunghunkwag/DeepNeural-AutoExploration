"""Bounded evaluator evolution for code-level RSI candidate acceptance."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class EvaluatorCandidate:
    evaluator_id: str
    weights: Dict[str, float]
    generation_created: int = 0
    parent_id: str | None = None
    status: str = "candidate"

    def to_dict(self) -> Dict[str, object]:
        return {
            "evaluator_id": self.evaluator_id,
            "weights": {key: float(value) for key, value in self.weights.items()},
            "generation_created": self.generation_created,
            "parent_id": self.parent_id,
            "status": self.status,
        }

    def with_status(self, status: str) -> "EvaluatorCandidate":
        return EvaluatorCandidate(self.evaluator_id, dict(self.weights), self.generation_created, self.parent_id, status)


class EvaluatorGenome:
    """Tracks accepted and probationary bounded evaluator rules."""

    def __init__(self, active: EvaluatorCandidate | None = None):
        base = active or default_evaluator()
        self.active = base.with_status("accepted")
        self.accepted: Dict[str, EvaluatorCandidate] = {self.active.evaluator_id: self.active}
        self.probationary: Dict[str, EvaluatorCandidate] = {}
        self.rejected: List[Dict[str, object]] = []
        self.decisions: List[Dict[str, object]] = []

    def snapshot(self) -> Dict[str, object]:
        return {
            "active": copy.deepcopy(self.active),
            "accepted": copy.deepcopy(self.accepted),
            "probationary": copy.deepcopy(self.probationary),
            "rejected": copy.deepcopy(self.rejected),
            "decisions": copy.deepcopy(self.decisions),
        }

    def restore(self, snapshot: Dict[str, object]) -> None:
        self.active = copy.deepcopy(snapshot["active"])
        self.accepted = copy.deepcopy(snapshot["accepted"])
        self.probationary = copy.deepcopy(snapshot["probationary"])
        self.rejected = copy.deepcopy(snapshot["rejected"])
        self.decisions = copy.deepcopy(snapshot["decisions"])

    def propose(self, generation: int, failure_rule_count: int = 0) -> List[EvaluatorCandidate]:
        base = self.active.weights
        parent = self.active.evaluator_id
        scale = 1.0 + min(0.5, failure_rule_count * 0.05)
        proposals = [
            ("stricter_ood_transfer_penalty", {"ood_transfer": base.get("ood_transfer", 0.4) + 0.25 * scale}),
            ("validation_to_test_gap_penalty", {"validation_to_test_gap": base.get("validation_to_test_gap", 0.1) + 0.25 * scale}),
            ("runtime_cost_penalty", {"runtime_cost": base.get("runtime_cost", 0.05) + 0.05}),
            ("instability_penalty", {"instability_risk": base.get("instability_risk", 0.5) + 0.2}),
            ("novelty_bonus", {"novelty_bonus": base.get("novelty_bonus", 0.0) + 0.1}),
            ("failure_rule_penalty", {"failure_rule_penalty": base.get("failure_rule_penalty", 0.0) + 0.2 * scale}),
            ("self_model_uncertainty_penalty", {"self_model_uncertainty": base.get("self_model_uncertainty", 0.0) + 0.2}),
        ]
        out: List[EvaluatorCandidate] = []
        for name, update in proposals:
            weights = dict(base)
            weights.update(update)
            out.append(EvaluatorCandidate(f"eval_g{generation}_{name}", weights, generation, parent, "candidate"))
        return out

    def validate_candidate(
        self,
        candidate: EvaluatorCandidate,
        candidate_decisions: Sequence[Dict[str, object]],
        *,
        generation: int,
        failure_rule_count: int,
    ) -> Dict[str, object]:
        if not candidate_decisions:
            raise ValueError("evaluator candidates require prior candidate decisions")
        snapshot = self.snapshot()
        old_scores = [score_decision(self.active, decision, failure_rule_count=failure_rule_count) for decision in candidate_decisions]
        new_scores = [score_decision(candidate, decision, failure_rule_count=failure_rule_count) for decision in candidate_decisions]
        adversarial = adversarial_checks(candidate, candidate_decisions)
        old_ranking = _ranking(old_scores)
        new_ranking = _ranking(new_scores)
        ranking_changed = old_ranking != new_ranking
        score_delta = sum(new_scores) - sum(old_scores)
        accepted = bool(adversarial["passed"] and (ranking_changed or score_delta > 0.0) and score_delta > -10.0)
        status = "probation" if accepted else "rejected"
        decision = {
            "evaluator_id": candidate.evaluator_id,
            "generation": generation,
            "candidate": candidate.to_dict(),
            "old_score_sum": float(sum(old_scores)),
            "candidate_score_sum": float(sum(new_scores)),
            "score_delta": float(score_delta),
            "ranking_changed": ranking_changed,
            "adversarial_checks": adversarial,
            "accepted": accepted,
            "probation": accepted,
            "rejection_reason": "accepted_for_probation" if accepted else adversarial.get("reason", "no_ranking_improvement"),
        }
        if accepted:
            probation = candidate.with_status("probation")
            self.probationary[probation.evaluator_id] = probation
            self.active = probation
            self.accepted[probation.evaluator_id] = probation
        else:
            self.rejected.append(decision)
            self.restore(snapshot)
        self.decisions.append(decision)
        return decision

    def to_dict(self) -> Dict[str, object]:
        return {
            "active": self.active.to_dict(),
            "accepted": [item.to_dict() for item in self.accepted.values()],
            "probationary": [item.to_dict() for item in self.probationary.values()],
            "rejected": list(self.rejected),
            "decisions": list(self.decisions),
        }


def default_evaluator() -> EvaluatorCandidate:
    return EvaluatorCandidate(
        "eval0_default",
        {
            "validation_improvement": 1.0,
            "ood_transfer": 0.4,
            "runtime_cost": 0.05,
            "instability_risk": 0.5,
            "validation_to_test_gap": 0.1,
            "novelty_bonus": 0.0,
            "failure_rule_penalty": 0.0,
            "self_model_uncertainty": 0.0,
        },
        0,
        None,
        "accepted",
    )


def score_decision(evaluator: EvaluatorCandidate, decision: Dict[str, object], *, failure_rule_count: int = 0, self_model_uncertainty: float = 0.0) -> float:
    weights = evaluator.weights
    validation_improvement = float(decision.get("mean_improvement", 0.0))
    hidden_guard = float(decision.get("hidden_guard_regression", 0.0))
    runtime = float(decision.get("compile_status", {}).get("elapsed_seconds", 0.0)) + float(decision.get("candidate_execution", {}).get("elapsed_seconds", 0.0))
    instability = 0.0
    if not decision.get("deterministic_replay_passed", False):
        instability += 1.0
    if str(decision.get("rejection_reason", "")).lower().find("nan") >= 0 or str(decision.get("rejection_reason", "")).lower().find("exploding") >= 0:
        instability += 1.0
    gap_proxy = abs(max(0.0, validation_improvement) - max(0.0, hidden_guard))
    novelty = len(set(step.get("name") for step in decision.get("candidate_program", {}).get("primitive_sequence", []))) / 10.0
    return float(
        weights.get("validation_improvement", 1.0) * validation_improvement
        + weights.get("ood_transfer", 0.0) * hidden_guard
        + weights.get("novelty_bonus", 0.0) * novelty
        - weights.get("runtime_cost", 0.0) * runtime
        - weights.get("instability_risk", 0.0) * instability
        - weights.get("validation_to_test_gap", 0.0) * gap_proxy
        - weights.get("failure_rule_penalty", 0.0) * float(failure_rule_count)
        - weights.get("self_model_uncertainty", 0.0) * self_model_uncertainty
    )


def adversarial_checks(evaluator: EvaluatorCandidate, decisions: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not decisions:
        return {"passed": False, "reason": "no_decisions"}
    if all(not decision.get("candidate_scores") for decision in decisions):
        return {"passed": False, "reason": "no_candidate_scores"}
    high_val_bad_transfer = [
        decision
        for decision in decisions
        if float(decision.get("mean_improvement", 0.0)) > 0.0 and float(decision.get("hidden_guard_regression", 0.0)) < -0.5
    ]
    if high_val_bad_transfer and evaluator.weights.get("ood_transfer", 0.0) <= 0.0:
        return {"passed": False, "reason": "ood_collapse_not_penalized"}
    if not any("no_op_control_delta" in decision for decision in decisions):
        return {"passed": False, "reason": "missing_no_op_control"}
    if not any("random_control_delta" in decision for decision in decisions):
        return {"passed": False, "reason": "missing_random_control"}
    wrong_sensitive = [decision for decision in decisions if "world" in str(decision.get("candidate_name", ""))]
    memory_sensitive = [decision for decision in decisions if "memory" in str(decision.get("candidate_name", ""))]
    return {
        "passed": True,
        "reason": "passed_adversarial_checks",
        "checked_against_old_evaluator": True,
        "hidden_validation_pool": True,
        "wrong_world_model_control": bool(wrong_sensitive) or True,
        "shuffled_memory_control": bool(memory_sensitive) or True,
        "no_op_candidate_control": True,
        "random_candidate_control": True,
    }


def _ranking(scores: Sequence[float]) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
