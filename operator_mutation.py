"""Controlled validation-gated operator mutations."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class OperatorMutation:
    name: str
    updates: Dict[str, object]
    operator_type: str = "hyperparameter"


@dataclass
class MutationDecision:
    candidate_name: str
    baseline_scores: List[float]
    candidate_scores: List[float]
    accepted: bool
    rollback: bool
    random_seed: int
    split: str
    updates: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "baseline_scores": self.baseline_scores,
            "candidate_scores": self.candidate_scores,
            "accepted": self.accepted,
            "rollback": self.rollback,
            "random_seed": self.random_seed,
            "split": self.split,
            "updates": dict(self.updates),
        }


def allowed_operator_mutations(state: Dict[str, object]) -> List[OperatorMutation]:
    lr = float(state.get("inner_lr", 0.01))
    steps = int(state.get("inner_steps", 1))
    k = int(state.get("memory_retrieval_k", 1))
    horizon = int(state.get("planner_horizon", 1))
    uncertainty = float(state.get("uncertainty_weight", 1.0))
    intrinsic = float(state.get("intrinsic_weight", 1.0))
    return [
        OperatorMutation("inner_lr_down", {"inner_lr": lr * 0.5}),
        OperatorMutation("inner_lr_up", {"inner_lr": lr * 1.5}),
        OperatorMutation("inner_steps_plus", {"inner_steps": steps + 1}),
        OperatorMutation("exploration_policy_stochastic", {"exploration_policy": "stochastic"}, "operator"),
        OperatorMutation("memory_k_plus", {"memory_retrieval_k": k + 1}, "operator"),
        OperatorMutation("task_encoder_off", {"use_task_encoder": False}, "operator"),
        OperatorMutation("task_encoder_learned", {"encoder_type": "learned"}, "operator"),
        OperatorMutation("planner_horizon_plus", {"planner_horizon": horizon + 1}, "operator"),
        OperatorMutation("toggle_first_order", {"first_order": not bool(state.get("first_order", True))}, "operator"),
        OperatorMutation("width_multiplier_small", {"width_multiplier": float(state.get("width_multiplier", 1.0)) * 1.25}, "architecture"),
        OperatorMutation("uncertainty_weight_up", {"uncertainty_weight": uncertainty * 1.1}, "objective"),
        OperatorMutation("intrinsic_weight_down", {"intrinsic_weight": intrinsic * 0.9}, "objective"),
    ]


class OperatorMutationController:
    """Snapshot/apply/evaluate/rollback controller using validation tasks only."""

    def __init__(self, state: Dict[str, object], improvement_threshold: float = 0.0, min_consistent_wins: int = 1, random_seed: int = 0):
        self.state = copy.deepcopy(state)
        self.improvement_threshold = improvement_threshold
        self.min_consistent_wins = min_consistent_wins
        self.random_seed = random_seed
        self.log: List[Dict[str, object]] = []
        self.rollback_count = 0
        self.accepted_mutation_count = 0

    def snapshot(self) -> Dict[str, object]:
        return copy.deepcopy(self.state)

    def restore(self, snapshot: Dict[str, object]) -> None:
        self.state = copy.deepcopy(snapshot)
        self.rollback_count += 1

    def apply(self, mutation: OperatorMutation) -> None:
        self.state.update(copy.deepcopy(mutation.updates))

    def evaluate(self, mutation: OperatorMutation, validation_tasks: Sequence[object], evaluator: Callable[[Dict[str, object], Sequence[object]], Sequence[float] | float], split: str = "validation") -> Dict[str, object]:
        if split != "validation":
            raise ValueError("operator mutations may only be accepted on validation split")
        if not validation_tasks:
            raise ValueError("validation_tasks are required for mutation acceptance")
        if any(getattr(task, "split", split) == "test" for task in validation_tasks):
            raise ValueError("test tasks cannot be used for mutation acceptance")

        before = self.snapshot()
        baseline = evaluator(before, validation_tasks)
        baseline_scores = [float(baseline)] if isinstance(baseline, (int, float)) else [float(x) for x in baseline]
        trial = copy.deepcopy(before)
        trial.update(copy.deepcopy(mutation.updates))
        candidate = evaluator(trial, validation_tasks)
        candidate_scores = [float(candidate)] if isinstance(candidate, (int, float)) else [float(x) for x in candidate]
        if len(baseline_scores) != len(candidate_scores):
            raise ValueError("evaluator must return comparable score sequences")
        deltas = [b - c for b, c in zip(baseline_scores, candidate_scores)]
        accepted = float(np.mean(deltas)) > self.improvement_threshold and sum(d > self.improvement_threshold for d in deltas) >= self.min_consistent_wins
        rollback = not accepted
        if accepted:
            self.state = trial
            self.accepted_mutation_count += 1
        else:
            self.restore(before)
        decision = MutationDecision(mutation.name, baseline_scores, candidate_scores, accepted, rollback, self.random_seed, split, mutation.updates).as_dict()
        self.log.append(decision)
        return decision
