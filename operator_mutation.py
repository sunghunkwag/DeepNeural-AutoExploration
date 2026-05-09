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
    description: str = ""
    affects_runtime_keys: Sequence[str] = field(default_factory=tuple)

    def runtime_keys(self) -> List[str]:
        return [str(k) for k in (self.affects_runtime_keys or tuple(self.updates.keys()))]


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
    operator_type: str = "hyperparameter"
    mean_improvement: float = 0.0
    win_rate: float = 0.0
    consistent_wins: int = 0
    runtime_effect_keys: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "operator_type": self.operator_type,
            "baseline_scores": self.baseline_scores,
            "candidate_scores": self.candidate_scores,
            "mean_improvement": self.mean_improvement,
            "win_rate": self.win_rate,
            "consistent_wins": self.consistent_wins,
            "accepted": self.accepted,
            "rollback": self.rollback,
            "random_seed": self.random_seed,
            "split": self.split,
            "updates": dict(self.updates),
            "runtime_effect_keys": list(self.runtime_effect_keys),
        }


def allowed_operator_mutations(state: Dict[str, object]) -> List[OperatorMutation]:
    lr = float(state.get("inner_lr", 0.01))
    steps = int(state.get("inner_steps", 1))
    k = int(state.get("memory_retrieval_k", 1))
    horizon = int(state.get("planner_horizon", 1))
    uncertainty = float(state.get("uncertainty_weight", 1.0))
    intrinsic = float(state.get("intrinsic_weight", 1.0))
    exploration_noise = float(state.get("exploration_noise_scale", 0.0))
    use_encoder = bool(state.get("use_task_encoder", True))
    use_memory = bool(state.get("use_memory_conditioning", True))
    use_world_model = bool(state.get("use_world_model_controller", True))
    return [
        OperatorMutation("inner_lr_down", {"inner_lr": lr * 0.5}, "hyperparameter", "Scale down inner-loop learning rate."),
        OperatorMutation("inner_lr_up", {"inner_lr": lr * 1.5}, "hyperparameter", "Scale up inner-loop learning rate."),
        OperatorMutation("inner_steps_plus", {"inner_steps": steps + 1}, "hyperparameter", "Add one functional MAML inner step."),
        OperatorMutation("inner_steps_minus", {"inner_steps": max(1, steps - 1)}, "hyperparameter", "Remove one functional MAML inner step."),
        OperatorMutation("memory_k_plus", {"memory_retrieval_k": k + 1}, "operator", "Retrieve one more non-test memory episode."),
        OperatorMutation("memory_k_minus", {"memory_retrieval_k": max(1, k - 1)}, "operator", "Retrieve one fewer memory episode."),
        OperatorMutation("task_encoder_toggle", {"use_task_encoder": not use_encoder}, "operator", "Toggle learned support-only task encoder."),
        OperatorMutation("memory_conditioning_toggle", {"use_memory_conditioning": not use_memory}, "operator", "Toggle memory-conditioned adaptation."),
        OperatorMutation("world_model_controller_toggle", {"use_world_model_controller": not use_world_model}, "operator", "Toggle world-model action selection."),
        OperatorMutation("planner_horizon_plus", {"planner_horizon": horizon + 1}, "operator", "Increase short-horizon planning/action horizon."),
        OperatorMutation("planner_horizon_minus", {"planner_horizon": max(1, horizon - 1)}, "operator", "Decrease short-horizon planning/action horizon."),
        OperatorMutation("intrinsic_weight_up", {"intrinsic_weight": intrinsic * 1.1}, "objective", "Increase intrinsic improvement weighting."),
        OperatorMutation("intrinsic_weight_down", {"intrinsic_weight": intrinsic * 0.9}, "objective", "Decrease intrinsic improvement weighting."),
        OperatorMutation("uncertainty_weight_up", {"uncertainty_weight": uncertainty * 1.1}, "objective", "Increase uncertainty penalty/weight."),
        OperatorMutation("uncertainty_weight_down", {"uncertainty_weight": uncertainty * 0.9}, "objective", "Decrease uncertainty penalty/weight."),
        OperatorMutation("exploration_noise_decay", {"exploration_noise_scale": max(0.0, exploration_noise * 0.7)}, "operator", "Decay the exploration noise schedule."),
        OperatorMutation("exploration_noise_up", {"exploration_noise_scale": exploration_noise + 0.01}, "operator", "Increase the exploration noise schedule."),
        OperatorMutation("width_multiplier_small", {"width_multiplier": float(state.get("width_multiplier", 1.0)) * 1.25}, "architecture", "Rebuild/select a wider compatible model variant."),
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
        mean_improvement = float(np.mean(deltas))
        wins = sum(d > self.improvement_threshold for d in deltas)
        win_rate = float(wins / max(1, len(deltas)))
        accepted = mean_improvement > self.improvement_threshold and wins >= self.min_consistent_wins
        rollback = not accepted
        if accepted:
            self.state = trial
            self.accepted_mutation_count += 1
        else:
            self.restore(before)
        decision = MutationDecision(
            mutation.name,
            baseline_scores,
            candidate_scores,
            accepted,
            rollback,
            self.random_seed,
            split,
            mutation.updates,
            operator_type=mutation.operator_type,
            mean_improvement=mean_improvement,
            win_rate=win_rate,
            consistent_wins=wins,
            runtime_effect_keys=mutation.runtime_keys(),
        ).as_dict()
        self.log.append(decision)
        return decision
