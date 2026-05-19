"""Bounded curriculum mutation for multi-domain neural exploration tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

from .search_process_model import TASK_FAMILIES, SearchProcessDiagnosis


@dataclass(frozen=True)
class CurriculumStrategyConfig:
    difficulty: str = "hard"
    family_sampling_weights: Dict[str, float] = field(default_factory=lambda: {family: 1.0 for family in TASK_FAMILIES})
    hard_mode_ratio: float = 1.0
    distractor_intensity: float = 0.35
    sequence_lag_level: int = 7
    causal_confounder_strength: float = 1.0
    memory_distractor_count: int = 10
    object_interaction_strength: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "family_sampling_weights": dict(self.family_sampling_weights),
            "hard_mode_ratio": self.hard_mode_ratio,
            "distractor_intensity": self.distractor_intensity,
            "sequence_lag_level": self.sequence_lag_level,
            "causal_confounder_strength": self.causal_confounder_strength,
            "memory_distractor_count": self.memory_distractor_count,
            "object_interaction_strength": self.object_interaction_strength,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "CurriculumStrategyConfig":
        base = cls()
        if not payload:
            return base
        difficulty = str(payload.get("difficulty", base.difficulty))
        if difficulty not in {"simple", "hard"}:
            difficulty = base.difficulty
        return cls(
            difficulty=difficulty,
            family_sampling_weights=_family_weights(payload.get("family_sampling_weights"), base.family_sampling_weights),
            hard_mode_ratio=_bounded_float(payload.get("hard_mode_ratio"), 0.0, 1.0, base.hard_mode_ratio),
            distractor_intensity=_bounded_float(payload.get("distractor_intensity"), 0.0, 1.0, base.distractor_intensity),
            sequence_lag_level=max(1, min(12, int(payload.get("sequence_lag_level", base.sequence_lag_level)))),
            causal_confounder_strength=_bounded_float(payload.get("causal_confounder_strength"), 0.0, 2.0, base.causal_confounder_strength),
            memory_distractor_count=max(2, min(16, int(payload.get("memory_distractor_count", base.memory_distractor_count)))),
            object_interaction_strength=_bounded_float(payload.get("object_interaction_strength"), 0.0, 2.0, base.object_interaction_strength),
        )


@dataclass(frozen=True)
class CurriculumMutationPlan:
    base_config: Dict[str, object]
    mutated_config: Dict[str, object]
    selected_mutations: list[str]
    rationale: list[str]
    evidence_used: Dict[str, object]
    used_heldout_labels_for_candidate_selection: bool = False
    heldout_used_only_for_post_freeze_diagnosis: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_config": dict(self.base_config),
            "mutated_config": dict(self.mutated_config),
            "selected_mutations": list(self.selected_mutations),
            "rationale": list(self.rationale),
            "evidence_used": dict(self.evidence_used),
            "used_heldout_labels_for_candidate_selection": self.used_heldout_labels_for_candidate_selection,
            "heldout_used_only_for_post_freeze_diagnosis": self.heldout_used_only_for_post_freeze_diagnosis,
        }


class CurriculumStrategyMutator:
    def mutate(
        self,
        diagnosis: SearchProcessDiagnosis,
        base_config: CurriculumStrategyConfig | None = None,
    ) -> CurriculumMutationPlan:
        base = base_config or CurriculumStrategyConfig()
        weights = dict(base.family_sampling_weights)
        selected: list[str] = []
        rationale: list[str] = []
        hard_ratio = base.hard_mode_ratio
        difficulty = base.difficulty
        distractor = base.distractor_intensity
        sequence_lag = base.sequence_lag_level
        confounder = base.causal_confounder_strength
        memory_count = base.memory_distractor_count
        object_strength = base.object_interaction_strength

        worst_family = _worst_family(diagnosis.task_family_hidden_losses)
        if worst_family:
            weights[worst_family] = min(3.0, weights.get(worst_family, 1.0) + 0.35)
            selected.append("rebalance_task_family_sampling")
            rationale.append(f"{worst_family} has the largest hidden-validation loss")
        if diagnosis.task_family_balance < 0.75:
            selected.append("detect_task_family_imbalance")
            rationale.append("task-family hidden losses are uneven")

        performance_supports_hard = (
            diagnosis.task_family_balance >= 0.70
            and diagnosis.evaluator_overfit_risk < 0.35
            and diagnosis.transfer_regression_risk < 0.35
            and max(diagnosis.mutation_success_rate_by_method.values() or [0.0]) > 0.0
        )
        if performance_supports_hard:
            hard_ratio = min(1.0, hard_ratio + 0.10)
            difficulty = "hard"
            selected.append("increase_hard_mode_focus")
            rationale.append("hidden validation is stable enough to increase curriculum pressure")
        else:
            hard_ratio = max(0.0, hard_ratio - 0.10) if diagnosis.evaluator_overfit_risk > 0.50 else hard_ratio
            selected.append("hold_or_reduce_hard_mode_focus")
            rationale.append("hard-mode focus is not increased without hidden-validation support")

        if diagnosis.residue_concentration_by_task_family.get("sequence_prediction", 0.0) > 0.25:
            sequence_lag = min(12, sequence_lag + 1)
            selected.append("increase_sequence_lag_level")
        if diagnosis.residue_concentration_by_task_family.get("memory_retrieval", 0.0) > 0.25:
            memory_count = min(16, memory_count + 1)
            distractor = min(1.0, distractor + 0.05)
            selected.append("increase_memory_distractors")
        if diagnosis.residue_concentration_by_task_family.get("causal_intervention", 0.0) > 0.20:
            confounder = min(2.0, confounder + 0.10)
            selected.append("increase_causal_confounder_strength")
        if diagnosis.residue_concentration_by_task_family.get("object_state_transition", 0.0) > 0.20:
            object_strength = min(2.0, object_strength + 0.10)
            selected.append("increase_object_interaction_strength")

        mutated = CurriculumStrategyConfig(
            difficulty=difficulty,
            family_sampling_weights=_normalize_weights(weights),
            hard_mode_ratio=hard_ratio,
            distractor_intensity=distractor,
            sequence_lag_level=sequence_lag,
            causal_confounder_strength=confounder,
            memory_distractor_count=memory_count,
            object_interaction_strength=object_strength,
        )
        return CurriculumMutationPlan(
            base_config=base.to_dict(),
            mutated_config=mutated.to_dict(),
            selected_mutations=sorted(set(selected)),
            rationale=rationale,
            evidence_used={
                "task_family_hidden_losses": diagnosis.task_family_hidden_losses,
                "task_family_balance": diagnosis.task_family_balance,
                "residue_concentration_by_task_family": diagnosis.residue_concentration_by_task_family,
                "transfer_regression_risk": diagnosis.transfer_regression_risk,
                "evaluator_overfit_risk": diagnosis.evaluator_overfit_risk,
            },
            used_heldout_labels_for_candidate_selection=False,
            heldout_used_only_for_post_freeze_diagnosis=True,
        )


def _worst_family(losses: Mapping[str, float]) -> str:
    if not losses:
        return ""
    return max(sorted(losses), key=lambda family: float(losses[family]))


def _family_weights(value: object, default: Mapping[str, float]) -> Dict[str, float]:
    out = dict(default)
    if isinstance(value, Mapping):
        for family in TASK_FAMILIES:
            raw = value.get(family)
            if isinstance(raw, (int, float)):
                out[family] = max(0.05, min(3.0, float(raw)))
    return _normalize_weights(out)


def _normalize_weights(values: Mapping[str, float]) -> Dict[str, float]:
    out = {family: max(0.05, min(3.0, float(values.get(family, 1.0)))) for family in TASK_FAMILIES}
    average = sum(out.values()) / max(1, len(out))
    return {family: float(value / max(1e-8, average)) for family, value in out.items()}


def _bounded_float(value: object, low: float, high: float, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    return float(max(low, min(high, value)))

