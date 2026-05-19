"""Bounded evaluator-objective mutation for candidate selection scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .meta_config_memory import MetaConfigMemory, apply_memory_guidance_to_config
from .search_process_model import SearchProcessDiagnosis


@dataclass(frozen=True)
class EvaluatorObjectiveConfig:
    validation_loss_weight: float = 0.30
    hidden_validation_loss_weight: float = 0.35
    transfer_score_weight: float = 0.12
    generalization_gap_penalty_weight: float = 0.08
    architecture_complexity_penalty_weight: float = 0.06
    representation_failure_penalty_weight: float = 0.08
    gradient_failure_penalty_weight: float = 0.08
    rollback_risk_weight: float = 0.08
    task_family_regression_penalty_weight: float = 0.15
    module_contribution_weight: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return {
            "validation_loss_weight": self.validation_loss_weight,
            "hidden_validation_loss_weight": self.hidden_validation_loss_weight,
            "transfer_score_weight": self.transfer_score_weight,
            "generalization_gap_penalty_weight": self.generalization_gap_penalty_weight,
            "architecture_complexity_penalty_weight": self.architecture_complexity_penalty_weight,
            "representation_failure_penalty_weight": self.representation_failure_penalty_weight,
            "gradient_failure_penalty_weight": self.gradient_failure_penalty_weight,
            "rollback_risk_weight": self.rollback_risk_weight,
            "task_family_regression_penalty_weight": self.task_family_regression_penalty_weight,
            "module_contribution_weight": self.module_contribution_weight,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "EvaluatorObjectiveConfig":
        base = cls()
        if not payload:
            return base
        return cls(**{key: _bounded_float(payload.get(key), 0.0, 1.0, getattr(base, key)) for key in base.to_dict()})


@dataclass(frozen=True)
class EvaluatorObjectiveMutationPlan:
    base_config: Dict[str, float]
    mutated_config: Dict[str, float]
    selected_mutations: list[str]
    rationale: list[str]
    evidence_used: Dict[str, object]
    used_heldout_labels: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_config": dict(self.base_config),
            "mutated_config": dict(self.mutated_config),
            "selected_mutations": list(self.selected_mutations),
            "rationale": list(self.rationale),
            "evidence_used": dict(self.evidence_used),
            "used_heldout_labels": self.used_heldout_labels,
        }


class EvaluatorObjectiveMutator:
    def mutate(
        self,
        diagnosis: SearchProcessDiagnosis,
        selection_score_components: Sequence[Mapping[str, object]] = (),
        base_config: EvaluatorObjectiveConfig | None = None,
        prior_meta_memory: MetaConfigMemory | None = None,
    ) -> EvaluatorObjectiveMutationPlan:
        base = base_config or EvaluatorObjectiveConfig()
        cfg = dict(base.to_dict())
        selected: list[str] = []
        rationale: list[str] = []
        component_summary = _component_summary(selection_score_components)

        if diagnosis.evaluator_overfit_risk > 0.25:
            cfg["validation_loss_weight"] = max(0.05, cfg["validation_loss_weight"] - 0.05)
            cfg["hidden_validation_loss_weight"] = min(1.0, cfg["hidden_validation_loss_weight"] + 0.08)
            selected.extend(["validation-overfitting", "hidden-validation underweighting"])
            rationale.append("positive validation changes are not reliably surviving hidden validation")
        if diagnosis.transfer_regression_risk > 0.15:
            cfg["transfer_score_weight"] = min(1.0, cfg["transfer_score_weight"] + 0.06)
            cfg["generalization_gap_penalty_weight"] = min(1.0, cfg["generalization_gap_penalty_weight"] + 0.04)
            selected.append("transfer underweighting")
            rationale.append("post-freeze transfer regression risk is elevated")
        if diagnosis.architecture_complexity_growth > 0.25:
            cfg["architecture_complexity_penalty_weight"] = min(1.0, cfg["architecture_complexity_penalty_weight"] + 0.05)
            selected.append("excessive complexity tolerance")
            rationale.append("architecture complexity grew faster than process-quality metrics")
        if max(diagnosis.rollback_rate_by_method.values() or [0.0]) > 0.5:
            cfg["rollback_risk_weight"] = min(1.0, cfg["rollback_risk_weight"] + 0.06)
            selected.append("under-penalized rollback risk")
            rationale.append("one or more mutation methods is rolling back too often")
        if diagnosis.task_family_balance < 0.75:
            cfg["task_family_regression_penalty_weight"] = min(1.0, cfg["task_family_regression_penalty_weight"] + 0.05)
            selected.append("task-family imbalance")
            rationale.append("hidden-validation losses are imbalanced across task families")
        if component_summary.get("module_contribution_score", 0.0) <= 0.01 and diagnosis.probe_actionability_score > 0.5:
            cfg["module_contribution_weight"] = min(1.0, cfg["module_contribution_weight"] + 0.03)
            selected.append("accepted-candidate fragility")
            rationale.append("module contribution signal is weak relative to available probe evidence")
        if diagnosis.probe_actionability_score < 0.5:
            cfg["representation_failure_penalty_weight"] = min(1.0, cfg["representation_failure_penalty_weight"] + 0.03)
            cfg["gradient_failure_penalty_weight"] = min(1.0, cfg["gradient_failure_penalty_weight"] + 0.03)
            selected.append("probe underweighting")
            rationale.append("probe actionability is low, so probe failures need clearer scoring pressure")

        mutated = EvaluatorObjectiveConfig.from_dict(cfg)
        guidance: Dict[str, object] = {}
        if prior_meta_memory is not None:
            adjusted, guidance = apply_memory_guidance_to_config(
                "evaluator_objective_config",
                base.to_dict(),
                mutated.to_dict(),
                prior_meta_memory,
            )
            if guidance.get("avoided_harmful_config_patterns"):
                selected.append("avoid_harmful_prior_config_pattern")
                rationale.append("prior meta-config memory marked one or more evaluator changes as harmful")
            if guidance.get("reused_successful_config_patterns"):
                selected.append("reuse_successful_prior_config_pattern")
                rationale.append("prior meta-config memory supports one or more evaluator changes")
            if guidance.get("lower_confidence_neutral_patterns"):
                rationale.append("prior neutral config changes lower confidence for repeated evaluator mutations")
            mutated = EvaluatorObjectiveConfig.from_dict(adjusted)
        return EvaluatorObjectiveMutationPlan(
            base_config=base.to_dict(),
            mutated_config=mutated.to_dict(),
            selected_mutations=sorted(set(selected)),
            rationale=rationale,
            evidence_used={
                "evaluator_overfit_risk": diagnosis.evaluator_overfit_risk,
                "transfer_regression_risk": diagnosis.transfer_regression_risk,
                "architecture_complexity_growth": diagnosis.architecture_complexity_growth,
                "task_family_balance": diagnosis.task_family_balance,
                "rollback_rate_by_method": diagnosis.rollback_rate_by_method,
                "selection_component_summary": component_summary,
                "meta_config_memory_guidance": guidance,
            },
            used_heldout_labels=False,
        )


def _component_summary(items: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    values: Dict[str, list[float]] = {}
    for item in items:
        raw = item.get("components", {})
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            if isinstance(value, (int, float)):
                values.setdefault(str(key), []).append(float(value))
    return {key: sum(vals) / max(1, len(vals)) for key, vals in values.items()}


def _bounded_float(value: object, low: float, high: float, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    return float(max(low, min(high, value)))

