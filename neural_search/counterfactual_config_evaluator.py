"""Lightweight estimated counterfactuals for config mutations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Mapping

from .meta_config_attribution import ConfigChangeAttributionReport
from .search_process_model import SearchProcessDiagnosis, SearchProcessModel


@dataclass(frozen=True)
class CounterfactualConfigReport:
    config_change_id: str
    config_path: str
    estimated_without_change: Dict[str, object]
    observed_with_change: Dict[str, object]
    estimated_effect_size: float
    confidence: float
    rationale: str
    estimate_label: str = "estimate_not_proof"
    claims_proof: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "config_change_id": self.config_change_id,
            "config_path": self.config_path,
            "estimated_without_change": dict(self.estimated_without_change),
            "observed_with_change": dict(self.observed_with_change),
            "estimated_effect_size": self.estimated_effect_size,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "estimate_label": self.estimate_label,
            "claims_proof": self.claims_proof,
        }


class CounterfactualConfigEvaluator:
    """Estimate what would likely happen without each config mutation."""

    def evaluate(
        self,
        attribution_report: ConfigChangeAttributionReport | Mapping[str, object],
        first_diagnosis: SearchProcessDiagnosis | Mapping[str, object],
        second_diagnosis: SearchProcessDiagnosis | Mapping[str, object],
        first_run: Mapping[str, object] | None = None,
        second_run: Mapping[str, object] | None = None,
    ) -> list[CounterfactualConfigReport]:
        attr = attribution_report.to_dict() if isinstance(attribution_report, ConfigChangeAttributionReport) else dict(attribution_report)
        first_diag = _diagnosis(first_diagnosis, first_run)
        second_diag = _diagnosis(second_diagnosis, second_run)
        reports: list[CounterfactualConfigReport] = []
        raw_changes = attr.get("changes", [])
        if not isinstance(raw_changes, list):
            return reports
        for change in raw_changes:
            if isinstance(change, Mapping):
                reports.append(self.evaluate_change(change, first_diag, second_diag))
        return reports

    def evaluate_change(
        self,
        change: Mapping[str, object],
        first_diagnosis: SearchProcessDiagnosis,
        second_diagnosis: SearchProcessDiagnosis,
    ) -> CounterfactualConfigReport:
        path = str(change.get("config_path", ""))
        change_id = str(change.get("change_id", path))
        if ".mutation_penalties." in path:
            method = path.rsplit(".", 1)[-1]
            before = first_diagnosis.rollback_rate_by_method.get(method, 0.0)
            after = second_diagnosis.rollback_rate_by_method.get(method, 0.0)
            return CounterfactualConfigReport(
                config_change_id=change_id,
                config_path=path,
                estimated_without_change={"rollback_risk": before, "basis": "prior method rollback rate"},
                observed_with_change={"rollback_rate": after},
                estimated_effect_size=float(before - after),
                confidence=0.68,
                rationale="Estimate only: without the added mutation penalty, the prior method rollback risk is expected to persist.",
            )
        if "hidden_validation" in path or "generalization_gap" in path:
            before_mismatch = _mean(first_diagnosis.validation_hidden_mismatch_by_method)
            after_mismatch = _mean(second_diagnosis.validation_hidden_mismatch_by_method)
            before_overfit = first_diagnosis.evaluator_overfit_risk
            after_overfit = second_diagnosis.evaluator_overfit_risk
            return CounterfactualConfigReport(
                config_change_id=change_id,
                config_path=path,
                estimated_without_change={"validation_hidden_mismatch": before_mismatch, "evaluator_overfit_risk": before_overfit},
                observed_with_change={"validation_hidden_mismatch": after_mismatch, "evaluator_overfit_risk": after_overfit},
                estimated_effect_size=float((before_mismatch - after_mismatch) + (before_overfit - after_overfit)),
                confidence=0.64,
                rationale="Estimate only: without stronger hidden-validation weighting, evaluator overfit risk is expected to remain near the first-run level.",
            )
        if ".mutation_boosts." in path or path == "focus_failure_types":
            before_resolution = _mean(first_diagnosis.residue_resolution_rate_by_failure_type)
            after_resolution = _mean(second_diagnosis.residue_resolution_rate_by_failure_type)
            before_persistent = len(first_diagnosis.persistent_residue_types)
            after_persistent = len(second_diagnosis.persistent_residue_types)
            return CounterfactualConfigReport(
                config_change_id=change_id,
                config_path=path,
                estimated_without_change={"residue_resolution": before_resolution, "persistent_residue_count": before_persistent},
                observed_with_change={"residue_resolution": after_resolution, "persistent_residue_count": after_persistent},
                estimated_effect_size=float((after_resolution - before_resolution) + 0.1 * (before_persistent - after_persistent)),
                confidence=0.58,
                rationale="Estimate only: without a residue-targeted boost, residue resolution probability is expected to remain low.",
            )
        if path.startswith("curriculum_strategy_config") or path == "focus_task_families":
            before_balance = first_diagnosis.task_family_balance
            after_balance = second_diagnosis.task_family_balance
            return CounterfactualConfigReport(
                config_change_id=change_id,
                config_path=path,
                estimated_without_change={"task_family_balance": before_balance},
                observed_with_change={"task_family_balance": after_balance},
                estimated_effect_size=float(after_balance - before_balance),
                confidence=0.52,
                rationale="Estimate only: without curriculum adjustment, task-family balance is expected to remain close to the prior diagnosis.",
            )
        before_process_risk = (
            first_diagnosis.evaluator_overfit_risk
            + first_diagnosis.transfer_regression_risk
            + first_diagnosis.architecture_complexity_growth
        )
        after_process_risk = (
            second_diagnosis.evaluator_overfit_risk
            + second_diagnosis.transfer_regression_risk
            + second_diagnosis.architecture_complexity_growth
        )
        return CounterfactualConfigReport(
            config_change_id=change_id,
            config_path=path,
            estimated_without_change={"aggregate_process_risk": before_process_risk},
            observed_with_change={"aggregate_process_risk": after_process_risk},
            estimated_effect_size=float(before_process_risk - after_process_risk),
            confidence=0.42,
            rationale="Estimate only: this lightweight counterfactual uses process statistics and is not proof of causality.",
        )


def _diagnosis(value: SearchProcessDiagnosis | Mapping[str, object], run: Mapping[str, object] | None) -> SearchProcessDiagnosis:
    if isinstance(value, SearchProcessDiagnosis):
        return value
    if isinstance(value, Mapping):
        return SearchProcessDiagnosis(
            rollback_rate_by_method=_mapping_float(value.get("rollback_rate_by_method", {})),
            validation_hidden_mismatch_by_method=_mapping_float(value.get("validation_hidden_mismatch_by_method", {})),
            residue_resolution_rate_by_failure_type=_mapping_float(value.get("residue_resolution_rate_by_failure_type", {})),
            persistent_residue_types=[str(item) for item in value.get("persistent_residue_types", [])] if isinstance(value.get("persistent_residue_types"), list) else [],
            evaluator_overfit_risk=_float(value.get("evaluator_overfit_risk")),
            architecture_complexity_growth=_float(value.get("architecture_complexity_growth")),
            task_family_balance=_float(value.get("task_family_balance")),
            transfer_regression_risk=_float(value.get("transfer_regression_risk")),
        )
    if run is not None:
        return SearchProcessModel().diagnose(run)
    return SearchProcessDiagnosis()


def _mapping_float(value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if isinstance(raw, (int, float))}


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mean(value: Mapping[str, float]) -> float:
    return float(mean(value.values())) if value else 0.0
