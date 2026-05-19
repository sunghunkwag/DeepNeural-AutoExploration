"""Quality evaluation for meta-meta-meta configuration decisions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Mapping

from .architecture_genome import stable_hash
from .meta_config_attribution import ConfigChangeAttributionReport
from .search_process_model import SearchProcessDiagnosis, SearchProcessModel, compute_process_improvement_components


@dataclass(frozen=True)
class MetaDecisionQualityReport:
    decision_id: str
    input_run_id: str
    expected_improvement_target: str
    quality_scores: Dict[str, float]
    helped_changes: list[str]
    harmed_changes: list[str]
    neutral_changes: list[str]
    recommendation: str
    should_reuse_config: bool
    should_revert_config: bool
    should_mutate_config_further: bool
    rationale: str
    evidence_used: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "input_run_id": self.input_run_id,
            "expected_improvement_target": self.expected_improvement_target,
            "quality_scores": dict(self.quality_scores),
            "helped_changes": list(self.helped_changes),
            "harmed_changes": list(self.harmed_changes),
            "neutral_changes": list(self.neutral_changes),
            "recommendation": self.recommendation,
            "should_reuse_config": self.should_reuse_config,
            "should_revert_config": self.should_revert_config,
            "should_mutate_config_further": self.should_mutate_config_further,
            "rationale": self.rationale,
            "evidence_used": dict(self.evidence_used),
        }


class MetaDecisionQualityEvaluator:
    """Score whether a prior meta-meta-meta decision helped the next run."""

    def evaluate(
        self,
        first_run: Mapping[str, object],
        second_run: Mapping[str, object],
        attribution_report: ConfigChangeAttributionReport | Mapping[str, object],
        process_improvement_components: Mapping[str, float] | None = None,
        *,
        first_diagnosis: SearchProcessDiagnosis | Mapping[str, object] | None = None,
        second_diagnosis: SearchProcessDiagnosis | Mapping[str, object] | None = None,
    ) -> MetaDecisionQualityReport:
        attr = attribution_report.to_dict() if isinstance(attribution_report, ConfigChangeAttributionReport) else dict(attribution_report)
        first_diag = _diagnosis(first_run, first_diagnosis)
        second_diag = _diagnosis(second_run, second_diagnosis)
        components = dict(process_improvement_components or compute_process_improvement_components(first_diag, second_diag, first_run, second_run))
        decision = _mapping(first_run.get("meta_meta_meta_decision")) or {}
        decision_id = str(decision.get("decision_id") or stable_hash({"first": _run_id(first_run), "second": _run_id(second_run)}, prefix="mmm_"))
        input_run_id = str(decision.get("input_run_id") or _run_id(first_run))
        expected_target = str(decision.get("expected_improvement_target") or "final_process_improvement_delta")
        helped = [str(item) for item in attr.get("helped_changes", [])] if isinstance(attr.get("helped_changes", []), list) else []
        harmed = [str(item) for item in attr.get("harmed_changes", [])] if isinstance(attr.get("harmed_changes", []), list) else []
        neutral = [str(item) for item in attr.get("neutral_changes", [])] if isinstance(attr.get("neutral_changes", []), list) else []
        mismatch_delta = _mean_mapping(second_diag.validation_hidden_mismatch_by_method) - _mean_mapping(first_diag.validation_hidden_mismatch_by_method)
        persistent_delta = len(second_diag.persistent_residue_types) - len(first_diag.persistent_residue_types)
        prediction_accuracy = _score_positive(_target_delta(expected_target, components, first_run, second_run))
        config_effectiveness = _ratio_score(len(helped), len(harmed), len(neutral))
        residue_resolution_effectiveness = max(_score_positive(components.get("residue_resolution_delta", 0.0)), _score_nonpositive(float(persistent_delta)))
        rollback_reduction_effectiveness = _score_nonpositive(components.get("rollback_rate_delta", 0.0))
        robustness_effectiveness = _score_positive(components.get("hidden_validation_robustness_delta", 0.0))
        transfer_safety = _score_nonpositive(components.get("transfer_regression_risk_delta", 0.0))
        complexity_control = _score_nonpositive(components.get("complexity_growth_delta", 0.0))
        curriculum_effectiveness = _score_nonnegative(components.get("task_family_balance_delta", 0.0))
        evaluator_reweighting_effectiveness = _score_nonpositive(mismatch_delta)
        quality_scores = {
            "prediction_accuracy": prediction_accuracy,
            "config_effectiveness": config_effectiveness,
            "residue_resolution_effectiveness": residue_resolution_effectiveness,
            "rollback_reduction_effectiveness": rollback_reduction_effectiveness,
            "robustness_effectiveness": robustness_effectiveness,
            "transfer_safety": transfer_safety,
            "complexity_control": complexity_control,
            "curriculum_effectiveness": curriculum_effectiveness,
            "evaluator_reweighting_effectiveness": evaluator_reweighting_effectiveness,
        }
        quality_scores["overall_meta_decision_quality_score"] = float(mean(quality_scores.values()))
        final_delta = float(components.get("final_process_improvement_delta", 0.0))
        rollback_delta = float(components.get("rollback_rate_delta", 0.0))
        overfit_delta = float(components.get("evaluator_overfit_risk_delta", 0.0))
        overall = quality_scores["overall_meta_decision_quality_score"]
        rollback_or_overfit_regressed = rollback_delta > 0.0 or overfit_delta > 0.0
        should_reuse = overall >= 0.60 and final_delta > 0.0 and len(helped) >= len(harmed) and not rollback_or_overfit_regressed
        should_revert = overall < 0.35 or (final_delta < 0.0 and len(harmed) > len(helped)) or (rollback_delta > 0.05 and overfit_delta > 0.05)
        should_mutate = not should_reuse and not should_revert
        if should_reuse:
            recommendation = "trust_previous_config"
            rationale = "second run improved the strict process score and most attributed changes were helpful"
        elif should_revert:
            recommendation = "revert_previous_config"
            rationale = "second run worsened or harmful attributed changes dominated the decision"
        else:
            recommendation = "revise_previous_config"
            rationale = "second-run evidence is mixed, so preserve helpful changes and revise neutral or harmful ones"
        return MetaDecisionQualityReport(
            decision_id=decision_id,
            input_run_id=input_run_id,
            expected_improvement_target=expected_target,
            quality_scores=quality_scores,
            helped_changes=helped,
            harmed_changes=harmed,
            neutral_changes=neutral,
            recommendation=recommendation,
            should_reuse_config=should_reuse,
            should_revert_config=should_revert,
            should_mutate_config_further=should_mutate,
            rationale=rationale,
            evidence_used={
                "process_improvement_components": components,
                "validation_hidden_mismatch_delta": mismatch_delta,
                "persistent_residue_count_delta": persistent_delta,
                "rollback_repair_needed": rollback_delta > 0.0,
                "evaluator_overfit_repair_needed": overfit_delta > 0.0,
                "attribution_id": attr.get("attribution_id", ""),
            },
        )


def _diagnosis(run: Mapping[str, object], value: SearchProcessDiagnosis | Mapping[str, object] | None) -> SearchProcessDiagnosis:
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
    existing = _mapping(run.get("search_process_diagnosis"))
    if existing:
        return _diagnosis(run, existing)
    return SearchProcessModel().diagnose(run)


def _target_delta(target: str, components: Mapping[str, float], first_run: Mapping[str, object], second_run: Mapping[str, object]) -> float:
    target_to_component = {
        "hidden_validation_robustness": "hidden_validation_robustness_delta",
        "residue_resolution_rate": "residue_resolution_delta",
        "task_family_balance": "task_family_balance_delta",
        "task_family_balance_score": "task_family_balance_delta",
        "heldout_generalization_delta": "transfer_regression_risk_delta",
        "cumulative_autonomous_improvement_score": "final_process_improvement_delta",
        "final_process_improvement_delta": "final_process_improvement_delta",
    }
    key = target_to_component.get(target, "final_process_improvement_delta")
    value = float(components.get(key, 0.0))
    if key == "transfer_regression_risk_delta":
        return -value
    if target == "cumulative_autonomous_improvement_score":
        second = _agi_metric(second_run, target)
        first = _agi_metric(first_run, target)
        return second - first if second != 0.0 or first != 0.0 else value
    return value


def _score_positive(delta: float) -> float:
    if delta > 0.0:
        return 1.0
    if delta == 0.0:
        return 0.5
    return 0.0


def _score_nonnegative(delta: float) -> float:
    return 1.0 if delta >= 0.0 else 0.0


def _score_nonpositive(delta: float) -> float:
    if delta < 0.0:
        return 1.0
    if delta == 0.0:
        return 0.75
    return 0.0


def _ratio_score(helped: int, harmed: int, neutral: int) -> float:
    total = helped + harmed + neutral
    if total == 0:
        return 0.5
    return float(max(0.0, min(1.0, (helped + 0.5 * neutral) / total - 0.25 * harmed / total)))


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _mapping_float(value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if isinstance(raw, (int, float))}


def _mean_mapping(value: Mapping[str, float]) -> float:
    return float(mean(value.values())) if value else 0.0


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _agi_metric(run: Mapping[str, object], name: str) -> float:
    agi = _mapping(run.get("agi_relevance_metrics")) or {}
    value = agi.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _run_id(run: Mapping[str, object]) -> str:
    payload = {"benchmark": run.get("benchmark", ""), "mode": run.get("mode", ""), "seed": run.get("seed", 0)}
    return stable_hash(payload, prefix="run_")
