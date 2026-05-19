"""Failure diagnosis for long-horizon autonomous exploration outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class FailureDiagnosisReport:
    rollback_root_causes: Dict[str, object]
    residue_resolution_failures: Dict[str, object]
    evaluator_overfit_causes: Dict[str, object]
    hidden_validation_robustness_causes: Dict[str, object]
    external_adapter_generalization_failure: Dict[str, object]
    memory_guidance_failure_modes: Dict[str, object]
    recommended_repairs: list[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "rollback_root_causes": dict(self.rollback_root_causes),
            "residue_resolution_failures": dict(self.residue_resolution_failures),
            "evaluator_overfit_causes": dict(self.evaluator_overfit_causes),
            "hidden_validation_robustness_causes": dict(self.hidden_validation_robustness_causes),
            "external_adapter_generalization_failure": dict(self.external_adapter_generalization_failure),
            "memory_guidance_failure_modes": dict(self.memory_guidance_failure_modes),
            "recommended_repairs": list(self.recommended_repairs),
        }


def diagnose_long_horizon_failure(
    long_horizon_result: Mapping[str, object],
    external_adapter_result: Mapping[str, object] | None = None,
) -> FailureDiagnosisReport:
    run_payloads = _load_run_payloads(long_horizon_result)
    attribution_reports = _list_of_mappings(long_horizon_result.get("attribution_reports", []))
    meta_decisions = _list_of_mappings(long_horizon_result.get("meta_decisions", []))
    rollback_by_method: Counter[str] = Counter()
    rollback_by_family: Counter[str] = Counter()
    hidden_decline_by_family: Dict[str, list[float]] = defaultdict(list)
    residue_counts_by_type: Dict[str, list[int]] = defaultdict(list)
    evaluator_weights: Dict[str, list[float]] = defaultdict(list)
    overfit_values: list[float] = []

    for run in run_payloads:
        decisions = _list_of_mappings(run.get("architecture_decisions", []))
        residues = _list_of_mappings(run.get("failure_residues", []))
        for decision in decisions:
            if bool(decision.get("rollback", not bool(decision.get("accepted", False)))):
                method = str(decision.get("mutation_method", "unknown"))
                rollback_by_method[method] += 1
                for family in _families_from_decision_or_residues(decision, residues):
                    rollback_by_family[family] += 1
            regressions = decision.get("task_family_regressions", {})
            if isinstance(regressions, Mapping):
                for family, value in regressions.items():
                    if isinstance(value, (int, float)):
                        hidden_decline_by_family[str(family)].append(float(value))
        residue_counts = Counter(str(item.get("failure_type", "")) for item in residues if item.get("failure_type"))
        for failure_type, count in residue_counts.items():
            residue_counts_by_type[failure_type].append(int(count))
        diagnosis = run.get("search_process_diagnosis", {})
        if isinstance(diagnosis, Mapping):
            overfit_values.append(_float(diagnosis.get("evaluator_overfit_risk")))
        plan = run.get("evaluator_objective_mutation_plan", {})
        mutated = plan.get("mutated_config", {}) if isinstance(plan, Mapping) else {}
        if isinstance(mutated, Mapping):
            for key, value in mutated.items():
                if isinstance(value, (int, float)):
                    evaluator_weights[str(key)].append(float(value))

    rollback_config_causes = _rollback_config_causes(attribution_reports)
    residue_failures = {
        failure_type: {
            "start_count": counts[0],
            "end_count": counts[-1],
            "worsened_or_persisted": counts[-1] >= counts[0],
        }
        for failure_type, counts in residue_counts_by_type.items()
        if counts
    }
    suspicious_weights = {
        key: {"start": values[0], "end": values[-1], "delta": values[-1] - values[0]}
        for key, values in evaluator_weights.items()
        if len(values) >= 2 and ("validation" in key or "hidden" in key or "rollback" in key or "transfer" in key)
    }
    memory_failure_modes = _memory_failure_modes(meta_decisions, attribution_reports)
    external_failure = _external_failure(external_adapter_result)
    repairs = _recommended_repairs(rollback_by_method, residue_failures, overfit_values, external_failure, memory_failure_modes)
    return FailureDiagnosisReport(
        rollback_root_causes={
            "rollback_causes_by_method": dict(rollback_by_method),
            "rollback_causes_by_task_family": dict(rollback_by_family),
            "rollback_causes_by_config_change": rollback_config_causes,
        },
        residue_resolution_failures=residue_failures,
        evaluator_overfit_causes={
            "start_evaluator_overfit_risk": overfit_values[0] if overfit_values else 0.0,
            "end_evaluator_overfit_risk": overfit_values[-1] if overfit_values else 0.0,
            "suspicious_objective_weight_changes": suspicious_weights,
        },
        hidden_validation_robustness_causes={
            "hidden_validation_decline_by_task_family": {
                family: float(mean(values)) for family, values in hidden_decline_by_family.items() if values
            },
        },
        external_adapter_generalization_failure=external_failure,
        memory_guidance_failure_modes=memory_failure_modes,
        recommended_repairs=repairs,
    )


def _load_run_payloads(result: Mapping[str, object]) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    paths = result.get("run_outputs", [])
    if not isinstance(paths, list):
        return out
    import json
    from pathlib import Path

    for path in paths:
        try:
            payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            out.append(payload)
    return out


def _rollback_config_causes(attributions: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for report in attributions:
        for change in _list_of_mappings(report.get("changes", [])):
            effect = change.get("observed_effect", {})
            if isinstance(effect, Mapping) and _float(effect.get("rollback_rate_delta")) > 0.0:
                counts[str(change.get("config_path", ""))] += 1
            elif bool(change.get("harmed")) and "rollback" in str(effect):
                counts[str(change.get("config_path", ""))] += 1
    return dict(counts)


def _memory_failure_modes(meta_decisions: Sequence[Mapping[str, object]], attributions: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    reused = Counter()
    avoided = Counter()
    harmed = set()
    helped = set()
    for report in attributions:
        for path in report.get("harmed_changes", []) if isinstance(report.get("harmed_changes", []), list) else []:
            harmed.add(str(path))
        for path in report.get("helped_changes", []) if isinstance(report.get("helped_changes", []), list) else []:
            helped.add(str(path))
    for decision in meta_decisions:
        for path in decision.get("reused_successful_config_patterns", []) if isinstance(decision.get("reused_successful_config_patterns", []), list) else []:
            reused[str(path)] += 1
        for path in decision.get("avoided_harmful_config_patterns", []) if isinstance(decision.get("avoided_harmful_config_patterns", []), list) else []:
            avoided[str(path)] += 1
    return {
        "reused_patterns_later_harmed": sorted(path for path in reused if path in harmed),
        "avoided_patterns_later_helped": sorted(path for path in avoided if path in helped),
        "reused_pattern_counts": dict(reused),
        "avoided_pattern_counts": dict(avoided),
    }


def _external_failure(result: Mapping[str, object] | None) -> Dict[str, object]:
    if not isinstance(result, Mapping):
        return {"available": False}
    return {
        "available": True,
        "validation_delta": _float(result.get("validation_delta")),
        "hidden_validation_delta": _float(result.get("hidden_validation_delta")),
        "heldout_after_freeze_delta": _float(result.get("heldout_after_freeze_delta")),
        "failed_generalization": _float(result.get("heldout_after_freeze_delta")) < 0.0,
        "external_failure_reason": result.get("external_failure_reason", ""),
    }


def _recommended_repairs(
    rollback_by_method: Counter[str],
    residue_failures: Mapping[str, object],
    overfit_values: Sequence[float],
    external_failure: Mapping[str, object],
    memory_modes: Mapping[str, object],
) -> list[str]:
    repairs = []
    if rollback_by_method:
        worst = rollback_by_method.most_common(1)[0][0]
        repairs.append(f"tighten rollback budget and raise mutation penalty for {worst}")
    if any(isinstance(value, Mapping) and value.get("worsened_or_persisted") for value in residue_failures.values()):
        repairs.append("track residue lifecycle and stop boosting failed residue repairs after rollback increases")
    if len(overfit_values) >= 2 and overfit_values[-1] > overfit_values[0]:
        repairs.append("apply evaluator mismatch guard: lower validation weight and raise hidden/rollback/task-family penalties")
    if external_failure.get("failed_generalization"):
        repairs.append("treat external heldout-after-freeze regression as transfer-risk evidence")
    if memory_modes.get("reused_patterns_later_harmed") or memory_modes.get("avoided_patterns_later_helped"):
        repairs.append("downgrade memory trust for rollback-correlated reuse and over-aggressive avoidance")
    return repairs


def _families_from_decision_or_residues(decision: Mapping[str, object], residues: Sequence[Mapping[str, object]]) -> list[str]:
    regressions = decision.get("task_family_regressions", {})
    if isinstance(regressions, Mapping) and regressions:
        return [str(key) for key in regressions]
    families = ("function_regression", "sequence_prediction", "object_state_transition", "causal_intervention", "memory_retrieval")
    found = []
    for residue in residues:
        task_id = str(residue.get("task_id", ""))
        for family in families:
            if family in task_id:
                found.append(family)
    return sorted(set(found)) or ["unknown"]


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
