"""Attribution for meta-meta-meta next-run configuration changes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Mapping

from .architecture_genome import stable_hash
from .search_process_model import SearchProcessDiagnosis, SearchProcessModel, compute_process_improvement_components
from .search_strategy_mutator import RESIDUE_TO_MUTATION


@dataclass(frozen=True)
class ConfigChangeRecord:
    change_id: str
    config_path: str
    old_value: object
    new_value: object
    change_type: str
    intended_effect: str
    observed_effect: Dict[str, object]
    attribution_confidence: float
    helped: bool
    harmed: bool
    neutral: bool
    evidence_used: Dict[str, object]
    used_heldout_for_candidate_selection: bool = False
    heldout_used_only_for_post_freeze_diagnosis: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "change_id": self.change_id,
            "config_path": self.config_path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_type": self.change_type,
            "intended_effect": self.intended_effect,
            "observed_effect": dict(self.observed_effect),
            "attribution_confidence": self.attribution_confidence,
            "helped": self.helped,
            "harmed": self.harmed,
            "neutral": self.neutral,
            "evidence_used": dict(self.evidence_used),
            "used_heldout_for_candidate_selection": self.used_heldout_for_candidate_selection,
            "heldout_used_only_for_post_freeze_diagnosis": self.heldout_used_only_for_post_freeze_diagnosis,
        }


@dataclass(frozen=True)
class ConfigChangeAttributionReport:
    attribution_id: str
    first_run_id: str
    second_run_id: str
    changes: list[ConfigChangeRecord]
    helped_changes: list[str]
    harmed_changes: list[str]
    neutral_changes: list[str]
    process_improvement_components: Dict[str, float]
    evidence_used: Dict[str, object]
    used_heldout_for_candidate_selection: bool = False
    heldout_used_only_for_post_freeze_diagnosis: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "attribution_id": self.attribution_id,
            "first_run_id": self.first_run_id,
            "second_run_id": self.second_run_id,
            "changes": [change.to_dict() for change in self.changes],
            "helped_changes": list(self.helped_changes),
            "harmed_changes": list(self.harmed_changes),
            "neutral_changes": list(self.neutral_changes),
            "process_improvement_components": dict(self.process_improvement_components),
            "evidence_used": dict(self.evidence_used),
            "used_heldout_for_candidate_selection": self.used_heldout_for_candidate_selection,
            "heldout_used_only_for_post_freeze_diagnosis": self.heldout_used_only_for_post_freeze_diagnosis,
        }


class MetaConfigAttribution:
    """Compare a recommended next-run config against the observed second run."""

    def attribute(
        self,
        first_run: Mapping[str, object],
        second_run: Mapping[str, object],
        *,
        first_next_run_config: Mapping[str, object] | None = None,
        second_applied_next_run_config: Mapping[str, object] | None = None,
        first_diagnosis: SearchProcessDiagnosis | Mapping[str, object] | None = None,
        second_diagnosis: SearchProcessDiagnosis | Mapping[str, object] | None = None,
    ) -> ConfigChangeAttributionReport:
        first_next = _mapping(first_next_run_config) or _mapping(first_run.get("next_run_config_recommendation"))
        second_applied = _mapping(second_applied_next_run_config) or _mapping(second_run.get("applied_next_run_config"))
        new_config = second_applied or first_next or {}
        old_config = _old_config_from_first_run(first_run)
        first_diag_obj = _diagnosis(first_run, first_diagnosis)
        second_diag_obj = _diagnosis(second_run, second_diagnosis)
        process_components = compute_process_improvement_components(first_diag_obj, second_diag_obj, first_run, second_run)
        changes: list[ConfigChangeRecord] = []
        for config_path, old_value, new_value in _iter_targeted_changes(old_config, new_config):
            record = self._record_change(
                config_path,
                old_value,
                new_value,
                first_run=first_run,
                second_run=second_run,
                first_diagnosis=first_diag_obj.to_dict(),
                second_diagnosis=second_diag_obj.to_dict(),
                process_components=process_components,
                applied_matches_recommendation=(not first_next or not second_applied or first_next == second_applied),
            )
            changes.append(record)
        payload = {
            "first": _run_id(first_run),
            "second": _run_id(second_run),
            "changes": [change.to_dict() for change in changes],
        }
        return ConfigChangeAttributionReport(
            attribution_id=stable_hash(payload, prefix="attr_"),
            first_run_id=_run_id(first_run),
            second_run_id=_run_id(second_run),
            changes=changes,
            helped_changes=[change.config_path for change in changes if change.helped],
            harmed_changes=[change.config_path for change in changes if change.harmed],
            neutral_changes=[change.config_path for change in changes if change.neutral],
            process_improvement_components=process_components,
            evidence_used={
                "first_next_run_config_present": bool(first_next),
                "second_applied_next_run_config_present": bool(second_applied),
                "applied_matches_recommendation": (not first_next or not second_applied or first_next == second_applied),
                "first_used_heldout_for_candidate_selection": first_diag_obj.used_heldout_for_candidate_selection,
                "second_used_heldout_for_candidate_selection": second_diag_obj.used_heldout_for_candidate_selection,
            },
            used_heldout_for_candidate_selection=False,
            heldout_used_only_for_post_freeze_diagnosis=True,
        )

    def _record_change(
        self,
        config_path: str,
        old_value: object,
        new_value: object,
        *,
        first_run: Mapping[str, object],
        second_run: Mapping[str, object],
        first_diagnosis: Mapping[str, object],
        second_diagnosis: Mapping[str, object],
        process_components: Mapping[str, float],
        applied_matches_recommendation: bool,
    ) -> ConfigChangeRecord:
        change_type = _change_type(old_value, new_value)
        intended = _intended_effect(config_path, old_value, new_value)
        observed = _observed_effect(config_path, old_value, new_value, first_run, second_run, first_diagnosis, second_diagnosis, process_components)
        helped = bool(observed.get("helped", False))
        harmed = bool(observed.get("harmed", False))
        neutral = not helped and not harmed
        confidence = _confidence(config_path, observed, applied_matches_recommendation)
        payload = {
            "path": config_path,
            "old": old_value,
            "new": new_value,
            "observed": observed,
        }
        return ConfigChangeRecord(
            change_id=stable_hash(payload, prefix="cfgchg_"),
            config_path=config_path,
            old_value=old_value,
            new_value=new_value,
            change_type=change_type,
            intended_effect=intended,
            observed_effect={key: value for key, value in observed.items() if key not in {"helped", "harmed"}},
            attribution_confidence=confidence,
            helped=helped,
            harmed=harmed,
            neutral=neutral,
            evidence_used={
                "first_search_process_diagnosis": dict(first_diagnosis),
                "second_search_process_diagnosis": dict(second_diagnosis),
                "process_improvement_components": dict(process_components),
                "applied_matches_recommendation": applied_matches_recommendation,
            },
            used_heldout_for_candidate_selection=False,
            heldout_used_only_for_post_freeze_diagnosis=True,
        )


def _old_config_from_first_run(first_run: Mapping[str, object]) -> Dict[str, object]:
    cfg = _mapping(first_run.get("config"))
    return {
        "search_strategy_config": _plan_base(first_run, "search_strategy_mutation_plan"),
        "evaluator_objective_config": _plan_base(first_run, "evaluator_objective_mutation_plan"),
        "curriculum_strategy_config": _plan_base(first_run, "curriculum_mutation_plan"),
        "recommended_generations": int(cfg.get("generations", 1)) if cfg else 1,
        "recommended_candidate_count": int(cfg.get("candidate_count", 3)) if cfg else 3,
        "recommended_focus_failure_types": [],
        "recommended_focus_task_families": [],
    }


def _plan_base(first_run: Mapping[str, object], key: str) -> Dict[str, object]:
    plan = _mapping(first_run.get(key))
    base = _mapping(plan.get("base_config")) if plan else None
    return dict(base or {})


def _iter_targeted_changes(old_config: Mapping[str, object], new_config: Mapping[str, object]) -> list[tuple[str, object, object]]:
    changes: list[tuple[str, object, object]] = []
    section_keys = ("search_strategy_config", "evaluator_objective_config", "curriculum_strategy_config")
    for section in section_keys:
        old_section = _mapping(old_config.get(section))
        new_section = _mapping(new_config.get(section))
        for suffix, old_value, new_value in _diff_nested(old_section or {}, new_section or {}):
            changes.append((f"{section}.{suffix}", old_value, new_value))
    for key in ("recommended_generations", "recommended_candidate_count"):
        if old_config.get(key) != new_config.get(key):
            changes.append((key, old_config.get(key), new_config.get(key)))
    for source_key, path in (
        ("recommended_focus_failure_types", "focus_failure_types"),
        ("recommended_focus_task_families", "focus_task_families"),
    ):
        old_value = list(old_config.get(source_key, []) or [])
        new_value = list(new_config.get(source_key, []) or [])
        if old_value != new_value:
            changes.append((path, old_value, new_value))
    return changes


def _diff_nested(old: Mapping[str, object], new: Mapping[str, object], prefix: str = "") -> list[tuple[str, object, object]]:
    keys = sorted(set(old) | set(new))
    changes: list[tuple[str, object, object]] = []
    for key in keys:
        path = f"{prefix}.{key}" if prefix else str(key)
        old_value = old.get(key)
        new_value = new.get(key)
        if isinstance(old_value, Mapping) and isinstance(new_value, Mapping):
            changes.extend(_diff_nested(old_value, new_value, path))
        elif old_value != new_value:
            changes.append((path, old_value, new_value))
    return changes


def _intended_effect(config_path: str, old_value: object, new_value: object) -> str:
    if "hidden_validation" in config_path or "generalization_gap" in config_path:
        return "reduce validation-hidden mismatch and improve hidden-validation robustness"
    if ".mutation_penalties." in config_path or config_path.endswith("rollback_risk_weight"):
        method = config_path.rsplit(".", 1)[-1]
        return f"reduce rollback rate for {method}"
    if ".mutation_boosts." in config_path:
        method = config_path.rsplit(".", 1)[-1]
        failures = _failures_for_method(method)
        if failures:
            return "reduce persistent residues for " + ", ".join(failures)
        return f"improve residue resolution through {method}"
    if config_path.startswith("curriculum_strategy_config") or config_path == "focus_task_families":
        return "improve task-family balance without increasing transfer regression"
    if config_path == "focus_failure_types":
        return "increase resolution pressure on persistent residue types"
    if config_path in {"recommended_generations", "recommended_candidate_count"}:
        direction = "increase" if _numeric(new_value) > _numeric(old_value) else "decrease"
        return f"{direction} search effort while monitoring rollback and complexity growth"
    if "complexity" in config_path:
        return "control architecture complexity growth"
    if "transfer" in config_path:
        return "reduce transfer regression risk"
    return "improve closed-loop search process quality"


def _observed_effect(
    config_path: str,
    old_value: object,
    new_value: object,
    first_run: Mapping[str, object],
    second_run: Mapping[str, object],
    first_diagnosis: Mapping[str, object],
    second_diagnosis: Mapping[str, object],
    process_components: Mapping[str, float],
) -> Dict[str, object]:
    effect: Dict[str, object] = {"config_path": config_path}
    if "hidden_validation" in config_path or "generalization_gap" in config_path:
        first_mismatch = _mean_mapping(first_diagnosis.get("validation_hidden_mismatch_by_method", {}))
        second_mismatch = _mean_mapping(second_diagnosis.get("validation_hidden_mismatch_by_method", {}))
        mismatch_delta = second_mismatch - first_mismatch
        overfit_delta = _float(second_diagnosis.get("evaluator_overfit_risk")) - _float(first_diagnosis.get("evaluator_overfit_risk"))
        robustness_delta = float(process_components.get("hidden_validation_robustness_delta", 0.0))
        effect.update(
            {
                "validation_hidden_mismatch_delta": mismatch_delta,
                "evaluator_overfit_risk_delta": overfit_delta,
                "hidden_validation_robustness_delta": robustness_delta,
            }
        )
        effect["helped"] = mismatch_delta < -1e-9 or overfit_delta < -1e-9 or robustness_delta > 1e-9
        effect["harmed"] = mismatch_delta > 1e-9 and overfit_delta > 1e-9 and robustness_delta < -1e-9
        return effect
    if ".mutation_penalties." in config_path:
        method = config_path.rsplit(".", 1)[-1]
        first_rate = _mapping_float(first_diagnosis.get("rollback_rate_by_method", {})).get(method, 0.0)
        second_rate = _mapping_float(second_diagnosis.get("rollback_rate_by_method", {})).get(method, 0.0)
        delta = second_rate - first_rate
        effect.update({"mutation_method": method, "rollback_rate_delta": delta, "first_rate": first_rate, "second_rate": second_rate})
        effect["helped"] = delta < -1e-9
        effect["harmed"] = delta > 1e-9
        return effect
    if ".mutation_boosts." in config_path:
        method = config_path.rsplit(".", 1)[-1]
        failures = _failures_for_method(method)
        first_resolution = _failure_resolution(first_diagnosis, failures)
        second_resolution = _failure_resolution(second_diagnosis, failures)
        resolution_delta = second_resolution - first_resolution
        first_persistent = _persistent_count(first_run, failures)
        second_persistent = _persistent_count(second_run, failures)
        persistent_delta = second_persistent - first_persistent
        effect.update(
            {
                "mutation_method": method,
                "target_failure_types": failures,
                "residue_resolution_delta": resolution_delta,
                "persistent_residue_count_delta": persistent_delta,
            }
        )
        effect["helped"] = resolution_delta > 1e-9 or persistent_delta < 0
        effect["harmed"] = resolution_delta < -1e-9 and persistent_delta > 0
        return effect
    if config_path == "focus_failure_types":
        failures = [str(item) for item in new_value] if isinstance(new_value, list) else []
        first_resolution = _failure_resolution(first_diagnosis, failures)
        second_resolution = _failure_resolution(second_diagnosis, failures)
        resolution_delta = second_resolution - first_resolution
        first_persistent = _persistent_count(first_run, failures)
        second_persistent = _persistent_count(second_run, failures)
        persistent_delta = second_persistent - first_persistent
        effect.update({"target_failure_types": failures, "residue_resolution_delta": resolution_delta, "persistent_residue_count_delta": persistent_delta})
        effect["helped"] = resolution_delta > 1e-9 or persistent_delta < 0
        effect["harmed"] = resolution_delta < -1e-9 and persistent_delta > 0
        return effect
    if config_path.startswith("curriculum_strategy_config") or config_path == "focus_task_families":
        balance_delta = float(process_components.get("task_family_balance_delta", 0.0))
        transfer_delta = float(process_components.get("transfer_regression_risk_delta", 0.0))
        effect.update({"task_family_balance_delta": balance_delta, "transfer_regression_risk_delta": transfer_delta})
        effect["helped"] = balance_delta > 1e-9 and transfer_delta <= 1e-9
        effect["harmed"] = balance_delta < -1e-9 or transfer_delta > 1e-9
        return effect
    if config_path in {"recommended_generations", "recommended_candidate_count"}:
        residue_delta = float(process_components.get("residue_resolution_delta", 0.0))
        rollback_delta = float(process_components.get("rollback_rate_delta", 0.0))
        complexity_delta = float(process_components.get("complexity_growth_delta", 0.0))
        effect.update({"residue_resolution_delta": residue_delta, "rollback_rate_delta": rollback_delta, "complexity_growth_delta": complexity_delta})
        effect["helped"] = residue_delta > 1e-9 and rollback_delta <= 1e-9 and complexity_delta <= 1e-9
        effect["harmed"] = rollback_delta > 1e-9 or complexity_delta > 1e-9
        return effect
    final_delta = float(process_components.get("final_process_improvement_delta", 0.0))
    effect.update({"final_process_improvement_delta": final_delta})
    effect["helped"] = final_delta > 1e-9
    effect["harmed"] = final_delta < -1e-9
    return effect


def _confidence(config_path: str, observed: Mapping[str, object], applied_matches_recommendation: bool) -> float:
    direct = any(token in config_path for token in ("hidden_validation", "mutation_penalties", "mutation_boosts", "focus_failure_types"))
    value = 0.62 if direct else 0.48
    if applied_matches_recommendation:
        value += 0.08
    if bool(observed.get("helped")) or bool(observed.get("harmed")):
        value += 0.08
    return float(max(0.05, min(0.95, value)))


def _diagnosis(run: Mapping[str, object], value: SearchProcessDiagnosis | Mapping[str, object] | None) -> SearchProcessDiagnosis:
    if isinstance(value, SearchProcessDiagnosis):
        return value
    if isinstance(value, Mapping):
        return _diagnosis_from_mapping(value)
    existing = _mapping(run.get("search_process_diagnosis"))
    if existing:
        return _diagnosis_from_mapping(existing)
    return SearchProcessModel().diagnose(run)


def _diagnosis_from_mapping(payload: Mapping[str, object]) -> SearchProcessDiagnosis:
    return SearchProcessDiagnosis(
        depth_usefulness_by_level=_mapping_float(payload.get("depth_usefulness_by_level", {})),
        mutation_success_rate_by_method=_mapping_float(payload.get("mutation_success_rate_by_method", {})),
        rollback_rate_by_method=_mapping_float(payload.get("rollback_rate_by_method", {})),
        validation_hidden_mismatch_by_method=_mapping_float(payload.get("validation_hidden_mismatch_by_method", {})),
        residue_resolution_rate_by_failure_type=_mapping_float(payload.get("residue_resolution_rate_by_failure_type", {})),
        persistent_residue_types=[str(item) for item in payload.get("persistent_residue_types", [])] if isinstance(payload.get("persistent_residue_types"), list) else [],
        probe_actionability_score=_float(payload.get("probe_actionability_score")),
        evaluator_overfit_risk=_float(payload.get("evaluator_overfit_risk")),
        search_space_expansion_precision=_float(payload.get("search_space_expansion_precision")),
        architecture_complexity_growth=_float(payload.get("architecture_complexity_growth")),
        task_family_balance=_float(payload.get("task_family_balance")),
        transfer_regression_risk=_float(payload.get("transfer_regression_risk")),
        task_family_hidden_losses=_mapping_float(payload.get("task_family_hidden_losses", {})),
        residue_concentration_by_task_family=_mapping_float(payload.get("residue_concentration_by_task_family", {})),
        mutation_success_by_task_family=_mapping_float(payload.get("mutation_success_by_task_family", {})),
        transfer_regression_by_task_family=_mapping_float(payload.get("transfer_regression_by_task_family", {})),
        heldout_used_only_for_post_freeze_diagnosis=bool(payload.get("heldout_used_only_for_post_freeze_diagnosis", True)),
        used_heldout_for_candidate_selection=bool(payload.get("used_heldout_for_candidate_selection", False)),
    )


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _mapping_float(value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if isinstance(raw, (int, float))}


def _mean_mapping(value: object) -> float:
    vals = list(_mapping_float(value).values())
    return float(mean(vals)) if vals else 0.0


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _numeric(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _change_type(old_value: object, new_value: object) -> str:
    if old_value is None:
        return "add"
    if new_value is None:
        return "remove"
    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
        if float(new_value) > float(old_value):
            return "increase_numeric"
        if float(new_value) < float(old_value):
            return "decrease_numeric"
        return "same_numeric"
    if isinstance(old_value, list) and isinstance(new_value, list):
        old_set = set(map(str, old_value))
        new_set = set(map(str, new_value))
        if old_set < new_set:
            return "list_add"
        if new_set < old_set:
            return "list_remove"
        return "list_replace"
    return "replace"


def _failures_for_method(method: str) -> list[str]:
    return sorted(failure for failure, mapped_method in RESIDUE_TO_MUTATION.items() if mapped_method == method)


def _failure_resolution(diagnosis: Mapping[str, object], failures: list[str]) -> float:
    rates = _mapping_float(diagnosis.get("residue_resolution_rate_by_failure_type", {}))
    if not failures:
        return _mean_mapping(rates)
    values = [rates.get(failure, 0.0) for failure in failures]
    return float(mean(values)) if values else 0.0


def _persistent_count(run: Mapping[str, object], failures: list[str]) -> int:
    failure_set = set(failures)
    residues = run.get("failure_residues", [])
    if not isinstance(residues, list):
        return 0
    counts: Counter[str] = Counter()
    for residue in residues:
        if isinstance(residue, Mapping):
            failure = str(residue.get("failure_type", ""))
            if not failure_set or failure in failure_set:
                counts[failure] += 1
    return int(sum(counts.values()))


def _run_id(run: Mapping[str, object]) -> str:
    existing = run.get("run_id")
    if isinstance(existing, str) and existing:
        return existing
    payload = {
        "benchmark": run.get("benchmark", ""),
        "mode": run.get("mode", ""),
        "seed": run.get("seed", 0),
        "final_genome": (_mapping(run.get("final_genome")) or {}).get("config_hash", ""),
    }
    return stable_hash(payload, prefix="run_")
