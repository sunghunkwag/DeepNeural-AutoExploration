"""Search-process diagnosis for bounded autonomous neural exploration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Mapping, Sequence


TASK_FAMILIES = (
    "function_regression",
    "sequence_prediction",
    "object_state_transition",
    "causal_intervention",
    "memory_retrieval",
)


@dataclass(frozen=True)
class SearchProcessDiagnosis:
    depth_usefulness_by_level: Dict[str, float] = field(default_factory=dict)
    mutation_success_rate_by_method: Dict[str, float] = field(default_factory=dict)
    rollback_rate_by_method: Dict[str, float] = field(default_factory=dict)
    validation_hidden_mismatch_by_method: Dict[str, float] = field(default_factory=dict)
    residue_resolution_rate_by_failure_type: Dict[str, float] = field(default_factory=dict)
    persistent_residue_types: list[str] = field(default_factory=list)
    probe_actionability_score: float = 0.0
    evaluator_overfit_risk: float = 0.0
    search_space_expansion_precision: float = 0.0
    architecture_complexity_growth: float = 0.0
    task_family_balance: float = 0.0
    transfer_regression_risk: float = 0.0
    task_family_hidden_losses: Dict[str, float] = field(default_factory=dict)
    residue_concentration_by_task_family: Dict[str, float] = field(default_factory=dict)
    mutation_success_by_task_family: Dict[str, float] = field(default_factory=dict)
    transfer_regression_by_task_family: Dict[str, float] = field(default_factory=dict)
    heldout_used_only_for_post_freeze_diagnosis: bool = True
    used_heldout_for_candidate_selection: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "depth_usefulness_by_level": dict(self.depth_usefulness_by_level),
            "mutation_success_rate_by_method": dict(self.mutation_success_rate_by_method),
            "rollback_rate_by_method": dict(self.rollback_rate_by_method),
            "validation_hidden_mismatch_by_method": dict(self.validation_hidden_mismatch_by_method),
            "residue_resolution_rate_by_failure_type": dict(self.residue_resolution_rate_by_failure_type),
            "persistent_residue_types": list(self.persistent_residue_types),
            "probe_actionability_score": self.probe_actionability_score,
            "evaluator_overfit_risk": self.evaluator_overfit_risk,
            "search_space_expansion_precision": self.search_space_expansion_precision,
            "architecture_complexity_growth": self.architecture_complexity_growth,
            "task_family_balance": self.task_family_balance,
            "transfer_regression_risk": self.transfer_regression_risk,
            "task_family_hidden_losses": dict(self.task_family_hidden_losses),
            "residue_concentration_by_task_family": dict(self.residue_concentration_by_task_family),
            "mutation_success_by_task_family": dict(self.mutation_success_by_task_family),
            "transfer_regression_by_task_family": dict(self.transfer_regression_by_task_family),
            "heldout_used_only_for_post_freeze_diagnosis": self.heldout_used_only_for_post_freeze_diagnosis,
            "used_heldout_for_candidate_selection": self.used_heldout_for_candidate_selection,
        }


class SearchProcessModel:
    """Summarize whether the search process itself is behaving usefully."""

    def diagnose(self, result: Mapping[str, object]) -> SearchProcessDiagnosis:
        decisions = _list_of_mappings(result.get("architecture_decisions", []))
        deep_decisions = _list_of_mappings(result.get("deep_search_decisions", []))
        residues = _list_of_mappings(result.get("failure_residues", []))
        residues_before = _list_of_mappings(result.get("failure_residues_before", []))
        expansions = _list_of_mappings(result.get("search_space_expansions", []))
        selection_components = _list_of_mappings(result.get("selection_score_components", []))
        base_genome = result.get("base_genome", {})
        final_genome = result.get("final_genome", {})
        final_metrics = result.get("final_metrics", result.get("metrics", {}))
        base_metrics = result.get("base_metrics", {})
        heldout = result.get("heldout_test_metrics_after_freeze", {})
        metrics = result.get("metrics", {})

        used_heldout_for_selection = any(bool(decision.get("used_heldout_labels", False)) for decision in decisions)
        depth_by_decision_id = {
            str(item.get("decision_id", "")): int(item.get("selected_depth_level", 0))
            for item in deep_decisions
        }
        depth_usefulness = _depth_usefulness(decisions, depth_by_decision_id)
        success, rollback, mismatch = _method_rates(decisions)
        residue_resolution = _residue_resolution_by_type(residues_before, residues)
        persistent = sorted(name for name, count in Counter(str(item.get("failure_type", "")) for item in residues).items() if name and count >= 2)
        probe_actionability = _probe_actionability(result, residues)
        evaluator_overfit = _evaluator_overfit_risk(decisions, selection_components)
        expansion_precision = _expansion_precision(expansions)
        complexity_growth = _complexity_growth(base_genome, final_genome)
        hidden_losses = _mapping_float(final_metrics, "hidden_validation_loss_by_family")
        balance = _balance_score(hidden_losses)
        transfer_risk, transfer_by_family = _transfer_regression_risk(base_metrics, final_metrics, heldout)
        residue_concentration = _residue_concentration_by_family(residues)
        mutation_success_by_family = _mutation_success_by_family(decisions)

        return SearchProcessDiagnosis(
            depth_usefulness_by_level=depth_usefulness,
            mutation_success_rate_by_method=success,
            rollback_rate_by_method=rollback,
            validation_hidden_mismatch_by_method=mismatch,
            residue_resolution_rate_by_failure_type=residue_resolution,
            persistent_residue_types=persistent,
            probe_actionability_score=probe_actionability,
            evaluator_overfit_risk=evaluator_overfit,
            search_space_expansion_precision=expansion_precision,
            architecture_complexity_growth=complexity_growth,
            task_family_balance=balance,
            transfer_regression_risk=transfer_risk,
            task_family_hidden_losses=hidden_losses,
            residue_concentration_by_task_family=residue_concentration,
            mutation_success_by_task_family=mutation_success_by_family,
            transfer_regression_by_task_family=transfer_by_family,
            heldout_used_only_for_post_freeze_diagnosis=bool(_mapping_bool(heldout, "heldout_test_evaluated_after_freeze", True)),
            used_heldout_for_candidate_selection=used_heldout_for_selection,
        )


def compute_process_improvement_components(
    first: SearchProcessDiagnosis | Mapping[str, object],
    second: SearchProcessDiagnosis | Mapping[str, object],
    first_result: Mapping[str, object] | None = None,
    second_result: Mapping[str, object] | None = None,
) -> Dict[str, float]:
    """Strict signed decomposition of second-run process change.

    Positive deltas mean the second run increased a desirable process signal.
    Risk deltas remain signed and are subtracted from the final score.
    """

    first_diag = _diagnosis_to_mapping(first)
    second_diag = _diagnosis_to_mapping(second)
    hidden_validation_robustness_delta = _run_metric(
        second_result,
        "hidden_validation_robustness",
        _derived_hidden_validation_robustness(second_diag),
    ) - _run_metric(
        first_result,
        "hidden_validation_robustness",
        _derived_hidden_validation_robustness(first_diag),
    )
    residue_resolution_delta = _run_metric(
        second_result,
        "residue_resolution_rate",
        _mean_mapping(second_diag.get("residue_resolution_rate_by_failure_type", {})),
    ) - _run_metric(
        first_result,
        "residue_resolution_rate",
        _mean_mapping(first_diag.get("residue_resolution_rate_by_failure_type", {})),
    )
    task_family_balance_delta = _diag_float(second_diag, "task_family_balance") - _diag_float(first_diag, "task_family_balance")
    rollback_rate_delta = _aggregate_rollback_rate(second_diag, second_result) - _aggregate_rollback_rate(first_diag, first_result)
    transfer_regression_risk_delta = _diag_float(second_diag, "transfer_regression_risk") - _diag_float(first_diag, "transfer_regression_risk")
    complexity_growth_delta = _diag_float(second_diag, "architecture_complexity_growth") - _diag_float(first_diag, "architecture_complexity_growth")
    evaluator_overfit_risk_delta = _diag_float(second_diag, "evaluator_overfit_risk") - _diag_float(first_diag, "evaluator_overfit_risk")
    final = (
        hidden_validation_robustness_delta
        + residue_resolution_delta
        + task_family_balance_delta
        - rollback_rate_delta
        - transfer_regression_risk_delta
        - complexity_growth_delta
        - evaluator_overfit_risk_delta
    )
    return {
        "hidden_validation_robustness_delta": float(hidden_validation_robustness_delta),
        "residue_resolution_delta": float(residue_resolution_delta),
        "task_family_balance_delta": float(task_family_balance_delta),
        "rollback_rate_delta": float(rollback_rate_delta),
        "transfer_regression_risk_delta": float(transfer_regression_risk_delta),
        "complexity_growth_delta": float(complexity_growth_delta),
        "evaluator_overfit_risk_delta": float(evaluator_overfit_risk_delta),
        "final_process_improvement_delta": float(final),
    }


def _depth_usefulness(decisions: Sequence[Mapping[str, object]], depth_by_decision_id: Mapping[str, int]) -> Dict[str, float]:
    values: Dict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        depth = depth_by_decision_id.get(str(decision.get("deep_search_decision_id", "")), 0)
        values[str(depth)].append(float(decision.get("selection_score_delta", 0.0)))
    return {level: float(mean(items)) for level, items in values.items() if items}


def _method_rates(decisions: Sequence[Mapping[str, object]]) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    counts: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    rollbacks: Counter[str] = Counter()
    mismatch: Dict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        method = str(decision.get("mutation_method", "unknown"))
        counts[method] += 1
        if bool(decision.get("accepted")):
            accepted[method] += 1
        else:
            rollbacks[method] += 1
        validation = float(decision.get("validation_improvement", 0.0))
        hidden = float(decision.get("hidden_validation_improvement", 0.0))
        mismatch[method].append(max(0.0, validation - hidden))
    success = {method: accepted[method] / max(1, count) for method, count in counts.items()}
    rollback = {method: rollbacks[method] / max(1, count) for method, count in counts.items()}
    mismatch_mean = {method: float(mean(values)) for method, values in mismatch.items()}
    return success, rollback, mismatch_mean


def _residue_resolution_by_type(before: Sequence[Mapping[str, object]], after: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    before_counts = Counter(str(item.get("failure_type", "")) for item in before if item.get("failure_type"))
    after_counts = Counter(str(item.get("failure_type", "")) for item in after if item.get("failure_type"))
    if not before_counts:
        before_counts = after_counts
    return {
        name: float(max(0.0, min(1.0, (count - after_counts.get(name, 0)) / max(1, count))))
        for name, count in before_counts.items()
    }


def _probe_actionability(result: Mapping[str, object], residues: Sequence[Mapping[str, object]]) -> float:
    traces = 0
    actionable = 0
    for key in ("representation_probe_results", "gradient_probe_results"):
        probe = result.get(key, {})
        if isinstance(probe, Mapping):
            modes = list(probe.get("failure_modes", []) or [])
            recommendations = probe.get("recommended_mutations", {})
            for mode in modes:
                traces += 1
                if isinstance(recommendations, Mapping) and recommendations.get(mode):
                    actionable += 1
    for residue in residues:
        traces += 1
        if residue.get("proposed_operator_or_module_family"):
            actionable += 1
    return float(actionable / max(1, traces))


def _evaluator_overfit_risk(decisions: Sequence[Mapping[str, object]], components: Sequence[Mapping[str, object]]) -> float:
    mismatch_events = 0
    positive_validation = 0
    for decision in decisions:
        validation = float(decision.get("validation_improvement", 0.0))
        hidden = float(decision.get("hidden_validation_improvement", 0.0))
        if validation > 0.0:
            positive_validation += 1
            if hidden <= 0.0:
                mismatch_events += 1
    component_gap = []
    for item in components:
        raw = item.get("components", {})
        if isinstance(raw, Mapping):
            val = float(raw.get("validation_loss", 0.0))
            hidden = float(raw.get("hidden_validation_loss", 0.0))
            component_gap.append(max(0.0, hidden - val) / (1.0 + val))
    mismatch_rate = mismatch_events / max(1, positive_validation)
    gap = mean(component_gap) if component_gap else 0.0
    return float(max(0.0, min(1.0, 0.65 * mismatch_rate + 0.35 * gap)))


def _expansion_precision(expansions: Sequence[Mapping[str, object]]) -> float:
    if not expansions:
        return 0.0
    values = []
    for item in expansions:
        confidence = float(item.get("confidence_score", 0.0))
        residue_ids = item.get("justifying_residue_ids", [])
        values.append(confidence if residue_ids else 0.0)
    return float(mean(values)) if values else 0.0


def _complexity_growth(base: object, final: object) -> float:
    if not isinstance(base, Mapping) or not isinstance(final, Mapping):
        return 0.0
    base_value = _genome_complexity(base)
    final_value = _genome_complexity(final)
    return float((final_value - base_value) / max(1.0, base_value))


def _genome_complexity(genome: Mapping[str, object]) -> float:
    return float(
        int(genome.get("hidden_dim", 0))
        + 4 * int(genome.get("residual_blocks", 0))
        + 5 * int(genome.get("memory_gates", 0))
        + 5 * int(genome.get("causal_bottlenecks", 0))
        + 5 * int(genome.get("object_binding_adapters", 0))
        + 5 * int(genome.get("sequence_state_adapters", 0))
    )


def _balance_score(values: Mapping[str, float]) -> float:
    if not values:
        return 0.0
    losses = [float(value) for value in values.values()]
    avg = mean(losses)
    variance = mean((value - avg) ** 2 for value in losses)
    return float(1.0 / (1.0 + variance / max(1e-8, abs(avg))))


def _transfer_regression_risk(
    base_metrics: object,
    final_metrics: object,
    heldout: object,
) -> tuple[float, Dict[str, float]]:
    final_map = final_metrics if isinstance(final_metrics, Mapping) else {}
    base_map = base_metrics if isinstance(base_metrics, Mapping) else {}
    heldout_map = heldout if isinstance(heldout, Mapping) else {}
    final_hidden = _mapping_float(final_map, "hidden_validation_loss_by_family")
    heldout_losses = _mapping_float(heldout_map, "heldout_test_loss_by_family")
    base_heldout = _mapping_float(base_map, "heldout_test_loss_by_family")
    by_family: Dict[str, float] = {}
    for family, loss in heldout_losses.items():
        hidden = final_hidden.get(family, loss)
        base_loss = base_heldout.get(family, loss)
        by_family[family] = max(0.0, loss - min(hidden, base_loss))
    risk = mean(by_family.values()) if by_family else 0.0
    return float(max(0.0, min(1.0, risk))), by_family


def _residue_concentration_by_family(residues: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for residue in residues:
        task_id = str(residue.get("task_id", ""))
        for family in TASK_FAMILIES:
            if family in task_id:
                counts[family] += 1
    total = sum(counts.values())
    return {family: counts[family] / max(1, total) for family in TASK_FAMILIES}


def _mutation_success_by_family(decisions: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    values: Dict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        improvements = decision.get("task_family_regressions", {})
        hidden_metrics = decision.get("hidden_validation_metrics", {})
        accepted = 1.0 if bool(decision.get("accepted")) else 0.0
        if isinstance(hidden_metrics, Mapping):
            for family in TASK_FAMILIES:
                if family in hidden_metrics:
                    penalty = 1.0 if isinstance(improvements, Mapping) and family in improvements else 0.0
                    values[family].append(max(0.0, accepted - penalty))
    return {family: float(mean(items)) if items else 0.0 for family, items in values.items()}


def _mapping_float(payload: object, key: str) -> Dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get(key, {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(name): float(value) for name, value in raw.items() if isinstance(value, (int, float))}


def _mapping_bool(payload: object, key: str, default: bool) -> bool:
    if not isinstance(payload, Mapping):
        return default
    return bool(payload.get(key, default))


def _diagnosis_to_mapping(value: SearchProcessDiagnosis | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(value, SearchProcessDiagnosis):
        return value.to_dict()
    return value


def _diag_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _run_metric(result: Mapping[str, object] | None, name: str, default: float) -> float:
    if isinstance(result, Mapping):
        agi = result.get("agi_relevance_metrics", {})
        if isinstance(agi, Mapping) and isinstance(agi.get(name), (int, float)):
            return float(agi[name])
        metrics = result.get("metrics", {})
        if isinstance(metrics, Mapping) and isinstance(metrics.get(name), (int, float)):
            return float(metrics[name])
    return float(default)


def _derived_hidden_validation_robustness(diagnosis: Mapping[str, object]) -> float:
    return float(max(0.0, min(1.0, 1.0 - _diag_float(diagnosis, "evaluator_overfit_risk"))))


def _mean_mapping(value: object) -> float:
    if not isinstance(value, Mapping) or not value:
        return 0.0
    values = [float(item) for item in value.values() if isinstance(item, (int, float))]
    return float(mean(values)) if values else 0.0


def _aggregate_rollback_rate(diagnosis: Mapping[str, object], result: Mapping[str, object] | None = None) -> float:
    if isinstance(result, Mapping):
        decisions = _list_of_mappings(result.get("architecture_decisions", []))
        if decisions:
            rollbacks = sum(1 for item in decisions if bool(item.get("rollback", not bool(item.get("accepted", False)))))
            return float(rollbacks / max(1, len(decisions)))
    return _mean_mapping(diagnosis.get("rollback_rate_by_method", {}))


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]

