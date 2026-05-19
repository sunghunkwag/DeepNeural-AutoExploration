"""Bounded mutation of search-strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

from .meta_config_memory import MetaConfigMemory, apply_memory_guidance_to_config
from .search_process_model import SearchProcessDiagnosis


RESIDUE_TO_MUTATION = {
    "object-binding failure": "object_binding_adapter",
    "task-family entanglement": "object_binding_adapter",
    "representation collapse": "wider_residual_stack",
    "architecture-representation bottleneck": "wider_residual_stack",
    "gradient bottleneck": "add_residual_block",
    "causal intervention failure": "add_causal_bottleneck",
    "sequence instability": "sequence_state_adapter",
    "dead module": "add_memory_gate",
    "over-compressed bottleneck": "repair_bottleneck",
}


@dataclass(frozen=True)
class SearchStrategyConfig:
    depth_thresholds: Dict[str, float] = field(default_factory=lambda: {"level1": 0.25, "level2": 0.55, "level3": 0.78, "level4": 0.88})
    mutation_penalties: Dict[str, float] = field(default_factory=dict)
    mutation_boosts: Dict[str, float] = field(default_factory=dict)
    validation_weight: float = 0.30
    hidden_validation_weight: float = 0.35
    transfer_weight: float = 0.12
    complexity_penalty_weight: float = 0.06
    rollback_risk_weight: float = 0.08
    probe_confirmation_required: bool = False
    exploration_temperature: float = 0.35
    stagnation_window: int = 3

    def to_dict(self) -> Dict[str, object]:
        return {
            "depth_thresholds": dict(self.depth_thresholds),
            "mutation_penalties": dict(self.mutation_penalties),
            "mutation_boosts": dict(self.mutation_boosts),
            "validation_weight": self.validation_weight,
            "hidden_validation_weight": self.hidden_validation_weight,
            "transfer_weight": self.transfer_weight,
            "complexity_penalty_weight": self.complexity_penalty_weight,
            "rollback_risk_weight": self.rollback_risk_weight,
            "probe_confirmation_required": self.probe_confirmation_required,
            "exploration_temperature": self.exploration_temperature,
            "stagnation_window": self.stagnation_window,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "SearchStrategyConfig":
        if not payload:
            return cls()
        base = cls()
        return cls(
            depth_thresholds=_bounded_float_map(payload.get("depth_thresholds"), base.depth_thresholds, 0.05, 0.95),
            mutation_penalties=_bounded_float_map(payload.get("mutation_penalties"), {}, 0.0, 1.0),
            mutation_boosts=_bounded_float_map(payload.get("mutation_boosts"), {}, 0.0, 1.0),
            validation_weight=_bounded_float(payload.get("validation_weight"), 0.0, 1.0, base.validation_weight),
            hidden_validation_weight=_bounded_float(payload.get("hidden_validation_weight"), 0.0, 1.0, base.hidden_validation_weight),
            transfer_weight=_bounded_float(payload.get("transfer_weight"), 0.0, 1.0, base.transfer_weight),
            complexity_penalty_weight=_bounded_float(payload.get("complexity_penalty_weight"), 0.0, 1.0, base.complexity_penalty_weight),
            rollback_risk_weight=_bounded_float(payload.get("rollback_risk_weight"), 0.0, 1.0, base.rollback_risk_weight),
            probe_confirmation_required=bool(payload.get("probe_confirmation_required", base.probe_confirmation_required)),
            exploration_temperature=_bounded_float(payload.get("exploration_temperature"), 0.0, 2.0, base.exploration_temperature),
            stagnation_window=max(1, min(12, int(payload.get("stagnation_window", base.stagnation_window)))),
        )


@dataclass(frozen=True)
class SearchStrategyMutationPlan:
    base_config: Dict[str, object]
    mutated_config: Dict[str, object]
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


class SearchStrategyMutator:
    def mutate(
        self,
        diagnosis: SearchProcessDiagnosis,
        base_config: SearchStrategyConfig | None = None,
        prior_meta_memory: MetaConfigMemory | None = None,
    ) -> SearchStrategyMutationPlan:
        base = base_config or SearchStrategyConfig()
        depth_thresholds = dict(base.depth_thresholds)
        penalties = dict(base.mutation_penalties)
        boosts = dict(base.mutation_boosts)
        validation_weight = base.validation_weight
        hidden_weight = base.hidden_validation_weight
        transfer_weight = base.transfer_weight
        complexity_weight = base.complexity_penalty_weight
        rollback_weight = base.rollback_risk_weight
        probe_required = base.probe_confirmation_required
        exploration = base.exploration_temperature
        selected: list[str] = []
        rationale: list[str] = []
        rollback_risk_budget = 0.65
        aggregate_rollback_rate = _mean(diagnosis.rollback_rate_by_method.values()) if diagnosis.rollback_rate_by_method else 0.0
        rollback_budget_exceeded = aggregate_rollback_rate > rollback_risk_budget

        deeper_level_failed = any(value < -0.02 for level, value in diagnosis.depth_usefulness_by_level.items() if int(level) >= 2)
        if _mean(diagnosis.depth_usefulness_by_level.values()) < 0.0 and diagnosis.persistent_residue_types and not deeper_level_failed:
            depth_thresholds = {key: max(0.05, value - 0.05) for key, value in depth_thresholds.items()}
            selected.append("decrease_depth_threshold")
            rationale.append("persistent residues remain while depth usefulness is non-positive")
        if deeper_level_failed:
            depth_thresholds["level2"] = min(0.95, depth_thresholds.get("level2", 0.55) + 0.05)
            depth_thresholds["level3"] = min(0.95, depth_thresholds.get("level3", 0.78) + 0.05)
            selected.append("increase_depth_threshold")
            rationale.append("deeper levels produced negative selection deltas")
        for method, rate in diagnosis.rollback_rate_by_method.items():
            if rate >= 0.5:
                penalties[method] = min(1.0, penalties.get(method, 0.0) + 0.30 * rate)
                if method in boosts:
                    boosts[method] = max(0.0, boosts.get(method, 0.0) - 0.15 * rate)
                selected.append("penalize_repeated_failed_mutation")
                rationale.append(f"{method} rollback rate {rate:.2f}")
        if rollback_budget_exceeded:
            rollback_weight = min(1.0, rollback_weight + 0.10)
            exploration = max(0.05, exploration - 0.10)
            depth_thresholds = {key: min(0.95, value + 0.03) for key, value in depth_thresholds.items()}
            selected.append("enforce_rollback_risk_budget")
            rationale.append(f"aggregate rollback rate {aggregate_rollback_rate:.2f} exceeded budget {rollback_risk_budget:.2f}")
        for failure_type in diagnosis.persistent_residue_types:
            method = RESIDUE_TO_MUTATION.get(failure_type)
            if method:
                method_rollback = diagnosis.rollback_rate_by_method.get(method, 0.0)
                if method_rollback > rollback_risk_budget:
                    penalties[method] = min(1.0, penalties.get(method, 0.0) + 0.20)
                    selected.append("defer_residue_boost_over_rollback_budget")
                    rationale.append(f"{failure_type} persists but {method} exceeded rollback budget")
                    continue
                boosts[method] = min(1.0, boosts.get(method, 0.0) + 0.20)
                selected.append("boost_mutation_for_persistent_residue")
                rationale.append(f"{failure_type} persists; boost {method}")
        if diagnosis.architecture_complexity_growth > 0.25:
            complexity_weight = min(1.0, complexity_weight + 0.05)
            selected.append("reduce_complexity_growth")
            rationale.append("architecture complexity grew faster than validation evidence justified")
        if diagnosis.evaluator_overfit_risk > 0.25:
            validation_weight = max(0.05, validation_weight - 0.05)
            hidden_weight = min(1.0, hidden_weight + 0.07)
            selected.extend(["increase_hidden_validation_weight", "reduce_validation_weight_if_mismatch_high"])
            rationale.append("validation-hidden mismatch indicates evaluator overfit risk")
        if not diagnosis.persistent_residue_types and max(diagnosis.mutation_success_rate_by_method.values() or [0.0]) > 0.0:
            exploration = max(0.05, exploration - 0.05)
            selected.append("increase_exploitation_after_hidden_validation_success")
            rationale.append("recent accepted candidates provide exploitable signal")
        elif diagnosis.persistent_residue_types and not any(rate > 0.0 for rate in diagnosis.mutation_success_rate_by_method.values()):
            exploration = min(2.0, exploration + 0.15)
            selected.append("increase_exploration_after_stagnation")
            rationale.append("persistent residues without accepted mutation success indicate stagnation")
        if diagnosis.probe_actionability_score < 0.50:
            probe_required = True
            selected.append("require_probe_confirmation_before_expansion")
            rationale.append("probe failures are not consistently producing actionable module recommendations")
        if diagnosis.transfer_regression_risk > 0.2:
            transfer_weight = min(1.0, transfer_weight + 0.05)
            selected.append("increase_hidden_validation_weight")
            rationale.append("post-freeze transfer regression risk should influence the next run")

        mutated = SearchStrategyConfig(
            depth_thresholds=depth_thresholds,
            mutation_penalties=penalties,
            mutation_boosts=boosts,
            validation_weight=validation_weight,
            hidden_validation_weight=hidden_weight,
            transfer_weight=transfer_weight,
            complexity_penalty_weight=complexity_weight,
            rollback_risk_weight=rollback_weight,
            probe_confirmation_required=probe_required,
            exploration_temperature=exploration,
            stagnation_window=base.stagnation_window,
        )
        guidance: Dict[str, object] = {}
        if prior_meta_memory is not None:
            adjusted, guidance = apply_memory_guidance_to_config(
                "search_strategy_config",
                base.to_dict(),
                mutated.to_dict(),
                prior_meta_memory,
            )
            if guidance.get("avoided_harmful_config_patterns"):
                selected.append("avoid_harmful_prior_config_pattern")
                rationale.append("prior meta-config memory marked one or more proposed strategy changes as harmful")
            if guidance.get("avoided_rollback_correlated_patterns"):
                selected.append("avoid_rollback_correlated_config_pattern")
                rationale.append("prior meta-config memory linked one or more strategy changes to rollback increases")
            if guidance.get("reused_successful_config_patterns"):
                selected.append("reuse_successful_prior_config_pattern")
                rationale.append("prior meta-config memory supports one or more proposed strategy changes")
            if guidance.get("lower_confidence_neutral_patterns"):
                rationale.append("prior neutral config changes lower confidence for repeated strategy mutations")
            mutated = SearchStrategyConfig.from_dict(adjusted)
        return SearchStrategyMutationPlan(
            base_config=base.to_dict(),
            mutated_config=mutated.to_dict(),
            selected_mutations=sorted(set(selected)),
            rationale=rationale,
            evidence_used={
                "persistent_residue_types": diagnosis.persistent_residue_types,
                "rollback_rate_by_method": diagnosis.rollback_rate_by_method,
                "rollback_risk_budget": rollback_risk_budget,
                "aggregate_rollback_rate": aggregate_rollback_rate,
                "rollback_budget_exceeded": rollback_budget_exceeded,
                "evaluator_overfit_risk": diagnosis.evaluator_overfit_risk,
                "probe_actionability_score": diagnosis.probe_actionability_score,
                "architecture_complexity_growth": diagnosis.architecture_complexity_growth,
                "transfer_regression_risk": diagnosis.transfer_regression_risk,
                "meta_config_memory_guidance": guidance,
            },
            used_heldout_labels=False,
        )


def _bounded_float(value: object, low: float, high: float, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    return float(max(low, min(high, value)))


def _bounded_float_map(value: object, default: Mapping[str, float], low: float, high: float) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return dict(default)
    out = dict(default)
    for key, raw in value.items():
        if isinstance(raw, (int, float)):
            out[str(key)] = float(max(low, min(high, raw)))
    return out


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(1, len(values))
