"""Adapter from long-horizon neural evidence to evaluator evolution signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

from evaluator_evolution import EvaluatorGenome


@dataclass
class EvaluatorEvolutionEvidenceAdapter:
    mismatch_patterns: Dict[str, int] = field(default_factory=dict)
    evaluator_repair_proposals: list[Dict[str, object]] = field(default_factory=list)
    evaluator_repair_acceptance_status: list[Dict[str, object]] = field(default_factory=list)
    proposed_weight_adjustments: Dict[str, float] = field(default_factory=dict)

    def observe_run(
        self,
        run_result: Mapping[str, object],
        quality_report: Mapping[str, object] | None = None,
    ) -> None:
        diagnosis = run_result.get("search_process_diagnosis", {})
        if isinstance(diagnosis, Mapping):
            if _mean_mapping(diagnosis.get("validation_hidden_mismatch_by_method", {})) > 0.05:
                self._inc("validation_hidden_mismatch")
                self.proposed_weight_adjustments["hidden_validation_loss_weight"] = self.proposed_weight_adjustments.get("hidden_validation_loss_weight", 0.0) + 0.04
            if _float(diagnosis.get("transfer_regression_risk")) > 0.05:
                self._inc("transfer_regression")
                self.proposed_weight_adjustments["transfer_score_weight"] = self.proposed_weight_adjustments.get("transfer_score_weight", 0.0) + 0.03
            if _float(diagnosis.get("architecture_complexity_growth")) > 0.20:
                self._inc("complexity_overgrowth")
                self.proposed_weight_adjustments["architecture_complexity_penalty_weight"] = self.proposed_weight_adjustments.get("architecture_complexity_penalty_weight", 0.0) + 0.03
        decisions = run_result.get("architecture_decisions", [])
        if isinstance(decisions, list):
            for decision in decisions:
                if not isinstance(decision, Mapping):
                    continue
                if _float(decision.get("hidden_validation_improvement")) < 0.0:
                    self._inc("hidden_validation_regression")
                regressions = decision.get("task_family_regressions", {})
                if isinstance(regressions, Mapping) and regressions:
                    self._inc("task_family_regression")
                if bool(decision.get("accepted")) and quality_report and quality_report.get("recommendation") == "revert_previous_config":
                    self._inc("accepted_candidate_later_judged_harmful")
        plan = run_result.get("evaluator_objective_mutation_plan", {})
        if isinstance(plan, Mapping):
            evidence = plan.get("evidence_used", {})
            if isinstance(evidence, Mapping):
                calibration = evidence.get("evaluator_calibration_report", {})
                if isinstance(calibration, Mapping):
                    if bool(calibration.get("validation_hidden_mismatch_guard_triggered")):
                        self._inc("validation_hidden_mismatch_guard_triggered")
                    if _float(calibration.get("high_rollback_mismatch_flag")) > 0.0:
                        self._inc("rollback_coupled_mismatch")
                repair_actions = evidence.get("evaluator_overfit_repair_actions", [])
                if isinstance(repair_actions, list):
                    for action in repair_actions:
                        self.proposed_weight_adjustments[str(action)] = self.proposed_weight_adjustments.get(str(action), 0.0) + 1.0
            self.evaluator_repair_proposals.append(
                {
                    "selected_mutations": list(plan.get("selected_mutations", [])) if isinstance(plan.get("selected_mutations", []), list) else [],
                    "mutated_config": dict(plan.get("mutated_config", {})) if isinstance(plan.get("mutated_config", {}), Mapping) else {},
                }
            )
        external = run_result.get("external_generalization_diagnosis", {})
        if isinstance(external, Mapping) and bool(external.get("heldout_after_freeze_regression_warning")):
            self._inc("external_heldout_after_freeze_regression")
            self.proposed_weight_adjustments["transfer_score_weight"] = self.proposed_weight_adjustments.get("transfer_score_weight", 0.0) + 0.05
            self.proposed_weight_adjustments["generalization_gap_penalty_weight"] = self.proposed_weight_adjustments.get("generalization_gap_penalty_weight", 0.0) + 0.05
        self._run_evaluator_evolution_probe(run_result)

    def summary(self) -> Dict[str, object]:
        return {
            "evaluator_repair_proposals": list(self.evaluator_repair_proposals),
            "evaluator_repair_acceptance_status": list(self.evaluator_repair_acceptance_status),
            "mismatch_patterns": dict(self.mismatch_patterns),
            "proposed_weight_adjustments": {key: float(value) for key, value in self.proposed_weight_adjustments.items()},
            "known_limitations": [
                "adapter proposes evaluator repairs from process evidence only",
                "probationary evaluator checks are lightweight and do not replace full benchmark validation",
            ],
        }

    def _run_evaluator_evolution_probe(self, run_result: Mapping[str, object]) -> None:
        compatible = _compatible_decisions(run_result)
        if not compatible:
            return
        genome = EvaluatorGenome()
        for candidate in genome.propose(generation=len(self.evaluator_repair_acceptance_status), failure_rule_count=sum(self.mismatch_patterns.values()))[:2]:
            decision = genome.validate_candidate(
                candidate,
                compatible,
                generation=candidate.generation_created,
                failure_rule_count=sum(self.mismatch_patterns.values()),
            )
            self.evaluator_repair_acceptance_status.append(decision)

    def _inc(self, key: str) -> None:
        self.mismatch_patterns[key] = self.mismatch_patterns.get(key, 0) + 1


def _compatible_decisions(run_result: Mapping[str, object]) -> list[Dict[str, object]]:
    raw = run_result.get("architecture_decisions", [])
    if not isinstance(raw, list):
        return []
    out: list[Dict[str, object]] = []
    for decision in raw:
        if not isinstance(decision, Mapping):
            continue
        out.append(
            {
                "accepted": bool(decision.get("accepted")),
                "rejection_reason": str(decision.get("rejection_reason", "")),
                "mean_improvement": _float(decision.get("validation_improvement")),
                "hidden_guard_regression": _float(decision.get("hidden_validation_improvement")),
                "candidate_scores": [_float((decision.get("validation_metrics", {}) or {}).get("loss"))] if isinstance(decision.get("validation_metrics", {}), Mapping) else [0.0],
                "candidate_program": {"primitive_sequence": [{"name": str(decision.get("mutation_method", "architecture_mutation"))}]},
                "no_op_control_delta": 0.0,
                "random_control_delta": 0.0,
                "deterministic_replay_passed": True,
            }
        )
    return out


def _mean_mapping(value: object) -> float:
    if not isinstance(value, Mapping) or not value:
        return 0.0
    vals = [float(raw) for raw in value.values() if isinstance(raw, (int, float))]
    return sum(vals) / max(1, len(vals))


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
