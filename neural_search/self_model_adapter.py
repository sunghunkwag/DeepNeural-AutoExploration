"""Long-horizon self-model evidence adapter."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Mapping, Sequence

from self_model import architecture_decision_to_effects


@dataclass(frozen=True)
class SelfModelPredictionRecord:
    run_index: int
    mutation_method: str
    failure_residue_type: str
    task_family: str
    validation_delta: float
    hidden_validation_delta: float
    rollback: bool
    predicted_helpful: bool
    actual_helpful: bool
    attribution_label: str
    meta_decision_quality: float
    self_model_effects: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_index": self.run_index,
            "mutation_method": self.mutation_method,
            "failure_residue_type": self.failure_residue_type,
            "task_family": self.task_family,
            "validation_delta": self.validation_delta,
            "hidden_validation_delta": self.hidden_validation_delta,
            "rollback": self.rollback,
            "predicted_helpful": self.predicted_helpful,
            "actual_helpful": self.actual_helpful,
            "attribution_label": self.attribution_label,
            "meta_decision_quality": self.meta_decision_quality,
            "self_model_effects": dict(self.self_model_effects),
        }


class LongHorizonSelfModelAdapter:
    """Records honest prediction/actual pairs from architecture decisions."""

    def __init__(self) -> None:
        self.records: list[SelfModelPredictionRecord] = []
        self._method_outcomes: Dict[str, list[bool]] = {}

    def observe_run(
        self,
        run_index: int,
        run_result: Mapping[str, object],
        attribution_report: Mapping[str, object] | None = None,
        quality_report: Mapping[str, object] | None = None,
    ) -> None:
        residues = _residues(run_result)
        failure_type = _dominant(residues, "failure_type")
        task_family = _task_family(residues)
        attribution_label = _attribution_label(attribution_report)
        quality = _quality_score(quality_report)
        decisions = run_result.get("architecture_decisions", [])
        if not isinstance(decisions, list):
            return
        for decision in decisions:
            if not isinstance(decision, Mapping):
                continue
            if bool(decision.get("used_heldout_labels", False)):
                raise ValueError("self-model adapter refuses heldout-labeled architecture decisions")
            method = str(decision.get("mutation_method", "unknown"))
            validation_delta = _float(decision.get("validation_improvement"))
            hidden_delta = _float(decision.get("hidden_validation_improvement"))
            rollback = bool(decision.get("rollback", not bool(decision.get("accepted", False))))
            predicted = self._predict_helpful(method, validation_delta, hidden_delta, rollback)
            actual = bool(not rollback and hidden_delta >= 0.0 and attribution_label != "harmed")
            effects = architecture_decision_to_effects(dict(decision))
            record = SelfModelPredictionRecord(
                run_index=int(run_index),
                mutation_method=method,
                failure_residue_type=failure_type,
                task_family=task_family,
                validation_delta=validation_delta,
                hidden_validation_delta=hidden_delta,
                rollback=rollback,
                predicted_helpful=predicted,
                actual_helpful=actual,
                attribution_label=attribution_label,
                meta_decision_quality=quality,
                self_model_effects=effects,
            )
            self.records.append(record)
            self._method_outcomes.setdefault(method, []).append(actual)

    def summary(self) -> Dict[str, object]:
        total = len(self.records)
        correct = sum(1 for record in self.records if record.predicted_helpful == record.actual_helpful)
        helpful_scores = {
            method: sum(1 for value in values if value) / max(1, len(values))
            for method, values in self._method_outcomes.items()
        }
        strongest_helpful = sorted(helpful_scores, key=lambda method: helpful_scores[method], reverse=True)[:5]
        strongest_harmful = sorted(helpful_scores, key=lambda method: helpful_scores[method])[:5]
        return {
            "prediction_count": total,
            "correct_prediction_count": correct,
            "self_model_prediction_accuracy": float(correct / max(1, total)),
            "strongest_predicted_helpful_mutations": strongest_helpful,
            "strongest_predicted_harmful_mutations": strongest_harmful,
            "prediction_records": [record.to_dict() for record in self.records],
            "known_limitations": [
                "heuristic online classifier, not a trained causal self-model",
                "uses validation and hidden-validation deltas, never heldout labels for selection",
            ],
        }

    def _predict_helpful(self, method: str, validation_delta: float, hidden_delta: float, rollback: bool) -> bool:
        history = self._method_outcomes.get(method, [])
        if history:
            return sum(1 for value in history if value) / len(history) >= 0.5
        return bool(not rollback and hidden_delta >= 0.0 and validation_delta >= -0.01)


def _residues(run_result: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = run_result.get("failure_residues", [])
    return [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []


def _dominant(items: Sequence[Mapping[str, object]], key: str) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key, ""))
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "none"
    return max(sorted(counts), key=lambda name: counts[name])


def _task_family(residues: Sequence[Mapping[str, object]]) -> str:
    families = ("function_regression", "sequence_prediction", "object_state_transition", "causal_intervention", "memory_retrieval")
    counts = {family: 0 for family in families}
    for residue in residues:
        task_id = str(residue.get("task_id", ""))
        for family in families:
            if family in task_id:
                counts[family] += 1
    best = max(sorted(counts), key=lambda name: counts[name])
    return best if counts[best] > 0 else "unknown"


def _attribution_label(report: Mapping[str, object] | None) -> str:
    if not report:
        return "neutral"
    if _list(report.get("harmed_changes")):
        return "harmed"
    if _list(report.get("helped_changes")):
        return "helped"
    return "neutral"


def _quality_score(report: Mapping[str, object] | None) -> float:
    if not report:
        return 0.0
    scores = report.get("quality_scores", {})
    if isinstance(scores, Mapping):
        return _float(scores.get("overall_meta_decision_quality_score"))
    return 0.0


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
