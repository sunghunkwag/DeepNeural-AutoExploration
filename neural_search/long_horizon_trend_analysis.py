"""Trend analysis for repeated autonomous neural exploration runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Dict, Mapping, Sequence


RISK_TRENDS = {
    "rollback_rate_trend",
    "persistent_residue_count_trend",
    "evaluator_overfit_risk_trend",
    "transfer_regression_risk_trend",
    "architecture_complexity_growth_trend",
}


@dataclass(frozen=True)
class LongHorizonTrend:
    start_value: float
    end_value: float
    delta: float
    slope: float
    improved: bool
    confidence: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_value": self.start_value,
            "end_value": self.end_value,
            "delta": self.delta,
            "slope": self.slope,
            "improved": self.improved,
            "confidence": self.confidence,
        }


def compute_long_horizon_trends(run_summaries: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    specs = {
        "rollback_rate_trend": "rollback_rate",
        "residue_resolution_trend": "residue_resolution_rate",
        "persistent_residue_count_trend": "persistent_residue_count",
        "hidden_validation_robustness_trend": "hidden_validation_robustness",
        "evaluator_overfit_risk_trend": "evaluator_overfit_risk",
        "transfer_regression_risk_trend": "transfer_regression_risk",
        "architecture_complexity_growth_trend": "architecture_complexity_growth",
        "task_family_balance_trend": "task_family_balance",
        "meta_decision_quality_trend": "meta_decision_quality",
        "process_improvement_delta_trend": "process_improvement_delta",
    }
    return {
        trend_name: _trend([_float(summary.get(value_key)) for summary in run_summaries], lower_is_better=trend_name in RISK_TRENDS).to_dict()
        for trend_name, value_key in specs.items()
    }


def _trend(values: Sequence[float], *, lower_is_better: bool) -> LongHorizonTrend:
    if not values:
        return LongHorizonTrend(0.0, 0.0, 0.0, 0.0, False, 0.0)
    start = float(values[0])
    end = float(values[-1])
    delta = end - start
    slope = _slope(values)
    improved = delta < 0.0 if lower_is_better else delta > 0.0
    scale = max(1.0, abs(start), abs(end), mean(abs(value) for value in values))
    confidence = min(1.0, abs(delta) / scale * sqrt(len(values)) / 2.0)
    if len(values) < 2:
        confidence = 0.0
    return LongHorizonTrend(
        start_value=start,
        end_value=end,
        delta=float(delta),
        slope=float(slope),
        improved=bool(improved),
        confidence=float(confidence),
    )


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = [float(index) for index in range(len(values))]
    x_mean = mean(xs)
    y_mean = mean(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return 0.0
    return float(sum((x - x_mean) * (float(y) - y_mean) for x, y in zip(xs, values)) / denom)


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
