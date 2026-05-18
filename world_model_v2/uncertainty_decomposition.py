"""Separate model, observation, and intervention uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, Sequence


def _variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return float(mean([(value - avg) ** 2 for value in values]))


@dataclass(frozen=True)
class UncertaintyComponents:
    model_uncertainty: float
    observation_uncertainty: float
    intervention_uncertainty: float

    @property
    def total(self) -> float:
        return self.model_uncertainty + self.observation_uncertainty + self.intervention_uncertainty

    def to_dict(self) -> Dict[str, float]:
        return {**asdict(self), "total": self.total}


class UncertaintyDecomposition:
    def decompose(
        self,
        *,
        model_scores: Sequence[float] = (),
        observation_errors: Sequence[float] = (),
        intervention_effects: Sequence[float] = (),
    ) -> UncertaintyComponents:
        return UncertaintyComponents(
            model_uncertainty=_variance([float(x) for x in model_scores]),
            observation_uncertainty=_variance([float(x) for x in observation_errors]),
            intervention_uncertainty=_variance([float(x) for x in intervention_effects]),
        )
