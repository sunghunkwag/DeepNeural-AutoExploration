"""Observed vs counterfactual object-state rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from .intervention_model import Intervention, InterventionModel
from .object_state import ObjectState


TransitionFn = Callable[[ObjectState], ObjectState]


def default_transition(state: ObjectState) -> ObjectState:
    """Advance x/y by vx/vy for every object when those attributes exist."""

    updated = state
    for object_id, record in state.objects.items():
        attrs = record.attributes
        if "x" in attrs and "vx" in attrs and isinstance(attrs["x"], (int, float)) and isinstance(attrs["vx"], (int, float)):
            updated = updated.set_attribute(object_id, "x", float(attrs["x"]) + float(attrs["vx"]))
        current = updated.objects[object_id].attributes
        if "y" in current and "vy" in current and isinstance(current["y"], (int, float)) and isinstance(current["vy"], (int, float)):
            updated = updated.set_attribute(object_id, "y", float(current["y"]) + float(current["vy"]))
    return updated.with_time(state.time_index + 1)


@dataclass(frozen=True)
class RolloutComparison:
    observed: List[ObjectState]
    counterfactual: List[ObjectState]
    attribute_deltas: Dict[str, float | str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "observed": [state.to_dict() for state in self.observed],
            "counterfactual": [state.to_dict() for state in self.counterfactual],
            "attribute_deltas": dict(self.attribute_deltas),
        }


class CounterfactualRollout:
    def __init__(
        self,
        transition_fn: Optional[TransitionFn] = None,
        intervention_model: Optional[InterventionModel] = None,
    ):
        self.transition_fn = transition_fn or default_transition
        self.intervention_model = intervention_model or InterventionModel()

    def rollout(self, state: ObjectState, *, horizon: int) -> List[ObjectState]:
        states = [state]
        current = state
        for _ in range(max(0, int(horizon))):
            current = self.transition_fn(current)
            states.append(current)
        return states

    def compare(
        self,
        observed_start: ObjectState,
        interventions: Iterable[Intervention],
        *,
        horizon: int = 1,
    ) -> RolloutComparison:
        observed = self.rollout(observed_start, horizon=horizon)
        intervened_start = self.intervention_model.apply(observed_start, interventions)
        counterfactual = self.rollout(intervened_start, horizon=horizon)
        deltas = _attribute_deltas(observed[-1], counterfactual[-1])
        return RolloutComparison(observed=observed, counterfactual=counterfactual, attribute_deltas=deltas)


def _attribute_deltas(left: ObjectState, right: ObjectState) -> Dict[str, float | str]:
    deltas: Dict[str, float | str] = {}
    for object_id, right_record in right.objects.items():
        left_record = left.objects.get(object_id)
        if left_record is None:
            continue
        for attr, right_value in right_record.attributes.items():
            left_value = left_record.attributes.get(attr)
            key = f"{object_id}.{attr}"
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                delta = float(right_value) - float(left_value)
                if delta != 0.0:
                    deltas[key] = delta
            elif left_value != right_value:
                deltas[key] = f"{left_value}->{right_value}"
    return deltas
