"""Simple do-style interventions over object attributes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

from .object_state import ObjectState


@dataclass(frozen=True)
class Intervention:
    object_id: str
    attribute: str
    value: object

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class InterventionModel:
    def apply(self, state: ObjectState, interventions: Iterable[Intervention]) -> ObjectState:
        updated = state
        for intervention in interventions:
            updated = updated.set_attribute(intervention.object_id, intervention.attribute, intervention.value)
        return updated
