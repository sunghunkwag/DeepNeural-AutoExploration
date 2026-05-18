"""Object-centric state representation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    latent: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectRelation:
    source_id: str
    target_id: str
    relation_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectState:
    objects: Dict[str, ObjectRecord] = field(default_factory=dict)
    relations: List[ObjectRelation] = field(default_factory=list)
    time_index: int = 0

    @classmethod
    def from_objects(
        cls,
        objects: List[ObjectRecord],
        relations: Optional[List[ObjectRelation]] = None,
        *,
        time_index: int = 0,
    ) -> "ObjectState":
        return cls({obj.object_id: obj for obj in objects}, list(relations or []), time_index=time_index)

    def add_object(self, object_id: str, attributes: Dict[str, Any], *, latent: bool = False) -> "ObjectState":
        updated = dict(self.objects)
        updated[object_id] = ObjectRecord(object_id, copy.deepcopy(attributes), latent)
        return ObjectState(updated, list(self.relations), self.time_index)

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "ObjectState":
        if source_id not in self.objects or target_id not in self.objects:
            raise KeyError("relations require existing source and target objects")
        relation = ObjectRelation(source_id, target_id, relation_type, copy.deepcopy(attributes or {}))
        return ObjectState(dict(self.objects), list(self.relations) + [relation], self.time_index)

    def get_attribute(self, object_id: str, attribute: str, default: Any = None) -> Any:
        return self.objects[object_id].attributes.get(attribute, default)

    def set_attribute(self, object_id: str, attribute: str, value: Any) -> "ObjectState":
        if object_id not in self.objects:
            raise KeyError(f"unknown object: {object_id}")
        updated = dict(self.objects)
        record = updated[object_id]
        attrs = copy.deepcopy(record.attributes)
        attrs[attribute] = value
        updated[object_id] = ObjectRecord(record.object_id, attrs, record.latent)
        return ObjectState(updated, list(self.relations), self.time_index)

    def with_time(self, time_index: int) -> "ObjectState":
        return ObjectState(dict(self.objects), list(self.relations), int(time_index))

    def to_dict(self) -> Dict[str, object]:
        return {
            "time_index": self.time_index,
            "objects": {key: value.to_dict() for key, value in self.objects.items()},
            "relations": [relation.to_dict() for relation in self.relations],
        }
