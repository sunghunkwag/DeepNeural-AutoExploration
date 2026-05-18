"""Data schema for structured failure residues."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict


def stable_residue_id(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "residue_" + hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class FailureResidue:
    task_id: str
    benchmark_name: str
    seed: int
    failure_type: str
    failed_operator_or_architecture: str
    suspected_bottleneck: str
    evidence_metrics: Dict[str, float] = field(default_factory=dict)
    missing_representation_hypothesis: str = ""
    proposed_operator_or_module_family: str = ""
    split_used: str = "validation"
    used_heldout_labels: bool = False
    residue_id: str = ""

    def __post_init__(self) -> None:
        if self.used_heldout_labels:
            raise ValueError("failure residues cannot be created from held-out labels used for candidate generation")
        if self.split_used == "test":
            raise ValueError("test split cannot be used to create candidate-generation residues")
        if not self.residue_id:
            object.__setattr__(self, "residue_id", stable_residue_id(self._identity_payload()))

    def _identity_payload(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("residue_id", None)
        return data

    def to_dict(self) -> Dict[str, object]:
        return {**self._identity_payload(), "residue_id": self.residue_id}
