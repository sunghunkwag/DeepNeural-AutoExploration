"""Shape-compatible neural architecture weight inheritance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WeightInheritanceReport:
    copied: List[str] = field(default_factory=list)
    skipped_shape_mismatch: Dict[str, str] = field(default_factory=dict)
    missing_in_parent: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "copied": list(self.copied),
            "copied_count": len(self.copied),
            "skipped_shape_mismatch": dict(self.skipped_shape_mismatch),
            "missing_in_parent": list(self.missing_in_parent),
        }


def inherit_shape_compatible_weights(parent: nn.Module, child: nn.Module) -> WeightInheritanceReport:
    """Copy state entries whose names and tensor shapes match exactly."""

    parent_state = parent.state_dict()
    child_state = child.state_dict()
    updated = {}
    copied: List[str] = []
    skipped: Dict[str, str] = {}
    missing: List[str] = []
    for name, child_value in child_state.items():
        parent_value = parent_state.get(name)
        if parent_value is None:
            updated[name] = child_value
            missing.append(name)
            continue
        if tuple(parent_value.shape) == tuple(child_value.shape):
            updated[name] = parent_value.detach().clone().to(dtype=child_value.dtype, device=child_value.device)
            copied.append(name)
        else:
            updated[name] = child_value
            skipped[name] = f"parent={tuple(parent_value.shape)} child={tuple(child_value.shape)}"
    child.load_state_dict(updated, strict=True)
    return WeightInheritanceReport(copied=copied, skipped_shape_mismatch=skipped, missing_in_parent=missing)
