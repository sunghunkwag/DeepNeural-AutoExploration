"""Search-space registry updates justified by failure residues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from failure_residue import MissingRepresentationHypothesis


@dataclass
class SearchSpaceRegistry:
    module_families: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def update_from_hypotheses(
        self,
        hypotheses: Sequence[MissingRepresentationHypothesis],
    ) -> Dict[str, Dict[str, object]]:
        updates: Dict[str, Dict[str, object]] = {}
        for hypothesis in hypotheses:
            name = hypothesis.proposed_module_family
            entry = self.module_families.get(
                name,
                {
                    "module_family": name,
                    "justification_count": 0,
                    "failure_types": [],
                    "task_ids": [],
                },
            )
            entry["justification_count"] = int(entry.get("justification_count", 0)) + hypothesis.support_count
            entry["failure_types"] = sorted(set(list(entry.get("failure_types", [])) + [hypothesis.failure_type]))
            entry["task_ids"] = sorted(set(list(entry.get("task_ids", [])) + list(hypothesis.task_ids)))
            self.module_families[name] = entry
            updates[name] = dict(entry)
        return updates

    def to_dict(self) -> Dict[str, object]:
        return {"module_families": dict(self.module_families)}
