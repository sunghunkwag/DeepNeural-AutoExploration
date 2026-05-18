"""Search-space registry updates justified by failure residues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from failure_residue import FailureResidue, MissingRepresentationHypothesis


SUPPORTED_BOUNDED_MODULE_FAMILIES = (
    "gated_memory_block",
    "causal_bottleneck_block",
    "wider_residual_stack",
    "object_binding_adapter",
    "sequence_state_adapter",
)


@dataclass
class SearchSpaceRegistry:
    module_families: Dict[str, Dict[str, object]] = field(default_factory=dict)
    min_support: int = 2

    def update_from_hypotheses(
        self,
        hypotheses: Sequence[MissingRepresentationHypothesis],
        residues: Sequence[FailureResidue] = (),
    ) -> Dict[str, Dict[str, object]]:
        updates: Dict[str, Dict[str, object]] = {}
        residue_count_by_family: Dict[str, int] = {}
        hidden_failure_by_family: Dict[str, int] = {}
        for residue in residues:
            family = residue.proposed_operator_or_module_family
            residue_count_by_family[family] = residue_count_by_family.get(family, 0) + 1
            if residue.split_used == "hidden_validation" or residue.evidence_metrics.get("hidden_validation_regression", 0.0) < 0.0:
                hidden_failure_by_family[family] = hidden_failure_by_family.get(family, 0) + 1
        for hypothesis in hypotheses:
            name = hypothesis.proposed_module_family
            if name not in SUPPORTED_BOUNDED_MODULE_FAMILIES:
                continue
            support_count = max(hypothesis.support_count, residue_count_by_family.get(name, 0))
            hidden_count = hidden_failure_by_family.get(name, 0)
            if support_count < self.min_support and hidden_count < self.min_support:
                continue
            entry = self.module_families.get(
                name,
                {
                    "module_family": name,
                    "justification_count": 0,
                    "hidden_validation_failure_count": 0,
                    "failure_types": [],
                    "task_ids": [],
                    "audit": [],
                },
            )
            entry["justification_count"] = int(entry.get("justification_count", 0)) + support_count
            entry["hidden_validation_failure_count"] = int(entry.get("hidden_validation_failure_count", 0)) + hidden_count
            entry["failure_types"] = sorted(set(list(entry.get("failure_types", [])) + [hypothesis.failure_type]))
            entry["task_ids"] = sorted(set(list(entry.get("task_ids", [])) + list(hypothesis.task_ids)))
            entry["audit"] = list(entry.get("audit", [])) + [
                {
                    "reason": hypothesis.rationale,
                    "support_count": support_count,
                    "hidden_validation_failure_count": hidden_count,
                }
            ]
            self.module_families[name] = entry
            updates[name] = dict(entry)
        return updates

    def to_dict(self) -> Dict[str, object]:
        return {"module_families": dict(self.module_families)}
