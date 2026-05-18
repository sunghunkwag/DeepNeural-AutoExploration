"""Rule-based inference from repeated residues to missing module families."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .residue_schema import FailureResidue


_FAILURE_TO_MODULE = {
    "object-binding failure": "object_binding_adapter",
    "architecture-representation bottleneck": "wider_residual_stack",
    "representation collapse": "wider_residual_stack",
    "gradient bottleneck": "wider_residual_stack",
    "causal intervention failure": "causal_bottleneck_block",
    "sequence instability": "sequence_state_adapter",
    "dead module": "gated_memory_block",
    "over-compressed bottleneck": "causal_bottleneck_block",
    "task-family entanglement": "object_binding_adapter",
}


@dataclass(frozen=True)
class MissingRepresentationHypothesis:
    failure_type: str
    proposed_module_family: str
    support_count: int
    task_ids: List[str]
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "failure_type": self.failure_type,
            "proposed_module_family": self.proposed_module_family,
            "support_count": self.support_count,
            "task_ids": list(self.task_ids),
            "rationale": self.rationale,
        }


class MissingOperatorInferencer:
    def __init__(self, min_support: int = 1):
        self.min_support = max(1, int(min_support))

    def infer(self, residues: Sequence[FailureResidue]) -> List[MissingRepresentationHypothesis]:
        grouped: Dict[tuple[str, str], List[FailureResidue]] = defaultdict(list)
        for residue in residues:
            module_family = _FAILURE_TO_MODULE.get(
                residue.failure_type,
                residue.proposed_operator_or_module_family,
            )
            grouped[(residue.failure_type, module_family)].append(residue)

        hypotheses: List[MissingRepresentationHypothesis] = []
        for (failure_type, module_family), group in grouped.items():
            if len(group) < self.min_support:
                continue
            task_ids = sorted({item.task_id for item in group})
            rationale = (
                f"{len(group)} residue(s) with failure type '{failure_type}' point to "
                f"the '{module_family}' family using validation-only evidence."
            )
            hypotheses.append(
                MissingRepresentationHypothesis(
                    failure_type=failure_type,
                    proposed_module_family=module_family,
                    support_count=len(group),
                    task_ids=task_ids,
                    rationale=rationale,
                )
            )
        return sorted(hypotheses, key=lambda item: (item.support_count, item.failure_type), reverse=True)
