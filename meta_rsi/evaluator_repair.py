"""Simple evaluator repair proposals from failure residues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from failure_residue import FailureResidue


@dataclass(frozen=True)
class EvaluatorRepairProposal:
    repair_id: str
    reason: str
    required_guard: str
    applies_to_failure_type: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "repair_id": self.repair_id,
            "reason": self.reason,
            "required_guard": self.required_guard,
            "applies_to_failure_type": self.applies_to_failure_type,
        }


class EvaluatorRepair:
    def propose(self, residues: Sequence[FailureResidue]) -> list[EvaluatorRepairProposal]:
        proposals: list[EvaluatorRepairProposal] = []
        if any(residue.failure_type == "evaluator-selection failure" for residue in residues):
            proposals.append(
                EvaluatorRepairProposal(
                    repair_id="repair_hidden_validation_guard",
                    reason="repeated selector failures require hidden-validation guard metrics before acceptance",
                    required_guard="hidden_validation_non_regression",
                    applies_to_failure_type="evaluator-selection failure",
                )
            )
        return proposals
