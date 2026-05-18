"""Generate next experiment plans from residues and architecture decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from failure_residue import FailureResidue, MissingRepresentationHypothesis, residues_to_experiment_plan
from neural_search import ArchitectureDecision


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    selection_split: str
    candidate_module_families: list[str]
    target_failure_types: list[str]
    accepted_candidate_ids: list[str] = field(default_factory=list)
    uses_heldout_labels: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "selection_split": self.selection_split,
            "candidate_module_families": list(self.candidate_module_families),
            "target_failure_types": list(self.target_failure_types),
            "accepted_candidate_ids": list(self.accepted_candidate_ids),
            "uses_heldout_labels": self.uses_heldout_labels,
        }


class ExperimentPlanner:
    def plan(
        self,
        residues: Sequence[FailureResidue],
        hypotheses: Sequence[MissingRepresentationHypothesis],
        decisions: Sequence[ArchitectureDecision],
        *,
        seed: int = 0,
    ) -> ExperimentPlan:
        base = residues_to_experiment_plan(residues, hypotheses, seed=seed)
        accepted_ids = [decision.candidate_id for decision in decisions if decision.accepted]
        return ExperimentPlan(
            plan_id=str(base["plan_id"]),
            selection_split=str(base["selection_split"]),
            candidate_module_families=list(base["candidate_module_families"]),
            target_failure_types=list(base["target_failure_types"]),
            accepted_candidate_ids=accepted_ids,
            uses_heldout_labels=False,
        )
