"""Convert failure residues into bounded experiment plans."""

from __future__ import annotations

from typing import Dict, Sequence

from .missing_operator_inferencer import MissingRepresentationHypothesis
from .residue_schema import FailureResidue, stable_residue_id


def residues_to_experiment_plan(
    residues: Sequence[FailureResidue],
    hypotheses: Sequence[MissingRepresentationHypothesis],
    *,
    seed: int = 0,
) -> Dict[str, object]:
    if any(residue.used_heldout_labels or residue.split_used == "test" for residue in residues):
        raise ValueError("experiment plans cannot use held-out label residues for candidate generation")
    payload = {
        "residue_ids": [residue.residue_id for residue in residues],
        "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
        "seed": int(seed),
    }
    return {
        "plan_id": stable_residue_id(payload).replace("residue_", "plan_"),
        "seed": int(seed),
        "selection_split": "validation",
        "uses_heldout_labels": False,
        "candidate_module_families": [hypothesis.proposed_module_family for hypothesis in hypotheses],
        "target_failure_types": sorted({residue.failure_type for residue in residues}),
        "residue_ids": [residue.residue_id for residue in residues],
        "steps": [
            "generate bounded neural architecture candidates",
            "train on procedural train split",
            "select only on validation metrics",
            "record hidden-validation metrics for audit",
            "freeze decisions before any held-out reporting",
        ],
    }
