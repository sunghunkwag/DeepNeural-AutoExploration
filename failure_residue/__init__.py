"""Structured failure residue extraction for bounded improvement loops."""

from .missing_operator_inferencer import MissingOperatorInferencer, MissingRepresentationHypothesis
from .residue_extractor import ARC_FAILURE_TAXONOMY, ResidueExtractor, classify_failure_type
from .residue_schema import FailureResidue
from .residue_to_experiment import residues_to_experiment_plan

__all__ = [
    "ARC_FAILURE_TAXONOMY",
    "FailureResidue",
    "MissingOperatorInferencer",
    "MissingRepresentationHypothesis",
    "ResidueExtractor",
    "classify_failure_type",
    "residues_to_experiment_plan",
]
