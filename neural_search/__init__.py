"""Bounded neural architecture search components.

This package represents candidate neural architectures as explicit genomes and
evaluates them through validation-only sandbox runs.  It is intentionally small
and CPU-runnable; it does not generate or execute arbitrary source code.
"""

from .architecture_genome import ArchitectureGenome
from .architecture_evaluator import ArchitectureDecision, ArchitectureEvaluator
from .mutation_operators import MUTATION_METHODS, ArchitectureMutator, mutate_architecture
from .neural_candidate_sandbox import NeuralCandidateSandbox, NeuralSandboxResult, NeuralSandboxTask
from .weight_inheritance import WeightInheritanceReport, inherit_shape_compatible_weights

__all__ = [
    "ArchitectureDecision",
    "ArchitectureEvaluator",
    "ArchitectureGenome",
    "ArchitectureMutator",
    "MUTATION_METHODS",
    "NeuralCandidateSandbox",
    "NeuralSandboxResult",
    "NeuralSandboxTask",
    "WeightInheritanceReport",
    "inherit_shape_compatible_weights",
    "mutate_architecture",
]
