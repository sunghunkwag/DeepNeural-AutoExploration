"""Bounded neural architecture search components.

This package represents candidate neural architectures as explicit genomes and
evaluates them through validation-only sandbox runs.  It is intentionally small
and CPU-runnable; it does not generate or execute arbitrary source code.
"""

from .architecture_genome import ArchitectureGenome
from .architecture_evaluator import ArchitectureDecision, ArchitectureEvaluator
from .agi_relevance_metrics import compute_agi_relevance_metrics
from .gradient_probe import GradientProbeResult, probe_gradients
from .mutation_operators import MUTATION_METHODS, ArchitectureMutator, mutate_architecture
from .mutation_policy import MutationPolicy, MutationProposal
from .neural_candidate_sandbox import NeuralCandidateSandbox, NeuralSandboxResult, NeuralSandboxTask
from .representation_probe import RepresentationProbeResult, probe_representations
from .search_history import SearchHistory, SearchHistoryRecord
from .task_families import MultiDomainBatch, MultiDomainTaskFamily, build_multidomain_task_families
from .weight_inheritance import WeightInheritanceReport, inherit_shape_compatible_weights

__all__ = [
    "ArchitectureDecision",
    "ArchitectureEvaluator",
    "ArchitectureGenome",
    "ArchitectureMutator",
    "GradientProbeResult",
    "MUTATION_METHODS",
    "MultiDomainBatch",
    "MultiDomainTaskFamily",
    "MutationPolicy",
    "MutationProposal",
    "NeuralCandidateSandbox",
    "NeuralSandboxResult",
    "NeuralSandboxTask",
    "RepresentationProbeResult",
    "SearchHistory",
    "SearchHistoryRecord",
    "WeightInheritanceReport",
    "build_multidomain_task_families",
    "compute_agi_relevance_metrics",
    "inherit_shape_compatible_weights",
    "mutate_architecture",
    "probe_gradients",
    "probe_representations",
]
