"""Deep search controller over the bounded mutation policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from failure_residue import FailureResidue

from .architecture_genome import ArchitectureGenome, stable_hash
from .search_credit_assignment import SearchCreditAssigner, SearchCreditReport
from .search_depth_scheduler import SearchDepthAssessment, SearchDepthScheduler
from .search_strategy import SearchStrategy


@dataclass(frozen=True)
class DeepSearchDecision:
    decision_id: str
    parent_genome_id: str
    selected_depth_level: int
    selected_mutation_methods: List[str]
    rationale: str
    evidence_used: Dict[str, object]
    expected_failure_modes_addressed: List[str]
    expected_task_families_helped: List[str]
    exploration_vs_exploitation_score: float
    complexity_penalty: float
    rollback_risk: float
    used_heldout_labels: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "parent_genome_id": self.parent_genome_id,
            "selected_depth_level": self.selected_depth_level,
            "selected_mutation_methods": list(self.selected_mutation_methods),
            "rationale": self.rationale,
            "evidence_used": dict(self.evidence_used),
            "expected_failure_modes_addressed": list(self.expected_failure_modes_addressed),
            "expected_task_families_helped": list(self.expected_task_families_helped),
            "exploration_vs_exploitation_score": self.exploration_vs_exploitation_score,
            "complexity_penalty": self.complexity_penalty,
            "rollback_risk": self.rollback_risk,
            "used_heldout_labels": self.used_heldout_labels,
        }


@dataclass(frozen=True)
class DeepSearchControllerOutput:
    decision: DeepSearchDecision
    depth_assessment: SearchDepthAssessment
    credit_report: SearchCreditReport

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision.to_dict(),
            "depth_assessment": self.depth_assessment.to_dict(),
            "credit_report": self.credit_report.to_dict(),
        }


class DeepSearchController:
    """Choose mutation depth and methods from residues, probes, and history."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.scheduler = SearchDepthScheduler()
        self.credit_assigner = SearchCreditAssigner()
        self.strategy = SearchStrategy()

    def decide(
        self,
        parent: ArchitectureGenome,
        *,
        residues: Sequence[FailureResidue] = (),
        history: object | None = None,
        representation_probe: object | None = None,
        gradient_probe: object | None = None,
        search_space_registry: object | None = None,
        generation: int = 0,
    ) -> DeepSearchControllerOutput:
        confidence = _registry_confidence(search_space_registry)
        credit = self.credit_assigner.assign(
            history=history,
            representation_probe=representation_probe,
            gradient_probe=gradient_probe,
        )
        assessment = self.scheduler.schedule(
            parent,
            residues=residues,
            history=history,
            representation_probe=representation_probe,
            gradient_probe=gradient_probe,
            search_space_confidence=confidence,
        )
        expansions = list(getattr(search_space_registry, "module_families", {}).values()) if search_space_registry is not None else []
        plan = self.strategy.plan(
            depth_level=assessment.selected_depth_level,
            residues=residues,
            representation_probe=representation_probe,
            gradient_probe=gradient_probe,
            credit_report=credit,
            expansions=expansions,
        )
        decision_payload = {
            "parent": parent.genome_id,
            "generation": int(generation),
            "seed": self.seed,
            "depth": assessment.selected_depth_level,
            "methods": plan.methods,
            "evidence": assessment.evidence_used,
        }
        decision = DeepSearchDecision(
            decision_id=stable_hash(decision_payload, prefix="deep_search_"),
            parent_genome_id=parent.genome_id,
            selected_depth_level=assessment.selected_depth_level,
            selected_mutation_methods=plan.methods,
            rationale=f"{assessment.rationale}; {plan.rationale}",
            evidence_used={
                **assessment.evidence_used,
                "method_credit": dict(credit.method_credit),
                "rollback_risk_by_method": dict(credit.rollback_risk_by_method),
            },
            expected_failure_modes_addressed=plan.failure_modes_addressed,
            expected_task_families_helped=plan.task_families_helped,
            exploration_vs_exploitation_score=assessment.exploration_vs_exploitation_score,
            complexity_penalty=assessment.complexity_penalty,
            rollback_risk=assessment.rollback_risk,
            used_heldout_labels=False,
        )
        return DeepSearchControllerOutput(decision=decision, depth_assessment=assessment, credit_report=credit)


def _registry_confidence(registry: object | None) -> Dict[str, float]:
    families = getattr(registry, "module_families", {}) if registry is not None else {}
    if not isinstance(families, Mapping):
        return {}
    confidence: Dict[str, float] = {}
    for name, entry in families.items():
        if isinstance(entry, Mapping):
            confidence[str(name)] = float(entry.get("confidence_score", 0.0))
    return confidence

