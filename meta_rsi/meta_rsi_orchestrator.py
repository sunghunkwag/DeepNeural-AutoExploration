"""Higher-level bounded improvement loop over residues and neural search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from failure_residue import MissingOperatorInferencer, ResidueExtractor
from neural_search import ArchitectureEvaluator, ArchitectureGenome, ArchitectureMutator

from .evaluator_repair import EvaluatorRepair
from .experiment_planner import ExperimentPlanner
from .search_space_expander import SearchSpaceRegistry


class MetaRSIOrchestrator:
    """Connect failure residues, neural search, search-space updates, and audit logs."""

    def __init__(
        self,
        *,
        seed: int = 0,
        base_genome: Optional[ArchitectureGenome] = None,
        registry: Optional[SearchSpaceRegistry] = None,
    ):
        self.seed = int(seed)
        self.base_genome = base_genome or ArchitectureGenome(seed=seed)
        self.registry = registry or SearchSpaceRegistry()
        self.extractor = ResidueExtractor()
        self.inferencer = MissingOperatorInferencer(min_support=1)
        self.mutator = ArchitectureMutator(seed=seed)
        self.evaluator = ArchitectureEvaluator(seed=seed)
        self.planner = ExperimentPlanner()
        self.evaluator_repair = EvaluatorRepair()

    def run(
        self,
        failure_evidence: Sequence[Mapping[str, object]],
        *,
        audit_path: str | Path | None = None,
    ) -> Dict[str, object]:
        residues = self.extractor.extract_many(failure_evidence)
        hypotheses = self.inferencer.infer(residues)
        registry_updates = self.registry.update_from_hypotheses(hypotheses)
        repair_proposals = self.evaluator_repair.propose(residues)

        current = self.base_genome
        candidates = self._generate_candidates(current, hypotheses)
        decisions = []
        for candidate in candidates:
            decision = self.evaluator.evaluate_candidate(current, candidate)
            decisions.append(decision)
            if decision.accepted:
                current = candidate

        if any(decision.accepted and decision.used_heldout_labels for decision in decisions):
            raise RuntimeError("accepted architecture used held-out labels")

        plan = self.planner.plan(residues, hypotheses, decisions, seed=self.seed)
        audit = {
            "seed": self.seed,
            "base_genome": self.base_genome.to_dict(),
            "final_genome": current.to_dict(),
            "residues": [residue.to_dict() for residue in residues],
            "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
            "generated_candidates": [candidate.to_dict() for candidate in candidates],
            "architecture_decisions": [decision.to_dict() for decision in decisions],
            "search_space_registry": self.registry.to_dict(),
            "registry_updates": registry_updates,
            "evaluator_repair_proposals": [proposal.to_dict() for proposal in repair_proposals],
            "experiment_plan": plan.to_dict(),
            "anti_cheat": {
                "uses_heldout_labels_for_selection": False,
                "accepted_candidates_use_heldout_labels": False,
                "candidate_generation_is_bounded": True,
            },
        }
        if audit_path is not None:
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        return audit

    def _generate_candidates(self, parent: ArchitectureGenome, hypotheses) -> list[ArchitectureGenome]:
        if not hypotheses:
            return [self.mutator.mutate(parent, "add_residual_block", seed=self.seed + 1)]
        candidates: list[ArchitectureGenome] = []
        seen_methods = set()
        for index, hypothesis in enumerate(hypotheses):
            method = _mutation_for_module_family(hypothesis.proposed_module_family, hypothesis.failure_type)
            if method in seen_methods:
                continue
            seen_methods.add(method)
            candidates.append(self.mutator.mutate(parent, method, seed=self.seed + 1 + index))
        return candidates


def _mutation_for_module_family(module_family: str, failure_type: str) -> str:
    lowered = f"{module_family} {failure_type}".lower()
    if "memory" in lowered or "object" in lowered or "binding" in lowered:
        return "object_binding_adapter" if "object" in lowered or "binding" in lowered else "add_memory_gate"
    if "causal" in lowered or "bottleneck" in lowered:
        return "add_causal_bottleneck"
    if "sequence" in lowered or "state" in lowered:
        return "sequence_state_adapter"
    if "width" in lowered or "architecture" in lowered or "neural" in lowered:
        return "widen_hidden_dim"
    if "evaluator" in lowered:
        return "add_causal_bottleneck"
    return "add_residual_block"
