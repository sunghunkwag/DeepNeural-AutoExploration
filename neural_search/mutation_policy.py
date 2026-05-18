"""Deterministic mutation policies for bounded neural architecture search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from failure_residue import FailureResidue

from .architecture_genome import ArchitectureGenome
from .mutation_operators import MUTATION_METHODS, mutate_architecture
from .search_history import SearchHistory


FAILURE_TO_MUTATION = {
    "object-binding failure": "object_binding_adapter",
    "representation collapse": "widen_hidden_dim",
    "architecture-representation bottleneck": "wider_residual_stack",
    "gradient bottleneck": "add_residual_block",
    "causal intervention failure": "add_causal_bottleneck",
    "sequence instability": "sequence_state_adapter",
    "dead module": "add_memory_gate",
    "over-compressed bottleneck": "repair_bottleneck",
    "task-family entanglement": "object_binding_adapter",
    "evaluator-selection failure": "add_causal_bottleneck",
    "primitive-library blind spot": "add_residual_block",
}


@dataclass(frozen=True)
class MutationProposal:
    candidate: ArchitectureGenome
    method: str
    policy: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "method": self.method,
            "policy": self.policy,
            "reason": self.reason,
        }


class MutationPolicy:
    def __init__(self, policy: str = "residue_guided", *, seed: int = 0):
        if policy not in {"random", "greedy", "residue_guided", "history_guided"}:
            raise ValueError(f"unknown mutation policy: {policy}")
        self.policy = policy
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def choose_methods(
        self,
        *,
        residues: Sequence[FailureResidue] = (),
        history: SearchHistory | None = None,
        count: int = 3,
    ) -> List[str]:
        if self.policy == "random":
            methods = list(MUTATION_METHODS)
            self.rng.shuffle(methods)
            return methods[:count]
        if self.policy == "greedy":
            return _dedupe(["widen_hidden_dim", "add_residual_block", "add_memory_gate", "add_causal_bottleneck"])[:count]
        if self.policy == "history_guided":
            best = history.best_method() if history is not None else None
            preferred = [best] if best else []
            preferred.extend(["wider_residual_stack", "widen_hidden_dim", "add_residual_block", "add_memory_gate"])
            return _dedupe([method for method in preferred if method])[:count]
        methods = [FAILURE_TO_MUTATION.get(residue.failure_type, "add_residual_block") for residue in residues]
        if not methods:
            methods = ["add_residual_block", "widen_hidden_dim", "add_memory_gate"]
        methods.extend(["widen_hidden_dim", "add_residual_block", "add_causal_bottleneck", "sequence_state_adapter"])
        return _dedupe(methods)[:count]

    def propose(
        self,
        parent: ArchitectureGenome,
        *,
        residues: Sequence[FailureResidue] = (),
        history: SearchHistory | None = None,
        count: int = 3,
    ) -> List[MutationProposal]:
        proposals: List[MutationProposal] = []
        for index, method in enumerate(self.choose_methods(residues=residues, history=history, count=count)):
            candidate = mutate_architecture(parent, method, self.seed + index + 1)
            reason = self._reason(method, residues, history)
            proposals.append(MutationProposal(candidate, method, self.policy, reason))
        return proposals

    def _reason(
        self,
        method: str,
        residues: Sequence[FailureResidue],
        history: SearchHistory | None,
    ) -> str:
        if self.policy == "residue_guided":
            matched = [residue.failure_type for residue in residues if FAILURE_TO_MUTATION.get(residue.failure_type) == method]
            return "residue-guided:" + ",".join(sorted(set(matched))) if matched else "residue-guided:fallback"
        if self.policy == "history_guided":
            best = history.best_method() if history is not None else None
            return f"history-guided:best={best or 'none'}"
        return f"{self.policy}:bounded architecture mutation"


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value in MUTATION_METHODS and value not in out:
            out.append(value)
    return out
