"""Deterministic mutation policies for bounded neural architecture search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

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
        deep_search_decision: object | None = None,
        credit_report: object | None = None,
        count: int = 3,
    ) -> List[str]:
        deep_methods = list(getattr(deep_search_decision, "selected_mutation_methods", []) or [])
        if deep_methods:
            return _rank_by_credit(_dedupe(deep_methods), credit_report, history)[:count]
        if self.policy == "random":
            methods = list(MUTATION_METHODS)
            self.rng.shuffle(methods)
            return _rank_by_credit(methods, credit_report, history)[:count]
        if self.policy == "greedy":
            return _rank_by_credit(_dedupe(["widen_hidden_dim", "add_residual_block", "add_memory_gate", "add_causal_bottleneck"]), credit_report, history)[:count]
        if self.policy == "history_guided":
            best = history.best_method() if history is not None else None
            preferred = [best] if best else []
            preferred.extend(["wider_residual_stack", "widen_hidden_dim", "add_residual_block", "add_memory_gate"])
            return _rank_by_credit(_dedupe([method for method in preferred if method]), credit_report, history)[:count]
        methods = [FAILURE_TO_MUTATION.get(residue.failure_type, "add_residual_block") for residue in residues]
        if not methods:
            methods = ["add_residual_block", "widen_hidden_dim", "add_memory_gate"]
        methods.extend(["widen_hidden_dim", "add_residual_block", "add_causal_bottleneck", "sequence_state_adapter"])
        return _rank_by_credit(_dedupe(methods), credit_report, history)[:count]

    def propose(
        self,
        parent: ArchitectureGenome,
        *,
        residues: Sequence[FailureResidue] = (),
        history: SearchHistory | None = None,
        deep_search_decision: object | None = None,
        credit_report: object | None = None,
        count: int = 3,
    ) -> List[MutationProposal]:
        proposals: List[MutationProposal] = []
        seen_signatures = _seen_structural_signatures(history)
        local_signatures = {_structural_signature(parent)}
        candidate_methods = self.choose_methods(
            residues=residues,
            history=history,
            deep_search_decision=deep_search_decision,
            credit_report=credit_report,
            count=max(count, len(MUTATION_METHODS)),
        )
        if len(candidate_methods) < count:
            candidate_methods.extend(method for method in MUTATION_METHODS if method not in candidate_methods)
        for index, method in enumerate(candidate_methods):
            candidate = mutate_architecture(parent, method, self.seed + index + 1)
            signature = _structural_signature(candidate)
            if signature in local_signatures or signature in seen_signatures:
                continue
            local_signatures.add(signature)
            reason = self._reason(method, residues, history)
            proposals.append(MutationProposal(candidate, method, self.policy, reason))
            if len(proposals) >= count:
                break
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


def _rank_by_credit(
    methods: List[str],
    credit_report: object | None,
    history: SearchHistory | None,
) -> List[str]:
    credits = getattr(credit_report, "method_credit", {}) if credit_report is not None else {}
    risks = getattr(credit_report, "rollback_risk_by_method", {}) if credit_report is not None else {}
    stats = history.method_stats() if history is not None else {}
    if not isinstance(credits, Mapping):
        credits = {}
    if not isinstance(risks, Mapping):
        risks = {}
    return sorted(
        methods,
        key=lambda method: (
            float(credits.get(method, 0.0))
            + 0.08 * float(stats.get(method, {}).get("acceptance_rate", 0.0))
            - 0.35 * float(risks.get(method, stats.get(method, {}).get("rollback_rate", 0.0))),
            -methods.index(method),
        ),
        reverse=True,
    )


def _structural_signature(genome: ArchitectureGenome) -> tuple[object, ...]:
    return (
        genome.input_dim,
        genome.output_dim,
        genome.hidden_dim,
        genome.residual_blocks,
        genome.memory_gates,
        genome.causal_bottlenecks,
        genome.object_binding_adapters,
        genome.sequence_state_adapters,
        genome.memory_slots,
        genome.object_slots,
        genome.sequence_chunks,
        genome.effective_bottleneck_dim,
        genome.activation,
        round(float(genome.dropout), 6),
    )


def _seen_structural_signatures(history: SearchHistory | None) -> set[tuple[object, ...]]:
    signatures: set[tuple[object, ...]] = set()
    if history is None:
        return signatures
    for record in history.records:
        for payload in (record.parent_genome, record.candidate_genome):
            if isinstance(payload, Mapping):
                signatures.add(
                    (
                        int(payload.get("input_dim", 0)),
                        int(payload.get("output_dim", 0)),
                        int(payload.get("hidden_dim", 0)),
                        int(payload.get("residual_blocks", 0)),
                        int(payload.get("memory_gates", 0)),
                        int(payload.get("causal_bottlenecks", 0)),
                        int(payload.get("object_binding_adapters", 0)),
                        int(payload.get("sequence_state_adapters", 0)),
                        int(payload.get("memory_slots", 1)),
                        int(payload.get("object_slots", 1)),
                        int(payload.get("sequence_chunks", 1)),
                        int(payload.get("effective_bottleneck_dim", payload.get("bottleneck_dim", 1) or 1)),
                        str(payload.get("activation", "")),
                        round(float(payload.get("dropout", 0.0)), 6),
                    )
                )
    return signatures
