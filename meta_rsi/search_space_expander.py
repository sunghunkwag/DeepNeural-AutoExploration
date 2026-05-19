"""Search-space registry updates justified by failure residues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from failure_residue import FailureResidue, MissingRepresentationHypothesis


SUPPORTED_BOUNDED_MODULE_FAMILIES = (
    "gated_memory_block",
    "causal_bottleneck_block",
    "wider_residual_stack",
    "object_binding_adapter",
    "sequence_state_adapter",
)


@dataclass
class SearchSpaceRegistry:
    module_families: Dict[str, Dict[str, object]] = field(default_factory=dict)
    min_support: int = 2

    def update_from_hypotheses(
        self,
        hypotheses: Sequence[MissingRepresentationHypothesis],
        residues: Sequence[FailureResidue] = (),
    ) -> Dict[str, Dict[str, object]]:
        updates: Dict[str, Dict[str, object]] = {}
        residue_count_by_family: Dict[str, int] = {}
        hidden_failure_by_family: Dict[str, int] = {}
        residue_ids_by_family: Dict[str, list[str]] = {}
        failure_types_by_family: Dict[str, set[str]] = {}
        for residue in residues:
            if residue.used_heldout_labels:
                raise ValueError("search-space expansion cannot use heldout-test evidence")
            family = residue.proposed_operator_or_module_family
            residue_count_by_family[family] = residue_count_by_family.get(family, 0) + 1
            residue_ids_by_family.setdefault(family, []).append(residue.residue_id)
            failure_types_by_family.setdefault(family, set()).add(residue.failure_type)
            if residue.split_used == "hidden_validation" or residue.evidence_metrics.get("hidden_validation_regression", 0.0) < 0.0:
                hidden_failure_by_family[family] = hidden_failure_by_family.get(family, 0) + 1
        for hypothesis in hypotheses:
            name = hypothesis.proposed_module_family
            if name not in SUPPORTED_BOUNDED_MODULE_FAMILIES:
                continue
            support_count = max(hypothesis.support_count, residue_count_by_family.get(name, 0))
            hidden_count = max(int(getattr(hypothesis, "hidden_validation_support", 0)), hidden_failure_by_family.get(name, 0))
            confidence = max(
                float(getattr(hypothesis, "confidence_score", 0.0)),
                min(1.0, (support_count + 0.75 * hidden_count) / max(2.0, float(self.min_support + 2))),
            )
            if support_count < self.min_support and hidden_count < self.min_support and confidence < 0.55:
                continue
            entry = self.module_families.get(
                name,
                {
                    "module_family": name,
                    "justification_count": 0,
                    "hidden_validation_failure_count": 0,
                    "confidence_score": 0.0,
                    "failure_types": [],
                    "task_ids": [],
                    "justifying_residue_ids": [],
                    "recommended_mutation_operators": [],
                    "bounded_rationale": _bounded_rationale(name),
                    "audit": [],
                },
            )
            entry["justification_count"] = int(entry.get("justification_count", 0)) + support_count
            entry["hidden_validation_failure_count"] = int(entry.get("hidden_validation_failure_count", 0)) + hidden_count
            entry["confidence_score"] = max(float(entry.get("confidence_score", 0.0)), confidence)
            entry["failure_types"] = sorted(
                set(list(entry.get("failure_types", [])) + [hypothesis.failure_type] + list(failure_types_by_family.get(name, set())))
            )
            entry["task_ids"] = sorted(set(list(entry.get("task_ids", [])) + list(hypothesis.task_ids)))
            entry["justifying_residue_ids"] = sorted(
                set(list(entry.get("justifying_residue_ids", [])) + residue_ids_by_family.get(name, []))
            )
            entry["recommended_mutation_operators"] = sorted(
                set(
                    list(entry.get("recommended_mutation_operators", []))
                    + list(getattr(hypothesis, "recommended_mutation_operators", []) or _recommended_mutations(name))
                )
            )
            entry["bounded_rationale"] = _bounded_rationale(name)
            entry["audit"] = list(entry.get("audit", [])) + [
                {
                    "reason": hypothesis.rationale,
                    "support_count": support_count,
                    "hidden_validation_failure_count": hidden_count,
                    "confidence_score": confidence,
                    "used_heldout_labels": False,
                    "recommended_mutation_operators": list(entry["recommended_mutation_operators"]),
                    "bounded_rationale": entry["bounded_rationale"],
                }
            ]
            self.module_families[name] = entry
            updates[name] = dict(entry)
        return updates

    def to_dict(self) -> Dict[str, object]:
        return {"module_families": dict(self.module_families)}


def _recommended_mutations(module_family: str) -> list[str]:
    return {
        "gated_memory_block": ["add_memory_gate"],
        "causal_bottleneck_block": ["add_causal_bottleneck", "repair_bottleneck"],
        "wider_residual_stack": ["wider_residual_stack", "widen_hidden_dim"],
        "object_binding_adapter": ["object_binding_adapter"],
        "sequence_state_adapter": ["sequence_state_adapter"],
    }.get(module_family, ["add_residual_block"])


def _bounded_rationale(module_family: str) -> str:
    return {
        "gated_memory_block": "bounded read/write gates over a fixed number of hidden-state memory slots",
        "causal_bottleneck_block": "fixed-width latent bottleneck with probeable usage statistics",
        "wider_residual_stack": "bounded increase in residual depth and hidden width",
        "object_binding_adapter": "fixed object-slot pairwise attention over hidden features",
        "sequence_state_adapter": "fixed-chunk recurrent transform over hidden features",
    }.get(module_family, "bounded module family with deterministic construction")
