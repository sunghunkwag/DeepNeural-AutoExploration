"""Mutation strategy helpers used by the deep search controller."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from failure_residue import FailureResidue

from .mutation_policy import FAILURE_TO_MUTATION


MODULE_FAMILY_TO_MUTATION = {
    "gated_memory_block": "add_memory_gate",
    "causal_bottleneck_block": "add_causal_bottleneck",
    "wider_residual_stack": "wider_residual_stack",
    "object_binding_adapter": "object_binding_adapter",
    "sequence_state_adapter": "sequence_state_adapter",
}


@dataclass(frozen=True)
class SearchStrategyPlan:
    methods: List[str]
    failure_modes_addressed: List[str] = field(default_factory=list)
    task_families_helped: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "methods": list(self.methods),
            "failure_modes_addressed": list(self.failure_modes_addressed),
            "task_families_helped": list(self.task_families_helped),
            "rationale": self.rationale,
        }


class SearchStrategy:
    """Build an ordered mutation plan from residues, probes, and method credit."""

    def plan(
        self,
        *,
        depth_level: int,
        residues: Sequence[FailureResidue] = (),
        representation_probe: object | None = None,
        gradient_probe: object | None = None,
        credit_report: object | None = None,
        expansions: Sequence[object] = (),
    ) -> SearchStrategyPlan:
        failure_modes = _ordered_failure_modes(residues, representation_probe, gradient_probe)
        task_families = _task_families(residues)
        methods: List[str] = []
        methods.extend(FAILURE_TO_MUTATION.get(mode, "") for mode in failure_modes)
        methods.extend(_probe_recommendations(representation_probe))
        methods.extend(_probe_recommendations(gradient_probe))
        if depth_level >= 3:
            methods.extend(_expansion_methods(expansions))
        if depth_level == 0:
            methods.extend(["widen_hidden_dim", "narrow_hidden_dim"])
            count = 1
        elif depth_level == 1:
            methods.extend(["widen_hidden_dim", "add_residual_block", "add_memory_gate"])
            count = 1
        elif depth_level == 2:
            methods.extend(["wider_residual_stack", "add_memory_gate", "add_causal_bottleneck", "object_binding_adapter"])
            count = 2
        elif depth_level == 3:
            methods.extend(["object_binding_adapter", "sequence_state_adapter", "add_causal_bottleneck", "wider_residual_stack"])
            count = 3
        else:
            methods.extend(["wider_residual_stack", "object_binding_adapter", "sequence_state_adapter", "add_causal_bottleneck"])
            count = 3
        methods = _rank_by_credit(_dedupe(methods), credit_report)[:count]
        if not methods:
            methods = ["add_residual_block"]
        return SearchStrategyPlan(
            methods=methods,
            failure_modes_addressed=failure_modes,
            task_families_helped=task_families,
            rationale=f"depth {depth_level} selected {','.join(methods)} from residues/probes/history",
        )


def _ordered_failure_modes(
    residues: Sequence[FailureResidue],
    representation_probe: object | None,
    gradient_probe: object | None,
) -> List[str]:
    counter = Counter(residue.failure_type for residue in residues)
    counter.update(getattr(representation_probe, "failure_modes", []) or [])
    counter.update(getattr(gradient_probe, "failure_modes", []) or [])
    return [name for name, _count in counter.most_common()]


def _task_families(residues: Sequence[FailureResidue]) -> List[str]:
    names = set()
    for residue in residues:
        task_id = residue.task_id
        for family in (
            "function_regression",
            "sequence_prediction",
            "object_state_transition",
            "causal_intervention",
            "memory_retrieval",
        ):
            if family in task_id:
                names.add(family)
    return sorted(names)


def _probe_recommendations(probe: object | None) -> List[str]:
    recommendations = getattr(probe, "recommended_mutations", {}) if probe is not None else {}
    methods: List[str] = []
    if isinstance(recommendations, dict):
        for values in recommendations.values():
            if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
                methods.extend(str(value) for value in values)
    return methods


def _expansion_methods(expansions: Sequence[object]) -> List[str]:
    methods: List[str] = []
    for item in expansions:
        if isinstance(item, dict):
            family = str(item.get("module_family", ""))
            methods.append(MODULE_FAMILY_TO_MUTATION.get(family, ""))
            methods.extend(str(value) for value in item.get("recommended_mutation_operators", []) if value)
    return methods


def _rank_by_credit(methods: List[str], credit_report: object | None) -> List[str]:
    credits = getattr(credit_report, "method_credit", {}) if credit_report is not None else {}
    risks = getattr(credit_report, "rollback_risk_by_method", {}) if credit_report is not None else {}
    if not isinstance(credits, dict):
        credits = {}
    if not isinstance(risks, dict):
        risks = {}
    return sorted(
        methods,
        key=lambda method: (float(credits.get(method, 0.0)) - 0.25 * float(risks.get(method, 0.0)), -methods.index(method)),
        reverse=True,
    )


def _dedupe(values: Iterable[str]) -> List[str]:
    allowed = {
        "add_residual_block",
        "remove_residual_block",
        "widen_hidden_dim",
        "narrow_hidden_dim",
        "add_memory_gate",
        "add_causal_bottleneck",
        "wider_residual_stack",
        "object_binding_adapter",
        "sequence_state_adapter",
        "repair_bottleneck",
    }
    out: List[str] = []
    for value in values:
        value = str(value)
        if value in allowed and value not in out:
            out.append(value)
    return out

