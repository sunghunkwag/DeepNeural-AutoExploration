"""Credit assignment for bounded architecture mutation methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class SearchCreditReport:
    method_credit: Dict[str, float] = field(default_factory=dict)
    task_family_credit: Dict[str, float] = field(default_factory=dict)
    module_contribution_score: float = 0.0
    rollback_risk_by_method: Dict[str, float] = field(default_factory=dict)
    rationale: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "method_credit": dict(self.method_credit),
            "task_family_credit": dict(self.task_family_credit),
            "module_contribution_score": self.module_contribution_score,
            "rollback_risk_by_method": dict(self.rollback_risk_by_method),
            "rationale": dict(self.rationale),
        }


class SearchCreditAssigner:
    """Assign conservative credit from hidden-validation and rollback outcomes."""

    def assign(
        self,
        *,
        history: object | None = None,
        representation_probe: object | None = None,
        gradient_probe: object | None = None,
    ) -> SearchCreditReport:
        records = list(getattr(history, "records", []) if history is not None else [])
        method_credit: Dict[str, float] = {}
        task_family_credit: Dict[str, float] = {}
        rollback_counts: Dict[str, float] = {}
        method_counts: Dict[str, float] = {}
        rationale: Dict[str, str] = {}

        for record in records:
            method = str(getattr(record, "mutation_method", "unknown"))
            method_counts[method] = method_counts.get(method, 0.0) + 1.0
            validation_gain = _record_float(record, "validation_improvement")
            hidden_gain = _record_float(record, "hidden_validation_improvement")
            transfer_gain = _record_float(record, "cross_domain_transfer_delta")
            accepted = bool(getattr(record, "accepted", False))
            credit = 0.25 * validation_gain + 0.55 * hidden_gain + 0.20 * transfer_gain
            if accepted:
                credit += 0.04
            else:
                credit -= 0.08
                rollback_counts[method] = rollback_counts.get(method, 0.0) + 1.0
            method_credit[method] = method_credit.get(method, 0.0) + credit
            per_family = getattr(record, "task_family_improvements", {})
            if isinstance(per_family, Mapping):
                for family, value in per_family.items():
                    task_family_credit[str(family)] = task_family_credit.get(str(family), 0.0) + float(value)

        for method, total in list(method_credit.items()):
            count = max(1.0, method_counts.get(method, 1.0))
            method_credit[method] = float(total / count)
            rejects = rollback_counts.get(method, 0.0)
            rationale[method] = (
                f"mean hidden-weighted credit={method_credit[method]:.4f}; "
                f"rollback_rate={rejects / count:.2f}"
            )
        rollback_risk_by_method = {
            method: float(rollback_counts.get(method, 0.0) / max(1.0, count))
            for method, count in method_counts.items()
        }
        module_contribution = _module_contribution(representation_probe, gradient_probe)
        return SearchCreditReport(
            method_credit=method_credit,
            task_family_credit=task_family_credit,
            module_contribution_score=module_contribution,
            rollback_risk_by_method=rollback_risk_by_method,
            rationale=rationale,
        )


def _record_float(record: object, name: str) -> float:
    if hasattr(record, name):
        return float(getattr(record, name))
    components = getattr(record, "selection_score_components", {})
    if isinstance(components, Mapping):
        return float(components.get(name, 0.0))
    return 0.0


def _module_contribution(representation_probe: object | None, gradient_probe: object | None) -> float:
    values = []
    ablations = getattr(representation_probe, "module_ablation_contribution", {}) if representation_probe is not None else {}
    if isinstance(ablations, Mapping) and ablations:
        values.append(mean(abs(float(value)) for value in ablations.values()))
    norms = getattr(gradient_probe, "gradient_norm_by_module", {}) if gradient_probe is not None else {}
    if isinstance(norms, Mapping) and norms:
        total = sum(abs(float(value)) for value in norms.values())
        values.append(total / max(1, len(norms)))
    return float(mean(values)) if values else 0.0

