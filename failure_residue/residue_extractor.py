"""Convert failed benchmark traces into structured failure residues."""

from __future__ import annotations

from typing import Dict, Mapping

from .residue_schema import FailureResidue


ARC_FAILURE_TAXONOMY = (
    "object-binding failure",
    "variable-shape failure",
    "color-rule ambiguity",
    "insufficient support evidence",
    "primitive-library blind spot",
    "evaluator-selection failure",
    "architecture-representation bottleneck",
)

NEURAL_FAILURE_TAXONOMY = (
    "representation collapse",
    "gradient bottleneck",
    "dead module",
    "over-compressed bottleneck",
    "task-family entanglement",
    "unstable hidden dynamics",
    "hidden-state saturation",
    "module over-dominance",
    "high validation-hidden-validation mismatch",
    "causal intervention failure",
    "sequence instability",
)


_FAILURE_INFO = {
    "object-binding failure": (
        "object identity or relation binding is not represented robustly",
        "object-centric binding module",
    ),
    "variable-shape failure": (
        "candidate representation assumes fixed output shape or aligned cells",
        "variable-shape adapter",
    ),
    "color-rule ambiguity": (
        "support evidence admits multiple color mappings or recoloring rules",
        "color-rule induction module",
    ),
    "insufficient support evidence": (
        "support examples are too sparse for a stable rule choice",
        "support-evidence expansion operator",
    ),
    "primitive-library blind spot": (
        "available symbolic or neural primitives do not cover the observed transform",
        "primitive-library expansion",
    ),
    "evaluator-selection failure": (
        "validation selector accepted or preferred a candidate that did not survive hidden validation",
        "evaluator repair guard",
    ),
    "architecture-representation bottleneck": (
        "neural architecture lacks a representation path for the failure pattern",
        "neural architecture expansion",
    ),
    "representation collapse": (
        "activation geometry has collapsed or carries too little variance",
        "wider_residual_stack",
    ),
    "gradient bottleneck": (
        "gradient flow is too small or sharply concentrated in a narrow path",
        "wider_residual_stack",
    ),
    "dead module": (
        "one or more bounded modules has negligible activation or gradient contribution",
        "gated_memory_block",
    ),
    "over-compressed bottleneck": (
        "causal bottleneck usage is too low for the task evidence",
        "causal_bottleneck_block",
    ),
    "task-family entanglement": (
        "task-family embeddings are poorly separated in hidden space",
        "object_binding_adapter",
    ),
    "unstable hidden dynamics": (
        "hidden states are too similar or unstable across task families",
        "sequence_state_adapter",
    ),
    "hidden-state saturation": (
        "hidden activations are saturated enough to limit representational range",
        "wider_residual_stack",
    ),
    "module over-dominance": (
        "one module dominates contribution or gradients over the rest of the architecture",
        "gated_memory_block",
    ),
    "high validation-hidden-validation mismatch": (
        "validation improves do not survive hidden-validation evidence",
        "causal_bottleneck_block",
    ),
    "causal intervention failure": (
        "candidate does not separate observed correlation from intervention behavior",
        "causal_bottleneck_block",
    ),
    "sequence instability": (
        "candidate does not maintain state over longer sequence dependencies",
        "sequence_state_adapter",
    ),
}


def _numeric_metrics(trace: Mapping[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    raw = trace.get("evidence_metrics", {})
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metrics[str(key)] = float(value)
    for key, value in trace.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and key not in {"seed"}:
            metrics.setdefault(str(key), float(value))
    return metrics


def classify_failure_type(trace: Mapping[str, object]) -> str:
    explicit = str(trace.get("failure_type", "")).strip()
    if explicit in ARC_FAILURE_TAXONOMY or explicit in NEURAL_FAILURE_TAXONOMY:
        return explicit

    metrics = _numeric_metrics(trace)
    reason = " ".join(
        str(trace.get(key, ""))
        for key in ("failure_type", "rejection_reason", "suspected_bottleneck", "notes")
    ).lower()
    if metrics.get("object_binding_error_rate", 0.0) > 0.0 or "object" in reason and "binding" in reason:
        return "object-binding failure"
    if metrics.get("shape_mismatch_rate", 0.0) > 0.0 or metrics.get("variable_shape_error_rate", 0.0) > 0.0 or "shape" in reason:
        return "variable-shape failure"
    if metrics.get("color_ambiguity_rate", 0.0) > 0.0 or "color" in reason and "ambigu" in reason:
        return "color-rule ambiguity"
    if metrics.get("support_pair_count", 2.0) <= 1.0 or metrics.get("support_conflict_count", 0.0) > 0.0:
        return "insufficient support evidence"
    if metrics.get("primitive_miss_rate", 0.0) > 0.0 or "primitive" in reason and "missing" in reason:
        return "primitive-library blind spot"
    if metrics.get("hidden_validation_regression", 0.0) < 0.0 or "selector" in reason or "hidden validation" in reason:
        return "evaluator-selection failure"
    if metrics.get("parameter_budget_pressure", 0.0) > 0.0 or "architecture" in reason or "representation" in reason:
        return "architecture-representation bottleneck"
    if "gradient" in reason:
        return "gradient bottleneck"
    if "sequence" in reason:
        return "sequence instability"
    if "causal" in reason or "intervention" in reason:
        return "causal intervention failure"
    return explicit or "validation failure"


class ResidueExtractor:
    """Extract leakage-safe failure residues from failed traces."""

    def extract(self, trace: Mapping[str, object]) -> FailureResidue:
        used_heldout = bool(trace.get("used_heldout_labels", trace.get("used_test_results", False)))
        split_used = str(trace.get("split_used", trace.get("split", "validation")))
        if used_heldout:
            raise ValueError("refusing residue extraction from held-out labels used for candidate generation")
        if split_used == "test":
            raise ValueError("refusing residue extraction from test split candidate-selection evidence")

        failure_type = classify_failure_type(trace)
        suspected, proposed = _FAILURE_INFO.get(
            failure_type,
            ("candidate failed validation without a more specific representation diagnosis", "bounded search-space review"),
        )
        metrics = _numeric_metrics(trace)
        failed = trace.get("failed_operator_or_architecture") or trace.get("candidate_id") or trace.get("program_id") or trace.get("candidate_name") or "unknown"
        return FailureResidue(
            task_id=str(trace.get("task_id", "unknown")),
            benchmark_name=str(trace.get("benchmark_name", trace.get("benchmark", "unknown"))),
            seed=int(trace.get("seed", trace.get("random_seed", 0))),
            failure_type=failure_type,
            failed_operator_or_architecture=str(failed),
            suspected_bottleneck=str(trace.get("suspected_bottleneck", suspected)),
            evidence_metrics=metrics,
            missing_representation_hypothesis=str(trace.get("missing_representation_hypothesis", suspected)),
            proposed_operator_or_module_family=str(trace.get("proposed_operator_or_module_family", proposed)),
            split_used=split_used,
            used_heldout_labels=False,
        )

    def extract_many(self, traces) -> list[FailureResidue]:
        return [self.extract(trace) for trace in traces]
