"""Evidence-based search depth scheduling for bounded neural exploration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

from failure_residue import FailureResidue

from .architecture_genome import ArchitectureGenome


SHALLOW_MUTATIONS = {
    "widen_hidden_dim",
    "narrow_hidden_dim",
    "add_residual_block",
    "remove_residual_block",
}


@dataclass(frozen=True)
class SearchDepthAssessment:
    selected_depth_level: int
    rationale: str
    evidence_used: Dict[str, float | str | list[str]] = field(default_factory=dict)
    exploration_vs_exploitation_score: float = 0.5
    complexity_penalty: float = 0.0
    rollback_risk: float = 0.0
    used_heldout_labels: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_depth_level": self.selected_depth_level,
            "rationale": self.rationale,
            "evidence_used": dict(self.evidence_used),
            "exploration_vs_exploitation_score": self.exploration_vs_exploitation_score,
            "complexity_penalty": self.complexity_penalty,
            "rollback_risk": self.rollback_risk,
            "used_heldout_labels": self.used_heldout_labels,
        }


class SearchDepthScheduler:
    """Choose bounded mutation depth from validation, history, and probe evidence."""

    def __init__(self, *, max_depth: int = 4) -> None:
        self.max_depth = max(0, min(4, int(max_depth)))

    def schedule(
        self,
        parent: ArchitectureGenome,
        *,
        residues: Sequence[FailureResidue] = (),
        history: object | None = None,
        representation_probe: object | None = None,
        gradient_probe: object | None = None,
        search_space_confidence: Mapping[str, float] | None = None,
    ) -> SearchDepthAssessment:
        records = list(getattr(history, "records", []) if history is not None else [])
        recent = records[-6:]
        rejected_recent = [record for record in recent if not bool(getattr(record, "accepted", False))]
        accepted_recent = [record for record in recent if bool(getattr(record, "accepted", False))]
        shallow_failures = sum(
            1
            for record in rejected_recent
            if str(getattr(record, "mutation_method", "")) in SHALLOW_MUTATIONS
        )
        hidden_regressions = sum(1 for record in recent if _record_float(record, "hidden_validation_improvement") < 0.0)
        validation_hidden_mismatch = sum(
            1
            for record in recent
            if _record_float(record, "validation_improvement") > 0.0
            and _record_float(record, "hidden_validation_improvement") <= 0.0
        )
        useful_shallow = sum(
            1
            for record in accepted_recent
            if str(getattr(record, "mutation_method", "")) in SHALLOW_MUTATIONS
            and _record_float(record, "hidden_validation_improvement") > 0.0
        )
        residue_counts = Counter(residue.failure_type for residue in residues)
        persistent_residues = [name for name, count in residue_counts.items() if count >= 2]
        representation_modes = list(getattr(representation_probe, "failure_modes", []) or [])
        gradient_modes = list(getattr(gradient_probe, "failure_modes", []) or [])
        structural_probe_failures = [
            mode
            for mode in representation_modes + gradient_modes
            if mode
            in {
                "representation collapse",
                "task-family entanglement",
                "gradient bottleneck",
                "dead module",
                "over-compressed bottleneck",
                "module over-dominance",
                "high validation-hidden-validation mismatch",
            }
        ]
        expansion_confidence = max(search_space_confidence.values()) if search_space_confidence else 0.0
        complexity_penalty = _architecture_complexity_penalty(parent)
        rollback_risk = _clip01(
            (len(rejected_recent) / max(1, len(recent)))
            + 0.08 * hidden_regressions
            + 0.06 * validation_hidden_mismatch
            + 0.35 * complexity_penalty
        )

        evidence_score = (
            0.35 * min(1.0, shallow_failures / 3.0)
            + 0.20 * min(1.0, hidden_regressions / 2.0)
            + 0.20 * min(1.0, len(persistent_residues) / 3.0)
            + 0.20 * min(1.0, len(structural_probe_failures) / 3.0)
            + 0.15 * min(1.0, expansion_confidence)
        )
        if useful_shallow:
            evidence_score -= 0.25 * min(1.0, useful_shallow / 2.0)
        if rollback_risk > 0.8:
            evidence_score -= 0.15
        depth = 0
        if evidence_score >= 0.78 and (persistent_residues or expansion_confidence >= 0.65):
            depth = 4 if hidden_regressions >= 3 else 3
        elif evidence_score >= 0.55:
            depth = 2
        elif evidence_score >= 0.25:
            depth = 1
        if shallow_failures >= 3 and hidden_regressions >= 2 and persistent_residues:
            depth = max(depth, 2)
        depth = min(depth, self.max_depth)
        if useful_shallow and depth > 1 and len(structural_probe_failures) < 2:
            depth = 1
        rationale = _rationale(depth, shallow_failures, hidden_regressions, persistent_residues, structural_probe_failures)
        exploration = _clip01(0.35 + 0.45 * evidence_score + 0.20 * min(1.0, len(persistent_residues) / 3.0) - 0.20 * rollback_risk)
        return SearchDepthAssessment(
            selected_depth_level=depth,
            rationale=rationale,
            evidence_used={
                "shallow_failures": float(shallow_failures),
                "hidden_validation_regressions": float(hidden_regressions),
                "validation_hidden_mismatch_count": float(validation_hidden_mismatch),
                "useful_shallow_mutations": float(useful_shallow),
                "persistent_residues": persistent_residues,
                "structural_probe_failures": structural_probe_failures,
                "search_space_expansion_confidence": float(expansion_confidence),
            },
            exploration_vs_exploitation_score=exploration,
            complexity_penalty=complexity_penalty,
            rollback_risk=rollback_risk,
            used_heldout_labels=False,
        )


def _architecture_complexity_penalty(genome: ArchitectureGenome) -> float:
    module_count = (
        genome.residual_blocks
        + genome.memory_gates
        + genome.causal_bottlenecks
        + genome.object_binding_adapters
        + genome.sequence_state_adapters
    )
    return _clip01((genome.hidden_dim / 128.0) + 0.035 * module_count)


def _record_float(record: object, name: str) -> float:
    if hasattr(record, name):
        return float(getattr(record, name))
    if hasattr(record, "selection_score_components") and isinstance(getattr(record, "selection_score_components"), Mapping):
        return float(getattr(record, "selection_score_components").get(name, 0.0))
    return 0.0


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _rationale(
    depth: int,
    shallow_failures: int,
    hidden_regressions: int,
    persistent_residues: Sequence[str],
    structural_probe_failures: Sequence[str],
) -> str:
    parts = [f"depth={depth}"]
    if shallow_failures:
        parts.append(f"{shallow_failures} shallow rollback(s)")
    if hidden_regressions:
        parts.append(f"{hidden_regressions} hidden-validation regression(s)")
    if persistent_residues:
        parts.append("persistent residues: " + ",".join(sorted(persistent_residues)))
    if structural_probe_failures:
        parts.append("probe bottlenecks: " + ",".join(sorted(set(structural_probe_failures))))
    return "; ".join(parts)
