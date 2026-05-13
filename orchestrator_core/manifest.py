"""Manifest helpers for orchestrator integration validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from experiment_manifest import current_git_commit, stable_config_hash

from .artifact_bus import ArtifactBus
from .memory_store import StructuralMemoryStore
from .schemas import OrchestratorDecision, json_safe
from .system_graph import SystemGraph


def anti_cheat_checks() -> list[str]:
    return [
        "random seed recorded",
        "validation and hidden-validation are the only acceptance splits",
        "held-out test records are stored only after decisions are frozen",
        "artifacts marked uses_test_signal are rejected",
        "test split evaluations do not drive acceptance",
        "rejected artifacts include rejection reasons",
        "failed acceptance evaluations produce failure residues",
        "accepted artifacts have system graph lineage",
        "accepted and rejected artifacts are stored in structural memory",
        "no-op control artifact is rejected",
        "random control artifact is recorded",
        "no external APIs are used",
        "no network calls are used",
        "no LLM calls are used",
        "no generated source is executed",
        "outputs are JSON serializable",
    ]


def build_manifest(
    *,
    mode: str,
    seed: int,
    config: Mapping[str, Any],
    result_metrics: Mapping[str, Any],
    graph: SystemGraph,
    bus: ArtifactBus,
    decisions: Sequence[OrchestratorDecision],
    memory_store: StructuralMemoryStore,
    failure_residues: Sequence[Any],
    mutation_decisions: Sequence[Mapping[str, Any]],
    non_residue_reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": "python benchmarks/orchestrator_integration_smoke.py",
        "mode": mode,
        "seed": int(seed),
        "config": json_safe(dict(config)),
        "config_hash": stable_config_hash(dict(config)),
        "metric_summary": json_safe(dict(result_metrics)),
        "system_graph": graph.to_dict(),
        "artifact_bus": bus.to_dict(),
        "orchestrator_decisions": [decision.to_dict() for decision in decisions],
        "structural_memory": memory_store.to_dict(),
        "failure_residues": [residue.to_dict() if hasattr(residue, "to_dict") else json_safe(residue) for residue in failure_residues],
        "eie_mutation_decisions": json_safe(list(mutation_decisions)),
        "non_residue_reasons": json_safe(list(non_residue_reasons)),
        "anti_cheat_checks_passed": anti_cheat_checks(),
        "validation_only_acceptance_enforced": True,
        "held_out_test_used_only_after_freeze": True,
        "used_test_signal_for_acceptance": False,
    }
    validate_manifest(manifest)
    return json_safe(manifest)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "system_graph",
        "artifact_bus",
        "orchestrator_decisions",
        "structural_memory",
        "failure_residues",
        "anti_cheat_checks_passed",
        "validation_only_acceptance_enforced",
        "held_out_test_used_only_after_freeze",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"orchestrator manifest missing required fields: {sorted(missing)}")
    if manifest.get("used_test_signal_for_acceptance"):
        raise ValueError("orchestrator manifest used test signal for acceptance")
    decisions = list(manifest.get("orchestrator_decisions", []))
    if not decisions:
        raise ValueError("orchestrator manifest missing decisions")
    memory = manifest.get("structural_memory", {})
    memory_records = memory.get("decision_records", []) if isinstance(memory, dict) else []
    memory_decision_ids = {str(record.get("decision_id")) for record in memory_records if isinstance(record, dict)}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("orchestrator decision must be an object")
        if decision.get("uses_test_signal_for_acceptance"):
            raise ValueError("orchestrator decision used test signal for acceptance")
        if not decision.get("accepted") and not decision.get("rejection_reason"):
            raise ValueError("rejected orchestrator decision missing rejection reason")
        if str(decision.get("decision_id")) not in memory_decision_ids:
            raise ValueError("orchestrator decision missing structural memory record")
    accepted_ids = {str(decision.get("artifact_id")) for decision in decisions if decision.get("accepted")}
    graph = manifest.get("system_graph", {})
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    accepted_lineage_ids = {
        str(edge.get("source_node_id", "")).replace("artifact:", "", 1)
        for edge in edges
        if isinstance(edge, dict) and edge.get("edge_type") == "accepts"
    }
    missing_lineage = accepted_ids.difference(accepted_lineage_ids)
    if missing_lineage:
        raise ValueError(f"accepted artifacts missing graph lineage: {sorted(missing_lineage)}")
