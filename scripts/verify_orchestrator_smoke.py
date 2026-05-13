"""Verify orchestrator integration benchmark outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FIELDS = {
    "mode",
    "artifact_count",
    "accepted_artifact_count",
    "rejected_artifact_count",
    "test_signal_artifact_rejected",
    "failure_residue_count",
    "eie_routed_residue_count",
    "memory_record_count",
    "graph_node_count",
    "graph_edge_count",
    "lineage_record_count",
    "used_test_signal_for_acceptance",
    "deterministic_replay_passed",
    "no_op_control_rejected",
    "random_control_recorded",
    "validation_evaluation_count",
    "hidden_validation_evaluation_count",
    "held_out_test_record_count",
    "full_validation_passed",
    "manifest_path",
}

REQUIRED_MANIFEST_FIELDS = {
    "system_graph",
    "artifact_bus",
    "orchestrator_decisions",
    "structural_memory",
    "failure_residues",
    "anti_cheat_checks_passed",
    "validation_only_acceptance_enforced",
    "held_out_test_used_only_after_freeze",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AssertionError(f"missing file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return payload


def _metric(result: Mapping[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    raise AssertionError(f"result missing required field: {key}")


def _manifest_path(result_path: Path, result: Mapping[str, Any]) -> Path:
    raw = result.get("manifest_path")
    if not raw:
        raise AssertionError("manifest_path is missing")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return path if path.exists() else result_path.parent / path.name


def verify(result_path: str | Path) -> None:
    result_path = Path(result_path)
    result = _load(result_path)
    for field in REQUIRED_FIELDS:
        _metric(result, field)

    if int(_metric(result, "artifact_count")) < 4:
        raise AssertionError("artifact_count must be at least 4")
    if int(_metric(result, "accepted_artifact_count")) < 1:
        raise AssertionError("accepted_artifact_count must be at least 1")
    if int(_metric(result, "rejected_artifact_count")) < 1:
        raise AssertionError("rejected_artifact_count must be at least 1")
    if not bool(_metric(result, "test_signal_artifact_rejected")):
        raise AssertionError("test_signal_artifact_rejected must be true")
    if int(_metric(result, "failure_residue_count")) < 1:
        raise AssertionError("failure_residue_count must be at least 1")
    if int(_metric(result, "eie_routed_residue_count")) < 1:
        raise AssertionError("eie_routed_residue_count must be at least 1")
    if int(_metric(result, "memory_record_count")) < 1:
        raise AssertionError("memory_record_count must be at least 1")
    if int(_metric(result, "graph_node_count")) < 4:
        raise AssertionError("graph_node_count must be at least 4")
    if int(_metric(result, "graph_edge_count")) < 3:
        raise AssertionError("graph_edge_count must be at least 3")
    if int(_metric(result, "lineage_record_count")) < 1:
        raise AssertionError("lineage_record_count must be at least 1")
    if bool(_metric(result, "used_test_signal_for_acceptance")):
        raise AssertionError("used_test_signal_for_acceptance must be false")
    if not bool(_metric(result, "deterministic_replay_passed")):
        raise AssertionError("deterministic_replay_passed must be true")
    if not bool(_metric(result, "no_op_control_rejected")):
        raise AssertionError("no_op_control_rejected must be true")
    if not bool(_metric(result, "random_control_recorded")):
        raise AssertionError("random_control_recorded must be true")

    manifest = _load(_manifest_path(result_path, result))
    missing = REQUIRED_MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise AssertionError(f"manifest missing required fields: {sorted(missing)}")
    if manifest.get("used_test_signal_for_acceptance"):
        raise AssertionError("manifest used_test_signal_for_acceptance must be false")
    if not manifest.get("validation_only_acceptance_enforced"):
        raise AssertionError("manifest did not enforce validation-only acceptance")
    if not manifest.get("held_out_test_used_only_after_freeze"):
        raise AssertionError("manifest did not record post-freeze held-out test usage")
    for decision in manifest.get("orchestrator_decisions", []):
        if decision.get("uses_test_signal_for_acceptance"):
            raise AssertionError("decision used test signal for acceptance")
        if not decision.get("accepted") and not decision.get("rejection_reason"):
            raise AssertionError("rejected decision missing rejection reason")

    if str(_metric(result, "mode")) == "full":
        _verify_full(result)


def _verify_full(result: Mapping[str, Any]) -> None:
    if int(_metric(result, "artifact_count")) < 10:
        raise AssertionError("full artifact_count must be at least 10")
    if int(_metric(result, "accepted_artifact_count")) < 2:
        raise AssertionError("full accepted_artifact_count must be at least 2")
    if int(_metric(result, "rejected_artifact_count")) < 3:
        raise AssertionError("full rejected_artifact_count must be at least 3")
    if int(_metric(result, "failure_residue_count")) < 3:
        raise AssertionError("full failure_residue_count must be at least 3")
    if int(_metric(result, "eie_routed_residue_count")) < 3:
        raise AssertionError("full eie_routed_residue_count must be at least 3")
    if int(_metric(result, "validation_evaluation_count")) < 3:
        raise AssertionError("full validation_evaluation_count must be at least 3")
    if int(_metric(result, "hidden_validation_evaluation_count")) < 3:
        raise AssertionError("full hidden_validation_evaluation_count must be at least 3")
    required_memory = int(_metric(result, "accepted_artifact_count")) + int(_metric(result, "rejected_artifact_count"))
    if int(_metric(result, "memory_record_count")) < required_memory:
        raise AssertionError("full memory_record_count is too low")
    if int(_metric(result, "graph_node_count")) < 6:
        raise AssertionError("full graph_node_count must be at least 6")
    if int(_metric(result, "graph_edge_count")) < 6:
        raise AssertionError("full graph_edge_count must be at least 6")
    if int(_metric(result, "lineage_record_count")) < int(_metric(result, "accepted_artifact_count")):
        raise AssertionError("full lineage_record_count must cover accepted artifacts")
    if not bool(_metric(result, "full_validation_passed")):
        raise AssertionError("full_validation_passed must be true")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/verify_orchestrator_smoke.py <result-json>")
    verify(sys.argv[1])


if __name__ == "__main__":
    main()
