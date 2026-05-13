import json
from pathlib import Path

import pytest

from benchmarks.orchestrator_integration_smoke import run
from scripts.verify_orchestrator_smoke import verify


@pytest.fixture(scope="module")
def smoke_payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("orchestrator") / "orchestrator_integration_smoke_seed42.json"
    result = run("smoke", 42, str(out))
    verify(out)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    return out, result, manifest


def test_smoke_benchmark_writes_result_and_manifest(smoke_payload):
    out, result, manifest = smoke_payload
    assert out.exists()
    assert Path(result["manifest_path"]).exists()
    assert manifest["system_graph"]["nodes"]
    assert manifest["artifact_bus"]["artifacts"]


def test_smoke_metrics_cover_required_routing(smoke_payload):
    _, result, _ = smoke_payload
    assert result["artifact_count"] >= 4
    assert result["accepted_artifact_count"] >= 1
    assert result["rejected_artifact_count"] >= 1
    assert result["failure_residue_count"] >= 1
    assert result["eie_routed_residue_count"] >= 1
    assert result["test_signal_artifact_rejected"] is True
    assert result["used_test_signal_for_acceptance"] is False
    assert result["deterministic_replay_passed"] is True


def test_smoke_memory_records_all_decisions(smoke_payload):
    _, result, manifest = smoke_payload
    memory_records = manifest["structural_memory"]["decision_records"]
    assert len(memory_records) == result["accepted_artifact_count"] + result["rejected_artifact_count"]
    assert all("rejection_reason" in record for record in memory_records)


def test_smoke_manifest_preserves_rejection_reasons(smoke_payload):
    _, _, manifest = smoke_payload
    rejected = [decision for decision in manifest["orchestrator_decisions"] if not decision["accepted"]]
    assert rejected
    assert all(decision["rejection_reason"] for decision in rejected)
    assert any(decision["rejection_reason"] == "artifact_uses_test_signal" for decision in rejected)
