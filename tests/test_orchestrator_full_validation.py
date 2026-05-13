import json
from pathlib import Path

import pytest

from benchmarks.orchestrator_integration_smoke import run
from scripts.verify_orchestrator_smoke import verify


@pytest.fixture(scope="module")
def full_payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("orchestrator_full") / "orchestrator_integration_full_seed42.json"
    result = run("full", 42, str(out))
    verify(out)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    return result, manifest


def test_full_validation_thresholds(full_payload):
    result, _ = full_payload
    assert result["artifact_count"] >= 10
    assert result["accepted_artifact_count"] >= 2
    assert result["rejected_artifact_count"] >= 3
    assert result["failure_residue_count"] >= 3
    assert result["eie_routed_residue_count"] >= 3
    assert result["validation_evaluation_count"] >= 3
    assert result["hidden_validation_evaluation_count"] >= 3
    assert result["full_validation_passed"] is True


def test_full_validation_has_all_artifact_types(full_payload):
    _, manifest = full_payload
    artifact_types = {artifact["artifact_type"] for artifact in manifest["artifact_bus"]["artifacts"]}
    assert {
        "operator_program",
        "primitive_candidate",
        "architecture_candidate",
        "evaluator_candidate",
        "observation_channel_candidate",
    }.issubset(artifact_types)


def test_full_validation_held_out_test_is_post_freeze(full_payload):
    result, manifest = full_payload
    assert result["held_out_test_record_count"] >= 1
    assert manifest["artifact_bus"]["decisions_frozen"] is True
    assert manifest["held_out_test_used_only_after_freeze"] is True
    assert all(not decision["uses_test_signal_for_acceptance"] for decision in manifest["orchestrator_decisions"])
