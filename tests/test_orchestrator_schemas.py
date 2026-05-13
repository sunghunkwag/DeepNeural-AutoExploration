import json

import pytest

from orchestrator_core.schemas import (
    ArtifactEvaluation,
    CandidateArtifact,
    OrchestratorDecision,
    SystemEdge,
    SystemNode,
)


def _artifact(**overrides):
    payload = {
        "artifact_id": "a0",
        "producer": "producer0",
        "artifact_type": "operator_program",
        "payload": {"score": 0.8},
        "declared_capabilities": ("operator_program",),
        "created_generation": 0,
        "uses_test_signal": False,
        "source_repo": "Hybrid-Program-Synthesis-with-RSI",
    }
    payload.update(overrides)
    return CandidateArtifact(**payload)


def test_system_node_and_edge_export_json():
    node = SystemNode("n0", "orchestrator", "repo", "module", ("route",), {"k": ("v",)})
    edge = SystemEdge("e0", "n0", "n1", "routes_to", "operator_program", {"weight": 1})
    json.dumps(node.to_dict(), sort_keys=True)
    json.dumps(edge.to_dict(), sort_keys=True)
    assert node.to_dict()["capabilities"] == ["route"]
    assert edge.to_dict()["metadata"]["weight"] == 1


def test_invalid_node_and_edge_types_are_rejected():
    with pytest.raises(ValueError):
        SystemNode("n0", "invalid", None, None)
    with pytest.raises(ValueError):
        SystemEdge("e0", "n0", "n1", "invalid", None)


def test_candidate_artifact_validation_and_json_export():
    artifact = _artifact()
    payload = artifact.to_dict()
    assert payload["artifact_type"] == "operator_program"
    assert payload["declared_capabilities"] == ["operator_program"]
    json.dumps(payload, sort_keys=True)
    with pytest.raises(ValueError):
        _artifact(artifact_type="invalid")


def test_artifact_evaluation_validation_rules():
    evaluation = ArtifactEvaluation(
        "a0",
        "eval0",
        "validation",
        {"score": 0.8},
        True,
        None,
        0.0,
        {},
        False,
    )
    assert evaluation.to_dict()["passed"] is True
    with pytest.raises(ValueError):
        ArtifactEvaluation("a0", "eval0", "train", {}, True, None, 0.0, {}, False)
    with pytest.raises(ValueError):
        ArtifactEvaluation("a0", "eval0", "validation", {}, False, None, 0.2, {}, False)
    with pytest.raises(ValueError):
        ArtifactEvaluation("a0", "eval0", "validation", {}, True, None, 1.2, {}, False)


def test_rejected_decision_requires_reason():
    with pytest.raises(ValueError):
        OrchestratorDecision("d0", "a0", False, "", False, (), None, False, False)
    decision = OrchestratorDecision("d1", "a0", True, "", False, (), "v0", False, False)
    assert decision.rejection_reason == "accepted"
    assert decision.with_memory_status(True).stored_in_memory is True
