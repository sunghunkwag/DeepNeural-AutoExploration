import json

import pytest

from orchestrator_core import ArtifactBus, ArtifactEvaluation, CandidateArtifact, OrchestratorDecision


def _artifact(artifact_id="a0", artifact_type="operator_program", uses_test_signal=False):
    return CandidateArtifact(
        artifact_id=artifact_id,
        producer="producer0",
        artifact_type=artifact_type,
        payload={"score": 0.8},
        declared_capabilities=(artifact_type,),
        created_generation=0,
        uses_test_signal=uses_test_signal,
        source_repo=None,
    )


def _evaluation(artifact_id="a0", split="validation", uses_test_signal=False, passed=True):
    return ArtifactEvaluation(
        artifact_id=artifact_id,
        evaluator_id="eval0",
        split=split,
        metrics={"score": 0.8},
        passed=passed,
        failure_type=None if passed else "validation_failure",
        severity=0.0 if passed else 0.4,
        evidence={},
        uses_test_signal=uses_test_signal,
    )


def test_bus_submits_artifacts_and_rejects_duplicates():
    bus = ArtifactBus()
    bus.submit(_artifact())
    with pytest.raises(ValueError):
        bus.submit(_artifact())
    assert len(bus.get_artifacts(artifact_type="operator_program")) == 1


def test_bus_blocks_test_signal_acceptance_requests():
    bus = ArtifactBus()
    with pytest.raises(ValueError):
        bus.submit(_artifact(uses_test_signal=True), acceptance_requested=True)


def test_bus_records_evaluations_and_blocks_test_before_freeze():
    bus = ArtifactBus()
    bus.submit(_artifact())
    bus.record_evaluation(_evaluation(split="validation"), acceptance_driving=True)
    with pytest.raises(ValueError):
        bus.record_evaluation(_evaluation(split="test"), acceptance_driving=False)
    bus.freeze_decisions()
    bus.record_evaluation(_evaluation(split="test"), acceptance_driving=False)
    assert len(bus.evaluations_for("a0")) == 2


def test_bus_rejects_invalid_acceptance_usage():
    bus = ArtifactBus()
    bus.submit(_artifact())
    with pytest.raises(ValueError):
        bus.record_evaluation(_evaluation(split="validation", uses_test_signal=True), acceptance_driving=True)


def test_bus_records_decisions_and_filters_status():
    bus = ArtifactBus()
    bus.submit(_artifact("a0"))
    bus.submit(_artifact("a1", "primitive_candidate"))
    bus.record_decision(OrchestratorDecision("d0", "a0", True, "accepted", False, (), "v0", True, False))
    bus.record_decision(OrchestratorDecision("d1", "a1", False, "validation_failure", False, (), None, True, False))
    assert [artifact.artifact_id for artifact in bus.get_artifacts(status="accepted")] == ["a0"]
    assert [artifact.artifact_id for artifact in bus.get_artifacts(status="rejected")] == ["a1"]
    json.dumps(bus.to_dict(), sort_keys=True)
    with pytest.raises(ValueError):
        bus.record_decision(OrchestratorDecision("d2", "a1", False, "again", False, (), None, True, False))
