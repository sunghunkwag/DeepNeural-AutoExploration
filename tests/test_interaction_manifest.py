import pytest

from interaction_manifest import (
    REQUIRED_INTERACTION_METRICS,
    InteractionBenchmarkManifest,
    build_interaction_manifest,
    stable_config_hash,
    validate_interaction_manifest,
)


def _metrics():
    values = {key: 0 for key in REQUIRED_INTERACTION_METRICS}
    values.update(
        {
            "interaction_quality": 0.8,
            "total_residue_severity": 1.2,
            "background_delivery_rate": 0.75,
            "deterministic_replay_passed": True,
            "no_op_control_score": 0.2,
            "random_control_score": 0.4,
            "always_interrupt_control_score": 0.1,
            "always_silence_control_score": 0.2,
            "residue_count": 3,
        }
    )
    return values


def _manifest(**overrides):
    config = {"mode": "smoke", "seed": 42}
    data = {
        "command": "python benchmarks/interaction_residue_benchmark.py --mode smoke --seed 42",
        "timestamp": "2026-05-12T00:00:00+00:00",
        "seed": 42,
        "mode": "smoke",
        "git_commit": "abc123",
        "number_of_events": 5,
        "number_of_actions": 5,
        "number_of_residues": 3,
        "task_family_names": ["turn_taking", "proactive_visual_event"],
        "used_test_labels": False,
        "deterministic_replay_passed": True,
        "control_policy_scores": {
            "baseline": 0.8,
            "no_op": 0.2,
            "random": 0.4,
            "always_interrupt": 0.1,
            "always_silence": 0.2,
        },
        "anti_cheat_check_results": {"non_monotonic_timestamps_rejected": True},
        "config_hash": stable_config_hash(config),
        "benchmark_metrics": _metrics(),
        "event_timestamps": [0, 200, 400, 600, 800],
        "bad_control_residue_counts": {"no_op": 2, "random": 1, "always_interrupt": 3, "always_silence": 2},
    }
    data.update(overrides)
    return InteractionBenchmarkManifest(**data)


def test_build_interaction_manifest_records_required_fields():
    manifest = build_interaction_manifest(
        command="python benchmarks/interaction_residue_benchmark.py --mode smoke --seed 42",
        seed=42,
        mode="smoke",
        config={"mode": "smoke", "seed": 42},
        benchmark_metrics=_metrics(),
        number_of_events=5,
        number_of_actions=5,
        number_of_residues=3,
        task_family_names=["turn_taking", "background_delegation"],
        used_test_labels=False,
        deterministic_replay_passed=True,
        control_policy_scores={
            "baseline": 0.8,
            "no_op": 0.2,
            "random": 0.4,
            "always_interrupt": 0.1,
            "always_silence": 0.2,
        },
        anti_cheat_check_results={"non_monotonic_timestamps_rejected": True},
        event_timestamps=[0, 200, 400, 600, 800],
        bad_control_residue_counts={"no_op": 2, "random": 1, "always_interrupt": 3, "always_silence": 2},
    )
    assert manifest.seed == 42
    assert manifest.mode == "smoke"
    assert manifest.number_of_residues == 3
    assert "required benchmark metrics recorded" in manifest.anti_cheat_checks_passed


def test_manifest_rejects_missing_controls():
    manifest = _manifest(control_policy_scores={"baseline": 0.8, "no_op": 0.2})
    with pytest.raises(ValueError, match="control policy scores missing"):
        validate_interaction_manifest(manifest)


def test_manifest_rejects_identical_scores_across_all_policies():
    manifest = _manifest(
        control_policy_scores={
            "baseline": 0.5,
            "no_op": 0.5,
            "random": 0.5,
            "always_interrupt": 0.5,
            "always_silence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="identical scores"):
        validate_interaction_manifest(manifest)


def test_manifest_rejects_all_zero_bad_control_residues():
    manifest = _manifest(bad_control_residue_counts={"no_op": 0, "random": 0, "always_interrupt": 0, "always_silence": 0})
    with pytest.raises(ValueError, match="zero residues"):
        validate_interaction_manifest(manifest)


def test_manifest_rejects_missing_required_metrics():
    metrics = _metrics()
    del metrics["random_control_score"]
    manifest = _manifest(benchmark_metrics=metrics)
    with pytest.raises(ValueError, match="required metrics"):
        validate_interaction_manifest(manifest)


def test_manifest_rejects_test_labels_and_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="test labels"):
        validate_interaction_manifest(_manifest(used_test_labels=True))
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_interaction_manifest(_manifest(event_timestamps=[0, 400, 200]))
