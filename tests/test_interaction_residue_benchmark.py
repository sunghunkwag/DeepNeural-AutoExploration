import json

import pytest

from benchmarks.interaction_residue_benchmark import (
    TASK_BACKGROUND,
    TASK_PROACTIVE_VISUAL,
    TASK_TURN_TAKING,
    assert_no_expected_actions,
    build_benchmark_tasks,
    evaluate_policy_on_tasks,
    filter_tasks_by_family,
    run,
)
from interaction_policies import AblatedNoBackgroundPolicy, AblatedNoVisualPolicy, AlwaysInterruptPolicy, BaselineInteractionPolicy, NoOpInteractionPolicy
from interaction_residue_layer import ACTION_LISTEN, MicroTurnEvent


def test_no_op_policy_scores_worse_than_baseline_on_proactive_visual_tasks():
    tasks = filter_tasks_by_family(build_benchmark_tasks(42, mode="smoke"), TASK_PROACTIVE_VISUAL)
    baseline = evaluate_policy_on_tasks(BaselineInteractionPolicy(), tasks, seed=42, policy_name="baseline")
    no_op = evaluate_policy_on_tasks(NoOpInteractionPolicy(), tasks, seed=42, policy_name="no_op")
    assert baseline["metrics"]["missed_visual_change_count"] == 0
    assert no_op["metrics"]["missed_visual_change_count"] > baseline["metrics"]["missed_visual_change_count"]
    assert no_op["metrics"]["interaction_quality"] < baseline["metrics"]["interaction_quality"]


def test_always_interrupt_scores_worse_than_baseline_on_turn_taking_tasks():
    tasks = filter_tasks_by_family(build_benchmark_tasks(42, mode="smoke"), TASK_TURN_TAKING)
    baseline = evaluate_policy_on_tasks(BaselineInteractionPolicy(), tasks, seed=42, policy_name="baseline")
    interrupt = evaluate_policy_on_tasks(AlwaysInterruptPolicy(), tasks, seed=42, policy_name="always_interrupt")
    assert baseline["metrics"]["bad_interrupt_count"] == 0
    assert interrupt["metrics"]["bad_interrupt_count"] > baseline["metrics"]["bad_interrupt_count"]
    assert interrupt["metrics"]["interaction_quality"] < baseline["metrics"]["interaction_quality"]


def test_no_visual_ablation_misses_visual_events():
    tasks = filter_tasks_by_family(build_benchmark_tasks(7, mode="smoke"), TASK_PROACTIVE_VISUAL)
    no_visual = evaluate_policy_on_tasks(AblatedNoVisualPolicy(), tasks, seed=7, policy_name="ablated_no_visual")
    assert no_visual["metrics"]["missed_visual_change_count"] == len(tasks)
    assert no_visual["metrics"]["residue_count"] >= len(tasks)


def test_no_background_ablation_fails_background_delivery_tasks():
    tasks = filter_tasks_by_family(build_benchmark_tasks(7, mode="smoke"), TASK_BACKGROUND)
    baseline = evaluate_policy_on_tasks(BaselineInteractionPolicy(), tasks, seed=7, policy_name="baseline")
    no_background = evaluate_policy_on_tasks(AblatedNoBackgroundPolicy(), tasks, seed=7, policy_name="ablated_no_background")
    assert baseline["metrics"]["background_delivery_rate"] == 1.0
    assert no_background["metrics"]["background_delivery_rate"] < baseline["metrics"]["background_delivery_rate"]
    assert no_background["metrics"]["late_background_result_count"] > 0


def test_benchmark_smoke_mode_writes_json_and_manifest_files(tmp_path):
    output = tmp_path / "interaction_residue_smoke_seed42.json"
    result = run("smoke", 42, str(output))
    manifest_path = tmp_path / "interaction_residue_smoke_seed42.manifest.json"

    saved = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["metrics"]["deterministic_replay_passed"] is True
    assert saved["metrics"]["residue_count"] == 0
    assert saved["control_policy_scores"]["no_op"] < saved["control_policy_scores"]["baseline"]
    assert manifest["seed"] == 42
    assert manifest["mode"] == "smoke"
    assert manifest["number_of_events"] == result["per_seed"][0]["number_of_events"]


def test_benchmark_rejects_expected_action_labels_in_events():
    with pytest.raises(ValueError, match="expected_action"):
        assert_no_expected_actions([MicroTurnEvent(0, expected_action=ACTION_LISTEN)])
