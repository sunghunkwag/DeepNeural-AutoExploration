import copy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.recursive_self_improvement_benchmark import (
    _build_memory,
    _evaluate_tasks,
    _initial_state,
    run,
)
from learned_task_encoder import LearnedTaskEncoder
from operator_mutation import OperatorMutation, OperatorMutationController, allowed_operator_mutations
from task_suite import ProceduralTaskSuite, assert_disjoint_task_ids, function_task_variance


def test_mixed_task_families_expose_support_query_and_ood_metadata():
    suite = ProceduralTaskSuite(seed=101)
    train = suite.sample_mixed_tasks("train", 9, ood=False)
    test = suite.sample_mixed_tasks("test", 9, ood=True)
    assert_disjoint_task_ids(train, test)
    families = {task.family for task in train + test}
    assert {"nonstationary_sequence", "delayed_dependency_sequence", "memory_association", "latent_navigation"}.issubset(families)
    for task in train + test:
        assert task.task_id
        assert task.support_x.shape[0] == task.support_y.shape[0]
        assert task.query_x.shape[0] == task.query_y.shape[0]
        assert function_task_variance(task) > 1e-6
    assert any(task.ood for task in test)


def test_allowed_mutations_cover_required_executable_types():
    names = {mutation.name for mutation in allowed_operator_mutations(_initial_state(3))}
    required = {
        "inner_lr_down",
        "inner_steps_plus",
        "memory_k_plus",
        "task_encoder_toggle",
        "memory_conditioning_toggle",
        "world_model_controller_toggle",
        "planner_horizon_plus",
        "intrinsic_weight_up",
        "exploration_noise_decay",
        "width_multiplier_small",
    }
    assert required.issubset(names)


def test_accepted_mutation_changes_actual_runtime_loss_and_rollback_is_exact():
    seed = 102
    suite = ProceduralTaskSuite(seed=seed)
    train = suite.sample_mixed_tasks("train", 4)
    val = suite.sample_mixed_tasks("validation", 2)
    state = _initial_state(seed)
    encoder = LearnedTaskEncoder(latent_dim=8, seed=seed)
    memory = _build_memory(train, state, encoder)

    before_losses, _, _ = _evaluate_tasks(val, state, encoder, memory, controller_kind="none")
    mutation = OperatorMutation("runtime_lr_change", {"inner_lr": float(state["inner_lr"]) * 0.25}, "hyperparameter")
    ctrl = OperatorMutationController(state, improvement_threshold=-1e9, min_consistent_wins=0, random_seed=seed)

    def evaluator(candidate_state, tasks):
        losses, _, _ = _evaluate_tasks(tasks, candidate_state, encoder, memory, controller_kind="none")
        return losses

    decision = ctrl.evaluate(mutation, val, evaluator, split="validation")
    after_losses, _, _ = _evaluate_tasks(val, ctrl.state, encoder, memory, controller_kind="none")
    assert decision["accepted"]
    assert ctrl.state["inner_lr"] != state["inner_lr"]
    assert before_losses != after_losses

    snapshot = copy.deepcopy(ctrl.state)
    bad = OperatorMutation("bad_lr", {"inner_lr": 99.0}, "hyperparameter")
    rejecting = OperatorMutationController(snapshot, improvement_threshold=1e9, min_consistent_wins=99, random_seed=seed)
    rejected = rejecting.evaluate(bad, val, evaluator, split="validation")
    assert not rejected["accepted"]
    assert rejected["rollback"]
    assert rejecting.state == snapshot


def test_recursive_benchmark_writes_json_manifest_and_anticheat(tmp_path):
    out = tmp_path / "recursive_smoke.json"
    result = run("smoke", 31, str(out))
    payload = json.loads(out.read_text())
    manifest_path = Path(payload["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert result["aggregate"]["controller_call_count"]["mean"] > 0
    assert result["aggregate"]["world_model_call_count"]["mean"] > 0
    assert "improvement_velocity" in result["aggregate"]
    assert "operator_reuse_success" in result["aggregate"]
    assert "accepted_mutation_quality" in result["aggregate"]
    assert result["aggregate"]["accepted_operator_count"]["mean"] >= 4
    assert payload["task_ids"]["train"] and payload["task_ids"]["validation"] and payload["task_ids"]["test"]
    assert manifest["seed_list"] == [31]
    assert manifest["accepted_mutations"] or manifest["rejected_mutations"]
    assert manifest["config_hash"]
    assert manifest["metric_summary"]["aggregate"]
    assert "no query_y stored in memory" in payload["anti_cheat_checks_passed"]
    assert "no test task used in mutation acceptance" in manifest["anti_cheat_checks_passed"]
    assert payload["per_seed"][0]["operator_genome"]["accepted_gene_ids"]


def test_recursive_smoke_is_deterministic_for_same_seed(tmp_path):
    first = run("smoke", 32, str(tmp_path / "first.json"))
    second = run("smoke", 32, str(tmp_path / "second.json"))
    for key in ["full_loop_delta_vs_baseline", "controller_prediction_error", "accepted_mutation_count", "rollback_count", "operator_reuse_success"]:
        assert first["aggregate"][key]["mean"] == pytest.approx(second["aggregate"][key]["mean"], rel=0, abs=1e-7)
    assert first["task_ids"] == second["task_ids"]


def test_recursive_benchmark_has_no_hardcoded_result_literals_or_family_branching():
    source = Path("benchmarks/recursive_self_improvement_benchmark.py").read_text()
    forbidden = ["24.53520743052165", "178.31387853622437", "201.10302019119263"]
    assert not any(value in source for value in forbidden)
    assert "if task.family" not in source
