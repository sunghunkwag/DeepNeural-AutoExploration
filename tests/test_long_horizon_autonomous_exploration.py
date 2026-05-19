import json
from pathlib import Path

import benchmarks.long_horizon_autonomous_exploration as benchmark


def _fake_loop(mode, seed, output, next_run_config=None):
    applied = None
    if next_run_config is not None:
        applied = json.loads(Path(next_run_config).read_text(encoding="utf-8"))
    improved = applied is not None
    decision = {
        "decision_id": f"arch-{seed}",
        "deep_search_decision_id": f"deep-{seed}",
        "candidate_id": f"cand-{seed}",
        "mutation_method": "widen_hidden_dim",
        "accepted": improved,
        "rollback": not improved,
        "selection_score_delta": 0.2 if improved else -0.1,
        "validation_improvement": 0.2,
        "hidden_validation_improvement": 0.1 if improved else -0.1,
        "validation_metrics": {"loss": 0.4},
        "hidden_validation_metrics": {"loss": 0.3 if improved else 0.7},
        "used_heldout_labels": False,
    }
    result = {
        "benchmark": "autonomous_neural_exploration",
        "mode": mode,
        "seed": seed,
        "config": {"generations": 1, "candidate_count": 2},
        "applied_next_run_config": applied,
        "architecture_decisions": [decision],
        "deep_search_decisions": [{"decision_id": f"deep-{seed}", "selected_depth_level": 2, "rollback_risk": 0.2}],
        "failure_residues": [{"failure_type": "object-binding failure", "task_id": "object_state_transition-a"}] if improved else [
            {"failure_type": "object-binding failure", "task_id": "object_state_transition-a"},
            {"failure_type": "object-binding failure", "task_id": "object_state_transition-b"},
        ],
        "failure_residues_before": [{"failure_type": "object-binding failure"}, {"failure_type": "object-binding failure"}],
        "failure_residues_after_accepted": [] if improved else [{"failure_type": "object-binding failure"}],
        "selection_score_components": [{"components": {"validation_loss": 0.2, "hidden_validation_loss": 0.3 if improved else 0.8}}],
        "search_space_expansions": [],
        "base_genome": {"hidden_dim": 10, "residual_blocks": 1},
        "final_genome": {"hidden_dim": 11 if improved else 10, "residual_blocks": 1},
        "base_metrics": {"heldout_test_loss_by_family": {"object_state_transition": 0.9}, "heldout_test_loss": 0.9},
        "final_metrics": {
            "validation_loss": 0.2,
            "hidden_validation_loss": 0.3 if improved else 0.8,
            "hidden_validation_loss_by_family": {"object_state_transition": 0.3 if improved else 0.8},
            "heldout_test_loss_by_family": {"object_state_transition": 0.5 if improved else 0.9},
            "heldout_test_loss": 0.5 if improved else 0.9,
            "cross_domain_transfer_score": 0.6,
            "heldout_test_evaluated_after_freeze": True,
        },
        "heldout_test_metrics_after_freeze": {
            "heldout_test_loss_by_family": {"object_state_transition": 0.5 if improved else 0.9},
            "heldout_test_loss": 0.5 if improved else 0.9,
            "heldout_test_evaluated_after_freeze": True,
        },
        "metrics": {"hidden_validation_robustness": 1.0 if improved else 0.0, "residue_resolution_rate": 1.0 if improved else 0.0},
        "agi_relevance_metrics": {
            "hidden_validation_robustness": 1.0 if improved else 0.0,
            "residue_resolution_rate": 1.0 if improved else 0.0,
            "cumulative_autonomous_improvement_score": 0.8 if improved else 0.2,
        },
        "final_bottleneck_diagnosis": {"primary_failure_type": "object-binding failure", "proposed_module_family": "object_binding_adapter"},
        "recommended_next_experiment": {"action": "continue_deep_search", "used_heldout_labels": False},
        "audit_trail": {"heldout_test_evaluated_after_freeze": True, "used_heldout_labels_for_selection": False},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def test_long_horizon_benchmark_runs_repeated_iterations_and_persists_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "run_autonomous_loop", _fake_loop)
    out = tmp_path / "long.json"
    result = benchmark.run("full", 42, 3, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    memory = json.loads(Path(result["meta_config_memory_path"]).read_text(encoding="utf-8"))
    assert result["run_count"] == 3
    assert len(result["run_outputs"]) == 3
    assert len(result["attribution_reports"]) == 2
    assert len(memory["records"]) == 2
    assert any(json.loads(Path(path).read_text(encoding="utf-8"))["applied_next_run_config"] is not None for path in result["run_outputs"][1:])
    assert "trend_analysis" in result
    assert manifest["run_count"] == 3
    assert out.exists()
