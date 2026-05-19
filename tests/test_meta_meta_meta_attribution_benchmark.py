import json
from pathlib import Path

import benchmarks.meta_meta_meta_search_benchmark as benchmark


def _fake_autonomous_loop(mode, seed, output, next_run_config=None):
    applied = None
    if next_run_config is not None:
        applied = json.loads(Path(next_run_config).read_text(encoding="utf-8"))
    improved = applied is not None
    decisions = [
        {
            "decision_id": f"arch-{seed}",
            "deep_search_decision_id": f"deep-{seed}",
            "mutation_method": "widen_hidden_dim",
            "accepted": improved,
            "rollback": not improved,
            "selection_score_delta": 0.1 if improved else -0.2,
            "validation_improvement": 0.2,
            "hidden_validation_improvement": 0.2 if improved else -0.1,
            "used_heldout_labels": False,
        }
    ]
    residues = (
        [{"failure_type": "object-binding failure", "task_id": "object_state_transition-a"}]
        if improved
        else [
            {"failure_type": "object-binding failure", "task_id": "object_state_transition-a"},
            {"failure_type": "object-binding failure", "task_id": "object_state_transition-b"},
        ]
    )
    result = {
        "benchmark": "autonomous_neural_exploration",
        "mode": mode,
        "seed": seed,
        "config": {"generations": 1, "candidate_count": 3},
        "applied_next_run_config": applied,
        "architecture_decisions": decisions,
        "deep_search_decisions": [{"decision_id": f"deep-{seed}", "selected_depth_level": 2, "rollback_risk": 0.2}],
        "failure_residues": residues,
        "failure_residues_before": [
            {"failure_type": "object-binding failure"},
            {"failure_type": "object-binding failure"},
        ],
        "failure_residues_after_accepted": [] if improved else [{"failure_type": "object-binding failure"}],
        "selection_score_components": [
            {"components": {"validation_loss": 0.2, "hidden_validation_loss": 0.25 if improved else 0.8}}
        ],
        "search_space_expansions": [],
        "base_genome": {"hidden_dim": 10, "residual_blocks": 1},
        "final_genome": {"hidden_dim": 11 if improved else 10, "residual_blocks": 1},
        "base_metrics": {
            "heldout_test_loss_by_family": {"object_state_transition": 0.9},
            "heldout_test_loss": 0.9,
        },
        "final_metrics": {
            "validation_loss": 0.2,
            "hidden_validation_loss": 0.25 if improved else 0.8,
            "hidden_validation_loss_by_family": {"object_state_transition": 0.25 if improved else 0.8},
            "heldout_test_loss_by_family": {"object_state_transition": 0.4 if improved else 0.9},
            "heldout_test_loss": 0.4 if improved else 0.9,
            "cross_domain_transfer_score": 0.6 if improved else 0.2,
            "heldout_test_evaluated_after_freeze": True,
        },
        "heldout_test_metrics_after_freeze": {
            "heldout_test_loss_by_family": {"object_state_transition": 0.4 if improved else 0.9},
            "heldout_test_loss": 0.4 if improved else 0.9,
            "heldout_test_evaluated_after_freeze": True,
        },
        "metrics": {
            "hidden_validation_robustness": 1.0 if improved else 0.0,
            "residue_resolution_rate": 1.0 if improved else 0.0,
            "cumulative_autonomous_improvement_score": 0.8 if improved else 0.1,
            "rollback_count": 0.0 if improved else 1.0,
        },
        "agi_relevance_metrics": {
            "hidden_validation_robustness": 1.0 if improved else 0.0,
            "residue_resolution_rate": 1.0 if improved else 0.0,
            "cumulative_autonomous_improvement_score": 0.8 if improved else 0.1,
        },
        "audit_trail": {
            "heldout_test_evaluated_after_freeze": True,
            "used_heldout_labels_for_selection": False,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def test_meta_meta_meta_benchmark_writes_attribution_quality_counterfactual_and_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "run_autonomous_loop", _fake_autonomous_loop)
    out = tmp_path / "meta_meta_meta.json"
    result = benchmark.run("smoke", 1801, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert out.exists()
    assert Path(result["meta_config_memory_path"]).exists()
    assert result["config_change_attribution_report"]["changes"]
    assert result["meta_decision_quality_report"]["recommendation"] in {
        "trust_previous_config",
        "revise_previous_config",
        "revert_previous_config",
    }
    assert result["counterfactual_config_report"]["all_reports_are_estimates"] is True
    assert result["counterfactual_config_report"]["claims_proof"] is False
    assert result["heldout_test_used_for_candidate_selection"] is False
    assert result["heldout_test_used_only_for_post_freeze_process_diagnosis"] is True
    assert "process_improvement_components" in manifest
    assert manifest["meta_config_memory_path"] == result["meta_config_memory_path"]
