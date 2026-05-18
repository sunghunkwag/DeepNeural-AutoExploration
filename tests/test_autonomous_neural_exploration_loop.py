import json
from pathlib import Path

from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop


def test_autonomous_loop_writes_result_and_manifest(tmp_path):
    out = tmp_path / "autonomous.json"
    result = run_autonomous_loop("smoke", 5, str(out))
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert out.exists()
    assert manifest_path.exists()
    assert manifest["seed"] == 5
    assert manifest["base_genome"]
    assert manifest["candidate_genomes"]
    assert manifest["mutation_policy"]
    assert "failure_residues" in manifest
    assert "representation_probe_results" in manifest
    assert "gradient_probe_results" in manifest
    assert manifest["heldout_test_metrics_after_freeze"]["heldout_test_evaluated_after_freeze"] is True
    assert result["freeze_event"]["heldout_test_evaluated_before_freeze"] is False


def test_autonomous_loop_rolls_back_rejections_and_accepts_only_improvements(tmp_path):
    result = run_autonomous_loop("smoke", 5, str(tmp_path / "autonomous.json"))
    rejected = result["rejected_candidates"]
    assert result["rollback_count"] == len(rejected)
    assert all(item["rollback"] is True for item in rejected)
    for accepted in result["accepted_candidates"]:
        assert accepted["validation_improvement"] > 0.0 or accepted["hidden_validation_improvement"] > 0.0
        assert accepted["used_heldout_labels"] is False


def test_autonomous_loop_is_deterministic_under_seed(tmp_path):
    first = run_autonomous_loop("smoke", 6, str(tmp_path / "first.json"))
    second = run_autonomous_loop("smoke", 6, str(tmp_path / "second.json"))
    assert first["final_genome"]["config_hash"] == second["final_genome"]["config_hash"]
    assert [item["candidate_id"] for item in first["architecture_decisions"]] == [
        item["candidate_id"] for item in second["architecture_decisions"]
    ]
    assert first["rollback_count"] == second["rollback_count"]
