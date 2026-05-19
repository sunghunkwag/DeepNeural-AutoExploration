import json
from pathlib import Path

from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop


def test_autonomous_loop_audit_contains_deep_search_decisions_and_scores(tmp_path):
    out = tmp_path / "autonomous.json"
    result = run_autonomous_loop("smoke", 1101, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["deep_search_decisions"]
    assert result["generation_candidate_table"]
    assert result["selection_score_components"]
    assert result["audit_trail"]["deep_search_decisions"]
    assert result["audit_trail"]["heldout_evaluation_timestamp_after_freeze"]
    assert result["freeze_event"]["heldout_test_evaluated_before_freeze"] is False
    assert all(decision["used_heldout_labels"] is False for decision in result["architecture_decisions"])
    assert manifest["deep_search_decisions"]
    assert manifest["research_goal_controller_input"]["recommended_next_experiment"]

