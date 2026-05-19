import json

from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop
from neural_search.meta_meta_meta_controller import MetaMetaMetaController


def test_meta_meta_meta_decision_is_json_serializable(tmp_path):
    result = run_autonomous_loop("smoke", 1301, str(tmp_path / "autonomous.json"))
    output = MetaMetaMetaController().run(result)
    encoded = json.dumps(output.decision.to_dict(), sort_keys=True)
    assert "used_heldout_for_candidate_selection" in encoded
    assert output.decision.used_heldout_for_candidate_selection is False
    assert output.decision.heldout_used_only_for_post_freeze_diagnosis is True
    assert output.next_run_config.recommended_candidate_count >= 1

