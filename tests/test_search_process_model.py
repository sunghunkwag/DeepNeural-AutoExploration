from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop
from neural_search.search_process_model import SearchProcessModel


def test_search_process_model_computes_metrics_from_real_autonomous_loop_output(tmp_path):
    result = run_autonomous_loop("smoke", 1201, str(tmp_path / "autonomous.json"))
    diagnosis = SearchProcessModel().diagnose(result)
    data = diagnosis.to_dict()
    assert data["depth_usefulness_by_level"]
    assert data["mutation_success_rate_by_method"]
    assert data["rollback_rate_by_method"]
    assert 0.0 <= diagnosis.probe_actionability_score <= 1.0
    assert 0.0 <= diagnosis.task_family_balance <= 1.0
    assert diagnosis.used_heldout_for_candidate_selection is False
    assert diagnosis.heldout_used_only_for_post_freeze_diagnosis is True

