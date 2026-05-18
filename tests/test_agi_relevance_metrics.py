from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop
from neural_search.agi_relevance_metrics import compute_agi_relevance_metrics


def test_agi_relevance_metrics_are_computed_from_benchmark_output(tmp_path):
    result = run_autonomous_loop("smoke", 7, str(tmp_path / "autonomous.json"))
    metrics = compute_agi_relevance_metrics(result)
    required = {
        "cross_domain_transfer_score",
        "new_task_adaptation_speed",
        "representation_reuse_score",
        "failure_recovery_score",
        "search_space_expansion_quality",
        "hidden_validation_robustness",
        "heldout_generalization_delta",
        "architecture_diversity_score",
        "mutation_policy_improvement_over_random",
        "cumulative_autonomous_improvement_score",
    }
    assert required <= set(metrics)
    assert metrics["cross_domain_transfer_score"] == result["metrics"]["cross_domain_transfer_score"]
    assert -1.0 <= metrics["cumulative_autonomous_improvement_score"] <= 2.0
