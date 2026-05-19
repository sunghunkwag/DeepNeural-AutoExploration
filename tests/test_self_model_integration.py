from neural_search.self_model_adapter import LongHorizonSelfModelAdapter


def test_self_model_adapter_records_prediction_actual_pairs():
    adapter = LongHorizonSelfModelAdapter()
    run = {
        "failure_residues": [{"failure_type": "gradient bottleneck", "task_id": "sequence_prediction-1"}],
        "architecture_decisions": [
            {
                "candidate_id": "c1",
                "mutation_method": "widen_hidden_dim",
                "accepted": True,
                "rollback": False,
                "validation_improvement": 0.2,
                "hidden_validation_improvement": 0.1,
                "selection_score_delta": 0.3,
                "used_heldout_labels": False,
            }
        ],
    }
    adapter.observe_run(0, run, {"helped_changes": ["x"]}, {"quality_scores": {"overall_meta_decision_quality_score": 0.8}})
    summary = adapter.summary()
    assert summary["prediction_count"] == 1
    assert summary["correct_prediction_count"] == 1
    assert summary["self_model_prediction_accuracy"] == 1.0
