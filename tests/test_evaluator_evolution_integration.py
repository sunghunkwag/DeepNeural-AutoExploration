from neural_search.evaluator_evolution_adapter import EvaluatorEvolutionEvidenceAdapter


def test_evaluator_evolution_receives_mismatch_evidence():
    adapter = EvaluatorEvolutionEvidenceAdapter()
    run = {
        "search_process_diagnosis": {
            "validation_hidden_mismatch_by_method": {"widen_hidden_dim": 0.2},
            "transfer_regression_risk": 0.1,
            "architecture_complexity_growth": 0.3,
        },
        "architecture_decisions": [
            {
                "accepted": False,
                "rejection_reason": "hidden_validation_regression",
                "validation_improvement": 0.2,
                "hidden_validation_improvement": -0.3,
                "validation_metrics": {"loss": 0.4},
                "mutation_method": "widen_hidden_dim",
                "task_family_regressions": {"sequence_prediction": 0.2},
            }
        ],
        "evaluator_objective_mutation_plan": {"selected_mutations": ["hidden-validation underweighting"], "mutated_config": {}},
    }
    adapter.observe_run(run)
    summary = adapter.summary()
    assert summary["mismatch_patterns"]["validation_hidden_mismatch"] == 1
    assert summary["mismatch_patterns"]["hidden_validation_regression"] == 1
    assert "hidden_validation_loss_weight" in summary["proposed_weight_adjustments"]
