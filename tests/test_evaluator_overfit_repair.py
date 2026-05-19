from neural_search.evaluator_objective_mutator import EvaluatorObjectiveMutator
from neural_search.search_process_model import SearchProcessDiagnosis


def test_evaluator_overfit_repair_changes_objective_weights_expected_direction():
    diagnosis = SearchProcessDiagnosis(
        evaluator_overfit_risk=0.6,
        rollback_rate_by_method={"widen_hidden_dim": 1.0},
        task_family_balance=0.4,
    )
    plan = EvaluatorObjectiveMutator().mutate(
        diagnosis,
        [
            {
                "components": {
                    "validation_loss": 0.2,
                    "hidden_validation_loss": 0.7,
                    "calibration_hidden_regression_flag": 1.0,
                    "calibration_high_rollback_mismatch_flag": 1.0,
                }
            }
        ],
    )
    evidence = plan.evidence_used

    assert plan.mutated_config["validation_loss_weight"] < plan.base_config["validation_loss_weight"]
    assert plan.mutated_config["hidden_validation_loss_weight"] > plan.base_config["hidden_validation_loss_weight"]
    assert plan.mutated_config["rollback_risk_weight"] > plan.base_config["rollback_risk_weight"]
    assert plan.mutated_config["task_family_regression_penalty_weight"] > plan.base_config["task_family_regression_penalty_weight"]
    assert evidence["evaluator_calibration_report"]["validation_hidden_mismatch_guard_triggered"] is True
    assert "apply_validation_hidden_mismatch_guard" in evidence["evaluator_overfit_repair_actions"]
    assert evidence["objective_weight_change_attribution"]
