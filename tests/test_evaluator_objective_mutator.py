from neural_search.evaluator_objective_mutator import EvaluatorObjectiveMutator
from neural_search.search_process_model import SearchProcessDiagnosis


def test_hidden_validation_mismatch_increases_hidden_validation_weight():
    diagnosis = SearchProcessDiagnosis(evaluator_overfit_risk=0.7)
    plan = EvaluatorObjectiveMutator().mutate(diagnosis)
    assert "hidden-validation underweighting" in plan.selected_mutations
    assert plan.mutated_config["hidden_validation_loss_weight"] > plan.base_config["hidden_validation_loss_weight"]
    assert plan.mutated_config["validation_loss_weight"] < plan.base_config["validation_loss_weight"]


def test_objective_mutator_changes_weights_from_selection_evidence():
    diagnosis = SearchProcessDiagnosis(
        evaluator_overfit_risk=0.3,
        transfer_regression_risk=0.3,
        rollback_rate_by_method={"add_memory_gate": 1.0},
        task_family_balance=0.5,
    )
    plan = EvaluatorObjectiveMutator().mutate(
        diagnosis,
        [{"components": {"module_contribution_score": 0.0, "validation_loss": 0.2, "hidden_validation_loss": 0.7}}],
    )
    assert plan.mutated_config["transfer_score_weight"] > plan.base_config["transfer_score_weight"]
    assert plan.mutated_config["rollback_risk_weight"] > plan.base_config["rollback_risk_weight"]
    assert plan.mutated_config["task_family_regression_penalty_weight"] > plan.base_config["task_family_regression_penalty_weight"]

