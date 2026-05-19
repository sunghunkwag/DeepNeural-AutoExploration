from neural_search.curriculum_strategy_mutator import CurriculumStrategyConfig, CurriculumStrategyMutator
from neural_search.search_process_model import SearchProcessDiagnosis


def test_curriculum_increases_hard_mode_only_when_performance_supports_it():
    base = CurriculumStrategyConfig(hard_mode_ratio=0.5)
    weak = SearchProcessDiagnosis(
        evaluator_overfit_risk=0.8,
        transfer_regression_risk=0.5,
        task_family_balance=0.9,
        mutation_success_rate_by_method={"widen_hidden_dim": 1.0},
    )
    weak_plan = CurriculumStrategyMutator().mutate(weak, base)
    assert weak_plan.mutated_config["hard_mode_ratio"] <= base.hard_mode_ratio

    strong = SearchProcessDiagnosis(
        evaluator_overfit_risk=0.1,
        transfer_regression_risk=0.1,
        task_family_balance=0.9,
        mutation_success_rate_by_method={"widen_hidden_dim": 1.0},
    )
    strong_plan = CurriculumStrategyMutator().mutate(strong, base)
    assert strong_plan.mutated_config["hard_mode_ratio"] > base.hard_mode_ratio
    assert "increase_hard_mode_focus" in strong_plan.selected_mutations


def test_curriculum_detects_task_family_imbalance():
    diagnosis = SearchProcessDiagnosis(
        task_family_hidden_losses={
            "function_regression": 0.1,
            "sequence_prediction": 0.2,
            "object_state_transition": 1.2,
        },
        task_family_balance=0.4,
    )
    plan = CurriculumStrategyMutator().mutate(diagnosis)
    assert "detect_task_family_imbalance" in plan.selected_mutations
    assert plan.mutated_config["family_sampling_weights"]["object_state_transition"] > 1.0

