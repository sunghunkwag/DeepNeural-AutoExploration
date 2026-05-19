from neural_search.search_process_model import SearchProcessDiagnosis
from neural_search.search_strategy_mutator import SearchStrategyMutator


def test_repeated_rollback_increases_mutation_penalty():
    diagnosis = SearchProcessDiagnosis(
        rollback_rate_by_method={"widen_hidden_dim": 1.0},
        mutation_success_rate_by_method={"widen_hidden_dim": 0.0},
    )
    plan = SearchStrategyMutator().mutate(diagnosis)
    assert "penalize_repeated_failed_mutation" in plan.selected_mutations
    assert plan.mutated_config["mutation_penalties"]["widen_hidden_dim"] > 0.0


def test_persistent_residue_increases_corresponding_mutation_boost():
    diagnosis = SearchProcessDiagnosis(
        persistent_residue_types=["object-binding failure"],
        mutation_success_rate_by_method={"object_binding_adapter": 0.0},
        depth_usefulness_by_level={"1": -0.1},
    )
    plan = SearchStrategyMutator().mutate(diagnosis)
    assert "boost_mutation_for_persistent_residue" in plan.selected_mutations
    assert plan.mutated_config["mutation_boosts"]["object_binding_adapter"] > 0.0


def test_search_depth_usefulness_changes_strategy_thresholds():
    diagnosis = SearchProcessDiagnosis(
        depth_usefulness_by_level={"2": -0.05},
        persistent_residue_types=["gradient bottleneck"],
        mutation_success_rate_by_method={"add_residual_block": 0.0},
    )
    plan = SearchStrategyMutator().mutate(diagnosis)
    assert "increase_depth_threshold" in plan.selected_mutations
    assert plan.mutated_config["depth_thresholds"]["level2"] > plan.base_config["depth_thresholds"]["level2"]

