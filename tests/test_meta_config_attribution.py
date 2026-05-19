from neural_search.curriculum_strategy_mutator import CurriculumStrategyConfig
from neural_search.evaluator_objective_mutator import EvaluatorObjectiveConfig
from neural_search.meta_config_attribution import MetaConfigAttribution
from neural_search.search_process_model import SearchProcessDiagnosis
from neural_search.search_strategy_mutator import SearchStrategyConfig


def _runs_and_diagnoses():
    base_search = SearchStrategyConfig().to_dict()
    next_search = dict(base_search)
    next_search["hidden_validation_weight"] = base_search["hidden_validation_weight"] + 0.10
    next_search["mutation_penalties"] = {"widen_hidden_dim": 0.40}
    next_search["mutation_boosts"] = {"object_binding_adapter": 0.20}
    base_eval = EvaluatorObjectiveConfig().to_dict()
    next_eval = dict(base_eval)
    next_eval["hidden_validation_loss_weight"] = base_eval["hidden_validation_loss_weight"] + 0.10
    base_curriculum = CurriculumStrategyConfig().to_dict()
    next_curriculum = dict(base_curriculum)
    next_curriculum["family_sampling_weights"] = dict(base_curriculum["family_sampling_weights"])
    next_curriculum["family_sampling_weights"]["object_state_transition"] = 1.50
    next_config = {
        "search_strategy_config": next_search,
        "evaluator_objective_config": next_eval,
        "curriculum_strategy_config": next_curriculum,
        "recommended_mode": "smoke",
        "recommended_seed_policy": "same_seed",
        "recommended_generations": 2,
        "recommended_candidate_count": 4,
        "recommended_focus_failure_types": ["object-binding failure"],
        "recommended_focus_task_families": ["object_state_transition"],
    }
    first = {
        "benchmark": "autonomous_neural_exploration",
        "mode": "smoke",
        "seed": 1,
        "config": {"generations": 1, "candidate_count": 3},
        "search_strategy_mutation_plan": {"base_config": base_search},
        "evaluator_objective_mutation_plan": {"base_config": base_eval},
        "curriculum_mutation_plan": {"base_config": base_curriculum},
        "next_run_config_recommendation": next_config,
        "agi_relevance_metrics": {"hidden_validation_robustness": 0.30, "residue_resolution_rate": 0.20},
        "architecture_decisions": [{"accepted": False, "rollback": True, "used_heldout_labels": False}],
        "failure_residues": [{"failure_type": "object-binding failure"}, {"failure_type": "object-binding failure"}],
    }
    second = {
        "benchmark": "autonomous_neural_exploration",
        "mode": "smoke",
        "seed": 1,
        "applied_next_run_config": next_config,
        "agi_relevance_metrics": {"hidden_validation_robustness": 0.80, "residue_resolution_rate": 0.70},
        "architecture_decisions": [{"accepted": True, "rollback": False, "used_heldout_labels": False}],
        "failure_residues": [{"failure_type": "object-binding failure"}],
    }
    first_diag = SearchProcessDiagnosis(
        rollback_rate_by_method={"widen_hidden_dim": 0.90},
        validation_hidden_mismatch_by_method={"widen_hidden_dim": 0.50},
        residue_resolution_rate_by_failure_type={"object-binding failure": 0.10},
        persistent_residue_types=["object-binding failure"],
        evaluator_overfit_risk=0.70,
        task_family_balance=0.40,
        transfer_regression_risk=0.30,
        architecture_complexity_growth=0.40,
    )
    second_diag = SearchProcessDiagnosis(
        rollback_rate_by_method={"widen_hidden_dim": 0.20},
        validation_hidden_mismatch_by_method={"widen_hidden_dim": 0.10},
        residue_resolution_rate_by_failure_type={"object-binding failure": 0.80},
        persistent_residue_types=[],
        evaluator_overfit_risk=0.20,
        task_family_balance=0.75,
        transfer_regression_risk=0.10,
        architecture_complexity_growth=0.20,
    )
    return first, second, next_config, first_diag, second_diag


def test_config_attribution_detects_strategy_evaluator_and_curriculum_changes():
    first, second, next_config, first_diag, second_diag = _runs_and_diagnoses()
    report = MetaConfigAttribution().attribute(
        first,
        second,
        first_next_run_config=next_config,
        second_applied_next_run_config=next_config,
        first_diagnosis=first_diag,
        second_diagnosis=second_diag,
    )
    paths = {change.config_path for change in report.changes}
    assert "search_strategy_config.mutation_penalties.widen_hidden_dim" in paths
    assert "search_strategy_config.mutation_boosts.object_binding_adapter" in paths
    assert "evaluator_objective_config.hidden_validation_loss_weight" in paths
    assert "curriculum_strategy_config.family_sampling_weights.object_state_transition" in paths
    assert report.used_heldout_for_candidate_selection is False
    assert report.heldout_used_only_for_post_freeze_diagnosis is True


def test_hidden_weight_penalty_and_boost_are_evaluated_against_process_changes():
    first, second, next_config, first_diag, second_diag = _runs_and_diagnoses()
    report = MetaConfigAttribution().attribute(
        first,
        second,
        first_next_run_config=next_config,
        second_applied_next_run_config=next_config,
        first_diagnosis=first_diag,
        second_diagnosis=second_diag,
    )
    by_path = {change.config_path: change for change in report.changes}
    hidden = by_path["search_strategy_config.hidden_validation_weight"]
    assert hidden.helped is True
    assert hidden.observed_effect["validation_hidden_mismatch_delta"] < 0.0
    penalty = by_path["search_strategy_config.mutation_penalties.widen_hidden_dim"]
    assert penalty.helped is True
    assert penalty.observed_effect["rollback_rate_delta"] < 0.0
    boost = by_path["search_strategy_config.mutation_boosts.object_binding_adapter"]
    assert boost.helped is True
    assert boost.observed_effect["residue_resolution_delta"] > 0.0
