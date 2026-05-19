from benchmarks.autonomous_neural_exploration_loop import _apply_strategy_config_to_methods
from benchmarks.long_horizon_autonomous_exploration import _apply_rollback_repair_to_next_config
from neural_search.next_run_config import NextRunConfig
from neural_search.search_process_model import SearchProcessDiagnosis
from neural_search.search_strategy_mutator import SearchStrategyMutator


def test_rollback_risk_budget_is_enforced_in_strategy_mutator():
    diagnosis = SearchProcessDiagnosis(
        rollback_rate_by_method={"widen_hidden_dim": 1.0},
        persistent_residue_types=["architecture-representation bottleneck"],
    )
    plan = SearchStrategyMutator().mutate(diagnosis)

    assert "enforce_rollback_risk_budget" in plan.selected_mutations
    assert plan.mutated_config["mutation_penalties"]["widen_hidden_dim"] > 0
    assert plan.evidence_used["rollback_budget_exceeded"] is True


def test_next_run_config_reduces_search_when_rollback_budget_exceeded():
    cfg = NextRunConfig(recommended_generations=3, recommended_candidate_count=4)
    result = {
        "architecture_decisions": [
            {
                "mutation_method": "widen_hidden_dim",
                "accepted": False,
                "rollback": True,
                "validation_improvement": 0.2,
                "hidden_validation_improvement": -0.2,
                "task_family_regressions": {"sequence_prediction": 0.1},
            }
            for _ in range(4)
        ]
    }
    repaired, report = _apply_rollback_repair_to_next_config(cfg, result, SearchProcessDiagnosis())

    assert report["rollback_budget_exceeded"] is True
    assert repaired.recommended_candidate_count == 3
    assert repaired.recommended_generations == 2
    assert repaired.search_strategy_config.mutation_penalties["widen_hidden_dim"] > 0
    assert repaired.evaluator_objective_config.hidden_validation_loss_weight > cfg.evaluator_objective_config.hidden_validation_loss_weight
    assert "reject_validation_only_hidden_regression_pattern" in report["rollback_repair_actions"]


def test_strategy_config_penalties_expand_to_safer_alternative_methods():
    cfg = NextRunConfig().search_strategy_config
    cfg = cfg.from_dict(
        {
            **cfg.to_dict(),
            "mutation_penalties": {"widen_hidden_dim": 1.0, "add_residual_block": 1.0},
            "mutation_boosts": {"wider_residual_stack": 0.8},
        }
    )

    methods = _apply_strategy_config_to_methods(["widen_hidden_dim"], cfg)

    assert methods[0] == "wider_residual_stack"
    assert "widen_hidden_dim" not in methods[:1]


def test_all_penalized_strategy_uses_probation_order_instead_of_repeating_deep_choice():
    cfg = NextRunConfig().search_strategy_config
    cfg = cfg.from_dict(
        {
            **cfg.to_dict(),
            "mutation_penalties": {
                "add_residual_block": 1.0,
                "remove_residual_block": 1.0,
                "widen_hidden_dim": 1.0,
                "narrow_hidden_dim": 1.0,
                "add_memory_gate": 1.0,
                "add_causal_bottleneck": 1.0,
                "object_binding_adapter": 1.0,
                "sequence_state_adapter": 1.0,
                "wider_residual_stack": 1.0,
                "repair_bottleneck": 1.0,
            },
            "mutation_boosts": {},
        }
    )

    methods = _apply_strategy_config_to_methods(["widen_hidden_dim"], cfg)

    assert methods[0] == "repair_bottleneck"
    assert methods.index("widen_hidden_dim") > 0
