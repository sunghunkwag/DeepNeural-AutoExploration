import json
from pathlib import Path

from neural_search.long_horizon_failure_diagnosis import diagnose_long_horizon_failure


def test_failure_diagnosis_identifies_rollback_and_overfit_causes(tmp_path):
    run0 = _run_payload(seed=42, accepted=True, overfit=0.0, residue_count=1)
    run1 = _run_payload(seed=43, accepted=False, overfit=0.6, residue_count=2)
    p0 = tmp_path / "run0.json"
    p1 = tmp_path / "run1.json"
    p0.write_text(json.dumps(run0), encoding="utf-8")
    p1.write_text(json.dumps(run1), encoding="utf-8")
    long_result = {
        "run_outputs": [str(p0), str(p1)],
        "attribution_reports": [
            {
                "changes": [
                    {
                        "config_path": "search_strategy_config.mutation_penalties.widen_hidden_dim",
                        "harmed": True,
                        "observed_effect": {"rollback_rate_delta": 0.5},
                    }
                ],
                "harmed_changes": ["search_strategy_config.mutation_penalties.widen_hidden_dim"],
            }
        ],
        "meta_decisions": [
            {"reused_successful_config_patterns": ["search_strategy_config.mutation_penalties.widen_hidden_dim"]}
        ],
    }
    external = {
        "validation_delta": 0.1,
        "hidden_validation_delta": 0.1,
        "heldout_after_freeze_delta": -0.02,
        "external_failure_reason": "heldout_after_freeze_regressed",
    }

    report = diagnose_long_horizon_failure(long_result, external).to_dict()

    assert report["rollback_root_causes"]["rollback_causes_by_method"]["widen_hidden_dim"] == 1
    assert report["rollback_root_causes"]["rollback_causes_by_config_change"]
    assert report["evaluator_overfit_causes"]["end_evaluator_overfit_risk"] > report["evaluator_overfit_causes"]["start_evaluator_overfit_risk"]
    assert report["external_adapter_generalization_failure"]["failed_generalization"] is True
    assert any("rollback" in repair for repair in report["recommended_repairs"])


def _run_payload(seed, accepted, overfit, residue_count):
    return {
        "benchmark": "autonomous_neural_exploration",
        "seed": seed,
        "architecture_decisions": [
            {
                "mutation_method": "widen_hidden_dim",
                "accepted": accepted,
                "rollback": not accepted,
                "validation_improvement": 0.2,
                "hidden_validation_improvement": -0.1 if not accepted else 0.1,
                "task_family_regressions": {"object_state_transition": 0.2} if not accepted else {},
            }
        ],
        "failure_residues": [
            {
                "failure_type": "object-binding failure",
                "task_id": f"object_state_transition-{index}",
                "proposed_operator_or_module_family": "object_binding_adapter",
            }
            for index in range(residue_count)
        ],
        "search_process_diagnosis": {"evaluator_overfit_risk": overfit},
        "evaluator_objective_mutation_plan": {
            "mutated_config": {"validation_loss_weight": 0.3 + overfit, "hidden_validation_loss_weight": 0.35}
        },
    }
