from neural_search.meta_config_attribution import ConfigChangeAttributionReport, ConfigChangeRecord
from neural_search.meta_decision_quality import MetaDecisionQualityEvaluator
from neural_search.search_process_model import SearchProcessDiagnosis, compute_process_improvement_components


def _record(path, helped=False, harmed=False):
    return ConfigChangeRecord(
        change_id=f"chg_{path}",
        config_path=path,
        old_value=0.1,
        new_value=0.2,
        change_type="increase_numeric",
        intended_effect="test",
        observed_effect={},
        attribution_confidence=0.7,
        helped=helped,
        harmed=harmed,
        neutral=not helped and not harmed,
        evidence_used={},
    )


def _report(helped=(), harmed=(), neutral=()):
    changes = [_record(path, helped=True) for path in helped]
    changes += [_record(path, harmed=True) for path in harmed]
    changes += [_record(path) for path in neutral]
    return ConfigChangeAttributionReport(
        attribution_id="attr_test",
        first_run_id="run_first",
        second_run_id="run_second",
        changes=changes,
        helped_changes=list(helped),
        harmed_changes=list(harmed),
        neutral_changes=list(neutral),
        process_improvement_components={},
        evidence_used={},
    )


def test_strict_process_improvement_components_preserve_negative_deltas():
    first = SearchProcessDiagnosis(
        rollback_rate_by_method={"widen_hidden_dim": 0.10},
        evaluator_overfit_risk=0.20,
        architecture_complexity_growth=0.10,
        task_family_balance=0.80,
        transfer_regression_risk=0.10,
    )
    second = SearchProcessDiagnosis(
        rollback_rate_by_method={"widen_hidden_dim": 0.30},
        evaluator_overfit_risk=0.30,
        architecture_complexity_growth=0.30,
        task_family_balance=0.70,
        transfer_regression_risk=0.20,
    )
    components = compute_process_improvement_components(first, second)
    assert components["rollback_rate_delta"] > 0.0
    assert components["task_family_balance_delta"] < 0.0
    assert components["final_process_improvement_delta"] < 0.0


def test_decision_quality_recommends_reuse_when_second_run_improves():
    first = {"meta_meta_meta_decision": {"decision_id": "d1", "input_run_id": "r1", "expected_improvement_target": "hidden_validation_robustness"}}
    second = {}
    components = {
        "hidden_validation_robustness_delta": 0.4,
        "residue_resolution_delta": 0.2,
        "task_family_balance_delta": 0.1,
        "rollback_rate_delta": -0.2,
        "transfer_regression_risk_delta": -0.1,
        "complexity_growth_delta": -0.1,
        "evaluator_overfit_risk_delta": -0.2,
        "final_process_improvement_delta": 1.3,
    }
    report = MetaDecisionQualityEvaluator().evaluate(
        first,
        second,
        _report(helped=["search_strategy_config.hidden_validation_weight"]),
        components,
        first_diagnosis=SearchProcessDiagnosis(validation_hidden_mismatch_by_method={"m": 0.4}, persistent_residue_types=["x"]),
        second_diagnosis=SearchProcessDiagnosis(validation_hidden_mismatch_by_method={"m": 0.1}, persistent_residue_types=[]),
    )
    assert report.should_reuse_config is True
    assert report.recommendation == "trust_previous_config"


def test_decision_quality_recommends_revise_or_revert_when_second_run_worsens():
    first = {"meta_meta_meta_decision": {"decision_id": "d2", "input_run_id": "r2", "expected_improvement_target": "residue_resolution_rate"}}
    components = {
        "hidden_validation_robustness_delta": -0.2,
        "residue_resolution_delta": -0.3,
        "task_family_balance_delta": -0.1,
        "rollback_rate_delta": 0.2,
        "transfer_regression_risk_delta": 0.2,
        "complexity_growth_delta": 0.2,
        "evaluator_overfit_risk_delta": 0.2,
        "final_process_improvement_delta": -1.4,
    }
    report = MetaDecisionQualityEvaluator().evaluate(
        first,
        {},
        _report(harmed=["search_strategy_config.mutation_boosts.object_binding_adapter"]),
        components,
        first_diagnosis=SearchProcessDiagnosis(validation_hidden_mismatch_by_method={"m": 0.1}),
        second_diagnosis=SearchProcessDiagnosis(validation_hidden_mismatch_by_method={"m": 0.4}, persistent_residue_types=["x", "y"]),
    )
    assert report.should_revert_config is True
    assert report.recommendation == "revert_previous_config"
