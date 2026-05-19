from neural_search.counterfactual_config_evaluator import CounterfactualConfigEvaluator
from neural_search.meta_config_attribution import ConfigChangeAttributionReport, ConfigChangeRecord
from neural_search.search_process_model import SearchProcessDiagnosis


def test_counterfactual_config_report_is_labeled_estimate_and_not_proof():
    change = ConfigChangeRecord(
        change_id="cfgchg_penalty",
        config_path="search_strategy_config.mutation_penalties.widen_hidden_dim",
        old_value=None,
        new_value=0.5,
        change_type="add",
        intended_effect="reduce rollback rate",
        observed_effect={},
        attribution_confidence=0.7,
        helped=True,
        harmed=False,
        neutral=False,
        evidence_used={},
    )
    attribution = ConfigChangeAttributionReport(
        attribution_id="attr",
        first_run_id="first",
        second_run_id="second",
        changes=[change],
        helped_changes=[change.config_path],
        harmed_changes=[],
        neutral_changes=[],
        process_improvement_components={},
        evidence_used={},
    )
    reports = CounterfactualConfigEvaluator().evaluate(
        attribution,
        SearchProcessDiagnosis(rollback_rate_by_method={"widen_hidden_dim": 0.9}),
        SearchProcessDiagnosis(rollback_rate_by_method={"widen_hidden_dim": 0.2}),
    )
    assert len(reports) == 1
    report = reports[0]
    assert report.estimate_label == "estimate_not_proof"
    assert report.claims_proof is False
    assert "Estimate only" in report.rationale
    assert "proof of causality" not in report.rationale
