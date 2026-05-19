from neural_search.meta_config_memory import MetaConfigMemory
from neural_search.meta_memory_effectiveness import evaluate_meta_memory_effectiveness


def test_memory_effectiveness_detects_helpful_and_harmful_patterns():
    memory = MetaConfigMemory()
    for idx in range(2):
        memory.add_decision(
            meta_decision_id=f"d{idx}",
            next_run_config_hash=f"h{idx}",
            applied_config={},
            attribution_report={},
            quality_report={},
            process_improvement_delta=-0.1,
            helped_changes=[],
            harmed_changes=["search_strategy_config.transfer_weight"],
            recommendation="revert_previous_config",
            seed=idx,
            mode="smoke",
        )
    decisions = [
        {
            "prior_meta_memory_used": True,
            "reused_successful_config_patterns": ["search_strategy_config.mutation_penalties.widen_hidden_dim"],
            "avoided_harmful_config_patterns": ["search_strategy_config.transfer_weight"],
        }
    ]
    qualities = [
        {
            "recommendation": "revise_previous_config",
            "evidence_used": {"process_improvement_components": {"final_process_improvement_delta": 0.2}},
        }
    ]
    report = evaluate_meta_memory_effectiveness(memory, decisions, qualities)
    assert report.reused_helpful_patterns_count == 1
    assert report.avoided_harmful_patterns_count == 1
    assert report.repeated_harmful_patterns_count == 1
    assert report.memory_guidance_helped_count == 1
