import json

from neural_search.meta_config_memory import MetaConfigMemory
from neural_search.search_process_model import SearchProcessDiagnosis
from neural_search.search_strategy_mutator import SearchStrategyConfig, SearchStrategyMutator


def test_meta_config_memory_keeps_bounded_json_serializable_records(tmp_path):
    memory = MetaConfigMemory(max_records=2)
    for index in range(3):
        memory.add_decision(
            meta_decision_id=f"d{index}",
            next_run_config_hash=f"h{index}",
            applied_config={"index": index},
            attribution_report={"neutral_changes": []},
            quality_report={"overall": index},
            process_improvement_delta=float(index),
            helped_changes=[f"search_strategy_config.hidden_validation_weight.{index}"],
            harmed_changes=[],
            recommendation="trust_previous_config",
            seed=index,
            mode="smoke",
        )
    assert len(memory.records) == 2
    encoded = json.dumps(memory.to_dict(), sort_keys=True)
    assert "d0" not in encoded
    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = MetaConfigMemory.load(path)
    assert len(loaded.records) == 2


def test_prior_harmful_config_patterns_are_avoided_when_memory_is_supplied():
    memory = MetaConfigMemory()
    memory.add_decision(
        meta_decision_id="d1",
        next_run_config_hash="h1",
        applied_config={},
        attribution_report={"harmed_changes": ["search_strategy_config.hidden_validation_weight"]},
        quality_report={},
        process_improvement_delta=-1.0,
        helped_changes=[],
        harmed_changes=["search_strategy_config.hidden_validation_weight"],
        recommendation="revert_previous_config",
        seed=1,
        mode="smoke",
    )
    base = SearchStrategyConfig()
    diagnosis = SearchProcessDiagnosis(evaluator_overfit_risk=0.8)
    plan = SearchStrategyMutator().mutate(diagnosis, base, memory)
    assert plan.mutated_config["hidden_validation_weight"] == base.hidden_validation_weight
    guidance = plan.evidence_used["meta_config_memory_guidance"]
    assert "search_strategy_config.hidden_validation_weight" in guidance["avoided_harmful_config_patterns"]
