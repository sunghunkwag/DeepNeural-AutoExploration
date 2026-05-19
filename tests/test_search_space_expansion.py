from failure_residue import FailureResidue, MissingOperatorInferencer
from meta_rsi.search_space_expander import SearchSpaceRegistry


def _residue(task_id, split="validation"):
    return FailureResidue(
        task_id=task_id,
        benchmark_name="test",
        seed=1,
        failure_type="object-binding failure",
        failed_operator_or_architecture="arch",
        suspected_bottleneck="object binding",
        evidence_metrics={"object_binding_error_rate": 0.5},
        proposed_operator_or_module_family="object_binding_adapter",
        split_used=split,
    )


def test_search_space_expansion_requires_repeated_evidence():
    one = [_residue("a")]
    hypotheses = MissingOperatorInferencer(min_support=1).infer(one)
    registry = SearchSpaceRegistry(min_support=2)
    assert registry.update_from_hypotheses(hypotheses, one) == {}

    two = [_residue("a"), _residue("b", split="hidden_validation")]
    hypotheses = MissingOperatorInferencer(min_support=1).infer(two)
    updates = registry.update_from_hypotheses(hypotheses, two)
    assert "object_binding_adapter" in updates
    assert updates["object_binding_adapter"]["justification_count"] >= 2
    assert updates["object_binding_adapter"]["audit"]
    assert updates["object_binding_adapter"]["confidence_score"] > 0.0
    assert updates["object_binding_adapter"]["recommended_mutation_operators"] == ["object_binding_adapter"]
    assert updates["object_binding_adapter"]["justifying_residue_ids"]


def test_inferencer_maps_probe_failure_to_bounded_module_family():
    residue = FailureResidue(
        task_id="probe",
        benchmark_name="test",
        seed=1,
        failure_type="sequence instability",
        failed_operator_or_architecture="arch",
        suspected_bottleneck="state",
        proposed_operator_or_module_family="generic",
        split_used="validation",
    )
    hypothesis = MissingOperatorInferencer(min_support=1).infer([residue])[0]
    assert hypothesis.proposed_module_family == "sequence_state_adapter"
