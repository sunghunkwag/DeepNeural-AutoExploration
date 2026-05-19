from failure_residue import FailureResidue
from neural_search import ArchitectureGenome
from neural_search.deep_search_controller import DeepSearchController
from neural_search.search_history import SearchHistory


def _residue(index=0):
    return FailureResidue(
        task_id=f"validation-object_state_transition-{index}",
        benchmark_name="test",
        seed=801,
        failure_type="object-binding failure",
        failed_operator_or_architecture="arch",
        suspected_bottleneck="object binding",
        evidence_metrics={"hidden_validation_regression": -0.1},
        proposed_operator_or_module_family="object_binding_adapter",
        split_used="hidden_validation",
    )


def test_deep_search_controller_uses_residue_history_and_probe_evidence():
    parent = ArchitectureGenome(seed=801)
    history = SearchHistory()
    for index in range(3):
        candidate = parent.with_updates(hidden_dim=parent.hidden_dim + index + 2, mutation_method="widen_hidden_dim", seed=802 + index)
        history.add(
            parent=parent,
            candidate=candidate,
            mutation_method="widen_hidden_dim",
            validation_metrics={"loss": 0.4},
            hidden_validation_metrics={"loss": 0.6},
            accepted=False,
            seed=802 + index,
            validation_improvement=0.02,
            hidden_validation_improvement=-0.05,
            rollback_reason="hidden_validation_regression",
        )
    output = DeepSearchController(seed=801).decide(parent, residues=[_residue(1), _residue(2)], history=history, generation=1)
    decision = output.decision
    assert decision.selected_depth_level >= 2
    assert "object_binding_adapter" in decision.selected_mutation_methods
    assert decision.used_heldout_labels is False
    assert decision.expected_failure_modes_addressed

