from failure_residue import FailureResidue
from neural_search import ArchitectureGenome
from neural_search.mutation_policy import MutationPolicy
from neural_search.search_history import SearchHistory


def _residue(failure_type):
    return FailureResidue(
        task_id=f"task-{failure_type}",
        benchmark_name="test",
        seed=1,
        failure_type=failure_type,
        failed_operator_or_architecture="arch",
        suspected_bottleneck="test",
        proposed_operator_or_module_family="wider_residual_stack",
        split_used="validation",
    )


def test_residue_guided_policy_changes_behavior_based_on_residues():
    parent = ArchitectureGenome(seed=1)
    object_policy = MutationPolicy("residue_guided", seed=7)
    object_methods = object_policy.choose_methods(residues=[_residue("object-binding failure")], count=1)
    causal_methods = object_policy.choose_methods(residues=[_residue("causal intervention failure")], count=1)
    assert object_methods == ["object_binding_adapter"]
    assert causal_methods == ["add_causal_bottleneck"]
    first = object_policy.propose(parent, residues=[_residue("object-binding failure")], count=1)[0]
    second = MutationPolicy("residue_guided", seed=7).propose(parent, residues=[_residue("object-binding failure")], count=1)[0]
    assert first.candidate.to_dict() == second.candidate.to_dict()


def test_history_guided_policy_reuses_successful_method():
    parent = ArchitectureGenome(seed=2)
    candidate = parent.with_updates(hidden_dim=20, mutation_method="widen_hidden_dim", seed=3)
    history = SearchHistory()
    history.add(
        parent=parent,
        candidate=candidate,
        mutation_method="widen_hidden_dim",
        validation_metrics={"loss": 0.1},
        hidden_validation_metrics={"loss": 0.2},
        accepted=True,
        seed=3,
    )
    methods = MutationPolicy("history_guided", seed=9).choose_methods(history=history, count=2)
    assert methods[0] == "widen_hidden_dim"
