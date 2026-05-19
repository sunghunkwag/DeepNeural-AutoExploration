from neural_search import ArchitectureGenome
from neural_search.mutation_policy import MutationPolicy
from neural_search.search_credit_assignment import SearchCreditAssigner
from neural_search.search_history import SearchHistory


def test_history_guidance_prefers_methods_with_hidden_validation_credit():
    parent = ArchitectureGenome(seed=601)
    rejected = parent.with_updates(residual_blocks=2, mutation_method="add_residual_block", seed=602)
    accepted = parent.with_updates(memory_gates=1, mutation_method="add_memory_gate", seed=603)
    history = SearchHistory()
    history.add(
        parent=parent,
        candidate=rejected,
        mutation_method="add_residual_block",
        validation_metrics={"loss": 0.4},
        hidden_validation_metrics={"loss": 0.6},
        accepted=False,
        seed=602,
        validation_improvement=0.02,
        hidden_validation_improvement=-0.10,
        rollback_reason="hidden_validation_regression",
    )
    history.add(
        parent=parent,
        candidate=accepted,
        mutation_method="add_memory_gate",
        validation_metrics={"loss": 0.35},
        hidden_validation_metrics={"loss": 0.30},
        accepted=True,
        seed=603,
        validation_improvement=0.04,
        hidden_validation_improvement=0.12,
    )
    credit = SearchCreditAssigner().assign(history=history)
    methods = MutationPolicy("history_guided", seed=604).choose_methods(history=history, credit_report=credit, count=2)
    assert methods[0] == "add_memory_gate"
    assert credit.method_credit["add_memory_gate"] > credit.method_credit["add_residual_block"]


def test_mutation_policy_avoids_duplicate_equivalent_candidates_in_generation():
    parent = ArchitectureGenome(seed=611)
    proposals = MutationPolicy("greedy", seed=612).propose(parent, count=4)
    signatures = {
        (
            proposal.candidate.hidden_dim,
            proposal.candidate.residual_blocks,
            proposal.candidate.memory_gates,
            proposal.candidate.causal_bottlenecks,
        )
        for proposal in proposals
    }
    assert len(signatures) == len(proposals)

