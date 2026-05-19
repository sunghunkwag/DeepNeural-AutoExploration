from neural_search import ArchitectureGenome
from neural_search.search_credit_assignment import SearchCreditAssigner
from neural_search.search_history import SearchHistory


def test_credit_assignment_penalizes_hidden_regression_and_rollbacks():
    parent = ArchitectureGenome(seed=1001)
    history = SearchHistory()
    bad = parent.with_updates(residual_blocks=2, mutation_method="add_residual_block", seed=1002)
    good = parent.with_updates(memory_gates=1, mutation_method="add_memory_gate", seed=1003)
    history.add(
        parent=parent,
        candidate=bad,
        mutation_method="add_residual_block",
        validation_metrics={"loss": 0.3},
        hidden_validation_metrics={"loss": 0.8},
        accepted=False,
        seed=1002,
        validation_improvement=0.02,
        hidden_validation_improvement=-0.12,
    )
    history.add(
        parent=parent,
        candidate=good,
        mutation_method="add_memory_gate",
        validation_metrics={"loss": 0.25},
        hidden_validation_metrics={"loss": 0.2},
        accepted=True,
        seed=1003,
        validation_improvement=0.03,
        hidden_validation_improvement=0.10,
    )
    report = SearchCreditAssigner().assign(history=history)
    assert report.method_credit["add_memory_gate"] > report.method_credit["add_residual_block"]
    assert report.rollback_risk_by_method["add_memory_gate"] < report.rollback_risk_by_method["add_residual_block"]

