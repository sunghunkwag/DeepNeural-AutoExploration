from failure_residue import FailureResidue
from neural_search import ArchitectureGenome
from neural_search.search_depth_scheduler import SearchDepthScheduler
from neural_search.search_history import SearchHistory


def _history_with_hidden_regressions(parent, count):
    history = SearchHistory()
    for index in range(count):
        candidate = parent.with_updates(hidden_dim=parent.hidden_dim + index + 2, mutation_method="widen_hidden_dim", seed=901 + index)
        history.add(
            parent=parent,
            candidate=candidate,
            mutation_method="widen_hidden_dim",
            validation_metrics={"loss": 0.4},
            hidden_validation_metrics={"loss": 0.7},
            accepted=False,
            seed=901 + index,
            validation_improvement=0.01,
            hidden_validation_improvement=-0.08,
        )
    return history


def _residue(index):
    return FailureResidue(
        task_id=f"task-{index}",
        benchmark_name="test",
        seed=901,
        failure_type="gradient bottleneck",
        failed_operator_or_architecture="arch",
        suspected_bottleneck="gradient",
        proposed_operator_or_module_family="wider_residual_stack",
        split_used="validation",
    )


def test_repeated_shallow_failure_increases_search_depth():
    parent = ArchitectureGenome(seed=901)
    assessment = SearchDepthScheduler().schedule(
        parent,
        history=_history_with_hidden_regressions(parent, 3),
        residues=[_residue(1), _residue(2)],
    )
    assert assessment.selected_depth_level >= 2
    assert assessment.rollback_risk > 0.0


def test_recent_hidden_validation_improvement_avoids_unnecessary_depth():
    parent = ArchitectureGenome(seed=911)
    history = SearchHistory()
    candidate = parent.with_updates(hidden_dim=parent.hidden_dim + 2, mutation_method="widen_hidden_dim", seed=912)
    history.add(
        parent=parent,
        candidate=candidate,
        mutation_method="widen_hidden_dim",
        validation_metrics={"loss": 0.2},
        hidden_validation_metrics={"loss": 0.2},
        accepted=True,
        seed=912,
        validation_improvement=0.05,
        hidden_validation_improvement=0.06,
    )
    assessment = SearchDepthScheduler().schedule(parent, history=history, residues=[])
    assert assessment.selected_depth_level <= 1

