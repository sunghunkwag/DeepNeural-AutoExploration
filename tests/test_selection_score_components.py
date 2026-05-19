from neural_search import ArchitectureGenome
from neural_search.multidomain_evaluator import MultiDomainEvaluator, selection_score_components
from neural_search.task_families import build_multidomain_task_families


def test_selection_score_records_all_required_components():
    families = build_multidomain_task_families(seed=501, samples_per_split=8, difficulty="hard")
    parent = ArchitectureGenome(input_dim=4, output_dim=2, hidden_dim=8, seed=501)
    candidate = parent.with_updates(hidden_dim=10, mutation_method="widen_hidden_dim", seed=502)
    evaluator = MultiDomainEvaluator(seed=501, train_steps=2, lr=0.01)
    parent_eval = evaluator.train_and_validate(parent, families)
    candidate_eval = evaluator.train_and_validate(candidate, families)
    decision = evaluator.selection_decision(parent, candidate, parent_eval, candidate_eval, rollback_risk=0.2)
    required = {
        "validation_loss",
        "hidden_validation_loss",
        "cross_domain_transfer_score",
        "generalization_gap_penalty",
        "architecture_complexity_penalty",
        "representation_failure_penalty",
        "gradient_failure_penalty",
        "rollback_risk",
        "task_family_regression_penalty",
        "module_contribution_score",
        "selection_score",
    }
    assert required <= set(decision.selection_score_components)
    assert decision.used_heldout_labels is False
    assert decision.split_used == "validation"


def test_selection_score_penalizes_complexity_and_rollback_risk():
    families = build_multidomain_task_families(seed=502, samples_per_split=8)
    compact = ArchitectureGenome(input_dim=4, output_dim=2, hidden_dim=8, seed=502)
    complex_genome = ArchitectureGenome(
        input_dim=4,
        output_dim=2,
        hidden_dim=32,
        residual_blocks=3,
        memory_gates=1,
        causal_bottlenecks=1,
        seed=503,
    )
    evaluator = MultiDomainEvaluator(seed=502, train_steps=1, lr=0.01)
    compact_eval = evaluator.train_and_validate(compact, families)
    compact_score = selection_score_components(compact, compact_eval)["selection_score"]
    risky_score = selection_score_components(complex_genome, compact_eval, rollback_risk=1.0)["selection_score"]
    assert risky_score < compact_score

