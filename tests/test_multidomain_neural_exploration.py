import torch

from benchmarks.deep_neural_exploration_benchmark import run as run_multidomain_benchmark
from neural_search import ArchitectureGenome
from neural_search.multidomain_evaluator import MultiDomainEvaluator
from neural_search.task_families import assert_disjoint_multidomain_splits, build_multidomain_task_families


def test_multidomain_task_families_produce_disjoint_splits():
    families = build_multidomain_task_families(seed=101, samples_per_split=8)
    assert len(families) == 5
    assert_disjoint_multidomain_splits(families)
    for family in families:
        assert set(family.splits) == {"train", "validation", "hidden_validation", "heldout_test"}
        assert family.splits["train"].x.shape[1] == 4
        assert family.splits["train"].y.shape[1] == 2


def test_multidomain_evaluator_trains_candidate_parameters():
    families = build_multidomain_task_families(seed=102, samples_per_split=8)
    genome = ArchitectureGenome(input_dim=4, output_dim=2, hidden_dim=8, seed=102)
    initial_model = genome.build_module()
    before = {name: value.detach().clone() for name, value in initial_model.state_dict().items() if value.dtype.is_floating_point}
    evaluator = MultiDomainEvaluator(seed=102, train_steps=3, lr=0.01)
    result = evaluator.train_and_validate(genome, families, initial_model=initial_model)
    after = result.model.state_dict()
    delta = sum((after[name] - before[name]).abs().sum().item() for name in before)
    assert delta > 0.0
    assert result.train_steps == 3
    assert result.validation_loss > 0.0
    assert result.hidden_validation_loss > 0.0


def test_deep_neural_exploration_smoke_reports_required_metrics(tmp_path):
    out = tmp_path / "deep_neural_exploration.json"
    result = run_multidomain_benchmark("smoke", 103, str(out))
    metrics = result["metrics"]
    assert out.exists()
    assert metrics["candidate_acceptance_count"] + metrics["candidate_rejection_count"] > 0
    assert metrics["heldout_test_evaluated_after_freeze"] is True
    assert metrics["cross_domain_transfer_score"] > 0.0
    assert "function_regression" in metrics["validation_loss_by_family"]
