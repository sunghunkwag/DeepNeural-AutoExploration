from neural_search import ArchitectureGenome
from neural_search.gradient_probe import gradient_failures_to_traces, probe_gradients
from neural_search.multidomain_evaluator import MultiDomainEvaluator
from neural_search.representation_probe import probe_representations, representation_failures_to_traces
from neural_search.task_families import build_multidomain_task_families


def _trained_model(seed=201):
    families = build_multidomain_task_families(seed=seed, samples_per_split=8)
    genome = ArchitectureGenome(input_dim=4, output_dim=2, hidden_dim=8, residual_blocks=1, seed=seed)
    result = MultiDomainEvaluator(seed=seed, train_steps=2).train_and_validate(genome, families)
    return genome, result.model, families


def test_representation_probe_reads_real_activations():
    genome, model, families = _trained_model()
    probe = probe_representations(model, families, genome_id=genome.genome_id)
    assert probe.activation_variance_by_module
    assert any(value >= 0.0 for value in probe.activation_variance_by_module.values())
    assert probe.module_ablation_contribution
    traces = representation_failures_to_traces(probe, benchmark_name="test", seed=201)
    assert all(trace["used_heldout_labels"] is False for trace in traces)


def test_gradient_probe_reads_real_gradients():
    genome, model, families = _trained_model(seed=202)
    probe = probe_gradients(model, families, genome_id=genome.genome_id)
    assert probe.gradient_norm_by_module
    assert probe.total_gradient_norm > 0.0
    traces = gradient_failures_to_traces(probe, benchmark_name="test", seed=202)
    assert all(trace["split_used"] == "validation" for trace in traces)
