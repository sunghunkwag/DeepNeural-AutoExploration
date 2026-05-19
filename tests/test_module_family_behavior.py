import torch

from neural_search import ArchitectureGenome


def _forward(genome):
    model = genome.build_module()
    x = torch.randn(6, genome.input_dim)
    return model, x, model(x)


def test_object_binding_adapter_changes_module_graph_and_forward_behavior():
    base = ArchitectureGenome(hidden_dim=12, seed=401)
    adapted = base.with_updates(object_binding_adapters=1, object_slots=3, mutation_method="object_binding_adapter", seed=402)
    base_model, x, base_out = _forward(base)
    adapted_model = adapted.build_module()
    adapted_out = adapted_model(x)
    assert len(base_model.object_binding_adapters) == 0
    assert len(adapted_model.object_binding_adapters) == 1
    assert not torch.allclose(base_out, adapted_out)
    assert adapted_model.object_binding_adapters[0].last_attention.shape[1:] == (3, 3)


def test_sequence_state_adapter_changes_module_graph_and_forward_behavior():
    base = ArchitectureGenome(hidden_dim=12, seed=411)
    adapted = base.with_updates(sequence_state_adapters=1, sequence_chunks=4, mutation_method="sequence_state_adapter", seed=412)
    base_model, x, base_out = _forward(base)
    adapted_model = adapted.build_module()
    adapted_out = adapted_model(x)
    assert len(base_model.sequence_state_adapters) == 0
    assert len(adapted_model.sequence_state_adapters) == 1
    assert not torch.allclose(base_out, adapted_out)
    assert adapted_model.sequence_state_adapters[0].last_state_norm > 0.0


def test_causal_bottleneck_exposes_probe_relevant_latent_usage():
    genome = ArchitectureGenome(hidden_dim=10, causal_bottlenecks=1, bottleneck_dim=3, seed=421)
    model, _x, _out = _forward(genome)
    bottleneck = model.causal_bottlenecks[0]
    assert bottleneck.last_latent is not None
    assert bottleneck.last_latent.shape[-1] == 3
    assert bottleneck.last_usage >= 0.0

