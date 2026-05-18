import torch
import pytest

from neural_search import (
    MUTATION_METHODS,
    ArchitectureEvaluator,
    ArchitectureGenome,
    NeuralCandidateSandbox,
    NeuralSandboxTask,
    inherit_shape_compatible_weights,
    mutate_architecture,
)


def _signature(module):
    return {
        "parameter_count": sum(param.numel() for param in module.parameters()),
        "parameter_shapes": sorted((name, tuple(param.shape)) for name, param in module.named_parameters()),
        "module_types": [type(item).__name__ for item in module.modules()],
    }


def test_architecture_mutations_change_built_networks():
    parent = ArchitectureGenome(input_dim=3, output_dim=2, hidden_dim=8, residual_blocks=2, seed=10)
    parent_module = parent.build_module()
    parent_signature = _signature(parent_module)
    x = torch.randn(4, 3)

    for index, method in enumerate(MUTATION_METHODS):
        child = mutate_architecture(parent, method, seed=100 + index)
        child_module = child.build_module()
        assert child.parent_id == parent.genome_id
        assert child.mutation_method == method
        assert child_module(x).shape == (4, 2)
        assert _signature(child_module) != parent_signature


def test_mutation_seed_is_deterministic():
    parent = ArchitectureGenome(input_dim=3, output_dim=2, hidden_dim=8, residual_blocks=1, seed=11)
    first = mutate_architecture(parent, "widen_hidden_dim", seed=22)
    second = mutate_architecture(parent, "widen_hidden_dim", seed=22)
    assert first.to_dict() == second.to_dict()


def test_weight_inheritance_copies_compatible_tensors():
    parent = ArchitectureGenome(input_dim=3, output_dim=2, hidden_dim=8, residual_blocks=1, seed=1)
    child = mutate_architecture(parent, "add_residual_block", seed=2)
    parent_module = parent.build_module()
    child_module = child.build_module()
    with torch.no_grad():
        for param in parent_module.parameters():
            param.fill_(0.123)

    report = inherit_shape_compatible_weights(parent_module, child_module)
    child_state = child_module.state_dict()
    assert report.copied
    for name in report.copied:
        assert torch.allclose(child_state[name], torch.full_like(child_state[name], 0.123))


def test_neural_candidate_sandbox_rejects_heldout_leakage():
    genome = ArchitectureGenome(input_dim=2, output_dim=1, hidden_dim=4, seed=3)
    task = NeuralSandboxTask(
        "leak",
        train_x=torch.randn(4, 2),
        train_y=torch.randn(4, 1),
        validation_x=torch.randn(4, 2),
        validation_y=torch.randn(4, 1),
        used_heldout_labels=True,
    )
    sandbox = NeuralCandidateSandbox(seed=3, train_steps=1, task_count=1, samples_per_split=8)
    with pytest.raises(ValueError):
        sandbox.evaluate(genome, [task])


def test_architecture_evaluator_audit_records_validation_only_decision():
    parent = ArchitectureGenome(input_dim=2, output_dim=1, hidden_dim=4, seed=4)
    child = mutate_architecture(parent, "add_memory_gate", seed=5)
    sandbox = NeuralCandidateSandbox(seed=4, train_steps=2, task_count=1, samples_per_split=8)
    evaluator = ArchitectureEvaluator(seed=4, sandbox=sandbox, improvement_threshold=-999.0)
    decision = evaluator.evaluate_candidate(parent, child)
    assert decision.split_used == "validation"
    assert not decision.used_heldout_labels
    assert decision.weight_inheritance["copied_count"] > 0
    assert decision.to_dict()["parent_id"] == parent.genome_id
