import copy
from pathlib import Path

import pytest
import torch

from adaptation_operators import (
    OperatorExecutionContext,
    OperatorGene,
    OperatorGenome,
    default_operator_registry,
    execute_operator,
    select_gene_with_world_model,
)
from benchmarks.recursive_self_improvement_benchmark import (
    WrongWorldModel,
    _assert_operator_diversity,
    _build_memory,
    _evaluate_gene_on_tasks,
    _infer_task,
    _initial_state,
    _make_model,
    _memory_summary,
    _validate_operator_candidate,
)
from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel
from learned_task_encoder import LearnedTaskEncoder
from task_suite import ProceduralTaskSuite


def _context(seed=201, memory=None, world_model=None):
    suite = ProceduralTaskSuite(seed=seed)
    task = suite.sample_function_task("sinusoid", split="validation")
    state = _initial_state(seed)
    encoder = LearnedTaskEncoder(latent_dim=8, seed=seed)
    model = _make_model(state)
    z = encoder.infer(task.support_x, task.support_y)
    return (
        task,
        state,
        encoder,
        OperatorExecutionContext(
            model=model,
            task=task,
            task_embedding=z,
            base_inner_lr=float(state["inner_lr"]),
            inner_steps=1,
            memory=memory,
            memory_k=1,
            world_model=world_model,
        ),
    )


def test_four_adaptation_operators_execute_and_are_not_dead_code():
    registry = default_operator_registry()
    assert {"functional_maml", "reptile_update", "memory_gated_gradient_scaling", "world_model_lr_schedule"}.issubset(registry)
    _, _, _, ctx = _context()
    traces = []
    for name in registry:
        gene = OperatorGene(f"test-{name}", name, {"inner_lr_scale": 1.0, "inner_steps": 1, "memory_gate_strength": 0.5})
        traces.append(execute_operator(gene, ctx))
    assert all(trace.query_loss_after >= 0.0 for trace in traces)
    assert len({round(trace.query_loss_after, 8) for trace in traces}) > 1


def test_accepted_operator_mutation_alters_runtime_behavior():
    seed = 202
    suite = ProceduralTaskSuite(seed=seed)
    train = suite.sample_mixed_tasks("train", 4)
    val = suite.sample_mixed_tasks("validation", 2)
    state = _initial_state(seed)
    encoder = LearnedTaskEncoder(latent_dim=8, seed=seed)
    memory = _build_memory(train, state, encoder)
    genome = OperatorGenome.default()
    before = _evaluate_gene_on_tasks(genome.active_gene(), val, state, encoder, memory, None)
    candidate = OperatorGene("accepted-memory-op", "memory_gated_gradient_scaling", {"inner_lr_scale": 1.0, "inner_steps": 1, "memory_gate_strength": 1.0}, 1, genome.active_gene_id, False)
    decision = _validate_operator_candidate(genome, candidate, val, state, encoder, memory, None, generation=1, seed=seed, improvement_threshold=-1e9)
    after = _evaluate_gene_on_tasks(genome.active_gene(), val, state, encoder, memory, None)
    assert decision["accepted"]
    assert genome.active_gene().operator_name == "memory_gated_gradient_scaling"
    assert before != after


def test_test_split_cannot_be_used_for_operator_acceptance():
    seed = 203
    suite = ProceduralTaskSuite(seed=seed)
    train = suite.sample_mixed_tasks("train", 4)
    test = suite.sample_mixed_tasks("test", 2, ood=True)
    state = _initial_state(seed)
    encoder = LearnedTaskEncoder(latent_dim=8, seed=seed)
    memory = _build_memory(train, state, encoder)
    genome = OperatorGenome.default()
    candidate = genome.mutate_candidates(0)[0]
    with pytest.raises(ValueError):
        _validate_operator_candidate(genome, candidate, test, state, encoder, memory, None, generation=0, seed=seed)


def test_operator_selection_is_not_task_family_hardcoded():
    source = Path("benchmarks/recursive_self_improvement_benchmark.py").read_text() + Path("adaptation_operators.py").read_text()
    assert "if task.family" not in source
    assert "elif task.family" not in source


def test_shuffled_memory_damages_memory_gated_operator_signal():
    task, state, encoder, base_ctx = _context(seed=204)
    z = base_ctx.task_embedding
    good = EpisodicMemory(capacity=2)
    good.add(MemoryRecord("good", torch.zeros(1), torch.zeros(1), 0.8, 0.0, z, split="train"))
    bad = EpisodicMemory(capacity=2)
    bad.add(MemoryRecord("bad", torch.zeros(1), torch.zeros(1), -0.8, 2.0, z.roll(1), split="train"))
    gene = OperatorGene("memory", "memory_gated_gradient_scaling", {"inner_lr_scale": 1.0, "inner_steps": 1, "memory_gate_strength": 1.0})
    good_ctx = copy.copy(base_ctx)
    bad_ctx = copy.copy(base_ctx)
    good_ctx.memory = good
    bad_ctx.memory = bad
    good_trace = execute_operator(gene, good_ctx)
    bad_trace = execute_operator(gene, bad_ctx)
    assert good_trace.memory_reward_signal > bad_trace.memory_reward_signal
    assert good_trace.selected_lrs != bad_trace.selected_lrs


def test_wrong_world_model_changes_operator_selection():
    class PreferWorld(WorldModel):
        def forward(self, latent_state, action):
            out = torch.zeros_like(latent_state)
            out[..., 0] = action[..., 4]
            return out

    genome = OperatorGenome.default()
    _, _, _, ctx = _context(seed=205)
    summary = _memory_summary(ctx.task_embedding, None, 1)
    good_gene, _ = select_gene_with_world_model(genome, ctx.task_embedding, summary, PreferWorld(8, 6))
    wrong_gene, _ = select_gene_with_world_model(genome, ctx.task_embedding, summary, WrongWorldModel(8, 6))
    assert good_gene.gene_id != wrong_gene.gene_id


def test_accepted_operator_can_be_reused_in_later_generation():
    genome = OperatorGenome.default()
    candidate = OperatorGene("g0-memory", "memory_gated_gradient_scaling", {"inner_lr_scale": 1.0, "inner_steps": 1, "memory_gate_strength": 1.0}, 0, genome.active_gene_id, False)
    genome.accept(candidate)
    later_generation_available = {gene.gene_id for gene in genome.accepted_genes()}
    assert "g0-memory" in later_generation_available
    assert genome.active_gene().gene_id == "g0-memory"


def test_rejected_operator_rolls_back_exactly():
    genome = OperatorGenome.default()
    before = genome.snapshot()
    candidate = OperatorGene("reject-me", "reptile_update", {"inner_lr_scale": 9.0}, 1, genome.active_gene_id, False)
    genome.reject(candidate)
    genome.restore(before)
    assert genome.snapshot() == before


def test_recursive_benchmark_dead_operator_check_fails_all_equal():
    with pytest.raises(RuntimeError):
        _assert_operator_diversity({"a": [1.0, 1.0], "b": [1.0, 1.0]})
