import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cognitive_core import (
    EpisodicMemory,
    ImprovementCandidate,
    MemoryRecord,
    Planner,
    SelfImprovementController,
    TaskInferenceModule,
    WorldModel,
    generate_linear_dynamics_batch,
)
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite, assert_disjoint_task_ids, function_task_variance


def test_train_eval_task_ids_do_not_overlap():
    suite = ProceduralTaskSuite(seed=1)
    train = suite.sample_function_tasks("train", 6)
    test = suite.sample_function_tasks("test", 6, ood=True)
    assert_disjoint_task_ids(train, test)


def test_query_targets_not_visible_to_support_task_inference_or_memory():
    suite = ProceduralTaskSuite(seed=2)
    task = suite.sample_function_task("sinusoid", split="train")
    infer = TaskInferenceModule()
    emb_a = infer.infer(task.support_x, task.support_y)
    altered = copy.copy(task)
    object.__setattr__(altered, "query_y", task.query_y + 1000.0)
    emb_b = infer.infer(altered.support_x, altered.support_y)
    assert torch.allclose(emb_a, emb_b)
    memory = EpisodicMemory(capacity=2)
    with pytest.raises(ValueError):
        memory.add(MemoryRecord("leak", task.query_y.flatten(), torch.zeros(1), 0.0, 0.0, emb_a, contains_query_targets=True))


def test_benchmark_function_families_are_not_constant():
    suite = ProceduralTaskSuite(seed=3)
    for family in ["sinusoid", "polynomial", "piecewise_linear", "noisy_affine", "compositional"]:
        task = suite.sample_function_task(family, split="train", n_query=32)
        assert function_task_variance(task) > 1e-5, family


def test_ood_sampler_changes_distribution():
    suite = ProceduralTaskSuite(seed=4)
    train = suite.sample_function_tasks("train", 20, ood=False)
    test = suite.sample_function_tasks("test", 20, ood=True)
    train_abs_x = torch.cat([t.query_x.abs().flatten() for t in train]).mean().item()
    test_abs_x = torch.cat([t.query_x.abs().flatten() for t in test]).mean().item()
    assert test_abs_x > train_abs_x * 1.12


def test_self_improvement_requires_validation_before_acceptance():
    ctrl = SelfImprovementController({"inner_lr": 0.1})
    candidate = ImprovementCandidate("fake", {"inner_lr": 0.2})
    with pytest.raises(ValueError):
        ctrl.evaluate_and_update(candidate, [], lambda state, tasks: [0.0])


def test_self_improvement_rollback_restores_previous_state():
    ctrl = SelfImprovementController({"inner_lr": 0.1}, improvement_threshold=0.0)
    candidate = ImprovementCandidate("bad", {"inner_lr": 10.0})
    decision = ctrl.evaluate_and_update(candidate, [object()], lambda state, tasks: [state["inner_lr"]])
    assert not decision["accepted"]
    assert ctrl.state == {"inner_lr": 0.1}
    assert ctrl.rollback_count == 1


def test_deterministic_mode_reproducible():
    cfg = DNAXConfig(input_dim=1, hidden_dims=[8], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False)
    torch.manual_seed(123)
    model_a = DeepNeuralAutoExplorer(cfg)
    torch.manual_seed(123)
    model_b = DeepNeuralAutoExplorer(cfg)
    x = torch.linspace(-1, 1, 5).unsqueeze(1)
    assert torch.allclose(model_a(x), model_b(x))


def test_stochastic_exploration_varies_outputs():
    cfg = DNAXConfig(input_dim=1, hidden_dims=[8], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False, exploration_noise_scale=0.5)
    torch.manual_seed(5)
    model = DeepNeuralAutoExplorer(cfg)
    x = torch.ones(5, 1)
    y1, _ = model.explore(x, perturb=True)
    y2, _ = model.explore(x, perturb=True)
    assert not torch.allclose(y1, y2)


def test_planner_uses_world_model_predictions():
    class AddWorld(WorldModel):
        def forward(self, latent_state, action):
            return latent_state + action

    class IgnoreActionWorld(WorldModel):
        def forward(self, latent_state, action):
            return latent_state

    actions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    add_plan = Planner(AddWorld(2, 2), actions, horizon=2).plan(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    ignore_plan = Planner(IgnoreActionWorld(2, 2), actions, horizon=2).plan(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    assert add_plan[1] < 1e-8
    assert ignore_plan[1] > 0.1


def test_memory_retrieval_uses_query_similarity():
    memory = EpisodicMemory(capacity=3)
    memory.add(MemoryRecord("near", torch.zeros(1), torch.zeros(1), 0.0, 0.0, torch.tensor([1.0, 0.0])))
    memory.add(MemoryRecord("far", torch.zeros(1), torch.zeros(1), 0.0, 0.0, torch.tensor([0.0, 1.0])))
    hit = memory.retrieve(torch.tensor([0.9, 0.1]), k=1)[0][0]
    assert hit.task_id == "near"


def test_world_model_loss_decreases_on_learnable_dynamics():
    torch.manual_seed(7)
    wm = WorldModel(2, 2, hidden_dim=32, seed=7)
    batch = generate_linear_dynamics_batch(n=96, seed=7)
    opt = torch.optim.Adam(wm.parameters(), lr=0.03)
    initial = float(wm.prediction_loss(*batch).detach())
    for _ in range(80):
        wm.train_step(batch, opt)
    final = float(wm.prediction_loss(*batch).detach())
    assert final < initial * 0.5
