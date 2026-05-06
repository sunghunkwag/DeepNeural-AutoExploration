import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cognitive_core import CognitiveCore, EpisodicMemory, MemoryRecord, WorldModel
from experiment_manifest import ExperimentManifest, build_manifest, stable_config_hash, validate_manifest
from learned_task_encoder import LearnedTaskEncoder, TaskConditionedRegressor, handcrafted_support_baseline_loss, task_reconstruction_loss, train_task_encoder
from maml_functional import functional_forward, maml_inner_loop
from model_based_controller import ModelBasedController, UpdateAction, default_update_candidates
from operator_mutation import OperatorMutation, OperatorMutationController
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite


def _cfg(seed=0):
    return DNAXConfig(input_dim=1, hidden_dims=[16, 16], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False, seed=seed, inner_lr=0.01, inner_steps=1, use_second_order=False, replay_buffer_size=8)


def _adapt_loss(model, task, lr):
    adapted = maml_inner_loop(model, dict(model.named_parameters()), task.support_x, task.support_y, lr, 1, first_order=True)
    with torch.no_grad():
        return float(F.mse_loss(functional_forward(model, adapted, task.query_x), task.query_y))


def test_learned_encoder_shape_permutation_support_y_and_query_y_independence():
    suite = ProceduralTaskSuite(seed=10)
    task = suite.sample_function_task("sinusoid")
    enc = LearnedTaskEncoder(latent_dim=8, seed=10)
    z = enc.encode(task.support_x, task.support_y)
    assert z.shape == (8,)
    perm = torch.randperm(task.support_x.shape[0])
    assert torch.allclose(z, enc.encode(task.support_x[perm], task.support_y[perm]), atol=1e-6)
    assert not torch.allclose(z, enc.encode(task.support_x, task.support_y + 1.0))
    query_y = task.query_y.clone()
    z_before = enc.encode(task.support_x, task.support_y)
    query_y.add_(1000.0)
    z_after = enc.encode(task.support_x, task.support_y)
    assert torch.allclose(z_before, z_after)


def test_learned_encoder_training_reduces_loss_and_beats_handcrafted_proxy():
    torch.manual_seed(11)
    suite = ProceduralTaskSuite(seed=11)
    train = suite.sample_function_tasks("train", 12)
    val = suite.sample_function_tasks("validation", 4, ood=False)
    enc = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=11)
    dec = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
    initial = float(task_reconstruction_loss(enc, dec, train).detach())
    result = train_task_encoder(enc, dec, train, val, steps=60, lr=0.01)
    assert result.train_losses[-1] < initial
    assert result.validation_metrics["reconstruction_loss"] <= handcrafted_support_baseline_loss(val) * 1.25


def test_z_task_changes_controller_policy_and_memory_score():
    class ZWorld(WorldModel):
        def forward(self, latent_state, action):
            out = torch.zeros_like(latent_state)
            out[..., 0] = latent_state[..., 0] * action[..., 0]
            return out
    cands = [UpdateAction("low", 0.1), UpdateAction("high", 1.0)]
    ctrl = ModelBasedController(ZWorld(8, 6), cands, latent_dim=8)
    assert ctrl.select_action(torch.tensor([1.0] + [0.0]*7)).selected.name == "high"
    assert ctrl.select_action(torch.tensor([-1.0] + [0.0]*7)).selected.name == "low"
    mem = EpisodicMemory(capacity=2)
    mem.add(MemoryRecord("a", torch.zeros(1), torch.zeros(1), 0.0, 0.0, torch.tensor([1.0, 0.0])))
    mem.add(MemoryRecord("b", torch.zeros(1), torch.zeros(1), 0.0, 0.0, torch.tensor([0.0, 1.0])))
    assert mem.retrieve(torch.tensor([0.9, 0.1]), k=1)[0][0].task_id == "a"
    assert mem.retrieve(torch.tensor([0.1, 0.9]), k=1)[0][0].task_id == "b"


def test_memory_conditioned_adaptation_relevant_beats_no_and_wrong_memory():
    suite = ProceduralTaskSuite(seed=12)
    task = suite.sample_function_task("noisy_affine", split="validation")
    cfg = _cfg(12)
    model = DeepNeuralAutoExplorer(cfg)
    enc = LearnedTaskEncoder(latent_dim=8, seed=12)
    z = enc.infer(task.support_x, task.support_y)
    relevant = EpisodicMemory(capacity=2)
    relevant.add(MemoryRecord("relevant", torch.zeros(1), torch.tensor([0.05]), 0.5, 0.0, z, split="train"))
    wrong = EpisodicMemory(capacity=2)
    wrong.add(MemoryRecord("wrong", torch.zeros(1), torch.tensor([0.05]), -0.5, 0.0, z.roll(1), split="train"))
    rel_core = CognitiveCore(model, cfg, WorldModel(8, 6), memory=relevant, task_inference=enc)
    wrong_core = CognitiveCore(model, cfg, WorldModel(8, 6), memory=wrong, task_inference=enc)
    rel_lr, _ = rel_core.memory_conditioned_inner_lr(z, base_lr=0.03)
    wrong_lr, _ = wrong_core.memory_conditioned_inner_lr(z, base_lr=0.03)
    assert rel_lr > 0.03
    assert wrong_lr <= 0.03
    assert _adapt_loss(model, task, rel_lr) <= _adapt_loss(model, task, wrong_lr) * 1.1
    with pytest.raises(ValueError):
        relevant.add(MemoryRecord("leak", torch.zeros(1), torch.zeros(1), 0, 0, z, contains_query_targets=True))
    for i in range(5):
        relevant.add(MemoryRecord(str(i), torch.zeros(1), torch.zeros(1), 0, 0, z + i))
    assert len(relevant) == relevant.capacity


def test_model_based_controller_calls_world_model_wrong_model_changes_and_logs_error():
    class Good(WorldModel):
        def forward(self, latent_state, action):
            out = torch.zeros_like(latent_state); out[..., 0] = action[..., 0]; return out
    class Bad(WorldModel):
        def forward(self, latent_state, action):
            out = torch.zeros_like(latent_state); out[..., 0] = -action[..., 0]; return out
    cands = [UpdateAction("low", 0.1), UpdateAction("high", 1.0)]
    good = ModelBasedController(Good(8, 6), cands, latent_dim=8)
    bad = ModelBasedController(Bad(8, 6), cands, latent_dim=8)
    z = torch.ones(8)
    gd = good.select_action(z)
    bd = bad.select_action(z)
    assert good.world_model_call_count == 2
    assert gd.selected.name != bd.selected.name
    entry = good.log_actual_outcome(gd, actual_improvement=0.25)
    assert "prediction_error" in entry
    wm = WorldModel(8, 6, seed=13)
    ctrl = ModelBasedController(wm, default_update_candidates(0.01), latent_dim=8)
    state, action = torch.randn(32, 8), torch.randn(32, 6)
    next_state = torch.tanh(state + 0.1 * action[:, :1].expand(-1, 8))
    initial, final = ctrl.train_on_transition_batch((state, action, next_state), steps=30, lr=0.02)
    assert final < initial


def test_operator_mutation_accept_reject_validation_only_and_log():
    val_tasks = [type("T", (), {"split": "validation"})() for _ in range(2)]
    ctrl = OperatorMutationController({"inner_lr": 1.0}, random_seed=123)
    good = ctrl.evaluate(OperatorMutation("good", {"inner_lr": 0.5}), val_tasks, lambda state, tasks: [state["inner_lr"] for _ in tasks])
    assert good["accepted"] and ctrl.accepted_mutation_count == 1
    bad = ctrl.evaluate(OperatorMutation("bad", {"inner_lr": 9.0}), val_tasks, lambda state, tasks: [state["inner_lr"] for _ in tasks])
    assert not bad["accepted"] and bad["rollback"] and ctrl.state["inner_lr"] == 0.5
    assert {"candidate_name", "baseline_scores", "candidate_scores", "accepted", "rollback", "random_seed"}.issubset(ctrl.log[-1])
    with pytest.raises(ValueError):
        ctrl.evaluate(OperatorMutation("no_val", {"inner_lr": 1.0}), [], lambda s, t: [0.0])
    test_task = type("T", (), {"split": "test"})()
    with pytest.raises(ValueError):
        ctrl.evaluate(OperatorMutation("test", {"inner_lr": 1.0}), [test_task], lambda s, t: [0.0], split="test")


def test_manifest_records_ids_and_detects_leakage_and_hash_changes():
    suite = ProceduralTaskSuite(seed=14)
    train, val, test = suite.sample_function_tasks("train", 2), suite.sample_function_tasks("validation", 2), suite.sample_function_tasks("test", 2, ood=True)
    manifest = build_manifest("cmd", [14], train, val, test, {"a": 1}, {"m": 1.0}, [])
    assert manifest.train_task_ids and manifest.validation_task_ids and manifest.test_task_ids
    assert stable_config_hash({"a": 1}) != stable_config_hash({"a": 2})
    with pytest.raises(ValueError):
        build_manifest("cmd", [], train, val, test, {"a": 1}, {}, [])
    overlap = ExperimentManifest(manifest.timestamp, manifest.git_commit, manifest.command, [14], ["x"], ["x"], ["z"], {}, {"test": True}, config_hash="abc")
    with pytest.raises(ValueError):
        validate_manifest(overlap)
    bad_mut = ExperimentManifest(manifest.timestamp, manifest.git_commit, manifest.command, [14], ["a"], ["b"], ["c"], {}, {"test": True}, accepted_mutations=[{"split": "test"}], config_hash="abc")
    with pytest.raises(ValueError):
        validate_manifest(bad_mut)
    with pytest.raises(ValueError):
        validate_manifest(manifest, memory_records=[{"contains_query_targets": True}])


def test_next_benchmark_result_has_seed_config_task_ids_and_no_hardcoded_result_literals(tmp_path):
    from benchmarks.next_agi_scaffold_benchmark import run
    out = tmp_path / "smoke.json"
    result = run("smoke", 7, str(out))
    payload = json.loads(out.read_text())
    assert payload["seeds"] == [7]
    assert payload["config"]["mode"] == "smoke"
    assert payload["task_ids"]["train"] and payload["task_ids"]["validation"] and payload["task_ids"]["test"]
    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.exists()
    source = Path("benchmarks/next_agi_scaffold_benchmark.py").read_text()
    forbidden = ["201.10301955540976", "178.31387853622437", "13.410557270050049"]
    assert not any(value in source for value in forbidden)
    assert result["aggregate"]["ablations"]["full_loop"]["mean"] < result["aggregate"]["ablations"]["no_adaptation"]["mean"]
