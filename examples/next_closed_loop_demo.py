"""Transparent CPU demo of the next closed-loop learning scaffold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from cognitive_core import CognitiveCore, EpisodicMemory, MemoryRecord, WorldModel
from experiment_manifest import build_manifest, write_manifest
from learned_task_encoder import LearnedTaskEncoder, TaskConditionedRegressor, train_task_encoder
from maml_functional import functional_forward, maml_inner_loop
from model_based_controller import ModelBasedController, default_update_candidates
from operator_mutation import OperatorMutationController, allowed_operator_mutations
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite, assert_disjoint_task_ids


def _adapt_loss(model, task, lr, steps=1):
    adapted = maml_inner_loop(model, dict(model.named_parameters()), task.support_x, task.support_y, lr, steps, first_order=True)
    with torch.no_grad():
        return float(F.mse_loss(functional_forward(model, adapted, task.query_x), task.query_y))


def main() -> None:
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    suite = ProceduralTaskSuite(seed=seed)
    train_tasks = suite.sample_function_tasks("train", 5)
    val_tasks = suite.sample_function_tasks("validation", 3)
    test_tasks = suite.sample_function_tasks("test", 3, ood=True)
    assert_disjoint_task_ids(train_tasks, val_tasks, test_tasks)
    cfg = DNAXConfig(input_dim=1, hidden_dims=[24, 24], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False, seed=seed, inner_lr=0.01, inner_steps=1, use_second_order=False, replay_buffer_size=32)
    model = DeepNeuralAutoExplorer(cfg)

    print("[1] train learned task encoder on procedural support/query tasks")
    encoder = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=seed)
    decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
    result = train_task_encoder(encoder, decoder, train_tasks, val_tasks, steps=15, lr=0.01)
    print(f"  encoder_loss initial={result.train_losses[0]:.4f} final={result.train_losses[-1]:.4f} val={result.validation_metrics['reconstruction_loss']:.4f}")

    print("[2] generate z_task from support only and retrieve memory")
    memory = EpisodicMemory(capacity=16)
    for task in train_tasks:
        z = encoder.infer(task.support_x, task.support_y)
        before = float(F.mse_loss(model(task.query_x), task.query_y).detach())
        after = _adapt_loss(model, task, cfg.inner_lr)
        memory.add(MemoryRecord(task.task_id, task.support_x.flatten()[:8], torch.tensor([cfg.inner_lr]), before - after, after, z, {"query_loss_after": after}, split="train"))
    probe = val_tasks[0]
    z_task = encoder.infer(probe.support_x, probe.support_y)
    retrieved = memory.retrieve(z_task, k=2)
    print("  z_task_shape=", tuple(z_task.shape), "retrieved=", [(r.task_id, round(score, 3)) for r, score in retrieved])

    print("[3] memory-conditioned adaptation")
    core = CognitiveCore(model, cfg, WorldModel(8, 6, seed=seed), memory=memory, task_inference=encoder)
    mem_log = core.adapt_memory_conditioned(probe, memory_k=2, use_memory=True)
    print(f"  conditioned_lr={mem_log['conditioned_inner_lr']:.5f} after={mem_log['query_loss_after']:.4f}")

    print("[4] world-model prediction and model-based update selection")
    world_model = WorldModel(8, 6, hidden_dim=32, seed=seed)
    controller = ModelBasedController(world_model, default_update_candidates(cfg.inner_lr), latent_dim=8)
    # A few tiny synthetic transition steps so predictions are learned, not analytic.
    state = torch.randn(24, 8) * 0.1
    action = torch.stack([default_update_candidates(cfg.inner_lr)[i % 3].vector() for i in range(24)])
    next_state = state.clone(); next_state[:, 0] = 0.02 + 0.4 * action[:, 0] + 0.01 * action[:, 1]
    initial, final = controller.train_on_transition_batch((state, action, next_state), steps=35, lr=0.02)
    decision = controller.select_action(z_task, memory, memory_k=2)
    selected_loss = _adapt_loss(model, probe, decision.selected.inner_lr, decision.selected.inner_steps)
    actual_improvement = float(F.mse_loss(model(probe.query_x), probe.query_y).detach()) - selected_loss
    pred_log = controller.log_actual_outcome(decision, actual_improvement)
    print(f"  wm_loss initial={initial:.4f} final={final:.4f}")
    print("  selected=", decision.selected.name, "predicted=", round(decision.predicted_improvement, 4), "actual=", round(actual_improvement, 4), "error=", round(pred_log["prediction_error"], 4))

    print("[5] validation-gated operator mutation with rollback if rejected")
    mutation = OperatorMutationController({"inner_lr": cfg.inner_lr, "inner_steps": 1, "memory_retrieval_k": 2, "planner_horizon": 1, "first_order": True}, random_seed=seed)
    def evaluator(state_dict, tasks):
        return [_adapt_loss(model, t, float(state_dict.get("inner_lr", cfg.inner_lr)), int(state_dict.get("inner_steps", 1))) for t in tasks]
    # Try an intentionally bad candidate to transparently demonstrate rollback.
    bad = allowed_operator_mutations(mutation.state)[1]
    bad = type(bad)("too_large_inner_lr", {"inner_lr": 5.0}, bad.operator_type)
    decision_log = mutation.evaluate(bad, val_tasks, evaluator, split="validation")
    print("  mutation=", json.dumps(decision_log, indent=2))

    print("[6] write manifest")
    metrics = {"demo_validation_loss": selected_loss, "prediction_error": pred_log["prediction_error"], "rollback_count": mutation.rollback_count}
    manifest = build_manifest("python examples/next_closed_loop_demo.py", [seed], train_tasks, val_tasks, test_tasks, {"seed": seed, "demo": True}, metrics, mutation.log)
    path = write_manifest(manifest, "results/next_closed_loop_demo_manifest.json")
    print(f"  manifest={path}")
    print("[done] support -> z_task -> memory -> adaptation -> world-model controller -> validation mutation -> manifest")


if __name__ == "__main__":
    main()
