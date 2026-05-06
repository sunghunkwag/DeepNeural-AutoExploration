"""Minimal transparent closed-loop demo for the AGI-oriented scaffold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cognitive_core import CognitiveCore, Planner, WorldModel, generate_linear_dynamics_batch
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    suite = ProceduralTaskSuite(seed=seed)
    cfg = DNAXConfig(input_dim=1, hidden_dims=[24, 24], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False, inner_lr=0.01, inner_steps=1, use_second_order=False, replay_buffer_size=32)
    model = DeepNeuralAutoExplorer(cfg)
    world_model = WorldModel(latent_dim=2, action_dim=2, hidden_dim=32, seed=seed)
    core = CognitiveCore(model, cfg, world_model)

    print("[1] sample task sequence and adapt with functional MAML")
    train_tasks = suite.sample_function_tasks("train", 3, ood=False)
    for task in train_tasks:
        log = core.adapt_and_evaluate(task)
        print(f"  task={task.task_id} family={task.family} before={log['query_loss_before']:.4f} after={log['query_loss_after']:.4f} improvement={log['adaptation_improvement']:.4f}")

    print("[2] retrieve similar episodic memory for a new context")
    probe = suite.sample_function_task("sinusoid", split="validation")
    emb = core.task_inference.infer(probe.support_x, probe.support_y)
    retrieved = core.memory.retrieve(emb, k=2)
    print("  retrieved=", [(r.task_id, round(score, 3)) for r, score in retrieved])

    print("[3] train world model on synthetic latent dynamics")
    batch = generate_linear_dynamics_batch(n=64, seed=seed)
    opt = torch.optim.Adam(world_model.parameters(), lr=0.03)
    initial = float(world_model.prediction_loss(*batch).detach())
    for _ in range(60):
        final = world_model.train_step(batch, opt)
    print(f"  world_model_loss initial={initial:.4f} final={final:.4f}")

    print("[4] plan over a tiny latent transition problem")
    class AdditiveWorldModel(WorldModel):
        def forward(self, latent_state, action):
            return latent_state + action
    planner = Planner(AdditiveWorldModel(2, 2), [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])], horizon=2)
    actions, score = planner.plan(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    print("  planned_actions=", [a.tolist() for a in actions], "goal_mse=", round(score, 6))

    print("[5] propose validation-gated self-improvement candidate")
    val_tasks = suite.sample_function_tasks("validation", 3, ood=False)
    candidate = core.self_improvement.propose_candidates()[0]
    decision = core.self_improvement.evaluate_and_update(candidate, val_tasks, core.validation_evaluator(val_tasks))
    print("  decision=", json.dumps(decision, indent=2))
    print("[done] closed-loop scaffold executed without using query targets in memory or final-test tuning")


if __name__ == "__main__":
    main()
