"""CPU-runnable AGI-scaffold benchmark with quick/full modes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from cognitive_core import CognitiveCore, EpisodicMemory, MemoryRecord, Planner, TaskInferenceModule, WorldModel, generate_linear_dynamics_batch
from maml_functional import functional_forward, maml_inner_loop
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite, assert_disjoint_task_ids, function_task_variance


def _stderr(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _evaluate_seed(seed: int, mode: str) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    suite = ProceduralTaskSuite(seed=seed)
    n_train = 5 if mode == "quick" else 15
    n_eval = 5 if mode == "quick" else 15
    train_tasks = suite.sample_function_tasks("train", n_train, ood=False)
    val_tasks = suite.sample_function_tasks("validation", max(3, n_eval // 2), ood=False)
    test_tasks = suite.sample_function_tasks("test", n_eval, ood=True)
    assert_disjoint_task_ids(train_tasks, val_tasks, test_tasks)

    cfg = DNAXConfig(input_dim=1, hidden_dims=[24, 24], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False, seed=seed, inner_lr=0.01, inner_steps=1, use_second_order=False, replay_buffer_size=64)
    model = DeepNeuralAutoExplorer(cfg)
    core = CognitiveCore(model, cfg, WorldModel(2, 2, hidden_dim=32, seed=seed))

    train_improvements = []
    val_losses = []
    test_losses = []
    baseline_test_losses = []
    embeddings_by_family = {}
    inference_proxy = []
    for task in train_tasks:
        assert function_task_variance(task) > 1e-5
        log = core.adapt_and_evaluate(task)
        train_improvements.append(float(log["adaptation_improvement"]))
        embeddings_by_family.setdefault(task.family, []).append(log["embedding"])
    for task in val_tasks:
        log = core.adapt_and_evaluate(task)
        val_losses.append(float(log["query_loss_after"]))
    for task in test_tasks:
        params = dict(model.named_parameters())
        with torch.no_grad():
            baseline_test_losses.append(float(F.mse_loss(model(task.query_x), task.query_y)))
        adapted = maml_inner_loop(model, params, task.support_x, task.support_y, cfg.inner_lr, cfg.inner_steps, first_order=True)
        with torch.no_grad():
            test_losses.append(float(F.mse_loss(functional_forward(model, adapted, task.query_x), task.query_y)))
        emb = core.task_inference.infer(task.support_x, task.support_y)
        same_family = embeddings_by_family.get(task.family)
        if same_family:
            center = torch.stack(same_family).mean(dim=0)
            inference_proxy.append(float(F.mse_loss(emb, center)))

    # World model prediction error and learning progress.
    wm = WorldModel(2, 2, hidden_dim=32, seed=seed)
    opt = torch.optim.Adam(wm.parameters(), lr=0.03)
    batch = generate_linear_dynamics_batch(n=64 if mode == "quick" else 128, seed=seed)
    initial_wm = float(wm.prediction_loss(*batch).detach())
    steps = 40 if mode == "quick" else 120
    for _ in range(steps):
        wm.train_step(batch, opt)
    final_wm = float(wm.prediction_loss(*batch).detach())

    # Memory retrieval precision on procedural associations.
    memory = EpisodicMemory(capacity=16)
    infer = TaskInferenceModule()
    correct = 0
    for idx, task in enumerate(train_tasks[:5]):
        emb = infer.infer(task.support_x, task.support_y)
        memory.add(MemoryRecord(task.task_id, task.support_x.flatten()[:8], torch.tensor([0.0]), 1.0, 0.0, emb, {"idx": float(idx)}))
    for task in train_tasks[:5]:
        emb = infer.infer(task.support_x, task.support_y)
        hit = memory.retrieve(emb, k=1)[0][0]
        correct += int(hit.task_id == task.task_id)
    memory_precision = correct / max(1, min(5, len(train_tasks)))

    # Planning success over learned/analytic latent transitions.
    class AdditiveWorldModel(WorldModel):
        def forward(self, latent_state, action):
            return latent_state + action
    planner = Planner(AdditiveWorldModel(2, 2), [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])], horizon=2)
    _, plan_score = planner.plan(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    planning_success = 1.0 if plan_score < 1e-8 else 0.0

    candidate = core.self_improvement.propose_candidates()[0]
    si_entry = core.self_improvement.evaluate_and_update(candidate, val_tasks[:3], core.validation_evaluator(val_tasks[:3]))

    return {
        "adaptation_improvement_after_k": float(np.mean(train_improvements)),
        "ood_generalization_gap": float(np.mean(test_losses) - np.mean(val_losses)),
        "task_inference_proxy_loss": float(np.mean(inference_proxy)) if inference_proxy else 0.0,
        "world_model_prediction_error": final_wm,
        "world_model_error_reduction": initial_wm - final_wm,
        "memory_retrieval_precision": float(memory_precision),
        "planning_success_rate": planning_success,
        "exploration_efficiency": float(np.mean(train_improvements) / (cfg.inner_steps + 1e-8)),
        "self_improvement_acceptance_rate": 1.0 if si_entry["accepted"] else 0.0,
        "rollback_count": float(core.self_improvement.rollback_count),
        "validation_vs_test_gap": float(np.mean(test_losses) - np.mean(val_losses)),
        "baseline_to_adapted_test_delta": float(np.mean(baseline_test_losses) - np.mean(test_losses)),
    }


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    seeds = [seed] if mode == "quick" else [seed, seed + 1, seed + 2]
    per_seed = [_evaluate_seed(s, mode) for s in seeds]
    keys = per_seed[0].keys()
    aggregate = {key: {"mean": mean([r[key] for r in per_seed]), "stderr": _stderr([r[key] for r in per_seed])} for key in keys}
    result = {"mode": mode, "seeds": seeds, "per_seed": per_seed, "aggregate": aggregate}
    if output is None:
        output = f"results/agi_scaffold_{mode}_seed{seed}.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AGI-oriented closed-loop scaffold benchmark")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={args.output or f'results/agi_scaffold_{args.mode}_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
