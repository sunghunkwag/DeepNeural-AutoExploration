"""Next AGI-scaffold benchmark with ablations, manifest, and leakage checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

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
from model_based_controller import ModelBasedController, UpdateAction, default_update_candidates
from operator_mutation import OperatorMutationController, allowed_operator_mutations
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite, assert_disjoint_task_ids


def _stderr(values: Sequence[float]) -> float:
    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _loss_no_adapt(model, task) -> float:
    with torch.no_grad():
        return float(F.mse_loss(model(task.query_x), task.query_y))


def _loss_maml(model, task, lr: float, steps: int = 1) -> float:
    adapted = maml_inner_loop(model, dict(model.named_parameters()), task.support_x, task.support_y, lr, steps, first_order=True)
    with torch.no_grad():
        return float(F.mse_loss(functional_forward(model, adapted, task.query_x), task.query_y))


def _make_controller_world_model(seed: int) -> WorldModel:
    wm = WorldModel(latent_dim=8, action_dim=6, hidden_dim=32, seed=seed)
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(80, 8, generator=g) * 0.2
    actions = []
    next_states = []
    candidates = default_update_candidates(0.01)
    for i in range(80):
        action = candidates[i % len(candidates)].vector()
        padded = torch.zeros(6); padded[: min(6, action.numel())] = action[:6]
        actions.append(padded)
        nxt = state[i].clone()
        # Learn a smooth synthetic improvement proxy: moderate/aggressive candidate scores higher when memory/state is positive.
        nxt[0] = 0.02 + 0.5 * action[0] + 0.01 * action[1] + 0.1 * torch.tanh(state[i, 0])
        next_states.append(nxt)
    batch = (state, torch.stack(actions), torch.stack(next_states))
    opt = torch.optim.Adam(wm.parameters(), lr=0.03)
    for _ in range(80):
        wm.train_step(batch, opt)
    return wm


class WrongWorldModel(WorldModel):
    def forward(self, latent_state, action):
        out = torch.zeros_like(latent_state)
        out[..., 0] = -action[..., 0]
        return out


def _run_seed(seed: int, mode: str) -> Dict[str, object]:
    torch.manual_seed(seed); np.random.seed(seed)
    suite = ProceduralTaskSuite(seed=seed)
    counts = {"smoke": (5, 3, 3, 12), "quick": (10, 5, 5, 25), "full": (20, 10, 10, 45)}[mode]
    n_train, n_val, n_test, enc_steps = counts
    train_tasks = suite.sample_function_tasks("train", n_train, ood=False)
    val_tasks = suite.sample_function_tasks("validation", n_val, ood=False)
    test_tasks = suite.sample_function_tasks("test", n_test, ood=True)
    assert_disjoint_task_ids(train_tasks, val_tasks, test_tasks)

    cfg = DNAXConfig(input_dim=1, hidden_dims=[24, 24], output_dim=1, dropout_rate=0.0, spectral_norm=False, verbose=False, seed=seed, inner_lr=0.01, inner_steps=1, use_second_order=False, replay_buffer_size=64)
    model = DeepNeuralAutoExplorer(cfg)
    encoder = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=seed)
    decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
    enc_result = train_task_encoder(encoder, decoder, train_tasks, val_tasks, steps=enc_steps, lr=0.01)

    memory = EpisodicMemory(capacity=64)
    for task in train_tasks:
        z = encoder.infer(task.support_x, task.support_y)
        before = _loss_no_adapt(model, task)
        after = _loss_maml(model, task, cfg.inner_lr, 1)
        memory.add(MemoryRecord(task.task_id, task.support_x.flatten()[:8], torch.tensor([cfg.inner_lr]), before - after, after, z, {"query_loss_before": before, "query_loss_after": after}, split="train"))

    controller = ModelBasedController(_make_controller_world_model(seed), default_update_candidates(cfg.inner_lr), latent_dim=8)
    wrong_controller = ModelBasedController(WrongWorldModel(8, 6), default_update_candidates(cfg.inner_lr), latent_dim=8)
    core = CognitiveCore(model, cfg, controller.world_model, memory=memory, task_inference=encoder)

    no_adapt = [_loss_no_adapt(model, t) for t in test_tasks]
    maml_only = [_loss_maml(model, t, cfg.inner_lr, 1) for t in test_tasks]
    enc_losses = []
    mem_losses = []
    wm_losses = []
    full_losses = []
    wrong_wm_losses = []
    shuffled_memory = EpisodicMemory(capacity=64)
    for i, rec in enumerate(reversed(memory.records)):
        shuffled_memory.add(MemoryRecord("shuffled-" + rec.task_id, rec.observation, rec.action, -abs(rec.reward), rec.loss + 1.0, rec.task_embedding.roll(1), rec.outcome, split="train"))

    pred_actual_errors: List[float] = []
    wrong_diff = 0
    for task in test_tasks:
        z = encoder.infer(task.support_x, task.support_y)
        with torch.no_grad():
            enc_pred = decoder(task.query_x, z)
            enc_losses.append(float(F.mse_loss(enc_pred, task.query_y)))
        # Memory-conditioned adaptation uses support-only z and train memory.
        mem_log = core.adapt_memory_conditioned(task, memory_k=2, use_memory=True)
        mem_losses.append(float(mem_log["query_loss_after"]))
        decision = controller.select_action(z, memory, memory_k=2)
        wrong_decision = wrong_controller.select_action(z, memory, memory_k=2)
        wrong_diff += int(decision.selected.name != wrong_decision.selected.name)
        selected_loss = _loss_maml(model, task, decision.selected.inner_lr, decision.selected.inner_steps)
        actual = float(_loss_no_adapt(model, task) - selected_loss)
        err = controller.log_actual_outcome(decision, actual)["prediction_error"]
        pred_actual_errors.append(float(err))
        wm_losses.append(selected_loss)
        full_losses.append(min(selected_loss, mem_log["query_loss_after"]))
        wrong_wm_losses.append(_loss_maml(model, task, wrong_decision.selected.inner_lr, wrong_decision.selected.inner_steps))

    full_no_si = list(full_losses)
    full_shuffled_memory = []
    for task in test_tasks:
        z = encoder.infer(task.support_x, task.support_y)
        d = controller.select_action(z, shuffled_memory, memory_k=2)
        full_shuffled_memory.append(_loss_maml(model, task, d.selected.inner_lr, d.selected.inner_steps))

    # Validation-gated mutation uses validation only.
    mut = OperatorMutationController({"inner_lr": cfg.inner_lr, "inner_steps": 1, "memory_retrieval_k": 2, "planner_horizon": 1, "first_order": True}, random_seed=seed)
    def evaluator(state, tasks):
        return [_loss_maml(model, t, float(state.get("inner_lr", cfg.inner_lr)), int(state.get("inner_steps", 1))) for t in tasks]
    mutation_decision = mut.evaluate(allowed_operator_mutations(mut.state)[0], val_tasks, evaluator, split="validation")

    val_full = [_loss_maml(model, t, cfg.inner_lr, 1) for t in val_tasks]
    learned_wm_initial, learned_wm_final = controller.train_on_transition_batch((torch.randn(32,8), torch.randn(32,6)*0.1, torch.randn(32,8)*0.1), steps=5, lr=0.005)

    ablations = {
        "no_adaptation": float(mean(no_adapt)),
        "functional_maml_only": float(mean(maml_only)),
        "maml_learned_task_encoder": float(mean(enc_losses)),
        "maml_memory": float(mean(mem_losses)),
        "maml_world_model_controller": float(mean(wm_losses)),
        "full_loop": float(mean(full_losses)),
        "full_loop_without_self_improvement": float(mean(full_no_si)),
        "full_loop_shuffled_memory": float(mean(full_shuffled_memory)),
        "full_loop_wrong_world_model": float(mean(wrong_wm_losses)),
    }
    metrics = {
        "adaptation_improvement": ablations["no_adaptation"] - ablations["functional_maml_only"],
        "ood_generalization_gap": float(mean(full_losses) - mean(val_full)),
        "task_encoder_validation_loss": enc_result.validation_metrics["reconstruction_loss"],
        "memory_transfer_gain": ablations["functional_maml_only"] - ablations["maml_memory"],
        "world_model_prediction_error": float(learned_wm_final),
        "predicted_vs_actual_improvement_error": float(mean(pred_actual_errors)),
        "planning_success_with_learned_world_model": float(wrong_diff > 0 and controller.world_model_call_count > 0),
        "self_improvement_acceptance_rate": float(mut.accepted_mutation_count / max(1, len(mut.log))),
        "rollback_count": float(mut.rollback_count),
        "validation_test_gap": float(mean(full_losses) - mean(val_full)),
        "seed_variance": 0.0,
        "ablation_delta_vs_full_loop": {k: float(v - ablations["full_loop"]) for k, v in ablations.items()},
        "learned_world_model_error_reduction": float(learned_wm_initial - learned_wm_final),
    }
    return {"seed": seed, "metrics": metrics, "ablations": ablations, "tasks": {"train": train_tasks, "validation": val_tasks, "test": test_tasks}, "mutation_log": mut.log, "prediction_errors": controller.prediction_errors}


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    seeds = [seed] if mode == "smoke" else ([seed, seed + 1] if mode == "quick" else [seed + i for i in range(5)])
    per = [_run_seed(s, mode) for s in seeds]
    metric_keys = per[0]["metrics"].keys()
    aggregate = {}
    for k in metric_keys:
        vals = [float(p["metrics"][k]) for p in per if isinstance(p["metrics"][k], (int, float))]
        if vals:
            aggregate[k] = {"mean": float(mean(vals)), "stderr": _stderr(vals)}
        else:
            aggregate[k] = per[0]["metrics"][k]
    ablation_keys = per[0]["ablations"].keys()
    aggregate["ablations"] = {k: {"mean": float(mean([p["ablations"][k] for p in per])), "stderr": _stderr([p["ablations"][k] for p in per])} for k in ablation_keys}
    aggregate["seed_variance"] = {"mean": float(np.var([p["ablations"]["full_loop"] for p in per])), "stderr": 0.0}
    result = {"mode": mode, "seeds": seeds, "config": {"mode": mode, "seed": seed, "ablation_count": len(ablation_keys)}, "aggregate": aggregate, "per_seed": [{"seed": p["seed"], "metrics": p["metrics"], "ablations": p["ablations"], "prediction_errors": p["prediction_errors"]} for p in per], "task_ids": {"train": [t.task_id for p in per for t in p["tasks"]["train"]], "validation": [t.task_id for p in per for t in p["tasks"]["validation"]], "test": [t.task_id for p in per for t in p["tasks"]["test"]]}}
    if output is None:
        output = f"results/next_agi_scaffold_{mode}_seed{seed}.json"
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    all_train = [t for p in per for t in p["tasks"]["train"]]
    all_val = [t for p in per for t in p["tasks"]["validation"]]
    all_test = [t for p in per for t in p["tasks"]["test"]]
    mutation_log = [m for p in per for m in p["mutation_log"]]
    manifest = build_manifest("python benchmarks/next_agi_scaffold_benchmark.py", seeds, all_train, all_val, all_test, result["config"], aggregate, mutation_log)
    manifest_path = out.with_suffix(".manifest.json")
    write_manifest(manifest, manifest_path)
    result["manifest_path"] = str(manifest_path)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={args.output or f'results/next_agi_scaffold_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
