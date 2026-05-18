"""Multi-domain deep neural architecture exploration benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_search import ArchitectureGenome
from neural_search.multidomain_evaluator import MultiDomainEvaluator
from neural_search.mutation_operators import mutate_architecture
from neural_search.task_families import assert_disjoint_multidomain_splits, build_multidomain_task_families, task_family_ids


def _mode_config(mode: str) -> dict[str, int | float]:
    return {
        "smoke": {"samples_per_split": 12, "train_steps": 6, "candidate_count": 2, "hidden_dim": 10},
        "quick": {"samples_per_split": 18, "train_steps": 10, "candidate_count": 4, "hidden_dim": 12},
    }[mode]


def run(mode: str, seed: int, output: str | None = None) -> dict[str, object]:
    cfg = _mode_config(mode)
    task_families = build_multidomain_task_families(seed=seed, samples_per_split=int(cfg["samples_per_split"]))
    assert_disjoint_multidomain_splits(task_families)
    evaluator = MultiDomainEvaluator(seed=seed, train_steps=int(cfg["train_steps"]), lr=0.018)
    base = ArchitectureGenome(input_dim=4, output_dim=2, hidden_dim=int(cfg["hidden_dim"]), residual_blocks=1, seed=seed)
    base_eval = evaluator.train_and_validate(base, task_families)
    methods = ["add_residual_block", "widen_hidden_dim", "add_memory_gate", "add_causal_bottleneck"][: int(cfg["candidate_count"])]
    decisions = []
    best_genome = base
    best_eval = base_eval
    for index, method in enumerate(methods):
        candidate = mutate_architecture(best_genome, method, seed + index + 1)
        candidate_eval = evaluator.train_and_validate(candidate, task_families)
        decision = evaluator.selection_decision(best_genome, candidate, best_eval, candidate_eval)
        decisions.append(decision.to_dict())
        if decision.accepted:
            best_genome = candidate
            best_eval = candidate_eval
    frozen_heldout = evaluator.evaluate_heldout_after_freeze(best_eval.model, best_genome, task_families)
    accepted = [item for item in decisions if item["accepted"]]
    metrics = {
        **frozen_heldout.to_dict(),
        "adaptation_speed": best_eval.adaptation_speed,
        "train_steps": best_eval.train_steps,
        "initial_train_loss": best_eval.initial_train_loss,
        "final_train_loss": best_eval.final_train_loss,
        "candidate_acceptance_count": float(len(accepted)),
        "candidate_rejection_count": float(len(decisions) - len(accepted)),
    }
    result = {
        "benchmark": "deep_neural_exploration_multidomain",
        "mode": mode,
        "seed": seed,
        "config": cfg,
        "task_ids": task_family_ids(task_families),
        "base_genome": base.to_dict(),
        "final_genome": best_genome.to_dict(),
        "candidate_decisions": decisions,
        "metrics": metrics,
    }
    if output is None:
        output = f"results/deep_neural_exploration_{mode}_seed{seed}.json"
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-domain neural exploration benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/deep_neural_exploration_{args.mode}_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
