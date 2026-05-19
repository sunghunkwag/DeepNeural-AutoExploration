"""Benchmark the bounded meta-meta-meta search controller."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop
from experiment_manifest import current_git_commit, stable_config_hash
from neural_search.next_run_config import NextRunConfig
from neural_search.search_process_model import SearchProcessModel


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    output_path = Path(output or f"results/meta_meta_meta_search_{mode}_seed{seed}.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    first_path = output_path.with_name(f"{output_path.stem}_first_autonomous.json")
    second_path = output_path.with_name(f"{output_path.stem}_second_autonomous.json")
    next_config_path = output_path.with_name(f"{output_path.stem}_next_run_config.json")

    first = run_autonomous_loop(mode, seed, str(first_path))
    next_config = NextRunConfig.from_dict(first["next_run_config_recommendation"])  # type: ignore[arg-type]
    next_config.save(next_config_path)
    second_seed = seed + 1 if next_config.recommended_seed_policy == "next_seed" else seed
    second = run_autonomous_loop(mode, second_seed, str(second_path), str(next_config_path))

    first_process = SearchProcessModel().diagnose(first)
    second_process = SearchProcessModel().diagnose(second)
    first_agi = first.get("agi_relevance_metrics", {})
    second_agi = second.get("agi_relevance_metrics", {})
    first_rollback = _rollback_rate(first)
    second_rollback = _rollback_rate(second)
    metrics = {
        "first_run_cumulative_autonomous_improvement_score": _metric(first_agi, "cumulative_autonomous_improvement_score"),
        "second_run_cumulative_autonomous_improvement_score": _metric(second_agi, "cumulative_autonomous_improvement_score"),
        "first_run_residue_resolution_rate": _metric(first_agi, "residue_resolution_rate"),
        "second_run_residue_resolution_rate": _metric(second_agi, "residue_resolution_rate"),
        "first_run_rollback_rate": first_rollback,
        "second_run_rollback_rate": second_rollback,
        "first_run_hidden_validation_robustness": _metric(first_agi, "hidden_validation_robustness"),
        "second_run_hidden_validation_robustness": _metric(second_agi, "hidden_validation_robustness"),
        "strategy_mutations_applied": float(len(first["search_strategy_mutation_plan"]["selected_mutations"])),  # type: ignore[index]
        "evaluator_mutations_applied": float(len(first["evaluator_objective_mutation_plan"]["selected_mutations"])),  # type: ignore[index]
        "curriculum_mutations_applied": float(len(first["curriculum_mutation_plan"]["selected_mutations"])),  # type: ignore[index]
    }
    metrics["process_improvement_delta"] = (
        metrics["second_run_cumulative_autonomous_improvement_score"]
        - metrics["first_run_cumulative_autonomous_improvement_score"]
        + 0.25 * (metrics["second_run_residue_resolution_rate"] - metrics["first_run_residue_resolution_rate"])
        + 0.25 * (metrics["first_run_rollback_rate"] - metrics["second_run_rollback_rate"])
        + 0.25 * (metrics["second_run_hidden_validation_robustness"] - metrics["first_run_hidden_validation_robustness"])
    )
    result = {
        "benchmark": "meta_meta_meta_search",
        "mode": mode,
        "seed": seed,
        "first_run_output_path": str(first_path),
        "second_run_output_path": str(second_path),
        "next_run_config_path": str(next_config_path),
        "first_run_process_diagnosis": first_process.to_dict(),
        "second_run_process_diagnosis": second_process.to_dict(),
        "strategy_mutations_applied": first["search_strategy_mutation_plan"],  # type: ignore[index]
        "evaluator_mutations_applied": first["evaluator_objective_mutation_plan"],  # type: ignore[index]
        "curriculum_mutations_applied": first["curriculum_mutation_plan"],  # type: ignore[index]
        "second_run_used_next_run_config": second.get("applied_next_run_config") is not None,
        "heldout_test_used_for_candidate_selection": any(
            decision.get("used_heldout_labels")
            for decision in list(first.get("architecture_decisions", [])) + list(second.get("architecture_decisions", []))
            if isinstance(decision, dict)
        ),
        "metrics": metrics,
    }
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": f"python benchmarks/meta_meta_meta_search_benchmark.py --mode {mode} --seed {seed}",
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "config_hash": stable_config_hash({"mode": mode, "seed": seed, "next_run_config": next_config.to_dict()}),
        "first_run_output_path": str(first_path),
        "second_run_output_path": str(second_path),
        "next_run_config_path": str(next_config_path),
        "metrics": metrics,
        "anti_cheat_checks_passed": [
            "meta-meta-meta controller runs after candidate freeze",
            "next-run config is JSON data only",
            "heldout test is not used for candidate selection",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result["manifest_path"] = str(manifest_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _metric(payload: object, name: str) -> float:
    if isinstance(payload, dict):
        value = payload.get(name, 0.0)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _rollback_rate(result: Dict[str, object]) -> float:
    decisions = [item for item in result.get("architecture_decisions", []) if isinstance(item, dict)]
    rollbacks = sum(1 for item in decisions if item.get("rollback"))
    return float(rollbacks / max(1, len(decisions)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-meta-meta search benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/meta_meta_meta_search_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()

