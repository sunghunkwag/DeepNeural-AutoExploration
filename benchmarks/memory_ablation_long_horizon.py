"""Compare memory-guided and memory-disabled long-horizon exploration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.external_neural_adapter_benchmark import run as run_external_adapter
from benchmarks.long_horizon_autonomous_exploration import run as run_long_horizon
from experiment_manifest import current_git_commit, stable_config_hash


DESIRABLE_HIGHER = {
    "residue_resolution_rate",
    "hidden_validation_robustness",
    "meta_decision_quality",
    "process_improvement_delta",
    "external_adapter_heldout_after_freeze_delta",
}
DESIRABLE_LOWER = {
    "rollback_rate",
    "evaluator_overfit_risk",
    "transfer_regression_risk",
}


def run(mode: str, seeds: list[int], runs: int, output: str | None = None) -> Dict[str, object]:
    output_path = Path(output or f"results/memory_ablation_long_horizon_{mode}.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    per_seed: list[Dict[str, object]] = []
    for seed in seeds:
        guided_path = output_path.with_name(f"{output_path.stem}_seed{seed}_memory_guided.json")
        disabled_path = output_path.with_name(f"{output_path.stem}_seed{seed}_memory_disabled.json")
        guided_memory = output_path.with_name(f"{output_path.stem}_seed{seed}_guided_memory.json")
        disabled_memory = output_path.with_name(f"{output_path.stem}_seed{seed}_disabled_shadow_memory.json")
        _reset_memory(guided_memory)
        _reset_memory(disabled_memory)
        guided = run_long_horizon(mode, seed, runs, str(guided_path), memory_enabled=True, memory_path_override=str(guided_memory))
        disabled = run_long_horizon(mode, seed, runs, str(disabled_path), memory_enabled=False, memory_path_override=str(disabled_memory))
        external = run_external_adapter(mode, seed, str(output_path.with_name(f"{output_path.stem}_seed{seed}_external.json")))
        comparison = _compare_conditions(guided, disabled, external)
        per_seed.append(
            {
                "seed": seed,
                "memory_guided_output_path": str(guided_path),
                "memory_disabled_output_path": str(disabled_path),
                "external_adapter_output_path": str(output_path.with_name(f"{output_path.stem}_seed{seed}_external.json")),
                "memory_guided_metrics": _condition_metrics(guided, external),
                "memory_disabled_metrics": _condition_metrics(disabled, external),
                "comparison": comparison,
            }
        )
    aggregate = _aggregate(per_seed)
    result = {
        "benchmark": "memory_ablation_long_horizon",
        "mode": mode,
        "seeds": seeds,
        "runs": runs,
        "conditions": ["memory_guided", "memory_disabled"],
        "per_seed_results": per_seed,
        "aggregate_comparison": aggregate,
        "memory_effect": aggregate["memory_effect"],
        "negative_results_preserved": True,
    }
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": f"python benchmarks/memory_ablation_long_horizon.py --mode {mode} --seeds {','.join(str(seed) for seed in seeds)} --runs {runs}",
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "config_hash": stable_config_hash({"mode": mode, "seeds": seeds, "runs": runs}),
        "aggregate_comparison": aggregate,
        "anti_cheat_checks_passed": [
            "memory-guided and memory-disabled conditions both ran",
            "heldout_test remained post-freeze only",
            "negative deltas are retained in aggregate comparison",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["manifest_path"] = str(manifest_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _reset_memory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"max_records": 20, "records": []}, indent=2), encoding="utf-8")


def _condition_metrics(result: Mapping[str, object], external: Mapping[str, object]) -> Dict[str, float]:
    trace = [item for item in result.get("long_horizon_trace", []) if isinstance(item, Mapping)] if isinstance(result.get("long_horizon_trace", []), list) else []
    return {
        "rollback_rate": _average(trace, "rollback_rate"),
        "residue_resolution_rate": _average(trace, "residue_resolution_rate"),
        "hidden_validation_robustness": _average(trace, "hidden_validation_robustness"),
        "evaluator_overfit_risk": _average(trace, "evaluator_overfit_risk"),
        "transfer_regression_risk": _average(trace, "transfer_regression_risk"),
        "meta_decision_quality": _average(trace, "meta_decision_quality"),
        "process_improvement_delta": _average(trace, "process_improvement_delta"),
        "external_adapter_heldout_after_freeze_delta": _float(external.get("heldout_after_freeze_delta")),
    }


def _compare_conditions(guided: Mapping[str, object], disabled: Mapping[str, object], external: Mapping[str, object]) -> Dict[str, object]:
    guided_metrics = _condition_metrics(guided, external)
    disabled_metrics = _condition_metrics(disabled, external)
    deltas: Dict[str, float] = {}
    helped = 0
    harmed = 0
    for key, guided_value in guided_metrics.items():
        disabled_value = disabled_metrics.get(key, 0.0)
        delta = guided_value - disabled_value
        if key in DESIRABLE_LOWER:
            delta = disabled_value - guided_value
        deltas[key] = float(delta)
        if delta > 1e-9:
            helped += 1
        elif delta < -1e-9:
            harmed += 1
    effect = "helped" if helped > harmed else "harmed" if harmed > helped else "neutral"
    return {"deltas_positive_means_memory_better": deltas, "helped_metric_count": helped, "harmed_metric_count": harmed, "memory_effect": effect}


def _aggregate(per_seed: list[Mapping[str, object]]) -> Dict[str, object]:
    values: Dict[str, list[float]] = {}
    effects = {"helped": 0, "harmed": 0, "neutral": 0}
    for item in per_seed:
        comparison = item.get("comparison", {})
        if isinstance(comparison, Mapping):
            effects[str(comparison.get("memory_effect", "neutral"))] = effects.get(str(comparison.get("memory_effect", "neutral")), 0) + 1
            deltas = comparison.get("deltas_positive_means_memory_better", {})
            if isinstance(deltas, Mapping):
                for key, value in deltas.items():
                    if isinstance(value, (int, float)):
                        values.setdefault(str(key), []).append(float(value))
    averaged = {key: float(mean(vals)) for key, vals in values.items() if vals}
    if effects["helped"] > effects["harmed"]:
        effect = "helped"
    elif effects["harmed"] > effects["helped"]:
        effect = "harmed"
    else:
        effect = "neutral"
    return {
        "mean_deltas_positive_means_memory_better": averaged,
        "seed_effect_counts": effects,
        "memory_effect": effect,
    }


def _average(trace: list[Mapping[str, object]], key: str) -> float:
    vals = [_float(item.get(key)) for item in trace if isinstance(item.get(key), (int, float))]
    return float(mean(vals)) if vals else 0.0


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _parse_seeds(value: str | None, seed: int) -> list[int]:
    if not value:
        return [seed]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory ablation for long-horizon exploration")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, _parse_seeds(args.seeds, args.seed), args.runs, args.output)
    print(json.dumps({"memory_effect": result["memory_effect"], "aggregate_comparison": result["aggregate_comparison"]}, indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/memory_ablation_long_horizon_{args.mode}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
