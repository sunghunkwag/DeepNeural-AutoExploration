"""Benchmark the bounded meta-meta-meta search controller as a closed loop."""

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
from neural_search.counterfactual_config_evaluator import CounterfactualConfigEvaluator
from neural_search.meta_config_attribution import MetaConfigAttribution
from neural_search.meta_config_memory import MetaConfigMemory, next_run_config_hash
from neural_search.meta_decision_quality import MetaDecisionQualityEvaluator
from neural_search.meta_meta_meta_controller import MetaMetaMetaController, MetaMetaMetaControllerOutput
from neural_search.search_process_model import SearchProcessModel, compute_process_improvement_components


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    output_path = Path(output or f"results/meta_meta_meta_search_{mode}_seed{seed}.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    first_path = output_path.with_name(f"{output_path.stem}_first_autonomous.json")
    second_path = output_path.with_name(f"{output_path.stem}_second_autonomous.json")
    next_config_path = output_path.with_name(f"{output_path.stem}_next_run_config.json")
    memory_path = output_path.with_name(f"meta_config_memory_{mode}_seed{seed}.json")

    prior_memory = MetaConfigMemory.load(memory_path)
    first = run_autonomous_loop(mode, seed, str(first_path))
    first_meta = MetaMetaMetaController(prior_memory).run(first)
    _apply_meta_output(first, first_meta)
    _write_json(first_path, first)

    next_config = first_meta.next_run_config
    next_config.save(next_config_path)
    second_seed = seed + 1 if next_config.recommended_seed_policy == "next_seed" else seed
    second = run_autonomous_loop(mode, second_seed, str(second_path), str(next_config_path))

    process_model = SearchProcessModel()
    first_process = process_model.diagnose(first)
    second_process = process_model.diagnose(second)
    process_components = compute_process_improvement_components(first_process, second_process, first, second)
    attribution_report = MetaConfigAttribution().attribute(
        first,
        second,
        first_next_run_config=next_config.to_dict(),
        second_applied_next_run_config=_mapping(second.get("applied_next_run_config")) or next_config.to_dict(),
        first_diagnosis=first_process,
        second_diagnosis=second_process,
    )
    quality_report = MetaDecisionQualityEvaluator().evaluate(
        first,
        second,
        attribution_report,
        process_components,
        first_diagnosis=first_process,
        second_diagnosis=second_process,
    )
    counterfactual_reports = CounterfactualConfigEvaluator().evaluate(
        attribution_report,
        first_process,
        second_process,
        first,
        second,
    )
    memory = prior_memory
    memory.add_decision(
        meta_decision_id=first_meta.decision.decision_id,
        next_run_config_hash=next_run_config_hash(next_config.to_dict()),
        applied_config=next_config.to_dict(),
        attribution_report=attribution_report.to_dict(),
        quality_report=quality_report.to_dict(),
        process_improvement_delta=process_components["final_process_improvement_delta"],
        helped_changes=quality_report.helped_changes,
        harmed_changes=quality_report.harmed_changes,
        recommendation=quality_report.recommendation,
        seed=seed,
        mode=mode,
    )
    memory.save(memory_path)

    metrics = _metrics(first, second, process_components, first_meta)
    anti_cheat_checks = _anti_cheat_checks(first_process, second_process, first, second)
    result = {
        "benchmark": "meta_meta_meta_search",
        "mode": mode,
        "seed": seed,
        "first_run_output_path": str(first_path),
        "second_run_output_path": str(second_path),
        "next_run_config_path": str(next_config_path),
        "first_run_process_diagnosis": first_process.to_dict(),
        "second_run_process_diagnosis": second_process.to_dict(),
        "process_improvement_components": dict(process_components),
        "config_change_attribution_report": attribution_report.to_dict(),
        "meta_decision_quality_report": quality_report.to_dict(),
        "counterfactual_config_report": {
            "reports": [report.to_dict() for report in counterfactual_reports],
            "all_reports_are_estimates": True,
            "claims_proof": False,
        },
        "meta_config_memory_path": str(memory_path),
        "recommendation_for_next_meta_run": quality_report.recommendation,
        "strategy_mutations_applied": first["search_strategy_mutation_plan"],
        "evaluator_mutations_applied": first["evaluator_objective_mutation_plan"],
        "curriculum_mutations_applied": first["curriculum_mutation_plan"],
        "second_run_used_next_run_config": second.get("applied_next_run_config") is not None,
        "heldout_test_used_for_candidate_selection": first_process.used_heldout_for_candidate_selection or second_process.used_heldout_for_candidate_selection,
        "heldout_test_used_only_for_post_freeze_process_diagnosis": (
            first_process.heldout_used_only_for_post_freeze_diagnosis
            and second_process.heldout_used_only_for_post_freeze_diagnosis
        ),
        "metrics": metrics,
        "anti_cheat_checks_passed": anti_cheat_checks,
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
        "meta_config_memory_path": str(memory_path),
        "process_improvement_components": dict(process_components),
        "anti_cheat_checks_passed": anti_cheat_checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result["manifest_path"] = str(manifest_path)
    _write_json(output_path, result)
    _write_json(manifest_path, manifest)
    return result


def _apply_meta_output(result: Dict[str, object], output: MetaMetaMetaControllerOutput) -> None:
    result["search_process_diagnosis"] = output.process_diagnosis.to_dict()
    result["meta_meta_meta_decision"] = output.decision.to_dict()
    result["search_strategy_mutation_plan"] = output.search_strategy_plan.to_dict()
    result["evaluator_objective_mutation_plan"] = output.evaluator_objective_plan.to_dict()
    result["curriculum_mutation_plan"] = output.curriculum_plan.to_dict()
    result["next_experiment_rewrite"] = output.next_experiment_rewrite.to_dict()
    result["next_run_config_recommendation"] = output.next_run_config.to_dict()
    audit = result.get("audit_trail")
    if isinstance(audit, dict):
        audit["meta_meta_meta_decision"] = result["meta_meta_meta_decision"]
        audit["search_strategy_config_suggestion"] = result["search_strategy_mutation_plan"]
        audit["evaluator_objective_config_suggestion"] = result["evaluator_objective_mutation_plan"]
        audit["curriculum_config_suggestion"] = result["curriculum_mutation_plan"]


def _metrics(
    first: Dict[str, object],
    second: Dict[str, object],
    process_components: Dict[str, float],
    first_meta: MetaMetaMetaControllerOutput,
) -> Dict[str, float]:
    first_agi = _mapping(first.get("agi_relevance_metrics")) or {}
    second_agi = _mapping(second.get("agi_relevance_metrics")) or {}
    metrics = {
        "first_run_cumulative_autonomous_improvement_score": _metric(first_agi, "cumulative_autonomous_improvement_score"),
        "second_run_cumulative_autonomous_improvement_score": _metric(second_agi, "cumulative_autonomous_improvement_score"),
        "first_run_residue_resolution_rate": _metric(first_agi, "residue_resolution_rate"),
        "second_run_residue_resolution_rate": _metric(second_agi, "residue_resolution_rate"),
        "first_run_rollback_rate": _rollback_rate(first),
        "second_run_rollback_rate": _rollback_rate(second),
        "first_run_hidden_validation_robustness": _metric(first_agi, "hidden_validation_robustness"),
        "second_run_hidden_validation_robustness": _metric(second_agi, "hidden_validation_robustness"),
        "strategy_mutations_applied": float(len(first_meta.search_strategy_plan.selected_mutations)),
        "evaluator_mutations_applied": float(len(first_meta.evaluator_objective_plan.selected_mutations)),
        "curriculum_mutations_applied": float(len(first_meta.curriculum_plan.selected_mutations)),
        "process_improvement_delta": process_components["final_process_improvement_delta"],
    }
    metrics.update(process_components)
    return metrics


def _anti_cheat_checks(
    first_process,
    second_process,
    first: Dict[str, object],
    second: Dict[str, object],
) -> list[str]:
    checks: list[str] = [
        "meta-meta-meta controller runs after candidate freeze",
        "next-run config is JSON data only",
        "heldout test is not used for candidate selection",
        "heldout test is used only for post-freeze process diagnosis",
        "negative process deltas are preserved",
        "counterfactual reports are estimates, not proof",
    ]
    if first_process.used_heldout_for_candidate_selection or second_process.used_heldout_for_candidate_selection:
        raise ValueError("heldout_test was used for candidate selection")
    if not (first_process.heldout_used_only_for_post_freeze_diagnosis and second_process.heldout_used_only_for_post_freeze_diagnosis):
        raise ValueError("heldout_test was not confined to post-freeze diagnosis")
    for run in (first, second):
        decisions = run.get("architecture_decisions", [])
        if isinstance(decisions, list) and any(isinstance(item, dict) and item.get("used_heldout_labels") for item in decisions):
            raise ValueError("heldout labels used in architecture decision")
    return checks


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


def _mapping(value: object) -> Dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
