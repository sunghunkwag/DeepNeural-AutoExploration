"""Local external-style neural adapter benchmark with heldout isolation."""

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

from experiment_manifest import current_git_commit, stable_config_hash
from neural_search import ArchitectureGenome
from neural_search.external_task_adapters import (
    assert_external_splits_disjoint,
    build_external_task_families,
    external_split_distribution_summary,
    serializable_external_task_families,
)
from neural_search.multidomain_evaluator import MultiDomainEvaluator
from neural_search.mutation_operators import mutate_architecture


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    cfg = _mode_config(mode)
    output_path = Path(output or f"results/external_neural_adapter_{mode}_seed{seed}.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    task_families = build_external_task_families(seed=seed, samples_per_split=int(cfg["samples_per_split"]))
    assert_external_splits_disjoint(task_families)
    evaluator = MultiDomainEvaluator(seed=seed, train_steps=int(cfg["train_steps"]), lr=0.018)
    base = ArchitectureGenome(hidden_dim=int(cfg["hidden_dim"]), residual_blocks=1, seed=seed)
    base_eval = evaluator.train_and_validate(base, task_families)
    current = base
    current_eval = base_eval
    decisions = []
    rollback_count = 0
    for index, method in enumerate(["widen_hidden_dim", "add_residual_block", "object_binding_adapter"][: int(cfg["candidate_count"])]):
        candidate = mutate_architecture(current, method, seed + 100 + index)
        candidate_eval = evaluator.train_and_validate(candidate, task_families)
        decision = evaluator.selection_decision(current, candidate, current_eval, candidate_eval)
        decision_dict = decision.to_dict()
        decision_dict = _apply_external_pre_freeze_generalization_guard(decision_dict)
        decisions.append(decision_dict)
        if decision_dict["accepted"]:
            current = candidate
            current_eval = candidate_eval
        else:
            rollback_count += 1
    final_heldout = evaluator.evaluate_heldout_after_freeze(current_eval.model, current, task_families)
    base_heldout = evaluator.evaluate_heldout_after_freeze(base_eval.model, base, task_families)
    validation_delta = base_eval.validation_loss - current_eval.validation_loss
    hidden_delta = base_eval.hidden_validation_loss - current_eval.hidden_validation_loss
    heldout_delta = base_heldout.heldout_test_loss - final_heldout.heldout_test_loss
    external_acceptance_status, external_failure_reason = _external_acceptance_status(validation_delta, hidden_delta, heldout_delta)
    generalization_diagnosis = _external_generalization_diagnosis(
        task_families,
        base_eval,
        current_eval,
        base_heldout,
        final_heldout,
        current,
        decisions,
        heldout_delta,
    )
    result = {
        "benchmark": "external_neural_adapter",
        "mode": mode,
        "seed": seed,
        "config": cfg,
        "adapter": "local_external_synthetic_tasks",
        "task_families": serializable_external_task_families(task_families),
        "baseline_metrics": base_heldout.to_dict(),
        "evolved_metrics": final_heldout.to_dict(),
        "validation_delta": float(validation_delta),
        "hidden_validation_delta": float(hidden_delta),
        "heldout_after_freeze_delta": float(heldout_delta),
        "external_generalization_diagnosis": generalization_diagnosis,
        "external_acceptance_status": external_acceptance_status,
        "external_failure_reason": external_failure_reason,
        "rollback_count": int(rollback_count),
        "residue_summary": _residue_summary(decisions),
        "limitation_summary": [
            "local generated external-style tasks only; no downloaded benchmark",
            "heldout_test is evaluated only after candidate selection is frozen",
            "small CPU-runnable neural search signal, not a broad external benchmark claim",
        ],
        "architecture_decisions": decisions,
        "heldout_test_used_for_candidate_selection": any(decision.get("used_heldout_labels") for decision in decisions),
        "heldout_test_used_only_after_freeze": bool(final_heldout.heldout_test_evaluated_after_freeze),
    }
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": f"python benchmarks/external_neural_adapter_benchmark.py --mode {mode} --seed {seed}",
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "config_hash": stable_config_hash({"mode": mode, "seed": seed, **cfg}),
        "validation_delta": float(validation_delta),
        "hidden_validation_delta": float(hidden_delta),
        "heldout_after_freeze_delta": float(heldout_delta),
        "external_acceptance_status": external_acceptance_status,
        "external_failure_reason": external_failure_reason,
        "rollback_count": int(rollback_count),
        "anti_cheat_checks_passed": [
            "external adapter splits are disjoint",
            "heldout_test evaluated only after freeze",
            "no external download required",
        ],
    }
    if result["heldout_test_used_for_candidate_selection"]:
        raise ValueError("heldout_test used for external adapter candidate selection")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["manifest_path"] = str(manifest_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _mode_config(mode: str) -> Dict[str, int]:
    return {
        "smoke": {"samples_per_split": 12, "train_steps": 6, "candidate_count": 2, "hidden_dim": 10},
        "quick": {"samples_per_split": 18, "train_steps": 8, "candidate_count": 3, "hidden_dim": 12},
        "full": {"samples_per_split": 24, "train_steps": 10, "candidate_count": 3, "hidden_dim": 12},
    }[mode]


def _residue_summary(decisions) -> Dict[str, object]:
    reasons: Dict[str, int] = {}
    for decision in decisions:
        if decision.get("accepted"):
            continue
        reason = str(decision.get("rejection_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"rejection_reasons": reasons, "persistent_residue_types": sorted(reasons)}


def _apply_external_pre_freeze_generalization_guard(decision: Dict[str, object]) -> Dict[str, object]:
    guarded = dict(decision)
    if not bool(guarded.get("accepted")):
        guarded["external_pre_freeze_generalization_guard_applied"] = False
        return guarded
    regressions = guarded.get("task_family_regressions", {})
    has_family_regression = isinstance(regressions, dict) and any(float(value) > 1e-6 for value in regressions.values())
    validation_gain = float(guarded.get("validation_improvement", 0.0))
    hidden_gain = float(guarded.get("hidden_validation_improvement", 0.0))
    score_delta = float(guarded.get("selection_score_delta", 0.0))
    hidden_margin = max(0.012, 1.25 * max(0.0, validation_gain))
    conservative_accept = (
        validation_gain > 0.0
        and hidden_gain >= hidden_margin
        and score_delta > 0.0
        and not has_family_regression
    )
    guarded["external_pre_freeze_generalization_guard_applied"] = True
    guarded["external_pre_freeze_generalization_guard_passed"] = conservative_accept
    guarded["external_pre_freeze_generalization_guard_evidence"] = {
        "validation_improvement": validation_gain,
        "hidden_validation_improvement": hidden_gain,
        "required_hidden_margin": hidden_margin,
        "selection_score_delta": score_delta,
        "has_task_family_regression": has_family_regression,
        "used_heldout_labels": False,
    }
    if not conservative_accept:
        guarded["accepted"] = False
        guarded["rollback"] = True
        guarded["rejection_reason"] = "external_pre_freeze_generalization_guard"
    return guarded


def _external_acceptance_status(validation_delta: float, hidden_delta: float, heldout_delta: float) -> tuple[str, str]:
    if validation_delta > 0.0 and hidden_delta > 0.0 and heldout_delta >= 0.0:
        return "externally_helpful", ""
    reasons = []
    if validation_delta <= 0.0:
        reasons.append("validation_delta_not_positive")
    if hidden_delta <= 0.0:
        reasons.append("hidden_validation_delta_not_positive")
    if heldout_delta < 0.0:
        reasons.append("heldout_after_freeze_regressed")
    return "failed_generalization", ",".join(reasons)


def _external_generalization_diagnosis(
    task_families,
    base_eval,
    current_eval,
    base_heldout,
    final_heldout,
    current_genome: ArchitectureGenome,
    decisions,
    heldout_delta: float,
) -> Dict[str, object]:
    gap_table: Dict[str, object] = {}
    for family in sorted(base_eval.validation_loss_by_family):
        base_val = float(base_eval.validation_loss_by_family.get(family, 0.0))
        final_val = float(current_eval.validation_loss_by_family.get(family, 0.0))
        base_hidden = float(base_eval.hidden_validation_loss_by_family.get(family, 0.0))
        final_hidden = float(current_eval.hidden_validation_loss_by_family.get(family, 0.0))
        base_ho = float(base_heldout.heldout_test_loss_by_family.get(family, 0.0))
        final_ho = float(final_heldout.heldout_test_loss_by_family.get(family, 0.0))
        gap_table[family] = {
            "validation_delta": base_val - final_val,
            "hidden_validation_delta": base_hidden - final_hidden,
            "heldout_after_freeze_delta": base_ho - final_ho,
            "hidden_heldout_gap_after_freeze": final_ho - final_hidden,
        }
    accepted_methods = [str(decision.get("mutation_method", "")) for decision in decisions if decision.get("accepted")]
    mutation_delta = {method: float(heldout_delta) for method in accepted_methods if method}
    rejected_methods = [str(decision.get("mutation_method", "")) for decision in decisions if not decision.get("accepted")]
    complexity = {
        "hidden_dim": int(current_genome.hidden_dim),
        "residual_blocks": int(current_genome.residual_blocks),
        "memory_gates": int(current_genome.memory_gates),
        "causal_bottlenecks": int(current_genome.causal_bottlenecks),
        "object_binding_adapters": int(current_genome.object_binding_adapters),
        "sequence_state_adapters": int(current_genome.sequence_state_adapters),
        "heldout_after_freeze_delta": float(heldout_delta),
    }
    difficult = sorted(
        gap_table,
        key=lambda family: float(gap_table[family]["hidden_heldout_gap_after_freeze"]) if isinstance(gap_table.get(family), dict) else 0.0,
        reverse=True,
    )
    return {
        "train_validation_hidden_heldout_gap_table": gap_table,
        "task_difficulty_split_analysis": difficult,
        "feature_distribution_shift_estimate": external_split_distribution_summary(task_families),
        "candidate_complexity_vs_heldout_delta": complexity,
        "mutation_method_vs_heldout_delta": mutation_delta,
        "rejected_mutation_methods": rejected_methods,
        "heldout_after_freeze_regression_warning": heldout_delta < 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External-style neural adapter benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps({key: result[key] for key in ("validation_delta", "hidden_validation_delta", "heldout_after_freeze_delta", "rollback_count")}, indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/external_neural_adapter_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
