"""Verify code-level RSI smoke benchmark outputs for CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_METRICS = {
    "candidate_count",
    "compiled_candidate_count",
    "accepted_program_count",
    "rollback_count",
    "improvement_velocity",
    "validation_to_test_gap",
    "dead_code_detector_result",
    "runtime_behavior_difference_observed",
    "self_model_prediction_error",
    "self_model_error_reduction",
    "failure_rule_count",
    "failure_rule_reuse_count",
    "candidate_quality_after_failure_rules",
    "evaluator_candidate_count",
    "accepted_evaluator_count",
    "probation_evaluator_count",
    "evaluator_overfit_detector",
    "candidate_quality_per_compute",
    "future_candidate_quality_prediction_error",
    "OOD_transfer_after_evaluator_evolution",
    "full_loop_vs_no_self_model",
    "full_loop_vs_no_failure_grammar",
    "full_loop_vs_no_evaluator_evolution",
}

REQUIRED_CHECKS = {
    "random seeds explicitly recorded",
    "train/validation/test task IDs disjoint",
    "hidden validation pool generated inside benchmark execution",
    "candidate programs compile into executable operators",
    "validation-only candidate acceptance enforced",
    "held-out OOD test used only after decisions are frozen",
    "runtime behavior difference observed",
    "dead-code detector passed",
    "deterministic replay checked",
    "accepted and rejected candidates logged with reasons",
    "self-model trained only on train/validation traces",
    "self-model predictions compared to actual outcomes",
    "failure grammar rewrites future candidates",
    "evaluator candidates require adversarial checks",
    "evaluator evolution recorded probationary decisions",
}


def _load(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_task_ids(payload: dict, source: str) -> None:
    ids = payload.get("task_ids") or {}
    for split in ("train", "validation", "test"):
        if not ids.get(split):
            raise AssertionError(f"{source} missing {split} task IDs")
    train = set(ids["train"])
    validation = set(ids["validation"]) | set(ids.get("hidden_validation", []))
    test = set(ids["test"])
    if train & validation or train & test or validation & test:
        raise AssertionError(f"{source} task IDs are not disjoint")


def _assert_checks(payload: dict, source: str) -> None:
    checks = set(str(item) for item in payload.get("anti_cheat_checks_passed", []))
    missing = REQUIRED_CHECKS.difference(checks)
    if missing:
        raise AssertionError(f"{source} missing anti-cheat checks: {sorted(missing)}")


def verify(result_path: str | Path) -> None:
    result_path = Path(result_path)
    result = _load(result_path)
    manifest_path = Path(str(result.get("manifest_path", result_path.with_suffix(".manifest.json"))))
    if not manifest_path.is_absolute():
        direct = manifest_path
        manifest_path = direct if direct.exists() else result_path.parent / direct.name
    manifest = _load(manifest_path)

    _assert_task_ids(result, "result")
    _assert_task_ids(manifest, "manifest")
    _assert_checks(result, "result")
    _assert_checks(manifest, "manifest")

    aggregate = result.get("aggregate") or {}
    for metric in REQUIRED_METRICS:
        if metric not in aggregate:
            raise AssertionError(f"aggregate missing metric: {metric}")
        if "mean" not in aggregate[metric]:
            raise AssertionError(f"aggregate metric missing mean: {metric}")
    if aggregate["candidate_count"]["mean"] <= 0:
        raise AssertionError("no synthesized candidate was generated")
    if aggregate["compiled_candidate_count"]["mean"] <= 0:
        raise AssertionError("no synthesized candidate compiled")
    if aggregate["accepted_program_count"]["mean"] <= 0:
        raise AssertionError("no synthesized candidate was accepted")
    if aggregate["dead_code_detector_result"]["mean"] < 1.0:
        raise AssertionError("dead-code detector did not pass")
    if aggregate["runtime_behavior_difference_observed"]["mean"] < 1.0:
        raise AssertionError("no runtime behavior difference observed")
    if aggregate["failure_rule_count"]["mean"] <= 0:
        raise AssertionError("failure grammar did not produce rules")
    if aggregate["failure_rule_reuse_count"]["mean"] <= 0:
        raise AssertionError("failure grammar rules were not reused")
    if aggregate["evaluator_candidate_count"]["mean"] <= 0:
        raise AssertionError("no evaluator candidates were evaluated")
    if aggregate["probation_evaluator_count"]["mean"] <= 0:
        raise AssertionError("no evaluator reached probation")

    generated = result.get("generated_candidates") or []
    if not generated:
        raise AssertionError("generated operator programs were not serialized")
    decisions = result.get("candidate_decisions") or []
    if not decisions:
        raise AssertionError("candidate decisions are missing")
    for decision in decisions:
        if decision.get("split") == "test" or decision.get("accepted_on_split") == "test":
            raise AssertionError("candidate accepted or evaluated for acceptance on test split")
        if not decision.get("candidate_program"):
            raise AssertionError("candidate decision missing serialized operator program")
        if "rejection_reason" not in decision:
            raise AssertionError("candidate decision missing accepted/rejected reason")

    if not manifest.get("config_hash"):
        raise AssertionError("manifest missing config_hash")
    if not manifest.get("generated_candidates"):
        raise AssertionError("manifest missing generated candidates")
    if not manifest.get("compile_status_per_candidate"):
        raise AssertionError("manifest missing compile status per candidate")
    if not manifest.get("runtime_behavior_difference_checks"):
        raise AssertionError("manifest missing runtime behavior checks")
    if not manifest.get("operator_source_hashes"):
        raise AssertionError("manifest missing operator source/program hashes")
    if not result.get("self_model_prediction_log") or not manifest.get("self_model_prediction_log"):
        raise AssertionError("self-model prediction logs missing")
    if not result.get("failure_rules") or not manifest.get("failure_rules"):
        raise AssertionError("failure grammar rules missing")
    if not result.get("evaluator_decisions") or not manifest.get("evaluator_decisions"):
        raise AssertionError("evaluator evolution decisions missing")
    if not manifest.get("failure_grammar", {}).get("rewrite_log"):
        raise AssertionError("failure grammar did not rewrite future candidates")
    for entry in manifest.get("self_model_prediction_log", []):
        if entry.get("actual_effects") is None:
            raise AssertionError("self-model prediction missing actual outcome comparison")
    for decision in manifest.get("evaluator_decisions", []):
        checks = decision.get("adversarial_checks", {})
        if not checks.get("passed"):
            continue
        for key in ("checked_against_old_evaluator", "hidden_validation_pool", "wrong_world_model_control", "shuffled_memory_control", "no_op_candidate_control", "random_candidate_control"):
            if not checks.get(key):
                raise AssertionError(f"evaluator decision missing adversarial check: {key}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/verify_code_level_rsi_smoke.py <result-json>")
    verify(sys.argv[1])


if __name__ == "__main__":
    main()
