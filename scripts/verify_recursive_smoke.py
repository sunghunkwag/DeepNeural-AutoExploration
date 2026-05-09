"""Validate recursive smoke benchmark outputs for CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


REQUIRED_CHECKS = {
    "random seeds explicitly recorded",
    "train/validation/test task IDs disjoint",
    "all task IDs recorded",
    "no test task used in mutation acceptance",
    "no query_y stored in memory",
    "benchmark result file includes config and task IDs",
    "OOD mode actually changes distribution",
    "no query_y used in task encoding",
    "no final test metric used in optimizer/controller",
    "no benchmark result hardcoded",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise AssertionError(f"missing expected file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"expected top-level JSON object in {path}")
    return payload


def _assert_non_empty_split_ids(payload: Dict[str, Any], source: str) -> None:
    for split in ("train", "validation", "test"):
        if source == "result":
            ids = payload.get("task_ids", {}).get(split)
        else:
            ids = payload.get(f"{split}_task_ids")
        if not ids:
            raise AssertionError(f"{source} missing {split} task IDs")


def _assert_checks(checks: Any, source: str) -> None:
    if not isinstance(checks, list):
        raise AssertionError(f"{source} anti-cheat checks must be a list")
    missing = REQUIRED_CHECKS.difference(str(item) for item in checks)
    if missing:
        raise AssertionError(f"{source} missing anti-cheat checks: {sorted(missing)}")


def verify(result_path: Path) -> None:
    result = _load_json(result_path)
    manifest_path = Path(str(result.get("manifest_path", result_path.with_suffix(".manifest.json"))))
    if not manifest_path.is_absolute():
        manifest_path = result_path.parent / manifest_path
    manifest = _load_json(manifest_path)

    _assert_non_empty_split_ids(result, "result")
    _assert_non_empty_split_ids(manifest, "manifest")
    _assert_checks(result.get("anti_cheat_checks_passed"), "result")
    _assert_checks(manifest.get("anti_cheat_checks_passed"), "manifest")

    if not manifest.get("config_hash"):
        raise AssertionError("manifest missing config_hash")
    if not manifest.get("metric_summary"):
        raise AssertionError("manifest missing metric_summary")
    if not result.get("aggregate"):
        raise AssertionError("result missing aggregate metrics")
    if not result.get("mutation_log"):
        raise AssertionError("result missing mutation log")

    aggregate = result["aggregate"]
    for key in ("controller_call_count", "world_model_call_count", "accepted_operator_count"):
        value = aggregate.get(key, {}).get("mean")
        if value is None or float(value) <= 0.0:
            raise AssertionError(f"aggregate metric {key} must be positive")
    for key in (
        "improvement_velocity",
        "operator_reuse_success",
        "controller_prediction_error_reduction",
        "ood_transfer_after_accepted_mutations",
        "accepted_mutation_quality",
    ):
        if key not in aggregate:
            raise AssertionError(f"aggregate missing operator-level metric {key}")

    per_seed = result.get("per_seed") or []
    if not per_seed or not per_seed[0].get("operator_genome", {}).get("accepted_gene_ids"):
        raise AssertionError("result missing accepted operator genome entries")

    for mutation in manifest.get("accepted_mutations", []) + manifest.get("rejected_mutations", []):
        if mutation.get("split") == "test" or mutation.get("accepted_on_split") == "test":
            raise AssertionError("mutation log used test split for acceptance")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/verify_recursive_smoke.py <result-json>")
    verify(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
