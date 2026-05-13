"""Verify bounded EIE smoke benchmark outputs for CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


REQUIRED_FIELDS = {
    "initial_residue_pressure",
    "final_residue_pressure",
    "residue_pressure_delta",
    "mutation_contract_count",
    "accepted_instrument_mutation_count",
    "rejected_instrument_mutation_count",
    "rollback_count",
    "problem_space_version_count",
    "observation_blindspot_reduction",
    "evaluator_penalty_strength_delta",
    "experiment_unit_coverage_gain",
    "operator_generator_diversity_gain",
    "validation_to_hidden_gap",
    "no_op_control_score",
    "random_control_score",
    "used_test_signal_for_acceptance",
    "manifest_path",
}

REQUIRED_MANIFEST_FIELDS = {
    "failure_residue_ledger",
    "instrument_mutation_contracts",
    "problem_space_version_graph",
    "anti_cheat_checks_passed",
    "validation_only_acceptance_enforced",
    "held_out_test_used_only_after_freeze",
}


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise AssertionError(f"missing file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return payload


def _metric(result: Dict[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    aggregate = result.get("aggregate")
    if isinstance(aggregate, dict) and key in aggregate and isinstance(aggregate[key], dict):
        return aggregate[key].get("mean")
    raise AssertionError(f"result missing required field: {key}")


def verify(result_path: str | Path) -> None:
    result_path = Path(result_path)
    result = _load(result_path)
    for field in REQUIRED_FIELDS:
        if field == "manifest_path":
            if not result.get("manifest_path"):
                raise AssertionError("manifest path is missing")
            continue
        _metric(result, field)

    if float(_metric(result, "initial_residue_pressure")) <= 0.0:
        raise AssertionError("initial_residue_pressure must be positive")
    if float(_metric(result, "mutation_contract_count")) < 1.0:
        raise AssertionError("mutation_contract_count must be at least 1")
    accepted = float(_metric(result, "accepted_instrument_mutation_count"))
    rejected = float(_metric(result, "rejected_instrument_mutation_count"))
    if accepted + rejected < 1.0:
        raise AssertionError("accepted + rejected mutation count must be at least 1")
    if bool(_metric(result, "used_test_signal_for_acceptance")):
        raise AssertionError("used_test_signal_for_acceptance must be false")
    if float(_metric(result, "problem_space_version_count")) < 1.0:
        raise AssertionError("problem_space_version_count must be at least 1")
    _metric(result, "no_op_control_score")
    _metric(result, "random_control_score")

    manifest_path = Path(str(result["manifest_path"]))
    if not manifest_path.is_absolute():
        direct = manifest_path
        manifest_path = direct if direct.exists() else result_path.parent / direct.name
    manifest = _load(manifest_path)
    missing = REQUIRED_MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise AssertionError(f"manifest missing required fields: {sorted(missing)}")
    if manifest.get("used_test_signal_for_acceptance"):
        raise AssertionError("manifest used_test_signal_for_acceptance must be false")
    if not manifest.get("validation_only_acceptance_enforced"):
        raise AssertionError("manifest did not enforce validation-only acceptance")
    if not manifest.get("held_out_test_used_only_after_freeze"):
        raise AssertionError("manifest did not record frozen held-out test usage")
    if not manifest.get("instrument_mutation_contracts"):
        raise AssertionError("manifest missing instrument mutation contracts")
    graph = manifest.get("problem_space_version_graph") or {}
    if not graph.get("versions"):
        raise AssertionError("manifest problem-space graph has no versions")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/verify_eie_smoke.py <result-json>")
    verify(sys.argv[1])


if __name__ == "__main__":
    main()
