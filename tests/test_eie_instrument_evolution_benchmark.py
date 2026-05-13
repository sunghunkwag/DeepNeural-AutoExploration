import json
from pathlib import Path

import pytest

from benchmarks.eie_instrument_evolution_benchmark import REQUIRED_METRICS, run
from scripts.verify_eie_smoke import verify


@pytest.fixture(scope="module")
def smoke_payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("eie") / "eie_instrument_evolution_smoke_seed42.json"
    result = run("smoke", 42, str(out))
    verify(out)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    return out, result, manifest


def test_smoke_benchmark_returns_required_metrics(smoke_payload):
    _, result, _ = smoke_payload
    for metric in REQUIRED_METRICS:
        assert metric in result["metrics"]
    assert result["metrics"]["initial_residue_pressure"] > 0.0
    assert result["metrics"]["mutation_contract_count"] >= 1.0


def test_smoke_benchmark_writes_json_output(smoke_payload):
    out, result, _ = smoke_payload
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"] == result["metrics"]


def test_smoke_benchmark_writes_manifest_output(smoke_payload):
    _, result, manifest = smoke_payload
    assert Path(result["manifest_path"]).exists()
    assert manifest["failure_residue_ledger"]["records"]
    assert manifest["instrument_mutation_contracts"]


def test_no_test_split_used_for_acceptance(smoke_payload):
    _, result, manifest = smoke_payload
    assert result["used_test_signal_for_acceptance"] is False
    assert manifest["used_test_signal_for_acceptance"] is False
    for decision in manifest["instrument_mutation_decisions"]:
        assert decision["used_test_signal_for_acceptance"] is False
        assert "test" not in decision.get("accepted_on_splits", [])


def test_at_least_one_mutation_contract_is_created(smoke_payload):
    _, result, manifest = smoke_payload
    assert result["metrics"]["mutation_contract_count"] >= 1.0
    assert len(manifest["instrument_mutation_contracts"]) >= 1


def test_no_op_and_random_controls_are_recorded(smoke_payload):
    _, result, manifest = smoke_payload
    assert "no_op_control_score" in result["metrics"]
    assert "random_control_score" in result["metrics"]
    contract_ids = [contract["contract_id"] for contract in manifest["instrument_mutation_contracts"]]
    assert any(contract_id.startswith("control_no_op") for contract_id in contract_ids)
    assert any(contract_id.startswith("control_random") for contract_id in contract_ids)
    no_op_decisions = [
        decision
        for decision in manifest["instrument_mutation_decisions"]
        if decision["contract"]["mutation_payload"].get("action") == "no_op"
    ]
    assert no_op_decisions
    assert all(not decision["accepted"] for decision in no_op_decisions)


def test_problem_space_graph_is_present_in_manifest(smoke_payload):
    _, result, manifest = smoke_payload
    graph = manifest["problem_space_version_graph"]
    assert graph["versions"]
    assert graph["current_version_id"]
    assert result["metrics"]["problem_space_version_count"] >= 1.0

