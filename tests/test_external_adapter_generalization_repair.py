import json
from pathlib import Path

from benchmarks.external_neural_adapter_benchmark import _external_acceptance_status, run


def test_external_adapter_reports_heldout_after_freeze_regression_honestly():
    status, reason = _external_acceptance_status(0.1, 0.1, -0.01)

    assert status == "failed_generalization"
    assert "heldout_after_freeze_regressed" in reason


def test_external_adapter_output_contains_generalization_diagnostics(tmp_path):
    out = tmp_path / "external.json"
    result = run("smoke", 42, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert "external_generalization_diagnosis" in result
    assert "train_validation_hidden_heldout_gap_table" in result["external_generalization_diagnosis"]
    assert result["external_acceptance_status"] in {"externally_helpful", "failed_generalization"}
    assert result["heldout_test_used_for_candidate_selection"] is False
    assert result["heldout_test_used_only_after_freeze"] is True
    assert out.exists()
    assert manifest["anti_cheat_checks_passed"]
