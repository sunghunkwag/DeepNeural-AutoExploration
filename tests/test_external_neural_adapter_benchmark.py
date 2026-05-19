import json
from pathlib import Path

from benchmarks.external_neural_adapter_benchmark import run as run_external
from neural_search.external_task_adapters import SPLITS, assert_external_splits_disjoint, build_external_task_families


def test_external_adapter_has_disjoint_splits():
    families = build_external_task_families(seed=7, samples_per_split=8)
    assert_external_splits_disjoint(families)
    assert all(set(family.splits) == set(SPLITS) for family in families)


def test_external_neural_adapter_benchmark_writes_outputs_and_preserves_heldout(tmp_path):
    out = tmp_path / "external.json"
    result = run_external("smoke", 7, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert out.exists()
    assert Path(result["manifest_path"]).exists()
    assert result["heldout_test_used_for_candidate_selection"] is False
    assert result["heldout_test_used_only_after_freeze"] is True
    assert "heldout_after_freeze_delta" in manifest
