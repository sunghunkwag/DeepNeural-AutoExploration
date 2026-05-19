import json
from pathlib import Path

from benchmarks.meta_meta_meta_search_benchmark import run as run_meta_benchmark


def test_meta_meta_meta_benchmark_writes_result_and_manifest(tmp_path):
    out = tmp_path / "meta_meta_meta.json"
    result = run_meta_benchmark("smoke", 1501, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert out.exists()
    assert Path(result["next_run_config_path"]).exists()
    assert Path(result["first_run_output_path"]).exists()
    assert Path(result["second_run_output_path"]).exists()
    assert result["second_run_used_next_run_config"] is True
    assert result["heldout_test_used_for_candidate_selection"] is False
    assert "process_improvement_delta" in result["metrics"]
    assert manifest["next_run_config_path"] == result["next_run_config_path"]

