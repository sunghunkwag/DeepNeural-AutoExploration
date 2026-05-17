import json
import math
from pathlib import Path

from benchmarks import arc_external_rsi_benchmark as arc_benchmark


def _grid(offset: int, size: int = 3):
    return [[(r + c + offset) % 4 for c in range(size)] for r in range(size)]


def _task_payload(offset: int):
    def transform(grid):
        return [[(cell + 1 + (offset % 2)) % 5 for cell in row] for row in grid]

    train_a = _grid(offset)
    train_b = _grid(offset + 1)
    test = _grid(offset + 2)
    return {
        "train": [
            {"input": train_a, "output": transform(train_a)},
            {"input": train_b, "output": transform(train_b)},
        ],
        "test": [{"input": test, "output": transform(test)}],
    }


def _write_arc_fixture(root: Path, count: int = 8) -> Path:
    task_dir = root / "data" / "training"
    task_dir.mkdir(parents=True)
    for idx in range(count):
        (task_dir / f"fixture_{idx:03d}.json").write_text(json.dumps(_task_payload(idx)), encoding="utf-8")
    bad = {
        "train": [{"input": [[1, 1], [1, 1]], "output": [[1]]}],
        "test": [{"input": [[1, 1], [1, 1]], "output": [[1]]}],
    }
    (task_dir / "bad_shape.json").write_text(json.dumps(bad), encoding="utf-8")
    return root


def test_arc_loader_uses_external_same_shape_tasks(tmp_path):
    root = _write_arc_fixture(tmp_path / "arc")
    loaded = arc_benchmark.load_arc_same_shape_tasks(root, seed=1, allow_download=False)
    assert loaded.filtered_task_count == 8
    assert len(loaded.tasks) == 8
    assert loaded.tasks[0].support_x.shape[1] == arc_benchmark.ARC_FEATURE_DIM
    assert loaded.tasks[0].metadata["source"] == arc_benchmark.ARC_REPO_URL


def test_arc_external_rsi_smoke_reports_baseline_to_evolved(tmp_path, monkeypatch):
    root = _write_arc_fixture(tmp_path / "arc", count=8)

    def tiny_config(mode):
        return {
            "generations": 1,
            "train": 2,
            "validation": 1,
            "hidden_validation": 1,
            "test": 2,
            "encoder_steps": 2,
            "controller_steps": 2,
            "max_candidates": 2,
            "timeout_seconds": 5.0,
            "max_grid_cells": 16,
        }

    monkeypatch.setattr(arc_benchmark, "_mode_config", tiny_config)
    out = tmp_path / "arc_external_rsi_smoke_seed3.json"
    result = arc_benchmark.run("smoke", 3, str(out), arc_data_dir=root, allow_download=False)
    metrics = result["metrics"]
    assert out.exists()
    assert Path(result["manifest_path"]).exists()
    assert metrics["candidate_count"] > 0
    assert math.isfinite(metrics["baseline_mean_loss"])
    assert math.isfinite(metrics["evolved_mean_loss"])
    assert "baseline_to_evolved_loss_delta" in metrics
    assert "cell_accuracy_delta" in metrics
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    task_ids = manifest["task_ids"]
    all_ids = task_ids["train"] + task_ids["validation"] + task_ids["hidden_validation"] + task_ids["test"]
    assert len(all_ids) == len(set(all_ids))
    assert "hidden validation gate used before held-out test scoring" in manifest["anti_cheat_checks_passed"]
