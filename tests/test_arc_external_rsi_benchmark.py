import json
import math
from pathlib import Path

import torch

from adaptation_operators import OperatorExecutionContext
from operator_dsl import OperatorProgram, PrimitiveStep, compile_operator_program
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
    assert len(loaded.tasks[0].metadata["support_offsets"]) == 3
    holdouts = arc_benchmark._support_holdout_tasks(loaded.tasks[0])
    assert len(holdouts) == 2
    assert holdouts[0].metadata["support_offsets"][-1] == holdouts[0].support_x.shape[0]


def test_arc_geometric_hdc_program_solves_support_only_flip():
    def hflip(grid):
        return [list(reversed(row)) for row in grid]

    payload = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": hflip([[1, 2], [3, 4]])},
            {"input": [[5, 6], [7, 8]], "output": hflip([[5, 6], [7, 8]])},
        ],
        "test": [{"input": [[2, 1], [4, 3]], "output": hflip([[2, 1], [4, 3]])}],
    }
    task = arc_benchmark._arc_payload_to_task("geom_hflip", "test", payload)
    program = OperatorProgram(
        "test_support_geometric_transform_mapping",
        (PrimitiveStep("support_geometric_transform_mapping"),),
    )
    context = OperatorExecutionContext(
        model=torch.nn.Linear(arc_benchmark.ARC_FEATURE_DIM, 10),
        task=task,
        task_embedding=torch.zeros(8),
        base_inner_lr=0.01,
        inner_steps=1,
    )
    _, details = compile_operator_program(program).adapt_params(context)
    assert details.prediction_override is not None
    metrics = arc_benchmark._grid_metrics_from_predictions(task, details.prediction_override)
    assert metrics["exact"] is True
    assert metrics["cell_accuracy"] == 1.0
    assert details.primitive_effects["support_geometric_hdc_dimension"] == 10000.0


def test_arc_nested_ring_reversal_program_solves_support_only_rings():
    def rings(colors):
        size = len(colors) * 2
        return [
            [colors[min(r, c, size - 1 - r, size - 1 - c)] for c in range(size)]
            for r in range(size)
        ]

    train_a = rings([4, 2, 1])
    train_b = rings([8, 6, 3])
    test = rings([7, 5, 9, 1])
    payload = {
        "train": [
            {"input": train_a, "output": rings(list(reversed([4, 2, 1])))},
            {"input": train_b, "output": rings(list(reversed([8, 6, 3])))},
        ],
        "test": [{"input": test, "output": rings(list(reversed([7, 5, 9, 1])))}],
    }
    task = arc_benchmark._arc_payload_to_task("nested_rings", "test", payload)
    program = OperatorProgram(
        "test_support_nested_ring_reversal_mapping",
        (PrimitiveStep("support_nested_ring_reversal_mapping"),),
    )
    context = OperatorExecutionContext(
        model=torch.nn.Linear(arc_benchmark.ARC_FEATURE_DIM, 10),
        task=task,
        task_embedding=torch.zeros(8),
        base_inner_lr=0.01,
        inner_steps=1,
    )
    _, details = compile_operator_program(program).adapt_params(context)
    assert details.prediction_override is not None
    metrics = arc_benchmark._grid_metrics_from_predictions(task, details.prediction_override)
    assert metrics["exact"] is True
    assert metrics["cell_accuracy"] == 1.0
    assert details.primitive_effects["support_nested_ring_hdc_dimension"] == 10000.0


def test_arc_translation_program_solves_support_only_shift():
    def shift_right(grid):
        return [[0] + row[:-1] for row in grid]

    train_a = [[0, 6, 6, 0], [0, 6, 0, 6], [0, 0, 6, 0]]
    train_b = [[0, 8, 8, 8], [0, 8, 0, 0], [0, 0, 8, 8]]
    test = [[0, 4, 4, 4], [0, 4, 0, 0], [0, 0, 4, 4]]
    payload = {
        "train": [
            {"input": train_a, "output": shift_right(train_a)},
            {"input": train_b, "output": shift_right(train_b)},
        ],
        "test": [{"input": test, "output": shift_right(test)}],
    }
    task = arc_benchmark._arc_payload_to_task("translation", "test", payload)
    program = OperatorProgram(
        "test_support_translation_mapping",
        (PrimitiveStep("support_translation_mapping"),),
    )
    context = OperatorExecutionContext(
        model=torch.nn.Linear(arc_benchmark.ARC_FEATURE_DIM, 10),
        task=task,
        task_embedding=torch.zeros(8),
        base_inner_lr=0.01,
        inner_steps=1,
    )
    _, details = compile_operator_program(program).adapt_params(context)
    assert details.prediction_override is not None
    metrics = arc_benchmark._grid_metrics_from_predictions(task, details.prediction_override)
    assert metrics["exact"] is True
    assert metrics["cell_accuracy"] == 1.0
    assert details.primitive_effects["support_translation_hdc_dimension"] == 10000.0
    assert details.primitive_effects["support_translation_dc"] == 1.0


def _run_support_exact_repair(payload):
    task = arc_benchmark._arc_payload_to_task("exact_repair", "test", payload)
    program = OperatorProgram(
        "test_support_exact_repair_search",
        (PrimitiveStep("support_exact_repair_search"),),
    )
    context = OperatorExecutionContext(
        model=torch.nn.Linear(arc_benchmark.ARC_FEATURE_DIM, 10),
        task=task,
        task_embedding=torch.zeros(8),
        base_inner_lr=0.01,
        inner_steps=1,
    )
    _, details = compile_operator_program(program).adapt_params(context)
    assert details.prediction_override is not None
    metrics = arc_benchmark._grid_metrics_from_predictions(task, details.prediction_override)
    assert metrics["exact"] is True
    assert metrics["cell_accuracy"] == 1.0
    assert details.primitive_effects["support_exact_repair_exact"] == 1.0
    return details


def test_arc_exact_repair_solves_binary_mask_recolor():
    payload = {
        "train": [
            {
                "input": [[4, 5, 4], [5, 5, 5], [4, 5, 4]],
                "output": [[0, 4, 0], [4, 4, 4], [0, 4, 0]],
            },
            {
                "input": [[5, 6, 6], [6, 5, 6], [6, 6, 5]],
                "output": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
            },
        ],
        "test": [
            {
                "input": [[3, 5, 3], [5, 3, 5], [3, 5, 3]],
                "output": [[0, 3, 0], [3, 0, 3], [0, 3, 0]],
            }
        ],
    }
    details = _run_support_exact_repair(payload)
    assert any(key.startswith("support_exact_repair_method:binary_mask") for key in details.primitive_effects)


def test_arc_exact_repair_solves_edge_reflection_and_rectangle_fill():
    reflection_payload = {
        "train": [
            {
                "input": [[2, 2, 2], [3, 3, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "output": [[2, 2, 2], [3, 3, 3], [0, 0, 0], [3, 3, 3], [2, 2, 2]],
            },
            {
                "input": [[8, 8, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "output": [[8, 8, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0], [8, 8, 8, 8]],
            },
        ],
        "test": [
            {
                "input": [[3, 3, 3, 3], [5, 5, 5, 5], [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                "output": [[3, 3, 3, 3], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [3, 3, 3, 3]],
            }
        ],
    }
    _run_support_exact_repair(reflection_payload)

    rectangle_payload = {
        "train": [
            {
                "input": [[4, 0, 4], [0, 0, 0], [4, 0, 4]],
                "output": [[4, 0, 4], [0, 2, 0], [4, 0, 4]],
            },
            {
                "input": [[4, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 4]],
                "output": [[4, 0, 0, 4], [0, 2, 2, 0], [0, 2, 2, 0], [4, 0, 0, 4]],
            },
        ],
        "test": [
            {
                "input": [[4, 0, 0, 0, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [4, 0, 0, 0, 4]],
                "output": [[4, 0, 0, 0, 4], [0, 2, 2, 2, 0], [0, 2, 2, 2, 0], [4, 0, 0, 0, 4]],
            }
        ],
    }
    _run_support_exact_repair(rectangle_payload)


def test_arc_exact_repair_solves_corner_projection_and_terminal_shift():
    corner_payload = {
        "train": [
            {
                "input": [[0, 0, 0, 0, 0, 0], [0, 4, 6, 0, 0, 0], [0, 2, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[1, 1, 0, 0, 2, 2], [0, 4, 6, 0, 0, 0], [0, 2, 1, 0, 0, 0], [6, 6, 0, 0, 4, 4]],
            },
            {
                "input": [[0, 0, 0, 0, 0, 0], [0, 0, 9, 3, 0, 0], [0, 0, 7, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[8, 8, 0, 0, 7, 7], [0, 0, 9, 3, 0, 0], [0, 0, 7, 8, 0, 0], [3, 3, 0, 0, 9, 9]],
            },
        ],
        "test": [
            {
                "input": [[0, 0, 0, 0, 0, 0], [0, 0, 3, 1, 0, 0], [0, 0, 2, 5, 0, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[5, 5, 0, 0, 2, 2], [0, 0, 3, 1, 0, 0], [0, 0, 2, 5, 0, 0], [1, 1, 0, 0, 3, 3]],
            }
        ],
    }
    _run_support_exact_repair(corner_payload)

    def terminal_shift(grid):
        out = []
        for row_index, row in enumerate(grid):
            next_empty = row_index + 1 >= len(grid) or all(value == 0 for value in grid[row_index + 1])
            longest = max((len(run) for run in "".join("1" if value else "0" for value in row).split("0")), default=0)
            if next_empty and longest >= 3:
                out.append(list(row))
            else:
                out.append([0] + row[:-1])
        return out

    train_a = [[0, 6, 6, 6, 0], [0, 6, 0, 6, 0], [0, 0, 6, 6, 6]]
    train_b = [[0, 8, 8, 8, 8], [0, 8, 0, 0, 8], [0, 0, 8, 8, 8]]
    test = [[0, 4, 4, 4, 0], [0, 4, 0, 4, 0], [0, 0, 4, 4, 4]]
    shift_payload = {
        "train": [
            {"input": train_a, "output": terminal_shift(train_a)},
            {"input": train_b, "output": terminal_shift(train_b)},
        ],
        "test": [{"input": test, "output": terminal_shift(test)}],
    }
    _run_support_exact_repair(shift_payload)


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
    assert manifest["generation_summaries"][0]["selector_guard"]["selected"] == "support_cv_per_task"
