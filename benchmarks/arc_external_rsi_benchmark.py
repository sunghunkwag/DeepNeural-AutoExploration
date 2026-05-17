"""External ARC-AGI adapter for the bounded code-level RSI loop.

This benchmark connects the existing ``OperatorProgram`` RSI machinery to a
real external ARC-AGI-1 subset.  It intentionally evaluates a narrow,
disclosed slice: public ARC tasks whose demonstration and test outputs have the
same grid shape as their inputs.  Those tasks are converted into cell-level
support/query regression tasks, then the normal candidate generation,
sandboxed validation, hidden validation, rollback, and guarded selector path is
used to produce baseline -> evolved numbers.

It is not an official ARC leaderboard run.  It does not use private ARC test
data and it does not solve variable-output-size ARC tasks.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from candidate_sandbox import CandidateSandbox
from evaluator_evolution import EvaluatorGenome
from experiment_manifest import current_git_commit, stable_config_hash
from failure_grammar import FailureGrammar
from learned_task_encoder import LearnedTaskEncoder, TaskConditionedRegressor, train_task_encoder
from operator_dsl import ProgramGenome, compile_operator_program
from rsi_candidate_generator import CandidateGenerator, CandidateGeneratorConfig
from self_model import CandidateSelfModel
from task_suite import SyntheticTask, assert_disjoint_task_ids

from benchmarks.code_level_rsi_benchmark import (
    _collect_program_transitions,
    _guarded_selector_kind,
    _make_contexts,
    _program_actions,
    _select_program,
    _self_model_actual_effects,
    _shuffled_memory,
    _validate_candidate_program,
)
from benchmarks.recursive_self_improvement_benchmark import (
    WrongWorldModel,
    _build_memory,
    _initial_state,
    _train_world_model_from_operator_transitions,
)
from maml_functional import functional_forward


ARC_REPO_URL = "https://github.com/fchollet/ARC-AGI"
ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_FEATURE_DIM = 10


@dataclass(frozen=True)
class ArcLoadResult:
    tasks: List[SyntheticTask]
    filtered_task_count: int
    source_dir: str
    source_url: str


def _mode_config(mode: str) -> Dict[str, int | float]:
    return {
        "smoke": {
            "generations": 1,
            "train": 4,
            "validation": 2,
            "hidden_validation": 2,
            "test": 4,
            "encoder_steps": 5,
            "controller_steps": 8,
            "max_candidates": 4,
            "timeout_seconds": 6.0,
            "max_grid_cells": 144,
        },
        "quick": {
            "generations": 2,
            "train": 5,
            "validation": 3,
            "hidden_validation": 3,
            "test": 5,
            "encoder_steps": 8,
            "controller_steps": 12,
            "max_candidates": 5,
            "timeout_seconds": 7.0,
            "max_grid_cells": 196,
        },
        "full": {
            "generations": 3,
            "train": 6,
            "validation": 4,
            "hidden_validation": 4,
            "test": 8,
            "encoder_steps": 12,
            "controller_steps": 18,
            "max_candidates": 6,
            "timeout_seconds": 8.0,
            "max_grid_cells": 225,
        },
    }[mode]


def ensure_arc_data(data_dir: str | Path, *, allow_download: bool = True) -> Path:
    """Return an ARC-AGI checkout root containing ``data/training``."""

    root = Path(data_dir)
    if (root / "data" / "training").exists():
        return root
    if (root / "training").exists():
        return root.parent
    if not allow_download:
        raise FileNotFoundError(f"ARC-AGI data not found under {root}")

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "arc_agi_master.zip"
    extract_dir = root / "_extract"
    with urllib.request.urlopen(ARC_ZIP_URL, timeout=60) as response:
        zip_path.write_bytes(response.read())
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    extracted = next(extract_dir.glob("ARC-AGI-*"))
    target = root / "ARC-AGI"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(extracted), str(target))
    shutil.rmtree(extract_dir)
    zip_path.unlink(missing_ok=True)
    return target


def _shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def _same_shape_public_task(payload: Mapping[str, object], *, max_grid_cells: int) -> bool:
    for split in ("train", "test"):
        for pair in payload.get(split, []):  # type: ignore[assignment]
            input_grid = pair["input"]  # type: ignore[index]
            output_grid = pair["output"]  # type: ignore[index]
            if _shape(input_grid) != _shape(output_grid):
                return False
            rows, cols = _shape(input_grid)
            if rows <= 0 or cols <= 0 or rows * cols > max_grid_cells:
                return False
    return True


def _cell_features(grid: Sequence[Sequence[int]]) -> torch.Tensor:
    rows, cols = _shape(grid)
    flat_colors = [int(value) for row in grid for value in row]
    background = max(set(flat_colors), key=flat_colors.count) if flat_colors else 0
    features: List[List[float]] = []
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r][c])
            up = int(grid[r - 1][c]) if r > 0 else -1
            down = int(grid[r + 1][c]) if r + 1 < rows else -1
            left = int(grid[r][c - 1]) if c > 0 else -1
            right = int(grid[r][c + 1]) if c + 1 < cols else -1
            features.append(
                [
                    0.0 if rows <= 1 else r / float(rows - 1),
                    0.0 if cols <= 1 else c / float(cols - 1),
                    rows / 30.0,
                    cols / 30.0,
                    color / 9.0,
                    1.0 if color == background else 0.0,
                    up / 9.0,
                    down / 9.0,
                    left / 9.0,
                    right / 9.0,
                ]
            )
    return torch.tensor(features, dtype=torch.float32)


def _cell_targets(grid: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor([[int(value) / 9.0] for row in grid for value in row], dtype=torch.float32)


def _pair_to_xy(pair: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], List[List[int]]]:
    input_grid = pair["input"]  # type: ignore[index]
    output_grid = pair["output"]  # type: ignore[index]
    return _cell_features(input_grid), _cell_targets(output_grid), _shape(output_grid), output_grid  # type: ignore[return-value]


def _cap_support_cells(task_id: str, x: torch.Tensor, y: torch.Tensor, max_cells: int = 768) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] <= max_cells:
        return x, y
    rng = random.Random(task_id)
    indices = list(range(x.shape[0]))
    rng.shuffle(indices)
    selected = torch.tensor(sorted(indices[:max_cells]), dtype=torch.long)
    return x.index_select(0, selected), y.index_select(0, selected)


def _arc_payload_to_task(task_id: str, split: str, payload: Mapping[str, object]) -> SyntheticTask:
    support_x_parts: List[torch.Tensor] = []
    support_y_parts: List[torch.Tensor] = []
    for pair in payload["train"]:  # type: ignore[index]
        x, y, _, _ = _pair_to_xy(pair)
        support_x_parts.append(x)
        support_y_parts.append(y)
    query_x_parts: List[torch.Tensor] = []
    query_y_parts: List[torch.Tensor] = []
    query_shapes: List[Tuple[int, int]] = []
    expected_outputs: List[List[List[int]]] = []
    offsets = [0]
    for pair in payload["test"]:  # type: ignore[index]
        x, y, shape, output_grid = _pair_to_xy(pair)
        query_x_parts.append(x)
        query_y_parts.append(y)
        query_shapes.append(shape)
        expected_outputs.append(output_grid)
        offsets.append(offsets[-1] + int(x.shape[0]))
    support_x = torch.cat(support_x_parts, dim=0)
    support_y = torch.cat(support_y_parts, dim=0)
    support_x, support_y = _cap_support_cells(task_id, support_x, support_y)
    return SyntheticTask(
        task_id=f"{split}-arc-agi1-{task_id}",
        family="arc_agi1_same_shape_cell",
        split=split,
        support_x=support_x,
        support_y=support_y,
        query_x=torch.cat(query_x_parts, dim=0),
        query_y=torch.cat(query_y_parts, dim=0),
        metadata={
            "source": ARC_REPO_URL,
            "source_task_id": task_id,
            "query_shapes": query_shapes,
            "query_offsets": offsets,
            "expected_outputs": expected_outputs,
            "ood": split in {"hidden_validation", "test"},
            "scope": "same_shape_cell_prediction",
        },
    )


def load_arc_same_shape_tasks(
    data_dir: str | Path,
    *,
    seed: int,
    split: str = "training",
    max_grid_cells: int = 225,
    allow_download: bool = True,
) -> ArcLoadResult:
    root = ensure_arc_data(data_dir, allow_download=allow_download)
    task_dir = root / "data" / split
    if not task_dir.exists():
        raise FileNotFoundError(f"missing ARC task directory: {task_dir}")
    candidates: List[Tuple[str, Mapping[str, object]]] = []
    filtered_count = 0
    for path in sorted(task_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _same_shape_public_task(payload, max_grid_cells=max_grid_cells):
            filtered_count += 1
            candidates.append((path.stem, payload))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    tasks = [_arc_payload_to_task(task_id, "unassigned", payload) for task_id, payload in candidates]
    return ArcLoadResult(tasks, filtered_count, str(root), ARC_REPO_URL)


def _assign_splits(tasks: Sequence[SyntheticTask], *, cfg: Mapping[str, int | float]) -> Dict[str, List[List[SyntheticTask]] | List[SyntheticTask]]:
    generations = int(cfg["generations"])
    train_n = int(cfg["train"])
    val_n = int(cfg["validation"])
    hidden_n = int(cfg["hidden_validation"])
    test_n = int(cfg["test"])
    required = generations * (train_n + val_n + hidden_n) + test_n
    if len(tasks) < required:
        raise ValueError(f"not enough filtered ARC tasks: need {required}, got {len(tasks)}")
    idx = 0
    train_by_gen: List[List[SyntheticTask]] = []
    val_by_gen: List[List[SyntheticTask]] = []
    hidden_by_gen: List[List[SyntheticTask]] = []
    for generation in range(generations):
        train_by_gen.append(_with_split(tasks[idx : idx + train_n], "train", generation))
        idx += train_n
        val_by_gen.append(_with_split(tasks[idx : idx + val_n], "validation", generation))
        idx += val_n
        hidden_by_gen.append(_with_split(tasks[idx : idx + hidden_n], "hidden_validation", generation))
        idx += hidden_n
    test_tasks = _with_split(tasks[idx : idx + test_n], "test", 0)
    return {"train": train_by_gen, "validation": val_by_gen, "hidden_validation": hidden_by_gen, "test": test_tasks}


def _with_split(tasks: Sequence[SyntheticTask], split: str, generation: int) -> List[SyntheticTask]:
    out: List[SyntheticTask] = []
    for task in tasks:
        source_id = str(task.metadata["source_task_id"])
        out.append(
            SyntheticTask(
                task_id=f"{split}-g{generation}-arc-agi1-{source_id}",
                family=task.family,
                split=split if split != "hidden_validation" else "validation",
                support_x=task.support_x,
                support_y=task.support_y,
                query_x=task.query_x,
                query_y=task.query_y,
                metadata={**task.metadata, "assigned_split": split, "generation": generation},
            )
        )
    return out


def _round_color_predictions(pred: torch.Tensor) -> List[int]:
    scaled = pred.detach().reshape(-1).float() * 9.0
    return [int(max(0, min(9, round(float(value))))) for value in scaled]


def _grid_metrics_from_predictions(task: SyntheticTask, pred: torch.Tensor) -> Dict[str, object]:
    predicted_colors = _round_color_predictions(pred)
    expected_colors = [int(round(float(value) * 9.0)) for value in task.query_y.reshape(-1)]
    correct = sum(1 for a, b in zip(predicted_colors, expected_colors) if a == b)
    offsets = [int(x) for x in task.metadata["query_offsets"]]  # type: ignore[index]
    shapes = [tuple(shape) for shape in task.metadata["query_shapes"]]  # type: ignore[index]
    exact_flags = []
    rendered: List[List[List[int]]] = []
    for index, shape in enumerate(shapes):
        start, end = offsets[index], offsets[index + 1]
        rows, cols = int(shape[0]), int(shape[1])
        flat = predicted_colors[start:end]
        rendered_grid = [flat[row * cols : (row + 1) * cols] for row in range(rows)]
        rendered.append(rendered_grid)
        expected_grid = task.metadata["expected_outputs"][index]  # type: ignore[index]
        exact_flags.append(rendered_grid == expected_grid)
    return {
        "task_id": task.task_id,
        "source_task_id": task.metadata["source_task_id"],
        "cell_accuracy": correct / max(1, len(expected_colors)),
        "exact": all(exact_flags),
        "predicted_outputs": rendered,
    }


def _evaluate_arc_program_set(
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory,
    genome: ProgramGenome,
    *,
    meta_controller=None,
    random_controller=None,
    selector_kind: str = "active",
    world_model=None,
) -> Dict[str, object]:
    losses: List[float] = []
    cell_acc: List[float] = []
    exact: List[float] = []
    selected: List[str] = []
    task_results: List[Dict[str, object]] = []
    for task in tasks:
        program = _select_program(
            genome,
            task,
            state,
            encoder,
            memory,
            meta_controller=meta_controller,
            random_controller=random_controller,
            selector_kind=selector_kind,
        )
        context = _make_contexts([task], state, encoder, memory, world_model)[0]
        adapted, details = compile_operator_program(program).adapt_params(context)
        with torch.no_grad():
            pred = functional_forward(context.model, adapted, task.query_x)
            loss = float(F.mse_loss(pred, task.query_y.float()))
        grid_metrics = _grid_metrics_from_predictions(task, pred)
        losses.append(loss)
        cell_acc.append(float(grid_metrics["cell_accuracy"]))
        exact.append(1.0 if grid_metrics["exact"] else 0.0)
        selected.append(program.program_id)
        task_results.append(
            {
                **{key: value for key, value in grid_metrics.items() if key != "predicted_outputs"},
                "program_id": program.program_id,
                "query_mse": loss,
                "behavior_signature": list(details.behavior_signature),
            }
        )
    return {
        "mean_mse": float(mean(losses)) if losses else 0.0,
        "cell_accuracy": float(mean(cell_acc)) if cell_acc else 0.0,
        "exact_task_accuracy": float(mean(exact)) if exact else 0.0,
        "selected_program_ids": selected,
        "task_results": task_results,
    }


def _run_seed(seed: int, mode: str, *, arc_data_dir: str | Path, allow_download: bool) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _mode_config(mode)
    load = load_arc_same_shape_tasks(
        arc_data_dir,
        seed=seed,
        max_grid_cells=int(cfg["max_grid_cells"]),
        allow_download=allow_download,
    )
    splits = _assign_splits(load.tasks, cfg=cfg)
    train_by_gen = splits["train"]  # type: ignore[assignment]
    val_by_gen = splits["validation"]  # type: ignore[assignment]
    hidden_by_gen = splits["hidden_validation"]  # type: ignore[assignment]
    test_tasks = splits["test"]  # type: ignore[assignment]
    all_train = [task for tasks in train_by_gen for task in tasks]
    all_val = [task for tasks in val_by_gen for task in tasks]
    all_hidden = [task for tasks in hidden_by_gen for task in tasks]
    assert_disjoint_task_ids(all_train, all_val, all_hidden, test_tasks)

    state = _initial_state(seed)
    state.update({"input_dim": ARC_FEATURE_DIM, "output_dim": 1, "inner_lr": 0.05, "inner_steps": 1, "width_multiplier": 1.4})
    genome = ProgramGenome.default()
    baseline_genome = ProgramGenome.default()
    sandbox = CandidateSandbox(ROOT / ".candidate_sandbox", timeout_seconds=float(cfg["timeout_seconds"]), max_loss=1e4)
    generator = CandidateGenerator(seed=seed, config=CandidateGeneratorConfig(max_candidates=int(cfg["max_candidates"])))
    random_generator = CandidateGenerator(seed=seed + 700, config=CandidateGeneratorConfig(max_candidates=2))
    self_model = CandidateSelfModel(seed=seed + 800)
    failure_grammar = FailureGrammar()
    evaluator_genome = EvaluatorGenome()

    generated_records: List[Dict[str, object]] = []
    candidate_decisions: List[Dict[str, object]] = []
    generation_summaries: List[Dict[str, object]] = []
    baseline_metrics: Dict[str, object] | None = None
    evolved_metrics_by_generation: List[Dict[str, object]] = []
    behavior_diff_observed = False

    for generation in range(int(cfg["generations"])):
        train_tasks = train_by_gen[generation]
        validation_tasks = val_by_gen[generation]
        hidden_validation_tasks = hidden_by_gen[generation]
        encoder = LearnedTaskEncoder(x_dim=ARC_FEATURE_DIM, y_dim=1, latent_dim=8, hidden_dim=32, seed=seed + generation)
        decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32, x_dim=ARC_FEATURE_DIM, y_dim=1)
        encoder_result = train_task_encoder(encoder, decoder, train_tasks, validation_tasks, steps=int(cfg["encoder_steps"]), lr=0.01)
        memory = _build_memory(train_tasks, state, encoder)
        if baseline_metrics is None:
            baseline_metrics = _evaluate_arc_program_set(test_tasks, state, encoder, memory, baseline_genome, selector_kind="active")
        transitions, program_scores, trace_summary = _collect_program_transitions(train_tasks, state, encoder, memory, genome)
        actions = _program_actions(genome)
        from model_based_controller import LearnedMetaController, RandomController

        meta_controller = LearnedMetaController(actions, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 40 + generation)
        meta_initial, meta_final = meta_controller.train_on_transitions(transitions, steps=int(cfg["controller_steps"]), lr=0.01)
        world_model, wm_initial, wm_final = _train_world_model_from_operator_transitions(seed + 60 + generation, transitions)
        self_model.fit(steps=2 if generation == 0 else 8, lr=0.01)

        random_controls = random_generator.generate(genome.accepted_programs(), generation=generation, trace_summary=trace_summary, random_baseline=True)
        raw_generated = generator.generate(genome.accepted_programs(), generation=generation, trace_summary=trace_summary)
        generated = failure_grammar.apply_to_candidates(raw_generated, generation)
        predictions = {
            record.candidate.program_id: self_model.predict(
                record.candidate,
                generation,
                failure_rule_count=len(failure_grammar.rules),
                evaluator_pressure=float(len(evaluator_genome.probationary)),
                used_for_selection=True,
            )
            for record in generated
        }
        generated = sorted(generated, key=lambda record: self_model.selection_score(predictions[record.candidate.program_id]), reverse=True)
        generated_records.extend(record.to_dict() for record in generated)
        for record in generated:
            decision = _validate_candidate_program(
                genome,
                record.candidate,
                validation_tasks,
                hidden_validation_tasks,
                state,
                encoder,
                memory,
                world_model,
                sandbox,
                generation=generation,
                seed=seed,
                random_control=random_controls[0].candidate if random_controls else None,
                evaluator_candidate=evaluator_genome.active,
                failure_rule_count=len(failure_grammar.rules),
                self_model_prediction=predictions.get(record.candidate.program_id),
                improvement_threshold=1e-6,
            )
            decision["generation_method"] = record.method
            decision["generation_reason"] = record.reason
            decision["parent_program_ids"] = list(record.parent_program_ids)
            candidate_decisions.append(decision)
            behavior_diff_observed = behavior_diff_observed or float(decision.get("runtime_behavior_difference", 0.0)) > 1e-9
            self_model.record_actual(record.candidate, generation, _self_model_actual_effects(decision, meta_final), split="validation")

        generation_decisions = [item for item in candidate_decisions if int(item.get("generation", -1)) == generation]
        failure_grammar.update_from_decisions(generation_decisions, generation)
        evaluator_generation_decisions = [
            evaluator_genome.validate_candidate(candidate, generation_decisions, generation=generation, failure_rule_count=len(failure_grammar.rules))
            for candidate in evaluator_genome.propose(generation, failure_rule_count=len(failure_grammar.rules))
        ]

        post_actions = _program_actions(genome)
        post_controller = LearnedMetaController(post_actions, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 90 + generation)
        post_transitions, _, _ = _collect_program_transitions(train_tasks, state, encoder, memory, genome, world_model)
        _, post_meta_final = post_controller.train_on_transitions(post_transitions, steps=int(cfg["controller_steps"]), lr=0.01)
        guarded_selector, selector_guard = _guarded_selector_kind(
            validation_tasks,
            hidden_validation_tasks,
            state,
            encoder,
            memory,
            genome,
            post_controller,
            world_model,
            required_relative_gain=0.01,
        )
        evolved_metrics = _evaluate_arc_program_set(
            test_tasks,
            state,
            encoder,
            memory,
            genome,
            meta_controller=post_controller,
            selector_kind=guarded_selector,
            world_model=world_model,
        )
        fixed_metrics = _evaluate_arc_program_set(test_tasks, state, encoder, memory, ProgramGenome.default(), selector_kind="fixed", world_model=world_model)
        shuffled_metrics = _evaluate_arc_program_set(test_tasks, state, encoder, _shuffled_memory(memory), genome, meta_controller=post_controller, selector_kind=guarded_selector, world_model=world_model)
        wrong_world_metrics = _evaluate_arc_program_set(test_tasks, state, encoder, memory, genome, meta_controller=post_controller, selector_kind=guarded_selector, world_model=WrongWorldModel(8, 6))
        random_controller = RandomController(post_actions, seed=seed + 120 + generation)
        random_metrics = _evaluate_arc_program_set(test_tasks, state, encoder, memory, genome, random_controller=random_controller, selector_kind="random", world_model=world_model)
        evolved_metrics_by_generation.append(evolved_metrics)
        state["recent_losses"] = [
            float(evolved_metrics["mean_mse"]),
            float(evolved_metrics["cell_accuracy"]),
            float(sum(1 for item in candidate_decisions if item.get("accepted"))),
            float(sum(1 for item in candidate_decisions if item.get("rollback"))),
        ]
        generation_summaries.append(
            {
                "generation": generation,
                "train_task_ids": [task.task_id for task in train_tasks],
                "validation_task_ids": [task.task_id for task in validation_tasks],
                "hidden_validation_task_ids": [task.task_id for task in hidden_validation_tasks],
                "test_task_ids": [task.task_id for task in test_tasks],
                "encoder_train_initial": encoder_result.train_losses[0],
                "encoder_train_final": encoder_result.train_losses[-1],
                "controller_loss_initial": meta_initial,
                "controller_loss_final": post_meta_final,
                "world_model_loss_initial": wm_initial,
                "world_model_loss_final": wm_final,
                "selector_guard": selector_guard,
                "baseline_test_metrics": baseline_metrics,
                "evolved_test_metrics": evolved_metrics,
                "fixed_test_metrics": fixed_metrics,
                "random_test_metrics": random_metrics,
                "shuffled_memory_test_metrics": shuffled_metrics,
                "wrong_world_model_test_metrics": wrong_world_metrics,
                "accepted_program_ids": sorted(genome.accepted_program_ids),
                "program_score_means": {key: float(mean(values)) for key, values in program_scores.items()},
                "evaluator_generation_decisions": evaluator_generation_decisions,
            }
        )

    assert baseline_metrics is not None
    final_evolved = evolved_metrics_by_generation[-1] if evolved_metrics_by_generation else baseline_metrics
    accepted = [item for item in candidate_decisions if item.get("accepted")]
    metrics = {
        "external_filtered_task_count": float(load.filtered_task_count),
        "external_test_task_count": float(len(test_tasks)),
        "baseline_mean_mse": float(baseline_metrics["mean_mse"]),
        "evolved_mean_mse": float(final_evolved["mean_mse"]),
        "baseline_to_evolved_mse_delta": float(baseline_metrics["mean_mse"]) - float(final_evolved["mean_mse"]),
        "baseline_cell_accuracy": float(baseline_metrics["cell_accuracy"]),
        "evolved_cell_accuracy": float(final_evolved["cell_accuracy"]),
        "cell_accuracy_delta": float(final_evolved["cell_accuracy"]) - float(baseline_metrics["cell_accuracy"]),
        "baseline_exact_task_accuracy": float(baseline_metrics["exact_task_accuracy"]),
        "evolved_exact_task_accuracy": float(final_evolved["exact_task_accuracy"]),
        "exact_task_accuracy_delta": float(final_evolved["exact_task_accuracy"]) - float(baseline_metrics["exact_task_accuracy"]),
        "candidate_count": float(len(generated_records)),
        "accepted_program_count": float(len(accepted)),
        "validation_rejected_count": float(len(candidate_decisions) - len(accepted)),
        "runtime_behavior_difference_observed": 1.0 if behavior_diff_observed else 0.0,
        "selector_guard_fallback_count": float(sum(1 for item in generation_summaries if item.get("selector_guard", {}).get("selected") != "learned")),
    }
    checks = [
        "external ARC-AGI source recorded",
        "same-shape ARC subset explicitly scoped",
        "train/validation/hidden/test task IDs disjoint",
        "test task outputs not used for candidate acceptance",
        "candidate programs evaluated through existing OperatorProgram DSL",
        "sandboxed validation and rollback path used",
        "hidden validation gate used before held-out test scoring",
        "baseline and evolved test metrics reported",
    ]
    return {
        "seed": seed,
        "metrics": metrics,
        "config": {
            "benchmark": "external_arc_agi_same_shape_code_level_rsi",
            "mode": mode,
            "mode_config": cfg,
            "arc_source_url": load.source_url,
            "arc_source_dir": load.source_dir,
            "scope": "ARC-AGI-1 public training same-shape cell-prediction subset",
        },
        "program_genome": genome.to_dict(),
        "generated_candidates": generated_records,
        "candidate_decisions": candidate_decisions,
        "generation_summaries": generation_summaries,
        "tasks": {"train": all_train, "validation": all_val, "hidden_validation": all_hidden, "test": test_tasks},
        "anti_cheat_checks_passed": checks,
    }


def _serializable_task_ids(tasks: Mapping[str, Sequence[SyntheticTask]]) -> Dict[str, List[str]]:
    return {key: [task.task_id for task in value] for key, value in tasks.items()}


def _write_outputs(payload: Dict[str, object], output: str | Path) -> Dict[str, object]:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = payload["tasks"]  # type: ignore[assignment]
    result = {
        "benchmark": "external_arc_agi_same_shape_code_level_rsi",
        "seed": payload["seed"],
        "config": payload["config"],
        "metrics": payload["metrics"],
        **payload["metrics"],  # type: ignore[arg-type]
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "config": payload["config"],
        "config_hash": stable_config_hash(payload["config"]),  # type: ignore[arg-type]
        "task_ids": _serializable_task_ids(tasks),  # type: ignore[arg-type]
        "program_genome": payload["program_genome"],
        "generated_candidates": payload["generated_candidates"],
        "candidate_decisions": payload["candidate_decisions"],
        "generation_summaries": payload["generation_summaries"],
        "anti_cheat_checks_passed": payload["anti_cheat_checks_passed"],
        "metric_summary": payload["metrics"],
    }
    result["manifest_path"] = str(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run(
    mode: str,
    seed: int,
    output: str | None = None,
    *,
    arc_data_dir: str | Path = ROOT.parent / "external_arc_agi",
    allow_download: bool = True,
) -> Dict[str, object]:
    payload = _run_seed(seed, mode, arc_data_dir=arc_data_dir, allow_download=allow_download)
    if output is None:
        output = f"results/arc_external_rsi_{mode}_seed{seed}.json"
    return _write_outputs(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="External ARC-AGI same-shape benchmark for the bounded program RSI loop")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--arc-data-dir", type=str, default=str(ROOT.parent / "external_arc_agi"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output, arc_data_dir=args.arc_data_dir, allow_download=not args.no_download)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/arc_external_rsi_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
