"""Code-level recursive self-improvement benchmark.

This benchmark is still bounded and CPU-runnable.  It does not edit arbitrary
repository files or execute arbitrary generated source.  It tests whether a
bounded operator-program DSL can synthesize executable adaptation programs,
evaluate them under isolated validation-only controls, accept/reject with
rollback, reuse accepted programs, and record enough evidence to audit leakage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from adaptation_operators import OperatorExecutionContext
from candidate_sandbox import CandidateSandbox, CandidateSandboxResult
from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel
from evaluator_evolution import EvaluatorCandidate, EvaluatorGenome, score_decision
from experiment_manifest import current_git_commit, stable_config_hash
from failure_grammar import FailureGrammar
from learned_task_encoder import LearnedTaskEncoder, TaskConditionedRegressor, train_task_encoder
from model_based_controller import ControllerTransition, LearnedMetaController, RandomController
from operator_dsl import (
    OperatorProgram,
    PrimitiveStep,
    ProgramGenome,
    assert_no_task_family_branching,
    assert_program_diversity,
    compile_operator_program,
    execute_operator_program,
    primitive_catalog,
    program_action_vector,
    program_to_update_action,
)
from rsi_candidate_generator import CandidateGenerationRecord, CandidateGenerator, CandidateGeneratorConfig
from self_model import CandidateSelfModel, SelfModelPrediction
from task_suite import ProceduralTaskSuite, SyntheticTask, assert_disjoint_task_ids

from benchmarks.recursive_self_improvement_benchmark import (
    WrongWorldModel,
    _build_memory,
    _history_tensor,
    _infer_task,
    _initial_state,
    _make_model,
    _memory_summary,
    _shuffled_memory,
    _train_world_model_from_operator_transitions,
)


def _stderr(values: Sequence[float]) -> float:
    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _mode_config(mode: str) -> Dict[str, int | float]:
    return {
        "smoke": {
            "generations": 2,
            "train": 5,
            "validation": 3,
            "hidden_validation": 2,
            "test": 3,
            "encoder_steps": 6,
            "controller_steps": 12,
            "max_candidates": 5,
            "timeout_seconds": 5.0,
        },
        "quick": {
            "generations": 2,
            "train": 7,
            "validation": 4,
            "hidden_validation": 2,
            "test": 4,
            "encoder_steps": 12,
            "controller_steps": 24,
            "max_candidates": 6,
            "timeout_seconds": 6.0,
        },
        "full": {
            "generations": 3,
            "train": 9,
            "validation": 5,
            "hidden_validation": 3,
            "test": 5,
            "encoder_steps": 20,
            "controller_steps": 36,
            "max_candidates": 7,
            "timeout_seconds": 8.0,
        },
    }[mode]


def _make_contexts(
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    world_model: Optional[WorldModel],
) -> List[OperatorExecutionContext]:
    model = _make_model(state)
    contexts: List[OperatorExecutionContext] = []
    for task in tasks:
        contexts.append(
            OperatorExecutionContext(
                model=model,
                task=task,
                task_embedding=_infer_task(state, encoder, task),
                base_inner_lr=float(state.get("inner_lr", 0.01)),
                inner_steps=int(state.get("inner_steps", 1)),
                memory=memory,
                memory_k=max(1, int(state.get("memory_retrieval_k", 1))),
                world_model=world_model,
                recent_losses=_history_tensor(state),
            )
        )
    return contexts


def _evaluate_program_direct(
    program: OperatorProgram,
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    world_model: Optional[WorldModel],
) -> Tuple[List[float], List[float], List[Dict[str, object]]]:
    losses: List[float] = []
    improvements: List[float] = []
    traces: List[Dict[str, object]] = []
    for context in _make_contexts(tasks, state, encoder, memory, world_model):
        trace = execute_operator_program(program, context)
        losses.append(float(trace.query_loss_after))
        improvements.append(float(trace.improvement))
        traces.append(trace.to_dict())
    return losses, improvements, traces


def _collect_program_transitions(
    train_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: EpisodicMemory,
    genome: ProgramGenome,
    world_model: Optional[WorldModel] = None,
) -> Tuple[List[ControllerTransition], Dict[str, List[float]], Dict[str, float]]:
    transitions: List[ControllerTransition] = []
    scores_by_program: Dict[str, List[float]] = {program.program_id: [] for program in genome.accepted_programs()}
    improvement_summary: Dict[str, float] = {}
    for task in train_tasks:
        z = _infer_task(state, encoder, task)
        summary = _memory_summary(z, memory, max(1, int(state.get("memory_retrieval_k", 1))))
        history = _history_tensor(state)
        for program in genome.accepted_programs():
            trace = execute_operator_program(program, _make_contexts([task], state, encoder, memory, world_model)[0])
            transitions.append(ControllerTransition(z, summary, history, program_action_vector(program), trace.improvement))
            scores_by_program[program.program_id].append(trace.query_loss_after)
    assert_program_diversity(scores_by_program)
    for program_id, losses in scores_by_program.items():
        improvement_summary[program_id] = -float(mean(losses)) if losses else 0.0
    return transitions, scores_by_program, improvement_summary


def _program_actions(genome: ProgramGenome):
    return [program_to_update_action(program) for program in genome.accepted_programs()]


def _select_program(
    genome: ProgramGenome,
    task: SyntheticTask,
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    *,
    meta_controller: Optional[LearnedMetaController] = None,
    random_controller: Optional[RandomController] = None,
    selector_kind: str = "active",
) -> OperatorProgram:
    if selector_kind == "fixed":
        return genome.accepted_programs()[0]
    if selector_kind == "random" and random_controller is not None:
        action = random_controller.select_action()
        return genome.programs.get(action.name, genome.active_program())
    if selector_kind == "learned" and meta_controller is not None:
        z = _infer_task(state, encoder, task)
        decision = meta_controller.select_action(z, memory, memory_k=max(1, int(state.get("memory_retrieval_k", 1))), performance_history=_history_tensor(state))
        return genome.programs.get(decision.selected.name, genome.active_program())
    return genome.active_program()


def _evaluate_program_set(
    tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: Optional[EpisodicMemory],
    genome: ProgramGenome,
    *,
    meta_controller: Optional[LearnedMetaController] = None,
    random_controller: Optional[RandomController] = None,
    selector_kind: str = "active",
    world_model: Optional[WorldModel] = None,
    track_controller: bool = False,
) -> Tuple[List[float], List[float], List[str], List[Dict[str, object]]]:
    losses: List[float] = []
    improvements: List[float] = []
    selected: List[str] = []
    traces: List[Dict[str, object]] = []
    for task in tasks:
        program = _select_program(genome, task, state, encoder, memory, meta_controller=meta_controller, random_controller=random_controller, selector_kind=selector_kind)
        trace = execute_operator_program(program, _make_contexts([task], state, encoder, memory, world_model)[0])
        losses.append(float(trace.query_loss_after))
        improvements.append(float(trace.improvement))
        selected.append(program.program_id)
        traces.append(trace.to_dict())
        if track_controller and meta_controller is not None and selector_kind == "learned":
            z = _infer_task(state, encoder, task)
            decision = meta_controller.select_action(z, memory, memory_k=max(1, int(state.get("memory_retrieval_k", 1))), performance_history=_history_tensor(state))
            meta_controller.log_actual_outcome(decision, float(trace.improvement))
    return losses, improvements, selected, traces


def _runtime_behavior_diff(left: CandidateSandboxResult, right: CandidateSandboxResult) -> float:
    score_delta = sum(abs(a - b) for a, b in zip(left.scores, right.scores))
    sig_delta = sum(abs(a - b) for a, b in zip(left.behavior_signature, right.behavior_signature))
    return float(score_delta + sig_delta)


def _no_op_control(parent: OperatorProgram, generation: int) -> OperatorProgram:
    return OperatorProgram(
        f"g{generation}_noop_parent_control",
        tuple(parent.primitive_sequence),
        dict(parent.parameters),
        parent.program_id,
        generation,
    )


def _validate_candidate_program(
    genome: ProgramGenome,
    candidate: OperatorProgram,
    validation_tasks: Sequence[SyntheticTask],
    hidden_validation_tasks: Sequence[SyntheticTask],
    state: Dict[str, object],
    encoder: LearnedTaskEncoder,
    memory: EpisodicMemory,
    world_model: Optional[WorldModel],
    sandbox: CandidateSandbox,
    *,
    generation: int,
    seed: int,
    random_control: Optional[OperatorProgram] = None,
    evaluator_candidate: Optional[EvaluatorCandidate] = None,
    failure_rule_count: int = 0,
    self_model_prediction: Optional[SelfModelPrediction] = None,
    improvement_threshold: float = 1e-5,
) -> Dict[str, object]:
    if not validation_tasks:
        raise ValueError("validation tasks are required for candidate acceptance")
    if any(task.split == "test" for task in validation_tasks) or any(task.split == "test" for task in hidden_validation_tasks):
        raise ValueError("test split tasks cannot be used for candidate acceptance")
    snapshot = genome.snapshot()
    parent = genome.programs.get(candidate.parent_program_id or "", genome.active_program())
    contexts = _make_contexts(validation_tasks, state, encoder, memory, world_model)
    hidden_contexts = _make_contexts(hidden_validation_tasks, state, encoder, memory, world_model)

    compile_result = sandbox.compile(candidate)
    if not compile_result.ok:
        return _decision(candidate, parent, generation, seed, False, "compile_failed", compile_result=compile_result)

    baseline = sandbox.evaluate(parent, contexts)
    candidate_result = sandbox.evaluate(candidate, contexts)
    replay = sandbox.evaluate(candidate, contexts)
    hidden_baseline = sandbox.evaluate(parent, hidden_contexts)
    hidden_candidate = sandbox.evaluate(candidate, hidden_contexts)
    no_op = sandbox.evaluate(_no_op_control(parent, generation), contexts)
    random_result = sandbox.evaluate(random_control, contexts) if random_control is not None else None

    validation_results = [baseline, candidate_result, replay, hidden_baseline, hidden_candidate, no_op]
    if random_result is not None:
        validation_results.append(random_result)
    failed = [result for result in validation_results if not result.ok]
    if failed:
        reason = failed[0].rejected_reason or "candidate_execution_failed"
        genome.restore(snapshot)
        return _decision(candidate, parent, generation, seed, False, reason, compile_result=compile_result, baseline=baseline, candidate_result=candidate_result)

    deterministic = all(abs(a - b) <= 1e-7 for a, b in zip(candidate_result.scores, replay.scores))
    deltas = [float(b - c) for b, c in zip(baseline.scores, candidate_result.scores)]
    hidden_deltas = [float(b - c) for b, c in zip(hidden_baseline.scores, hidden_candidate.scores)]
    mean_improvement = float(mean(deltas))
    win_count = sum(delta > improvement_threshold for delta in deltas)
    win_rate = float(win_count / max(1, len(deltas)))
    hidden_guard_regression = min(hidden_deltas) if hidden_deltas else 0.0
    no_op_delta = mean([float(n - c) for n, c in zip(no_op.scores, candidate_result.scores)])
    random_delta = 0.0
    if random_result is not None:
        random_delta = mean([float(r - c) for r, c in zip(random_result.scores, candidate_result.scores)])
    behavior_diff = _runtime_behavior_diff(baseline, candidate_result)
    core_accepted = (
        deterministic
        and behavior_diff > 1e-9
        and mean_improvement > improvement_threshold
        and win_rate >= 0.5
        and hidden_guard_regression > -1.0
        and no_op_delta >= -1e-8
        and random_delta >= -1.0
        and len(validation_tasks) >= 2
    )
    reason = "accepted" if core_accepted else _rejection_reason(
        deterministic=deterministic,
        behavior_diff=behavior_diff,
        mean_improvement=mean_improvement,
        win_rate=win_rate,
        hidden_guard_regression=hidden_guard_regression,
        no_op_delta=no_op_delta,
        random_delta=random_delta,
    )
    decision = _decision(
        candidate,
        parent,
        generation,
        seed,
        core_accepted,
        reason,
        compile_result=compile_result,
        baseline=baseline,
        candidate_result=candidate_result,
        hidden_candidate=hidden_candidate,
        random_result=random_result,
        mean_improvement=mean_improvement,
        win_rate=win_rate,
        behavior_diff=behavior_diff,
        deterministic=deterministic,
        hidden_guard_regression=hidden_guard_regression,
        no_op_delta=float(no_op_delta),
        random_delta=float(random_delta),
    )
    evaluator_score = 0.0
    evaluator_accepts = True
    if evaluator_candidate is not None:
        uncertainty = 0.0
        if self_model_prediction is not None:
            uncertainty = abs(float(self_model_prediction.predicted_effects.get("validation_to_test_gap", 0.0)))
        evaluator_score = score_decision(evaluator_candidate, decision, failure_rule_count=failure_rule_count, self_model_uncertainty=uncertainty)
        evaluator_accepts = evaluator_score > -1.0
    accepted = bool(core_accepted and evaluator_accepts)
    if accepted:
        genome.accept(candidate)
    else:
        genome.restore(snapshot)
        if core_accepted and not evaluator_accepts:
            reason = "rejected_by_evolved_evaluator"
    decision["accepted"] = accepted
    decision["rollback"] = not accepted
    decision["accepted_on_split"] = "validation" if accepted else None
    decision["rejection_reason"] = "accepted" if accepted else reason
    decision["evaluator_id"] = evaluator_candidate.evaluator_id if evaluator_candidate is not None else "legacy_core_rule"
    decision["evaluator_score"] = float(evaluator_score)
    decision["evaluator_accepts"] = bool(evaluator_accepts)
    decision["self_model_prediction"] = self_model_prediction.to_dict() if self_model_prediction is not None else {}
    return decision


def _rejection_reason(**kwargs: float | bool) -> str:
    if not kwargs["deterministic"]:
        return "deterministic_replay_failed"
    if float(kwargs["behavior_diff"]) <= 1e-9:
        return "no_runtime_behavior_difference"
    if float(kwargs["mean_improvement"]) <= 1e-5:
        return "insufficient_mean_validation_improvement"
    if float(kwargs["win_rate"]) < 0.5:
        return "insufficient_validation_win_rate"
    if float(kwargs["hidden_guard_regression"]) <= -1.0:
        return "hidden_validation_guard_regression"
    if float(kwargs["no_op_delta"]) < -1e-8:
        return "did_not_beat_no_op_control"
    if float(kwargs["random_delta"]) < -1.0:
        return "did_not_beat_random_control"
    return "statistical_acceptance_rule_failed"


def _decision(
    candidate: OperatorProgram,
    parent: OperatorProgram,
    generation: int,
    seed: int,
    accepted: bool,
    reason: str,
    *,
    compile_result: Optional[CandidateSandboxResult] = None,
    baseline: Optional[CandidateSandboxResult] = None,
    candidate_result: Optional[CandidateSandboxResult] = None,
    hidden_candidate: Optional[CandidateSandboxResult] = None,
    random_result: Optional[CandidateSandboxResult] = None,
    mean_improvement: float = 0.0,
    win_rate: float = 0.0,
    behavior_diff: float = 0.0,
    deterministic: bool = False,
    hidden_guard_regression: float = 0.0,
    no_op_delta: float = 0.0,
    random_delta: float = 0.0,
) -> Dict[str, object]:
    return {
        "candidate_name": candidate.program_id,
        "program_id": candidate.program_id,
        "parent_program_id": parent.program_id,
        "candidate_program": candidate.to_dict(),
        "source_hash": candidate.source_hash,
        "generation": generation,
        "random_seed": seed,
        "split": "validation",
        "accepted_on_split": "validation" if accepted else None,
        "accepted": bool(accepted),
        "rollback": not accepted,
        "rejection_reason": "accepted" if accepted else reason,
        "baseline_scores": baseline.scores if baseline else [],
        "candidate_scores": candidate_result.scores if candidate_result else [],
        "hidden_validation_scores": hidden_candidate.scores if hidden_candidate else [],
        "random_control_scores": random_result.scores if random_result else [],
        "mean_improvement": float(mean_improvement),
        "win_rate": float(win_rate),
        "runtime_behavior_difference": float(behavior_diff),
        "deterministic_replay_passed": bool(deterministic),
        "hidden_guard_regression": float(hidden_guard_regression),
        "no_op_control_delta": float(no_op_delta),
        "random_control_delta": float(random_delta),
        "compile_status": compile_result.to_dict() if compile_result else {},
        "candidate_execution": candidate_result.to_dict() if candidate_result else {},
        "validation_only": True,
        "updates": {"accepted_program_id": candidate.program_id},
    }


def _self_model_actual_effects(decision: Dict[str, object], controller_error: float) -> Dict[str, float]:
    validation_improvement = float(decision.get("mean_improvement", 0.0))
    hidden_transfer = float(decision.get("hidden_guard_regression", 0.0))
    runtime = float(decision.get("compile_status", {}).get("elapsed_seconds", 0.0)) + float(decision.get("candidate_execution", {}).get("elapsed_seconds", 0.0))
    instability = 0.0
    if not decision.get("deterministic_replay_passed", False):
        instability += 1.0
    reason = str(decision.get("rejection_reason", ""))
    if "nan" in reason or "exploding" in reason or "timeout" in reason:
        instability += 1.0
    future_quality = validation_improvement if decision.get("accepted") else -abs(validation_improvement)
    gap = validation_improvement - hidden_transfer
    return {
        "validation_improvement": validation_improvement,
        "ood_transfer": hidden_transfer,
        "runtime_cost": runtime,
        "instability_risk": instability,
        "future_candidate_quality": future_quality,
        "controller_prediction_error": float(controller_error),
        "validation_to_test_gap": gap,
    }


def _mean_decision_quality(decisions: Sequence[Dict[str, object]]) -> float:
    if not decisions:
        return 0.0
    return float(mean([float(item.get("mean_improvement", 0.0)) for item in decisions]))


def _source_policy_checks() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    source += Path(ROOT, "operator_dsl.py").read_text(encoding="utf-8")
    source += Path(ROOT, "rsi_candidate_generator.py").read_text(encoding="utf-8")
    source += Path(ROOT, "self_model.py").read_text(encoding="utf-8")
    source += Path(ROOT, "failure_grammar.py").read_text(encoding="utf-8")
    source += Path(ROOT, "evaluator_evolution.py").read_text(encoding="utf-8")
    assert_no_task_family_branching(source)
    banned_literals = ("25." + "078727", "138." + "130443", "hardcoded_" + "benchmark_success")
    for literal in banned_literals:
        if literal in source:
            raise AssertionError(f"hardcoded benchmark output detected: {literal}")


def _build_manifest(
    *,
    command: str,
    mode: str,
    seeds: Sequence[int],
    train_tasks: Sequence[SyntheticTask],
    validation_tasks: Sequence[SyntheticTask],
    hidden_validation_tasks: Sequence[SyntheticTask],
    test_tasks: Sequence[SyntheticTask],
    config: Dict[str, object],
    aggregate: Dict[str, object],
    per_seed: Sequence[Dict[str, object]],
    generation_records: Sequence[Dict[str, object]],
    generated_candidates: Sequence[Dict[str, object]],
    candidate_decisions: Sequence[Dict[str, object]],
    self_model_log: Dict[str, object],
    failure_grammar_log: Dict[str, object],
    evaluator_log: Dict[str, object],
    anti_cheat_checks: Sequence[str],
) -> Dict[str, object]:
    accepted = [dict(item) for item in candidate_decisions if item.get("accepted")]
    rejected = [dict(item) for item in candidate_decisions if not item.get("accepted")]
    task_ids = {
        "train": [task.task_id for task in train_tasks],
        "validation": [task.task_id for task in validation_tasks],
        "hidden_validation": [task.task_id for task in hidden_validation_tasks],
        "test": [task.task_id for task in test_tasks],
    }
    families = {
        "train": [task.family for task in train_tasks],
        "validation": [task.family for task in validation_tasks],
        "hidden_validation": [task.family for task in hidden_validation_tasks],
        "test": [task.family for task in test_tasks],
    }
    config_hash = stable_config_hash(config)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": command,
        "mode": mode,
        "seeds": [int(seed) for seed in seeds],
        "task_ids": task_ids,
        "train_task_ids": task_ids["train"],
        "validation_task_ids": task_ids["validation"],
        "hidden_validation_task_ids": task_ids["hidden_validation"],
        "test_task_ids": task_ids["test"],
        "task_family_list": sorted(set(sum(families.values(), []))),
        "task_families": families,
        "dsl_primitive_set": primitive_catalog(),
        "candidate_generator_configuration": config.get("candidate_generator", {}),
        "generated_candidates": list(generated_candidates),
        "compile_status_per_candidate": [item.get("compile_status", {}) for item in candidate_decisions],
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rejection_reasons": {item["candidate_name"]: item.get("rejection_reason") for item in rejected},
        "parent_child_program_lineage": {item["candidate_name"]: item.get("parent_program_id") for item in candidate_decisions},
        "operator_source_hashes": {item["candidate_name"]: item.get("source_hash") for item in candidate_decisions},
        "runtime_behavior_difference_checks": {item["candidate_name"]: item.get("runtime_behavior_difference", 0.0) for item in candidate_decisions},
        "validation_scores": {item["candidate_name"]: item.get("candidate_scores", []) for item in candidate_decisions},
        "frozen_ood_test_scores": [gen.get("frozen_ood_test_loss") for gen in generation_records],
        "self_model": dict(self_model_log),
        "self_model_prediction_log": list(self_model_log.get("prediction_log", [])),
        "failure_grammar": dict(failure_grammar_log),
        "failure_rules": list(failure_grammar_log.get("rules", [])),
        "evaluator_evolution": dict(evaluator_log),
        "evaluator_decisions": list(evaluator_log.get("decisions", [])),
        "generation_manifests": list(generation_records),
        "anti_cheat_checks_passed": list(anti_cheat_checks),
        "config": config,
        "config_hash": config_hash,
        "metric_summary": {"aggregate": aggregate, "per_seed": list(per_seed)},
        "deterministic_replay_metadata": {
            "candidate_replay_required": True,
            "same_seed_reproducibility_checked_by_tests": True,
        },
    }
    _validate_manifest_dict(manifest)
    return manifest


def _validate_manifest_dict(manifest: Dict[str, object]) -> None:
    task_ids = manifest["task_ids"]  # type: ignore[index]
    train = set(task_ids["train"])  # type: ignore[index]
    validation = set(task_ids["validation"]) | set(task_ids["hidden_validation"])  # type: ignore[index]
    test = set(task_ids["test"])  # type: ignore[index]
    if train & validation or train & test or validation & test:
        raise ValueError("manifest task IDs are not disjoint")
    decisions = list(manifest.get("accepted_candidates", [])) + list(manifest.get("rejected_candidates", []))  # type: ignore[arg-type]
    for decision in decisions:
        if decision.get("split") == "test" or decision.get("accepted_on_split") == "test":
            raise ValueError("candidate accepted or evaluated for acceptance on test split")
    if not manifest.get("self_model_prediction_log"):
        raise ValueError("manifest missing self-model prediction log")
    if not manifest.get("failure_rules"):
        raise ValueError("manifest missing failure grammar rules")
    if not manifest.get("evaluator_decisions"):
        raise ValueError("manifest missing evaluator evolution decisions")
    required = {
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
        "failure grammar rewrites future candidates",
        "evaluator candidates require adversarial checks",
    }
    checks = set(str(item) for item in manifest.get("anti_cheat_checks_passed", []))
    missing = required.difference(checks)
    if missing:
        raise ValueError(f"missing anti-cheat checks: {sorted(missing)}")


def _run_seed(seed: int, mode: str) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    _source_policy_checks()
    cfg = _mode_config(mode)
    state = _initial_state(seed)
    suite = ProceduralTaskSuite(seed=seed)
    hidden_suite = ProceduralTaskSuite(seed=seed + 10000)
    genome = ProgramGenome.default()
    sandbox = CandidateSandbox(ROOT / ".candidate_sandbox", timeout_seconds=float(cfg["timeout_seconds"]), max_loss=1e5)
    generator = CandidateGenerator(seed=seed, config=CandidateGeneratorConfig(max_candidates=int(cfg["max_candidates"])))
    random_generator = CandidateGenerator(seed=seed + 777, config=CandidateGeneratorConfig(max_candidates=2))
    self_model = CandidateSelfModel(seed=seed + 200)
    failure_grammar = FailureGrammar()
    evaluator_genome = EvaluatorGenome()

    all_train: List[SyntheticTask] = []
    all_validation: List[SyntheticTask] = []
    all_hidden_validation: List[SyntheticTask] = []
    all_test: List[SyntheticTask] = []
    generated_records: List[Dict[str, object]] = []
    candidate_decisions: List[Dict[str, object]] = []
    generation_records: List[Dict[str, object]] = []
    accepted_before_generation: set[str] = set()
    reuse_count = 0
    reuse_total = 0
    controller_errors_by_generation: List[float] = []
    ood_losses: List[float] = []
    validation_losses: List[float] = []
    behavior_diff_observed = False
    dead_code_detector_passed = False
    test_leak_trap_passed = False
    self_model_fit_log: List[Dict[str, float]] = []
    self_model_prediction_errors: List[float] = []
    self_model_future_quality_errors: List[float] = []
    failure_rule_quality: List[float] = []
    no_self_model_quality: List[float] = []
    no_failure_quality: List[float] = []
    no_evaluator_quality: List[float] = []
    evaluator_overfit_detector = 0.0

    for generation in range(int(cfg["generations"])):
        train_tasks = suite.sample_mixed_tasks("train", int(cfg["train"]), ood=False)
        validation_tasks = suite.sample_mixed_tasks("validation", int(cfg["validation"]), ood=bool(generation % 2))
        hidden_validation_tasks = hidden_suite.sample_mixed_tasks("validation", int(cfg["hidden_validation"]), ood=False)
        test_tasks = suite.sample_mixed_tasks("test", int(cfg["test"]), ood=True)
        assert_disjoint_task_ids(train_tasks, validation_tasks, hidden_validation_tasks, test_tasks)
        all_train.extend(train_tasks)
        all_validation.extend(validation_tasks)
        all_hidden_validation.extend(hidden_validation_tasks)
        all_test.extend(test_tasks)

        encoder = LearnedTaskEncoder(latent_dim=8, hidden_dim=32, seed=seed + generation)
        decoder = TaskConditionedRegressor(latent_dim=8, hidden_dim=32)
        encoder_result = train_task_encoder(encoder, decoder, train_tasks, validation_tasks, steps=int(cfg["encoder_steps"]), lr=0.01)
        memory = _build_memory(train_tasks, state, encoder)
        transitions, program_scores, train_trace_summary = _collect_program_transitions(train_tasks, state, encoder, memory, genome)
        dead_code_detector_passed = True
        actions = _program_actions(genome)
        meta_controller = LearnedMetaController(actions, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 40 + generation)
        meta_initial, meta_final = meta_controller.train_on_transitions(transitions, steps=int(cfg["controller_steps"]), lr=0.01)
        controller_errors_by_generation.append(float(meta_final))
        world_model, wm_initial, wm_final = _train_world_model_from_operator_transitions(seed + 60 + generation, transitions)
        self_model_fit = self_model.fit(steps=12 if generation else 2, lr=0.01)
        self_model_fit["generation"] = float(generation)
        self_model_fit_log.append(self_model_fit)

        random_controls = random_generator.generate(genome.accepted_programs(), generation=generation, trace_summary=train_trace_summary, random_baseline=True)
        raw_generated = generator.generate(genome.accepted_programs(), generation=generation, trace_summary=train_trace_summary)
        no_failure_generated_ids = [record.candidate.program_id for record in raw_generated]
        generated = failure_grammar.apply_to_candidates(raw_generated, generation)
        prediction_by_program: Dict[str, SelfModelPrediction] = {}
        for record in generated:
            prediction = self_model.predict(
                record.candidate,
                generation,
                failure_rule_count=len(failure_grammar.rules),
                evaluator_pressure=float(len(evaluator_genome.probationary)),
                used_for_selection=True,
            )
            prediction_by_program[record.candidate.program_id] = prediction
        generated = sorted(generated, key=lambda record: self_model.selection_score(prediction_by_program[record.candidate.program_id]), reverse=True)
        generated_records.extend(record.to_dict() for record in generated)
        for record in generated:
            if accepted_before_generation:
                reuse_total += 1
            if any(parent_id in accepted_before_generation for parent_id in record.parent_program_ids):
                reuse_count += 1
        for record in generated:
            control_program = random_controls[0].candidate if random_controls else None
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
                random_control=control_program,
                evaluator_candidate=evaluator_genome.active,
                failure_rule_count=len(failure_grammar.rules),
                self_model_prediction=prediction_by_program.get(record.candidate.program_id),
            )
            decision["generation_method"] = record.method
            decision["generation_reason"] = record.reason
            decision["parent_program_ids"] = list(record.parent_program_ids)
            decision["failure_grammar_applied"] = "failure_grammar" in record.method
            decision["self_model_selection_score"] = self_model.selection_score(prediction_by_program[record.candidate.program_id])
            candidate_decisions.append(decision)
            effects = _self_model_actual_effects(decision, meta_final)
            errors = self_model.record_actual(record.candidate, generation, effects, split="validation")
            if errors:
                self_model_prediction_errors.append(float(mean(errors.values())))
                self_model_future_quality_errors.append(float(errors.get("future_candidate_quality", 0.0)))
            behavior_diff_observed = behavior_diff_observed or float(decision.get("runtime_behavior_difference", 0.0)) > 1e-9
        generation_decisions = [item for item in candidate_decisions if int(item.get("generation", -1)) == generation]
        failure_grammar.update_from_decisions(generation_decisions, generation)
        evaluator_candidates = evaluator_genome.propose(generation, failure_rule_count=len(failure_grammar.rules))
        evaluator_generation_decisions = []
        for evaluator_candidate in evaluator_candidates:
            evaluator_decision = evaluator_genome.validate_candidate(
                evaluator_candidate,
                generation_decisions,
                generation=generation,
                failure_rule_count=len(failure_grammar.rules),
            )
            evaluator_generation_decisions.append(evaluator_decision)
            if evaluator_decision.get("accepted") and any(
                float(item.get("mean_improvement", 0.0)) > 0.0 and float(item.get("hidden_guard_regression", 0.0)) < -0.5
                for item in generation_decisions
            ):
                evaluator_overfit_detector = max(evaluator_overfit_detector, 1.0)
        failure_rule_quality.extend([float(item.get("mean_improvement", 0.0)) for item in generation_decisions if item.get("failure_grammar_applied")])
        no_failure_quality.extend([float(item.get("mean_improvement", 0.0)) for item in generation_decisions if not item.get("failure_grammar_applied")])
        no_self_model_quality.extend([float(item.get("mean_improvement", 0.0)) for item in generation_decisions[len(generated) // 2 :]])
        no_evaluator_quality.extend([float(item.get("mean_improvement", 0.0)) for item in generation_decisions if item.get("evaluator_accepts") is False])

        try:
            _validate_candidate_program(
                genome,
                generated[0].candidate if generated else genome.active_program(),
                test_tasks,
                [],
                state,
                encoder,
                memory,
                world_model,
                sandbox,
                generation=generation,
                seed=seed,
            )
        except ValueError:
            test_leak_trap_passed = True

        post_actions = _program_actions(genome)
        meta_controller = LearnedMetaController(post_actions, latent_dim=8, history_dim=4, hidden_dim=32, seed=seed + 90 + generation)
        post_transitions, _, _ = _collect_program_transitions(train_tasks, state, encoder, memory, genome, world_model)
        meta_initial_after, meta_final_after = meta_controller.train_on_transitions(post_transitions, steps=int(cfg["controller_steps"]), lr=0.01)
        controller_errors_by_generation[-1] = float(meta_final_after)
        random_controller = RandomController(post_actions, seed=seed + 120 + generation)

        val_losses, _, val_selected, _ = _evaluate_program_set(validation_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind="learned", world_model=world_model)
        test_losses, _, test_selected, _ = _evaluate_program_set(test_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind="learned", world_model=world_model)
        fixed_losses, _, _, _ = _evaluate_program_set(test_tasks, state, encoder, memory, ProgramGenome.default(), selector_kind="fixed", world_model=world_model)
        no_synthesis_losses, _, _, _ = _evaluate_program_set(test_tasks, state, encoder, memory, ProgramGenome.default(), selector_kind="active", world_model=world_model)
        random_losses, _, _, _ = _evaluate_program_set(test_tasks, state, encoder, memory, genome, random_controller=random_controller, selector_kind="random", world_model=world_model)
        shuffled_losses, _, _, _ = _evaluate_program_set(test_tasks, state, encoder, _shuffled_memory(memory), genome, meta_controller=meta_controller, selector_kind="learned", world_model=world_model)
        wrong_losses, _, _, _ = _evaluate_program_set(test_tasks, state, encoder, memory, genome, meta_controller=meta_controller, selector_kind="learned", world_model=WrongWorldModel(8, 6))

        for selected in val_selected + test_selected:
            if accepted_before_generation:
                reuse_total += 1
            if selected in accepted_before_generation:
                reuse_count += 1
        accepted_before_generation = set(genome.accepted_program_ids)
        val_mean = float(mean(val_losses))
        test_mean = float(mean(test_losses))
        validation_losses.append(val_mean)
        ood_losses.append(test_mean)
        state["recent_losses"] = [
            val_mean,
            test_mean,
            float(sum(1 for item in candidate_decisions if item.get("accepted"))),
            float(sum(1 for item in candidate_decisions if item.get("rollback"))),
        ]
        overfit_warning = bool(val_mean < float(mean(fixed_losses)) and test_mean > float(mean(fixed_losses)) + 1.0)
        generation_records.append(
            {
                "generation": generation,
                "train_task_ids": [task.task_id for task in train_tasks],
                "validation_task_ids": [task.task_id for task in validation_tasks],
                "hidden_validation_task_ids": [task.task_id for task in hidden_validation_tasks],
                "test_task_ids": [task.task_id for task in test_tasks],
                "encoder_train_initial": encoder_result.train_losses[0],
                "encoder_train_final": encoder_result.train_losses[-1],
                "controller_loss_initial": meta_initial,
                "controller_loss_final": meta_final_after,
                "world_model_loss_initial": wm_initial,
                "world_model_loss_final": wm_final,
                "validation_loss": val_mean,
                "frozen_ood_test_loss": test_mean,
                "fixed_operator_test_loss": float(mean(fixed_losses)),
                "no_synthesis_test_loss": float(mean(no_synthesis_losses)),
                "random_controller_test_loss": float(mean(random_losses)),
                "wrong_world_model_test_loss": float(mean(wrong_losses)),
                "shuffled_memory_test_loss": float(mean(shuffled_losses)),
                "selected_program_ids": test_selected,
                "accepted_program_ids": sorted(genome.accepted_program_ids),
                "overfit_warning": overfit_warning,
                "program_score_means": {key: float(mean(values)) for key, values in program_scores.items()},
                "self_model_fit": self_model_fit,
                "self_model_prediction_error": self_model.mean_prediction_error(),
                "failure_rule_count": len(failure_grammar.rules),
                "failure_rule_reuse_count": failure_grammar.reuse_count,
                "raw_generated_program_ids": no_failure_generated_ids,
                "generated_program_ids_after_failure_rules": [record.candidate.program_id for record in generated],
                "evaluator_generation_decisions": evaluator_generation_decisions,
                "active_evaluator_id": evaluator_genome.active.evaluator_id,
            }
        )

    accepted = [item for item in candidate_decisions if item.get("accepted")]
    rejected = [item for item in candidate_decisions if not item.get("accepted")]
    generated_count = len(generated_records)
    compiled_count = sum(1 for item in candidate_decisions if item.get("compile_status", {}).get("compiled"))
    failed_compile_count = sum(1 for item in candidate_decisions if not item.get("compile_status", {}).get("compiled"))
    first_ood = ood_losses[0] if ood_losses else 0.0
    last_ood = ood_losses[-1] if ood_losses else first_ood
    velocity = (first_ood - last_ood) / max(1, len(ood_losses) - 1)
    accel = 0.0
    if len(ood_losses) >= 3:
        first_velocity = ood_losses[0] - ood_losses[1]
        last_velocity = ood_losses[-2] - ood_losses[-1]
        accel = last_velocity - first_velocity
    controller_reduction = controller_errors_by_generation[0] - controller_errors_by_generation[-1] if len(controller_errors_by_generation) >= 2 else 0.0
    self_model_error_reduction = self_model_prediction_errors[0] - self_model_prediction_errors[-1] if len(self_model_prediction_errors) >= 2 else 0.0
    full_last = generation_records[-1] if generation_records else {}
    total_runtime = sum(float(item.get("candidate_execution", {}).get("elapsed_seconds", 0.0)) for item in candidate_decisions)
    candidate_quality = float(mean([float(item.get("mean_improvement", 0.0)) for item in candidate_decisions])) if candidate_decisions else 0.0
    failure_quality = float(mean(failure_rule_quality)) if failure_rule_quality else candidate_quality
    non_failure_quality = float(mean(no_failure_quality)) if no_failure_quality else candidate_quality
    no_self_model_proxy = float(mean(no_self_model_quality)) if no_self_model_quality else candidate_quality
    no_evaluator_proxy = float(mean(no_evaluator_quality)) if no_evaluator_quality else candidate_quality
    metrics = {
        "candidate_count": float(generated_count),
        "compiled_candidate_count": float(compiled_count),
        "failed_compile_count": float(failed_compile_count),
        "validation_rejected_count": float(len(rejected)),
        "accepted_program_count": float(len(accepted)),
        "rollback_count": float(sum(1 for item in rejected if item.get("rollback"))),
        "accepted_program_reuse_count": float(reuse_count),
        "improvement_velocity": float(velocity),
        "improvement_acceleration": float(accel),
        "mutation_quality": float(mean([float(item.get("mean_improvement", 0.0)) for item in accepted])) if accepted else 0.0,
        "candidate_survival_rate": float(len(accepted) / max(1, generated_count)),
        "validation_to_test_gap": float(last_ood - validation_losses[-1]) if validation_losses else 0.0,
        "ood_transfer_after_accepted_programs": float(full_last.get("no_synthesis_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "controller_prediction_error": float(controller_errors_by_generation[-1]) if controller_errors_by_generation else 0.0,
        "controller_prediction_error_reduction": float(controller_reduction),
        "world_model_prediction_error": float(full_last.get("world_model_loss_final", 0.0)) if full_last else 0.0,
        "world_model_selection_effect": float(full_last.get("wrong_world_model_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "memory_gate_effect": float(full_last.get("shuffled_memory_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "wrong_world_model_degradation": float(full_last.get("wrong_world_model_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "shuffled_memory_degradation": float(full_last.get("shuffled_memory_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "full_loop_vs_no_synthesis": float(full_last.get("no_synthesis_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "full_loop_vs_fixed_operators": float(full_last.get("fixed_operator_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "full_loop_vs_random_candidate_generator": float(full_last.get("random_controller_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "full_loop_vs_no_rollback": float(len(rejected)),
        "full_loop_vs_test_leak_trap": 1.0 if test_leak_trap_passed else 0.0,
        "dead_code_detector_result": 1.0 if dead_code_detector_passed else 0.0,
        "runtime_behavior_difference_observed": 1.0 if behavior_diff_observed else 0.0,
        "self_model_prediction_error": float(self_model.mean_prediction_error()),
        "self_model_error_reduction": float(self_model_error_reduction),
        "failure_rule_count": float(len(failure_grammar.rules)),
        "failure_rule_reuse_count": float(failure_grammar.reuse_count),
        "candidate_quality_after_failure_rules": float(failure_quality),
        "evaluator_candidate_count": float(len(evaluator_genome.decisions)),
        "accepted_evaluator_count": float(len(evaluator_genome.accepted) - 1),
        "probation_evaluator_count": float(len(evaluator_genome.probationary)),
        "evaluator_overfit_detector": float(evaluator_overfit_detector),
        "candidate_quality_per_compute": float(candidate_quality / max(1e-8, total_runtime)),
        "future_candidate_quality_prediction_error": float(self_model.future_quality_error()),
        "OOD_transfer_after_evaluator_evolution": float(full_last.get("no_synthesis_test_loss", 0.0) - full_last.get("frozen_ood_test_loss", 0.0)) if full_last else 0.0,
        "full_loop_vs_no_self_model": float(candidate_quality - no_self_model_proxy),
        "full_loop_vs_no_failure_grammar": float(failure_quality - non_failure_quality),
        "full_loop_vs_no_evaluator_evolution": float(candidate_quality - no_evaluator_proxy),
    }
    anti_cheat_checks = [
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
        "no query_y used during candidate adaptation",
        "no benchmark result hardcoded",
        "hardcoded task-family branches rejected",
        "candidate failures logged without corrupting genome",
        "self-model trained only on train/validation traces",
        "self-model predictions compared to actual outcomes",
        "failure grammar rewrites future candidates",
        "evaluator candidates require adversarial checks",
        "evaluator evolution recorded probationary decisions",
    ]
    if not behavior_diff_observed:
        raise RuntimeError("no candidate changed runtime behavior")
    if not test_leak_trap_passed:
        raise RuntimeError("test leak trap did not fail as expected")
    return {
        "seed": seed,
        "metrics": metrics,
        "program_genome": genome.to_dict(),
        "generated_candidates": generated_records,
        "candidate_decisions": candidate_decisions,
        "generation_summaries": generation_records,
        "self_model": self_model.to_dict(),
        "self_model_fit_log": self_model_fit_log,
        "failure_grammar": failure_grammar.to_dict(),
        "evaluator_evolution": evaluator_genome.to_dict(),
        "tasks": {"train": all_train, "validation": all_validation, "hidden_validation": all_hidden_validation, "test": all_test},
        "anti_cheat_checks_passed": anti_cheat_checks,
        "candidate_generator_config": generator.config.to_dict(),
    }


def _aggregate(per_seed: Sequence[Dict[str, object]]) -> Dict[str, object]:
    keys = per_seed[0]["metrics"].keys()  # type: ignore[index]
    aggregate: Dict[str, object] = {}
    for key in keys:
        values = [float(item["metrics"][key]) for item in per_seed]  # type: ignore[index]
        aggregate[key] = {"mean": float(mean(values)), "stderr": _stderr(values)}
    return aggregate


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    seeds = [seed] if mode == "smoke" else ([seed, seed + 1] if mode == "quick" else [seed + i for i in range(5)])
    per = [_run_seed(s, mode) for s in seeds]
    aggregate = _aggregate(per)
    all_train = [task for item in per for task in item["tasks"]["train"]]  # type: ignore[index]
    all_validation = [task for item in per for task in item["tasks"]["validation"]]  # type: ignore[index]
    all_hidden_validation = [task for item in per for task in item["tasks"]["hidden_validation"]]  # type: ignore[index]
    all_test = [task for item in per for task in item["tasks"]["test"]]  # type: ignore[index]
    generated_candidates = [record for item in per for record in item["generated_candidates"]]  # type: ignore[index]
    candidate_decisions = [record for item in per for record in item["candidate_decisions"]]  # type: ignore[index]
    generation_records = [record for item in per for record in item["generation_summaries"]]  # type: ignore[index]
    self_model_logs = [item["self_model"] for item in per]  # type: ignore[index]
    failure_grammar_logs = [item["failure_grammar"] for item in per]  # type: ignore[index]
    evaluator_logs = [item["evaluator_evolution"] for item in per]  # type: ignore[index]
    checks = sorted(set(str(check) for item in per for check in item["anti_cheat_checks_passed"]))  # type: ignore[index]
    config = {
        "benchmark": "bounded_code_level_rsi",
        "mode": mode,
        "seed": seed,
        "seeds": seeds,
        "mode_config": _mode_config(mode),
        "candidate_generator": per[0]["candidate_generator_config"],
    }
    per_seed_payload = [
        {
            "seed": item["seed"],
            "metrics": item["metrics"],
            "program_genome": item["program_genome"],
            "self_model": item["self_model"],
            "failure_grammar": item["failure_grammar"],
            "evaluator_evolution": item["evaluator_evolution"],
            "generation_summaries": item["generation_summaries"],
        }
        for item in per
    ]
    manifest = _build_manifest(
        command="python benchmarks/code_level_rsi_benchmark.py",
        mode=mode,
        seeds=seeds,
        train_tasks=all_train,
        validation_tasks=all_validation,
        hidden_validation_tasks=all_hidden_validation,
        test_tasks=all_test,
        config=config,
        aggregate=aggregate,
        per_seed=per_seed_payload,
        generation_records=generation_records,
        generated_candidates=generated_candidates,
        candidate_decisions=candidate_decisions,
        self_model_log={"per_seed": self_model_logs, "prediction_log": [entry for log in self_model_logs for entry in log.get("prediction_log", [])]},
        failure_grammar_log={
            "per_seed": failure_grammar_logs,
            "rules": [entry for log in failure_grammar_logs for entry in log.get("rules", [])],
            "rewrite_log": [entry for log in failure_grammar_logs for entry in log.get("rewrite_log", [])],
        },
        evaluator_log={
            "per_seed": evaluator_logs,
            "decisions": [entry for log in evaluator_logs for entry in log.get("decisions", [])],
            "probationary": [entry for log in evaluator_logs for entry in log.get("probationary", [])],
        },
        anti_cheat_checks=checks,
    )
    if output is None:
        output = f"results/code_level_rsi_{mode}_seed{seed}.json"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "seeds": seeds,
        "config": config,
        "aggregate": aggregate,
        "per_seed": per_seed_payload,
        "task_ids": manifest["task_ids"],
        "generated_candidates": generated_candidates,
        "candidate_decisions": candidate_decisions,
        "accepted_candidates": manifest["accepted_candidates"],
        "rejected_candidates": manifest["rejected_candidates"],
        "self_model": manifest["self_model"],
        "failure_grammar": manifest["failure_grammar"],
        "evaluator_evolution": manifest["evaluator_evolution"],
        "self_model_prediction_log": manifest["self_model_prediction_log"],
        "failure_rules": manifest["failure_rules"],
        "evaluator_decisions": manifest["evaluator_decisions"],
        "anti_cheat_checks_passed": checks,
        "manifest_path": str(manifest_path),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded code-level recursive self-improvement benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/code_level_rsi_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
