"""CPU-runnable benchmark for the bounded interaction operating scaffold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background_reasoning_simulator import BackgroundReasoningSimulator
from interaction_manifest import build_interaction_manifest, write_interaction_manifest
from interaction_policies import (
    AblatedNoBackgroundPolicy,
    AblatedNoVisualPolicy,
    AlwaysInterruptPolicy,
    AlwaysListenPolicy,
    AlwaysSilencePolicy,
    BaselineInteractionPolicy,
    NoOpInteractionPolicy,
    RandomInteractionPolicy,
)
from interaction_residue_layer import (
    ACTION_DELIVER_BACKGROUND_RESULT,
    ACTION_DELEGATE_BACKGROUND,
    ACTION_SILENCE,
    InteractionAction,
    InteractionPolicy,
    InteractionResidue,
    InteractionResidueLayer,
    MICRO_TURN_MS,
    MicroTurnEvent,
)
from interaction_stream_runtime import InteractionRecorder, InteractionReplay, InteractionStream


TASK_TURN_TAKING = "turn_taking"
TASK_PROACTIVE_VISUAL = "proactive_visual_event"
TASK_BACKGROUND = "background_delegation"
TASK_MIXED = "mixed_stream"
TASK_ADVERSARIAL = "adversarial_controls"

REQUIRED_METRICS = [
    "interaction_quality",
    "bad_interrupt_count",
    "missed_visual_change_count",
    "late_background_result_count",
    "premature_answer_count",
    "idle_when_questioned_count",
    "backchannel_overuse_count",
    "residue_count",
    "total_residue_severity",
    "background_delivery_rate",
    "deterministic_replay_passed",
    "no_op_control_score",
    "random_control_score",
    "always_interrupt_control_score",
    "always_silence_control_score",
]


@dataclass(frozen=True)
class InteractionBenchmarkTask:
    task_id: str
    family: str
    events: tuple[MicroTurnEvent, ...]


def assert_no_expected_actions(events: Iterable[MicroTurnEvent]) -> None:
    leaked = [event.timestamp_ms for event in events if event.expected_action is not None]
    if leaked:
        raise ValueError(f"benchmark events must not carry expected_action labels: {leaked}")


def build_benchmark_tasks(seed: int, *, mode: str = "smoke") -> List[InteractionBenchmarkTask]:
    rng = random.Random(seed)
    repeats = {"smoke": 1, "quick": 2, "full": 3}[mode]
    tasks: List[InteractionBenchmarkTask] = []
    timestamp = 0
    visual_labels = ["bug_detected", "safety_violation", "process_anomaly", "tool_failure"]

    def add_task(family: str, events: Sequence[MicroTurnEvent]) -> None:
        nonlocal timestamp
        task_id = f"{family}_{len(tasks)}_seed{seed}"
        tasks.append(InteractionBenchmarkTask(task_id, family, tuple(events)))
        timestamp = max(event.timestamp_ms for event in events) + 1200

    for index in range(repeats):
        add_task(
            TASK_TURN_TAKING,
            [
                MicroTurnEvent(
                    timestamp,
                    text_delta="I am still explaining the failure",
                    audio_activity=0.86,
                    user_speaking_prob=0.94,
                ),
                MicroTurnEvent(
                    timestamp + MICRO_TURN_MS,
                    audio_activity=0.72,
                    user_speaking_prob=0.78,
                    semantic_uncertainty=0.70,
                ),
            ],
        )

        add_task(
            TASK_PROACTIVE_VISUAL,
            [
                MicroTurnEvent(
                    timestamp,
                    visual_events=(rng.choice(visual_labels),),
                    anomaly_confidence=0.86 + 0.02 * (index % 3),
                    user_speaking_prob=0.08,
                )
            ],
        )

        add_task(
            TASK_BACKGROUND,
            [
                MicroTurnEvent(
                    timestamp,
                    text_delta="Why did the validation run regress?",
                    semantic_uncertainty=0.91,
                    user_speaking_prob=0.0,
                )
            ],
        )

        add_task(
            TASK_MIXED,
            [
                MicroTurnEvent(
                    timestamp,
                    text_delta="I am checking the trace",
                    audio_activity=0.82,
                    user_speaking_prob=0.84,
                    semantic_uncertainty=0.30,
                ),
                MicroTurnEvent(
                    timestamp + MICRO_TURN_MS,
                    visual_events=("tool_failure",),
                    anomaly_confidence=0.90,
                    user_speaking_prob=0.16,
                ),
                MicroTurnEvent(
                    timestamp + 2 * MICRO_TURN_MS,
                    text_delta="What should we inspect next?",
                    semantic_uncertainty=0.88,
                    user_speaking_prob=0.0,
                ),
                MicroTurnEvent(
                    timestamp + 6 * MICRO_TURN_MS,
                    background_result_ready=True,
                    background_result_summary="Previous background analysis found a stale tool result.",
                ),
            ],
        )

        add_task(
            TASK_ADVERSARIAL,
            [
                MicroTurnEvent(
                    timestamp,
                    visual_events=("bug_detected_corrupted",),
                    anomaly_confidence=0.95,
                    user_speaking_prob=0.0,
                )
            ],
        )

    assert_no_expected_actions(event for task in tasks for event in task.events)
    return tasks


def filter_tasks_by_family(tasks: Sequence[InteractionBenchmarkTask], family: str) -> List[InteractionBenchmarkTask]:
    return [task for task in tasks if task.family == family]


def evaluate_policy_on_tasks(
    policy: object,
    tasks: Sequence[InteractionBenchmarkTask],
    *,
    seed: int = 0,
    policy_name: str = "policy",
    record_trace: bool = False,
    simulator_delay_ms: int = 400,
    background_deadline_ms: int = 600,
    force_missing_background: bool = False,
    force_delayed_background: bool = False,
) -> Dict[str, object]:
    all_task_events = [event for task in tasks for event in task.events]
    assert_no_expected_actions(all_task_events)

    layer = InteractionResidueLayer(policy=policy)  # type: ignore[arg-type]
    simulator = BackgroundReasoningSimulator(
        layer.background_bus,
        delay_ms=simulator_delay_ms,
        deterministic=True,
        seed=seed,
    )
    stream = InteractionStream()
    recorder = InteractionRecorder() if record_trace else None
    extra_residues: List[InteractionResidue] = []
    background_requests = 0
    on_time_background_deliveries = 0

    def process_event(
        event: MicroTurnEvent,
        *,
        task_id: str,
        allow_background_poll: bool = True,
    ) -> InteractionAction:
        nonlocal background_requests, on_time_background_deliveries
        stream.append(event)
        if recorder is not None:
            recorder.record_event(event, split="validation")
        action = layer.step(event)
        if recorder is not None:
            recorder.record_action(action, policy_name=policy_name)
            recorder.record_state(layer.state, label=f"{task_id}:after_action")
            recorder.record_background_state(layer.background_bus)

        if event.background_result_ready:
            background_requests += 1
            if action.kind == ACTION_DELIVER_BACKGROUND_RESULT:
                on_time_background_deliveries += 1

        if _requires_background_reasoning(event):
            background_requests += 1
            if action.kind != ACTION_DELEGATE_BACKGROUND:
                extra_residues.append(
                    _late_background_residue(
                        event,
                        actual_action=action.kind,
                        evidence={"reason": "policy_did_not_delegate"},
                    )
                )
                return action

            pending = sorted(layer.background_bus.pending_tasks(), key=lambda task: task.created_timestamp_ms)
            delegated = pending[-1] if pending else None
            if delegated is None:
                extra_residues.append(
                    _late_background_residue(
                        event,
                        actual_action=action.kind,
                        evidence={"reason": "delegated_task_not_recorded"},
                    )
                )
                return action

            if force_missing_background:
                simulator.missing_task_ids.add(delegated.task_id)
            if force_delayed_background:
                simulator.extra_delay_ms_by_task[delegated.task_id] = background_deadline_ms + MICRO_TURN_MS

            poll_time = event.timestamp_ms + simulator_delay_ms + simulator.extra_delay_ms_by_task.get(delegated.task_id, 0)
            result_events = simulator.poll(poll_time) if allow_background_poll else []
            if not result_events:
                extra_residues.append(
                    _late_background_residue(
                        event,
                        actual_action=ACTION_SILENCE,
                        evidence={"reason": "background_result_missing", "task_id": delegated.task_id},
                    )
                )
                return action

            delivered = False
            late = False
            for result_event in result_events:
                result_action = process_event(result_event, task_id=task_id, allow_background_poll=False)
                delivered = delivered or result_action.kind == ACTION_DELIVER_BACKGROUND_RESULT
                late = late or result_event.timestamp_ms - event.timestamp_ms > background_deadline_ms
            if delivered and not late:
                on_time_background_deliveries += 1
            else:
                extra_residues.append(
                    _late_background_residue(
                        event,
                        actual_action=ACTION_DELIVER_BACKGROUND_RESULT if delivered else ACTION_SILENCE,
                        evidence={"reason": "background_result_late" if late else "background_result_not_delivered"},
                    )
                )
        return action

    for task in tasks:
        for event in task.events:
            process_event(event, task_id=task.task_id)

    report = layer.evaluate()
    if recorder is not None:
        recorder.record_evaluator_report(report)

    residue_objects = list(report["residue_objects"]) + extra_residues
    total_severity = sum(float(residue.severity) for residue in residue_objects)
    failure_counts: Dict[str, int] = {}
    for residue in residue_objects:
        failure_counts[residue.failure_type] = failure_counts.get(residue.failure_type, 0) + 1

    action_count = len(layer.actions)
    event_count = len(layer.events)
    interaction_quality = max(0.0, 1.0 - total_severity / max(1.0, float(event_count)))
    metrics = {
        "interaction_quality": float(interaction_quality),
        "bad_interrupt_count": int(failure_counts.get("bad_interrupt", 0)),
        "missed_visual_change_count": int(failure_counts.get("missed_visual_change", 0)),
        "late_background_result_count": int(failure_counts.get("late_background_result", 0)),
        "premature_answer_count": int(failure_counts.get("premature_answer", 0)),
        "idle_when_questioned_count": int(failure_counts.get("idle_when_questioned", 0)),
        "backchannel_overuse_count": int(failure_counts.get("backchannel_overuse", 0)),
        "residue_count": int(len(residue_objects)),
        "total_residue_severity": float(total_severity),
        "background_delivery_rate": float(on_time_background_deliveries / max(1, background_requests)),
        "event_count": event_count,
        "action_count": action_count,
        "failure_counts": failure_counts,
    }

    return {
        "policy_name": policy_name,
        "metrics": metrics,
        "events": [event.to_dict() for event in layer.events],
        "actions": [action.to_dict() for action in layer.actions],
        "event_timestamps": [event.timestamp_ms for event in layer.events],
        "residues": [residue.to_dict() for residue in residue_objects],
        "trace_records": list(recorder.records) if recorder is not None else [],
    }


def run_adversarial_background_control(seed: int) -> Dict[str, object]:
    tasks = filter_tasks_by_family(build_benchmark_tasks(seed, mode="smoke"), TASK_BACKGROUND)
    missing = evaluate_policy_on_tasks(
        BaselineInteractionPolicy(),
        tasks,
        seed=seed,
        policy_name="baseline_missing_background",
        force_missing_background=True,
    )
    delayed = evaluate_policy_on_tasks(
        BaselineInteractionPolicy(),
        tasks,
        seed=seed,
        policy_name="baseline_delayed_background",
        force_delayed_background=True,
    )
    return {
        "missing_background_result_count": missing["metrics"]["late_background_result_count"],
        "delayed_background_result_count": delayed["metrics"]["late_background_result_count"],
    }


def _requires_background_reasoning(event: MicroTurnEvent) -> bool:
    return (
        not event.background_result_ready
        and event.semantic_uncertainty >= 0.82
        and InteractionPolicy._looks_like_question(event.text_delta)
    )


def _late_background_residue(
    event: MicroTurnEvent,
    *,
    actual_action: str,
    evidence: Dict[str, object],
) -> InteractionResidue:
    return InteractionResidue(
        failure_type="late_background_result",
        timestamp_ms=event.timestamp_ms,
        expected_action=ACTION_DELIVER_BACKGROUND_RESULT,
        actual_action=actual_action,
        severity=0.80,
        evidence=dict(evidence),
        source_event=event.to_dict(),
    )


def _run_anti_cheat_checks(tasks: Sequence[InteractionBenchmarkTask], *, deterministic_replay_passed: bool) -> Dict[str, bool]:
    try:
        InteractionStream([MicroTurnEvent(200), MicroTurnEvent(0)])
        shuffled_rejected = False
    except ValueError:
        shuffled_rejected = True

    try:
        stream = InteractionStream()
        stream.append(MicroTurnEvent(0))
        stream.append(MicroTurnEvent(0))
        non_monotonic_rejected = False
    except ValueError:
        non_monotonic_rejected = True

    all_events = [event for task in tasks for event in task.events]
    expected_action_labels_absent = all(event.expected_action is None for event in all_events)
    corrupted_label_exercised = any("corrupted" in label for event in all_events for label in event.visual_events)
    return {
        "shuffled_timestamps_rejected": shuffled_rejected,
        "non_monotonic_timestamps_rejected": non_monotonic_rejected,
        "expected_action_labels_absent": expected_action_labels_absent,
        "corrupted_visual_label_control_exercised": corrupted_label_exercised,
        "deterministic_replay_passed": deterministic_replay_passed,
    }


def _run_seed(seed: int, mode: str) -> Dict[str, object]:
    tasks = build_benchmark_tasks(seed, mode=mode)
    baseline = evaluate_policy_on_tasks(
        BaselineInteractionPolicy(),
        tasks,
        seed=seed,
        policy_name="baseline",
        record_trace=True,
    )
    replay_report = InteractionReplay(baseline["trace_records"]).replay(
        InteractionResidueLayer(policy=BaselineInteractionPolicy()),
        deterministic=True,
    )
    deterministic_replay_passed = bool(replay_report["deterministic_replay_passed"])

    no_op = evaluate_policy_on_tasks(NoOpInteractionPolicy(), tasks, seed=seed, policy_name="no_op")
    random_policy = evaluate_policy_on_tasks(RandomInteractionPolicy(seed=seed), tasks, seed=seed, policy_name="random")
    always_interrupt = evaluate_policy_on_tasks(AlwaysInterruptPolicy(), tasks, seed=seed, policy_name="always_interrupt")
    always_silence = evaluate_policy_on_tasks(AlwaysSilencePolicy(), tasks, seed=seed, policy_name="always_silence")
    always_listen = evaluate_policy_on_tasks(AlwaysListenPolicy(), tasks, seed=seed, policy_name="always_listen")
    no_visual = evaluate_policy_on_tasks(AblatedNoVisualPolicy(), tasks, seed=seed, policy_name="ablated_no_visual")
    no_background = evaluate_policy_on_tasks(
        AblatedNoBackgroundPolicy(),
        tasks,
        seed=seed,
        policy_name="ablated_no_background",
    )
    background_adversarial = run_adversarial_background_control(seed)

    control_policy_scores = {
        "baseline": float(baseline["metrics"]["interaction_quality"]),
        "no_op": float(no_op["metrics"]["interaction_quality"]),
        "random": float(random_policy["metrics"]["interaction_quality"]),
        "always_interrupt": float(always_interrupt["metrics"]["interaction_quality"]),
        "always_silence": float(always_silence["metrics"]["interaction_quality"]),
        "always_listen": float(always_listen["metrics"]["interaction_quality"]),
        "ablated_no_visual": float(no_visual["metrics"]["interaction_quality"]),
        "ablated_no_background": float(no_background["metrics"]["interaction_quality"]),
    }

    metrics = dict(baseline["metrics"])
    metrics.update(
        {
            "deterministic_replay_passed": deterministic_replay_passed,
            "no_op_control_score": control_policy_scores["no_op"],
            "random_control_score": control_policy_scores["random"],
            "always_interrupt_control_score": control_policy_scores["always_interrupt"],
            "always_silence_control_score": control_policy_scores["always_silence"],
        }
    )

    bad_control_residue_counts = {
        "no_op": int(no_op["metrics"]["residue_count"]),
        "random": int(random_policy["metrics"]["residue_count"]),
        "always_interrupt": int(always_interrupt["metrics"]["residue_count"]),
        "always_silence": int(always_silence["metrics"]["residue_count"]),
    }

    anti_cheat = _run_anti_cheat_checks(tasks, deterministic_replay_passed=deterministic_replay_passed)
    anti_cheat["missing_background_result_penalized"] = bool(
        background_adversarial["missing_background_result_count"] > 0
    )
    anti_cheat["delayed_background_result_penalized"] = bool(
        background_adversarial["delayed_background_result_count"] > 0
    )
    anti_cheat["controls_executed"] = all(name in control_policy_scores for name in ("no_op", "random", "always_interrupt", "always_silence"))

    return {
        "seed": seed,
        "tasks": [{"task_id": task.task_id, "family": task.family} for task in tasks],
        "task_family_names": sorted({task.family for task in tasks}),
        "metrics": metrics,
        "control_policy_scores": control_policy_scores,
        "bad_control_residue_counts": bad_control_residue_counts,
        "anti_cheat_check_results": anti_cheat,
        "policy_results": {
            "baseline": _compact_policy_result(baseline),
            "no_op": _compact_policy_result(no_op),
            "random": _compact_policy_result(random_policy),
            "always_interrupt": _compact_policy_result(always_interrupt),
            "always_silence": _compact_policy_result(always_silence),
            "always_listen": _compact_policy_result(always_listen),
            "ablated_no_visual": _compact_policy_result(no_visual),
            "ablated_no_background": _compact_policy_result(no_background),
        },
        "event_timestamps": list(baseline["event_timestamps"]),
        "number_of_events": int(baseline["metrics"]["event_count"]),
        "number_of_actions": int(baseline["metrics"]["action_count"]),
        "number_of_residues": int(baseline["metrics"]["residue_count"]),
    }


def _compact_policy_result(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "policy_name": result["policy_name"],
        "metrics": result["metrics"],
        "residue_count": result["metrics"]["residue_count"],  # type: ignore[index]
    }


def _aggregate(per_seed: Sequence[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {}
    count_metrics = {
        "bad_interrupt_count",
        "missed_visual_change_count",
        "late_background_result_count",
        "premature_answer_count",
        "idle_when_questioned_count",
        "backchannel_overuse_count",
        "residue_count",
    }
    sum_metrics = {"total_residue_severity"}
    mean_metrics = {
        "interaction_quality",
        "background_delivery_rate",
        "no_op_control_score",
        "random_control_score",
        "always_interrupt_control_score",
        "always_silence_control_score",
    }
    for key in count_metrics:
        aggregate[key] = int(sum(int(item["metrics"][key]) for item in per_seed))  # type: ignore[index]
    for key in sum_metrics:
        aggregate[key] = float(sum(float(item["metrics"][key]) for item in per_seed))  # type: ignore[index]
    for key in mean_metrics:
        aggregate[key] = float(mean(float(item["metrics"][key]) for item in per_seed))  # type: ignore[index]
    aggregate["deterministic_replay_passed"] = all(bool(item["metrics"]["deterministic_replay_passed"]) for item in per_seed)  # type: ignore[index]
    return aggregate


def _aggregate_scores(per_seed: Sequence[Dict[str, object]]) -> Dict[str, float]:
    names = per_seed[0]["control_policy_scores"].keys()  # type: ignore[index]
    return {
        name: float(mean(float(item["control_policy_scores"][name]) for item in per_seed))  # type: ignore[index]
        for name in names
    }


def _aggregate_bad_control_counts(per_seed: Sequence[Dict[str, object]]) -> Dict[str, int]:
    names = per_seed[0]["bad_control_residue_counts"].keys()  # type: ignore[index]
    return {
        name: int(sum(int(item["bad_control_residue_counts"][name]) for item in per_seed))  # type: ignore[index]
        for name in names
    }


def _aggregate_anti_cheat(per_seed: Sequence[Dict[str, object]]) -> Dict[str, bool]:
    names = per_seed[0]["anti_cheat_check_results"].keys()  # type: ignore[index]
    return {
        name: all(bool(item["anti_cheat_check_results"][name]) for item in per_seed)  # type: ignore[index]
        for name in names
    }


def _offset_timestamps(per_seed: Sequence[Dict[str, object]]) -> List[int]:
    timestamps: List[int] = []
    offset = 0
    for item in per_seed:
        seed_timestamps = [int(value) for value in item["event_timestamps"]]  # type: ignore[index]
        timestamps.extend(offset + value for value in seed_timestamps)
        offset += (max(seed_timestamps) + 100000) if seed_timestamps else 100000
    return timestamps


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    if mode not in {"smoke", "quick", "full"}:
        raise ValueError("mode must be smoke, quick, or full")
    seeds = [seed] if mode == "smoke" else ([seed, seed + 1] if mode == "quick" else [seed + i for i in range(5)])
    per_seed = [_run_seed(item_seed, mode) for item_seed in seeds]
    metrics = _aggregate(per_seed)
    control_scores = _aggregate_scores(per_seed)
    bad_control_counts = _aggregate_bad_control_counts(per_seed)
    anti_cheat = _aggregate_anti_cheat(per_seed)
    task_family_names = sorted({family for item in per_seed for family in item["task_family_names"]})  # type: ignore[index]

    config = {
        "mode": mode,
        "seed": seed,
        "seeds": seeds,
        "micro_turn_ms": MICRO_TURN_MS,
        "task_family_names": task_family_names,
        "policy_count": len(control_scores),
    }
    result = {
        "mode": mode,
        "seeds": seeds,
        "config": config,
        "metrics": metrics,
        "control_policy_scores": control_scores,
        "bad_control_residue_counts": bad_control_counts,
        "anti_cheat_check_results": anti_cheat,
        "per_seed": per_seed,
    }

    output_path = Path(output or f"results/interaction_residue_{mode}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    manifest = build_interaction_manifest(
        command=f"python benchmarks/interaction_residue_benchmark.py --mode {mode} --seed {seed}",
        seed=seed,
        mode=mode,
        config=config,
        benchmark_metrics=metrics,
        number_of_events=sum(int(item["number_of_events"]) for item in per_seed),
        number_of_actions=sum(int(item["number_of_actions"]) for item in per_seed),
        number_of_residues=sum(int(item["number_of_residues"]) for item in per_seed),
        task_family_names=task_family_names,
        used_test_labels=False,
        deterministic_replay_passed=bool(metrics["deterministic_replay_passed"]),
        control_policy_scores=control_scores,
        anti_cheat_check_results=anti_cheat,
        event_timestamps=_offset_timestamps(per_seed),
        bad_control_residue_counts=bad_control_counts,
    )
    manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
    write_interaction_manifest(manifest, manifest_path)
    result["manifest_path"] = str(manifest_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/interaction_residue_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
