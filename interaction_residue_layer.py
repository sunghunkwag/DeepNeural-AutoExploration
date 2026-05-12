"""Micro-turn interaction residue layer for bounded RSI experiments.

This module adds a small, CPU-runnable interaction shell for micro-turn
simulation without copying any private model weights, private code, or closed
architecture.  The goal is to make interaction failures observable, scoreable,
and exportable into the repository's bounded RSI mechanics.

Core loop:

    MicroTurnEvent -> InteractionPolicy -> InteractionAction
        -> InteractionEvaluator -> InteractionResidue -> RSI decision records

The exported decision records are shaped so they can be consumed by existing
bounded evaluator-evolution code.  They deliberately do not use held-out test
signals and they include simple no-op/random controls for anti-shortcut checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


MICRO_TURN_MS = 200

ACTION_SILENCE = "silence"
ACTION_LISTEN = "listen"
ACTION_BACKCHANNEL = "backchannel"
ACTION_INTERRUPT_WARNING = "interrupt_warning"
ACTION_ANSWER = "answer"
ACTION_DELEGATE_BACKGROUND = "delegate_background"
ACTION_DELIVER_BACKGROUND_RESULT = "deliver_background_result"

CRITICAL_VISUAL_EVENTS = frozenset(
    {
        "bug_detected",
        "safety_violation",
        "process_anomaly",
        "new_participant",
        "tool_failure",
    }
)

INTERACTION_FAILURE_TYPES = frozenset(
    {
        "bad_interrupt",
        "missed_visual_change",
        "late_background_result",
        "premature_answer",
        "idle_when_questioned",
        "backchannel_overuse",
        "expected_action_mismatch",
    }
)


@dataclass(frozen=True)
class MicroTurnEvent:
    """A 200 ms-aligned observation packet.

    The fields are intentionally scalar and inspectable.  This keeps the first
    integration runnable on ordinary CPUs while preserving the architectural
    shape of a streaming interaction system.
    """

    timestamp_ms: int
    text_delta: str = ""
    audio_activity: float = 0.0
    visual_events: Tuple[str, ...] = field(default_factory=tuple)
    user_speaking_prob: float = 0.0
    model_speaking: bool = False
    background_result_ready: bool = False
    background_result_summary: str = ""
    anomaly_confidence: float = 0.0
    semantic_uncertainty: float = 0.0
    expected_action: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        for name, value in (
            ("audio_activity", self.audio_activity),
            ("user_speaking_prob", self.user_speaking_prob),
            ("anomaly_confidence", self.anomaly_confidence),
            ("semantic_uncertainty", self.semantic_uncertainty),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        object.__setattr__(self, "visual_events", tuple(str(item) for item in self.visual_events))

    @property
    def micro_turn_index(self) -> int:
        return self.timestamp_ms // MICRO_TURN_MS

    def has_critical_visual_event(self) -> bool:
        return any(event in CRITICAL_VISUAL_EVENTS for event in self.visual_events)

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "micro_turn_index": self.micro_turn_index,
            "text_delta": self.text_delta,
            "audio_activity": float(self.audio_activity),
            "visual_events": list(self.visual_events),
            "user_speaking_prob": float(self.user_speaking_prob),
            "model_speaking": bool(self.model_speaking),
            "background_result_ready": bool(self.background_result_ready),
            "background_result_summary": self.background_result_summary,
            "anomaly_confidence": float(self.anomaly_confidence),
            "semantic_uncertainty": float(self.semantic_uncertainty),
            "expected_action": self.expected_action,
        }


@dataclass(frozen=True)
class InteractionAction:
    """A policy decision emitted for one micro-turn."""

    kind: str
    message: str = ""
    confidence: float = 1.0
    delegated_task: str | None = None
    latency_budget_ms: int = MICRO_TURN_MS

    def __post_init__(self) -> None:
        allowed = {
            ACTION_SILENCE,
            ACTION_LISTEN,
            ACTION_BACKCHANNEL,
            ACTION_INTERRUPT_WARNING,
            ACTION_ANSWER,
            ACTION_DELEGATE_BACKGROUND,
            ACTION_DELIVER_BACKGROUND_RESULT,
        }
        if self.kind not in allowed:
            raise ValueError(f"unknown interaction action: {self.kind}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "message": self.message,
            "confidence": float(self.confidence),
            "delegated_task": self.delegated_task,
            "latency_budget_ms": int(self.latency_budget_ms),
        }


@dataclass
class InteractionState:
    """Small mutable state kept across micro-turns."""

    turn_count: int = 0
    last_action_kind: str = ACTION_SILENCE
    last_user_activity_ms: int | None = None
    background_pending: bool = False
    consecutive_backchannels: int = 0
    interrupt_count: int = 0
    delivered_background_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "turn_count": self.turn_count,
            "last_action_kind": self.last_action_kind,
            "last_user_activity_ms": self.last_user_activity_ms,
            "background_pending": self.background_pending,
            "consecutive_backchannels": self.consecutive_backchannels,
            "interrupt_count": self.interrupt_count,
            "delivered_background_count": self.delivered_background_count,
        }


@dataclass(frozen=True)
class BackgroundTask:
    task_id: str
    created_timestamp_ms: int
    summary: str
    completed: bool = False
    result_summary: str = ""

    def complete(self, result_summary: str) -> "BackgroundTask":
        return BackgroundTask(self.task_id, self.created_timestamp_ms, self.summary, True, result_summary)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "created_timestamp_ms": self.created_timestamp_ms,
            "summary": self.summary,
            "completed": self.completed,
            "result_summary": self.result_summary,
        }


class BackgroundAgentBus:
    """Minimal asynchronous-task bus for interaction/background separation."""

    def __init__(self) -> None:
        self._tasks: Dict[str, BackgroundTask] = {}
        self._counter = 0

    def submit(self, timestamp_ms: int, summary: str) -> BackgroundTask:
        self._counter += 1
        task = BackgroundTask(f"bg_{self._counter}", timestamp_ms, summary)
        self._tasks[task.task_id] = task
        return task

    def complete(self, task_id: str, result_summary: str) -> BackgroundTask:
        if task_id not in self._tasks:
            raise KeyError(f"unknown background task: {task_id}")
        task = self._tasks[task_id].complete(result_summary)
        self._tasks[task_id] = task
        return task

    def ready_tasks(self) -> List[BackgroundTask]:
        return [task for task in self._tasks.values() if task.completed]

    def pending_tasks(self) -> List[BackgroundTask]:
        return [task for task in self._tasks.values() if not task.completed]

    def to_dict(self) -> Dict[str, object]:
        return {"tasks": [task.to_dict() for task in self._tasks.values()]}


class InteractionPolicy:
    """Deterministic baseline policy over micro-turn observations.

    This is not a trained neural policy.  It is the control policy needed for
    the first integration benchmark: simple enough to audit, but expressive
    enough to produce measurable timing, interruption, and delegation failures.
    """

    def __init__(
        self,
        *,
        speech_continuation_threshold: float = 0.72,
        hard_interrupt_speech_threshold: float = 0.88,
        proactive_visual_threshold: float = 0.70,
        delegation_uncertainty_threshold: float = 0.82,
        backchannel_uncertainty_threshold: float = 0.65,
        max_consecutive_backchannels: int = 2,
    ) -> None:
        self.speech_continuation_threshold = speech_continuation_threshold
        self.hard_interrupt_speech_threshold = hard_interrupt_speech_threshold
        self.proactive_visual_threshold = proactive_visual_threshold
        self.delegation_uncertainty_threshold = delegation_uncertainty_threshold
        self.backchannel_uncertainty_threshold = backchannel_uncertainty_threshold
        self.max_consecutive_backchannels = max_consecutive_backchannels

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        if event.background_result_ready:
            return InteractionAction(
                ACTION_DELIVER_BACKGROUND_RESULT,
                message=event.background_result_summary or "Background result is ready.",
                confidence=0.95,
            )

        if self._should_interrupt_for_visual_event(event):
            return InteractionAction(
                ACTION_INTERRUPT_WARNING,
                message="Immediate visual change requires attention.",
                confidence=max(0.75, event.anomaly_confidence),
            )

        if event.user_speaking_prob >= self.speech_continuation_threshold:
            if (
                event.semantic_uncertainty >= self.backchannel_uncertainty_threshold
                and state.consecutive_backchannels < self.max_consecutive_backchannels
            ):
                return InteractionAction(ACTION_BACKCHANNEL, message="I am following.", confidence=0.80)
            return InteractionAction(ACTION_LISTEN, confidence=0.90)

        if self._looks_like_question(event.text_delta):
            if event.semantic_uncertainty >= self.delegation_uncertainty_threshold:
                return InteractionAction(
                    ACTION_DELEGATE_BACKGROUND,
                    message="I will compute this in the background.",
                    delegated_task=event.text_delta.strip(),
                    confidence=0.75,
                )
            return InteractionAction(ACTION_ANSWER, message="Answer now.", confidence=0.80)

        if event.semantic_uncertainty >= self.delegation_uncertainty_threshold and event.text_delta.strip():
            return InteractionAction(
                ACTION_DELEGATE_BACKGROUND,
                message="This needs slower background reasoning.",
                delegated_task=event.text_delta.strip(),
                confidence=0.70,
            )

        return InteractionAction(ACTION_SILENCE, confidence=0.80)

    def _should_interrupt_for_visual_event(self, event: MicroTurnEvent) -> bool:
        if not event.has_critical_visual_event():
            return False
        if event.anomaly_confidence < self.proactive_visual_threshold:
            return False
        if event.user_speaking_prob >= self.hard_interrupt_speech_threshold:
            return False
        return True

    @staticmethod
    def _looks_like_question(text_delta: str) -> bool:
        stripped = text_delta.strip().lower()
        if not stripped:
            return False
        return stripped.endswith("?") or stripped.startswith(("why ", "how ", "what ", "when ", "where ", "can ", "does "))


@dataclass(frozen=True)
class InteractionResidue:
    """Compressed evidence that the interaction policy failed a micro-turn."""

    failure_type: str
    timestamp_ms: int
    expected_action: str
    actual_action: str
    severity: float
    evidence: Dict[str, object]
    source_event: Dict[str, object]

    def __post_init__(self) -> None:
        if self.failure_type not in INTERACTION_FAILURE_TYPES:
            raise ValueError(f"unknown interaction failure type: {self.failure_type}")
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("severity must be in [0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return {
            "failure_type": self.failure_type,
            "timestamp_ms": self.timestamp_ms,
            "expected_action": self.expected_action,
            "actual_action": self.actual_action,
            "severity": float(self.severity),
            "evidence": dict(self.evidence),
            "source_event": dict(self.source_event),
        }

    def to_evaluator_decision(self, generation: int = 0) -> Dict[str, object]:
        """Export residue as an evaluator-evolution-compatible decision.

        Existing evaluator evolution expects validation-only candidate decision
        dictionaries.  Interaction residue is not an operator-program failure,
        but the same acceptance discipline can evaluate whether future policies
        reduce the residue under controls.
        """

        primitive_name = f"interaction_{self.failure_type}"
        return {
            "accepted": False,
            "generation": generation,
            "candidate_name": primitive_name,
            "rejection_reason": self.failure_type,
            "mean_improvement": -float(self.severity),
            "hidden_guard_regression": -float(self.severity),
            "runtime_behavior_difference": 1.0,
            "deterministic_replay_passed": True,
            "candidate_scores": {"interaction_quality_delta": -float(self.severity)},
            "no_op_control_delta": 0.0,
            "random_control_delta": -float(self.severity) * 0.5,
            "compile_status": {"elapsed_seconds": 0.0},
            "candidate_execution": {"elapsed_seconds": MICRO_TURN_MS / 1000.0},
            "candidate_program": {
                "program_id": primitive_name,
                "primitive_sequence": [
                    {"name": "micro_turn_observation"},
                    {"name": "interaction_policy"},
                    {"name": primitive_name},
                ],
            },
            "interaction_residue": self.to_dict(),
            "used_test_results": False,
        }


class InteractionEvaluator:
    """Scores interaction traces and emits residue records."""

    def __init__(
        self,
        *,
        speech_continuation_threshold: float = 0.72,
        hard_interrupt_speech_threshold: float = 0.88,
        proactive_visual_threshold: float = 0.70,
        delegation_uncertainty_threshold: float = 0.82,
        max_consecutive_backchannels: int = 2,
    ) -> None:
        self.speech_continuation_threshold = speech_continuation_threshold
        self.hard_interrupt_speech_threshold = hard_interrupt_speech_threshold
        self.proactive_visual_threshold = proactive_visual_threshold
        self.delegation_uncertainty_threshold = delegation_uncertainty_threshold
        self.max_consecutive_backchannels = max_consecutive_backchannels

    def evaluate_trace(
        self,
        events: Sequence[MicroTurnEvent],
        actions: Sequence[InteractionAction],
    ) -> Dict[str, object]:
        if len(events) != len(actions):
            raise ValueError("events and actions must have equal length")

        residues: List[InteractionResidue] = []
        consecutive_backchannels = 0
        action_counts: Dict[str, int] = {}

        for event, action in zip(events, actions):
            action_counts[action.kind] = action_counts.get(action.kind, 0) + 1
            if action.kind == ACTION_BACKCHANNEL:
                consecutive_backchannels += 1
            else:
                consecutive_backchannels = 0

            residues.extend(self._evaluate_pair(event, action, consecutive_backchannels))

        total_severity = sum(residue.severity for residue in residues)
        max_possible = max(1.0, float(len(events)))
        quality = max(0.0, 1.0 - total_severity / max_possible)
        return {
            "micro_turn_count": len(events),
            "action_counts": action_counts,
            "residue_count": len(residues),
            "total_residue_severity": float(total_severity),
            "interaction_quality": float(quality),
            "residues": [residue.to_dict() for residue in residues],
            "residue_objects": residues,
            "used_test_results": False,
        }

    def _evaluate_pair(
        self,
        event: MicroTurnEvent,
        action: InteractionAction,
        consecutive_backchannels: int,
    ) -> List[InteractionResidue]:
        residues: List[InteractionResidue] = []

        if event.expected_action is not None and action.kind != event.expected_action:
            residues.append(
                self._residue(
                    "expected_action_mismatch",
                    event,
                    event.expected_action,
                    action.kind,
                    0.70,
                    {"expected_action": event.expected_action},
                )
            )

        if action.kind == ACTION_INTERRUPT_WARNING and event.user_speaking_prob >= self.hard_interrupt_speech_threshold:
            residues.append(
                self._residue(
                    "bad_interrupt",
                    event,
                    ACTION_LISTEN,
                    action.kind,
                    min(1.0, event.user_speaking_prob),
                    {"user_speaking_prob": event.user_speaking_prob},
                )
            )

        if action.kind == ACTION_ANSWER and event.user_speaking_prob >= self.speech_continuation_threshold:
            residues.append(
                self._residue(
                    "premature_answer",
                    event,
                    ACTION_LISTEN,
                    action.kind,
                    min(1.0, event.user_speaking_prob),
                    {"user_speaking_prob": event.user_speaking_prob},
                )
            )

        if (
            action.kind == ACTION_ANSWER
            and event.semantic_uncertainty >= self.delegation_uncertainty_threshold
            and self._looks_like_question(event.text_delta)
        ):
            residues.append(
                self._residue(
                    "premature_answer",
                    event,
                    ACTION_DELEGATE_BACKGROUND,
                    action.kind,
                    max(0.55, event.semantic_uncertainty),
                    {"semantic_uncertainty": event.semantic_uncertainty, "text_delta": event.text_delta},
                )
            )

        if self._visual_event_requires_intervention(event) and action.kind != ACTION_INTERRUPT_WARNING:
            residues.append(
                self._residue(
                    "missed_visual_change",
                    event,
                    ACTION_INTERRUPT_WARNING,
                    action.kind,
                    max(0.50, event.anomaly_confidence),
                    {"visual_events": list(event.visual_events), "anomaly_confidence": event.anomaly_confidence},
                )
            )

        if event.background_result_ready and action.kind != ACTION_DELIVER_BACKGROUND_RESULT:
            residues.append(
                self._residue(
                    "late_background_result",
                    event,
                    ACTION_DELIVER_BACKGROUND_RESULT,
                    action.kind,
                    0.80,
                    {"background_result_summary": event.background_result_summary},
                )
            )

        if self._looks_like_question(event.text_delta) and event.user_speaking_prob < self.speech_continuation_threshold:
            if action.kind not in {ACTION_ANSWER, ACTION_DELEGATE_BACKGROUND, ACTION_DELIVER_BACKGROUND_RESULT}:
                residues.append(
                    self._residue(
                        "idle_when_questioned",
                        event,
                        ACTION_ANSWER,
                        action.kind,
                        0.60,
                        {"text_delta": event.text_delta},
                    )
                )

        if consecutive_backchannels > self.max_consecutive_backchannels:
            residues.append(
                self._residue(
                    "backchannel_overuse",
                    event,
                    ACTION_LISTEN,
                    action.kind,
                    0.45,
                    {"consecutive_backchannels": consecutive_backchannels},
                )
            )

        return residues

    def _visual_event_requires_intervention(self, event: MicroTurnEvent) -> bool:
        return (
            event.has_critical_visual_event()
            and event.anomaly_confidence >= self.proactive_visual_threshold
            and event.user_speaking_prob < self.hard_interrupt_speech_threshold
        )

    @staticmethod
    def _looks_like_question(text_delta: str) -> bool:
        return InteractionPolicy._looks_like_question(text_delta)

    @staticmethod
    def _residue(
        failure_type: str,
        event: MicroTurnEvent,
        expected_action: str,
        actual_action: str,
        severity: float,
        evidence: Dict[str, object],
    ) -> InteractionResidue:
        return InteractionResidue(
            failure_type=failure_type,
            timestamp_ms=event.timestamp_ms,
            expected_action=expected_action,
            actual_action=actual_action,
            severity=float(max(0.0, min(1.0, severity))),
            evidence=evidence,
            source_event=event.to_dict(),
        )


class InteractionResidueLayer:
    """Runnable bridge from real-time micro-turns to bounded RSI records."""

    def __init__(
        self,
        policy: InteractionPolicy | None = None,
        evaluator: InteractionEvaluator | None = None,
        background_bus: BackgroundAgentBus | None = None,
    ) -> None:
        self.policy = policy or InteractionPolicy()
        self.evaluator = evaluator or InteractionEvaluator()
        self.background_bus = background_bus or BackgroundAgentBus()
        self.state = InteractionState()
        self.events: List[MicroTurnEvent] = []
        self.actions: List[InteractionAction] = []

    def step(self, event: MicroTurnEvent) -> InteractionAction:
        action = self.policy.decide(event, self.state)
        if action.kind == ACTION_DELEGATE_BACKGROUND and action.delegated_task:
            self.background_bus.submit(event.timestamp_ms, action.delegated_task)
            self.state.background_pending = True
        self._update_state(event, action)
        self.events.append(event)
        self.actions.append(action)
        return action

    def run(self, events: Iterable[MicroTurnEvent]) -> List[InteractionAction]:
        return [self.step(event) for event in events]

    def evaluate(self) -> Dict[str, object]:
        return self.evaluator.evaluate_trace(self.events, self.actions)

    def export_evaluator_decisions(self, generation: int = 0) -> List[Dict[str, object]]:
        report = self.evaluate()
        residues = report["residue_objects"]
        return [residue.to_evaluator_decision(generation=generation) for residue in residues]

    def reset(self) -> None:
        self.state = InteractionState()
        self.events.clear()
        self.actions.clear()

    def _update_state(self, event: MicroTurnEvent, action: InteractionAction) -> None:
        self.state.turn_count += 1
        self.state.last_action_kind = action.kind
        if event.user_speaking_prob > 0.0 or event.audio_activity > 0.0:
            self.state.last_user_activity_ms = event.timestamp_ms
        if action.kind == ACTION_BACKCHANNEL:
            self.state.consecutive_backchannels += 1
        else:
            self.state.consecutive_backchannels = 0
        if action.kind == ACTION_INTERRUPT_WARNING:
            self.state.interrupt_count += 1
        if action.kind == ACTION_DELIVER_BACKGROUND_RESULT:
            self.state.delivered_background_count += 1
            self.state.background_pending = bool(self.background_bus.pending_tasks())

    def to_dict(self) -> Dict[str, object]:
        return {
            "state": self.state.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "actions": [action.to_dict() for action in self.actions],
            "background_bus": self.background_bus.to_dict(),
        }


def build_demo_trace() -> List[MicroTurnEvent]:
    """Return a deterministic trace that exercises the layer end-to-end."""

    return [
        MicroTurnEvent(0, text_delta="I am editing this function", user_speaking_prob=0.95, audio_activity=0.8),
        MicroTurnEvent(200, text_delta="", user_speaking_prob=0.76, audio_activity=0.7, semantic_uncertainty=0.7),
        MicroTurnEvent(
            400,
            text_delta="arr[i + 1] = value",
            visual_events=("bug_detected",),
            anomaly_confidence=0.88,
            user_speaking_prob=0.10,
        ),
        MicroTurnEvent(600, text_delta="Why did that fail?", user_speaking_prob=0.05, semantic_uncertainty=0.90),
        MicroTurnEvent(
            800,
            background_result_ready=True,
            background_result_summary="The likely failure is an off-by-one write.",
        ),
    ]
