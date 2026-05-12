"""Stream recording and replay runtime for bounded interaction experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from interaction_residue_layer import (
    BackgroundAgentBus,
    InteractionAction,
    InteractionResidueLayer,
    InteractionState,
    MicroTurnEvent,
)


EVENT_FIELDS = {
    "timestamp_ms",
    "text_delta",
    "audio_activity",
    "visual_events",
    "user_speaking_prob",
    "model_speaking",
    "background_result_ready",
    "background_result_summary",
    "anomaly_confidence",
    "semantic_uncertainty",
    "expected_action",
}

ACTION_FIELDS = {"kind", "message", "confidence", "delegated_task", "latency_budget_ms"}

HIDDEN_LABEL_KEYS = {
    "expected_action",
    "expected_answer",
    "held_out_expected_answer",
    "held_out_label",
    "hidden_label",
    "test_label",
    "test_expected_answer",
}


def event_from_dict(payload: Dict[str, object]) -> MicroTurnEvent:
    data = {key: payload[key] for key in EVENT_FIELDS if key in payload}
    if "visual_events" in data:
        data["visual_events"] = tuple(data["visual_events"] or ())
    return MicroTurnEvent(**data)  # type: ignore[arg-type]


def action_from_dict(payload: Dict[str, object]) -> InteractionAction:
    data = {key: payload[key] for key in ACTION_FIELDS if key in payload}
    return InteractionAction(**data)  # type: ignore[arg-type]


def _json_safe(value: object) -> object:
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())  # type: ignore[attr-defined]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "residue_objects"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _assert_no_hidden_labels(payload: object, *, validation_only: bool, path: str = "record") -> None:
    if validation_only:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in HIDDEN_LABEL_KEYS and value not in (None, "", []):
                raise ValueError(f"hidden or held-out label field requires validation_only=True: {path}.{key}")
            _assert_no_hidden_labels(value, validation_only=validation_only, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _assert_no_hidden_labels(item, validation_only=validation_only, path=f"{path}[{index}]")


class InteractionStream:
    """Append-only stream of strictly increasing micro-turn events."""

    def __init__(self, events: Iterable[MicroTurnEvent] = ()) -> None:
        self._events: List[MicroTurnEvent] = []
        for event in events:
            self.append(event)

    def append(self, event: MicroTurnEvent) -> None:
        if self._events and event.timestamp_ms <= self._events[-1].timestamp_ms:
            raise ValueError("interaction stream timestamps must be strictly increasing")
        self._events.append(event)

    def extend(self, events: Iterable[MicroTurnEvent]) -> None:
        for event in events:
            self.append(event)

    def deterministic_replay(self) -> List[MicroTurnEvent]:
        return list(self._events)

    @property
    def events(self) -> List[MicroTurnEvent]:
        return list(self._events)

    def to_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps({"type": "event", "event": event.to_dict()}, sort_keys=True) + "\n")
        return target

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "InteractionStream":
        stream = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "event":
                    stream.append(event_from_dict(record["event"]))
                elif "timestamp_ms" in record:
                    stream.append(event_from_dict(record))
        return stream

    def __iter__(self) -> Iterator[MicroTurnEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)


class InteractionRecorder:
    """Writes event/action/state traces without leaking held-out labels."""

    def __init__(self) -> None:
        self.records: List[Dict[str, object]] = []

    def record_event(self, event: MicroTurnEvent, *, split: str = "validation", validation_only: bool = False) -> None:
        payload = event.to_dict()
        _assert_no_hidden_labels(payload, validation_only=validation_only)
        self.records.append(
            {
                "type": "event",
                "split": split,
                "validation_only": bool(validation_only),
                "event": payload,
            }
        )

    def record_action(self, action: InteractionAction, *, policy_name: str = "policy") -> None:
        self.records.append({"type": "action", "policy_name": policy_name, "action": action.to_dict()})

    def record_state(self, state: InteractionState | Dict[str, object], *, label: str = "state") -> None:
        payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
        self.records.append({"type": "state", "label": label, "state": _json_safe(payload)})

    def record_evaluator_report(self, report: Dict[str, object], *, validation_only: bool = False) -> None:
        payload = _json_safe(report)
        _assert_no_hidden_labels(payload, validation_only=validation_only)
        self.records.append(
            {
                "type": "evaluator_report",
                "validation_only": bool(validation_only),
                "report": payload,
            }
        )

    def record_background_state(self, background_state: BackgroundAgentBus | Dict[str, object]) -> None:
        payload = background_state.to_dict() if hasattr(background_state, "to_dict") else dict(background_state)
        self.records.append({"type": "background_state", "background_state": _json_safe(payload)})

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return target

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "InteractionRecorder":
        recorder = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            recorder.records = [json.loads(line) for line in handle if line.strip()]
        return recorder


class InteractionReplay:
    """Replay JSONL traces through a supplied interaction residue layer."""

    def __init__(self, records: Sequence[Dict[str, object]]) -> None:
        self.records = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "InteractionReplay":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls([json.loads(line) for line in handle if line.strip()])

    def replay(self, layer: InteractionResidueLayer, *, deterministic: bool = True) -> Dict[str, object]:
        events, recorded_actions = self._events_and_actions()
        self._validate_event_order(events)
        if hasattr(layer, "reset"):
            layer.reset()
        actions = [layer.step(event) for event in events]

        divergences: List[Dict[str, object]] = []
        if deterministic:
            for index, (actual, expected) in enumerate(zip(actions, recorded_actions)):
                if actual.to_dict() != expected.to_dict():
                    divergences.append(
                        {
                            "index": index,
                            "timestamp_ms": events[index].timestamp_ms,
                            "expected_action": expected.to_dict(),
                            "actual_action": actual.to_dict(),
                        }
                    )
            if len(recorded_actions) != len(actions):
                divergences.append(
                    {
                        "index": min(len(recorded_actions), len(actions)),
                        "expected_action_count": len(recorded_actions),
                        "actual_action_count": len(actions),
                    }
                )

        return {
            "event_count": len(events),
            "recorded_action_count": len(recorded_actions),
            "replayed_action_count": len(actions),
            "divergence_count": len(divergences),
            "divergences": divergences,
            "deterministic_replay_passed": not divergences,
            "actions": [action.to_dict() for action in actions],
        }

    def _events_and_actions(self) -> tuple[List[MicroTurnEvent], List[InteractionAction]]:
        events: List[MicroTurnEvent] = []
        actions: List[InteractionAction] = []
        for record in self.records:
            if record.get("type") == "event":
                events.append(event_from_dict(record["event"]))  # type: ignore[arg-type]
            elif record.get("type") == "action":
                actions.append(action_from_dict(record["action"]))  # type: ignore[arg-type]
        return events, actions

    @staticmethod
    def _validate_event_order(events: Sequence[MicroTurnEvent]) -> None:
        for previous, current in zip(events, events[1:]):
            if current.timestamp_ms <= previous.timestamp_ms:
                raise ValueError("replay trace timestamps are non-monotonic")
