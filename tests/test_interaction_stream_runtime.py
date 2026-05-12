import json

import pytest

from interaction_residue_layer import (
    ACTION_INTERRUPT_WARNING,
    ACTION_LISTEN,
    InteractionResidueLayer,
    MicroTurnEvent,
)
from interaction_stream_runtime import InteractionRecorder, InteractionReplay, InteractionStream


def test_timestamps_must_be_strictly_increasing():
    stream = InteractionStream()
    stream.append(MicroTurnEvent(0))
    stream.append(MicroTurnEvent(200))
    with pytest.raises(ValueError, match="strictly increasing"):
        stream.append(MicroTurnEvent(200))


def test_shuffled_timestamps_fail_stream_import(tmp_path):
    trace_path = tmp_path / "shuffled.jsonl"
    records = [
        {"type": "event", "event": MicroTurnEvent(200).to_dict()},
        {"type": "event", "event": MicroTurnEvent(0).to_dict()},
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        InteractionStream.from_jsonl(trace_path)


def test_jsonl_replay_reproduces_actions_in_deterministic_mode(tmp_path):
    events = [
        MicroTurnEvent(0, user_speaking_prob=0.92, audio_activity=0.8),
        MicroTurnEvent(200, visual_events=("bug_detected",), anomaly_confidence=0.91, user_speaking_prob=0.05),
    ]
    layer = InteractionResidueLayer()
    recorder = InteractionRecorder()
    for event in events:
        recorder.record_event(event)
        recorder.record_action(layer.step(event), policy_name="baseline")
    trace_path = recorder.write_jsonl(tmp_path / "trace.jsonl")

    replay = InteractionReplay.from_jsonl(trace_path)
    report = replay.replay(InteractionResidueLayer(), deterministic=True)

    assert report["deterministic_replay_passed"] is True
    assert report["divergence_count"] == 0
    assert [action["kind"] for action in report["actions"]] == [ACTION_LISTEN, ACTION_INTERRUPT_WARNING]


def test_recorder_rejects_expected_action_without_validation_only():
    recorder = InteractionRecorder()
    event = MicroTurnEvent(0, expected_action=ACTION_LISTEN)
    with pytest.raises(ValueError, match="validation_only"):
        recorder.record_event(event)


def test_recorder_allows_expected_action_when_marked_validation_only():
    recorder = InteractionRecorder()
    event = MicroTurnEvent(0, expected_action=ACTION_LISTEN)
    recorder.record_event(event, validation_only=True)
    assert recorder.records[0]["validation_only"] is True
    assert recorder.records[0]["event"]["expected_action"] == ACTION_LISTEN
