import pytest

from interaction_adapters import (
    CombinedMicroTurnAdapter,
    SyntheticAudioActivityAdapter,
    SyntheticVisualEventAdapter,
    TextDeltaAdapter,
)


def test_text_delta_adapter_assigns_incremental_timestamps_and_text():
    adapter = TextDeltaAdapter(start_timestamp_ms=400)
    events = adapter.adapt_many(["hel", "lo"])
    assert [event.timestamp_ms for event in events] == [400, 600]
    assert "".join(event.text_delta for event in events) == "hello"


def test_audio_activity_adapter_maps_scalar_to_speech_fields():
    event = SyntheticAudioActivityAdapter().adapt(0.73, timestamp_ms=200)
    assert event.audio_activity == 0.73
    assert event.user_speaking_prob == 0.73
    with pytest.raises(ValueError, match="activity"):
        SyntheticAudioActivityAdapter().adapt(1.5, timestamp_ms=400)


def test_visual_event_adapter_accepts_symbolic_events_and_rejects_unknown_when_strict():
    loose_event = SyntheticVisualEventAdapter().adapt("bug_detected_corrupted", timestamp_ms=0, confidence=0.8)
    assert loose_event.visual_events == ("bug_detected_corrupted",)
    assert loose_event.has_critical_visual_event() is False

    strict = SyntheticVisualEventAdapter(strict=True)
    good_event = strict.adapt("safety_violation", timestamp_ms=200, confidence=0.9)
    assert good_event.has_critical_visual_event() is True
    with pytest.raises(ValueError, match="unknown visual event"):
        strict.adapt("not_a_supported_event", timestamp_ms=400)


def test_combined_adapter_merges_updates_into_200_ms_packets():
    adapter = CombinedMicroTurnAdapter()
    adapter.add_text(40, "hel", semantic_uncertainty=0.2)
    adapter.add_text(120, "lo", semantic_uncertainty=0.6)
    adapter.add_audio(150, 0.74)
    adapter.add_visual(199, "tool_failure", confidence=0.91)
    adapter.add_background_result(260, "done")

    packets = adapter.packets()

    assert [packet.timestamp_ms for packet in packets] == [0, 200]
    assert packets[0].text_delta == "hello"
    assert packets[0].audio_activity == 0.74
    assert packets[0].user_speaking_prob == 0.74
    assert packets[0].visual_events == ("tool_failure",)
    assert packets[0].anomaly_confidence == 0.91
    assert packets[0].semantic_uncertainty == 0.6
    assert packets[1].background_result_ready is True
    assert packets[1].background_result_summary == "done"
