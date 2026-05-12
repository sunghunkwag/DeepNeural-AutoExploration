"""Dependency-light adapters for synthetic micro-turn interaction streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from interaction_residue_layer import CRITICAL_VISUAL_EVENTS, MICRO_TURN_MS, MicroTurnEvent


def _clamp_probability(value: float, *, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _aligned_timestamp(timestamp_ms: int, alignment_ms: int = MICRO_TURN_MS) -> int:
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    return int(timestamp_ms // alignment_ms) * alignment_ms


class TextDeltaAdapter:
    """Converts incremental text chunks into micro-turn events."""

    def __init__(self, *, start_timestamp_ms: int = 0, step_ms: int = MICRO_TURN_MS) -> None:
        self.next_timestamp_ms = int(start_timestamp_ms)
        self.step_ms = int(step_ms)

    def adapt(
        self,
        chunk: str,
        *,
        timestamp_ms: int | None = None,
        user_speaking_prob: float = 0.0,
        semantic_uncertainty: float = 0.0,
    ) -> MicroTurnEvent:
        timestamp = self.next_timestamp_ms if timestamp_ms is None else int(timestamp_ms)
        if timestamp_ms is None:
            self.next_timestamp_ms += self.step_ms
        return MicroTurnEvent(
            timestamp,
            text_delta=str(chunk),
            user_speaking_prob=_clamp_probability(user_speaking_prob, name="user_speaking_prob"),
            semantic_uncertainty=_clamp_probability(semantic_uncertainty, name="semantic_uncertainty"),
        )

    def adapt_many(self, chunks: Iterable[str], *, start_timestamp_ms: int | None = None) -> List[MicroTurnEvent]:
        if start_timestamp_ms is not None:
            self.next_timestamp_ms = int(start_timestamp_ms)
        return [self.adapt(chunk) for chunk in chunks]


class SyntheticAudioActivityAdapter:
    """Maps scalar speech activity to audio and user-speaking event fields."""

    def adapt(self, activity: float, *, timestamp_ms: int) -> MicroTurnEvent:
        value = _clamp_probability(activity, name="activity")
        return MicroTurnEvent(timestamp_ms, audio_activity=value, user_speaking_prob=value)

    def fields(self, activity: float) -> Dict[str, float]:
        value = _clamp_probability(activity, name="activity")
        return {"audio_activity": value, "user_speaking_prob": value}


class SyntheticVisualEventAdapter:
    """Converts symbolic visual events into micro-turn visual fields."""

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = bool(strict)

    def adapt(self, visual_event: str, *, timestamp_ms: int, confidence: float = 1.0) -> MicroTurnEvent:
        label = str(visual_event)
        if self.strict and label not in CRITICAL_VISUAL_EVENTS:
            raise ValueError(f"unknown visual event label: {label}")
        return MicroTurnEvent(
            timestamp_ms,
            visual_events=(label,),
            anomaly_confidence=_clamp_probability(confidence, name="confidence"),
        )

    def fields(self, visual_event: str, *, confidence: float = 1.0) -> Dict[str, object]:
        label = str(visual_event)
        if self.strict and label not in CRITICAL_VISUAL_EVENTS:
            raise ValueError(f"unknown visual event label: {label}")
        return {"visual_events": (label,), "anomaly_confidence": _clamp_probability(confidence, name="confidence")}


@dataclass
class _Bucket:
    timestamp_ms: int
    text_chunks: List[str] = field(default_factory=list)
    audio_activity: float = 0.0
    user_speaking_prob: float = 0.0
    visual_events: List[str] = field(default_factory=list)
    anomaly_confidence: float = 0.0
    semantic_uncertainty: float = 0.0
    background_result_ready: bool = False
    background_result_summary: str = ""


class CombinedMicroTurnAdapter:
    """Merges text, audio, visual, and background updates into aligned packets."""

    def __init__(self, *, alignment_ms: int = MICRO_TURN_MS) -> None:
        if alignment_ms <= 0:
            raise ValueError("alignment_ms must be positive")
        self.alignment_ms = int(alignment_ms)
        self._buckets: Dict[int, _Bucket] = {}

    def add_text(
        self,
        timestamp_ms: int,
        chunk: str,
        *,
        semantic_uncertainty: float = 0.0,
        user_speaking_prob: float = 0.0,
    ) -> None:
        bucket = self._bucket(timestamp_ms)
        bucket.text_chunks.append(str(chunk))
        bucket.semantic_uncertainty = max(
            bucket.semantic_uncertainty,
            _clamp_probability(semantic_uncertainty, name="semantic_uncertainty"),
        )
        bucket.user_speaking_prob = max(
            bucket.user_speaking_prob,
            _clamp_probability(user_speaking_prob, name="user_speaking_prob"),
        )

    def add_audio(self, timestamp_ms: int, activity: float) -> None:
        bucket = self._bucket(timestamp_ms)
        value = _clamp_probability(activity, name="activity")
        bucket.audio_activity = max(bucket.audio_activity, value)
        bucket.user_speaking_prob = max(bucket.user_speaking_prob, value)

    def add_visual(self, timestamp_ms: int, visual_event: str, *, confidence: float = 1.0) -> None:
        bucket = self._bucket(timestamp_ms)
        bucket.visual_events.append(str(visual_event))
        bucket.anomaly_confidence = max(
            bucket.anomaly_confidence,
            _clamp_probability(confidence, name="confidence"),
        )

    def add_background_result(self, timestamp_ms: int, summary: str) -> None:
        bucket = self._bucket(timestamp_ms)
        bucket.background_result_ready = True
        bucket.background_result_summary = str(summary)

    def packets(self) -> List[MicroTurnEvent]:
        events = []
        for timestamp in sorted(self._buckets):
            bucket = self._buckets[timestamp]
            events.append(
                MicroTurnEvent(
                    timestamp_ms=timestamp,
                    text_delta="".join(bucket.text_chunks),
                    audio_activity=bucket.audio_activity,
                    visual_events=tuple(bucket.visual_events),
                    user_speaking_prob=bucket.user_speaking_prob,
                    background_result_ready=bucket.background_result_ready,
                    background_result_summary=bucket.background_result_summary,
                    anomaly_confidence=bucket.anomaly_confidence,
                    semantic_uncertainty=bucket.semantic_uncertainty,
                )
            )
        return events

    def clear(self) -> None:
        self._buckets.clear()

    def _bucket(self, timestamp_ms: int) -> _Bucket:
        timestamp = _aligned_timestamp(int(timestamp_ms), self.alignment_ms)
        if timestamp not in self._buckets:
            self._buckets[timestamp] = _Bucket(timestamp)
        return self._buckets[timestamp]
