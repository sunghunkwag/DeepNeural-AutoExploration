"""Policy variants and controls for interaction scaffold benchmarks."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import List

from interaction_residue_layer import (
    ACTION_ANSWER,
    ACTION_BACKCHANNEL,
    ACTION_DELEGATE_BACKGROUND,
    ACTION_DELIVER_BACKGROUND_RESULT,
    ACTION_INTERRUPT_WARNING,
    ACTION_LISTEN,
    ACTION_SILENCE,
    InteractionAction,
    InteractionPolicy,
    InteractionState,
    MicroTurnEvent,
)


VALID_ACTIONS: List[str] = [
    ACTION_SILENCE,
    ACTION_LISTEN,
    ACTION_BACKCHANNEL,
    ACTION_INTERRUPT_WARNING,
    ACTION_ANSWER,
    ACTION_DELEGATE_BACKGROUND,
    ACTION_DELIVER_BACKGROUND_RESULT,
]


class BaselineInteractionPolicy:
    """Thin wrapper around the existing deterministic interaction policy."""

    def __init__(self, base_policy: InteractionPolicy | None = None) -> None:
        self.base_policy = base_policy or InteractionPolicy()

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        return self.base_policy.decide(event, state)


class NoOpInteractionPolicy:
    """Always emits silence."""

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        return InteractionAction(ACTION_SILENCE, confidence=1.0)


class AlwaysSilencePolicy(NoOpInteractionPolicy):
    """Named always-silence control used by benchmark reports."""


class RandomInteractionPolicy:
    """Seeded random valid-action policy."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        kind = self.rng.choice(VALID_ACTIONS)
        if kind == ACTION_DELEGATE_BACKGROUND:
            return InteractionAction(
                kind,
                message="Randomly delegated.",
                delegated_task=event.text_delta.strip() or "random delegated task",
                confidence=0.5,
            )
        if kind == ACTION_DELIVER_BACKGROUND_RESULT:
            return InteractionAction(
                kind,
                message=event.background_result_summary or "Randomly delivered result.",
                confidence=0.5,
            )
        if kind == ACTION_INTERRUPT_WARNING:
            return InteractionAction(kind, message="Random interrupt.", confidence=0.5)
        if kind == ACTION_BACKCHANNEL:
            return InteractionAction(kind, message="Random backchannel.", confidence=0.5)
        if kind == ACTION_ANSWER:
            return InteractionAction(kind, message="Random answer.", confidence=0.5)
        return InteractionAction(kind, confidence=0.5)


class AlwaysInterruptPolicy:
    """Always emits an interrupt-warning action."""

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        return InteractionAction(ACTION_INTERRUPT_WARNING, message="Always interrupt.", confidence=1.0)


class AlwaysListenPolicy:
    """Always emits listen."""

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        return InteractionAction(ACTION_LISTEN, confidence=1.0)


class AblatedNoVisualPolicy:
    """Runs the baseline policy after removing visual fields from the event."""

    def __init__(self, base_policy: InteractionPolicy | None = None) -> None:
        self.base_policy = base_policy or InteractionPolicy()

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        visual_blind_event = replace(event, visual_events=(), anomaly_confidence=0.0)
        return self.base_policy.decide(visual_blind_event, state)


class AblatedNoBackgroundPolicy:
    """Runs the baseline policy without result readiness or delegation cues."""

    def __init__(self, base_policy: InteractionPolicy | None = None) -> None:
        self.base_policy = base_policy or InteractionPolicy()

    def decide(self, event: MicroTurnEvent, state: InteractionState) -> InteractionAction:
        background_blind_event = replace(
            event,
            background_result_ready=False,
            background_result_summary="",
            semantic_uncertainty=0.0,
        )
        return self.base_policy.decide(background_blind_event, state)
