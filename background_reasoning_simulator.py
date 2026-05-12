"""Deterministic background reasoning simulator for interaction benchmarks."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Set

from interaction_residue_layer import BackgroundAgentBus, BackgroundTask, MICRO_TURN_MS, MicroTurnEvent


class BackgroundReasoningSimulator:
    """Completes delegated background tasks after simulated delay.

    The simulator is intentionally local and deterministic by default. It does
    not perform external calls; it only exercises the boundary between a fast
    interaction policy and slower background work.
    """

    def __init__(
        self,
        bus: BackgroundAgentBus,
        *,
        delay_ms: int = 400,
        deterministic: bool = True,
        seed: int = 0,
        missing_task_ids: Sequence[str] = (),
        corrupted_task_ids: Sequence[str] = (),
        extra_delay_ms_by_task: Dict[str, int] | None = None,
        missing_rate: float = 0.0,
        corruption_rate: float = 0.0,
    ) -> None:
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if not 0.0 <= float(missing_rate) <= 1.0:
            raise ValueError("missing_rate must be in [0, 1]")
        if not 0.0 <= float(corruption_rate) <= 1.0:
            raise ValueError("corruption_rate must be in [0, 1]")
        self.bus = bus
        self.delay_ms = int(delay_ms)
        self.deterministic = bool(deterministic)
        self.rng = random.Random(seed)
        self.missing_task_ids: Set[str] = set(missing_task_ids)
        self.corrupted_task_ids: Set[str] = set(corrupted_task_ids)
        self.extra_delay_ms_by_task = dict(extra_delay_ms_by_task or {})
        self.missing_rate = float(missing_rate)
        self.corruption_rate = float(corruption_rate)
        self._decided_missing: Dict[str, bool] = {}
        self._decided_corrupt: Dict[str, bool] = {}
        self._emitted_task_ids: Set[str] = set()

    def accept_delegated_tasks(self) -> List[BackgroundTask]:
        return self.bus.pending_tasks()

    def poll(self, timestamp_ms: int) -> List[MicroTurnEvent]:
        emitted: List[MicroTurnEvent] = []
        for task in list(self.bus.pending_tasks()):
            if task.task_id in self._emitted_task_ids:
                continue
            if self._is_missing(task):
                continue
            ready_at = task.created_timestamp_ms + self.delay_ms + self.extra_delay_ms_by_task.get(task.task_id, 0)
            if timestamp_ms < ready_at:
                continue
            summary = self._result_summary(task)
            completed = self.bus.complete(task.task_id, summary)
            self._emitted_task_ids.add(completed.task_id)
            emitted.append(
                MicroTurnEvent(
                    timestamp_ms=int(timestamp_ms) + len(emitted) * MICRO_TURN_MS,
                    background_result_ready=True,
                    background_result_summary=completed.result_summary,
                )
            )
        return emitted

    def advance_to(self, timestamp_ms: int) -> List[MicroTurnEvent]:
        return self.poll(timestamp_ms)

    def has_pending(self) -> bool:
        return bool(self.bus.pending_tasks())

    def _is_missing(self, task: BackgroundTask) -> bool:
        if task.task_id in self.missing_task_ids:
            return True
        if task.task_id not in self._decided_missing:
            self._decided_missing[task.task_id] = self.rng.random() < self.missing_rate
        return self._decided_missing[task.task_id]

    def _is_corrupted(self, task: BackgroundTask) -> bool:
        if task.task_id in self.corrupted_task_ids:
            return True
        if task.task_id not in self._decided_corrupt:
            self._decided_corrupt[task.task_id] = self.rng.random() < self.corruption_rate
        return self._decided_corrupt[task.task_id]

    def _result_summary(self, task: BackgroundTask) -> str:
        if self._is_corrupted(task):
            return f"CORRUPTED_RESULT for {task.task_id}"
        return f"Background result for {task.task_id}: {task.summary}"
