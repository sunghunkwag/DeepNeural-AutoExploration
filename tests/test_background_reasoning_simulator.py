from background_reasoning_simulator import BackgroundReasoningSimulator
from interaction_residue_layer import BackgroundAgentBus


def test_simulator_completes_delegated_task_after_delay():
    bus = BackgroundAgentBus()
    task = bus.submit(100, "inspect failing trace")
    simulator = BackgroundReasoningSimulator(bus, delay_ms=300, seed=7)

    assert simulator.poll(399) == []
    events = simulator.poll(400)

    assert len(events) == 1
    assert events[0].timestamp_ms == 400
    assert events[0].background_result_ready is True
    assert task.task_id in events[0].background_result_summary
    assert len(bus.ready_tasks()) == 1
    assert len(bus.pending_tasks()) == 0


def test_missing_control_keeps_result_from_emitting():
    bus = BackgroundAgentBus()
    task = bus.submit(0, "slow result")
    simulator = BackgroundReasoningSimulator(bus, delay_ms=100, missing_task_ids=[task.task_id])

    events = simulator.poll(1000)

    assert events == []
    assert [pending.task_id for pending in bus.pending_tasks()] == [task.task_id]


def test_corrupted_control_emits_corrupted_summary_deterministically():
    bus = BackgroundAgentBus()
    task = bus.submit(0, "compute exact answer")
    simulator = BackgroundReasoningSimulator(bus, delay_ms=0, corrupted_task_ids=[task.task_id])

    first = simulator.poll(0)
    second = simulator.poll(0)

    assert len(first) == 1
    assert "CORRUPTED_RESULT" in first[0].background_result_summary
    assert second == []
