from interaction_residue_layer import (
    ACTION_ANSWER,
    ACTION_BACKCHANNEL,
    ACTION_DELIVER_BACKGROUND_RESULT,
    ACTION_DELEGATE_BACKGROUND,
    ACTION_INTERRUPT_WARNING,
    ACTION_LISTEN,
    ACTION_SILENCE,
    BackgroundAgentBus,
    InteractionAction,
    InteractionEvaluator,
    InteractionPolicy,
    InteractionResidueLayer,
    MicroTurnEvent,
    build_demo_trace,
)
from evaluator_evolution import EvaluatorGenome, score_decision


def test_policy_listens_while_user_is_still_speaking():
    policy = InteractionPolicy()
    event = MicroTurnEvent(0, text_delta="I am still explaining", user_speaking_prob=0.91, audio_activity=0.8)
    action = policy.decide(event, state=InteractionResidueLayer().state)
    assert action.kind == ACTION_LISTEN


def test_policy_backchannels_under_uncertain_continuing_speech():
    layer = InteractionResidueLayer()
    event = MicroTurnEvent(200, user_speaking_prob=0.78, audio_activity=0.7, semantic_uncertainty=0.70)
    action = layer.step(event)
    assert action.kind == ACTION_BACKCHANNEL
    assert layer.state.consecutive_backchannels == 1


def test_policy_interrupts_for_high_confidence_visual_bug_when_user_is_not_speaking():
    policy = InteractionPolicy()
    event = MicroTurnEvent(
        400,
        text_delta="arr[i + 1] = value",
        visual_events=("bug_detected",),
        anomaly_confidence=0.90,
        user_speaking_prob=0.05,
    )
    action = policy.decide(event, state=InteractionResidueLayer().state)
    assert action.kind == ACTION_INTERRUPT_WARNING
    assert action.confidence >= 0.90


def test_policy_does_not_hard_interrupt_during_strong_user_speech():
    policy = InteractionPolicy()
    event = MicroTurnEvent(
        600,
        text_delta="I am not done",
        visual_events=("bug_detected",),
        anomaly_confidence=0.95,
        user_speaking_prob=0.93,
    )
    action = policy.decide(event, state=InteractionResidueLayer().state)
    assert action.kind in {ACTION_LISTEN, ACTION_BACKCHANNEL}


def test_policy_delegates_uncertain_question_to_background_bus():
    layer = InteractionResidueLayer()
    event = MicroTurnEvent(800, text_delta="Why did the recursive loop regress?", semantic_uncertainty=0.93)
    action = layer.step(event)
    pending = layer.background_bus.pending_tasks()
    assert action.kind == ACTION_DELEGATE_BACKGROUND
    assert len(pending) == 1
    assert pending[0].summary == "Why did the recursive loop regress?"


def test_background_bus_records_completion():
    bus = BackgroundAgentBus()
    task = bus.submit(1000, "inspect validation residue")
    completed = bus.complete(task.task_id, "validation overfit detected")
    assert completed.completed is True
    assert len(bus.ready_tasks()) == 1
    assert len(bus.pending_tasks()) == 0


def test_evaluator_detects_bad_interrupt():
    evaluator = InteractionEvaluator()
    event = MicroTurnEvent(0, user_speaking_prob=0.95, audio_activity=0.8)
    action = InteractionAction(ACTION_INTERRUPT_WARNING)
    report = evaluator.evaluate_trace([event], [action])
    residues = report["residues"]
    assert report["residue_count"] == 1
    assert residues[0]["failure_type"] == "bad_interrupt"
    assert residues[0]["expected_action"] == ACTION_LISTEN


def test_evaluator_detects_missed_visual_change():
    evaluator = InteractionEvaluator()
    event = MicroTurnEvent(
        200,
        visual_events=("safety_violation",),
        anomaly_confidence=0.91,
        user_speaking_prob=0.10,
    )
    action = InteractionAction(ACTION_SILENCE)
    report = evaluator.evaluate_trace([event], [action])
    residues = report["residues"]
    assert report["residue_count"] == 1
    assert residues[0]["failure_type"] == "missed_visual_change"
    assert residues[0]["expected_action"] == ACTION_INTERRUPT_WARNING


def test_evaluator_detects_late_background_result():
    evaluator = InteractionEvaluator()
    event = MicroTurnEvent(400, background_result_ready=True, background_result_summary="done")
    action = InteractionAction(ACTION_SILENCE)
    report = evaluator.evaluate_trace([event], [action])
    assert report["residues"][0]["failure_type"] == "late_background_result"
    assert report["residues"][0]["expected_action"] == ACTION_DELIVER_BACKGROUND_RESULT


def test_evaluator_detects_idle_question():
    evaluator = InteractionEvaluator()
    event = MicroTurnEvent(600, text_delta="What is the bug?", user_speaking_prob=0.0)
    action = InteractionAction(ACTION_SILENCE)
    report = evaluator.evaluate_trace([event], [action])
    assert report["residues"][0]["failure_type"] == "idle_when_questioned"
    assert report["residues"][0]["expected_action"] == ACTION_ANSWER


def test_layer_runs_demo_trace_without_residue_under_baseline_policy():
    layer = InteractionResidueLayer()
    actions = layer.run(build_demo_trace())
    report = layer.evaluate()
    assert len(actions) == 5
    assert report["micro_turn_count"] == 5
    assert report["interaction_quality"] == 1.0
    assert report["residue_count"] == 0


def test_layer_exports_evaluator_decisions_for_forced_policy_failure():
    layer = InteractionResidueLayer()
    event = MicroTurnEvent(
        0,
        visual_events=("process_anomaly",),
        anomaly_confidence=0.93,
        user_speaking_prob=0.0,
    )
    layer.events.append(event)
    layer.actions.append(InteractionAction(ACTION_SILENCE))
    decisions = layer.export_evaluator_decisions(generation=3)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision["generation"] == 3
    assert decision["accepted"] is False
    assert decision["rejection_reason"] == "missed_visual_change"
    assert decision["used_test_results"] is False
    assert "no_op_control_delta" in decision
    assert "random_control_delta" in decision


def test_exported_interaction_decision_can_be_scored_by_existing_evaluator():
    layer = InteractionResidueLayer()
    event = MicroTurnEvent(0, background_result_ready=True, background_result_summary="computed")
    layer.events.append(event)
    layer.actions.append(InteractionAction(ACTION_SILENCE))
    decision = layer.export_evaluator_decisions(generation=1)[0]
    genome = EvaluatorGenome()
    score = score_decision(genome.active, decision, failure_rule_count=1)
    assert isinstance(score, float)
    assert score < 0.0


def test_existing_evaluator_genome_can_validate_interaction_residue_candidates():
    layer = InteractionResidueLayer()
    event = MicroTurnEvent(0, text_delta="What now?", user_speaking_prob=0.0)
    layer.events.append(event)
    layer.actions.append(InteractionAction(ACTION_SILENCE))
    decisions = layer.export_evaluator_decisions(generation=2)
    genome = EvaluatorGenome()
    candidate = genome.propose(generation=2, failure_rule_count=1)[0]
    result = genome.validate_candidate(candidate, decisions, generation=2, failure_rule_count=1)
    assert result["evaluator_id"] == candidate.evaluator_id
    assert "adversarial_checks" in result


def test_micro_turn_event_rejects_invalid_probability():
    try:
        MicroTurnEvent(0, user_speaking_prob=1.2)
    except ValueError as exc:
        assert "user_speaking_prob" in str(exc)
    else:
        raise AssertionError("invalid probability was accepted")
