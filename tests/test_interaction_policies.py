from interaction_policies import (
    AblatedNoBackgroundPolicy,
    AblatedNoVisualPolicy,
    AlwaysInterruptPolicy,
    BaselineInteractionPolicy,
    NoOpInteractionPolicy,
    RandomInteractionPolicy,
)
from interaction_residue_layer import (
    ACTION_DELIVER_BACKGROUND_RESULT,
    ACTION_INTERRUPT_WARNING,
    ACTION_SILENCE,
    InteractionEvaluator,
    InteractionResidueLayer,
    MicroTurnEvent,
)


def test_no_op_policy_always_returns_silence():
    action = NoOpInteractionPolicy().decide(MicroTurnEvent(0, visual_events=("bug_detected",), anomaly_confidence=1.0), InteractionResidueLayer().state)
    assert action.kind == ACTION_SILENCE
    assert action.confidence == 1.0


def test_random_policy_is_seeded_and_reproducible():
    events = [MicroTurnEvent(i * 200, text_delta=f"What now {i}?") for i in range(6)]
    first = RandomInteractionPolicy(seed=123)
    second = RandomInteractionPolicy(seed=123)
    first_actions = [first.decide(event, InteractionResidueLayer().state).to_dict() for event in events]
    second_actions = [second.decide(event, InteractionResidueLayer().state).to_dict() for event in events]
    assert first_actions == second_actions
    assert len({action["kind"] for action in first_actions}) > 1


def test_always_interrupt_policy_interrupts_even_during_speech():
    event = MicroTurnEvent(0, user_speaking_prob=0.97, audio_activity=0.9)
    action = AlwaysInterruptPolicy().decide(event, InteractionResidueLayer().state)
    report = InteractionEvaluator().evaluate_trace([event], [action])
    assert action.kind == ACTION_INTERRUPT_WARNING
    assert report["residues"][0]["failure_type"] == "bad_interrupt"


def test_no_visual_ablation_misses_visual_events():
    event = MicroTurnEvent(0, visual_events=("safety_violation",), anomaly_confidence=0.93, user_speaking_prob=0.0)
    action = AblatedNoVisualPolicy().decide(event, InteractionResidueLayer().state)
    report = InteractionEvaluator().evaluate_trace([event], [action])
    assert action.kind != ACTION_INTERRUPT_WARNING
    assert report["residues"][0]["failure_type"] == "missed_visual_change"


def test_no_background_ablation_fails_background_delivery_tasks():
    event = MicroTurnEvent(0, background_result_ready=True, background_result_summary="result")
    action = AblatedNoBackgroundPolicy().decide(event, InteractionResidueLayer().state)
    report = InteractionEvaluator().evaluate_trace([event], [action])
    assert action.kind != ACTION_DELIVER_BACKGROUND_RESULT
    assert report["residues"][0]["failure_type"] == "late_background_result"


def test_baseline_delegates_high_uncertainty_question_but_no_background_answers_prematurely():
    event = MicroTurnEvent(0, text_delta="Why did this regress?", semantic_uncertainty=0.93)
    baseline_action = BaselineInteractionPolicy().decide(event, InteractionResidueLayer().state)
    ablated_action = AblatedNoBackgroundPolicy().decide(event, InteractionResidueLayer().state)
    report = InteractionEvaluator().evaluate_trace([event], [ablated_action])
    assert baseline_action.kind != ablated_action.kind
    assert report["residues"][0]["failure_type"] == "premature_answer"
