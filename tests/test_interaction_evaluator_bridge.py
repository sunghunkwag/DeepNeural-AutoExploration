import pytest

from evaluator_evolution import EvaluatorGenome, score_decision
from interaction_evaluator_bridge import InteractionEvaluatorBridge
from interaction_residue_layer import ACTION_SILENCE, InteractionAction, InteractionEvaluator, MicroTurnEvent


def _sample_residue():
    event = MicroTurnEvent(0, visual_events=("process_anomaly",), anomaly_confidence=0.93, user_speaking_prob=0.0)
    report = InteractionEvaluator().evaluate_trace([event], [InteractionAction(ACTION_SILENCE)])
    return report["residue_objects"][0]


def test_evaluator_bridge_exports_records_without_test_results():
    bridge = InteractionEvaluatorBridge(generation=4)
    records = bridge.to_decision_records([_sample_residue()], split="validation", control_policy_scores={"no_op": 0.2, "random": 0.4})
    record = records[0]
    assert record["generation"] == 4
    assert record["used_test_results"] is False
    assert record["candidate_metadata"]["source"] == "interaction_operating_scaffold"
    assert record["candidate_metadata"]["split"] == "validation"


def test_evaluator_bridge_includes_no_op_and_random_controls():
    bridge = InteractionEvaluatorBridge()
    record = bridge.to_decision_records([_sample_residue()], control_policy_scores={"no_op": 0.1, "random": 0.3})[0]
    assert "no_op_control_delta" in record
    assert "random_control_delta" in record
    assert record["control_policy_scores"] == {"no_op": 0.1, "random": 0.3}


def test_bridge_records_are_accepted_by_existing_evaluator_scoring_and_validation():
    bridge = InteractionEvaluatorBridge(generation=2)
    decisions = bridge.to_decision_records([_sample_residue()], control_policy_scores={"no_op": 0.2, "random": 0.4})
    genome = EvaluatorGenome()
    score = score_decision(genome.active, decisions[0], failure_rule_count=1)
    result = bridge.validate_candidate(genome, decisions, generation=2, failure_rule_count=1)
    assert score < 0.0
    assert result["generation"] == 2
    assert result["adversarial_checks"]["passed"] is True


def test_bridge_rejects_test_split_data():
    bridge = InteractionEvaluatorBridge()
    with pytest.raises(ValueError, match="test split"):
        bridge.to_decision_records([_sample_residue()], split="test")
