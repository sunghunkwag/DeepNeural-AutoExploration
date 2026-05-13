import json

import pytest

from eie_core import (
    EIEController,
    EvaluatorSpec,
    ExperimentUnit,
    FailureResidue,
    FailureResidueLedger,
    InstrumentMutationContract,
    ObservationChannel,
    OperatorGeneratorSpec,
    ProblemSpaceVersion,
    ProblemSpaceVersionGraph,
)


def _residue(residue_id="r0", suspected="observation_channel:obs0", split="validation", severity=0.7):
    return FailureResidue(
        residue_id=residue_id,
        failure_type="hidden_gap_blind_spot",
        source="unit_test",
        evidence={"blind_spot": "hidden_gap", "missing_signal": "hidden_gap_probe"},
        severity=severity,
        suspected_instrument=suspected,
        generation=0,
        split=split,
    )


def _controller() -> EIEController:
    observation = ObservationChannel("obs0", ("validation_loss",), ("hidden_gap",), 1.0)
    evaluator = EvaluatorSpec("eval0", {"validation_to_hidden_gap": 0.1}, ())
    experiment = ExperimentUnit(
        "exp0",
        "synthetic",
        {"acceptance_splits": ["validation", "hidden_validation"]},
        {"hidden_validation_perturbations": [], "adversarial_controls": []},
        ("residue_pressure",),
    )
    generator = OperatorGeneratorSpec("opgen0", ("constant_lr", "maml_step"), {"brittle_primitives": ["memory_gate"]})
    version = ProblemSpaceVersion(
        "v0",
        None,
        ("obs0",),
        ("eval0",),
        ("exp0",),
        ("opgen0",),
        None,
        0,
    )
    return EIEController(
        observation_channels=(observation,),
        evaluator_specs=(evaluator,),
        experiment_units=(experiment,),
        operator_generators=(generator,),
        version_graph=ProblemSpaceVersionGraph(version),
    )


def test_failure_residue_severity_validation():
    with pytest.raises(ValueError):
        _residue(severity=1.5)
    with pytest.raises(ValueError):
        _residue(severity=-0.1)


def test_failure_residue_ledger_pressure_summary():
    ledger = FailureResidueLedger()
    ledger.record(_residue("r1", severity=0.5), acceptance_driving=True)
    ledger.record(_residue("r2", suspected="evaluator:eval0", severity=0.25), acceptance_driving=True)
    summary = ledger.pressure_summary()
    assert summary["observation_channel:obs0"]["pressure"] == pytest.approx(0.5)
    assert summary["evaluator:eval0"]["pressure"] == pytest.approx(0.25)
    assert ledger.unresolved_pressure_score() == pytest.approx(0.75)


def test_test_split_cannot_drive_instrument_acceptance():
    ledger = FailureResidueLedger()
    with pytest.raises(ValueError):
        ledger.record(_residue("heldout", split="test"), acceptance_driving=True)
    ledger.record(_residue("heldout_ok", split="test"), acceptance_driving=False)
    assert ledger.unresolved_pressure_score() == 0.0


def test_instrument_mutation_contract_rejects_test_signal_during_acceptance():
    contract = InstrumentMutationContract(
        "bad",
        "evaluator",
        "eval0",
        "test leakage",
        0.1,
        {"action": "strengthen_penalty", "penalty": "validation_to_hidden_gap"},
        0,
        uses_test_signal=True,
    )
    with pytest.raises(ValueError):
        contract.assert_allowed_for_acceptance()


def test_problem_space_version_graph_lineage():
    graph = ProblemSpaceVersionGraph(ProblemSpaceVersion("v0", None, ("obs0",), ("eval0",), ("exp0",), ("opgen0",), None, 0))
    graph.add_version(ProblemSpaceVersion("v1", "v0", ("obs0",), ("eval0",), ("exp0",), ("opgen0",), "c0", 1))
    graph.add_version(ProblemSpaceVersion("v2", "v1", ("obs0",), ("eval0",), ("exp0",), ("opgen0",), "c1", 2))
    assert graph.current_version.version_id == "v2"
    assert graph.lineage() == ["v0", "v1", "v2"]
    assert len(graph.to_dict()["edges"]) == 2


def test_eie_controller_proposes_mutation_from_repeated_residues():
    controller = _controller()
    controller.observe_failures([_residue("r1"), _residue("r2", split="hidden_validation")], acceptance_driving=True)
    diagnosis = controller.diagnose()
    contract = controller.propose_mutation_contract(diagnosis, generation=0)
    assert contract.target_instrument_type == "observation_channel"
    assert contract.mutation_payload["action"] == "reduce_blind_spot"


def test_no_op_mutation_is_rejected():
    controller = _controller()
    controller.observe_failures([_residue("r1"), _residue("r2", split="hidden_validation")], acceptance_driving=True)
    before_version = controller.version_graph.current_version.version_id
    decision = controller.evaluate_contract(controller.make_no_op_contract(target_instrument_type="observation_channel", target_instrument_id="obs0", generation=0))
    assert not decision["accepted"]
    assert decision["rollback"]
    assert controller.version_graph.current_version.version_id == before_version


def test_successful_mutation_reduces_residue_pressure():
    controller = _controller()
    controller.observe_failures([_residue("r1"), _residue("r2", split="hidden_validation")], acceptance_driving=True)
    result = controller.run_generation(generation=0)
    decision = result["decision"]
    assert decision["accepted"]
    assert decision["after_pressure"]["total_pressure"] < decision["before_pressure"]["total_pressure"]
    assert controller.measure_residue_pressure()["total_pressure"] == 0.0
    assert controller.version_graph.current_version.version_id == "v1"


def test_rollback_preserves_previous_version():
    controller = _controller()
    controller.observe_failures([_residue("r1")], acceptance_driving=True)
    accepted = controller.run_generation(generation=0)
    assert accepted["decision"]["accepted"]
    version_after_accept = controller.version_graph.current_version.version_id
    controller.observe_failures([_residue("r3", suspected="experiment_unit:exp0")], acceptance_driving=True)
    rejected = controller.evaluate_contract(controller.make_no_op_contract(target_instrument_type="experiment_unit", target_instrument_id="exp0", generation=1))
    assert not rejected["accepted"]
    assert controller.version_graph.current_version.version_id == version_after_accept
    assert controller.version_graph.version_count == 2


def test_json_exports_are_stable_and_serializable():
    controller = _controller()
    controller.observe_failures([_residue("r1")], acceptance_driving=True)
    controller.run_generation(generation=0)
    payload = controller.to_dict()
    first = json.dumps(payload, sort_keys=True)
    second = json.dumps(controller.to_dict(), sort_keys=True)
    assert first == second
    assert "problem_space_version_graph" in payload

