"""Adapters for deterministic artifact fixtures and EIE residue routing."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from eie_core import (
    ACCEPTANCE_SPLITS as EIE_ACCEPTANCE_SPLITS,
    EIEController,
    EvaluatorSpec,
    ExperimentUnit,
    FailureResidue,
    FailureResidueLedger,
    ObservationChannel,
    OperatorGeneratorSpec,
    ProblemSpaceVersion,
    ProblemSpaceVersionGraph,
)

from .schemas import ACCEPTANCE_SPLITS, ArtifactEvaluation, CandidateArtifact, OrchestratorDecision


HPS_SOURCE_REPO = "Hybrid-Program-Synthesis-with-RSI"


class HybridProgramSynthesisAdapter:
    def __init__(self, *, source_repo: str = HPS_SOURCE_REPO) -> None:
        self.source_repo = source_repo

    def smoke_artifacts(self, seed: int) -> tuple[CandidateArtifact, ...]:
        rng = random.Random(seed)
        return (
            self._artifact("op_valid_seed42_g0", "hps_fixture_alpha", "operator_program", 0, 0.86, 0.80, "valid_operator"),
            self._artifact("op_invalid_seed42_g0", "hps_fixture_alpha", "operator_program", 0, 0.42, 0.36, "invalid_operator"),
            self._artifact("primitive_valid_seed42_g0", "hps_fixture_alpha", "primitive_candidate", 0, 0.78, 0.73, "valid_primitive"),
            self._artifact(
                "op_test_signal_seed42_g0",
                "hps_fixture_alpha",
                "operator_program",
                0,
                0.91,
                0.89,
                "test_signal_trap",
                uses_test_signal=True,
            ),
            self._artifact("noop_control_seed42_g0", "hps_fixture_alpha", "operator_program", 0, 0.50, 0.50, "no_op"),
            self._artifact(
                "random_control_seed42_g0",
                "hps_fixture_alpha",
                "primitive_candidate",
                0,
                0.48 + rng.random() * 0.02,
                0.47 + rng.random() * 0.02,
                "random",
            ),
        )

    def full_artifacts(self, seed: int) -> tuple[CandidateArtifact, ...]:
        rng = random.Random(seed)
        artifacts = [
            self._artifact("op_valid_a_seed42_g0", "hps_fixture_alpha", "operator_program", 0, 0.88, 0.82, "valid_operator"),
            self._artifact("op_valid_b_seed42_g1", "hps_fixture_beta", "operator_program", 1, 0.84, 0.79, "valid_operator"),
            self._artifact("op_invalid_a_seed42_g0", "hps_fixture_alpha", "operator_program", 0, 0.39, 0.34, "invalid_operator"),
            self._artifact("op_invalid_b_seed42_g1", "hps_fixture_beta", "operator_program", 1, 0.46, 0.31, "invalid_operator"),
            self._artifact("primitive_valid_seed42_g1", "hps_fixture_alpha", "primitive_candidate", 1, 0.77, 0.74, "valid_primitive"),
            self._artifact("primitive_invalid_seed42_g1", "hps_fixture_beta", "primitive_candidate", 1, 0.51, 0.41, "invalid_primitive"),
            self._artifact("architecture_candidate_seed42_g2", "architecture_fixture", "architecture_candidate", 2, 0.76, 0.72, "architecture_candidate"),
            self._artifact("evaluator_candidate_seed42_g2", "evaluation_fixture", "evaluator_candidate", 2, 0.58, 0.44, "evaluator_gap"),
            self._artifact("observation_channel_seed42_g2", "observation_fixture", "observation_channel_candidate", 2, 0.79, 0.75, "observation_channel"),
            self._artifact(
                "op_test_signal_seed42_g2",
                "hps_fixture_beta",
                "operator_program",
                2,
                0.94,
                0.91,
                "test_signal_trap",
                uses_test_signal=True,
            ),
            self._artifact("noop_control_seed42_g2", "hps_fixture_alpha", "operator_program", 2, 0.50, 0.50, "no_op"),
            self._artifact(
                "random_control_seed42_g2",
                "hps_fixture_beta",
                "primitive_candidate",
                2,
                0.49 + rng.random() * 0.03,
                0.45 + rng.random() * 0.03,
                "random",
            ),
        ]
        return tuple(artifacts)

    def _artifact(
        self,
        artifact_id: str,
        producer: str,
        artifact_type: str,
        generation: int,
        validation_score: float,
        hidden_validation_score: float,
        role: str,
        *,
        uses_test_signal: bool = False,
    ) -> CandidateArtifact:
        payload = {
            "role": role,
            "validation_score": float(validation_score),
            "hidden_validation_score": float(hidden_validation_score),
            "control_kind": role if role in {"no_op", "random"} else "candidate",
            "program_family": artifact_type,
        }
        return CandidateArtifact(
            artifact_id=artifact_id,
            producer=producer,
            artifact_type=artifact_type,
            payload=payload,
            declared_capabilities=(artifact_type, "deterministic_fixture"),
            created_generation=generation,
            uses_test_signal=uses_test_signal,
            source_repo=self.source_repo,
        )


def make_default_eie_controller() -> EIEController:
    observation = ObservationChannel(
        "orch_obs0",
        signal_names=("validation_score", "hidden_validation_score"),
        blind_spots=("artifact_failure_pattern",),
        cost=1.0,
    )
    evaluator = EvaluatorSpec(
        "orch_eval0",
        scoring_terms={"validation_to_hidden_gap": 0.10, "runtime_cost": 0.05, "instability": 0.10},
        adversarial_checks=("deterministic_replay",),
    )
    experiment = ExperimentUnit(
        "orch_exp0",
        "orchestrator_artifact_validation",
        {"acceptance_splits": list(EIE_ACCEPTANCE_SPLITS), "held_out_split": "test"},
        {"hidden_validation_perturbations": [], "adversarial_controls": ["no_op_control", "random_control"]},
        ("validation_score", "hidden_validation_score"),
    )
    generator = OperatorGeneratorSpec(
        "orch_opgen0",
        primitive_search_space=("candidate_operator", "primitive_candidate"),
        mutation_policy={"brittle_primitives": ["unstable_primitive"], "max_candidates": 8},
    )
    version = ProblemSpaceVersion(
        version_id="v0",
        parent_version_id=None,
        observation_channel_ids=(observation.channel_id,),
        evaluator_ids=(evaluator.evaluator_id,),
        experiment_unit_ids=(experiment.unit_id,),
        operator_generator_ids=(generator.generator_id,),
        created_by_contract_id=None,
        generation=0,
    )
    return EIEController(
        observation_channels=(observation,),
        evaluator_specs=(evaluator,),
        experiment_units=(experiment,),
        operator_generators=(generator,),
        version_graph=ProblemSpaceVersionGraph(version),
        ledger=FailureResidueLedger(),
    )


class DeepNeuralEIEAdapter:
    def __init__(self, controller: EIEController | None = None) -> None:
        self.controller = controller or make_default_eie_controller()
        self.routed_residues: list[FailureResidue] = []
        self.mutation_decisions: list[dict[str, object]] = []
        self.decisions: list[OrchestratorDecision] = []
        self.non_residue_reasons: list[dict[str, object]] = []

    def process_artifact(
        self,
        artifact: CandidateArtifact,
        evaluations: Sequence[ArtifactEvaluation],
        *,
        generation: int,
        require_hidden_validation: bool = True,
    ) -> OrchestratorDecision:
        for evaluation in evaluations:
            if evaluation.artifact_id != artifact.artifact_id:
                raise ValueError("evaluation artifact_id does not match artifact")
        acceptance_evaluations = [item for item in evaluations if item.split in ACCEPTANCE_SPLITS]
        validation_splits = {item.split for item in acceptance_evaluations}
        residues: list[FailureResidue] = []
        rejection_reason = ""

        if artifact.uses_test_signal:
            accepted = False
            rejection_reason = "artifact_uses_test_signal"
        elif any(item.uses_test_signal for item in acceptance_evaluations):
            accepted = False
            rejection_reason = "evaluation_uses_test_signal"
        elif "validation" not in validation_splits:
            accepted = False
            rejection_reason = "missing_validation_evidence"
        elif require_hidden_validation and "hidden_validation" not in validation_splits:
            accepted = False
            rejection_reason = "missing_hidden_validation_evidence"
        else:
            accepted = all(item.passed for item in acceptance_evaluations)
            if not accepted:
                rejection_reason = self._first_rejection_reason(acceptance_evaluations)
            else:
                rejection_reason = "accepted"

        for index, evaluation in enumerate(acceptance_evaluations):
            if not evaluation.passed:
                residues.append(self._failure_residue(artifact, evaluation, index))

        if not accepted and not residues:
            self.non_residue_reasons.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "reason": rejection_reason,
                    "split": "acceptance",
                }
            )

        mutation_result: dict[str, object] | None = None
        if residues:
            self.controller.observe_failures(residues, acceptance_driving=True)
            self.routed_residues.extend(residues)
            mutation_result = self.controller.run_generation(generation=generation)
            decision_payload = mutation_result.get("decision", {})
            if isinstance(decision_payload, dict):
                self.mutation_decisions.append(decision_payload)

        problem_space_version_id = self.controller.version_graph.current_version.version_id
        decision = OrchestratorDecision(
            decision_id=f"decision:{artifact.artifact_id}",
            artifact_id=artifact.artifact_id,
            accepted=accepted,
            rejection_reason=rejection_reason,
            routed_to_eie=bool(residues),
            residue_ids=tuple(residue.residue_id for residue in residues),
            problem_space_version_id=problem_space_version_id,
            stored_in_memory=False,
            uses_test_signal_for_acceptance=False,
        )
        self.decisions.append(decision)
        return decision

    def record_post_freeze_test_non_residue(self, evaluation: ArtifactEvaluation) -> None:
        if evaluation.split != "test":
            raise ValueError("only test split evaluations can be recorded as post-freeze non-residue records")
        if not evaluation.passed:
            self.non_residue_reasons.append(
                {
                    "artifact_id": evaluation.artifact_id,
                    "reason": "held_out_test_record_after_freeze_not_used_for_acceptance",
                    "split": "test",
                    "failure_type": evaluation.failure_type,
                }
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "controller": self.controller.to_dict(),
            "routed_residues": [residue.to_dict() for residue in self.routed_residues],
            "mutation_decisions": list(self.mutation_decisions),
            "orchestrator_decisions": [decision.to_dict() for decision in self.decisions],
            "non_residue_reasons": list(self.non_residue_reasons),
        }

    def _failure_residue(
        self,
        artifact: CandidateArtifact,
        evaluation: ArtifactEvaluation,
        index: int,
    ) -> FailureResidue:
        suspected_instrument = str(
            evaluation.evidence.get("suspected_instrument")
            or _suspected_instrument_for_artifact(artifact)
        )
        evidence = dict(evaluation.evidence)
        evidence.update(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "producer": artifact.producer,
            }
        )
        return FailureResidue(
            residue_id=f"residue:{artifact.artifact_id}:{evaluation.split}:{index}",
            failure_type=evaluation.failure_type or "artifact_evaluation_failed",
            source=f"orchestrator:{evaluation.evaluator_id}",
            evidence=evidence,
            severity=float(evaluation.severity),
            suspected_instrument=suspected_instrument,
            generation=int(artifact.created_generation),
            split=evaluation.split,
        )

    @staticmethod
    def _first_rejection_reason(evaluations: Iterable[ArtifactEvaluation]) -> str:
        failed = [item for item in evaluations if not item.passed]
        if not failed:
            return "acceptance_rule_not_satisfied"
        explicit = failed[0].evidence.get("rejection_reason")
        return str(explicit or failed[0].failure_type or "artifact_evaluation_failed")


def _suspected_instrument_for_artifact(artifact: CandidateArtifact) -> str:
    role = str(artifact.payload.get("role", ""))
    if artifact.artifact_type in {"operator_program", "primitive_candidate"}:
        if role == "no_op":
            return "experiment_unit:orch_exp0"
        if role == "random":
            return "operator_generator:orch_opgen0"
        return "operator_generator:orch_opgen0"
    if artifact.artifact_type == "evaluator_candidate":
        return "evaluator:orch_eval0"
    if artifact.artifact_type == "observation_channel_candidate":
        return "observation_channel:orch_obs0"
    return "problem_space:v0"
