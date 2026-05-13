"""In-memory artifact bus with validation-only acceptance controls."""

from __future__ import annotations

from typing import Any

from .schemas import (
    ACCEPTANCE_SPLITS,
    ALLOWED_ARTIFACT_TYPES,
    ArtifactEvaluation,
    CandidateArtifact,
    OrchestratorDecision,
    json_safe,
)


class ArtifactBus:
    def __init__(self) -> None:
        self._artifacts: dict[str, CandidateArtifact] = {}
        self._evaluations: dict[str, list[ArtifactEvaluation]] = {}
        self._decisions: dict[str, OrchestratorDecision] = {}
        self._decision_ids: set[str] = set()
        self._decisions_frozen = False

    @property
    def decisions_frozen(self) -> bool:
        return self._decisions_frozen

    @property
    def artifacts(self) -> dict[str, CandidateArtifact]:
        return dict(self._artifacts)

    @property
    def evaluations(self) -> dict[str, list[ArtifactEvaluation]]:
        return {key: list(value) for key, value in self._evaluations.items()}

    @property
    def decisions(self) -> dict[str, OrchestratorDecision]:
        return dict(self._decisions)

    def submit(self, artifact: CandidateArtifact, *, acceptance_requested: bool = False) -> None:
        if artifact.artifact_type not in ALLOWED_ARTIFACT_TYPES:
            raise ValueError(f"invalid artifact_type: {artifact.artifact_type}")
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"duplicate artifact_id: {artifact.artifact_id}")
        if acceptance_requested and artifact.uses_test_signal:
            raise ValueError("artifacts marked uses_test_signal cannot be submitted for acceptance")
        self._artifacts[artifact.artifact_id] = artifact
        self._evaluations[artifact.artifact_id] = []

    def record_evaluation(self, evaluation: ArtifactEvaluation, *, acceptance_driving: bool = True) -> None:
        if evaluation.artifact_id not in self._artifacts:
            raise KeyError(f"unknown artifact_id: {evaluation.artifact_id}")
        if evaluation.split == "test" and not self._decisions_frozen:
            raise ValueError("test split evaluations can only be recorded after decisions are frozen")
        if acceptance_driving:
            if evaluation.split not in ACCEPTANCE_SPLITS:
                raise ValueError(f"invalid acceptance split: {evaluation.split}")
            if evaluation.uses_test_signal:
                raise ValueError("evaluation marked uses_test_signal cannot drive acceptance")
        self._evaluations[evaluation.artifact_id].append(evaluation)

    def freeze_decisions(self) -> None:
        self._decisions_frozen = True

    def record_decision(self, decision: OrchestratorDecision) -> None:
        if decision.artifact_id not in self._artifacts:
            raise KeyError(f"unknown artifact_id: {decision.artifact_id}")
        if decision.artifact_id in self._decisions:
            raise ValueError(f"decision already recorded for artifact_id: {decision.artifact_id}")
        if decision.decision_id in self._decision_ids:
            raise ValueError(f"duplicate decision_id: {decision.decision_id}")
        if decision.uses_test_signal_for_acceptance:
            raise ValueError("orchestrator decisions cannot use test signal for acceptance")
        self._decisions[decision.artifact_id] = decision
        self._decision_ids.add(decision.decision_id)

    def get_artifacts(
        self,
        *,
        artifact_type: str | None = None,
        producer: str | None = None,
        status: str | None = None,
    ) -> tuple[CandidateArtifact, ...]:
        if artifact_type is not None and artifact_type not in ALLOWED_ARTIFACT_TYPES:
            raise ValueError(f"invalid artifact_type: {artifact_type}")
        if status is not None and status not in {"accepted", "rejected", "pending"}:
            raise ValueError(f"invalid status: {status}")
        matches: list[CandidateArtifact] = []
        for artifact in sorted(self._artifacts.values(), key=lambda item: item.artifact_id):
            if artifact_type is not None and artifact.artifact_type != artifact_type:
                continue
            if producer is not None and artifact.producer != producer:
                continue
            decision = self._decisions.get(artifact.artifact_id)
            current_status = "pending" if decision is None else ("accepted" if decision.accepted else "rejected")
            if status is not None and current_status != status:
                continue
            matches.append(artifact)
        return tuple(matches)

    def evaluations_for(self, artifact_id: str) -> tuple[ArtifactEvaluation, ...]:
        if artifact_id not in self._artifacts:
            raise KeyError(f"unknown artifact_id: {artifact_id}")
        return tuple(self._evaluations[artifact_id])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in sorted(self._artifacts.values(), key=lambda item: item.artifact_id)],
            "evaluations": {
                artifact_id: [evaluation.to_dict() for evaluation in evaluations]
                for artifact_id, evaluations in sorted(self._evaluations.items())
            },
            "decisions": [decision.to_dict() for decision in sorted(self._decisions.values(), key=lambda item: item.decision_id)],
            "decisions_frozen": bool(self._decisions_frozen),
            "artifact_count": len(self._artifacts),
            "evaluation_count": sum(len(items) for items in self._evaluations.values()),
        }

    def json_safe(self) -> dict[str, Any]:
        return json_safe(self.to_dict())
