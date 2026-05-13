"""Local JSON-safe structural memory for orchestrator decisions."""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import CandidateArtifact, OrchestratorDecision, json_safe


class StructuralMemoryStore:
    def __init__(self) -> None:
        self._accepted_lineage: list[dict[str, Any]] = []
        self._rejected_lineage: list[dict[str, Any]] = []
        self._failure_residues: list[dict[str, Any]] = []
        self._mutation_decisions: list[dict[str, Any]] = []
        self._decision_records: list[dict[str, Any]] = []

    @property
    def accepted_lineage(self) -> tuple[dict[str, Any], ...]:
        return tuple(json_safe(item) for item in self._accepted_lineage)

    @property
    def rejected_lineage(self) -> tuple[dict[str, Any], ...]:
        return tuple(json_safe(item) for item in self._rejected_lineage)

    @property
    def decision_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(json_safe(item) for item in self._decision_records)

    def store_decision(
        self,
        artifact: CandidateArtifact,
        decision: OrchestratorDecision,
        lineage: Mapping[str, Any],
    ) -> OrchestratorDecision:
        if not decision.accepted and not decision.rejection_reason:
            raise ValueError("rejected decisions require a rejection reason")
        entry = {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "producer": artifact.producer,
            "decision_id": decision.decision_id,
            "accepted": bool(decision.accepted),
            "rejection_reason": decision.rejection_reason,
            "residue_ids": list(decision.residue_ids),
            "problem_space_version_id": decision.problem_space_version_id,
            "lineage": json_safe(dict(lineage)),
        }
        self._decision_records.append(entry)
        if decision.accepted:
            self._accepted_lineage.append(entry)
        else:
            self._rejected_lineage.append(entry)
        return decision.with_memory_status(True)

    def store_failure_residue(self, residue: Any) -> None:
        if hasattr(residue, "to_dict"):
            payload = residue.to_dict()
        else:
            payload = dict(residue)
        self._failure_residues.append(json_safe(payload))

    def store_mutation_decision(self, decision: Mapping[str, Any]) -> None:
        self._mutation_decisions.append(json_safe(dict(decision)))

    @property
    def record_count(self) -> int:
        return (
            len(self._decision_records)
            + len(self._failure_residues)
            + len(self._mutation_decisions)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_lineage": json_safe(self._accepted_lineage),
            "rejected_lineage": json_safe(self._rejected_lineage),
            "failure_residues": json_safe(self._failure_residues),
            "eie_mutation_decisions": json_safe(self._mutation_decisions),
            "decision_records": json_safe(self._decision_records),
            "record_count": int(self.record_count),
        }

    def json_safe(self) -> dict[str, Any]:
        return json_safe(self.to_dict())
