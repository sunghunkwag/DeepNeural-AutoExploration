"""Typed schemas for the orchestrator integration scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


ALLOWED_NODE_TYPES = (
    "orchestrator",
    "candidate_generator",
    "evaluator",
    "memory_store",
    "world_model",
    "representation_bridge",
    "cognitive_engine",
    "artifact",
    "failure_residue",
    "problem_space_version",
)

ALLOWED_EDGE_TYPES = (
    "produces",
    "consumes",
    "evaluates",
    "rejects",
    "accepts",
    "stores",
    "mutates",
    "explains",
    "routes_to",
)

ALLOWED_ARTIFACT_TYPES = (
    "operator_program",
    "primitive_candidate",
    "architecture_candidate",
    "evaluator_candidate",
    "observation_channel_candidate",
)

ACCEPTANCE_SPLITS = ("validation", "hidden_validation")
ALLOWED_EVALUATION_SPLITS = ACCEPTANCE_SPLITS + ("test",)


def _tuple_str(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)


def _json_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    payload = json_safe(dict(values))
    json.dumps(payload, sort_keys=True)
    return payload


@dataclass(frozen=True)
class SystemNode:
    node_id: str
    node_type: str
    repo_name: str | None
    module_name: str | None
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        if self.node_type not in ALLOWED_NODE_TYPES:
            raise ValueError(f"invalid node_type: {self.node_type}")
        object.__setattr__(self, "capabilities", _tuple_str(self.capabilities))
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "repo_name": self.repo_name,
            "module_name": self.module_name,
            "capabilities": list(self.capabilities),
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class SystemEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    artifact_type: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id is required")
        if not self.source_node_id or not self.target_node_id:
            raise ValueError("edge endpoints are required")
        if self.edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(f"invalid edge_type: {self.edge_type}")
        if self.artifact_type is not None and self.artifact_type not in ALLOWED_ARTIFACT_TYPES:
            raise ValueError(f"invalid artifact_type: {self.artifact_type}")
        object.__setattr__(self, "metadata", _json_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "artifact_type": self.artifact_type,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class CandidateArtifact:
    artifact_id: str
    producer: str
    artifact_type: str
    payload: dict[str, Any]
    declared_capabilities: tuple[str, ...]
    created_generation: int
    uses_test_signal: bool
    source_repo: str | None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if not self.producer:
            raise ValueError("producer is required")
        if self.artifact_type not in ALLOWED_ARTIFACT_TYPES:
            raise ValueError(f"invalid artifact_type: {self.artifact_type}")
        object.__setattr__(self, "payload", _json_dict(self.payload))
        object.__setattr__(self, "declared_capabilities", _tuple_str(self.declared_capabilities))
        object.__setattr__(self, "created_generation", int(self.created_generation))
        object.__setattr__(self, "uses_test_signal", bool(self.uses_test_signal))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "producer": self.producer,
            "artifact_type": self.artifact_type,
            "payload": json_safe(self.payload),
            "declared_capabilities": list(self.declared_capabilities),
            "created_generation": int(self.created_generation),
            "uses_test_signal": bool(self.uses_test_signal),
            "source_repo": self.source_repo,
        }


@dataclass(frozen=True)
class ArtifactEvaluation:
    artifact_id: str
    evaluator_id: str
    split: str
    metrics: dict[str, Any]
    passed: bool
    failure_type: str | None
    severity: float
    evidence: dict[str, Any]
    uses_test_signal: bool

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if not self.evaluator_id:
            raise ValueError("evaluator_id is required")
        if self.split not in ALLOWED_EVALUATION_SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        severity = float(self.severity)
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        if not self.passed and not self.failure_type:
            raise ValueError("failed evaluations require failure_type")
        object.__setattr__(self, "metrics", _json_dict(self.metrics))
        object.__setattr__(self, "evidence", _json_dict(self.evidence))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "uses_test_signal", bool(self.uses_test_signal))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "evaluator_id": self.evaluator_id,
            "split": self.split,
            "metrics": json_safe(self.metrics),
            "passed": bool(self.passed),
            "failure_type": self.failure_type,
            "severity": float(self.severity),
            "evidence": json_safe(self.evidence),
            "uses_test_signal": bool(self.uses_test_signal),
        }


@dataclass(frozen=True)
class OrchestratorDecision:
    decision_id: str
    artifact_id: str
    accepted: bool
    rejection_reason: str
    routed_to_eie: bool
    residue_ids: tuple[str, ...]
    problem_space_version_id: str | None
    stored_in_memory: bool
    uses_test_signal_for_acceptance: bool

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("decision_id is required")
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if not self.accepted and not self.rejection_reason:
            raise ValueError("rejected decisions require rejection_reason")
        if self.accepted and not self.rejection_reason:
            object.__setattr__(self, "rejection_reason", "accepted")
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "routed_to_eie", bool(self.routed_to_eie))
        object.__setattr__(self, "residue_ids", _tuple_str(self.residue_ids))
        object.__setattr__(self, "stored_in_memory", bool(self.stored_in_memory))
        object.__setattr__(self, "uses_test_signal_for_acceptance", bool(self.uses_test_signal_for_acceptance))

    def with_memory_status(self, stored: bool) -> "OrchestratorDecision":
        return OrchestratorDecision(
            decision_id=self.decision_id,
            artifact_id=self.artifact_id,
            accepted=self.accepted,
            rejection_reason=self.rejection_reason,
            routed_to_eie=self.routed_to_eie,
            residue_ids=self.residue_ids,
            problem_space_version_id=self.problem_space_version_id,
            stored_in_memory=stored,
            uses_test_signal_for_acceptance=self.uses_test_signal_for_acceptance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "artifact_id": self.artifact_id,
            "accepted": bool(self.accepted),
            "rejection_reason": self.rejection_reason,
            "routed_to_eie": bool(self.routed_to_eie),
            "residue_ids": list(self.residue_ids),
            "problem_space_version_id": self.problem_space_version_id,
            "stored_in_memory": bool(self.stored_in_memory),
            "uses_test_signal_for_acceptance": bool(self.uses_test_signal_for_acceptance),
        }
