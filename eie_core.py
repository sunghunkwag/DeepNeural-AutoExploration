"""Epistemic Instrument Evolution core primitives.

This module implements a bounded, auditable EIE loop.  EIE here means that
failure residues can trigger controlled changes to the instruments used for
observation, evaluation, experiment construction, operator generation, and
problem-space versioning.  Acceptance is deliberately narrow: validation and
hidden-validation pressure must be measured before and after the proposed
instrument mutation, and held-out test residues are never allowed to drive an
acceptance decision.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


ACCEPTANCE_SPLITS = ("validation", "hidden_validation")
INSTRUMENT_TYPES = (
    "observation_channel",
    "evaluator",
    "experiment_unit",
    "operator_generator",
    "problem_space",
)


def _tuple_str(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(str(value) for value in values)


def _json_dict(values: Mapping[str, object]) -> Dict[str, object]:
    return {str(key): value for key, value in values.items()}


def _parse_suspected_instrument(value: str) -> Tuple[str, str | None]:
    if ":" in value:
        instrument_type, instrument_id = value.split(":", 1)
        return instrument_type, instrument_id
    if value in INSTRUMENT_TYPES:
        return value, None
    return "problem_space", value or None


def _pressure_key(instrument_type: str, instrument_id: str | None) -> str:
    return f"{instrument_type}:{instrument_id}" if instrument_id else instrument_type


def _penalty_for_failure(failure_type: str, evidence: Mapping[str, object] | None = None) -> str:
    evidence = evidence or {}
    if "penalty" in evidence:
        return str(evidence["penalty"])
    lowered = failure_type.lower()
    if "hidden" in lowered or "gap" in lowered or "overfit" in lowered:
        return "validation_to_hidden_gap"
    if "runtime" in lowered or "cost" in lowered:
        return "runtime_cost"
    if "instability" in lowered or "deterministic" in lowered:
        return "instability"
    if "no_op" in lowered:
        return "no_op_control_failure"
    if "random" in lowered:
        return "random_control_failure"
    return "validation_to_hidden_gap"


@dataclass(frozen=True)
class FailureResidue:
    """Immutable evidence that an instrument failed to expose or control risk."""

    residue_id: str
    failure_type: str
    source: str
    evidence: Dict[str, object]
    severity: float
    suspected_instrument: str
    generation: int
    split: str

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        if not self.residue_id:
            raise ValueError("residue_id is required")
        if not self.failure_type:
            raise ValueError("failure_type is required")
        object.__setattr__(self, "severity", float(self.severity))
        object.__setattr__(self, "evidence", _json_dict(self.evidence))
        object.__setattr__(self, "split", str(self.split))

    def to_dict(self) -> Dict[str, object]:
        return {
            "residue_id": self.residue_id,
            "failure_type": self.failure_type,
            "source": self.source,
            "evidence": dict(self.evidence),
            "severity": float(self.severity),
            "suspected_instrument": self.suspected_instrument,
            "generation": int(self.generation),
            "split": self.split,
        }


class FailureResidueLedger:
    """Records residues and computes validation-only pressure summaries."""

    def __init__(self) -> None:
        self._records: List[Dict[str, object]] = []
        self._resolved_ids: set[str] = set()

    @property
    def records(self) -> List[FailureResidue]:
        return [entry["residue"] for entry in self._records]  # type: ignore[return-value]

    def record(self, residue: FailureResidue, *, acceptance_driving: bool = False) -> None:
        if acceptance_driving and residue.split == "test":
            raise ValueError("held-out test residues cannot drive instrument acceptance")
        if any(existing["residue"].residue_id == residue.residue_id for existing in self._records):  # type: ignore[index]
            raise ValueError(f"duplicate residue_id: {residue.residue_id}")
        self._records.append(
            {
                "residue": residue,
                "acceptance_driving": bool(acceptance_driving),
                "resolved": False,
            }
        )

    def extend(self, residues: Iterable[FailureResidue], *, acceptance_driving: bool = False) -> None:
        for residue in residues:
            self.record(residue, acceptance_driving=acceptance_driving)

    def resolve(self, residue_id: str) -> None:
        self._resolved_ids.add(residue_id)
        for entry in self._records:
            residue = entry["residue"]
            if isinstance(residue, FailureResidue) and residue.residue_id == residue_id:
                entry["resolved"] = True

    def pressure_summary(
        self,
        *,
        splits: Sequence[str] = ACCEPTANCE_SPLITS,
        unresolved_only: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        allowed = set(splits)
        summary: Dict[str, Dict[str, float]] = {}
        for entry in self._records:
            residue = entry["residue"]
            if not isinstance(residue, FailureResidue):
                continue
            if residue.split not in allowed:
                continue
            if unresolved_only and bool(entry.get("resolved")):
                continue
            instrument_type, instrument_id = _parse_suspected_instrument(residue.suspected_instrument)
            key = _pressure_key(instrument_type, instrument_id)
            bucket = summary.setdefault(key, {"pressure": 0.0, "count": 0.0})
            bucket["pressure"] += float(residue.severity)
            bucket["count"] += 1.0
        return {key: {"pressure": float(value["pressure"]), "count": float(value["count"])} for key, value in sorted(summary.items())}

    def unresolved_pressure_score(self, *, splits: Sequence[str] = ACCEPTANCE_SPLITS) -> float:
        return float(sum(bucket["pressure"] for bucket in self.pressure_summary(splits=splits).values()))

    def residues_for(self, suspected_instrument: str, *, splits: Sequence[str] = ACCEPTANCE_SPLITS) -> List[FailureResidue]:
        allowed = set(splits)
        target_type, target_id = _parse_suspected_instrument(suspected_instrument)
        matches: List[FailureResidue] = []
        for entry in self._records:
            residue = entry["residue"]
            if not isinstance(residue, FailureResidue) or residue.split not in allowed or bool(entry.get("resolved")):
                continue
            instrument_type, instrument_id = _parse_suspected_instrument(residue.suspected_instrument)
            if instrument_type == target_type and (target_id is None or instrument_id == target_id):
                matches.append(residue)
        return matches

    def to_dict(self) -> Dict[str, object]:
        return {
            "records": [
                {
                    **entry["residue"].to_dict(),  # type: ignore[union-attr]
                    "acceptance_driving": bool(entry.get("acceptance_driving", False)),
                    "resolved": bool(entry.get("resolved", False)),
                }
                for entry in self._records
            ],
            "pressure_summary": self.pressure_summary(),
            "unresolved_pressure_score": self.unresolved_pressure_score(),
        }


@dataclass(frozen=True)
class ObservationChannel:
    """Defines the observable signals and known blind spots for an EIE loop."""

    channel_id: str
    signal_names: Tuple[str, ...] = field(default_factory=tuple)
    blind_spots: Tuple[str, ...] = field(default_factory=tuple)
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.channel_id:
            raise ValueError("channel_id is required")
        object.__setattr__(self, "signal_names", _tuple_str(self.signal_names))
        object.__setattr__(self, "blind_spots", _tuple_str(self.blind_spots))
        object.__setattr__(self, "cost", float(self.cost))

    def mutate(self, payload: Mapping[str, object]) -> "ObservationChannel":
        action = str(payload.get("action", ""))
        signals = list(self.signal_names)
        blind_spots = list(self.blind_spots)
        if action == "add_signal":
            signal = str(payload.get("signal", ""))
            if signal and signal not in signals:
                signals.append(signal)
        elif action == "reduce_blind_spot":
            blind_spot = str(payload.get("blind_spot", ""))
            if blind_spot in blind_spots:
                blind_spots.remove(blind_spot)
            signal = str(payload.get("signal", ""))
            if signal and signal not in signals:
                signals.append(signal)
        elif action == "no_op":
            pass
        else:
            raise ValueError(f"unsupported observation mutation: {action}")
        return ObservationChannel(self.channel_id, tuple(sorted(signals)), tuple(sorted(blind_spots)), self.cost)

    def to_dict(self) -> Dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "signal_names": list(self.signal_names),
            "blind_spots": list(self.blind_spots),
            "cost": float(self.cost),
        }


@dataclass(frozen=True)
class EvaluatorSpec:
    """Defines validation scoring terms and adversarial checks."""

    evaluator_id: str
    scoring_terms: Dict[str, float] = field(default_factory=dict)
    adversarial_checks: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.evaluator_id:
            raise ValueError("evaluator_id is required")
        object.__setattr__(self, "scoring_terms", {str(k): float(v) for k, v in self.scoring_terms.items()})
        object.__setattr__(self, "adversarial_checks", _tuple_str(self.adversarial_checks))

    def mutate(self, payload: Mapping[str, object]) -> "EvaluatorSpec":
        action = str(payload.get("action", ""))
        scoring_terms = dict(self.scoring_terms)
        checks = list(self.adversarial_checks)
        if action in {"add_penalty", "strengthen_penalty"}:
            penalty = str(payload.get("penalty", ""))
            if penalty not in {
                "validation_to_hidden_gap",
                "runtime_cost",
                "instability",
                "no_op_control_failure",
                "random_control_failure",
            }:
                raise ValueError(f"unsupported evaluator penalty: {penalty}")
            delta = float(payload.get("delta", 0.25))
            scoring_terms[penalty] = max(0.0, scoring_terms.get(penalty, 0.0) + delta)
            check = str(payload.get("adversarial_check", penalty))
            if check and check not in checks:
                checks.append(check)
        elif action == "no_op":
            pass
        else:
            raise ValueError(f"unsupported evaluator mutation: {action}")
        return EvaluatorSpec(self.evaluator_id, scoring_terms, tuple(sorted(checks)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "evaluator_id": self.evaluator_id,
            "scoring_terms": {key: float(value) for key, value in sorted(self.scoring_terms.items())},
            "adversarial_checks": list(self.adversarial_checks),
        }


@dataclass(frozen=True)
class ExperimentUnit:
    """Defines a bounded experiment family and its validation controls."""

    unit_id: str
    task_family: str
    split_policy: Dict[str, object]
    perturbation_policy: Dict[str, object]
    metrics: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id is required")
        object.__setattr__(self, "split_policy", _json_dict(self.split_policy))
        object.__setattr__(self, "perturbation_policy", _json_dict(self.perturbation_policy))
        object.__setattr__(self, "metrics", _tuple_str(self.metrics))

    def mutate(self, payload: Mapping[str, object]) -> "ExperimentUnit":
        action = str(payload.get("action", ""))
        split_policy = copy.deepcopy(self.split_policy)
        perturbation_policy = copy.deepcopy(self.perturbation_policy)
        metrics = list(self.metrics)
        if action == "add_hidden_validation_perturbation":
            perturbation = str(payload.get("perturbation", "hidden_validation_shift"))
            items = list(perturbation_policy.get("hidden_validation_perturbations", []))
            if perturbation not in items:
                items.append(perturbation)
            perturbation_policy["hidden_validation_perturbations"] = sorted(items)
        elif action == "add_adversarial_control":
            control = str(payload.get("control", "adversarial_control"))
            controls = list(perturbation_policy.get("adversarial_controls", []))
            if control not in controls:
                controls.append(control)
            perturbation_policy["adversarial_controls"] = sorted(controls)
            metric = f"{control}_score"
            if metric not in metrics:
                metrics.append(metric)
        elif action == "no_op":
            pass
        else:
            raise ValueError(f"unsupported experiment mutation: {action}")
        return ExperimentUnit(self.unit_id, self.task_family, split_policy, perturbation_policy, tuple(sorted(metrics)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "task_family": self.task_family,
            "split_policy": dict(self.split_policy),
            "perturbation_policy": copy.deepcopy(self.perturbation_policy),
            "metrics": list(self.metrics),
        }


@dataclass(frozen=True)
class OperatorGeneratorSpec:
    """Defines the primitive search space and bounded mutation policy."""

    generator_id: str
    primitive_search_space: Tuple[str, ...] = field(default_factory=tuple)
    mutation_policy: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generator_id:
            raise ValueError("generator_id is required")
        object.__setattr__(self, "primitive_search_space", _tuple_str(self.primitive_search_space))
        object.__setattr__(self, "mutation_policy", _json_dict(self.mutation_policy))

    def mutate(self, payload: Mapping[str, object]) -> "OperatorGeneratorSpec":
        action = str(payload.get("action", ""))
        primitives = list(self.primitive_search_space)
        policy = copy.deepcopy(self.mutation_policy)
        if action == "expand_primitive_diversity":
            primitive = str(payload.get("primitive", "stability_probe"))
            if primitive and primitive not in primitives:
                primitives.append(primitive)
        elif action == "reduce_brittle_primitive":
            primitive = str(payload.get("primitive", ""))
            brittle = list(policy.get("brittle_primitives", []))
            if primitive in brittle:
                brittle.remove(primitive)
            policy["brittle_primitives"] = sorted(brittle)
            avoided = list(policy.get("avoid_primitives", []))
            if primitive and primitive not in avoided:
                avoided.append(primitive)
            policy["avoid_primitives"] = sorted(avoided)
        elif action == "no_op":
            pass
        else:
            raise ValueError(f"unsupported operator-generator mutation: {action}")
        return OperatorGeneratorSpec(self.generator_id, tuple(sorted(primitives)), policy)

    def to_dict(self) -> Dict[str, object]:
        return {
            "generator_id": self.generator_id,
            "primitive_search_space": list(self.primitive_search_space),
            "mutation_policy": copy.deepcopy(self.mutation_policy),
        }


@dataclass(frozen=True)
class InstrumentMutationContract:
    """A proposed bounded mutation to one EIE instrument."""

    contract_id: str
    target_instrument_type: str
    target_instrument_id: str
    reason: str
    expected_residue_reduction: float
    mutation_payload: Dict[str, object]
    created_generation: int
    uses_test_signal: bool = False

    def __post_init__(self) -> None:
        if self.target_instrument_type not in INSTRUMENT_TYPES:
            raise ValueError(f"unknown instrument type: {self.target_instrument_type}")
        if not self.contract_id:
            raise ValueError("contract_id is required")
        object.__setattr__(self, "expected_residue_reduction", float(self.expected_residue_reduction))
        object.__setattr__(self, "mutation_payload", _json_dict(self.mutation_payload))

    def assert_allowed_for_acceptance(self) -> None:
        if self.uses_test_signal:
            raise ValueError("instrument mutation acceptance cannot use held-out test signals")

    def to_dict(self) -> Dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "target_instrument_type": self.target_instrument_type,
            "target_instrument_id": self.target_instrument_id,
            "reason": self.reason,
            "expected_residue_reduction": float(self.expected_residue_reduction),
            "mutation_payload": dict(self.mutation_payload),
            "created_generation": int(self.created_generation),
            "uses_test_signal": bool(self.uses_test_signal),
        }


@dataclass(frozen=True)
class ProblemSpaceVersion:
    """A lineage node for the current observable/evaluable problem space."""

    version_id: str
    parent_version_id: str | None
    observation_channel_ids: Tuple[str, ...]
    evaluator_ids: Tuple[str, ...]
    experiment_unit_ids: Tuple[str, ...]
    operator_generator_ids: Tuple[str, ...]
    created_by_contract_id: str | None
    generation: int

    def __post_init__(self) -> None:
        if not self.version_id:
            raise ValueError("version_id is required")
        object.__setattr__(self, "observation_channel_ids", _tuple_str(self.observation_channel_ids))
        object.__setattr__(self, "evaluator_ids", _tuple_str(self.evaluator_ids))
        object.__setattr__(self, "experiment_unit_ids", _tuple_str(self.experiment_unit_ids))
        object.__setattr__(self, "operator_generator_ids", _tuple_str(self.operator_generator_ids))

    def to_dict(self) -> Dict[str, object]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "observation_channel_ids": list(self.observation_channel_ids),
            "evaluator_ids": list(self.evaluator_ids),
            "experiment_unit_ids": list(self.experiment_unit_ids),
            "operator_generator_ids": list(self.operator_generator_ids),
            "created_by_contract_id": self.created_by_contract_id,
            "generation": int(self.generation),
        }


class ProblemSpaceVersionGraph:
    """Maintains problem-space versions and accepted mutation edges."""

    def __init__(self, initial_version: ProblemSpaceVersion) -> None:
        if initial_version.parent_version_id is not None:
            raise ValueError("initial problem-space version must not have a parent")
        self._versions: Dict[str, ProblemSpaceVersion] = {initial_version.version_id: initial_version}
        self._edges: List[Dict[str, object]] = []
        self._current_version_id = initial_version.version_id

    @property
    def current_version(self) -> ProblemSpaceVersion:
        return self._versions[self._current_version_id]

    @property
    def version_count(self) -> int:
        return len(self._versions)

    def add_version(self, version: ProblemSpaceVersion) -> None:
        if version.version_id in self._versions:
            raise ValueError(f"duplicate problem-space version: {version.version_id}")
        if version.parent_version_id not in self._versions:
            raise ValueError(f"unknown parent problem-space version: {version.parent_version_id}")
        self._versions[version.version_id] = version
        self._edges.append(
            {
                "parent_version_id": version.parent_version_id,
                "child_version_id": version.version_id,
                "contract_id": version.created_by_contract_id,
                "generation": int(version.generation),
            }
        )
        self._current_version_id = version.version_id

    def lineage(self, version_id: str | None = None) -> List[str]:
        current_id = version_id or self._current_version_id
        lineage: List[str] = []
        while current_id is not None:
            version = self._versions[current_id]
            lineage.append(version.version_id)
            current_id = version.parent_version_id
        return list(reversed(lineage))

    def to_dict(self) -> Dict[str, object]:
        return {
            "current_version_id": self._current_version_id,
            "versions": [version.to_dict() for version in sorted(self._versions.values(), key=lambda item: item.version_id)],
            "edges": list(self._edges),
            "lineage": self.lineage(),
        }


@dataclass(frozen=True)
class EIEDiagnosis:
    """Deterministic diagnosis of which instrument is under residue pressure."""

    target_instrument_type: str
    target_instrument_id: str
    pressure_score: float
    reason: str
    pressure_summary: Dict[str, Dict[str, float]]

    @staticmethod
    def from_pressure_summary(summary: Mapping[str, Mapping[str, float]]) -> "EIEDiagnosis":
        if not summary:
            return EIEDiagnosis("problem_space", "problem_space", 0.0, "no unresolved pressure", {})
        order = {name: idx for idx, name in enumerate(INSTRUMENT_TYPES)}
        candidates: List[Tuple[float, int, str, str | None, str]] = []
        for key, bucket in summary.items():
            instrument_type, instrument_id = _parse_suspected_instrument(key)
            pressure = float(bucket.get("pressure", 0.0))
            candidates.append((-pressure, order.get(instrument_type, len(order)), instrument_type, instrument_id, key))
        candidates.sort()
        pressure = -candidates[0][0]
        instrument_type = candidates[0][2]
        instrument_id = candidates[0][3] or instrument_type
        return EIEDiagnosis(
            instrument_type,
            instrument_id,
            float(pressure),
            f"highest unresolved pressure is {pressure:.6f} on {candidates[0][4]}",
            {key: {"pressure": float(value.get("pressure", 0.0)), "count": float(value.get("count", 0.0))} for key, value in sorted(summary.items())},
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_instrument_type": self.target_instrument_type,
            "target_instrument_id": self.target_instrument_id,
            "pressure_score": float(self.pressure_score),
            "reason": self.reason,
            "pressure_summary": copy.deepcopy(self.pressure_summary),
        }


class EIEController:
    """Coordinates bounded EIE diagnosis, mutation, validation, and rollback."""

    def __init__(
        self,
        *,
        observation_channels: Sequence[ObservationChannel],
        evaluator_specs: Sequence[EvaluatorSpec],
        experiment_units: Sequence[ExperimentUnit],
        operator_generators: Sequence[OperatorGeneratorSpec],
        version_graph: ProblemSpaceVersionGraph,
        ledger: FailureResidueLedger | None = None,
    ) -> None:
        self.observation_channels = {item.channel_id: item for item in observation_channels}
        self.evaluator_specs = {item.evaluator_id: item for item in evaluator_specs}
        self.experiment_units = {item.unit_id: item for item in experiment_units}
        self.operator_generators = {item.generator_id: item for item in operator_generators}
        self.version_graph = version_graph
        self.ledger = ledger or FailureResidueLedger()
        self.contracts: List[InstrumentMutationContract] = []
        self.decisions: List[Dict[str, object]] = []
        self.rollback_count = 0
        self.used_test_signal_for_acceptance = False
        self.acceptance_pressure_measurements: List[Dict[str, object]] = []

    def observe_failures(self, residues: Iterable[FailureResidue], *, acceptance_driving: bool = True) -> None:
        self.ledger.extend(residues, acceptance_driving=acceptance_driving)

    def diagnose(self) -> EIEDiagnosis:
        return EIEDiagnosis.from_pressure_summary(self.ledger.pressure_summary())

    def propose_mutation_contract(self, diagnosis: EIEDiagnosis, *, generation: int) -> InstrumentMutationContract:
        target_type = diagnosis.target_instrument_type
        target_id = self._resolve_target_id(target_type, diagnosis.target_instrument_id)
        payload = self._proposal_payload(target_type, target_id)
        return InstrumentMutationContract(
            contract_id=f"contract_g{generation}_{len(self.contracts)}_{target_type}_{target_id}",
            target_instrument_type=target_type,
            target_instrument_id=target_id,
            reason=diagnosis.reason,
            expected_residue_reduction=max(0.0, diagnosis.pressure_score * 0.25),
            mutation_payload=payload,
            created_generation=generation,
            uses_test_signal=False,
        )

    def evaluate_contract(self, contract: InstrumentMutationContract) -> Dict[str, object]:
        contract.assert_allowed_for_acceptance()
        self.contracts.append(contract)
        before = self.measure_residue_pressure()
        snapshots = self._snapshot_instruments()
        mutated = self._mutated_snapshot(contract, snapshots)
        after = self.measure_residue_pressure(instruments=mutated)
        measurement = {
            "contract_id": contract.contract_id,
            "before_pressure": before,
            "after_pressure": after,
            "splits": list(ACCEPTANCE_SPLITS),
        }
        self.acceptance_pressure_measurements.append(measurement)
        accepted = bool(after["total_pressure"] < before["total_pressure"])
        reason = "accepted" if accepted else "residue_pressure_not_reduced"
        lineage_recorded = False
        if accepted:
            self._commit_snapshot(mutated)
            for residue in self.ledger.residues_for(_pressure_key(contract.target_instrument_type, contract.target_instrument_id)):
                self.ledger.resolve(residue.residue_id)
            current = self.version_graph.current_version
            next_version = ProblemSpaceVersion(
                version_id=f"v{self.version_graph.version_count}",
                parent_version_id=current.version_id,
                observation_channel_ids=tuple(sorted(self.observation_channels)),
                evaluator_ids=tuple(sorted(self.evaluator_specs)),
                experiment_unit_ids=tuple(sorted(self.experiment_units)),
                operator_generator_ids=tuple(sorted(self.operator_generators)),
                created_by_contract_id=contract.contract_id,
                generation=contract.created_generation,
            )
            self.version_graph.add_version(next_version)
            lineage_recorded = True
        else:
            self.rollback_count += 1
        decision = {
            "contract_id": contract.contract_id,
            "contract": contract.to_dict(),
            "accepted": accepted,
            "rollback": not accepted,
            "rejection_reason": reason,
            "before_pressure": before,
            "after_pressure": after,
            "pressure_delta": float(before["total_pressure"] - after["total_pressure"]),
            "lineage_recorded": lineage_recorded,
            "validation_only": True,
            "used_test_signal_for_acceptance": False,
            "accepted_on_splits": list(ACCEPTANCE_SPLITS) if accepted else [],
            "generation": int(contract.created_generation),
        }
        if accepted and not lineage_recorded:
            raise RuntimeError("accepted instrument mutation did not record problem-space lineage")
        self.decisions.append(decision)
        return decision

    def run_generation(self, *, generation: int) -> Dict[str, object]:
        diagnosis = self.diagnose()
        contract = self.propose_mutation_contract(diagnosis, generation=generation)
        decision = self.evaluate_contract(contract)
        return {"diagnosis": diagnosis.to_dict(), "decision": decision}

    def measure_residue_pressure(
        self,
        *,
        instruments: Mapping[str, Mapping[str, object]] | None = None,
        splits: Sequence[str] = ACCEPTANCE_SPLITS,
    ) -> Dict[str, object]:
        allowed = set(splits)
        snapshots = instruments or self._snapshot_instruments()
        by_instrument: Dict[str, float] = {}
        by_split: Dict[str, float] = {split: 0.0 for split in allowed}
        total = 0.0
        count = 0
        for entry in self.ledger._records:
            residue = entry["residue"]
            if not isinstance(residue, FailureResidue):
                continue
            if residue.split not in allowed or bool(entry.get("resolved")):
                continue
            adjusted = self._adjusted_severity(residue, snapshots)
            instrument_type, instrument_id = _parse_suspected_instrument(residue.suspected_instrument)
            key = _pressure_key(instrument_type, instrument_id)
            by_instrument[key] = by_instrument.get(key, 0.0) + adjusted
            by_split[residue.split] = by_split.get(residue.split, 0.0) + adjusted
            total += adjusted
            count += 1
        return {
            "total_pressure": float(total),
            "by_instrument": {key: float(value) for key, value in sorted(by_instrument.items())},
            "by_split": {key: float(value) for key, value in sorted(by_split.items())},
            "residue_count": count,
        }

    def evaluate_held_out_test_after_freeze(self) -> Dict[str, object]:
        return {
            "split": "test",
            "used_for_acceptance": False,
            "pressure": self.measure_residue_pressure(splits=("test",)),
        }

    def make_no_op_contract(self, *, target_instrument_type: str, target_instrument_id: str, generation: int) -> InstrumentMutationContract:
        return InstrumentMutationContract(
            contract_id=f"control_no_op_g{generation}_{len(self.contracts)}",
            target_instrument_type=target_instrument_type,
            target_instrument_id=target_instrument_id,
            reason="no-op control must not reduce residue pressure",
            expected_residue_reduction=0.0,
            mutation_payload={"action": "no_op"},
            created_generation=generation,
            uses_test_signal=False,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "observation_channels": [item.to_dict() for item in sorted(self.observation_channels.values(), key=lambda x: x.channel_id)],
            "evaluator_specs": [item.to_dict() for item in sorted(self.evaluator_specs.values(), key=lambda x: x.evaluator_id)],
            "experiment_units": [item.to_dict() for item in sorted(self.experiment_units.values(), key=lambda x: x.unit_id)],
            "operator_generators": [item.to_dict() for item in sorted(self.operator_generators.values(), key=lambda x: x.generator_id)],
            "failure_residue_ledger": self.ledger.to_dict(),
            "instrument_mutation_contracts": [contract.to_dict() for contract in self.contracts],
            "instrument_mutation_decisions": list(self.decisions),
            "problem_space_version_graph": self.version_graph.to_dict(),
            "rollback_count": int(self.rollback_count),
            "used_test_signal_for_acceptance": bool(self.used_test_signal_for_acceptance),
            "acceptance_pressure_measurements": list(self.acceptance_pressure_measurements),
        }

    def _resolve_target_id(self, target_type: str, target_id: str) -> str:
        registry = self._registry_for(target_type)
        if target_id in registry:
            return target_id
        if len(registry) == 1:
            return next(iter(registry))
        if target_type == "problem_space":
            return self.version_graph.current_version.version_id
        raise KeyError(f"unknown target instrument: {target_type}:{target_id}")

    def _registry_for(self, target_type: str) -> MutableMapping[str, object]:
        if target_type == "observation_channel":
            return self.observation_channels
        if target_type == "evaluator":
            return self.evaluator_specs
        if target_type == "experiment_unit":
            return self.experiment_units
        if target_type == "operator_generator":
            return self.operator_generators
        if target_type == "problem_space":
            return {}
        raise ValueError(f"unknown instrument type: {target_type}")

    def _proposal_payload(self, target_type: str, target_id: str) -> Dict[str, object]:
        residues = self.ledger.residues_for(_pressure_key(target_type, target_id))
        evidence = residues[0].evidence if residues else {}
        failure_type = residues[0].failure_type if residues else "generic_failure"
        if target_type == "observation_channel":
            blind_spot = str(evidence.get("blind_spot", "hidden_validation_gap"))
            signal = str(evidence.get("missing_signal", f"{blind_spot}_signal"))
            return {"action": "reduce_blind_spot", "blind_spot": blind_spot, "signal": signal}
        if target_type == "evaluator":
            penalty = _penalty_for_failure(failure_type, evidence)
            return {"action": "strengthen_penalty", "penalty": penalty, "delta": 0.5, "adversarial_check": penalty}
        if target_type == "experiment_unit":
            if "control" in failure_type or "control" in evidence:
                return {"action": "add_adversarial_control", "control": str(evidence.get("control", "no_op_control"))}
            return {"action": "add_hidden_validation_perturbation", "perturbation": str(evidence.get("perturbation", "hidden_validation_shift"))}
        if target_type == "operator_generator":
            primitive = str(evidence.get("primitive", "stability_probe"))
            if evidence.get("brittle", False):
                return {"action": "reduce_brittle_primitive", "primitive": primitive}
            return {"action": "expand_primitive_diversity", "primitive": primitive}
        return {"action": "no_op"}

    def _snapshot_instruments(self) -> Dict[str, Dict[str, object]]:
        return {
            "observation_channel": copy.deepcopy(self.observation_channels),
            "evaluator": copy.deepcopy(self.evaluator_specs),
            "experiment_unit": copy.deepcopy(self.experiment_units),
            "operator_generator": copy.deepcopy(self.operator_generators),
        }

    def _mutated_snapshot(
        self,
        contract: InstrumentMutationContract,
        snapshot: Mapping[str, Mapping[str, object]],
    ) -> Dict[str, Dict[str, object]]:
        mutated = {key: copy.deepcopy(dict(value)) for key, value in snapshot.items()}
        target_type = contract.target_instrument_type
        target_id = contract.target_instrument_id
        if target_type == "problem_space":
            return mutated
        registry = mutated[target_type]
        target = registry.get(target_id)
        if target is None:
            raise KeyError(f"unknown mutation target: {target_type}:{target_id}")
        if not hasattr(target, "mutate"):
            raise TypeError(f"target instrument does not support mutation: {target_type}:{target_id}")
        registry[target_id] = target.mutate(contract.mutation_payload)  # type: ignore[union-attr]
        return mutated

    def _commit_snapshot(self, snapshot: Mapping[str, Mapping[str, object]]) -> None:
        self.observation_channels = copy.deepcopy(snapshot["observation_channel"])  # type: ignore[assignment]
        self.evaluator_specs = copy.deepcopy(snapshot["evaluator"])  # type: ignore[assignment]
        self.experiment_units = copy.deepcopy(snapshot["experiment_unit"])  # type: ignore[assignment]
        self.operator_generators = copy.deepcopy(snapshot["operator_generator"])  # type: ignore[assignment]

    def _adjusted_severity(self, residue: FailureResidue, snapshots: Mapping[str, Mapping[str, object]]) -> float:
        instrument_type, instrument_id = _parse_suspected_instrument(residue.suspected_instrument)
        if instrument_type == "observation_channel":
            channel = self._lookup_snapshot(snapshots, "observation_channel", instrument_id)
            if isinstance(channel, ObservationChannel):
                blind_spot = str(residue.evidence.get("blind_spot", ""))
                missing_signal = str(residue.evidence.get("missing_signal", ""))
                if blind_spot and blind_spot not in channel.blind_spots:
                    return float(residue.severity * 0.35)
                if missing_signal and missing_signal in channel.signal_names:
                    return float(residue.severity * 0.50)
        if instrument_type == "evaluator":
            evaluator = self._lookup_snapshot(snapshots, "evaluator", instrument_id)
            if isinstance(evaluator, EvaluatorSpec):
                penalty = _penalty_for_failure(residue.failure_type, residue.evidence)
                strength = evaluator.scoring_terms.get(penalty, 0.0)
                if penalty in evaluator.adversarial_checks:
                    strength += 0.25
                return float(residue.severity / (1.0 + max(0.0, strength)))
        if instrument_type == "experiment_unit":
            unit = self._lookup_snapshot(snapshots, "experiment_unit", instrument_id)
            if isinstance(unit, ExperimentUnit):
                perturbations = set(unit.perturbation_policy.get("hidden_validation_perturbations", []))
                controls = set(unit.perturbation_policy.get("adversarial_controls", []))
                expected_perturbation = str(residue.evidence.get("perturbation", ""))
                expected_control = str(residue.evidence.get("control", ""))
                if expected_perturbation and expected_perturbation in perturbations:
                    return float(residue.severity * 0.45)
                if expected_control and expected_control in controls:
                    return float(residue.severity * 0.45)
        if instrument_type == "operator_generator":
            generator = self._lookup_snapshot(snapshots, "operator_generator", instrument_id)
            if isinstance(generator, OperatorGeneratorSpec):
                primitive = str(residue.evidence.get("primitive", ""))
                brittle = set(generator.mutation_policy.get("brittle_primitives", []))
                if primitive and primitive not in brittle and primitive in generator.mutation_policy.get("avoid_primitives", []):
                    return float(residue.severity * 0.40)
                diversity = len(set(generator.primitive_search_space))
                if diversity >= int(residue.evidence.get("min_diversity", 999)):
                    return float(residue.severity * 0.55)
        return float(residue.severity)

    @staticmethod
    def _lookup_snapshot(snapshots: Mapping[str, Mapping[str, object]], instrument_type: str, instrument_id: str | None) -> object | None:
        registry = snapshots.get(instrument_type, {})
        if instrument_id is not None:
            return registry.get(instrument_id)
        if len(registry) == 1:
            return next(iter(registry.values()))
        return None
