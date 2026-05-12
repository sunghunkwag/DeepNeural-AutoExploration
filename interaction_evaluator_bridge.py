"""Bridge from interaction residues to evaluator-evolution decisions."""

from __future__ import annotations

from typing import Dict, List, Sequence

from evaluator_evolution import EvaluatorCandidate, EvaluatorGenome, score_decision
from interaction_residue_layer import InteractionResidue


class InteractionEvaluatorBridge:
    """Exports interaction residue evidence for the bounded evaluator path."""

    def __init__(self, *, generation: int = 0) -> None:
        self.generation = int(generation)

    def to_decision_records(
        self,
        residues: Sequence[InteractionResidue],
        *,
        split: str = "validation",
        generation: int | None = None,
        control_policy_scores: Dict[str, float] | None = None,
    ) -> List[Dict[str, object]]:
        if split == "test":
            raise ValueError("interaction evaluator bridge cannot use test split data")
        controls = dict(control_policy_scores or {})
        controls.setdefault("no_op", 0.0)
        controls.setdefault("random", 0.0)
        records: List[Dict[str, object]] = []
        for residue in residues:
            decision = residue.to_evaluator_decision(self.generation if generation is None else int(generation))
            decision["used_test_results"] = False
            decision["candidate_metadata"] = {
                "source": "interaction_operating_scaffold",
                "split": split,
                "interaction_failure_type": residue.failure_type,
                "timestamp_ms": residue.timestamp_ms,
                "candidate_family": "interaction_policy",
            }
            decision["control_policy_scores"] = {
                "no_op": float(controls["no_op"]),
                "random": float(controls["random"]),
            }
            decision.setdefault("no_op_control_delta", float(controls["no_op"]))
            decision.setdefault("random_control_delta", float(controls["random"]))
            self.validate_decision_record(decision)
            records.append(decision)
        return records

    @staticmethod
    def validate_decision_record(decision: Dict[str, object]) -> None:
        if decision.get("used_test_results") is not False:
            raise ValueError("interaction decisions must set used_test_results = False")
        metadata = decision.get("candidate_metadata", {})
        if isinstance(metadata, dict) and metadata.get("split") == "test":
            raise ValueError("interaction decisions cannot be exported from test split data")
        if "no_op_control_delta" not in decision:
            raise ValueError("interaction decision missing no-op control")
        if "random_control_delta" not in decision:
            raise ValueError("interaction decision missing random control")
        controls = decision.get("control_policy_scores", {})
        if not isinstance(controls, dict) or "no_op" not in controls or "random" not in controls:
            raise ValueError("interaction decision missing control policy scores")

    def score_records(
        self,
        genome: EvaluatorGenome,
        decisions: Sequence[Dict[str, object]],
        *,
        failure_rule_count: int = 1,
    ) -> List[float]:
        for decision in decisions:
            self.validate_decision_record(decision)
        return [score_decision(genome.active, decision, failure_rule_count=failure_rule_count) for decision in decisions]

    def validate_candidate(
        self,
        genome: EvaluatorGenome,
        decisions: Sequence[Dict[str, object]],
        *,
        candidate: EvaluatorCandidate | None = None,
        generation: int | None = None,
        failure_rule_count: int = 1,
    ) -> Dict[str, object]:
        if not decisions:
            raise ValueError("candidate validation requires at least one interaction decision")
        for decision in decisions:
            self.validate_decision_record(decision)
        selected = candidate or genome.propose(
            self.generation if generation is None else int(generation),
            failure_rule_count=failure_rule_count,
        )[0]
        return genome.validate_candidate(
            selected,
            decisions,
            generation=self.generation if generation is None else int(generation),
            failure_rule_count=failure_rule_count,
        )
