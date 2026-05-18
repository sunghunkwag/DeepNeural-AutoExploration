"""Validation-gated architecture evaluation and audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from .architecture_genome import ArchitectureGenome
from .neural_candidate_sandbox import NeuralCandidateSandbox, NeuralSandboxTask
from .weight_inheritance import inherit_shape_compatible_weights


@dataclass
class ArchitectureDecision:
    candidate_id: str
    parent_id: str
    mutation_method: str
    accepted: bool
    rollback: bool
    validation_metrics: Dict[str, float]
    hidden_validation_metrics: Dict[str, float] = field(default_factory=dict)
    parent_validation_metrics: Dict[str, float] = field(default_factory=dict)
    used_heldout_labels: bool = False
    split_used: str = "validation"
    random_seed: int = 0
    config_hash: str = ""
    weight_inheritance: Dict[str, object] = field(default_factory=dict)
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "mutation_method": self.mutation_method,
            "accepted": self.accepted,
            "rollback": self.rollback,
            "validation_metrics": dict(self.validation_metrics),
            "hidden_validation_metrics": dict(self.hidden_validation_metrics),
            "parent_validation_metrics": dict(self.parent_validation_metrics),
            "used_heldout_labels": self.used_heldout_labels,
            "split_used": self.split_used,
            "random_seed": self.random_seed,
            "config_hash": self.config_hash,
            "weight_inheritance": dict(self.weight_inheritance),
            "rejection_reason": self.rejection_reason,
        }


class ArchitectureEvaluator:
    def __init__(
        self,
        *,
        seed: int = 0,
        sandbox: Optional[NeuralCandidateSandbox] = None,
        improvement_threshold: float = 1e-4,
    ):
        self.seed = int(seed)
        self.sandbox = sandbox or NeuralCandidateSandbox(seed=seed)
        self.improvement_threshold = float(improvement_threshold)
        self.audit_log: list[ArchitectureDecision] = []

    def evaluate_candidate(
        self,
        parent: ArchitectureGenome,
        candidate: ArchitectureGenome,
        tasks: Optional[Sequence[NeuralSandboxTask]] = None,
    ) -> ArchitectureDecision:
        active_tasks = list(tasks) if tasks is not None else self.sandbox.procedural_tasks(parent)
        parent_result = self.sandbox.evaluate(parent, active_tasks)
        if not parent_result.ok or parent_result.model is None:
            decision = ArchitectureDecision(
                candidate_id=candidate.genome_id,
                parent_id=parent.genome_id,
                mutation_method=candidate.mutation_method,
                accepted=False,
                rollback=True,
                validation_metrics={},
                parent_validation_metrics=parent_result.validation_metrics,
                used_heldout_labels=parent_result.used_heldout_labels,
                random_seed=candidate.seed,
                config_hash=candidate.config_hash,
                rejection_reason=f"parent_evaluation_failed:{parent_result.error}",
            )
            self.audit_log.append(decision)
            return decision

        child_model = candidate.build_module()
        inheritance = inherit_shape_compatible_weights(parent_result.model, child_model)
        candidate_result = self.sandbox.evaluate(candidate, active_tasks, initial_model=child_model)
        parent_loss = float(parent_result.validation_metrics.get("loss", float("inf")))
        candidate_loss = float(candidate_result.validation_metrics.get("loss", float("inf")))
        accepted = bool(
            candidate_result.ok
            and not candidate_result.used_heldout_labels
            and candidate_loss < parent_loss - self.improvement_threshold
        )
        reason = "accepted" if accepted else "validation_loss_not_improved"
        if not candidate_result.ok:
            reason = f"candidate_evaluation_failed:{candidate_result.error}"
        decision = ArchitectureDecision(
            candidate_id=candidate.genome_id,
            parent_id=parent.genome_id,
            mutation_method=candidate.mutation_method,
            accepted=accepted,
            rollback=not accepted,
            validation_metrics=candidate_result.validation_metrics,
            hidden_validation_metrics=candidate_result.hidden_validation_metrics,
            parent_validation_metrics=parent_result.validation_metrics,
            used_heldout_labels=candidate_result.used_heldout_labels,
            split_used=candidate_result.split_used,
            random_seed=candidate.seed,
            config_hash=candidate.config_hash,
            weight_inheritance=inheritance.to_dict(),
            rejection_reason=reason,
        )
        self.audit_log.append(decision)
        return decision
