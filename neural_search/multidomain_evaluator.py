"""Multi-domain training and validation for architecture candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .architecture_genome import ArchitectureGenome
from .module_library import set_deterministic_seed
from .task_families import MultiDomainBatch, MultiDomainTaskFamily, SPLITS


@dataclass
class MultiDomainEvaluation:
    genome_id: str
    train_loss_by_family: Dict[str, float]
    validation_loss_by_family: Dict[str, float]
    hidden_validation_loss_by_family: Dict[str, float]
    heldout_test_loss_by_family: Dict[str, float] = field(default_factory=dict)
    cross_domain_transfer_score: float = 0.0
    generalization_gap: float = 0.0
    adaptation_speed: float = 0.0
    train_steps: int = 0
    initial_train_loss: float = 0.0
    final_train_loss: float = 0.0
    heldout_test_evaluated_after_freeze: bool = False
    model: Optional[torch.nn.Module] = field(default=None, repr=False, compare=False)

    @property
    def validation_loss(self) -> float:
        return _mean_dict(self.validation_loss_by_family)

    @property
    def hidden_validation_loss(self) -> float:
        return _mean_dict(self.hidden_validation_loss_by_family)

    @property
    def train_loss(self) -> float:
        return _mean_dict(self.train_loss_by_family)

    @property
    def heldout_test_loss(self) -> float:
        return _mean_dict(self.heldout_test_loss_by_family)

    def selection_score(self) -> float:
        return 0.6 * self.validation_loss + 0.4 * self.hidden_validation_loss

    def to_dict(self, *, include_model: bool = False) -> Dict[str, object]:
        data = {
            "genome_id": self.genome_id,
            "train_loss_by_family": dict(self.train_loss_by_family),
            "validation_loss_by_family": dict(self.validation_loss_by_family),
            "hidden_validation_loss_by_family": dict(self.hidden_validation_loss_by_family),
            "heldout_test_loss_by_family": dict(self.heldout_test_loss_by_family),
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "hidden_validation_loss": self.hidden_validation_loss,
            "heldout_test_loss": self.heldout_test_loss,
            "cross_domain_transfer_score": self.cross_domain_transfer_score,
            "generalization_gap": self.generalization_gap,
            "adaptation_speed": self.adaptation_speed,
            "train_steps": self.train_steps,
            "initial_train_loss": self.initial_train_loss,
            "final_train_loss": self.final_train_loss,
            "heldout_test_evaluated_after_freeze": self.heldout_test_evaluated_after_freeze,
        }
        if include_model:
            data["model"] = self.model
        return data


@dataclass(frozen=True)
class CandidateSelectionDecision:
    candidate_id: str
    parent_id: str
    accepted: bool
    rollback: bool
    validation_improvement: float
    hidden_validation_improvement: float
    selection_score_delta: float
    mutation_method: str
    random_seed: int
    config_hash: str
    validation_metrics: Dict[str, float]
    hidden_validation_metrics: Dict[str, float]
    rejection_reason: str
    split_used: str = "validation"
    used_heldout_labels: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "accepted": self.accepted,
            "rollback": self.rollback,
            "validation_improvement": self.validation_improvement,
            "hidden_validation_improvement": self.hidden_validation_improvement,
            "selection_score_delta": self.selection_score_delta,
            "mutation_method": self.mutation_method,
            "random_seed": self.random_seed,
            "config_hash": self.config_hash,
            "validation_metrics": dict(self.validation_metrics),
            "hidden_validation_metrics": dict(self.hidden_validation_metrics),
            "rejection_reason": self.rejection_reason,
            "split_used": self.split_used,
            "used_heldout_labels": self.used_heldout_labels,
        }


class MultiDomainEvaluator:
    def __init__(self, *, seed: int = 0, train_steps: int = 20, lr: float = 0.015):
        self.seed = int(seed)
        self.train_steps = max(1, int(train_steps))
        self.lr = float(lr)

    def train_and_validate(
        self,
        genome: ArchitectureGenome,
        task_families: Sequence[MultiDomainTaskFamily],
        *,
        initial_model: Optional[torch.nn.Module] = None,
    ) -> MultiDomainEvaluation:
        set_deterministic_seed(self.seed + genome.seed)
        model = initial_model if initial_model is not None else genome.build_module()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        train_batches = [family.splits["train"] for family in task_families]
        initial_train_loss = self._mean_loss(model, train_batches)
        first_step_loss = initial_train_loss
        for step in range(self.train_steps):
            for batch in train_batches:
                optimizer.zero_grad()
                loss = F.mse_loss(model(batch.x), batch.y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite multi-domain train loss")
                loss.backward()
                optimizer.step()
            if step == 0:
                first_step_loss = self._mean_loss(model, train_batches)
        model.eval()
        final_train_loss = self._mean_loss(model, train_batches)
        train_by_family = self._loss_by_family(model, task_families, "train")
        val_by_family = self._loss_by_family(model, task_families, "validation")
        hidden_by_family = self._loss_by_family(model, task_families, "hidden_validation")
        train_mean = _mean_dict(train_by_family)
        hidden_mean = _mean_dict(hidden_by_family)
        adaptation_speed = max(0.0, initial_train_loss - first_step_loss) / max(1, self.train_steps)
        return MultiDomainEvaluation(
            genome_id=genome.genome_id,
            train_loss_by_family=train_by_family,
            validation_loss_by_family=val_by_family,
            hidden_validation_loss_by_family=hidden_by_family,
            cross_domain_transfer_score=1.0 / (1.0 + hidden_mean),
            generalization_gap=hidden_mean - train_mean,
            adaptation_speed=adaptation_speed,
            train_steps=self.train_steps,
            initial_train_loss=initial_train_loss,
            final_train_loss=final_train_loss,
            model=model,
        )

    def evaluate_heldout_after_freeze(
        self,
        model: torch.nn.Module,
        genome: ArchitectureGenome,
        task_families: Sequence[MultiDomainTaskFamily],
    ) -> MultiDomainEvaluation:
        model.eval()
        train_by_family = self._loss_by_family(model, task_families, "train")
        val_by_family = self._loss_by_family(model, task_families, "validation")
        hidden_by_family = self._loss_by_family(model, task_families, "hidden_validation")
        heldout_by_family = self._loss_by_family(model, task_families, "heldout_test")
        train_mean = _mean_dict(train_by_family)
        hidden_mean = _mean_dict(hidden_by_family)
        heldout_mean = _mean_dict(heldout_by_family)
        return MultiDomainEvaluation(
            genome_id=genome.genome_id,
            train_loss_by_family=train_by_family,
            validation_loss_by_family=val_by_family,
            hidden_validation_loss_by_family=hidden_by_family,
            heldout_test_loss_by_family=heldout_by_family,
            cross_domain_transfer_score=1.0 / (1.0 + heldout_mean),
            generalization_gap=hidden_mean - train_mean,
            adaptation_speed=0.0,
            heldout_test_evaluated_after_freeze=True,
            model=model,
        )

    def selection_decision(
        self,
        parent: ArchitectureGenome,
        candidate: ArchitectureGenome,
        parent_eval: MultiDomainEvaluation,
        candidate_eval: MultiDomainEvaluation,
        *,
        improvement_threshold: float = 1e-5,
    ) -> CandidateSelectionDecision:
        validation_improvement = parent_eval.validation_loss - candidate_eval.validation_loss
        hidden_improvement = parent_eval.hidden_validation_loss - candidate_eval.hidden_validation_loss
        score_delta = parent_eval.selection_score() - candidate_eval.selection_score()
        accepted = bool(
            score_delta > improvement_threshold
            and (validation_improvement > 0.0 or hidden_improvement > 0.0)
        )
        reason = "accepted" if accepted else "validation_hidden_score_not_improved"
        return CandidateSelectionDecision(
            candidate_id=candidate.genome_id,
            parent_id=parent.genome_id,
            accepted=accepted,
            rollback=not accepted,
            validation_improvement=float(validation_improvement),
            hidden_validation_improvement=float(hidden_improvement),
            selection_score_delta=float(score_delta),
            mutation_method=candidate.mutation_method,
            random_seed=candidate.seed,
            config_hash=candidate.config_hash,
            validation_metrics={"loss": candidate_eval.validation_loss, **candidate_eval.validation_loss_by_family},
            hidden_validation_metrics={"loss": candidate_eval.hidden_validation_loss, **candidate_eval.hidden_validation_loss_by_family},
            rejection_reason=reason,
        )

    def _loss_by_family(
        self,
        model: torch.nn.Module,
        task_families: Sequence[MultiDomainTaskFamily],
        split: str,
    ) -> Dict[str, float]:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        out: Dict[str, float] = {}
        with torch.no_grad():
            for family in task_families:
                batch = family.splits[split]
                out[family.family] = float(F.mse_loss(model(batch.x), batch.y))
        return out

    def _mean_loss(self, model: torch.nn.Module, batches: Iterable[MultiDomainBatch]) -> float:
        values: List[float] = []
        with torch.no_grad():
            for batch in batches:
                values.append(float(F.mse_loss(model(batch.x), batch.y)))
        return float(mean(values)) if values else 0.0


def _mean_dict(values: Mapping[str, float] | Dict[str, float]) -> float:
    vals = [float(value) for value in values.values()]
    return float(mean(vals)) if vals else 0.0
