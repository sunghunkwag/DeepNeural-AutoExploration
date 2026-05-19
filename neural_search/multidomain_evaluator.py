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
    selection_score_components: Dict[str, float] = field(default_factory=dict)
    parent_selection_score: float = 0.0
    candidate_selection_score: float = 0.0
    task_family_regressions: Dict[str, float] = field(default_factory=dict)

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
            "selection_score_components": dict(self.selection_score_components),
            "parent_selection_score": self.parent_selection_score,
            "candidate_selection_score": self.candidate_selection_score,
            "task_family_regressions": dict(self.task_family_regressions),
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
        parent_representation_probe: object | None = None,
        candidate_representation_probe: object | None = None,
        parent_gradient_probe: object | None = None,
        candidate_gradient_probe: object | None = None,
        rollback_risk: float = 0.0,
        module_contribution_score: float = 0.0,
        objective_config: object | None = None,
    ) -> CandidateSelectionDecision:
        validation_improvement = parent_eval.validation_loss - candidate_eval.validation_loss
        hidden_improvement = parent_eval.hidden_validation_loss - candidate_eval.hidden_validation_loss
        transfer_delta = candidate_eval.cross_domain_transfer_score - parent_eval.cross_domain_transfer_score
        task_regressions = _task_family_regressions(parent_eval, candidate_eval)
        regression_penalty = sum(task_regressions.values()) / max(1, len(candidate_eval.hidden_validation_loss_by_family))
        parent_components = selection_score_components(
            parent,
            parent_eval,
            representation_probe=parent_representation_probe,
            gradient_probe=parent_gradient_probe,
            rollback_risk=0.0,
            task_family_regression_penalty=0.0,
            module_contribution_score=module_contribution_score,
            objective_config=objective_config,
        )
        candidate_components = selection_score_components(
            candidate,
            candidate_eval,
            representation_probe=candidate_representation_probe,
            gradient_probe=candidate_gradient_probe,
            rollback_risk=rollback_risk,
            task_family_regression_penalty=regression_penalty,
            module_contribution_score=module_contribution_score,
            objective_config=objective_config,
        )
        parent_score = parent_components["selection_score"]
        candidate_score = candidate_components["selection_score"]
        score_delta = candidate_score - parent_score
        unacceptable_hidden_regression = hidden_improvement < -0.02
        unacceptable_transfer_regression = transfer_delta < -0.05
        too_many_family_regressions = len(task_regressions) > max(1, len(candidate_eval.hidden_validation_loss_by_family) // 2)
        accepted = bool(
            score_delta > improvement_threshold
            and (validation_improvement > 0.0 or hidden_improvement > 0.0)
            and not unacceptable_hidden_regression
            and not unacceptable_transfer_regression
            and not too_many_family_regressions
        )
        if accepted:
            reason = "accepted"
        elif unacceptable_hidden_regression:
            reason = "hidden_validation_regression"
        elif unacceptable_transfer_regression:
            reason = "transfer_regression"
        elif too_many_family_regressions:
            reason = "task_family_regression"
        else:
            reason = "selection_score_not_improved"
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
            selection_score_components=candidate_components,
            parent_selection_score=float(parent_score),
            candidate_selection_score=float(candidate_score),
            task_family_regressions=task_regressions,
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


def architecture_complexity_penalty(genome: ArchitectureGenome) -> float:
    module_count = (
        genome.residual_blocks
        + genome.memory_gates
        + genome.causal_bottlenecks
        + genome.object_binding_adapters
        + genome.sequence_state_adapters
    )
    return float(min(1.0, genome.hidden_dim / 128.0 + 0.035 * module_count))


def selection_score_components(
    genome: ArchitectureGenome,
    evaluation: MultiDomainEvaluation,
    *,
    representation_probe: object | None = None,
    gradient_probe: object | None = None,
    rollback_risk: float = 0.0,
    task_family_regression_penalty: float = 0.0,
    module_contribution_score: float = 0.0,
    objective_config: object | None = None,
) -> Dict[str, float]:
    weights = _objective_weights(objective_config)
    representation_failure_penalty = _probe_failure_penalty(representation_probe, collapse_weight=True)
    gradient_failure_penalty = _probe_failure_penalty(gradient_probe, collapse_weight=False)
    generalization_gap_penalty = max(0.0, evaluation.generalization_gap)
    complexity_penalty = architecture_complexity_penalty(genome)
    validation_component = 1.0 / (1.0 + max(0.0, evaluation.validation_loss))
    hidden_component = 1.0 / (1.0 + max(0.0, evaluation.hidden_validation_loss))
    module_contribution_component = min(1.0, max(0.0, float(module_contribution_score)))
    score = (
        weights["validation_loss_weight"] * validation_component
        + weights["hidden_validation_loss_weight"] * hidden_component
        + weights["transfer_score_weight"] * evaluation.cross_domain_transfer_score
        + 0.05 * evaluation.adaptation_speed
        + weights["module_contribution_weight"] * module_contribution_component
        - weights["generalization_gap_penalty_weight"] * generalization_gap_penalty
        - weights["architecture_complexity_penalty_weight"] * complexity_penalty
        - weights["representation_failure_penalty_weight"] * representation_failure_penalty
        - weights["gradient_failure_penalty_weight"] * gradient_failure_penalty
        - weights["rollback_risk_weight"] * max(0.0, float(rollback_risk))
        - weights["task_family_regression_penalty_weight"] * max(0.0, float(task_family_regression_penalty))
    )
    return {
        "validation_loss": float(evaluation.validation_loss),
        "hidden_validation_loss": float(evaluation.hidden_validation_loss),
        "cross_domain_transfer_score": float(evaluation.cross_domain_transfer_score),
        "generalization_gap_penalty": float(generalization_gap_penalty),
        "architecture_complexity_penalty": float(complexity_penalty),
        "representation_failure_penalty": float(representation_failure_penalty),
        "gradient_failure_penalty": float(gradient_failure_penalty),
        "rollback_risk": float(rollback_risk),
        "task_family_regression_penalty": float(task_family_regression_penalty),
        "module_contribution_score": float(module_contribution_component),
        "selection_score": float(score),
        **{f"objective_{key}": float(value) for key, value in weights.items()},
    }


def _probe_failure_penalty(probe: object | None, *, collapse_weight: bool) -> float:
    if probe is None:
        return 0.0
    modes = list(getattr(probe, "failure_modes", []) or [])
    penalty = 0.08 * len(modes)
    if collapse_weight:
        penalty += 0.20 * max(0.0, float(getattr(probe, "representation_collapse_score", 0.0)))
        penalty += 0.10 * max(0.0, float(getattr(probe, "validation_hidden_mismatch", 0.0)))
    else:
        penalty += 0.15 if float(getattr(probe, "total_gradient_norm", 1.0)) < 1e-6 else 0.0
    return float(min(1.0, penalty))


def _task_family_regressions(
    parent_eval: MultiDomainEvaluation,
    candidate_eval: MultiDomainEvaluation,
) -> Dict[str, float]:
    regressions: Dict[str, float] = {}
    for family, parent_loss in parent_eval.hidden_validation_loss_by_family.items():
        candidate_loss = candidate_eval.hidden_validation_loss_by_family.get(family)
        if candidate_loss is None:
            continue
        regression = float(candidate_loss - parent_loss)
        if regression > 1e-5:
            regressions[family] = regression
    return regressions


def _objective_weights(config: object | None) -> Dict[str, float]:
    defaults = {
        "validation_loss_weight": 0.30,
        "hidden_validation_loss_weight": 0.35,
        "transfer_score_weight": 0.12,
        "generalization_gap_penalty_weight": 0.08,
        "architecture_complexity_penalty_weight": 0.06,
        "representation_failure_penalty_weight": 0.08,
        "gradient_failure_penalty_weight": 0.08,
        "rollback_risk_weight": 0.08,
        "task_family_regression_penalty_weight": 0.15,
        "module_contribution_weight": 0.05,
    }
    if config is None:
        return defaults
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif isinstance(config, Mapping):
        raw = config
    else:
        raw = {}
    if not isinstance(raw, Mapping):
        return defaults
    out = dict(defaults)
    for key in defaults:
        value = raw.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(max(0.0, min(1.0, value)))
    return out
