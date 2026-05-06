"""
AGI-oriented closed-loop cognitive scaffold.

This module adds a minimal but real loop around the existing RSI-DNAX model:
task inference, functional MAML adaptation, episodic memory, latent world-model
learning, short-horizon planning, intrinsic objectives, and validation-gated
self-improvement with rollback.  It is an experimental scaffold, not an AGI
claim.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maml_functional import functional_forward, maml_inner_loop
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer


@dataclass(frozen=True)
class MemoryRecord:
    """One non-query-leaking episode summary stored by EpisodicMemory."""

    task_id: str
    observation: torch.Tensor
    action: torch.Tensor
    reward: float
    loss: float
    task_embedding: torch.Tensor
    outcome: Dict[str, float] = field(default_factory=dict)
    split: str = "train"
    contains_query_targets: bool = False


class EpisodicMemory:
    """Bounded cosine-similarity memory for support/context summaries.

    The memory refuses records marked as containing query targets so that test
    labels cannot enter the adaptation/retrieval path by accident.
    """

    def __init__(self, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.records: List[MemoryRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: MemoryRecord) -> None:
        if record.contains_query_targets:
            raise ValueError("EpisodicMemory rejects records containing query targets")
        detached = MemoryRecord(
            task_id=record.task_id,
            observation=record.observation.detach().clone().float(),
            action=record.action.detach().clone().float(),
            reward=float(record.reward),
            loss=float(record.loss),
            task_embedding=record.task_embedding.detach().clone().float(),
            outcome=dict(record.outcome),
            split=record.split,
            contains_query_targets=False,
        )
        self.records.append(detached)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity :]

    def retrieve(self, query_embedding: torch.Tensor, k: int = 1) -> List[Tuple[MemoryRecord, float]]:
        if k <= 0 or not self.records:
            return []
        q = query_embedding.detach().float().flatten()
        scored: List[Tuple[MemoryRecord, float]] = []
        for record in self.records:
            e = record.task_embedding.float().flatten()
            if e.numel() != q.numel():
                raise ValueError("query embedding dimension does not match memory records")
            denom = (q.norm() * e.norm()).clamp_min(1e-8)
            score = float(torch.dot(q, e) / denom)
            scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


class TaskInferenceModule:
    """Infers compact task embeddings from support/context examples only."""

    def __init__(self, embedding_dim: int = 8):
        if embedding_dim < 8:
            raise ValueError("embedding_dim must be at least 8")
        self.embedding_dim = embedding_dim

    def infer(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        x = support_x.detach().float().reshape(support_x.shape[0], -1)
        y = support_y.detach().float().reshape(support_y.shape[0], -1)
        x_mean, x_std = x.mean(), x.std(unbiased=False)
        y_mean, y_std = y.mean(), y.std(unbiased=False)
        x_flat = x[:, 0]
        y_flat = y[:, 0]
        if x_flat.numel() > 1:
            xm = x_flat - x_flat.mean()
            ym = y_flat - y_flat.mean()
            corr = torch.dot(xm, ym) / ((xm.norm() * ym.norm()).clamp_min(1e-8))
            order = torch.argsort(x_flat)
            dy = torch.diff(y_flat[order])
            dx = torch.diff(x_flat[order]).abs().clamp_min(1e-6)
            slope_abs = (dy / dx).abs().mean() if dy.numel() else torch.tensor(0.0)
            roughness = torch.diff(dy).abs().mean() if dy.numel() > 1 else torch.tensor(0.0)
        else:
            corr = torch.tensor(0.0)
            slope_abs = torch.tensor(0.0)
            roughness = torch.tensor(0.0)
        features = torch.stack([x_mean, x_std, y_mean, y_std, corr, slope_abs, roughness, torch.tensor(float(x.shape[0]))])
        if self.embedding_dim == 8:
            return features
        pad = torch.zeros(self.embedding_dim - 8)
        return torch.cat([features, pad], dim=0)


class WorldModel(nn.Module):
    """Small latent transition model p(z_{t+1} | z_t, a_t)."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 32, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent_state.float(), action.float()], dim=-1))

    def predict_next(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(latent_state, action)

    def prediction_loss(self, latent_state: torch.Tensor, action: torch.Tensor, next_latent_state: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(latent_state, action), next_latent_state.float())

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
        self.train()
        latent, action, next_latent = batch
        optimizer.zero_grad()
        loss = self.prediction_loss(latent, action, next_latent)
        loss.backward()
        optimizer.step()
        return float(loss.detach())


def generate_linear_dynamics_batch(n: int = 64, latent_dim: int = 2, action_dim: int = 2, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Procedural learnable dynamics: z' = z + padded(action) + small drift."""

    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(n, latent_dim, generator=g)
    action = torch.randn(n, action_dim, generator=g) * 0.5
    projected = torch.zeros(n, latent_dim)
    cols = min(latent_dim, action_dim)
    projected[:, :cols] = action[:, :cols]
    drift = 0.1 * torch.sin(latent)
    next_latent = latent + projected + drift
    return latent, action, next_latent


class Planner:
    """Deterministic short-horizon latent search using a WorldModel."""

    def __init__(self, world_model: WorldModel, action_set: Sequence[torch.Tensor], horizon: int = 3):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not action_set:
            raise ValueError("action_set cannot be empty")
        self.world_model = world_model
        self.action_set = [a.detach().float().flatten() for a in action_set]
        self.horizon = horizon

    def plan(self, start_state: torch.Tensor, goal_state: torch.Tensor) -> Tuple[List[torch.Tensor], float]:
        start = start_state.detach().float().reshape(1, -1)
        goal = goal_state.detach().float().reshape(1, -1)
        best_sequence: Optional[Tuple[torch.Tensor, ...]] = None
        best_score = float("inf")
        with torch.no_grad():
            for sequence in itertools.product(self.action_set, repeat=self.horizon):
                state = start
                for action in sequence:
                    state = self.world_model.forward(state, action.reshape(1, -1))
                score = float(F.mse_loss(state, goal))
                if score < best_score - 1e-12:
                    best_score = score
                    best_sequence = sequence
        assert best_sequence is not None
        return [a.clone() for a in best_sequence], best_score


class IntrinsicObjectiveSystem:
    """Computes intrinsic signals without using final test labels."""

    def compute(
        self,
        *,
        uncertainty_before: float,
        uncertainty_after: float,
        prediction_error_before: float,
        prediction_error_after: float,
        novelty: float,
        previous_loss: Optional[float],
        current_loss: float,
    ) -> Dict[str, float]:
        uncertainty_reduction = uncertainty_before - uncertainty_after
        prediction_error_reduction = prediction_error_before - prediction_error_after
        improvement_rate = 0.0 if previous_loss is None else previous_loss - current_loss
        total = uncertainty_reduction + prediction_error_reduction + novelty + improvement_rate
        return {
            "uncertainty_reduction": float(uncertainty_reduction),
            "prediction_error_reduction": float(prediction_error_reduction),
            "novelty": float(novelty),
            "improvement_rate": float(improvement_rate),
            "intrinsic_total": float(total),
        }


@dataclass
class ImprovementCandidate:
    name: str
    updates: Dict[str, float]


class SelfImprovementController:
    """Validation-gated hyperparameter/operator proposal controller."""

    def __init__(self, state: Dict[str, float], improvement_threshold: float = 0.0, min_consistent_wins: int = 1):
        self.state = dict(state)
        self.improvement_threshold = improvement_threshold
        self.min_consistent_wins = min_consistent_wins
        self.log: List[Dict[str, object]] = []
        self.rollback_count = 0
        self.accepted_mutation_count = 0

    def snapshot(self) -> Dict[str, float]:
        return copy.deepcopy(self.state)

    def restore(self, snapshot: Dict[str, float]) -> None:
        self.state = copy.deepcopy(snapshot)
        self.rollback_count += 1

    def propose_candidates(self) -> List[ImprovementCandidate]:
        lr = self.state.get("inner_lr", 0.01)
        noise = self.state.get("exploration_noise_scale", 0.02)
        return [
            ImprovementCandidate("lower_inner_lr", {"inner_lr": lr * 0.5}),
            ImprovementCandidate("raise_inner_lr", {"inner_lr": lr * 1.5}),
            ImprovementCandidate("decay_exploration_noise", {"exploration_noise_scale": noise * 0.8}),
        ]

    def evaluate_and_update(
        self,
        candidate: ImprovementCandidate,
        validation_tasks: Optional[Sequence[object]],
        evaluator: Callable[[Dict[str, float], Sequence[object]], Sequence[float] | float],
    ) -> Dict[str, object]:
        if validation_tasks is None or len(validation_tasks) == 0:
            raise ValueError("validation_tasks are required before accepting a self-improvement candidate")
        before = self.snapshot()
        baseline_scores = evaluator(before, validation_tasks)
        if isinstance(baseline_scores, (float, int)):
            baseline_scores = [float(baseline_scores)]
        trial = copy.deepcopy(before)
        trial.update(candidate.updates)
        candidate_scores = evaluator(trial, validation_tasks)
        if isinstance(candidate_scores, (float, int)):
            candidate_scores = [float(candidate_scores)]
        if len(candidate_scores) != len(baseline_scores):
            raise ValueError("evaluator must return comparable score sequences")
        deltas = [float(b - c) for b, c in zip(baseline_scores, candidate_scores)]  # lower is better
        mean_improvement = float(np.mean(deltas))
        wins = sum(delta > self.improvement_threshold for delta in deltas)
        accepted = wins >= self.min_consistent_wins and mean_improvement > self.improvement_threshold
        if accepted:
            self.state = trial
            self.accepted_mutation_count += 1
        else:
            self.restore(before)
        entry = {
            "candidate": candidate.name,
            "updates": dict(candidate.updates),
            "baseline": [float(x) for x in baseline_scores],
            "candidate_scores": [float(x) for x in candidate_scores],
            "mean_improvement": mean_improvement,
            "accepted": accepted,
            "rollback_count": self.rollback_count,
            "rollback": not accepted,
            "random_seed": int(self.state.get("seed", 0)) if isinstance(self.state.get("seed", 0), (int, float)) else 0,
            "split": "validation",
        }
        self.log.append(entry)
        return entry


class CognitiveCore:
    """Coordinates the closed loop around DeepNeuralAutoExplorer and MAML."""

    def __init__(
        self,
        model: DeepNeuralAutoExplorer,
        config: DNAXConfig,
        world_model: WorldModel,
        memory: Optional[EpisodicMemory] = None,
        task_inference: Optional[TaskInferenceModule] = None,
    ):
        self.model = model
        self.config = config
        self.world_model = world_model
        self.memory = memory or EpisodicMemory(capacity=config.replay_buffer_size)
        self.task_inference = task_inference or TaskInferenceModule()
        self.intrinsic = IntrinsicObjectiveSystem()
        self.self_improvement = SelfImprovementController(
            {"inner_lr": config.inner_lr, "inner_steps": config.inner_steps, "exploration_noise_scale": config.exploration_noise_scale, "seed": config.seed or 0},
            improvement_threshold=0.0,
        )

    def adapt_and_evaluate(self, task, inner_lr: Optional[float] = None, inner_steps: Optional[int] = None) -> Dict[str, object]:
        params = dict(self.model.named_parameters())
        sx, sy = task.support_x, task.support_y
        qx, qy = task.query_x, task.query_y
        with torch.no_grad():
            before = F.mse_loss(self.model(qx), qy).item()
        adapted = maml_inner_loop(
            self.model,
            params,
            sx,
            sy,
            inner_lr if inner_lr is not None else self.config.inner_lr,
            inner_steps if inner_steps is not None else self.config.inner_steps,
            first_order=not self.config.use_second_order,
        )
        with torch.no_grad():
            pred = functional_forward(self.model, adapted, qx)
            after = F.mse_loss(pred, qy).item()
        embedding = self.task_inference.infer(sx, sy)
        retrieved = self.memory.retrieve(embedding, k=1)
        self.memory.add(
            MemoryRecord(
                task_id=task.task_id,
                observation=torch.cat([sx.flatten(), sy.flatten()])[: int(getattr(self.task_inference, "embedding_dim", getattr(self.task_inference, "latent_dim", 8)))],
                action=torch.tensor([inner_lr if inner_lr is not None else self.config.inner_lr]),
                reward=before - after,
                loss=after,
                task_embedding=embedding,
                outcome={"query_loss_before": before, "query_loss_after": after},
                split=task.split,
                contains_query_targets=False,
            )
        )
        return {
            "task_id": task.task_id,
            "embedding": embedding,
            "retrieved": retrieved,
            "adaptation_improvement": before - after,
            "query_loss_before": before,
            "query_loss_after": after,
        }


    def memory_conditioned_inner_lr(self, embedding: torch.Tensor, base_lr: Optional[float] = None, k: int = 1) -> Tuple[float, List[Tuple[MemoryRecord, float]]]:
        """Use retrieved non-test episodes to modulate adaptation learning rate."""
        lr = float(base_lr if base_lr is not None else self.config.inner_lr)
        retrieved = self.memory.retrieve(embedding, k=k)
        if not retrieved:
            return lr, retrieved
        weighted_reward = 0.0
        weight_sum = 0.0
        for record, score in retrieved:
            if record.split == "test":
                continue
            weight = max(0.0, float(score))
            weighted_reward += weight * float(record.reward)
            weight_sum += weight
        if weight_sum <= 1e-8:
            return lr, retrieved
        signal = max(-0.5, min(0.5, weighted_reward / weight_sum))
        return float(max(1e-4, min(0.2, lr * (1.0 + signal)))), retrieved

    def adapt_memory_conditioned(self, task, memory_k: int = 1, use_memory: bool = True) -> Dict[str, object]:
        """Adapt with a learning rate selected from support-only task latent and memory."""
        embedding = self.task_inference.infer(task.support_x, task.support_y)
        if use_memory:
            conditioned_lr, retrieved = self.memory_conditioned_inner_lr(embedding, self.config.inner_lr, k=memory_k)
        else:
            conditioned_lr, retrieved = self.config.inner_lr, []
        log = self.adapt_and_evaluate(task, inner_lr=conditioned_lr)
        log["conditioned_inner_lr"] = conditioned_lr
        log["retrieved_for_adaptation"] = retrieved
        return log

    def validation_evaluator(self, tasks: Sequence[object]) -> Callable[[Dict[str, float], Sequence[object]], List[float]]:
        def _eval(state: Dict[str, float], validation_tasks: Sequence[object]) -> List[float]:
            return [float(self.adapt_and_evaluate(t, inner_lr=state.get("inner_lr", self.config.inner_lr))["query_loss_after"]) for t in validation_tasks]

        return _eval
