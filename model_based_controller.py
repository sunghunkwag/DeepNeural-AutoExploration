"""Model-based update controller for the upgraded closed-loop scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cognitive_core import EpisodicMemory, MemoryRecord, WorldModel


@dataclass(frozen=True)
class UpdateAction:
    name: str
    inner_lr: float
    inner_steps: int = 1
    memory_k: int = 1
    first_order: bool = True
    planner_horizon: int = 1
    exploration_noise_scale: float = 0.0

    def vector(self) -> torch.Tensor:
        return torch.tensor([
            float(self.inner_lr),
            float(self.inner_steps),
            float(self.memory_k),
            1.0 if self.first_order else 0.0,
            float(self.planner_horizon),
            float(self.exploration_noise_scale),
        ])


@dataclass
class ControllerDecision:
    selected: UpdateAction
    predicted_improvement: float
    scores: Dict[str, float]
    retrieved: List[Tuple[MemoryRecord, float]] = field(default_factory=list)


class ModelBasedController:
    """Ranks candidate adaptation/update actions by calling a learned WorldModel."""

    def __init__(self, world_model: WorldModel, candidates: Sequence[UpdateAction], latent_dim: Optional[int] = None):
        if not candidates:
            raise ValueError("candidates cannot be empty")
        self.world_model = world_model
        self.candidates = list(candidates)
        self.latent_dim = latent_dim or world_model.latent_dim
        self.prediction_errors: List[Dict[str, float | str]] = []
        self.world_model_call_count = 0

    def _fit(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        flat = tensor.detach().float().flatten()
        if flat.numel() >= dim:
            return flat[:dim]
        return torch.cat([flat, torch.zeros(dim - flat.numel())])

    def summarize_memory(self, retrieved: Sequence[Tuple[MemoryRecord, float]]) -> torch.Tensor:
        if not retrieved:
            return torch.zeros(self.latent_dim)
        weights = torch.tensor([max(0.0, score) for _, score in retrieved], dtype=torch.float32)
        if float(weights.sum()) <= 1e-8:
            weights = torch.ones(len(retrieved))
        embeddings = torch.stack([self._fit(record.task_embedding, self.latent_dim) for record, _ in retrieved])
        rewards = torch.tensor([record.reward for record, _ in retrieved], dtype=torch.float32).reshape(-1, 1)
        summary = (embeddings * weights.reshape(-1, 1)).sum(dim=0) / weights.sum().clamp_min(1e-8)
        summary[0] = summary[0] + 0.1 * float(rewards.mean())
        return summary

    def latent_state(self, z_task: torch.Tensor, memory_summary: torch.Tensor) -> torch.Tensor:
        z = self._fit(z_task, self.latent_dim)
        m = self._fit(memory_summary, self.latent_dim)
        return (z + 0.5 * m).reshape(1, -1)

    def score_action(self, z_task: torch.Tensor, memory_summary: torch.Tensor, action: UpdateAction) -> float:
        state = self.latent_state(z_task, memory_summary)
        action_vec = self._fit(action.vector(), self.world_model.action_dim).reshape(1, -1)
        with torch.no_grad():
            self.world_model_call_count += 1
            predicted_next = self.world_model.forward(state, action_vec)
        # first latent dimension is interpreted as expected validation improvement.
        return float(predicted_next.reshape(-1)[0])

    def select_action(self, z_task: torch.Tensor, memory: Optional[EpisodicMemory] = None, memory_k: int = 1) -> ControllerDecision:
        retrieved = memory.retrieve(z_task, k=memory_k) if memory is not None else []
        summary = self.summarize_memory(retrieved)
        scores = {candidate.name: self.score_action(z_task, summary, candidate) for candidate in self.candidates}
        selected = max(self.candidates, key=lambda c: scores[c.name])
        return ControllerDecision(selected, scores[selected.name], scores, retrieved)

    def log_actual_outcome(self, decision: ControllerDecision, actual_improvement: float) -> Dict[str, float | str]:
        error = abs(float(decision.predicted_improvement) - float(actual_improvement))
        entry = {
            "candidate": decision.selected.name,
            "predicted_improvement": float(decision.predicted_improvement),
            "actual_improvement": float(actual_improvement),
            "prediction_error": float(error),
        }
        self.prediction_errors.append(entry)
        return entry

    def train_on_transition_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], steps: int = 50, lr: float = 0.02) -> Tuple[float, float]:
        opt = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        initial = float(self.world_model.prediction_loss(*batch).detach())
        for _ in range(steps):
            self.world_model.train_step(batch, opt)
        final = float(self.world_model.prediction_loss(*batch).detach())
        return initial, final


def default_update_candidates(base_lr: float = 0.01) -> List[UpdateAction]:
    return [
        UpdateAction("conservative_lr", base_lr * 0.5, inner_steps=1, memory_k=1, first_order=True, planner_horizon=1),
        UpdateAction("base_lr", base_lr, inner_steps=1, memory_k=2, first_order=True, planner_horizon=2),
        UpdateAction("aggressive_lr", base_lr * 1.5, inner_steps=2, memory_k=2, first_order=True, planner_horizon=2),
    ]


def memory_conditioned_inner_lr(base_lr: float, retrieved: Sequence[Tuple[MemoryRecord, float]], min_lr: float = 1e-4, max_lr: float = 0.2) -> float:
    """Convert retrieved rewards/similarities into an adaptation learning-rate prior."""
    if not retrieved:
        return float(base_lr)
    weighted = 0.0
    denom = 0.0
    for record, similarity in retrieved:
        if record.split == "test":
            continue
        weight = max(0.0, float(similarity))
        weighted += weight * float(record.reward)
        denom += weight
    if denom <= 1e-8:
        return float(base_lr)
    reward_signal = weighted / denom
    scale = 1.0 + max(-0.5, min(0.5, reward_signal))
    return float(max(min_lr, min(max_lr, base_lr * scale)))
