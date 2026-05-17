"""Learned support-set task/context encoder for procedural regression tasks.

The encoder is intentionally small and CPU-runnable.  It uses a DeepSets-style
pair encoder over support pairs ``[x_i, y_i]`` followed by permutation-invariant
mean/max aggregation.  Query targets are never part of the encoder API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_core import TaskInferenceModule
from task_suite import ProceduralTaskSuite, SyntheticTask


class LearnedTaskEncoder(nn.Module):
    """Permutation-robust DeepSets context encoder.

    ``encode(support_x, support_y)`` returns a single latent vector with shape
    ``(latent_dim,)`` and depends only on support examples.
    """

    def __init__(self, x_dim: int = 1, y_dim: int = 1, hidden_dim: int = 32, latent_dim: int = 8, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.pair_encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        return self.encode(support_x, support_y)

    def encode(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        x = support_x.float().reshape(support_x.shape[0], -1)
        y = support_y.float().reshape(support_y.shape[0], -1)
        pairs = torch.cat([x, y], dim=-1)
        h = self.pair_encoder(pairs)
        context = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=-1)
        return self.projector(context)

    def infer(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        """Compatibility with ``TaskInferenceModule.infer``."""
        with torch.no_grad():
            return self.encode(support_x, support_y).detach()


class TaskConditionedRegressor(nn.Module):
    """Small decoder used to train/evaluate learned task embeddings."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 32, x_dim: int = 1, y_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, query_x: torch.Tensor, z_task: torch.Tensor) -> torch.Tensor:
        qx = query_x.float().reshape(query_x.shape[0], -1)
        z = z_task.float().reshape(1, -1).expand(qx.shape[0], -1)
        return self.net(torch.cat([qx, z], dim=-1))


@dataclass
class EncoderTrainingResult:
    train_losses: List[float]
    validation_metrics: Dict[str, float]


def task_reconstruction_loss(encoder: LearnedTaskEncoder, decoder: TaskConditionedRegressor, tasks: Sequence[SyntheticTask]) -> torch.Tensor:
    losses = []
    for task in tasks:
        z = encoder.encode(task.support_x, task.support_y)
        pred = decoder(task.query_x, z)
        losses.append(F.mse_loss(pred, task.query_y.float()))
    return torch.stack(losses).mean()


def train_task_encoder(
    encoder: LearnedTaskEncoder,
    decoder: TaskConditionedRegressor,
    train_tasks: Sequence[SyntheticTask],
    validation_tasks: Sequence[SyntheticTask] = (),
    *,
    steps: int = 40,
    lr: float = 0.01,
) -> EncoderTrainingResult:
    """Train the encoder/decoder on a support-only reconstruction proxy."""

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    train_losses: List[float] = []
    for _ in range(max(1, steps)):
        opt.zero_grad()
        loss = task_reconstruction_loss(encoder, decoder, train_tasks)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.detach()))
    metrics = evaluate_task_encoder(encoder, decoder, validation_tasks or train_tasks)
    return EncoderTrainingResult(train_losses, metrics)


def evaluate_task_encoder(encoder: LearnedTaskEncoder, decoder: TaskConditionedRegressor, tasks: Sequence[SyntheticTask]) -> Dict[str, float]:
    if not tasks:
        return {"reconstruction_loss": 0.0, "handcrafted_baseline_loss": 0.0, "learned_minus_baseline": 0.0}
    with torch.no_grad():
        learned_loss = float(task_reconstruction_loss(encoder, decoder, tasks).detach())
        baseline_loss = handcrafted_support_baseline_loss(tasks)
    return {
        "reconstruction_loss": learned_loss,
        "handcrafted_baseline_loss": baseline_loss,
        "learned_minus_baseline": learned_loss - baseline_loss,
    }


def handcrafted_support_baseline_loss(tasks: Sequence[SyntheticTask]) -> float:
    """A hand-crafted non-learned baseline using support mean/slope statistics."""
    infer = TaskInferenceModule()
    losses: List[float] = []
    for task in tasks:
        features = infer.infer(task.support_x, task.support_y)
        y_mean = features[2]
        corr = features[4]
        slope_abs = features[5].clamp(max=3.0)
        support_x = task.support_x.float().reshape(task.support_x.shape[0], -1)
        query_x = task.query_x.float().reshape(task.query_x.shape[0], -1)
        x_center = support_x.mean(dim=0, keepdim=True)
        query_centered = (query_x - x_center).mean(dim=1, keepdim=True)
        # A crude signed local linear predictor from the old statistics.
        pred = y_mean + corr.sign() * slope_abs * 0.25 * query_centered
        losses.append(float(F.mse_loss(pred, task.query_y.float())))
    return float(sum(losses) / max(1, len(losses)))


def make_encoder_training_tasks(seed: int = 0, n_train: int = 10, n_val: int = 5, ood_val: bool = True) -> Tuple[List[SyntheticTask], List[SyntheticTask]]:
    suite = ProceduralTaskSuite(seed=seed)
    return suite.sample_function_tasks("train", n_train, ood=False), suite.sample_function_tasks("validation", n_val, ood=ood_val)
