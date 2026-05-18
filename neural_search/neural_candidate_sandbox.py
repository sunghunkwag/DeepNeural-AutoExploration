"""Validation-only sandbox for neural architecture candidates."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .architecture_genome import ArchitectureGenome
from .module_library import set_deterministic_seed


@dataclass(frozen=True)
class NeuralSandboxTask:
    task_id: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    hidden_validation_x: Optional[torch.Tensor] = None
    hidden_validation_y: Optional[torch.Tensor] = None
    split_used: str = "validation"
    used_heldout_labels: bool = False


@dataclass
class NeuralSandboxResult:
    genome_id: str
    ok: bool
    validation_metrics: Dict[str, float]
    hidden_validation_metrics: Dict[str, float] = field(default_factory=dict)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    split_used: str = "validation"
    used_heldout_labels: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0
    model: Optional[torch.nn.Module] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, object]:
        return {
            "genome_id": self.genome_id,
            "ok": self.ok,
            "validation_metrics": dict(self.validation_metrics),
            "hidden_validation_metrics": dict(self.hidden_validation_metrics),
            "train_metrics": dict(self.train_metrics),
            "split_used": self.split_used,
            "used_heldout_labels": self.used_heldout_labels,
            "error": self.error,
            "elapsed_seconds": float(self.elapsed_seconds),
        }


class NeuralCandidateSandbox:
    """Train and score candidates on procedural validation tasks.

    Held-out test labels are not part of this sandbox.  If a task is explicitly
    marked as using held-out labels, evaluation fails before training.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        train_steps: int = 12,
        lr: float = 0.02,
        task_count: int = 3,
        samples_per_split: int = 24,
    ):
        self.seed = int(seed)
        self.train_steps = max(1, int(train_steps))
        self.lr = float(lr)
        self.task_count = max(1, int(task_count))
        self.samples_per_split = max(8, int(samples_per_split))

    def procedural_tasks(self, genome: ArchitectureGenome) -> List[NeuralSandboxTask]:
        tasks: List[NeuralSandboxTask] = []
        for index in range(self.task_count):
            generator = torch.Generator().manual_seed(self.seed + 101 * index + genome.input_dim + genome.output_dim)
            weight = torch.randn(genome.input_dim, genome.output_dim, generator=generator) * 0.7
            bias = torch.randn(genome.output_dim, generator=generator) * 0.1

            def make_split(offset: float) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.randn(self.samples_per_split, genome.input_dim, generator=generator) + offset
                y = torch.sin(x @ weight + bias)
                return x, y

            train_x, train_y = make_split(0.0)
            val_x, val_y = make_split(0.15)
            hidden_x, hidden_y = make_split(-0.2)
            tasks.append(
                NeuralSandboxTask(
                    task_id=f"procedural-{index}",
                    train_x=train_x,
                    train_y=train_y,
                    validation_x=val_x,
                    validation_y=val_y,
                    hidden_validation_x=hidden_x,
                    hidden_validation_y=hidden_y,
                )
            )
        return tasks

    def evaluate(
        self,
        genome: ArchitectureGenome,
        tasks: Optional[Sequence[NeuralSandboxTask]] = None,
        *,
        initial_model: Optional[torch.nn.Module] = None,
    ) -> NeuralSandboxResult:
        start = time.monotonic()
        active_tasks = list(tasks) if tasks is not None else self.procedural_tasks(genome)
        if not active_tasks:
            raise ValueError("at least one sandbox task is required")
        for task in active_tasks:
            if task.used_heldout_labels:
                raise ValueError("neural sandbox refuses tasks marked as using held-out labels")
            if task.split_used == "test":
                raise ValueError("test split cannot be used for candidate selection")

        set_deterministic_seed(genome.seed + self.seed)
        model = initial_model if initial_model is not None else genome.build_module()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        try:
            model.train()
            train_losses: List[float] = []
            for _ in range(self.train_steps):
                for task in active_tasks:
                    optimizer.zero_grad()
                    loss = F.mse_loss(model(task.train_x), task.train_y.float())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite train loss")
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.detach()))

            model.eval()
            validation_losses: List[float] = []
            hidden_losses: List[float] = []
            with torch.no_grad():
                for task in active_tasks:
                    validation_loss = F.mse_loss(model(task.validation_x), task.validation_y.float())
                    validation_losses.append(float(validation_loss))
                    if task.hidden_validation_x is not None and task.hidden_validation_y is not None:
                        hidden_loss = F.mse_loss(model(task.hidden_validation_x), task.hidden_validation_y.float())
                        hidden_losses.append(float(hidden_loss))
            if any(not math.isfinite(value) for value in validation_losses + hidden_losses):
                raise FloatingPointError("non-finite validation loss")
            return NeuralSandboxResult(
                genome_id=genome.genome_id,
                ok=True,
                validation_metrics={"loss": float(mean(validation_losses))},
                hidden_validation_metrics={"loss": float(mean(hidden_losses))} if hidden_losses else {},
                train_metrics={"loss": float(mean(train_losses)) if train_losses else 0.0},
                elapsed_seconds=time.monotonic() - start,
                model=model,
            )
        except Exception as exc:
            return NeuralSandboxResult(
                genome_id=genome.genome_id,
                ok=False,
                validation_metrics={},
                error=repr(exc),
                elapsed_seconds=time.monotonic() - start,
                model=model,
            )
