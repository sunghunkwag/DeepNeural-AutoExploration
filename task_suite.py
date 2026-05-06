"""Procedural task suite for the AGI-oriented scaffold."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    family: str
    split: str
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    metadata: Dict[str, object] = field(default_factory=dict)


class ProceduralTaskSuite:
    """Generates non-identical train/validation/test tasks from named families."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.counter = 0

    def _id(self, split: str, family: str) -> str:
        self.counter += 1
        return f"{split}-{family}-{self.seed}-{self.counter}-{uuid.uuid4().hex[:8]}"

    def _x_range(self, split: str, ood: bool) -> Tuple[float, float]:
        if ood and split in {"validation", "test", "eval"}:
            return (-7.0, 7.0)
        return (-5.0, 5.0)

    def _sample_x(self, n: int, split: str, ood: bool) -> torch.Tensor:
        lo, hi = self._x_range(split, ood)
        x = self.rng.uniform(lo, hi, size=(n, 1)).astype(np.float32)
        return torch.from_numpy(x)

    def sample_function_task(self, family: str = "sinusoid", split: str = "train", n_support: int = 5, n_query: int = 16, ood: bool = False) -> SyntheticTask:
        sx = self._sample_x(n_support, split, ood)
        qx = self._sample_x(n_query, split, ood)
        params: Dict[str, float] = {}

        if family == "sinusoid":
            amp_hi = 7.0 if (ood and split != "train") else 5.0
            a = float(self.rng.uniform(0.3, amp_hi))
            phase = float(self.rng.uniform(0, math.pi))
            fn = lambda x: a * torch.sin(x + phase)
            params = {"amplitude": a, "phase": phase}
        elif family == "polynomial":
            coeff = self.rng.uniform(-0.6, 0.6, size=3).astype(np.float32)
            if ood and split != "train":
                coeff += np.array([0.4, -0.3, 0.2], dtype=np.float32)
            c = torch.from_numpy(coeff)
            fn = lambda x: c[0] * x.pow(2) + c[1] * x + c[2]
            params = {"coefficients": coeff.tolist()}
        elif family == "piecewise_linear":
            left = float(self.rng.uniform(-2, 2))
            right = float(self.rng.uniform(-2, 2))
            kink = float(self.rng.uniform(-1, 1))
            fn = lambda x: torch.where(x < kink, left * (x - kink), right * (x - kink))
            params = {"left": left, "right": right, "kink": kink}
        elif family == "noisy_affine":
            slope = float(self.rng.uniform(-3, 3))
            bias = float(self.rng.uniform(-1, 1))
            noise = 0.05 if split == "train" else 0.08
            fn = lambda x: slope * x + bias + noise * torch.randn_like(x)
            params = {"slope": slope, "bias": bias, "noise": noise}
        elif family == "compositional":
            scale = float(self.rng.uniform(0.5, 2.0))
            shift = float(self.rng.uniform(-1.0, 1.0))
            fn = lambda x: torch.sin(scale * x) + 0.25 * torch.tanh(x + shift) + 0.05 * x.pow(2)
            params = {"scale": scale, "shift": shift}
        else:
            raise ValueError(f"unknown function family: {family}")

        sy = fn(sx).float()
        qy = fn(qx).float()
        return SyntheticTask(self._id(split, family), family, split, sx, sy, qx, qy, {"params": params, "ood": ood})

    def sample_function_tasks(self, split: str, count: int, ood: bool = False) -> List[SyntheticTask]:
        families = ["sinusoid", "polynomial", "piecewise_linear", "noisy_affine", "compositional"]
        return [self.sample_function_task(families[i % len(families)], split=split, ood=ood) for i in range(count)]

    def sample_sequence_task(self, family: str = "repeating", split: str = "train", length: int = 24) -> Dict[str, object]:
        vocab = 5
        if family == "repeating":
            pattern = self.rng.integers(0, vocab, size=int(self.rng.integers(2, 5)))
            seq = np.resize(pattern, length + 1)
        elif family == "switching_rules":
            a = self.rng.integers(0, vocab, size=length + 1)
            switch = length // 2
            seq = np.concatenate([(a[:switch] + 1) % vocab, (a[switch:] + 2) % vocab])[: length + 1]
        elif family == "delayed_dependency":
            seq = self.rng.integers(0, vocab, size=length + 1)
            seq[3:] = seq[:-3]
        elif family == "nonstationary":
            seq = np.cumsum(self.rng.integers(0, 2, size=length + 1)) % vocab
            seq[length // 2 :] = (seq[length // 2 :] + 2) % vocab
        else:
            raise ValueError(f"unknown sequence family: {family}")
        return {"task_id": self._id(split, family), "family": family, "split": split, "inputs": seq[:-1].tolist(), "targets": seq[1:].tolist()}

    def sample_dynamics_task(self, split: str = "train", n: int = 32, intervention: bool = False) -> Dict[str, torch.Tensor | str]:
        state = torch.from_numpy(self.rng.normal(size=(n, 2)).astype(np.float32))
        action = torch.from_numpy(self.rng.normal(scale=0.4, size=(n, 2)).astype(np.float32))
        matrix = torch.tensor([[1.0, 0.15], [-0.05, 0.95]])
        if intervention:
            action = action + torch.tensor([0.5, -0.25])
        next_state = state @ matrix.T + action + 0.05 * torch.sin(state)
        return {"task_id": self._id(split, "latent_dynamics"), "split": split, "state": state, "action": action, "next_state": next_state}

    def sample_memory_task(self, split: str = "train", n_pairs: int = 8, n_distractors: int = 4) -> Dict[str, object]:
        keys = torch.from_numpy(self.rng.normal(size=(n_pairs, 4)).astype(np.float32))
        values = torch.from_numpy(self.rng.normal(size=(n_pairs, 3)).astype(np.float32))
        query_index = int(self.rng.integers(0, n_pairs))
        distractors = torch.from_numpy(self.rng.normal(loc=3.0, size=(n_distractors, 4)).astype(np.float32))
        return {
            "task_id": self._id(split, "memory_retrieval"),
            "split": split,
            "keys": keys,
            "values": values,
            "query": keys[query_index] + 0.01 * torch.randn(4),
            "target_value": values[query_index],
            "distractors": distractors,
        }

    def sample_planning_task(self, split: str = "train", size: int = 4) -> Dict[str, object]:
        start = torch.tensor([0.0, 0.0])
        goal = torch.tensor([float(size - 1), float(size - 1)])
        actions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0]), torch.tensor([0.0, -1.0])]
        return {"task_id": self._id(split, "grid_planning"), "split": split, "start": start, "goal": goal, "actions": actions, "size": size}


def assert_disjoint_task_ids(*task_groups: List[SyntheticTask]) -> None:
    seen = set()
    for group in task_groups:
        for task in group:
            if task.task_id in seen:
                raise AssertionError(f"overlapping task id: {task.task_id}")
            seen.add(task.task_id)


def function_task_variance(task: SyntheticTask) -> float:
    return float(torch.var(torch.cat([task.support_y.flatten(), task.query_y.flatten()]), unbiased=False))
