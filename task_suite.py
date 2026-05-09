"""Procedural task suite for the AGI-oriented scaffold."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

    @property
    def ood(self) -> bool:
        return bool(self.metadata.get("ood", False))


class ProceduralTaskSuite:
    """Generates non-identical train/validation/test tasks from named families."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.counter = 0

    def _id(self, split: str, family: str) -> str:
        self.counter += 1
        return f"{split}-{family}-{self.seed}-{self.counter:04d}"

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

    def sample_sequence_regression_task(
        self,
        family: str = "nonstationary_sequence",
        split: str = "train",
        n_support: int = 8,
        n_query: int = 16,
        ood: bool = False,
    ) -> SyntheticTask:
        """Return a sequence-style task in the same support/query tensor format.

        The model still sees scalar context pairs, but the targets are generated
        from delayed or regime-switching dynamics so support context and memory
        are useful for selecting update settings.
        """

        total = n_support + n_query
        t = torch.linspace(0.0, 1.0, total).unsqueeze(1)
        phase = float(self.rng.uniform(-0.7, 0.7))
        freq = float(self.rng.uniform(1.0, 2.2))
        if ood and split != "train":
            freq += 1.2
            phase += 0.4

        if family == "nonstationary_sequence":
            switch = int(total * (0.45 if not ood else 0.35))
            left = torch.sin((freq * math.pi) * t + phase)
            right = torch.cos(((freq + 0.7) * math.pi) * t - phase)
            y = torch.where(torch.arange(total).reshape(-1, 1) < switch, left, right)
            y = y + 0.05 * torch.linspace(-1.0, 1.0, total).unsqueeze(1)
            params = {"freq": freq, "phase": phase, "switch": switch}
        elif family == "delayed_dependency_sequence":
            base = torch.sin((freq * math.pi) * t + phase)
            delayed = torch.roll(base, shifts=2 if not ood else 3, dims=0)
            delayed[:3] = base[:3].mean()
            y = 0.65 * base + 0.35 * delayed
            params = {"freq": freq, "phase": phase, "delay": 2 if not ood else 3}
        else:
            raise ValueError(f"unknown sequence regression family: {family}")

        sx, sy = t[:n_support].float(), y[:n_support].float()
        qx, qy = t[n_support:].float(), y[n_support:].float()
        return SyntheticTask(self._id(split, family), family, split, sx, sy, qx, qy, {"params": params, "ood": ood})

    def sample_memory_association_task(
        self,
        split: str = "train",
        n_support: int = 8,
        n_query: int = 16,
        ood: bool = False,
    ) -> SyntheticTask:
        """Scalar key/value association task with distractor-style query keys."""

        keys = np.linspace(-1.0, 1.0, n_support).astype(np.float32)
        self.rng.shuffle(keys)
        phase = float(self.rng.uniform(-0.5, 0.5))
        values = np.sin(2.3 * keys + phase).astype(np.float32)
        if ood and split != "train":
            query_keys = self.rng.uniform(-1.6, 1.6, size=n_query).astype(np.float32)
            query_values = np.sin(2.9 * query_keys + phase + 0.25).astype(np.float32)
        else:
            nearest = self.rng.choice(keys, size=n_query, replace=True)
            query_keys = (nearest + self.rng.normal(scale=0.03, size=n_query)).astype(np.float32)
            query_values = np.sin(2.3 * query_keys + phase).astype(np.float32)
        sx = torch.from_numpy(keys.reshape(-1, 1))
        sy = torch.from_numpy(values.reshape(-1, 1))
        qx = torch.from_numpy(query_keys.reshape(-1, 1))
        qy = torch.from_numpy(query_values.reshape(-1, 1))
        return SyntheticTask(
            self._id(split, "memory_association"),
            "memory_association",
            split,
            sx,
            sy,
            qx,
            qy,
            {"params": {"phase": phase}, "ood": ood, "has_distractors": True},
        )

    def sample_latent_navigation_task(
        self,
        split: str = "train",
        n_support: int = 8,
        n_query: int = 16,
        ood: bool = False,
    ) -> SyntheticTask:
        """A tiny planning proxy: predict distance-to-goal under latent motion."""

        goal = float(self.rng.uniform(0.7, 1.1) + (0.4 if ood and split != "train" else 0.0))
        speed = float(self.rng.uniform(0.4, 0.9))
        t = torch.linspace(0.0, 1.0, n_support + n_query).unsqueeze(1)
        position = speed * t + 0.08 * torch.sin(4.0 * t)
        y = (goal - position).abs()
        return SyntheticTask(
            self._id(split, "latent_navigation"),
            "latent_navigation",
            split,
            t[:n_support].float(),
            y[:n_support].float(),
            t[n_support:].float(),
            y[n_support:].float(),
            {"params": {"goal": goal, "speed": speed}, "ood": ood},
        )

    def sample_mixed_tasks(self, split: str, count: int, ood: bool = False) -> List[SyntheticTask]:
        families = [
            "sinusoid",
            "polynomial",
            "piecewise_linear",
            "noisy_affine",
            "compositional",
            "nonstationary_sequence",
            "delayed_dependency_sequence",
            "memory_association",
            "latent_navigation",
        ]
        tasks: List[SyntheticTask] = []
        for i in range(count):
            family = families[i % len(families)]
            if family in {"nonstationary_sequence", "delayed_dependency_sequence"}:
                tasks.append(self.sample_sequence_regression_task(family, split=split, ood=ood))
            elif family == "memory_association":
                tasks.append(self.sample_memory_association_task(split=split, ood=ood))
            elif family == "latent_navigation":
                tasks.append(self.sample_latent_navigation_task(split=split, ood=ood))
            else:
                tasks.append(self.sample_function_task(family, split=split, ood=ood))
        return tasks

    def sample_sequence_task(self, family: str = "repeating", split: str = "train", length: int = 24, ood: bool = False) -> Dict[str, object]:
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
        return {"task_id": self._id(split, family), "family": family, "split": split, "ood": ood, "inputs": seq[:-1].tolist(), "targets": seq[1:].tolist()}

    def sample_dynamics_task(self, split: str = "train", n: int = 32, intervention: bool = False) -> Dict[str, torch.Tensor | str | bool]:
        state = torch.from_numpy(self.rng.normal(size=(n, 2)).astype(np.float32))
        action = torch.from_numpy(self.rng.normal(scale=0.4, size=(n, 2)).astype(np.float32))
        matrix = torch.tensor([[1.0, 0.15], [-0.05, 0.95]])
        if intervention:
            action = action + torch.tensor([0.5, -0.25])
        next_state = state @ matrix.T + action + 0.05 * torch.sin(state)
        return {"task_id": self._id(split, "latent_dynamics"), "family": "latent_dynamics", "split": split, "ood": intervention, "state": state, "action": action, "next_state": next_state}

    def sample_memory_task(self, split: str = "train", n_pairs: int = 8, n_distractors: int = 4, ood: bool = False) -> Dict[str, object]:
        keys = torch.from_numpy(self.rng.normal(size=(n_pairs, 4)).astype(np.float32))
        values = torch.from_numpy(self.rng.normal(size=(n_pairs, 3)).astype(np.float32))
        query_index = int(self.rng.integers(0, n_pairs))
        distractors = torch.from_numpy(self.rng.normal(loc=3.0, size=(n_distractors, 4)).astype(np.float32))
        return {
            "task_id": self._id(split, "memory_retrieval"),
            "family": "memory_retrieval",
            "split": split,
            "ood": ood,
            "keys": keys,
            "values": values,
            "query": keys[query_index] + 0.01 * torch.randn(4),
            "target_value": values[query_index],
            "distractors": distractors,
        }

    def sample_planning_task(self, split: str = "train", size: int = 4, ood: bool = False) -> Dict[str, object]:
        start = torch.tensor([0.0, 0.0])
        goal = torch.tensor([float(size - 1), float(size - 1)])
        actions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0]), torch.tensor([0.0, -1.0])]
        return {"task_id": self._id(split, "grid_planning"), "family": "grid_planning", "split": split, "ood": ood, "start": start, "goal": goal, "actions": actions, "size": size}


def assert_disjoint_task_ids(*task_groups: Sequence[object]) -> None:
    seen = set()
    for group in task_groups:
        for task in group:
            task_id = str(task.get("task_id")) if isinstance(task, dict) else str(getattr(task, "task_id"))
            if task_id in seen:
                raise AssertionError(f"overlapping task id: {task_id}")
            seen.add(task_id)


def function_task_variance(task: SyntheticTask) -> float:
    return float(torch.var(torch.cat([task.support_y.flatten(), task.query_y.flatten()]), unbiased=False))
