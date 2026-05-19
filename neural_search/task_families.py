"""Deterministic multi-domain task families for neural exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import torch


TASK_FAMILIES = (
    "function_regression",
    "sequence_prediction",
    "object_state_transition",
    "causal_intervention",
    "memory_retrieval",
)

SPLITS = ("train", "validation", "hidden_validation", "heldout_test")


@dataclass(frozen=True)
class MultiDomainBatch:
    task_id: str
    family: str
    split: str
    x: torch.Tensor
    y: torch.Tensor
    used_heldout_labels: bool = False
    difficulty: str = "simple"

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "x_shape": list(self.x.shape),
            "y_shape": list(self.y.shape),
            "used_heldout_labels": self.used_heldout_labels,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class MultiDomainTaskFamily:
    family: str
    splits: Dict[str, MultiDomainBatch]
    difficulty: str = "simple"

    def split(self, name: str) -> MultiDomainBatch:
        return self.splits[name]

    def selection_splits(self) -> List[MultiDomainBatch]:
        return [self.splits[name] for name in ("train", "validation", "hidden_validation")]

    def to_dict(self) -> Dict[str, object]:
        return {"family": self.family, "difficulty": self.difficulty, "splits": {key: value.to_dict() for key, value in self.splits.items()}}


def _generator(seed: int, family_index: int, split_index: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) + 1009 * family_index + 97 * split_index)


def _split_offset(split: str) -> float:
    return {"train": 0.0, "validation": 0.2, "hidden_validation": -0.35, "heldout_test": 0.65}[split]


def build_multidomain_task_families(
    *,
    seed: int = 0,
    samples_per_split: int = 24,
    input_dim: int = 4,
    output_dim: int = 2,
    difficulty: str = "simple",
) -> List[MultiDomainTaskFamily]:
    if input_dim != 4 or output_dim != 2:
        raise ValueError("multi-domain neural exploration tasks use input_dim=4 and output_dim=2")
    if difficulty not in {"simple", "hard"}:
        raise ValueError("difficulty must be 'simple' or 'hard'")
    return [
        MultiDomainTaskFamily(
            family=family,
            splits={
                split: _make_batch(family, split, seed=seed, samples=max(8, int(samples_per_split)), difficulty=difficulty)
                for split in SPLITS
            },
            difficulty=difficulty,
        )
        for family in TASK_FAMILIES
    ]


def all_batches(task_families: Iterable[MultiDomainTaskFamily], splits: Iterable[str] = SPLITS) -> List[MultiDomainBatch]:
    split_set = set(splits)
    return [family.splits[split] for family in task_families for split in SPLITS if split in split_set]


def assert_disjoint_multidomain_splits(task_families: Iterable[MultiDomainTaskFamily]) -> None:
    seen = set()
    for batch in all_batches(task_families):
        if batch.task_id in seen:
            raise AssertionError(f"overlapping multi-domain task id: {batch.task_id}")
        seen.add(batch.task_id)
    for family in task_families:
        if set(family.splits) != set(SPLITS):
            raise AssertionError(f"{family.family} missing required splits")


def task_family_ids(task_families: Iterable[MultiDomainTaskFamily]) -> Dict[str, List[str]]:
    return {split: [family.splits[split].task_id for family in task_families] for split in SPLITS}


def _make_batch(family: str, split: str, *, seed: int, samples: int, difficulty: str) -> MultiDomainBatch:
    family_index = TASK_FAMILIES.index(family)
    split_index = SPLITS.index(split)
    g = _generator(seed, family_index, split_index)
    offset = _split_offset(split)
    if family == "function_regression":
        x, y = _function_regression(samples, g, offset, difficulty)
    elif family == "sequence_prediction":
        x, y = _sequence_prediction(samples, g, offset, difficulty)
    elif family == "object_state_transition":
        x, y = _object_state_transition(samples, g, offset, difficulty)
    elif family == "causal_intervention":
        x, y = _causal_intervention(samples, g, offset, difficulty)
    elif family == "memory_retrieval":
        x, y = _memory_retrieval(samples, g, offset, difficulty)
    else:
        raise ValueError(f"unknown task family: {family}")
    return MultiDomainBatch(
        task_id=f"{split}-{family}-seed{seed}",
        family=family,
        split=split,
        x=x.float(),
        y=y.float(),
        used_heldout_labels=False,
        difficulty=difficulty,
    )


def _function_regression(samples: int, g: torch.Generator, offset: float, difficulty: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.linspace(-1.5 + offset, 1.5 + offset, samples).reshape(-1, 1)
    if difficulty == "simple":
        x = torch.cat([z, z.pow(2), torch.sin(2.0 * z), torch.cos(3.0 * z)], dim=1)
        scale = 1.0 + 0.2 * offset
        y = torch.cat([torch.sin(scale * z) + 0.2 * z.pow(2), torch.cos((1.5 + offset) * z)], dim=1)
    else:
        phase = 0.35 + offset
        high = torch.sin(5.0 * z + phase) * torch.cos(2.7 * z - offset)
        cusp = torch.abs(z - 0.25 * torch.sign(z + 0.01))
        x = torch.cat([z, torch.sin(3.0 * z), torch.cos(5.0 * z), torch.sign(torch.sin(2.0 * z + phase))], dim=1)
        y = torch.cat([high + 0.35 * cusp, torch.sin(z.pow(2) * (2.0 + abs(offset))) + 0.25 * torch.tanh(4.0 * z)], dim=1)
    return x, y


def _sequence_prediction(samples: int, g: torch.Generator, offset: float, difficulty: str) -> tuple[torch.Tensor, torch.Tensor]:
    lag = 3 if difficulty == "simple" else 7
    total = samples + lag + 2
    base = torch.randn(total, generator=g) * (0.10 if difficulty == "simple" else 0.18)
    signal = torch.sin(torch.linspace(0.0, 4.0 + offset, total)) + base
    if difficulty == "hard":
        signal = signal + 0.35 * torch.roll(signal, shifts=3) * torch.sin(torch.linspace(0.0, 7.0, total))
    prev2 = signal[lag - 2 : lag - 2 + samples]
    prev1 = signal[lag - 1 : lag - 1 + samples]
    current = signal[lag : lag + samples]
    distant = signal[:samples]
    t = torch.linspace(0.0, 1.0, samples)
    x = torch.stack([prev2, prev1, current, distant if difficulty == "hard" else t + offset], dim=1)
    if difficulty == "simple":
        next_value = 0.25 * prev2 + 0.35 * prev1 + 0.55 * current + 0.1 * torch.sin(5.0 * t + offset)
    else:
        gate = torch.sigmoid(4.0 * distant)
        next_value = gate * current + (1.0 - gate) * distant + 0.2 * torch.sin(9.0 * t + offset)
    y = torch.stack([next_value, next_value - current], dim=1)
    return x, y


def _object_state_transition(samples: int, g: torch.Generator, offset: float, difficulty: str) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.randn(samples, 2, generator=g) * (0.6 + abs(offset))
    vel = torch.randn(samples, 2, generator=g) * 0.25 + torch.tensor([0.15 + offset * 0.1, -0.05])
    if difficulty == "simple":
        x = torch.cat([pos, vel], dim=1)
        next_pos = pos + vel + 0.08 * torch.sin(pos)
    else:
        other = torch.stack([torch.sin(pos[:, 1] + offset), torch.cos(pos[:, 0] - offset)], dim=1)
        relative = other - pos
        distance = torch.norm(relative, dim=1, keepdim=True).clamp_min(0.15)
        attraction = 0.22 * relative / distance.pow(2).clamp_max(12.0)
        collision = torch.where(distance < 0.75, -0.35 * relative / distance, torch.zeros_like(relative))
        x = torch.cat([pos[:, :1], pos[:, 1:2], vel[:, :1], other[:, :1]], dim=1)
        next_pos = pos + vel + attraction + collision + 0.05 * torch.sin(pos + other)
    return x, next_pos


def _causal_intervention(samples: int, g: torch.Generator, offset: float, difficulty: str) -> tuple[torch.Tensor, torch.Tensor]:
    attr = torch.randn(samples, 1, generator=g) * 0.7
    confounder = torch.randn(samples, 1, generator=g) * 0.5 + offset
    context = 0.65 * attr + confounder if difficulty == "hard" else torch.randn(samples, 1, generator=g) * 0.4 + offset
    intervention = torch.randn(samples, 1, generator=g) * 0.5 + 0.25 + (0.2 * confounder if difficulty == "hard" else 0.0)
    do_flag = (torch.arange(samples).reshape(-1, 1) % 2).float()
    x = torch.cat([attr, context, intervention, do_flag], dim=1)
    natural = attr + 0.35 * context + (0.4 * confounder if difficulty == "hard" else 0.0)
    post = torch.where(do_flag > 0.5, intervention + 0.15 * context, natural)
    effect = post - natural
    return x, torch.cat([post, effect], dim=1)


def _memory_retrieval(samples: int, g: torch.Generator, offset: float, difficulty: str) -> tuple[torch.Tensor, torch.Tensor]:
    key_count = 6 if difficulty == "simple" else 10
    keys = torch.randn(key_count, 2, generator=g)
    values = torch.stack([torch.sin(keys[:, 0] * 1.7 + offset) + 0.2 * keys[:, 1], torch.cos(keys[:, 1] * 1.3 - offset) - 0.15 * keys[:, 0]], dim=1)
    indices = torch.arange(samples) % key_count
    noise = torch.randn(samples, 2, generator=g) * ((0.03 + 0.02 * abs(offset)) if difficulty == "simple" else 0.09)
    distractor_indices = (indices + 1 + (torch.arange(samples) % max(1, key_count - 1))) % key_count
    distractor_mix = 0.0 if difficulty == "simple" else 0.35
    query = (1.0 - distractor_mix) * keys.index_select(0, indices) + distractor_mix * keys.index_select(0, distractor_indices) + noise
    context = torch.stack([
        torch.full((samples,), float(key_count) / 10.0),
        torch.linspace(-1.0, 1.0, samples) + offset + (0.2 * (indices.float() % 3.0) if difficulty == "hard" else 0.0),
    ], dim=1)
    x = torch.cat([query, context], dim=1)
    y = values.index_select(0, indices)
    return x, y


def split_batches_by_name(task_families: Iterable[MultiDomainTaskFamily]) -> Dict[str, List[MultiDomainBatch]]:
    return {split: [family.splits[split] for family in task_families] for split in SPLITS}


def serializable_task_families(task_families: Iterable[MultiDomainTaskFamily]) -> List[Dict[str, object]]:
    return [family.to_dict() for family in task_families]
