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

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "x_shape": list(self.x.shape),
            "y_shape": list(self.y.shape),
            "used_heldout_labels": self.used_heldout_labels,
        }


@dataclass(frozen=True)
class MultiDomainTaskFamily:
    family: str
    splits: Dict[str, MultiDomainBatch]

    def split(self, name: str) -> MultiDomainBatch:
        return self.splits[name]

    def selection_splits(self) -> List[MultiDomainBatch]:
        return [self.splits[name] for name in ("train", "validation", "hidden_validation")]

    def to_dict(self) -> Dict[str, object]:
        return {"family": self.family, "splits": {key: value.to_dict() for key, value in self.splits.items()}}


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
) -> List[MultiDomainTaskFamily]:
    if input_dim != 4 or output_dim != 2:
        raise ValueError("multi-domain neural exploration tasks use input_dim=4 and output_dim=2")
    return [
        MultiDomainTaskFamily(
            family=family,
            splits={
                split: _make_batch(family, split, seed=seed, samples=max(8, int(samples_per_split)))
                for split in SPLITS
            },
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


def _make_batch(family: str, split: str, *, seed: int, samples: int) -> MultiDomainBatch:
    family_index = TASK_FAMILIES.index(family)
    split_index = SPLITS.index(split)
    g = _generator(seed, family_index, split_index)
    offset = _split_offset(split)
    if family == "function_regression":
        x, y = _function_regression(samples, g, offset)
    elif family == "sequence_prediction":
        x, y = _sequence_prediction(samples, g, offset)
    elif family == "object_state_transition":
        x, y = _object_state_transition(samples, g, offset)
    elif family == "causal_intervention":
        x, y = _causal_intervention(samples, g, offset)
    elif family == "memory_retrieval":
        x, y = _memory_retrieval(samples, g, offset)
    else:
        raise ValueError(f"unknown task family: {family}")
    return MultiDomainBatch(
        task_id=f"{split}-{family}-seed{seed}",
        family=family,
        split=split,
        x=x.float(),
        y=y.float(),
        used_heldout_labels=False,
    )


def _function_regression(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.linspace(-1.5 + offset, 1.5 + offset, samples).reshape(-1, 1)
    x = torch.cat([z, z.pow(2), torch.sin(2.0 * z), torch.cos(3.0 * z)], dim=1)
    scale = 1.0 + 0.2 * offset
    y = torch.cat([torch.sin(scale * z) + 0.2 * z.pow(2), torch.cos((1.5 + offset) * z)], dim=1)
    return x, y


def _sequence_prediction(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.randn(samples + 3, generator=g) * 0.15
    signal = torch.sin(torch.linspace(0.0, 4.0 + offset, samples + 3)) + base
    prev2 = signal[:-3]
    prev1 = signal[1:-2]
    current = signal[2:-1]
    t = torch.linspace(0.0, 1.0, samples)
    x = torch.stack([prev2, prev1, current, t + offset], dim=1)
    next_value = 0.25 * prev2 + 0.35 * prev1 + 0.55 * current + 0.1 * torch.sin(5.0 * t + offset)
    y = torch.stack([next_value, next_value - current], dim=1)
    return x, y


def _object_state_transition(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.randn(samples, 2, generator=g) * (0.6 + abs(offset))
    vel = torch.randn(samples, 2, generator=g) * 0.25 + torch.tensor([0.15 + offset * 0.1, -0.05])
    x = torch.cat([pos, vel], dim=1)
    next_pos = pos + vel + 0.08 * torch.sin(pos)
    return x, next_pos


def _causal_intervention(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    attr = torch.randn(samples, 1, generator=g) * 0.7
    context = torch.randn(samples, 1, generator=g) * 0.4 + offset
    intervention = torch.randn(samples, 1, generator=g) * 0.5 + 0.25
    do_flag = (torch.arange(samples).reshape(-1, 1) % 2).float()
    x = torch.cat([attr, context, intervention, do_flag], dim=1)
    natural = attr + 0.35 * context
    post = torch.where(do_flag > 0.5, intervention + 0.15 * context, natural)
    effect = post - natural
    return x, torch.cat([post, effect], dim=1)


def _memory_retrieval(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    key_count = 6
    keys = torch.randn(key_count, 2, generator=g)
    values = torch.stack([torch.sin(keys[:, 0] * 1.7 + offset), torch.cos(keys[:, 1] * 1.3 - offset)], dim=1)
    indices = torch.arange(samples) % key_count
    noise = torch.randn(samples, 2, generator=g) * (0.03 + 0.02 * abs(offset))
    query = keys.index_select(0, indices) + noise
    context = torch.stack([
        torch.full((samples,), float(key_count) / 10.0),
        torch.linspace(-1.0, 1.0, samples) + offset,
    ], dim=1)
    x = torch.cat([query, context], dim=1)
    y = values.index_select(0, indices)
    return x, y


def split_batches_by_name(task_families: Iterable[MultiDomainTaskFamily]) -> Dict[str, List[MultiDomainBatch]]:
    return {split: [family.splits[split] for family in task_families] for split in SPLITS}


def serializable_task_families(task_families: Iterable[MultiDomainTaskFamily]) -> List[Dict[str, object]]:
    return [family.to_dict() for family in task_families]
