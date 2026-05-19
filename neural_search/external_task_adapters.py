"""Local external-style neural task adapters with isolated splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .task_families import MultiDomainBatch, MultiDomainTaskFamily, SPLITS


EXTERNAL_FAMILIES = (
    "external_synthetic_classification",
    "external_image_tensor",
    "external_character_sequence",
)


@dataclass(frozen=True)
class ExternalAdapterSpec:
    name: str
    input_dim: int = 4
    output_dim: int = 2
    splits: tuple[str, ...] = SPLITS

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "input_dim": self.input_dim, "output_dim": self.output_dim, "splits": list(self.splits)}


def build_external_task_families(*, seed: int = 0, samples_per_split: int = 24) -> List[MultiDomainTaskFamily]:
    return [
        MultiDomainTaskFamily(
            family=family,
            difficulty="external",
            splits={
                split: _make_external_batch(family, split, seed=seed, samples=max(8, int(samples_per_split)))
                for split in SPLITS
            },
        )
        for family in EXTERNAL_FAMILIES
    ]


def assert_external_splits_disjoint(task_families: List[MultiDomainTaskFamily]) -> None:
    seen: set[str] = set()
    for family in task_families:
        if set(family.splits) != set(SPLITS):
            raise AssertionError(f"external family missing split: {family.family}")
        for split, batch in family.splits.items():
            if batch.task_id in seen:
                raise AssertionError(f"overlapping external task id: {batch.task_id}")
            if split == "heldout_test" and batch.used_heldout_labels:
                raise AssertionError("heldout labels must not be marked as used during task construction")
            seen.add(batch.task_id)


def serializable_external_task_families(task_families: List[MultiDomainTaskFamily]) -> List[Dict[str, object]]:
    return [family.to_dict() for family in task_families]


def external_split_distribution_summary(task_families: List[MultiDomainTaskFamily]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for family in task_families:
        train = family.splits["train"].x
        heldout = family.splits["heldout_test"].x
        summary[family.family] = {
            "train_feature_mean": [float(value) for value in train.mean(dim=0)],
            "heldout_feature_mean": [float(value) for value in heldout.mean(dim=0)],
            "feature_distribution_shift_l2": float(torch.norm(train.mean(dim=0) - heldout.mean(dim=0))),
            "train_target_mean": [float(value) for value in family.splits["train"].y.mean(dim=0)],
            "heldout_target_mean": [float(value) for value in family.splits["heldout_test"].y.mean(dim=0)],
        }
    return summary


def _make_external_batch(family: str, split: str, *, seed: int, samples: int) -> MultiDomainBatch:
    family_index = EXTERNAL_FAMILIES.index(family)
    split_index = SPLITS.index(split)
    g = torch.Generator().manual_seed(int(seed) + 7001 * family_index + 313 * split_index)
    offset = {"train": 0.0, "validation": 0.15, "hidden_validation": -0.25, "heldout_test": 0.45}[split]
    if family == "external_synthetic_classification":
        x, y = _synthetic_classification(samples, g, offset)
    elif family == "external_image_tensor":
        x, y = _image_like_tensor(samples, g, offset)
    elif family == "external_character_sequence":
        x, y = _character_sequence(samples, g, offset)
    else:
        raise ValueError(f"unknown external family: {family}")
    return MultiDomainBatch(
        task_id=f"{split}-{family}-seed{seed}",
        family=family,
        split=split,
        x=x.float(),
        y=y.float(),
        used_heldout_labels=False,
        difficulty="external",
    )


def _synthetic_classification(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(samples, 4, generator=g) + torch.tensor([offset, -offset, 0.5 * offset, 0.0])
    logit = x[:, 0] * x[:, 1] - 0.6 * x[:, 2] + torch.sin(x[:, 3] + offset)
    prob = torch.sigmoid(logit)
    y = torch.stack([prob, 1.0 - prob], dim=1)
    return x, y


def _image_like_tensor(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.rand(samples, 4, generator=g)
    top = pixels[:, 0] + pixels[:, 1] + offset
    bottom = pixels[:, 2] + pixels[:, 3] - offset
    edge = torch.tanh(2.0 * (top - bottom))
    texture = torch.sin(3.0 * pixels[:, 0] + 2.0 * pixels[:, 3] + offset)
    y = torch.stack([(edge + 1.0) / 2.0, (texture + 1.0) / 2.0], dim=1)
    return pixels * 2.0 - 1.0, y


def _character_sequence(samples: int, g: torch.Generator, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.randint(0, 8, (samples,), generator=g).float()
    prev = torch.roll(token, shifts=1)
    nxt = (token + prev + int(abs(offset) * 10)) % 8
    pos = torch.linspace(0.0, 1.0, samples)
    x = torch.stack([token / 7.0, prev / 7.0, pos, torch.sin(6.28 * pos + offset)], dim=1)
    y = torch.stack([nxt / 7.0, torch.cos(nxt + offset)], dim=1)
    return x, y
