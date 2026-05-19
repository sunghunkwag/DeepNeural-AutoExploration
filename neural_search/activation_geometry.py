"""Activation geometry utilities for representation probes."""

from __future__ import annotations

from itertools import combinations
from statistics import mean
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def flatten_activation(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    return value.reshape(value.shape[0], -1)


def activation_variance(tensor: torch.Tensor) -> float:
    flat = flatten_activation(tensor)
    return float(torch.var(flat, dim=0, unbiased=False).mean()) if flat.numel() else 0.0


def activation_saturation(tensor: torch.Tensor, threshold: float = 0.97) -> float:
    flat = flatten_activation(tensor)
    if not flat.numel():
        return 0.0
    squashed = torch.tanh(flat).abs()
    return float((squashed > float(threshold)).float().mean())


def module_overdominance_score(values: Mapping[str, float]) -> float:
    positives = [abs(float(value)) for value in values.values()]
    if not positives:
        return 0.0
    total = sum(positives)
    if total <= 1e-12:
        return 0.0
    return float(max(positives) / total)


def representation_collapse_score(activation_variances: Mapping[str, float]) -> float:
    if not activation_variances:
        return 1.0
    mean_var = mean(float(value) for value in activation_variances.values())
    return float(1.0 / (1.0 + max(0.0, mean_var)))


def centroid_separability(family_embeddings: Mapping[str, torch.Tensor]) -> float:
    centroids = {key: flatten_activation(value).mean(dim=0) for key, value in family_embeddings.items() if value.numel()}
    if len(centroids) <= 1:
        return 0.0
    distances = []
    for left, right in combinations(sorted(centroids), 2):
        distances.append(float(torch.norm(centroids[left] - centroids[right], p=2)))
    within = []
    for key, values in family_embeddings.items():
        flat = flatten_activation(values)
        if flat.shape[0] > 1:
            within.append(float(torch.norm(flat - flat.mean(dim=0, keepdim=True), dim=1).mean()))
    return float(mean(distances) / (1.0 + (mean(within) if within else 0.0)))


def mean_pairwise_cosine_similarity(family_embeddings: Mapping[str, torch.Tensor]) -> float:
    centroids = [flatten_activation(value).mean(dim=0) for _, value in sorted(family_embeddings.items()) if value.numel()]
    if len(centroids) <= 1:
        return 1.0
    values = []
    for left, right in combinations(centroids, 2):
        values.append(float(F.cosine_similarity(left, right, dim=0)))
    return float(mean(values)) if values else 1.0
