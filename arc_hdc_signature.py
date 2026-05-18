"""Hyperdimensional ARC grid signatures inspired by OMEGA-THDSE.

This module implements a small, dependency-light FHRR-style phase-vector
encoder for ARC grids.  It uses 10,000-dimensional phase vectors, role binding
by element-wise phase addition, and bundling by circular mean.  The encoder is
not a magic intelligence mechanism; it is a deterministic structural scoring
tool that can be used to break ties between support-only ARC transformation
candidates without looking at held-out test labels.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np


HDC_DIMENSION = 10_000
_TWO_PI = np.float32(2.0 * np.pi)


def _seed_for(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@lru_cache(maxsize=512)
def phase_vector(key: str, dimension: int = HDC_DIMENSION) -> np.ndarray:
    """Return a deterministic phase vector for ``key``."""

    rng = np.random.default_rng(_seed_for(f"{dimension}:{key}"))
    return rng.uniform(0.0, float(_TWO_PI), size=dimension).astype(np.float32)


def bind_phases(*vectors: np.ndarray) -> np.ndarray:
    """Bind FHRR phase vectors by element-wise phase addition."""

    if not vectors:
        raise ValueError("bind_phases requires at least one vector")
    total = np.zeros_like(vectors[0], dtype=np.float32)
    for vector in vectors:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != total.shape:
            raise ValueError(f"phase dimension mismatch: {arr.shape} != {total.shape}")
        total = total + arr
    return np.mod(total, _TWO_PI)


def bundle_phases(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Bundle phase vectors by circular mean."""

    if not vectors:
        raise ValueError("bundle_phases requires at least one vector")
    shape = np.asarray(vectors[0]).shape
    sin_sum = np.zeros(shape, dtype=np.float32)
    cos_sum = np.zeros(shape, dtype=np.float32)
    for vector in vectors:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(f"phase dimension mismatch: {arr.shape} != {shape}")
        sin_sum += np.sin(arr)
        cos_sum += np.cos(arr)
    return np.mod(np.arctan2(sin_sum, cos_sum).astype(np.float32), _TWO_PI)


def fhrr_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return mean cosine similarity between two FHRR phase vectors."""

    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"phase dimension mismatch: {a.shape} != {b.shape}")
    return float(np.mean(np.cos(a - b)))


GridKey = Tuple[Tuple[int, ...], ...]


def _grid_key(grid: Sequence[Sequence[int]]) -> GridKey:
    return tuple(tuple(int(value) for value in row) for row in grid)


@lru_cache(maxsize=2048)
def _cached_grid_signature(grid: GridKey, role: str, dimension: int) -> np.ndarray:
    cells = []
    role_vec = phase_vector(f"role:{role}", dimension)
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            cells.append(
                bind_phases(
                    role_vec,
                    phase_vector(f"row:{row_index}", dimension),
                    phase_vector(f"col:{col_index}", dimension),
                    phase_vector(f"color:{int(value)}", dimension),
                )
            )
    if not cells:
        return phase_vector(f"empty:{role}", dimension)
    return bundle_phases(cells)


def grid_signature(grid: Sequence[Sequence[int]], *, role: str) -> np.ndarray:
    """Encode an ARC grid as a role-bound 10,000-dimensional phase vector."""

    return _cached_grid_signature(_grid_key(grid), role, HDC_DIMENSION)


def grid_pair_similarity(predicted: Sequence[Sequence[int]], expected: Sequence[Sequence[int]]) -> float:
    """Compare two grids in the same high-dimensional role-bound space."""

    return fhrr_similarity(
        grid_signature(predicted, role="predicted"),
        grid_signature(expected, role="predicted"),
    )
