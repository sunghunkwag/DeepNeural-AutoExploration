"""Deterministic bounded architecture mutation operators."""

from __future__ import annotations

import random
from typing import Iterable, List

from .architecture_genome import ArchitectureGenome


MUTATION_METHODS = (
    "add_residual_block",
    "remove_residual_block",
    "widen_hidden_dim",
    "narrow_hidden_dim",
    "add_memory_gate",
    "add_causal_bottleneck",
    "object_binding_adapter",
    "sequence_state_adapter",
    "wider_residual_stack",
    "repair_bottleneck",
)


def _width_step(hidden_dim: int, rng: random.Random) -> int:
    base = max(2, hidden_dim // 4)
    jitter = rng.choice([0, 1, 2])
    return base + jitter


def mutate_architecture(parent: ArchitectureGenome, method: str, seed: int) -> ArchitectureGenome:
    if method not in MUTATION_METHODS:
        raise ValueError(f"unknown architecture mutation: {method}")
    rng = random.Random(int(seed))
    updates = {"mutation_method": method, "seed": int(seed), "parent_id": parent.genome_id, "genome_id": ""}
    if method == "add_residual_block":
        updates["residual_blocks"] = parent.residual_blocks + 1
    elif method == "remove_residual_block":
        updates["residual_blocks"] = max(0, parent.residual_blocks - 1)
    elif method == "widen_hidden_dim":
        updates["hidden_dim"] = parent.hidden_dim + _width_step(parent.hidden_dim, rng)
    elif method == "narrow_hidden_dim":
        updates["hidden_dim"] = max(2, parent.hidden_dim - _width_step(parent.hidden_dim, rng))
    elif method == "add_memory_gate":
        updates["memory_gates"] = parent.memory_gates + 1
        updates["memory_slots"] = min(8, parent.memory_slots + 1)
    elif method == "add_causal_bottleneck":
        updates["causal_bottlenecks"] = parent.causal_bottlenecks + 1
        updates["bottleneck_dim"] = max(1, min(parent.effective_bottleneck_dim, parent.hidden_dim // 2))
    elif method == "object_binding_adapter":
        updates["object_binding_adapters"] = parent.object_binding_adapters + 1
        updates["object_slots"] = min(6, parent.object_slots + 1)
    elif method == "sequence_state_adapter":
        updates["sequence_state_adapters"] = parent.sequence_state_adapters + 1
        updates["sequence_chunks"] = min(8, parent.sequence_chunks + 1)
    elif method == "wider_residual_stack":
        updates["residual_blocks"] = parent.residual_blocks + 2
        updates["hidden_dim"] = parent.hidden_dim + _width_step(parent.hidden_dim, rng)
    elif method == "repair_bottleneck":
        updates["causal_bottlenecks"] = parent.causal_bottlenecks + 1
        updates["bottleneck_dim"] = max(2, min(parent.hidden_dim, parent.effective_bottleneck_dim * 2))
    return parent.with_updates(**updates)


class ArchitectureMutator:
    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def mutate(self, parent: ArchitectureGenome, method: str, seed: int | None = None) -> ArchitectureGenome:
        return mutate_architecture(parent, method, self.seed if seed is None else int(seed))

    def mutate_many(
        self,
        parent: ArchitectureGenome,
        methods: Iterable[str] | None = None,
        *,
        start_seed: int | None = None,
    ) -> List[ArchitectureGenome]:
        base_seed = self.seed if start_seed is None else int(start_seed)
        return [
            mutate_architecture(parent, method, base_seed + index)
            for index, method in enumerate(methods or MUTATION_METHODS)
        ]
