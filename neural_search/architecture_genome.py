"""Explicit neural architecture genomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional


def stable_hash(payload: object, prefix: str = "") -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


@dataclass(frozen=True)
class ArchitectureGenome:
    """A bounded, buildable architecture description.

    The genome is intentionally explicit: residual depth, hidden width, memory
    gates, and causal bottlenecks are represented as fields rather than being
    hidden inside a fixed DNAX configuration.
    """

    input_dim: int = 4
    output_dim: int = 2
    hidden_dim: int = 16
    residual_blocks: int = 1
    memory_gates: int = 0
    causal_bottlenecks: int = 0
    object_binding_adapters: int = 0
    sequence_state_adapters: int = 0
    bottleneck_dim: Optional[int] = None
    activation: str = "tanh"
    dropout: float = 0.0
    parent_id: Optional[str] = None
    mutation_method: str = "initial"
    seed: int = 0
    genome_id: str = ""

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if self.residual_blocks < 0:
            raise ValueError("residual_blocks cannot be negative")
        if self.memory_gates < 0:
            raise ValueError("memory_gates cannot be negative")
        if self.causal_bottlenecks < 0:
            raise ValueError("causal_bottlenecks cannot be negative")
        if self.object_binding_adapters < 0:
            raise ValueError("object_binding_adapters cannot be negative")
        if self.sequence_state_adapters < 0:
            raise ValueError("sequence_state_adapters cannot be negative")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.bottleneck_dim is not None and self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive when provided")
        if not self.genome_id:
            object.__setattr__(self, "genome_id", stable_hash(self._identity_payload(), prefix="arch_"))

    @property
    def effective_bottleneck_dim(self) -> int:
        if self.bottleneck_dim is not None:
            return max(1, min(int(self.bottleneck_dim), self.hidden_dim))
        return max(1, self.hidden_dim // 2)

    @property
    def config_hash(self) -> str:
        return stable_hash(self._identity_payload())

    def _identity_payload(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("genome_id", None)
        return data

    def to_dict(self) -> Dict[str, object]:
        return {
            **self._identity_payload(),
            "genome_id": self.genome_id,
            "config_hash": self.config_hash,
            "effective_bottleneck_dim": self.effective_bottleneck_dim,
        }

    def with_updates(self, **updates: object) -> "ArchitectureGenome":
        updates.setdefault("parent_id", self.genome_id)
        updates.setdefault("genome_id", "")
        return replace(self, **updates)

    def build_module(self):
        from .module_library import build_architecture_module

        return build_architecture_module(self)
