"""Buildable PyTorch modules for architecture genomes."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .architecture_genome import ArchitectureGenome


BOUNDED_MODULE_FAMILIES = (
    "gated_memory_block",
    "causal_bottleneck_block",
    "wider_residual_stack",
    "object_binding_adapter",
    "sequence_state_adapter",
)


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    lowered = name.lower()
    if lowered == "relu":
        return torch.relu
    if lowered in {"gelu", "gel"}:
        return torch.nn.functional.gelu
    if lowered in {"silu", "swish"}:
        return torch.nn.functional.silu
    if lowered == "tanh":
        return torch.tanh
    raise ValueError(f"unsupported activation: {name}")


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, activation: str, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        return self.activation(residual + self.dropout(out))


class MemoryGate(nn.Module):
    """A small gated state update over the hidden representation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        candidate = torch.tanh(self.candidate(x))
        return gate * candidate + (1.0 - gate) * x


class ObjectBindingAdapter(nn.Module):
    """Bounded pairwise feature mixer for object/relation-like task features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = max(1.0, float(q.shape[-1]) ** 0.5)
        if x.ndim == 2:
            scores = torch.softmax((q @ k.transpose(0, 1)) / scale, dim=-1)
            mixed = scores @ v
        else:
            scores = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
            mixed = scores @ v
        return x + self.output(mixed)


class SequenceStateAdapter(nn.Module):
    """Small recurrent-style state adapter over feature order within a batch."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.mix = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[0] <= 1:
            return x + torch.tanh(self.mix(x))
        state = torch.zeros_like(x[0])
        outputs = []
        for row in x:
            state = self.update(row, state)
            outputs.append(state)
        stacked = torch.stack(outputs, dim=0)
        return x + torch.tanh(self.mix(stacked))


class CausalBottleneck(nn.Module):
    """A bounded bottleneck intended to pressure compact causal features."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, hidden_dim)
        self.latent_gate = nn.Parameter(torch.zeros(bottleneck_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(self.encoder(x)) * torch.sigmoid(self.latent_gate)
        return x + self.decoder(latent)


class NeuralArchitectureModule(nn.Module):
    """Executable module produced from an ArchitectureGenome."""

    def __init__(self, genome: ArchitectureGenome):
        super().__init__()
        self.genome = genome
        self.input_layer = nn.Linear(genome.input_dim, genome.hidden_dim)
        self.residual_blocks = nn.ModuleList(
            ResidualMLPBlock(genome.hidden_dim, genome.activation, genome.dropout)
            for _ in range(genome.residual_blocks)
        )
        self.memory_gates = nn.ModuleList(MemoryGate(genome.hidden_dim) for _ in range(genome.memory_gates))
        self.object_binding_adapters = nn.ModuleList(
            ObjectBindingAdapter(genome.hidden_dim) for _ in range(genome.object_binding_adapters)
        )
        self.sequence_state_adapters = nn.ModuleList(
            SequenceStateAdapter(genome.hidden_dim) for _ in range(genome.sequence_state_adapters)
        )
        self.causal_bottlenecks = nn.ModuleList(
            CausalBottleneck(genome.hidden_dim, genome.effective_bottleneck_dim)
            for _ in range(genome.causal_bottlenecks)
        )
        self.output_layer = nn.Linear(genome.hidden_dim, genome.output_dim)
        self.activation = _activation(genome.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_layer(x.float()))
        for index, block in enumerate(self.residual_blocks):
            h = block(h)
            if index < len(self.memory_gates):
                h = self.memory_gates[index](h)
        for index in range(len(self.residual_blocks), len(self.memory_gates)):
            h = self.memory_gates[index](h)
        for adapter in self.object_binding_adapters:
            h = adapter(h)
        for adapter in self.sequence_state_adapters:
            h = adapter(h)
        for bottleneck in self.causal_bottlenecks:
            h = bottleneck(h)
        return self.output_layer(h)


def build_architecture_module(genome: ArchitectureGenome) -> NeuralArchitectureModule:
    set_deterministic_seed(genome.seed)
    module = NeuralArchitectureModule(genome)
    module.train()
    return module
