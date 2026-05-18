"""Buildable PyTorch modules for architecture genomes."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .architecture_genome import ArchitectureGenome


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
        for bottleneck in self.causal_bottlenecks:
            h = bottleneck(h)
        return self.output_layer(h)


def build_architecture_module(genome: ArchitectureGenome) -> NeuralArchitectureModule:
    set_deterministic_seed(genome.seed)
    module = NeuralArchitectureModule(genome)
    module.train()
    return module
