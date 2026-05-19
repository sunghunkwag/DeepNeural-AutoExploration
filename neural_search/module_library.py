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
    """Read/write-style gated memory block over the hidden representation."""

    def __init__(self, hidden_dim: int, memory_slots: int = 2):
        super().__init__()
        self.memory_slots = max(1, int(memory_slots))
        self.memory_bank = nn.Parameter(torch.randn(self.memory_slots, hidden_dim) * 0.02)
        self.read_gate = nn.Linear(hidden_dim, self.memory_slots)
        self.write_gate = nn.Linear(hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.last_read_weights: torch.Tensor | None = None
        self.last_write_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_weights = torch.softmax(self.read_gate(x), dim=-1)
        memory_read = read_weights @ self.memory_bank
        write_gate = torch.sigmoid(self.write_gate(x))
        candidate = torch.tanh(self.candidate(torch.cat([x, memory_read], dim=-1)))
        self.last_read_weights = read_weights.detach()
        self.last_write_gate = write_gate.detach()
        return write_gate * candidate + (1.0 - write_gate) * x


class ObjectBindingAdapter(nn.Module):
    """Bounded pairwise feature mixer for object/relation-like task features."""

    def __init__(self, hidden_dim: int, object_slots: int = 2):
        super().__init__()
        self.object_slots = max(1, int(object_slots))
        self.slot_dim = (hidden_dim + self.object_slots - 1) // self.object_slots
        padded_dim = self.slot_dim * self.object_slots
        self.pad_dim = padded_dim - hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._to_slots(self.query(x))
        k = self._to_slots(self.key(x))
        v = self._to_slots(self.value(x))
        scale = max(1.0, float(self.slot_dim) ** 0.5)
        attention = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
        mixed = (attention @ v).reshape(x.shape[0], -1)
        if self.pad_dim:
            mixed = mixed[:, :-self.pad_dim]
        self.last_attention = attention.detach()
        return x + self.output(mixed)

    def _to_slots(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_dim:
            x = torch.nn.functional.pad(x, (0, self.pad_dim))
        return x.reshape(x.shape[0], self.object_slots, self.slot_dim)


class SequenceStateAdapter(nn.Module):
    """Small recurrent-style state adapter over feature order within a batch."""

    def __init__(self, hidden_dim: int, sequence_chunks: int = 4):
        super().__init__()
        self.sequence_chunks = max(1, int(sequence_chunks))
        self.chunk_dim = (hidden_dim + self.sequence_chunks - 1) // self.sequence_chunks
        padded_dim = self.chunk_dim * self.sequence_chunks
        self.pad_dim = padded_dim - hidden_dim
        self.update = nn.GRUCell(self.chunk_dim, self.chunk_dim)
        self.mix = nn.Linear(padded_dim, hidden_dim)
        self.last_state_norm: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(x, (0, self.pad_dim)) if self.pad_dim else x
        chunks = padded.reshape(x.shape[0], self.sequence_chunks, self.chunk_dim)
        state = torch.zeros(x.shape[0], self.chunk_dim, dtype=x.dtype, device=x.device)
        outputs = []
        for index in range(self.sequence_chunks):
            state = self.update(chunks[:, index, :], state)
            outputs.append(state)
        stacked = torch.stack(outputs, dim=1).reshape(x.shape[0], -1)
        self.last_state_norm = float(stacked.detach().norm(dim=-1).mean())
        return x + torch.tanh(self.mix(stacked))


class CausalBottleneck(nn.Module):
    """A bounded bottleneck intended to pressure compact causal features."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, hidden_dim)
        self.latent_gate = nn.Parameter(torch.zeros(bottleneck_dim))
        self.last_latent: torch.Tensor | None = None
        self.last_usage: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(self.encoder(x)) * torch.sigmoid(self.latent_gate)
        self.last_latent = latent.detach()
        self.last_usage = float(latent.detach().abs().mean())
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
        self.memory_gates = nn.ModuleList(MemoryGate(genome.hidden_dim, genome.memory_slots) for _ in range(genome.memory_gates))
        self.object_binding_adapters = nn.ModuleList(
            ObjectBindingAdapter(genome.hidden_dim, genome.object_slots) for _ in range(genome.object_binding_adapters)
        )
        self.sequence_state_adapters = nn.ModuleList(
            SequenceStateAdapter(genome.hidden_dim, genome.sequence_chunks) for _ in range(genome.sequence_state_adapters)
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
