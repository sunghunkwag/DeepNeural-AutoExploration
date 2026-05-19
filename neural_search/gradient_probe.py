"""Gradient probes over real model gradients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .task_families import MultiDomainTaskFamily


@dataclass(frozen=True)
class GradientProbeResult:
    genome_id: str
    gradient_norm_by_module: Dict[str, float]
    total_gradient_norm: float
    module_overdominance_score: float = 0.0
    recommended_mutations: Dict[str, List[str]] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "genome_id": self.genome_id,
            "gradient_norm_by_module": dict(self.gradient_norm_by_module),
            "total_gradient_norm": self.total_gradient_norm,
            "module_overdominance_score": self.module_overdominance_score,
            "recommended_mutations": {key: list(value) for key, value in self.recommended_mutations.items()},
            "failure_modes": list(self.failure_modes),
        }


def probe_gradients(
    model: torch.nn.Module,
    task_families: Sequence[MultiDomainTaskFamily],
    *,
    split: str = "train",
    genome_id: str = "",
) -> GradientProbeResult:
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    for family in task_families:
        batch = family.splits[split]
        losses.append(F.mse_loss(model(batch.x), batch.y))
    loss = sum(losses) / max(1, len(losses))
    loss.backward()
    norms: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        module_name = _module_name(name)
        norms[module_name] = norms.get(module_name, 0.0) + float(param.grad.detach().norm())
    total = float(sum(value * value for value in norms.values()) ** 0.5)
    dominance = _dominance(norms)
    failures = _failure_modes(norms, total, dominance)
    return GradientProbeResult(
        genome_id=genome_id,
        gradient_norm_by_module=norms,
        total_gradient_norm=total,
        module_overdominance_score=dominance,
        recommended_mutations={mode: _mutations_for_mode(mode) for mode in failures},
        failure_modes=failures,
    )


def gradient_failures_to_traces(
    result: GradientProbeResult,
    *,
    benchmark_name: str,
    seed: int,
    split_used: str = "validation",
) -> List[Dict[str, object]]:
    return [
        {
            "task_id": f"gradient-probe-{mode}",
            "benchmark_name": benchmark_name,
            "seed": int(seed),
            "failure_type": mode,
            "failed_operator_or_architecture": result.genome_id,
            "evidence_metrics": {
                "total_gradient_norm": result.total_gradient_norm,
                "min_module_gradient_norm": min(result.gradient_norm_by_module.values()) if result.gradient_norm_by_module else 0.0,
                "max_module_gradient_norm": max(result.gradient_norm_by_module.values()) if result.gradient_norm_by_module else 0.0,
                "module_overdominance_score": result.module_overdominance_score,
            },
            "proposed_operator_or_module_family": _module_family_for_mode(mode),
            "missing_representation_hypothesis": f"gradient probe detected {mode}; try {', '.join(result.recommended_mutations.get(mode, []))}",
            "split_used": split_used,
            "used_heldout_labels": False,
        }
        for mode in result.failure_modes
    ]


def _module_name(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    if len(parts) >= 2 and parts[0] in {"residual_blocks", "memory_gates", "causal_bottlenecks", "object_binding_adapters", "sequence_state_adapters"}:
        return ".".join(parts[:2])
    return parts[0]


def _failure_modes(norms: Dict[str, float], total: float, dominance: float) -> List[str]:
    modes: List[str] = []
    if total < 1e-6:
        modes.append("gradient bottleneck")
    if norms:
        values = [float(value) for value in norms.values()]
        if min(values) < 1e-8 and max(values) > 1e-5:
            modes.append("dead module")
        if max(values) / max(1e-8, min(value for value in values if value > 0.0) if any(value > 0 for value in values) else 1e-8) > 500.0:
            modes.append("gradient bottleneck")
        if dominance > 0.80 and len(values) > 1:
            modes.append("module over-dominance")
    return sorted(set(modes))


def _dominance(norms: Dict[str, float]) -> float:
    values = [abs(float(value)) for value in norms.values()]
    total = sum(values)
    return 0.0 if total <= 1e-12 else float(max(values) / total)


def _module_family_for_mode(mode: str) -> str:
    return {
        "gradient bottleneck": "wider_residual_stack",
        "dead module": "gated_memory_block",
        "module over-dominance": "gated_memory_block",
    }.get(mode, "wider_residual_stack")


def _mutations_for_mode(mode: str) -> List[str]:
    return {
        "gradient bottleneck": ["add_residual_block", "repair_bottleneck"],
        "dead module": ["add_memory_gate"],
        "module over-dominance": ["add_memory_gate", "widen_hidden_dim"],
    }.get(mode, ["add_residual_block"])
