"""Representation probes over actual model activations."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from .activation_geometry import (
    activation_variance,
    activation_saturation,
    centroid_separability,
    flatten_activation,
    mean_pairwise_cosine_similarity,
    module_overdominance_score,
    representation_collapse_score,
)
from .task_families import MultiDomainBatch, MultiDomainTaskFamily


@dataclass(frozen=True)
class RepresentationProbeResult:
    genome_id: str
    activation_variance_by_module: Dict[str, float]
    representation_collapse_score: float
    task_embedding_separability: float
    hidden_state_similarity: float
    module_ablation_contribution: Dict[str, float]
    hidden_state_saturation: float = 0.0
    module_overdominance_score: float = 0.0
    validation_hidden_mismatch: float = 0.0
    failure_modes: List[str] = field(default_factory=list)
    recommended_mutations: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "genome_id": self.genome_id,
            "activation_variance_by_module": dict(self.activation_variance_by_module),
            "representation_collapse_score": self.representation_collapse_score,
            "task_embedding_separability": self.task_embedding_separability,
            "hidden_state_similarity": self.hidden_state_similarity,
            "hidden_state_saturation": self.hidden_state_saturation,
            "module_overdominance_score": self.module_overdominance_score,
            "validation_hidden_mismatch": self.validation_hidden_mismatch,
            "module_ablation_contribution": dict(self.module_ablation_contribution),
            "failure_modes": list(self.failure_modes),
            "recommended_mutations": {key: list(value) for key, value in self.recommended_mutations.items()},
        }


def probe_representations(
    model: torch.nn.Module,
    task_families: Sequence[MultiDomainTaskFamily],
    *,
    split: str = "validation",
    genome_id: str = "",
    max_ablation_modules: int = 6,
    validation_hidden_mismatch: float = 0.0,
) -> RepresentationProbeResult:
    model.eval()
    captured: Dict[str, List[torch.Tensor]] = {}
    hooks = []
    for name, module in _probe_modules(model):
        hooks.append(module.register_forward_hook(_capture_hook(name, captured)))
    with torch.no_grad():
        for family in task_families:
            batch = family.splits[split]
            model(batch.x)
    for hook in hooks:
        hook.remove()
    variances = {
        name: activation_variance(torch.cat(values, dim=0))
        for name, values in captured.items()
        if values
    }
    saturations = [activation_saturation(torch.cat(values, dim=0)) for values in captured.values() if values]
    saturation = float(mean(saturations)) if saturations else 0.0
    family_embeddings = _family_embeddings(model, task_families, split=split)
    separability = centroid_separability(family_embeddings)
    similarity = mean_pairwise_cosine_similarity(family_embeddings)
    collapse = representation_collapse_score(variances)
    ablations = module_ablation_contribution(
        model,
        [family.splits[split] for family in task_families],
        max_modules=max_ablation_modules,
    )
    dominance = module_overdominance_score(ablations)
    failure_modes = _failure_modes(variances, collapse, separability, similarity, ablations, saturation, dominance, validation_hidden_mismatch)
    recommendations = {mode: _mutations_for_mode(mode) for mode in failure_modes}
    return RepresentationProbeResult(
        genome_id=genome_id,
        activation_variance_by_module=variances,
        representation_collapse_score=collapse,
        task_embedding_separability=separability,
        hidden_state_similarity=similarity,
        hidden_state_saturation=saturation,
        module_overdominance_score=dominance,
        validation_hidden_mismatch=float(validation_hidden_mismatch),
        module_ablation_contribution=ablations,
        failure_modes=failure_modes,
        recommended_mutations=recommendations,
    )


def representation_failures_to_traces(
    result: RepresentationProbeResult,
    *,
    benchmark_name: str,
    seed: int,
    split_used: str = "validation",
) -> List[Dict[str, object]]:
    return [
        {
            "task_id": f"representation-probe-{mode}",
            "benchmark_name": benchmark_name,
            "seed": int(seed),
            "failure_type": mode,
            "failed_operator_or_architecture": result.genome_id,
            "evidence_metrics": {
                "representation_collapse_score": result.representation_collapse_score,
                "task_embedding_separability": result.task_embedding_separability,
                "hidden_state_similarity": result.hidden_state_similarity,
                "hidden_state_saturation": result.hidden_state_saturation,
                "module_overdominance_score": result.module_overdominance_score,
                "validation_hidden_mismatch": result.validation_hidden_mismatch,
            },
            "proposed_operator_or_module_family": _module_family_for_mode(mode),
            "missing_representation_hypothesis": f"probe detected {mode}; try {', '.join(result.recommended_mutations.get(mode, []))}",
            "split_used": split_used,
            "used_heldout_labels": False,
        }
        for mode in result.failure_modes
    ]


def module_ablation_contribution(
    model: torch.nn.Module,
    batches: Sequence[MultiDomainBatch],
    *,
    max_modules: int = 6,
) -> Dict[str, float]:
    baseline = _mean_loss(model, batches)
    contributions: Dict[str, float] = {}
    for name, module in _probe_modules(model)[:max_modules]:
        def zero_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                return torch.zeros_like(output)
            return output

        hook = module.register_forward_hook(zero_hook)
        try:
            ablated = _mean_loss(model, batches)
            contributions[name] = float(ablated - baseline)
        finally:
            hook.remove()
    return contributions


def _capture_hook(name: str, captured: Dict[str, List[torch.Tensor]]):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            captured.setdefault(name, []).append(flatten_activation(output).detach().cpu())
    return hook


def _probe_modules(model: torch.nn.Module):
    module_names = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        if len(list(module.children())) == 0 or module.__class__.__name__ in {
            "ResidualMLPBlock",
            "MemoryGate",
            "CausalBottleneck",
            "ObjectBindingAdapter",
            "SequenceStateAdapter",
        }:
            module_names.append((name, module))
    return module_names


def _family_embeddings(
    model: torch.nn.Module,
    task_families: Sequence[MultiDomainTaskFamily],
    *,
    split: str,
) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}
    target_module = getattr(model, "output_layer", None)
    if target_module is None:
        return embeddings
    for family in task_families:
        captured: List[torch.Tensor] = []
        hook = target_module.register_forward_hook(lambda _m, inputs, _o: captured.append(flatten_activation(inputs[0]).detach().cpu()))
        with torch.no_grad():
            model(family.splits[split].x)
        hook.remove()
        if captured:
            embeddings[family.family] = torch.cat(captured, dim=0)
    return embeddings


def _mean_loss(model: torch.nn.Module, batches: Iterable[MultiDomainBatch]) -> float:
    values = []
    with torch.no_grad():
        for batch in batches:
            values.append(float(F.mse_loss(model(batch.x), batch.y)))
    return float(mean(values)) if values else 0.0


def _failure_modes(
    variances: Dict[str, float],
    collapse: float,
    separability: float,
    similarity: float,
    ablations: Dict[str, float],
    saturation: float,
    dominance: float,
    validation_hidden_mismatch: float,
) -> List[str]:
    modes: List[str] = []
    if collapse > 0.98 or (variances and max(variances.values()) < 1e-4):
        modes.append("representation collapse")
    if separability < 0.03:
        modes.append("task-family entanglement")
    if similarity > 0.98:
        modes.append("unstable hidden dynamics")
    if saturation > 0.45:
        modes.append("hidden-state saturation")
    if any(abs(value) < 1e-7 for value in ablations.values()):
        modes.append("dead module")
    if dominance > 0.75 and len(ablations) > 1:
        modes.append("module over-dominance")
    if validation_hidden_mismatch > 0.20:
        modes.append("high validation-hidden-validation mismatch")
    if any("causal_bottlenecks" in name and value < 1e-5 for name, value in variances.items()):
        modes.append("over-compressed bottleneck")
    return sorted(set(modes))


def _module_family_for_mode(mode: str) -> str:
    return {
        "representation collapse": "wider_residual_stack",
        "task-family entanglement": "object_binding_adapter",
        "unstable hidden dynamics": "sequence_state_adapter",
        "dead module": "gated_memory_block",
        "over-compressed bottleneck": "causal_bottleneck_block",
        "hidden-state saturation": "wider_residual_stack",
        "module over-dominance": "gated_memory_block",
        "high validation-hidden-validation mismatch": "causal_bottleneck_block",
    }.get(mode, "wider_residual_stack")


def _mutations_for_mode(mode: str) -> List[str]:
    return {
        "representation collapse": ["wider_residual_stack", "widen_hidden_dim"],
        "task-family entanglement": ["object_binding_adapter", "add_causal_bottleneck"],
        "unstable hidden dynamics": ["sequence_state_adapter"],
        "hidden-state saturation": ["widen_hidden_dim", "add_residual_block"],
        "dead module": ["add_memory_gate"],
        "module over-dominance": ["add_memory_gate", "remove_residual_block"],
        "over-compressed bottleneck": ["repair_bottleneck"],
        "high validation-hidden-validation mismatch": ["add_causal_bottleneck", "object_binding_adapter"],
    }.get(mode, ["add_residual_block"])
