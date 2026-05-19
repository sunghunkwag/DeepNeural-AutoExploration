"""Evaluate whether meta-config memory improves repeated controller decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .meta_config_memory import MetaConfigMemory


@dataclass(frozen=True)
class MetaMemoryEffectivenessReport:
    reused_helpful_patterns_count: int
    avoided_harmful_patterns_count: int
    repeated_harmful_patterns_count: int
    memory_guidance_applied_count: int
    memory_guidance_helped_count: int
    memory_guidance_harmed_count: int
    memory_effectiveness_score: float
    recommendation: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "reused_helpful_patterns_count": self.reused_helpful_patterns_count,
            "avoided_harmful_patterns_count": self.avoided_harmful_patterns_count,
            "repeated_harmful_patterns_count": self.repeated_harmful_patterns_count,
            "memory_guidance_applied_count": self.memory_guidance_applied_count,
            "memory_guidance_helped_count": self.memory_guidance_helped_count,
            "memory_guidance_harmed_count": self.memory_guidance_harmed_count,
            "memory_effectiveness_score": self.memory_effectiveness_score,
            "recommendation": self.recommendation,
        }


def evaluate_meta_memory_effectiveness(
    memory: MetaConfigMemory | Mapping[str, object],
    meta_decisions: Sequence[Mapping[str, object]],
    quality_reports: Sequence[Mapping[str, object]],
) -> MetaMemoryEffectivenessReport:
    mem = memory if isinstance(memory, MetaConfigMemory) else MetaConfigMemory.from_dict(memory)
    reused = sum(len(_list(decision.get("reused_successful_config_patterns"))) for decision in meta_decisions)
    avoided = sum(len(_list(decision.get("avoided_harmful_config_patterns"))) for decision in meta_decisions)
    repeated_harmful = len(mem.repeatedly_harmful_paths(min_count=2))
    guidance_applied = sum(
        1
        for decision in meta_decisions
        if bool(decision.get("prior_meta_memory_used"))
        or _list(decision.get("reused_successful_config_patterns"))
        or _list(decision.get("avoided_harmful_config_patterns"))
    )
    helped = 0
    harmed = 0
    for index, decision in enumerate(meta_decisions):
        if not (
            bool(decision.get("prior_meta_memory_used"))
            or _list(decision.get("reused_successful_config_patterns"))
            or _list(decision.get("avoided_harmful_config_patterns"))
        ):
            continue
        quality = quality_reports[index] if index < len(quality_reports) else {}
        evidence = quality.get("evidence_used", {}) if isinstance(quality, Mapping) else {}
        components = evidence.get("process_improvement_components", {}) if isinstance(evidence, Mapping) else {}
        delta = _float(components.get("final_process_improvement_delta")) if isinstance(components, Mapping) else 0.0
        recommendation = str(quality.get("recommendation", "")) if isinstance(quality, Mapping) else ""
        if delta >= 0.0 and recommendation != "revert_previous_config":
            helped += 1
        elif delta < 0.0 or recommendation == "revert_previous_config":
            harmed += 1
    denominator = max(1, guidance_applied + repeated_harmful)
    score = max(0.0, min(1.0, (helped + 0.5 * avoided - harmed - 0.25 * repeated_harmful) / denominator))
    if score >= 0.65:
        recommendation = "continue_using_memory_guidance"
    elif score <= 0.25:
        recommendation = "revise_memory_policy"
    else:
        recommendation = "use_memory_selectively"
    return MetaMemoryEffectivenessReport(
        reused_helpful_patterns_count=int(reused),
        avoided_harmful_patterns_count=int(avoided),
        repeated_harmful_patterns_count=int(repeated_harmful),
        memory_guidance_applied_count=int(guidance_applied),
        memory_guidance_helped_count=int(helped),
        memory_guidance_harmed_count=int(harmed),
        memory_effectiveness_score=float(score),
        recommendation=recommendation,
    )


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
