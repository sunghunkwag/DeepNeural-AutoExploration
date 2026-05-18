"""Conservative progress metrics for AGI-relevant research signals."""

from __future__ import annotations

from statistics import mean
from typing import Dict, Mapping, Sequence


def compute_agi_relevance_metrics(payload: Mapping[str, object]) -> Dict[str, float]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    decisions = list(payload.get("architecture_decisions", payload.get("candidate_decisions", [])))  # type: ignore[arg-type]
    residues = list(payload.get("failure_residues", payload.get("residues", [])))  # type: ignore[arg-type]
    expansions = list(payload.get("search_space_expansions", []))  # type: ignore[arg-type]
    candidates = list(payload.get("candidate_genomes", []))  # type: ignore[arg-type]
    base_metrics = payload.get("base_metrics", {})
    final_metrics = payload.get("final_metrics", metrics)
    if not isinstance(base_metrics, Mapping):
        base_metrics = {}
    if not isinstance(final_metrics, Mapping):
        final_metrics = {}

    cross_domain = float(metrics.get("cross_domain_transfer_score", final_metrics.get("cross_domain_transfer_score", 0.0)))
    adaptation_speed = float(metrics.get("adaptation_speed", final_metrics.get("adaptation_speed", 0.0)))
    representation_reuse = _representation_reuse(payload)
    failure_recovery = _failure_recovery_score(decisions, residues)
    expansion_quality = _expansion_quality(expansions)
    hidden_robustness = _hidden_validation_robustness(decisions, final_metrics)
    heldout_delta = _heldout_delta(base_metrics, final_metrics)
    diversity = _architecture_diversity(candidates)
    policy_over_random = float(metrics.get("mutation_policy_improvement_over_random", 0.0))
    components = [
        cross_domain,
        adaptation_speed,
        representation_reuse,
        failure_recovery,
        expansion_quality,
        hidden_robustness,
        max(-1.0, min(1.0, heldout_delta)),
        diversity,
        max(-1.0, min(1.0, policy_over_random)),
    ]
    return {
        "cross_domain_transfer_score": cross_domain,
        "new_task_adaptation_speed": adaptation_speed,
        "representation_reuse_score": representation_reuse,
        "failure_recovery_score": failure_recovery,
        "search_space_expansion_quality": expansion_quality,
        "hidden_validation_robustness": hidden_robustness,
        "heldout_generalization_delta": heldout_delta,
        "architecture_diversity_score": diversity,
        "mutation_policy_improvement_over_random": policy_over_random,
        "cumulative_autonomous_improvement_score": float(mean(components)) if components else 0.0,
    }


def _representation_reuse(payload: Mapping[str, object]) -> float:
    probe = payload.get("representation_probe_results", payload.get("representation_probe", {}))
    if isinstance(probe, Sequence) and not isinstance(probe, (str, bytes, dict)):
        values = []
        for item in probe:
            if isinstance(item, Mapping):
                values.append(float(item.get("task_embedding_separability", 0.0)) / (1.0 + abs(float(item.get("hidden_state_similarity", 0.0)))))
        return float(mean(values)) if values else 0.0
    if isinstance(probe, Mapping):
        return float(probe.get("task_embedding_separability", 0.0)) / (1.0 + abs(float(probe.get("hidden_state_similarity", 0.0))))
    return 0.0


def _failure_recovery_score(decisions: Sequence[object], residues: Sequence[object]) -> float:
    accepted = sum(1 for item in decisions if isinstance(item, Mapping) and bool(item.get("accepted")))
    residue_count = len(residues)
    if residue_count == 0:
        return 0.0
    return float(min(1.0, accepted / max(1, residue_count)))


def _expansion_quality(expansions: Sequence[object]) -> float:
    if not expansions:
        return 0.0
    justified = 0
    for item in expansions:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("justification_count", 0)) > 0 or int(item.get("hidden_validation_failure_count", 0)) > 0:
            justified += 1
    return float(justified / max(1, len(expansions)))


def _hidden_validation_robustness(decisions: Sequence[object], final_metrics: Mapping[str, object]) -> float:
    improvements = [
        float(item.get("hidden_validation_improvement", 0.0))
        for item in decisions
        if isinstance(item, Mapping) and bool(item.get("accepted"))
    ]
    if improvements:
        return float(sum(1 for value in improvements if value >= 0.0) / len(improvements))
    hidden_loss = float(final_metrics.get("hidden_validation_loss", 0.0))
    validation_loss = float(final_metrics.get("validation_loss", 0.0))
    return 1.0 / (1.0 + max(0.0, hidden_loss - validation_loss))


def _heldout_delta(base_metrics: Mapping[str, object], final_metrics: Mapping[str, object]) -> float:
    base = float(base_metrics.get("heldout_test_loss", base_metrics.get("hidden_validation_loss", 0.0)))
    final = float(final_metrics.get("heldout_test_loss", final_metrics.get("hidden_validation_loss", 0.0)))
    return base - final


def _architecture_diversity(candidates: Sequence[object]) -> float:
    hashes = set()
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            hashes.add(str(candidate.get("config_hash", candidate.get("genome_id", ""))))
    if not candidates:
        return 0.0
    return float(len(hashes) / max(1, len(candidates)))
