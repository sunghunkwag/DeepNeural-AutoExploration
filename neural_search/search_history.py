"""Audit history for neural architecture search decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .architecture_genome import ArchitectureGenome


@dataclass(frozen=True)
class SearchHistoryRecord:
    parent_genome: Dict[str, object]
    candidate_genome: Dict[str, object]
    mutation_method: str
    validation_metrics: Dict[str, float]
    hidden_validation_metrics: Dict[str, float]
    accepted: bool
    failure_residue_ids: List[str]
    config_hash: str
    seed: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "parent_genome": dict(self.parent_genome),
            "candidate_genome": dict(self.candidate_genome),
            "mutation_method": self.mutation_method,
            "validation_metrics": dict(self.validation_metrics),
            "hidden_validation_metrics": dict(self.hidden_validation_metrics),
            "accepted": self.accepted,
            "failure_residue_ids": list(self.failure_residue_ids),
            "config_hash": self.config_hash,
            "seed": self.seed,
        }


class SearchHistory:
    def __init__(self) -> None:
        self.records: List[SearchHistoryRecord] = []

    def add(
        self,
        *,
        parent: ArchitectureGenome,
        candidate: ArchitectureGenome,
        mutation_method: str,
        validation_metrics: Dict[str, float],
        hidden_validation_metrics: Dict[str, float],
        accepted: bool,
        failure_residue_ids: Optional[Iterable[str]] = None,
        seed: int,
    ) -> SearchHistoryRecord:
        record = SearchHistoryRecord(
            parent_genome=parent.to_dict(),
            candidate_genome=candidate.to_dict(),
            mutation_method=mutation_method,
            validation_metrics=dict(validation_metrics),
            hidden_validation_metrics=dict(hidden_validation_metrics),
            accepted=bool(accepted),
            failure_residue_ids=list(failure_residue_ids or []),
            config_hash=candidate.config_hash,
            seed=int(seed),
        )
        self.records.append(record)
        return record

    def accepted_records(self) -> List[SearchHistoryRecord]:
        return [record for record in self.records if record.accepted]

    def rejected_records(self) -> List[SearchHistoryRecord]:
        return [record for record in self.records if not record.accepted]

    def best_method(self) -> str | None:
        accepted = self.accepted_records()
        if not accepted:
            return None
        def score(record: SearchHistoryRecord) -> float:
            return -float(record.validation_metrics.get("loss", 0.0)) - 0.5 * float(record.hidden_validation_metrics.get("loss", 0.0))
        return max(accepted, key=score).mutation_method

    def method_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for record in self.records:
            item = stats.setdefault(record.mutation_method, {"count": 0.0, "accepted": 0.0, "mean_validation_loss": 0.0})
            count = item["count"]
            item["mean_validation_loss"] = (item["mean_validation_loss"] * count + float(record.validation_metrics.get("loss", 0.0))) / (count + 1.0)
            item["count"] = count + 1.0
            item["accepted"] += 1.0 if record.accepted else 0.0
        return stats

    def to_dict(self) -> Dict[str, object]:
        return {
            "records": [record.to_dict() for record in self.records],
            "accepted_count": len(self.accepted_records()),
            "rejected_count": len(self.rejected_records()),
            "method_stats": self.method_stats(),
        }
