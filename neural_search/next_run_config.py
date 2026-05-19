"""Safe bounded next-run configuration for autonomous neural exploration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

from .curriculum_strategy_mutator import CurriculumStrategyConfig
from .evaluator_objective_mutator import EvaluatorObjectiveConfig
from .search_strategy_mutator import SearchStrategyConfig


@dataclass(frozen=True)
class NextRunConfig:
    search_strategy_config: SearchStrategyConfig = field(default_factory=SearchStrategyConfig)
    evaluator_objective_config: EvaluatorObjectiveConfig = field(default_factory=EvaluatorObjectiveConfig)
    curriculum_strategy_config: CurriculumStrategyConfig = field(default_factory=CurriculumStrategyConfig)
    recommended_mode: str = "smoke"
    recommended_seed_policy: str = "same_seed"
    recommended_generations: int = 1
    recommended_candidate_count: int = 3
    recommended_focus_failure_types: list[str] = field(default_factory=list)
    recommended_focus_task_families: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "search_strategy_config": self.search_strategy_config.to_dict(),
            "evaluator_objective_config": self.evaluator_objective_config.to_dict(),
            "curriculum_strategy_config": self.curriculum_strategy_config.to_dict(),
            "recommended_mode": self.recommended_mode,
            "recommended_seed_policy": self.recommended_seed_policy,
            "recommended_generations": self.recommended_generations,
            "recommended_candidate_count": self.recommended_candidate_count,
            "recommended_focus_failure_types": list(self.recommended_focus_failure_types),
            "recommended_focus_task_families": list(self.recommended_focus_task_families),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NextRunConfig":
        if not payload:
            return cls()
        mode = str(payload.get("recommended_mode", "smoke"))
        if mode not in {"smoke", "quick"}:
            mode = "smoke"
        seed_policy = str(payload.get("recommended_seed_policy", "same_seed"))
        if seed_policy not in {"same_seed", "next_seed"}:
            seed_policy = "same_seed"
        return cls(
            search_strategy_config=SearchStrategyConfig.from_dict(_mapping(payload.get("search_strategy_config"))),
            evaluator_objective_config=EvaluatorObjectiveConfig.from_dict(_mapping(payload.get("evaluator_objective_config"))),
            curriculum_strategy_config=CurriculumStrategyConfig.from_dict(_mapping(payload.get("curriculum_strategy_config"))),
            recommended_mode=mode,
            recommended_seed_policy=seed_policy,
            recommended_generations=max(1, min(4, int(payload.get("recommended_generations", 1)))),
            recommended_candidate_count=max(1, min(6, int(payload.get("recommended_candidate_count", 3)))),
            recommended_focus_failure_types=_string_list(payload.get("recommended_focus_failure_types"), max_len=12),
            recommended_focus_task_families=_string_list(payload.get("recommended_focus_task_families"), max_len=5),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NextRunConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("next-run config must be a JSON object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _string_list(value: object, *, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item[:80])
        if len(out) >= max_len:
            break
    return out

