"""Bounded JSON memory for meta-meta-meta config mutations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

from .architecture_genome import stable_hash


@dataclass(frozen=True)
class MetaConfigMemoryRecord:
    meta_decision_id: str
    next_run_config_hash: str
    applied_config: Dict[str, object]
    attribution_report: Dict[str, object]
    quality_report: Dict[str, object]
    process_improvement_delta: float
    helped_changes: list[str]
    harmed_changes: list[str]
    recommendation: str
    seed: int
    mode: str
    timestamp: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "meta_decision_id": self.meta_decision_id,
            "next_run_config_hash": self.next_run_config_hash,
            "applied_config": dict(self.applied_config),
            "attribution_report": dict(self.attribution_report),
            "quality_report": dict(self.quality_report),
            "process_improvement_delta": self.process_improvement_delta,
            "helped_changes": list(self.helped_changes),
            "harmed_changes": list(self.harmed_changes),
            "recommendation": self.recommendation,
            "seed": self.seed,
            "mode": self.mode,
            "timestamp": self.timestamp,
        }


@dataclass
class MetaConfigMemory:
    records: list[MetaConfigMemoryRecord] = field(default_factory=list)
    max_records: int = 20

    def add_record(self, record: MetaConfigMemoryRecord | Mapping[str, object]) -> None:
        item = record if isinstance(record, MetaConfigMemoryRecord) else _record_from_mapping(record)
        self.records.append(item)
        self.records = self.records[-max(1, int(self.max_records)) :]

    def add_decision(
        self,
        *,
        meta_decision_id: str,
        next_run_config_hash: str,
        applied_config: Mapping[str, object],
        attribution_report: Mapping[str, object],
        quality_report: Mapping[str, object],
        process_improvement_delta: float,
        helped_changes: list[str],
        harmed_changes: list[str],
        recommendation: str,
        seed: int,
        mode: str,
        timestamp: str | None = None,
    ) -> None:
        self.add_record(
            MetaConfigMemoryRecord(
                meta_decision_id=meta_decision_id,
                next_run_config_hash=next_run_config_hash,
                applied_config=dict(applied_config),
                attribution_report=dict(attribution_report),
                quality_report=dict(quality_report),
                process_improvement_delta=float(process_improvement_delta),
                helped_changes=list(helped_changes),
                harmed_changes=list(harmed_changes),
                recommendation=recommendation,
                seed=int(seed),
                mode=str(mode),
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            )
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_records": self.max_records,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "MetaConfigMemory":
        if not payload:
            return cls()
        memory = cls(max_records=max(1, int(payload.get("max_records", 20))))
        raw_records = payload.get("records", [])
        if isinstance(raw_records, list):
            for item in raw_records:
                if isinstance(item, Mapping):
                    memory.add_record(item)
        return memory

    @classmethod
    def load(cls, path: str | Path, *, max_records: int = 20) -> "MetaConfigMemory":
        p = Path(path)
        if not p.exists():
            return cls(max_records=max_records)
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return cls(max_records=max_records)
        memory = cls.from_dict(payload)
        memory.max_records = max_records
        memory.records = memory.records[-max(1, max_records) :]
        return memory

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def helpful_change_paths(self, section_prefix: str = "", *, min_count: int = 1) -> list[str]:
        return _paths_with_min_count(self.records, "helped_changes", section_prefix, min_count)

    def harmful_change_paths(self, section_prefix: str = "", *, min_count: int = 1) -> list[str]:
        return _paths_with_min_count(self.records, "harmed_changes", section_prefix, min_count)

    def repeatedly_harmful_paths(self, *, min_count: int = 2) -> list[str]:
        return self.harmful_change_paths("", min_count=min_count)

    def rollback_correlated_harmful_paths(self, section_prefix: str = "", *, min_count: int = 1) -> list[str]:
        counts: Dict[str, int] = {}
        for record in self.records:
            attr = record.attribution_report
            if not isinstance(attr, Mapping):
                continue
            components = attr.get("process_improvement_components", {})
            transition_rollback_delta = _float((components or {}).get("rollback_rate_delta")) if isinstance(components, Mapping) else 0.0
            changes = attr.get("changes", [])
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                path = str(change.get("config_path", ""))
                if section_prefix and not path.startswith(section_prefix):
                    continue
                effect = change.get("observed_effect", {})
                rollback_delta = _float(effect.get("rollback_rate_delta")) if isinstance(effect, Mapping) else 0.0
                if rollback_delta > 0.0 or (transition_rollback_delta > 0.0 and bool(change.get("harmed"))):
                    counts[path] = counts.get(path, 0) + 1
        return sorted(path for path, count in counts.items() if count >= min_count)

    def repeatedly_neutral_paths(self, section_prefix: str = "", *, min_count: int = 2) -> list[str]:
        counts: Dict[str, int] = {}
        for record in self.records:
            attr = record.attribution_report
            neutral = attr.get("neutral_changes", [])
            if not isinstance(neutral, list):
                continue
            for path in neutral:
                path_str = str(path)
                if not section_prefix or path_str.startswith(section_prefix):
                    counts[path_str] = counts.get(path_str, 0) + 1
        return sorted(path for path, count in counts.items() if count >= min_count)

    def latest_recommendation(self) -> str:
        return self.records[-1].recommendation if self.records else ""


def next_run_config_hash(config: Mapping[str, object]) -> str:
    return stable_hash(dict(config), prefix="cfg_")


def apply_memory_guidance_to_config(
    section_prefix: str,
    base_config: Mapping[str, object],
    proposed_config: Mapping[str, object],
    memory: MetaConfigMemory | None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    if memory is None or not memory.records:
        return dict(proposed_config), {
            "reused_successful_config_patterns": [],
            "avoided_harmful_config_patterns": [],
            "avoided_rollback_correlated_patterns": [],
            "lower_confidence_neutral_patterns": [],
        }
    adjusted = _deep_copy_mapping(proposed_config)
    helpful = set(memory.helpful_change_paths(section_prefix))
    harmful = set(memory.harmful_change_paths(section_prefix))
    rollback_harmful = set(memory.rollback_correlated_harmful_paths(section_prefix))
    neutral = set(memory.repeatedly_neutral_paths(section_prefix))
    changed_paths = [path for path, old, new in _diff_nested(base_config, proposed_config, section_prefix) if old != new]
    avoided: list[str] = []
    avoided_rollback: list[str] = []
    reused: list[str] = []
    lowered: list[str] = []
    for path in changed_paths:
        if path in harmful or path in rollback_harmful:
            _set_path(adjusted, path[len(section_prefix) + 1 :], _get_path(base_config, path[len(section_prefix) + 1 :]))
            if path in harmful:
                avoided.append(path)
            if path in rollback_harmful:
                avoided_rollback.append(path)
        elif path in helpful:
            reused.append(path)
        elif path in neutral:
            lowered.append(path)
    return adjusted, {
        "reused_successful_config_patterns": sorted(reused),
        "avoided_harmful_config_patterns": sorted(avoided),
        "avoided_rollback_correlated_patterns": sorted(avoided_rollback),
        "lower_confidence_neutral_patterns": sorted(lowered),
    }


def _record_from_mapping(payload: Mapping[str, object]) -> MetaConfigMemoryRecord:
    return MetaConfigMemoryRecord(
        meta_decision_id=str(payload.get("meta_decision_id", "")),
        next_run_config_hash=str(payload.get("next_run_config_hash", "")),
        applied_config=dict(payload.get("applied_config", {})) if isinstance(payload.get("applied_config", {}), Mapping) else {},
        attribution_report=dict(payload.get("attribution_report", {})) if isinstance(payload.get("attribution_report", {}), Mapping) else {},
        quality_report=dict(payload.get("quality_report", {})) if isinstance(payload.get("quality_report", {}), Mapping) else {},
        process_improvement_delta=float(payload.get("process_improvement_delta", 0.0)) if isinstance(payload.get("process_improvement_delta", 0.0), (int, float)) else 0.0,
        helped_changes=[str(item) for item in payload.get("helped_changes", [])] if isinstance(payload.get("helped_changes", []), list) else [],
        harmed_changes=[str(item) for item in payload.get("harmed_changes", [])] if isinstance(payload.get("harmed_changes", []), list) else [],
        recommendation=str(payload.get("recommendation", "")),
        seed=int(payload.get("seed", 0)) if isinstance(payload.get("seed", 0), int) else 0,
        mode=str(payload.get("mode", "")),
        timestamp=str(payload.get("timestamp", "")),
    )


def _paths_with_min_count(records: list[MetaConfigMemoryRecord], field_name: str, section_prefix: str, min_count: int) -> list[str]:
    counts: Dict[str, int] = {}
    for record in records:
        values = getattr(record, field_name)
        for path in values:
            if not section_prefix or path.startswith(section_prefix):
                counts[path] = counts.get(path, 0) + 1
    return sorted(path for path, count in counts.items() if count >= min_count)


def _deep_copy_mapping(value: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, raw in value.items():
        if isinstance(raw, Mapping):
            out[str(key)] = _deep_copy_mapping(raw)
        elif isinstance(raw, list):
            out[str(key)] = list(raw)
        else:
            out[str(key)] = raw
    return out


def _diff_nested(old: Mapping[str, object], new: Mapping[str, object], prefix: str) -> list[tuple[str, object, object]]:
    changes: list[tuple[str, object, object]] = []
    for key in sorted(set(old) | set(new)):
        path = f"{prefix}.{key}"
        old_value = old.get(key)
        new_value = new.get(key)
        if isinstance(old_value, Mapping) and isinstance(new_value, Mapping):
            changes.extend(_diff_nested(old_value, new_value, path))
        elif old_value != new_value:
            changes.append((path, old_value, new_value))
    return changes


def _set_path(config: Dict[str, object], path: str, value: object) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        return
    cursor: Dict[str, object] = config
    for part in parts[:-1]:
        raw = cursor.get(part)
        if not isinstance(raw, dict):
            raw = {}
            cursor[part] = raw
        cursor = raw
    cursor[parts[-1]] = value


def _get_path(config: Mapping[str, object], path: str) -> object:
    cursor: object = config
    for part in [part for part in path.split(".") if part]:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(part)
    return cursor


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
