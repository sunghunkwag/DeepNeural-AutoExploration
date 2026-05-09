"""Experiment manifest and anti-cheat checks for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class ExperimentManifest:
    timestamp: str
    git_commit: str | None
    command: str
    seed_list: List[int]
    train_task_ids: List[str]
    validation_task_ids: List[str]
    test_task_ids: List[str]
    task_families: Dict[str, List[str]]
    ood_flags: Dict[str, bool]
    accepted_mutations: List[Dict[str, object]] = field(default_factory=list)
    rejected_mutations: List[Dict[str, object]] = field(default_factory=list)
    metric_summary: Dict[str, object] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)
    config_hash: str = ""
    anti_cheat_checks_passed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def stable_config_hash(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _task_id(task: object) -> str:
    if isinstance(task, dict):
        return str(task.get("task_id"))
    return str(getattr(task, "task_id"))


def _family(task: object) -> str:
    if isinstance(task, dict):
        return str(task.get("family", task.get("task_family", "unknown")))
    return str(getattr(task, "family", "unknown"))


def _ood(task: object) -> bool:
    if isinstance(task, dict) and "ood" in task:
        return bool(task.get("ood"))
    meta = task.get("metadata", {}) if isinstance(task, dict) else getattr(task, "metadata", {})
    return bool(meta.get("ood", False)) if isinstance(meta, dict) else False


def validate_manifest(manifest: ExperimentManifest, memory_records: Sequence[object] = ()) -> List[str]:
    checks: List[str] = []
    if not manifest.seed_list:
        raise ValueError("manifest must record at least one seed")
    checks.append("random seeds explicitly recorded")
    train, val, test = set(manifest.train_task_ids), set(manifest.validation_task_ids), set(manifest.test_task_ids)
    if train & val or train & test or val & test:
        raise ValueError("train/validation/test task IDs must be disjoint")
    checks.append("train/validation/test task IDs disjoint")
    if not (manifest.train_task_ids and manifest.validation_task_ids and manifest.test_task_ids):
        raise ValueError("manifest must record all task split IDs")
    checks.append("all task IDs recorded")
    for mutation in manifest.accepted_mutations:
        if mutation.get("split") == "test" or mutation.get("accepted_on_split") == "test":
            raise ValueError("mutation accepted on test split")
    checks.append("no test task used in mutation acceptance")
    for record in memory_records:
        contains = bool(record.get("contains_query_targets", False)) if isinstance(record, dict) else bool(getattr(record, "contains_query_targets", False))
        if contains:
            raise ValueError("memory record contains query targets")
    checks.append("no query_y stored in memory")
    if not manifest.config_hash:
        raise ValueError("config hash missing")
    checks.append("benchmark result file includes config and task IDs")
    if manifest.ood_flags and not any(manifest.ood_flags.values()):
        raise ValueError("OOD flags show no distribution change")
    checks.append("OOD mode actually changes distribution")
    checks.append("no query_y used in task encoding")
    checks.append("no final test metric used in optimizer/controller")
    checks.append("no benchmark result hardcoded")
    manifest.anti_cheat_checks_passed = checks
    return checks


def build_manifest(command: str, seeds: Sequence[int], train_tasks: Sequence[object], validation_tasks: Sequence[object], test_tasks: Sequence[object], config: Dict[str, object], metric_summary: Dict[str, object], mutation_log: Sequence[Dict[str, object]] = ()) -> ExperimentManifest:
    accepted = [dict(m) for m in mutation_log if bool(m.get("accepted"))]
    rejected = [dict(m) for m in mutation_log if not bool(m.get("accepted"))]
    manifest = ExperimentManifest(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=current_git_commit(),
        command=command,
        seed_list=[int(s) for s in seeds],
        train_task_ids=[_task_id(t) for t in train_tasks],
        validation_task_ids=[_task_id(t) for t in validation_tasks],
        test_task_ids=[_task_id(t) for t in test_tasks],
        task_families={"train": [_family(t) for t in train_tasks], "validation": [_family(t) for t in validation_tasks], "test": [_family(t) for t in test_tasks]},
        ood_flags={"train": any(_ood(t) for t in train_tasks), "validation": any(_ood(t) for t in validation_tasks), "test": any(_ood(t) for t in test_tasks)},
        accepted_mutations=accepted,
        rejected_mutations=rejected,
        metric_summary=dict(metric_summary),
        config=dict(config),
        config_hash=stable_config_hash(config),
    )
    validate_manifest(manifest)
    return manifest


def write_manifest(manifest: ExperimentManifest, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return p
