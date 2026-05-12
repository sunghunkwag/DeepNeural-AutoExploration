"""Manifest and anti-cheat checks for interaction scaffold benchmarks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


REQUIRED_INTERACTION_METRICS = {
    "interaction_quality",
    "bad_interrupt_count",
    "missed_visual_change_count",
    "late_background_result_count",
    "premature_answer_count",
    "idle_when_questioned_count",
    "backchannel_overuse_count",
    "residue_count",
    "total_residue_severity",
    "background_delivery_rate",
    "deterministic_replay_passed",
    "no_op_control_score",
    "random_control_score",
    "always_interrupt_control_score",
    "always_silence_control_score",
}


@dataclass
class InteractionBenchmarkManifest:
    command: str
    timestamp: str
    seed: int
    mode: str
    git_commit: str | None
    number_of_events: int
    number_of_actions: int
    number_of_residues: int
    task_family_names: List[str]
    used_test_labels: bool
    deterministic_replay_passed: bool
    control_policy_scores: Dict[str, float]
    anti_cheat_check_results: Dict[str, bool]
    config_hash: str
    benchmark_metrics: Dict[str, object]
    event_timestamps: List[int] = field(default_factory=list)
    bad_control_residue_counts: Dict[str, int] = field(default_factory=dict)
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


def validate_interaction_manifest(manifest: InteractionBenchmarkManifest) -> List[str]:
    checks: List[str] = []
    if manifest.used_test_labels:
        raise ValueError("test labels were used for generation")
    checks.append("test labels not used for generation")

    for previous, current in zip(manifest.event_timestamps, manifest.event_timestamps[1:]):
        if current <= previous:
            raise ValueError("event timestamps are non-monotonic")
    checks.append("event timestamps strictly increasing")

    if not manifest.deterministic_replay_passed:
        raise ValueError("deterministic replay failed")
    checks.append("deterministic replay passed")

    required_controls = {"baseline", "no_op", "random", "always_interrupt", "always_silence"}
    missing_controls = sorted(required_controls - set(manifest.control_policy_scores))
    if missing_controls:
        raise ValueError(f"control policy scores missing: {missing_controls}")
    checks.append("required control policy scores recorded")

    missing_metrics = sorted(REQUIRED_INTERACTION_METRICS - set(manifest.benchmark_metrics))
    if missing_metrics:
        raise ValueError(f"benchmark output lacks required metrics: {missing_metrics}")
    checks.append("required benchmark metrics recorded")

    score_values = {round(float(value), 12) for value in manifest.control_policy_scores.values()}
    if len(score_values) <= 1:
        raise ValueError("all policies received identical scores")
    checks.append("policy controls are not all identical")

    bad_counts = dict(manifest.bad_control_residue_counts)
    if not bad_counts:
        raise ValueError("bad control residue counts missing")
    if all(int(value) == 0 for value in bad_counts.values()):
        raise ValueError("all tasks produced zero residues under intentionally bad controls")
    checks.append("bad controls produced nonzero residues")

    if not manifest.config_hash:
        raise ValueError("config hash missing")
    checks.append("config hash recorded")

    failed_anti_cheat = sorted(key for key, value in manifest.anti_cheat_check_results.items() if not bool(value))
    if failed_anti_cheat:
        raise ValueError(f"anti-cheat checks failed: {failed_anti_cheat}")
    checks.append("anti-cheat checks passed")

    if manifest.number_of_events <= 0 or manifest.number_of_actions <= 0:
        raise ValueError("manifest must record nonzero events and actions")
    checks.append("event and action counts recorded")

    manifest.anti_cheat_checks_passed = checks
    return checks


def build_interaction_manifest(
    *,
    command: str,
    seed: int,
    mode: str,
    config: Dict[str, object],
    benchmark_metrics: Dict[str, object],
    number_of_events: int,
    number_of_actions: int,
    number_of_residues: int,
    task_family_names: Sequence[str],
    used_test_labels: bool,
    deterministic_replay_passed: bool,
    control_policy_scores: Dict[str, float],
    anti_cheat_check_results: Dict[str, bool],
    event_timestamps: Sequence[int],
    bad_control_residue_counts: Dict[str, int],
) -> InteractionBenchmarkManifest:
    manifest = InteractionBenchmarkManifest(
        command=command,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=int(seed),
        mode=str(mode),
        git_commit=current_git_commit(),
        number_of_events=int(number_of_events),
        number_of_actions=int(number_of_actions),
        number_of_residues=int(number_of_residues),
        task_family_names=sorted({str(name) for name in task_family_names}),
        used_test_labels=bool(used_test_labels),
        deterministic_replay_passed=bool(deterministic_replay_passed),
        control_policy_scores={key: float(value) for key, value in control_policy_scores.items()},
        anti_cheat_check_results={key: bool(value) for key, value in anti_cheat_check_results.items()},
        config_hash=stable_config_hash(config),
        benchmark_metrics=dict(benchmark_metrics),
        event_timestamps=[int(item) for item in event_timestamps],
        bad_control_residue_counts={key: int(value) for key, value in bad_control_residue_counts.items()},
    )
    validate_interaction_manifest(manifest)
    return manifest


def write_interaction_manifest(manifest: InteractionBenchmarkManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target
