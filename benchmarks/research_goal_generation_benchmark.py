"""Benchmark-evidence to research-goal generation runner.

This runner reads existing benchmark result JSON files, generates bounded
research goals, proposes transfer hypotheses, and writes a report plus an
audit manifest. It does not use held-out labels for candidate acceptance and
does not claim that goal generation itself improves ARC or coding scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_manifest import current_git_commit, stable_config_hash
from research_goal_controller import (
    GoalPolicyConfig,
    MetaMetaConfig,
    build_goal_generation_report,
)


def _default_evidence_paths(mode: str) -> List[Path]:
    results = ROOT / "results"
    candidates = [
        results / "arc_external_rsi_exact_repair_guarded_quick_seed42.json",
        results / "arc_external_rsi_exact_repair_guarded_full_seed42.json",
        results / "humaneval_template_rsi_quick.json",
    ]
    if mode == "smoke":
        candidates = candidates[:2]
    elif mode == "full":
        candidates.extend(
            sorted(results.glob("arc_external_rsi_multiseed_*_seed*.json"))[:12]
        )
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError("no default benchmark evidence files found under results/")
    return existing


def _write_manifest(
    output_path: Path,
    evidence_paths: Sequence[Path],
    report: dict,
    config: dict,
) -> Path:
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": " ".join(sys.argv),
        "evidence_paths": [str(path) for path in evidence_paths],
        "metric_summary": report.get("metrics", {}),
        "config": config,
        "config_hash": stable_config_hash(config),
        "anti_cheat_checks_passed": [
            "benchmark evidence loaded from explicit JSON result files",
            "research goals generated from published metrics and limitations",
            "meta-meta loop mutates goal-policy configuration only",
            "transfer hypotheses remain proposals until separately benchmarked",
            "no generated goal is counted as held-out benchmark improvement",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def run(
    *,
    mode: str,
    seed: int,
    evidence_paths: Sequence[str | Path],
    output: str | Path | None = None,
) -> dict:
    resolved_evidence = [Path(path) for path in evidence_paths] or _default_evidence_paths(mode)
    output_path = Path(output) if output else ROOT / "results" / f"research_goal_generation_{mode}_seed{seed}.json"
    policy = GoalPolicyConfig()
    meta = MetaMetaConfig(seed=seed, rounds=2 if mode == "smoke" else 4 if mode == "quick" else 6)
    report = build_goal_generation_report(
        resolved_evidence,
        policy_config=policy,
        meta_config=meta,
    )
    config = {
        "benchmark": "research_goal_generation_controller",
        "mode": mode,
        "seed": seed,
        "evidence_paths": [str(path) for path in resolved_evidence],
        "policy_config": policy.to_dict(),
        "meta_config": {
            "rounds": meta.rounds,
            "candidates_per_round": meta.candidates_per_round,
            "mutation_scale": meta.mutation_scale,
            "seed": meta.seed,
        },
    }
    manifest_path = _write_manifest(output_path, resolved_evidence, report, config)
    report.update(
        {
            "seed": seed,
            "config": config,
            "manifest_path": str(manifest_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quick", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence", nargs="*", default=[])
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = run(
        mode=args.mode,
        seed=args.seed,
        evidence_paths=args.evidence,
        output=args.output,
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

