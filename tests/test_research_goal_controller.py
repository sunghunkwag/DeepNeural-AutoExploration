import json

from benchmarks.research_goal_generation_benchmark import run as run_goal_benchmark
from research_goal_controller import (
    BenchmarkEvidence,
    EvidenceGoalGenerator,
    GoalPolicyConfig,
    MetaMetaConfig,
    MetaMetaGoalController,
    build_goal_generation_report,
)


def _arc_result(mode="quick", seed=42, exact=0.4, cell=0.9577142857):
    return {
        "benchmark": "external_arc_agi_same_shape_code_level_rsi",
        "seed": seed,
        "config": {
            "mode": mode,
            "scope": "ARC-AGI-1 public training same-shape cell-prediction subset",
        },
        "metrics": {
            "baseline_cell_accuracy": 0.6445714286,
            "evolved_cell_accuracy": cell,
            "cell_accuracy_delta": cell - 0.6445714286,
            "baseline_exact_task_accuracy": 0.0,
            "evolved_exact_task_accuracy": exact,
            "exact_task_accuracy_delta": exact,
            "candidate_count": 16.0,
            "accepted_program_count": 15.0,
            "validation_rejected_count": 1.0,
        },
    }


def _coding_result():
    return {
        "benchmark": "humaneval_template_rsi",
        "seed": 42,
        "config": {"mode": "quick"},
        "metrics": {
            "baseline_pass_at_1": 0.0,
            "evolved_pass_at_1": 1.0,
        },
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_benchmark_evidence_loads_arc_metrics_and_tags(tmp_path):
    path = _write_json(tmp_path / "arc_external_rsi_exact_repair_guarded_quick_seed42.json", _arc_result())
    evidence = BenchmarkEvidence.from_result(json.loads(path.read_text(encoding="utf-8")), str(path))
    assert evidence.mode == "quick"
    assert evidence.seed == 42
    assert evidence.metrics["evolved_exact_task_accuracy"] == 0.4
    assert "arc" in evidence.mechanism_tags
    assert "exact_repair" in evidence.mechanism_tags
    assert "same_shape_arc_only" in evidence.limitations


def test_goal_generator_creates_gap_transfer_and_cross_domain_goals(tmp_path):
    quick = _write_json(tmp_path / "arc_external_rsi_exact_repair_guarded_quick_seed42.json", _arc_result())
    full = _write_json(
        tmp_path / "arc_external_rsi_exact_repair_guarded_full_seed42.json",
        _arc_result(mode="full", exact=0.625, cell=0.9244444444),
    )
    coding = _write_json(tmp_path / "humaneval_template_rsi_quick.json", _coding_result())
    evidence = [
        BenchmarkEvidence.from_result(json.loads(path.read_text(encoding="utf-8")), str(path))
        for path in (quick, full, coding)
    ]
    goals = EvidenceGoalGenerator().generate(evidence)
    domains = {goal.domain for goal in goals}
    strategies = {goal.strategy for goal in goals}
    assert "arc_agi1_exact_grid" in domains
    assert "external_coding_generalization" in domains
    assert "cross_domain_operator_transfer" in domains
    assert "transfer" in strategies
    assert all(goal.source_evidence_ids for goal in goals)


def test_meta_meta_controller_improves_weak_goal_policy(tmp_path):
    quick = _write_json(tmp_path / "arc_external_rsi_exact_repair_guarded_quick_seed42.json", _arc_result())
    full = _write_json(
        tmp_path / "arc_external_rsi_exact_repair_guarded_full_seed42.json",
        _arc_result(mode="full", exact=0.625, cell=0.9244444444),
    )
    evidence = [
        BenchmarkEvidence.from_result(json.loads(path.read_text(encoding="utf-8")), str(path))
        for path in (quick, full)
    ]
    weak_policy = GoalPolicyConfig(
        exact_target=0.40,
        cell_target=0.70,
        transfer_threshold=0.90,
        max_goals=3,
    )
    result = MetaMetaGoalController(MetaMetaConfig(rounds=3, candidates_per_round=2, seed=7)).improve(
        weak_policy,
        evidence,
    )
    assert result.final_score >= result.initial_score
    assert result.final_score > 0.0
    assert result.goals


def test_goal_generation_report_and_benchmark_runner_write_outputs(tmp_path):
    quick = _write_json(tmp_path / "arc_external_rsi_exact_repair_guarded_quick_seed42.json", _arc_result())
    full = _write_json(
        tmp_path / "arc_external_rsi_exact_repair_guarded_full_seed42.json",
        _arc_result(mode="full", exact=0.625, cell=0.9244444444),
    )
    report = build_goal_generation_report([quick, full])
    assert report["metrics"]["generated_goal_count"] > 0
    assert report["metrics"]["external_evidence_count"] == 2.0
    output = tmp_path / "research_goal_generation_smoke_seed42.json"
    run_report = run_goal_benchmark(mode="smoke", seed=42, evidence_paths=[quick, full], output=output)
    manifest = output.with_suffix(".manifest.json")
    assert output.exists()
    assert manifest.exists()
    assert run_report["manifest_path"] == str(manifest)
    assert run_report["metrics"]["generated_goal_count"] > 0
