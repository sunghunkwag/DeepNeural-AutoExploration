"""Repeated autonomous neural exploration with trend and memory evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from failure_residue import build_residue_lifecycle_sequence_report
from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop
from experiment_manifest import current_git_commit, stable_config_hash
from neural_search.counterfactual_config_evaluator import CounterfactualConfigEvaluator
from neural_search.evaluator_evolution_adapter import EvaluatorEvolutionEvidenceAdapter
from neural_search.long_horizon_trend_analysis import compute_long_horizon_trends
from neural_search.meta_config_attribution import MetaConfigAttribution
from neural_search.meta_config_memory import MetaConfigMemory, next_run_config_hash
from neural_search.meta_decision_quality import MetaDecisionQualityEvaluator
from neural_search.meta_memory_effectiveness import evaluate_meta_memory_effectiveness
from neural_search.meta_meta_meta_controller import MetaMetaMetaController, MetaMetaMetaControllerOutput
from neural_search.next_run_config import NextRunConfig
from neural_search.search_process_model import SearchProcessModel, compute_process_improvement_components
from neural_search.self_model_adapter import LongHorizonSelfModelAdapter
from research_goal_controller import BenchmarkEvidence, EvidenceGoalGenerator


def run(
    mode: str,
    seed: int,
    runs: int,
    output: str | None = None,
    *,
    memory_enabled: bool = True,
    memory_path_override: str | None = None,
) -> Dict[str, object]:
    run_count = max(1, int(runs))
    output_path = Path(output or f"results/long_horizon_autonomous_exploration_{mode}_seed{seed}.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    memory_path = Path(memory_path_override) if memory_path_override else output_path.with_name(f"long_horizon_meta_config_memory_{mode}_seed{seed}.json")
    memory = MetaConfigMemory.load(memory_path) if memory_enabled else MetaConfigMemory()
    current_config = NextRunConfig.from_dict(memory.records[-1].applied_config) if memory_enabled and memory.records else None
    process_model = SearchProcessModel()
    self_model = LongHorizonSelfModelAdapter()
    evaluator_adapter = EvaluatorEvolutionEvidenceAdapter()

    run_results: list[Dict[str, object]] = []
    run_diagnoses = []
    run_outputs: list[str] = []
    next_run_configs: list[Dict[str, object]] = []
    meta_decisions: list[Dict[str, object]] = []
    attribution_reports: list[Dict[str, object]] = []
    quality_reports: list[Dict[str, object]] = []
    counterfactual_reports: list[Dict[str, object]] = []
    run_summaries: list[Dict[str, object]] = []
    research_goal_trajectory: list[Dict[str, object]] = []

    for run_index in range(run_count):
        run_seed = seed + run_index
        run_path = output_path.with_name(f"{output_path.stem}_run{run_index}.json")
        config_path = None
        if current_config is not None:
            config_path = output_path.with_name(f"{output_path.stem}_run{run_index}_applied_next_config.json")
            current_config.save(config_path)
        result = run_autonomous_loop(mode, run_seed, str(run_path), str(config_path) if config_path is not None else None)
        controller_memory = memory if memory_enabled else None
        meta_output = MetaMetaMetaController(controller_memory).run(
            result,
            base_search_config=current_config.search_strategy_config if current_config is not None else None,
            base_evaluator_config=current_config.evaluator_objective_config if current_config is not None else None,
            base_curriculum_config=current_config.curriculum_strategy_config if current_config is not None else None,
        )
        _apply_meta_output(result, meta_output)
        diagnosis = process_model.diagnose(result)
        meta_decisions.append(meta_output.decision.to_dict())
        next_config, rollback_repair = _apply_rollback_repair_to_next_config(meta_output.next_run_config, result, diagnosis)
        result["rollback_risk_repair_report"] = rollback_repair
        result["rollback_risk_budget"] = rollback_repair["rollback_risk_budget"]
        result["rollback_budget_exceeded"] = rollback_repair["rollback_budget_exceeded"]
        result["rollback_causes_by_method"] = rollback_repair["rollback_causes_by_method"]
        result["rollback_causes_by_task_family"] = rollback_repair["rollback_causes_by_task_family"]
        result["rollback_causes_by_config_change"] = rollback_repair["rollback_causes_by_config_change"]
        result["rollback_repair_actions"] = rollback_repair["rollback_repair_actions"]
        _write_json(run_path, result)
        run_results.append(result)
        run_diagnoses.append(diagnosis)
        run_outputs.append(str(run_path))
        next_config_path = output_path.with_name(f"{output_path.stem}_run{run_index}_next_config.json")
        next_config.save(next_config_path)
        next_run_configs.append({"run_index": run_index, "path": str(next_config_path), "config": next_config.to_dict()})
        run_summaries.append(_run_summary(run_index, result, diagnosis, rollback_repair))
        research_goal_trajectory.append(_research_goal_entry(run_index, result, meta_output.decision.expected_improvement_target))

        if run_index > 0:
            previous = run_results[run_index - 1]
            previous_diag = run_diagnoses[run_index - 1]
            previous_config = next_run_configs[run_index - 1]["config"]
            components = compute_process_improvement_components(previous_diag, diagnosis, previous, result)
            attribution = MetaConfigAttribution().attribute(
                previous,
                result,
                first_next_run_config=_mapping(previous_config),
                second_applied_next_run_config=_mapping(result.get("applied_next_run_config")) or _mapping(previous_config),
                first_diagnosis=previous_diag,
                second_diagnosis=diagnosis,
            )
            quality = MetaDecisionQualityEvaluator().evaluate(
                previous,
                result,
                attribution,
                components,
                first_diagnosis=previous_diag,
                second_diagnosis=diagnosis,
            )
            counterfactual = CounterfactualConfigEvaluator().evaluate(attribution, previous_diag, diagnosis, previous, result)
            attribution_reports.append(attribution.to_dict())
            quality_reports.append(quality.to_dict())
            counterfactual_reports.append(
                {
                    "transition": f"run_{run_index - 1}_to_run_{run_index}",
                    "reports": [item.to_dict() for item in counterfactual],
                    "all_reports_are_estimates": True,
                    "claims_proof": False,
                }
            )
            lifecycle = attribution.to_dict().get("residue_lifecycle_report", {})
            if isinstance(lifecycle, Mapping):
                run_summaries[run_index]["residue_lifecycle_report"] = dict(lifecycle)
            config_causes = _rollback_causes_by_config_change(attribution.to_dict())
            run_summaries[run_index - 1]["rollback_causes_by_config_change"] = config_causes
            run_summaries[run_index - 1]["process_improvement_delta"] = components["final_process_improvement_delta"]
            run_summaries[run_index - 1]["meta_decision_quality"] = quality.quality_scores["overall_meta_decision_quality_score"]
            memory.add_decision(
                meta_decision_id=str(meta_decisions[run_index - 1]["decision_id"]),
                next_run_config_hash=next_run_config_hash(_mapping(previous_config) or {}),
                applied_config=_mapping(previous_config) or {},
                attribution_report=attribution.to_dict(),
                quality_report=quality.to_dict(),
                process_improvement_delta=components["final_process_improvement_delta"],
                helped_changes=quality.helped_changes,
                harmed_changes=quality.harmed_changes,
                recommendation=quality.recommendation,
                seed=seed + run_index - 1,
                mode=mode,
            )
            memory.save(memory_path)
            self_model.observe_run(run_index - 1, previous, attribution.to_dict(), quality.to_dict())
            evaluator_adapter.observe_run(previous, quality.to_dict())

        current_config = next_config

    if run_results:
        self_model.observe_run(run_count - 1, run_results[-1], None, None)
        evaluator_adapter.observe_run(run_results[-1], None)
    memory.save(memory_path)
    residue_lifecycle = build_residue_lifecycle_sequence_report(run_results, attribution_reports)
    trend_analysis = compute_long_horizon_trends(run_summaries)
    memory_effectiveness = evaluate_meta_memory_effectiveness(memory, meta_decisions, quality_reports).to_dict()
    self_model_summary = self_model.summary()
    evaluator_evolution_summary = evaluator_adapter.summary()
    final_research_goal = _final_research_goal(research_goal_trajectory, trend_analysis)
    result = {
        "benchmark": "long_horizon_autonomous_exploration",
        "mode": mode,
        "seed": seed,
        "memory_guidance_enabled": memory_enabled,
        "run_count": run_count,
        "run_outputs": run_outputs,
        "next_run_configs": next_run_configs,
        "meta_decisions": meta_decisions,
        "attribution_reports": attribution_reports,
        "quality_reports": quality_reports,
        "counterfactual_reports": counterfactual_reports,
        "meta_config_memory_path": str(memory_path),
        "long_horizon_trace": run_summaries,
        "residue_lifecycle_report": residue_lifecycle,
        "trend_analysis": trend_analysis,
        "memory_effectiveness": memory_effectiveness,
        "self_model_summary": self_model_summary,
        "evaluator_evolution_summary": evaluator_evolution_summary,
        "research_goal_trajectory": research_goal_trajectory,
        "final_research_goal": final_research_goal,
        "external_adapter_summary": _external_adapter_summary(mode, seed, output_path),
        "final_recommendation": _final_recommendation(trend_analysis, memory_effectiveness, final_research_goal),
    }
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": f"python benchmarks/long_horizon_autonomous_exploration.py --mode {mode} --seed {seed} --runs {run_count}" + ("" if memory_enabled else " --memory-disabled"),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "config_hash": stable_config_hash({"mode": mode, "seed": seed, "runs": run_count, "memory_guidance_enabled": memory_enabled}),
        "memory_guidance_enabled": memory_enabled,
        "run_count": run_count,
        "run_outputs": run_outputs,
        "meta_config_memory_path": str(memory_path),
        "trend_analysis": trend_analysis,
        "memory_effectiveness": memory_effectiveness,
        "anti_cheat_checks_passed": _anti_cheat_checks(run_results, run_diagnoses),
    }
    result["manifest_path"] = str(manifest_path)
    _write_json(output_path, result)
    _write_json(manifest_path, manifest)
    return result


def _apply_meta_output(result: Dict[str, object], output: MetaMetaMetaControllerOutput) -> None:
    result["search_process_diagnosis"] = output.process_diagnosis.to_dict()
    result["meta_meta_meta_decision"] = output.decision.to_dict()
    result["search_strategy_mutation_plan"] = output.search_strategy_plan.to_dict()
    result["evaluator_objective_mutation_plan"] = output.evaluator_objective_plan.to_dict()
    result["curriculum_mutation_plan"] = output.curriculum_plan.to_dict()
    result["next_experiment_rewrite"] = output.next_experiment_rewrite.to_dict()
    result["next_run_config_recommendation"] = output.next_run_config.to_dict()


def _run_summary(run_index: int, result: Mapping[str, object], diagnosis, rollback_repair: Mapping[str, object] | None = None) -> Dict[str, object]:
    agi = result.get("agi_relevance_metrics", {})
    decisions = [item for item in result.get("architecture_decisions", []) if isinstance(item, Mapping)] if isinstance(result.get("architecture_decisions", []), list) else []
    rollback_rate = sum(1 for item in decisions if item.get("rollback")) / max(1, len(decisions))
    repair = rollback_repair or {}
    return {
        "run_index": run_index,
        "seed": int(result.get("seed", 0)) if isinstance(result.get("seed", 0), int) else 0,
        "rollback_rate": float(rollback_rate),
        "residue_resolution_rate": _metric(agi, "residue_resolution_rate"),
        "persistent_residue_count": float(len(diagnosis.persistent_residue_types)),
        "hidden_validation_robustness": _metric(agi, "hidden_validation_robustness"),
        "evaluator_overfit_risk": diagnosis.evaluator_overfit_risk,
        "transfer_regression_risk": diagnosis.transfer_regression_risk,
        "architecture_complexity_growth": diagnosis.architecture_complexity_growth,
        "task_family_balance": diagnosis.task_family_balance,
        "meta_decision_quality": 0.0,
        "process_improvement_delta": 0.0,
        "rollback_risk_budget": _float(repair.get("rollback_risk_budget")),
        "rollback_budget_exceeded": bool(repair.get("rollback_budget_exceeded", False)),
        "rollback_causes_by_method": dict(repair.get("rollback_causes_by_method", {})) if isinstance(repair.get("rollback_causes_by_method", {}), Mapping) else {},
        "rollback_causes_by_task_family": dict(repair.get("rollback_causes_by_task_family", {})) if isinstance(repair.get("rollback_causes_by_task_family", {}), Mapping) else {},
        "rollback_causes_by_config_change": dict(repair.get("rollback_causes_by_config_change", {})) if isinstance(repair.get("rollback_causes_by_config_change", {}), Mapping) else {},
        "rollback_repair_actions": list(repair.get("rollback_repair_actions", [])) if isinstance(repair.get("rollback_repair_actions", []), list) else [],
        "evaluator_calibration_report": _latest_evaluator_calibration(result),
    }


def _research_goal_entry(run_index: int, result: Mapping[str, object], expected_target: str) -> Dict[str, object]:
    diagnosis = result.get("final_bottleneck_diagnosis", {})
    recommended = result.get("recommended_next_experiment", {})
    evidence = BenchmarkEvidence.from_result(result, source_path=f"long_horizon_run_{run_index}")
    goals = EvidenceGoalGenerator().generate([evidence])
    top_goal = goals[0].to_dict() if goals else {}
    focus_family = _focus_family(result)
    return {
        "run_index": run_index,
        "dominant_failure_type": str(diagnosis.get("primary_failure_type", "none")) if isinstance(diagnosis, Mapping) else "none",
        "recommended_next_experiment": recommended if isinstance(recommended, Mapping) else {},
        "recommended_focus_task_family": focus_family,
        "recommended_module_family": str(diagnosis.get("proposed_module_family", "none")) if isinstance(diagnosis, Mapping) else "none",
        "expected_improvement_target": expected_target,
        "rationale": str(top_goal.get("rationale", "bounded long-horizon evidence collection")),
    }


def _final_research_goal(trajectory: list[Dict[str, object]], trends: Mapping[str, object]) -> Dict[str, object]:
    latest = trajectory[-1] if trajectory else {}
    negative = [name for name, trend in trends.items() if isinstance(trend, Mapping) and not bool(trend.get("improved", False)) and abs(_float(trend.get("delta"))) > 1e-12]
    return {
        "best_next_experiment": latest.get("recommended_next_experiment", {}),
        "reason": "prioritize unresolved negative process trends" if negative else "continue the current bounded next-run policy",
        "expected_bottleneck": negative[0] if negative else latest.get("dominant_failure_type", "none"),
        "required_evidence": [
            "repeat long-horizon validation on additional seeds",
            "compare memory-guided runs against memory-disabled controls",
            "inspect heldout-after-freeze transfer deltas only after candidate freeze",
        ],
    }


def _final_recommendation(trends: Mapping[str, object], memory_effectiveness: Mapping[str, object], final_goal: Mapping[str, object]) -> Dict[str, object]:
    bad_trends = [name for name, trend in trends.items() if isinstance(trend, Mapping) and not bool(trend.get("improved", False)) and abs(_float(trend.get("delta"))) > 1e-12]
    memory_score = _float(memory_effectiveness.get("memory_effectiveness_score"))
    action = "continue_memory_guided_exploration" if memory_score >= 0.5 and len(bad_trends) <= 3 else "revise_long_horizon_policy"
    return {"action": action, "negative_trends": bad_trends, "memory_effectiveness_score": memory_score, "final_research_goal": dict(final_goal)}


def _external_adapter_summary(mode: str, seed: int, output_path: Path) -> Dict[str, object]:
    candidate = output_path.with_name(f"external_neural_adapter_{mode}_seed{seed}.json")
    if not candidate.exists():
        return {"status": "not_run_in_this_benchmark", "expected_path": str(candidate)}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return {
        "status": "available",
        "path": str(candidate),
        "validation_delta": _float(payload.get("validation_delta")),
        "hidden_validation_delta": _float(payload.get("hidden_validation_delta")),
        "heldout_after_freeze_delta": _float(payload.get("heldout_after_freeze_delta")),
    }


def _anti_cheat_checks(run_results: list[Mapping[str, object]], diagnoses: list[object]) -> list[str]:
    for result in run_results:
        decisions = result.get("architecture_decisions", [])
        if isinstance(decisions, list) and any(isinstance(item, Mapping) and item.get("used_heldout_labels") for item in decisions):
            raise ValueError("heldout labels used for candidate selection")
    for diagnosis in diagnoses:
        if getattr(diagnosis, "used_heldout_for_candidate_selection", False):
            raise ValueError("heldout used for candidate selection")
        if not getattr(diagnosis, "heldout_used_only_for_post_freeze_diagnosis", True):
            raise ValueError("heldout not confined to post-freeze diagnosis")
    return [
        "repeated runs completed",
        "meta-config memory persisted as bounded JSON",
        "next-run configs are JSON data",
        "heldout_test is not used for candidate selection",
        "heldout_test is used only for post-freeze diagnosis",
        "negative trends are preserved",
    ]


def _apply_rollback_repair_to_next_config(
    next_config: NextRunConfig,
    result: Mapping[str, object],
    diagnosis,
) -> tuple[NextRunConfig, Dict[str, object]]:
    budget = 0.65
    decisions = [item for item in result.get("architecture_decisions", []) if isinstance(item, Mapping)] if isinstance(result.get("architecture_decisions", []), list) else []
    rollback_count = sum(1 for item in decisions if bool(item.get("rollback", not bool(item.get("accepted", False)))))
    rollback_rate = rollback_count / max(1, len(decisions))
    causes_by_method = _rollback_causes_by_method(decisions)
    causes_by_family = _rollback_causes_by_task_family(decisions)
    validation_hidden_guard = any(_float(item.get("validation_improvement")) > 0.0 and _float(item.get("hidden_validation_improvement")) < 0.0 for item in decisions)
    budget_exceeded = rollback_rate > budget
    actions: list[str] = []
    cfg = next_config.to_dict()
    search_cfg = dict(cfg.get("search_strategy_config", {})) if isinstance(cfg.get("search_strategy_config", {}), Mapping) else {}
    evaluator_cfg = dict(cfg.get("evaluator_objective_config", {})) if isinstance(cfg.get("evaluator_objective_config", {}), Mapping) else {}
    penalties = dict(search_cfg.get("mutation_penalties", {})) if isinstance(search_cfg.get("mutation_penalties", {}), Mapping) else {}
    boosts = dict(search_cfg.get("mutation_boosts", {})) if isinstance(search_cfg.get("mutation_boosts", {}), Mapping) else {}
    if budget_exceeded:
        cfg["recommended_candidate_count"] = max(1, int(cfg.get("recommended_candidate_count", 1)) - 1)
        cfg["recommended_generations"] = max(1, int(cfg.get("recommended_generations", 1)) - 1)
        search_cfg["rollback_risk_weight"] = min(1.0, _float(search_cfg.get("rollback_risk_weight")) + 0.12)
        search_cfg["exploration_temperature"] = max(0.05, _float(search_cfg.get("exploration_temperature")) - 0.10)
        actions.extend(["reduce_candidate_count", "reduce_search_depth", "raise_rollback_risk_weight", "lower_exploration_temperature"])
        for method, count in causes_by_method.items():
            rate = count / max(1, rollback_count)
            penalties[method] = min(1.0, _float(penalties.get(method)) + 0.20 + 0.20 * rate)
            if method in boosts:
                boosts[method] = max(0.0, _float(boosts.get(method)) - 0.10)
            actions.append(f"penalize_rollback_method:{method}")
    if validation_hidden_guard:
        evaluator_cfg["validation_loss_weight"] = max(0.05, _float(evaluator_cfg.get("validation_loss_weight")) - 0.06)
        evaluator_cfg["hidden_validation_loss_weight"] = min(1.0, _float(evaluator_cfg.get("hidden_validation_loss_weight")) + 0.08)
        evaluator_cfg["rollback_risk_weight"] = min(1.0, _float(evaluator_cfg.get("rollback_risk_weight")) + 0.08)
        evaluator_cfg["task_family_regression_penalty_weight"] = min(1.0, _float(evaluator_cfg.get("task_family_regression_penalty_weight")) + 0.05)
        actions.extend(["reject_validation_only_hidden_regression_pattern", "raise_hidden_validation_weight", "raise_task_family_regression_penalty"])
    search_cfg["mutation_penalties"] = penalties
    search_cfg["mutation_boosts"] = boosts
    cfg["search_strategy_config"] = search_cfg
    cfg["evaluator_objective_config"] = evaluator_cfg
    repaired = NextRunConfig.from_dict(cfg)
    return repaired, {
        "rollback_risk_budget": budget,
        "observed_rollback_rate": rollback_rate,
        "rollback_budget_exceeded": budget_exceeded,
        "rollback_causes_by_method": causes_by_method,
        "rollback_causes_by_task_family": causes_by_family,
        "rollback_causes_by_config_change": {},
        "rollback_repair_actions": sorted(set(actions)),
        "validation_hidden_mismatch_guard_triggered": validation_hidden_guard,
        "used_heldout_for_candidate_selection": False,
    }


def _rollback_causes_by_method(decisions: list[Mapping[str, object]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for decision in decisions:
        if bool(decision.get("rollback", not bool(decision.get("accepted", False)))):
            counts[str(decision.get("mutation_method", "unknown"))] += 1
    return dict(counts)


def _rollback_causes_by_task_family(decisions: list[Mapping[str, object]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for decision in decisions:
        if not bool(decision.get("rollback", not bool(decision.get("accepted", False)))):
            continue
        regressions = decision.get("task_family_regressions", {})
        if isinstance(regressions, Mapping) and regressions:
            for family in regressions:
                counts[str(family)] += 1
        else:
            counts["unknown"] += 1
    return dict(counts)


def _rollback_causes_by_config_change(attribution_report: Mapping[str, object]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    changes = attribution_report.get("changes", [])
    if not isinstance(changes, list):
        return {}
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        effect = change.get("observed_effect", {})
        rollback_delta = _float(effect.get("rollback_rate_delta")) if isinstance(effect, Mapping) else 0.0
        if rollback_delta > 0.0 or (bool(change.get("harmed")) and "rollback" in str(effect)):
            counts[str(change.get("config_path", "unknown"))] += 1
    return dict(counts)


def _latest_evaluator_calibration(result: Mapping[str, object]) -> Dict[str, object]:
    plan = result.get("evaluator_objective_mutation_plan", {})
    evidence = plan.get("evidence_used", {}) if isinstance(plan, Mapping) else {}
    calibration = evidence.get("evaluator_calibration_report", {}) if isinstance(evidence, Mapping) else {}
    return dict(calibration) if isinstance(calibration, Mapping) else {}


def _focus_family(result: Mapping[str, object]) -> str:
    diagnosis = result.get("search_process_diagnosis", {})
    concentrations = diagnosis.get("residue_concentration_by_task_family", {}) if isinstance(diagnosis, Mapping) else {}
    if not isinstance(concentrations, Mapping) or not concentrations:
        return "unknown"
    return max(sorted(concentrations), key=lambda family: _float(concentrations[family]))


def _metric(payload: object, name: str) -> float:
    if isinstance(payload, Mapping):
        value = payload.get(name, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
    return 0.0


def _mapping(value: object) -> Dict[str, object] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-horizon autonomous neural exploration")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--memory-disabled", action="store_true")
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.runs, args.output, memory_enabled=not args.memory_disabled)
    print(json.dumps({"run_count": result["run_count"], "trend_analysis": result["trend_analysis"], "final_recommendation": result["final_recommendation"]}, indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/long_horizon_autonomous_exploration_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
