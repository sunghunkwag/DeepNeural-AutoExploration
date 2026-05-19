"""Meta-meta-meta controller for bounded search-process adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from .architecture_genome import stable_hash
from .curriculum_strategy_mutator import CurriculumMutationPlan, CurriculumStrategyConfig, CurriculumStrategyMutator
from .evaluator_objective_mutator import EvaluatorObjectiveConfig, EvaluatorObjectiveMutationPlan, EvaluatorObjectiveMutator
from .next_run_config import NextRunConfig
from .search_process_model import SearchProcessDiagnosis, SearchProcessModel
from .search_strategy_mutator import SearchStrategyConfig, SearchStrategyMutationPlan, SearchStrategyMutator


@dataclass(frozen=True)
class NextExperimentRewritePlan:
    rewrite_id: str
    action: str
    focus_failure_types: list[str]
    focus_task_families: list[str]
    config_changes: Dict[str, object]
    rationale: str
    used_heldout_for_candidate_selection: bool = False
    heldout_used_only_for_post_freeze_diagnosis: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "rewrite_id": self.rewrite_id,
            "action": self.action,
            "focus_failure_types": list(self.focus_failure_types),
            "focus_task_families": list(self.focus_task_families),
            "config_changes": dict(self.config_changes),
            "rationale": self.rationale,
            "used_heldout_for_candidate_selection": self.used_heldout_for_candidate_selection,
            "heldout_used_only_for_post_freeze_diagnosis": self.heldout_used_only_for_post_freeze_diagnosis,
        }


@dataclass(frozen=True)
class MetaMetaMetaDecision:
    decision_id: str
    input_run_id: str
    process_diagnosis: Dict[str, object]
    selected_strategy_mutations: Dict[str, object]
    selected_evaluator_mutations: Dict[str, object]
    selected_curriculum_mutations: Dict[str, object]
    next_experiment_rewrite: Dict[str, object]
    rationale: str
    evidence_used: Dict[str, object]
    expected_improvement_target: str
    risk_assessment: Dict[str, object]
    used_heldout_for_candidate_selection: bool = False
    heldout_used_only_for_post_freeze_diagnosis: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "input_run_id": self.input_run_id,
            "process_diagnosis": dict(self.process_diagnosis),
            "selected_strategy_mutations": dict(self.selected_strategy_mutations),
            "selected_evaluator_mutations": dict(self.selected_evaluator_mutations),
            "selected_curriculum_mutations": dict(self.selected_curriculum_mutations),
            "next_experiment_rewrite": dict(self.next_experiment_rewrite),
            "rationale": self.rationale,
            "evidence_used": dict(self.evidence_used),
            "expected_improvement_target": self.expected_improvement_target,
            "risk_assessment": dict(self.risk_assessment),
            "used_heldout_for_candidate_selection": self.used_heldout_for_candidate_selection,
            "heldout_used_only_for_post_freeze_diagnosis": self.heldout_used_only_for_post_freeze_diagnosis,
        }


@dataclass(frozen=True)
class MetaMetaMetaControllerOutput:
    decision: MetaMetaMetaDecision
    process_diagnosis: SearchProcessDiagnosis
    search_strategy_plan: SearchStrategyMutationPlan
    evaluator_objective_plan: EvaluatorObjectiveMutationPlan
    curriculum_plan: CurriculumMutationPlan
    next_experiment_rewrite: NextExperimentRewritePlan
    next_run_config: NextRunConfig

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision.to_dict(),
            "process_diagnosis": self.process_diagnosis.to_dict(),
            "search_strategy_plan": self.search_strategy_plan.to_dict(),
            "evaluator_objective_plan": self.evaluator_objective_plan.to_dict(),
            "curriculum_plan": self.curriculum_plan.to_dict(),
            "next_experiment_rewrite": self.next_experiment_rewrite.to_dict(),
            "next_run_config": self.next_run_config.to_dict(),
        }


class MetaMetaMetaController:
    """Evaluate and adapt the search process after candidate decisions are frozen."""

    def __init__(self) -> None:
        self.process_model = SearchProcessModel()
        self.strategy_mutator = SearchStrategyMutator()
        self.evaluator_mutator = EvaluatorObjectiveMutator()
        self.curriculum_mutator = CurriculumStrategyMutator()

    def run(
        self,
        result: Mapping[str, object],
        *,
        base_search_config: SearchStrategyConfig | None = None,
        base_evaluator_config: EvaluatorObjectiveConfig | None = None,
        base_curriculum_config: CurriculumStrategyConfig | None = None,
    ) -> MetaMetaMetaControllerOutput:
        diagnosis = self.process_model.diagnose(result)
        strategy_plan = self.strategy_mutator.mutate(diagnosis, base_search_config)
        evaluator_plan = self.evaluator_mutator.mutate(
            diagnosis,
            _list_of_mappings(result.get("selection_score_components", [])),
            base_evaluator_config,
        )
        curriculum_plan = self.curriculum_mutator.mutate(diagnosis, base_curriculum_config)
        next_config = _build_next_config(result, diagnosis, strategy_plan, evaluator_plan, curriculum_plan)
        rewrite = _rewrite_plan(result, diagnosis, next_config)
        input_run_id = _input_run_id(result)
        decision_payload = {
            "input_run_id": input_run_id,
            "diagnosis": diagnosis.to_dict(),
            "strategy": strategy_plan.mutated_config,
            "evaluator": evaluator_plan.mutated_config,
            "curriculum": curriculum_plan.mutated_config,
            "rewrite": rewrite.to_dict(),
        }
        decision = MetaMetaMetaDecision(
            decision_id=stable_hash(decision_payload, prefix="mmm_"),
            input_run_id=input_run_id,
            process_diagnosis=diagnosis.to_dict(),
            selected_strategy_mutations=strategy_plan.to_dict(),
            selected_evaluator_mutations=evaluator_plan.to_dict(),
            selected_curriculum_mutations=curriculum_plan.to_dict(),
            next_experiment_rewrite=rewrite.to_dict(),
            rationale=_rationale(diagnosis, strategy_plan, evaluator_plan, curriculum_plan),
            evidence_used={
                "deep_search_decision_count": len(_list_of_mappings(result.get("deep_search_decisions", []))),
                "architecture_decision_count": len(_list_of_mappings(result.get("architecture_decisions", []))),
                "failure_residue_count": len(_list_of_mappings(result.get("failure_residues", []))),
                "heldout_metrics_used_post_freeze_only": True,
            },
            expected_improvement_target=_expected_target(diagnosis),
            risk_assessment={
                "evaluator_overfit_risk": diagnosis.evaluator_overfit_risk,
                "transfer_regression_risk": diagnosis.transfer_regression_risk,
                "complexity_growth": diagnosis.architecture_complexity_growth,
                "used_heldout_for_candidate_selection": diagnosis.used_heldout_for_candidate_selection,
            },
            used_heldout_for_candidate_selection=False,
            heldout_used_only_for_post_freeze_diagnosis=True,
        )
        return MetaMetaMetaControllerOutput(
            decision=decision,
            process_diagnosis=diagnosis,
            search_strategy_plan=strategy_plan,
            evaluator_objective_plan=evaluator_plan,
            curriculum_plan=curriculum_plan,
            next_experiment_rewrite=rewrite,
            next_run_config=next_config,
        )


def _build_next_config(
    result: Mapping[str, object],
    diagnosis: SearchProcessDiagnosis,
    strategy_plan: SearchStrategyMutationPlan,
    evaluator_plan: EvaluatorObjectiveMutationPlan,
    curriculum_plan: CurriculumMutationPlan,
) -> NextRunConfig:
    cfg = result.get("config", {})
    mode = str(result.get("mode", "smoke"))
    if isinstance(cfg, Mapping):
        current_generations = int(cfg.get("generations", 1))
        current_count = int(cfg.get("candidate_count", 3))
    else:
        current_generations = 1
        current_count = 3
    if diagnosis.persistent_residue_types and diagnosis.evaluator_overfit_risk < 0.5:
        generations = min(4, current_generations + 1)
    else:
        generations = current_generations
    if max(diagnosis.rollback_rate_by_method.values() or [0.0]) > 0.7:
        candidate_count = max(1, current_count - 1)
    elif diagnosis.persistent_residue_types:
        candidate_count = min(6, current_count + 1)
    else:
        candidate_count = current_count
    return NextRunConfig(
        search_strategy_config=SearchStrategyConfig.from_dict(strategy_plan.mutated_config),
        evaluator_objective_config=EvaluatorObjectiveConfig.from_dict(evaluator_plan.mutated_config),
        curriculum_strategy_config=CurriculumStrategyConfig.from_dict(curriculum_plan.mutated_config),
        recommended_mode=mode if mode in {"smoke", "quick"} else "smoke",
        recommended_seed_policy="same_seed",
        recommended_generations=generations,
        recommended_candidate_count=candidate_count,
        recommended_focus_failure_types=diagnosis.persistent_residue_types,
        recommended_focus_task_families=_focus_families(diagnosis),
    )


def _rewrite_plan(result: Mapping[str, object], diagnosis: SearchProcessDiagnosis, next_config: NextRunConfig) -> NextExperimentRewritePlan:
    action = "modify_search_strategy"
    if diagnosis.evaluator_overfit_risk > 0.35:
        action = "modify_evaluator_objective"
    elif diagnosis.task_family_balance < 0.75:
        action = "modify_task_curriculum"
    elif diagnosis.persistent_residue_types:
        action = "prioritize_persistent_residue_modules"
    payload = {
        "run": _input_run_id(result),
        "action": action,
        "config": next_config.to_dict(),
    }
    return NextExperimentRewritePlan(
        rewrite_id=stable_hash(payload, prefix="rewrite_"),
        action=action,
        focus_failure_types=list(next_config.recommended_focus_failure_types),
        focus_task_families=list(next_config.recommended_focus_task_families),
        config_changes=next_config.to_dict(),
        rationale=f"next run should {action.replace('_', ' ')} based on process diagnosis",
        used_heldout_for_candidate_selection=False,
        heldout_used_only_for_post_freeze_diagnosis=True,
    )


def _input_run_id(result: Mapping[str, object]) -> str:
    payload = {
        "benchmark": result.get("benchmark", ""),
        "mode": result.get("mode", ""),
        "seed": result.get("seed", 0),
        "final_genome": (result.get("final_genome", {}) or {}).get("config_hash", "") if isinstance(result.get("final_genome", {}), Mapping) else "",
    }
    return stable_hash(payload, prefix="run_")


def _rationale(
    diagnosis: SearchProcessDiagnosis,
    strategy_plan: SearchStrategyMutationPlan,
    evaluator_plan: EvaluatorObjectiveMutationPlan,
    curriculum_plan: CurriculumMutationPlan,
) -> str:
    parts = []
    if diagnosis.persistent_residue_types:
        parts.append("persistent residues remain: " + ",".join(diagnosis.persistent_residue_types))
    if diagnosis.evaluator_overfit_risk > 0.25:
        parts.append("validation-hidden mismatch requires objective adjustment")
    if diagnosis.task_family_balance < 0.75:
        parts.append("task-family balance needs curriculum adjustment")
    parts.append(f"strategy mutations={len(strategy_plan.selected_mutations)}")
    parts.append(f"evaluator mutations={len(evaluator_plan.selected_mutations)}")
    parts.append(f"curriculum mutations={len(curriculum_plan.selected_mutations)}")
    return "; ".join(parts)


def _expected_target(diagnosis: SearchProcessDiagnosis) -> str:
    if diagnosis.evaluator_overfit_risk > 0.25:
        return "hidden_validation_robustness"
    if diagnosis.persistent_residue_types:
        return "residue_resolution_rate"
    if diagnosis.transfer_regression_risk > 0.15:
        return "heldout_generalization_delta"
    return "cumulative_autonomous_improvement_score"


def _focus_families(diagnosis: SearchProcessDiagnosis) -> list[str]:
    items = sorted(diagnosis.residue_concentration_by_task_family.items(), key=lambda item: item[1], reverse=True)
    return [family for family, value in items if value > 0.0][:3]


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]

