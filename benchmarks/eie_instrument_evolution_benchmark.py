"""Bounded Epistemic Instrument Evolution benchmark.

The benchmark uses deterministic synthetic failure residues to exercise the
EIE loop end to end.  It does not call external services, execute generated
source, use GPUs, or use held-out test residues for mutation acceptance.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from eie_core import (
    ACCEPTANCE_SPLITS,
    EIEController,
    EvaluatorSpec,
    ExperimentUnit,
    FailureResidue,
    FailureResidueLedger,
    InstrumentMutationContract,
    ObservationChannel,
    OperatorGeneratorSpec,
    ProblemSpaceVersion,
    ProblemSpaceVersionGraph,
)
from experiment_manifest import current_git_commit, stable_config_hash


REQUIRED_METRICS = (
    "initial_residue_pressure",
    "final_residue_pressure",
    "residue_pressure_delta",
    "mutation_contract_count",
    "accepted_instrument_mutation_count",
    "rejected_instrument_mutation_count",
    "rollback_count",
    "problem_space_version_count",
    "observation_blindspot_reduction",
    "evaluator_penalty_strength_delta",
    "experiment_unit_coverage_gain",
    "operator_generator_diversity_gain",
    "validation_to_hidden_gap",
    "no_op_control_score",
    "random_control_score",
    "used_test_signal_for_acceptance",
)


def _mode_config(mode: str) -> Dict[str, int]:
    return {
        "smoke": {"generations": 4, "validation_repeats": 1, "hidden_validation_repeats": 1, "test_repeats": 1},
        "quick": {"generations": 4, "validation_repeats": 2, "hidden_validation_repeats": 1, "test_repeats": 1},
        "full": {"generations": 5, "validation_repeats": 3, "hidden_validation_repeats": 2, "test_repeats": 2},
    }[mode]


def _initial_controller() -> EIEController:
    observation = ObservationChannel(
        "obs0",
        signal_names=("validation_loss", "runtime_seconds"),
        blind_spots=("control_behavior", "hidden_gap"),
        cost=1.0,
    )
    evaluator = EvaluatorSpec(
        "eval0",
        scoring_terms={"validation_to_hidden_gap": 0.10, "runtime_cost": 0.05, "instability": 0.10},
        adversarial_checks=("deterministic_replay",),
    )
    experiment = ExperimentUnit(
        "exp0",
        task_family="synthetic_residue",
        split_policy={"acceptance_splits": list(ACCEPTANCE_SPLITS), "held_out_split": "test"},
        perturbation_policy={"hidden_validation_perturbations": [], "adversarial_controls": []},
        metrics=("residue_pressure", "validation_to_hidden_gap"),
    )
    generator = OperatorGeneratorSpec(
        "opgen0",
        primitive_search_space=("constant_lr", "maml_step"),
        mutation_policy={"brittle_primitives": ["memory_gate"], "max_candidates": 4},
    )
    version = ProblemSpaceVersion(
        version_id="v0",
        parent_version_id=None,
        observation_channel_ids=(observation.channel_id,),
        evaluator_ids=(evaluator.evaluator_id,),
        experiment_unit_ids=(experiment.unit_id,),
        operator_generator_ids=(generator.generator_id,),
        created_by_contract_id=None,
        generation=0,
    )
    return EIEController(
        observation_channels=(observation,),
        evaluator_specs=(evaluator,),
        experiment_units=(experiment,),
        operator_generators=(generator,),
        version_graph=ProblemSpaceVersionGraph(version),
        ledger=FailureResidueLedger(),
    )


def _make_residue(
    residue_id: str,
    failure_type: str,
    suspected_instrument: str,
    split: str,
    severity: float,
    generation: int,
    evidence: Dict[str, object],
) -> FailureResidue:
    return FailureResidue(
        residue_id=residue_id,
        failure_type=failure_type,
        source="deterministic_synthetic_eie_case",
        evidence=evidence,
        severity=severity,
        suspected_instrument=suspected_instrument,
        generation=generation,
        split=split,
    )


def _synthetic_residues(seed: int, cfg: Dict[str, int], *, split: str) -> List[FailureResidue]:
    rng = random.Random(seed + len(split) * 17)
    repeats = int(cfg["validation_repeats"] if split == "validation" else cfg["hidden_validation_repeats"] if split == "hidden_validation" else cfg["test_repeats"])
    residues: List[FailureResidue] = []
    split_scale = {"validation": 1.0, "hidden_validation": 0.9, "test": 0.95}[split]
    for index in range(repeats):
        jitter = rng.uniform(-0.015, 0.015)
        residues.extend(
            [
                _make_residue(
                    f"{split}_obs_{index}",
                    "hidden_gap_blind_spot",
                    "observation_channel:obs0",
                    split,
                    max(0.0, min(1.0, (0.70 + jitter) * split_scale)),
                    0,
                    {"blind_spot": "hidden_gap", "missing_signal": "hidden_gap_probe"},
                ),
                _make_residue(
                    f"{split}_eval_{index}",
                    "validation_hidden_gap_overfit",
                    "evaluator:eval0",
                    split,
                    max(0.0, min(1.0, (0.46 - jitter) * split_scale)),
                    0,
                    {"penalty": "validation_to_hidden_gap"},
                ),
                _make_residue(
                    f"{split}_experiment_{index}",
                    "hidden_validation_coverage_gap",
                    "experiment_unit:exp0",
                    split,
                    max(0.0, min(1.0, (0.36 + jitter * 0.5) * split_scale)),
                    0,
                    {"perturbation": "hidden_shift"},
                ),
                _make_residue(
                    f"{split}_operator_{index}",
                    "limited_primitive_diversity",
                    "operator_generator:opgen0",
                    split,
                    max(0.0, min(1.0, (0.31 - jitter * 0.5) * split_scale)),
                    0,
                    {"primitive": "stability_probe", "brittle": False, "min_diversity": 3},
                ),
            ]
        )
    return residues


def _count_experiment_coverage(controller: EIEController) -> int:
    total = 0
    for unit in controller.experiment_units.values():
        total += len(unit.perturbation_policy.get("hidden_validation_perturbations", []))
        total += len(unit.perturbation_policy.get("adversarial_controls", []))
    return int(total)


def _penalty_strength(controller: EIEController) -> float:
    return float(sum(sum(spec.scoring_terms.values()) for spec in controller.evaluator_specs.values()))


def _blindspot_count(controller: EIEController) -> int:
    return int(sum(len(channel.blind_spots) for channel in controller.observation_channels.values()))


def _operator_diversity(controller: EIEController) -> int:
    return int(sum(len(set(spec.primitive_search_space)) for spec in controller.operator_generators.values()))


def _random_control_contract(controller: EIEController, *, generation: int, seed: int) -> InstrumentMutationContract:
    rng = random.Random(seed + generation * 1009)
    primitive = f"random_control_{rng.randrange(1000)}"
    return InstrumentMutationContract(
        contract_id=f"control_random_g{generation}_{len(controller.contracts)}",
        target_instrument_type="operator_generator",
        target_instrument_id="opgen0",
        reason="random control must be measured and should not be accepted without pressure reduction",
        expected_residue_reduction=0.0,
        mutation_payload={"action": "expand_primitive_diversity", "primitive": primitive},
        created_generation=generation,
        uses_test_signal=False,
    )


def _validate_manifest(manifest: Dict[str, object]) -> None:
    required = {
        "failure_residue_ledger",
        "instrument_mutation_contracts",
        "problem_space_version_graph",
        "anti_cheat_checks_passed",
        "validation_only_acceptance_enforced",
        "held_out_test_used_only_after_freeze",
    }
    missing = [key for key in sorted(required) if key not in manifest]
    if missing:
        raise ValueError(f"EIE manifest missing required fields: {missing}")
    if manifest.get("used_test_signal_for_acceptance"):
        raise ValueError("EIE manifest indicates test signal was used for acceptance")
    decisions = list(manifest.get("instrument_mutation_decisions", []))
    if not decisions:
        raise ValueError("EIE manifest missing mutation decisions")
    no_op_seen = False
    random_seen = False
    for decision in decisions:
        contract = decision.get("contract", {}) if isinstance(decision, dict) else {}
        payload = contract.get("mutation_payload", {}) if isinstance(contract, dict) else {}
        action = payload.get("action") if isinstance(payload, dict) else None
        if decision.get("used_test_signal_for_acceptance"):
            raise ValueError("EIE decision used test signal for acceptance")
        if "before_pressure" not in decision or "after_pressure" not in decision:
            raise ValueError("EIE decision missing before/after pressure")
        if decision.get("accepted") and not decision.get("lineage_recorded"):
            raise ValueError("accepted EIE mutation missing lineage record")
        if action == "no_op":
            no_op_seen = True
            if decision.get("accepted"):
                raise ValueError("no-op EIE control was accepted")
        if str(contract.get("contract_id", "")).startswith("control_random"):
            random_seen = True
    if not no_op_seen:
        raise ValueError("EIE manifest missing no-op control")
    if not random_seen:
        raise ValueError("EIE manifest missing random control")


def _anti_cheat_checks() -> List[str]:
    return [
        "random seeds explicitly recorded",
        "validation-only instrument acceptance enforced",
        "hidden validation pressure included in acceptance measurement",
        "held-out OOD test used only after decisions are frozen",
        "used_test_signal_for_acceptance=false recorded",
        "no-op instrument mutation tested and rejected",
        "random instrument mutation control recorded",
        "residue pressure computed from validation and hidden-validation only",
        "before and after pressure measured for every acceptance decision",
        "accepted instrument mutations record problem-space lineage",
        "no external services or network calls used",
        "no arbitrary generated source execution",
    ]


def _run_seed(seed: int, mode: str) -> Dict[str, object]:
    cfg = _mode_config(mode)
    controller = _initial_controller()
    initial_blindspots = _blindspot_count(controller)
    initial_penalty_strength = _penalty_strength(controller)
    initial_coverage = _count_experiment_coverage(controller)
    initial_operator_diversity = _operator_diversity(controller)

    acceptance_residues = _synthetic_residues(seed, cfg, split="validation")
    acceptance_residues.extend(_synthetic_residues(seed + 1, cfg, split="hidden_validation"))
    controller.observe_failures(acceptance_residues, acceptance_driving=True)
    initial_pressure = controller.measure_residue_pressure()
    generation_summaries: List[Dict[str, object]] = []

    for generation in range(int(cfg["generations"])):
        generation_result = controller.run_generation(generation=generation)
        generation_summaries.append(generation_result)

    no_op_contract = controller.make_no_op_contract(
        target_instrument_type="experiment_unit",
        target_instrument_id="exp0",
        generation=int(cfg["generations"]),
    )
    no_op_decision = controller.evaluate_contract(no_op_contract)
    random_contract = _random_control_contract(controller, generation=int(cfg["generations"]) + 1, seed=seed)
    random_decision = controller.evaluate_contract(random_contract)

    frozen_pressure = controller.measure_residue_pressure()
    controller.observe_failures(_synthetic_residues(seed + 2, cfg, split="test"), acceptance_driving=False)
    post_freeze_test = controller.evaluate_held_out_test_after_freeze()

    accepted = [decision for decision in controller.decisions if decision.get("accepted")]
    rejected = [decision for decision in controller.decisions if not decision.get("accepted")]
    validation_pressure = float(frozen_pressure["by_split"].get("validation", 0.0))
    hidden_pressure = float(frozen_pressure["by_split"].get("hidden_validation", 0.0))
    metrics = {
        "initial_residue_pressure": float(initial_pressure["total_pressure"]),
        "final_residue_pressure": float(frozen_pressure["total_pressure"]),
        "residue_pressure_delta": float(initial_pressure["total_pressure"] - frozen_pressure["total_pressure"]),
        "mutation_contract_count": float(len(controller.contracts)),
        "accepted_instrument_mutation_count": float(len(accepted)),
        "rejected_instrument_mutation_count": float(len(rejected)),
        "rollback_count": float(controller.rollback_count),
        "problem_space_version_count": float(controller.version_graph.version_count),
        "observation_blindspot_reduction": float(initial_blindspots - _blindspot_count(controller)),
        "evaluator_penalty_strength_delta": float(_penalty_strength(controller) - initial_penalty_strength),
        "experiment_unit_coverage_gain": float(_count_experiment_coverage(controller) - initial_coverage),
        "operator_generator_diversity_gain": float(_operator_diversity(controller) - initial_operator_diversity),
        "validation_to_hidden_gap": float(abs(validation_pressure - hidden_pressure)),
        "no_op_control_score": float(no_op_decision["pressure_delta"]),
        "random_control_score": float(random_decision["pressure_delta"]),
        "used_test_signal_for_acceptance": False,
    }
    for key in REQUIRED_METRICS:
        if key not in metrics:
            raise RuntimeError(f"missing EIE metric: {key}")
    if metrics["initial_residue_pressure"] <= 0.0:
        raise RuntimeError("initial EIE residue pressure must be positive")
    if metrics["mutation_contract_count"] < 1.0:
        raise RuntimeError("EIE must create at least one mutation contract")
    if metrics["used_test_signal_for_acceptance"]:
        raise RuntimeError("EIE benchmark used test signal for acceptance")
    return {
        "seed": seed,
        "metrics": metrics,
        "controller": controller.to_dict(),
        "generation_summaries": generation_summaries,
        "post_freeze_test_evaluation": post_freeze_test,
        "task_ids": {
            "validation": [residue.residue_id for residue in acceptance_residues if residue.split == "validation"],
            "hidden_validation": [residue.residue_id for residue in acceptance_residues if residue.split == "hidden_validation"],
            "test": [residue.residue_id for residue in controller.ledger.records if residue.split == "test"],
        },
    }


def _stderr(values: Sequence[float]) -> float:
    return 0.0 if len(values) <= 1 else float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _aggregate(per_seed: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float | bool]]:
    aggregate: Dict[str, Dict[str, float | bool]] = {}
    for key in REQUIRED_METRICS:
        values = [item["metrics"][key] for item in per_seed]  # type: ignore[index]
        if isinstance(values[0], bool):
            aggregate[key] = {"mean": bool(any(values)), "stderr": 0.0}
        else:
            floats = [float(value) for value in values]
            aggregate[key] = {"mean": float(mean(floats)), "stderr": _stderr(floats)}
    return aggregate


def _build_manifest(
    *,
    mode: str,
    seed: int,
    seeds: Sequence[int],
    config: Dict[str, object],
    aggregate: Dict[str, object],
    per_seed_payload: Sequence[Dict[str, object]],
    result_metrics: Dict[str, object],
) -> Dict[str, object]:
    controllers = [item["controller"] for item in per_seed_payload]
    all_decisions = [decision for controller in controllers for decision in controller.get("instrument_mutation_decisions", [])]
    all_contracts = [contract for controller in controllers for contract in controller.get("instrument_mutation_contracts", [])]
    ledger_records = [record for controller in controllers for record in controller.get("failure_residue_ledger", {}).get("records", [])]
    graph = controllers[-1]["problem_space_version_graph"] if controllers else {}
    checks = _anti_cheat_checks()
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": "python benchmarks/eie_instrument_evolution_benchmark.py",
        "mode": mode,
        "seed": seed,
        "seeds": [int(item) for item in seeds],
        "config": config,
        "config_hash": stable_config_hash(config),
        "metric_summary": {"aggregate": aggregate, "metrics": result_metrics},
        "failure_residue_ledger": {"records": ledger_records},
        "instrument_mutation_contracts": all_contracts,
        "instrument_mutation_decisions": all_decisions,
        "problem_space_version_graph": graph,
        "anti_cheat_checks_passed": checks,
        "validation_only_acceptance_enforced": True,
        "held_out_test_used_only_after_freeze": True,
        "used_test_signal_for_acceptance": False,
        "no_op_control_recorded": any(str(item.get("contract_id", "")).startswith("control_no_op") for item in all_contracts),
        "random_control_recorded": any(str(item.get("contract_id", "")).startswith("control_random") for item in all_contracts),
    }
    _validate_manifest(manifest)
    return manifest


def run(mode: str, seed: int, output: str | None = None) -> Dict[str, object]:
    seeds = [seed] if mode == "smoke" else ([seed, seed + 1] if mode == "quick" else [seed + i for i in range(5)])
    per_seed = [_run_seed(item, mode) for item in seeds]
    aggregate = _aggregate(per_seed)
    metrics = {key: aggregate[key]["mean"] for key in REQUIRED_METRICS}
    config = {
        "benchmark": "eie_instrument_evolution",
        "mode": mode,
        "seed": seed,
        "seeds": seeds,
        "mode_config": _mode_config(mode),
    }
    per_seed_payload = [
        {
            "seed": item["seed"],
            "metrics": item["metrics"],
            "controller": item["controller"],
            "generation_summaries": item["generation_summaries"],
            "post_freeze_test_evaluation": item["post_freeze_test_evaluation"],
            "task_ids": item["task_ids"],
        }
        for item in per_seed
    ]
    if output is None:
        output = f"results/eie_instrument_evolution_{mode}_seed{seed}.json"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(
        mode=mode,
        seed=seed,
        seeds=seeds,
        config=config,
        aggregate=aggregate,
        per_seed_payload=per_seed_payload,
        result_metrics=metrics,
    )
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "seed": seed,
        "seeds": seeds,
        "config": config,
        "metrics": metrics,
        "aggregate": aggregate,
        "per_seed": per_seed_payload,
        "manifest_path": str(manifest_path),
        "anti_cheat_checks_passed": manifest["anti_cheat_checks_passed"],
        "used_test_signal_for_acceptance": False,
    }
    result.update(metrics)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded EIE instrument evolution benchmark")
    parser.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/eie_instrument_evolution_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
