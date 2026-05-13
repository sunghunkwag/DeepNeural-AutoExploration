"""Orchestrator integration smoke and full validation benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator_core import (
    ArtifactBus,
    ArtifactEvaluation,
    DeepNeuralEIEAdapter,
    DeterministicRouter,
    HybridProgramSynthesisAdapter,
    StructuralMemoryStore,
    SystemGraph,
    SystemNode,
    build_manifest,
)


EVALUATOR_ID = "eie_evaluator"


def _threshold(split: str) -> float:
    return 0.70 if split == "validation" else 0.66


def _mode_artifacts(mode: str, seed: int) -> tuple[Any, ...]:
    adapter = HybridProgramSynthesisAdapter()
    return adapter.smoke_artifacts(seed) if mode == "smoke" else adapter.full_artifacts(seed)


def _static_graph() -> SystemGraph:
    graph = SystemGraph()
    graph.add_node(
        SystemNode(
            node_id="orchestrator",
            node_type="orchestrator",
            repo_name="DeepNeural-AutoExploration",
            module_name="orchestrator_core.router.DeterministicRouter",
            capabilities=("artifact_coordination", "control_routing"),
            metadata={},
        )
    )
    graph.add_node(
        SystemNode(
            node_id=EVALUATOR_ID,
            node_type="evaluator",
            repo_name="DeepNeural-AutoExploration",
            module_name="orchestrator_core.adapters.DeepNeuralEIEAdapter",
            capabilities=("validation_gate", "failure_residue_routing"),
            metadata={},
        )
    )
    graph.add_node(
        SystemNode(
            node_id="structural_memory",
            node_type="memory_store",
            repo_name="DeepNeural-AutoExploration",
            module_name="orchestrator_core.memory_store.StructuralMemoryStore",
            capabilities=("decision_lineage", "json_safe_records"),
            metadata={},
        )
    )
    graph.add_node(
        SystemNode(
            node_id="problem_space_version:v0",
            node_type="problem_space_version",
            repo_name="DeepNeural-AutoExploration",
            module_name="eie_core.ProblemSpaceVersion",
            capabilities=("problem_space_lineage",),
            metadata={"version_id": "v0"},
        )
    )
    return graph


def _evaluate_artifact(artifact: Any, split: str) -> ArtifactEvaluation:
    score_key = "validation_score" if split == "validation" else "hidden_validation_score"
    score = float(artifact.payload.get(score_key, 0.0))
    role = str(artifact.payload.get("role", "candidate"))
    control_kind = str(artifact.payload.get("control_kind", "candidate"))
    passed = score >= _threshold(split) and control_kind not in {"no_op", "random"}
    failure_type = None
    severity = 0.0
    evidence: dict[str, Any] = {
        "score": score,
        "threshold": _threshold(split),
        "role": role,
        "control_kind": control_kind,
    }
    if not passed:
        failure_type = _failure_type(artifact, role, control_kind, split)
        severity = min(1.0, max(0.05, _threshold(split) - score + 0.25))
        evidence.update(
            {
                "rejection_reason": failure_type,
                "primitive": "unstable_primitive" if artifact.artifact_type == "primitive_candidate" else "candidate_operator",
                "min_diversity": 3,
                "control": f"{control_kind}_control" if control_kind in {"no_op", "random"} else "",
                "penalty": "validation_to_hidden_gap",
                "blind_spot": "artifact_failure_pattern",
                "missing_signal": "artifact_failure_probe",
            }
        )
    return ArtifactEvaluation(
        artifact_id=artifact.artifact_id,
        evaluator_id=EVALUATOR_ID,
        split=split,
        metrics={"score": score, "threshold": _threshold(split)},
        passed=passed,
        failure_type=failure_type,
        severity=severity,
        evidence=evidence,
        uses_test_signal=bool(artifact.uses_test_signal),
    )


def _failure_type(artifact: Any, role: str, control_kind: str, split: str) -> str:
    if control_kind == "no_op":
        return "no_op_control_failure"
    if control_kind == "random":
        return "random_control_failure"
    if artifact.artifact_type == "evaluator_candidate":
        return "validation_hidden_gap_overfit"
    if artifact.artifact_type == "observation_channel_candidate":
        return "hidden_gap_blind_spot"
    if artifact.artifact_type == "architecture_candidate":
        return "problem_space_validation_gap"
    if artifact.artifact_type == "primitive_candidate":
        return "limited_primitive_diversity"
    return f"{role}_{split}_failure"


def _acceptance_evaluations(artifact: Any) -> tuple[ArtifactEvaluation, ...]:
    return (_evaluate_artifact(artifact, "validation"), _evaluate_artifact(artifact, "hidden_validation"))


def _held_out_test_evaluation(artifact: Any, index: int) -> ArtifactEvaluation:
    base = float(artifact.payload.get("hidden_validation_score", 0.0))
    score = max(0.0, min(1.0, base - 0.01 * (index % 3)))
    passed = score >= 0.64 and not artifact.uses_test_signal
    failure_type = None if passed else "held_out_test_record_failed"
    return ArtifactEvaluation(
        artifact_id=artifact.artifact_id,
        evaluator_id=EVALUATOR_ID,
        split="test",
        metrics={"score": score, "threshold": 0.64},
        passed=passed,
        failure_type=failure_type,
        severity=0.0 if passed else min(1.0, max(0.05, 0.64 - score + 0.20)),
        evidence={"score": score, "post_freeze": True},
        uses_test_signal=False,
    )


def _execute(mode: str, seed: int) -> dict[str, Any]:
    graph = _static_graph()
    bus = ArtifactBus()
    memory = StructuralMemoryStore()
    eie_adapter = DeepNeuralEIEAdapter()
    router = DeterministicRouter(
        graph=graph,
        bus=bus,
        eie_adapter=eie_adapter,
        memory_store=memory,
        seed=seed,
        evaluator_node_id=EVALUATOR_ID,
    )
    artifacts = _mode_artifacts(mode, seed)
    decisions = []
    for artifact in artifacts:
        decision = router.route_artifact(
            artifact,
            _acceptance_evaluations(artifact),
            require_hidden_validation=True,
        )
        decisions.append(decision)

    bus.freeze_decisions()
    test_records = []
    test_limit = len(artifacts) if mode == "full" else min(2, len(artifacts))
    for index, artifact in enumerate(artifacts[:test_limit]):
        evaluation = _held_out_test_evaluation(artifact, index)
        bus.record_evaluation(evaluation, acceptance_driving=False)
        eie_adapter.record_post_freeze_test_non_residue(evaluation)
        test_records.append(evaluation)

    metrics = _metrics(
        mode=mode,
        artifacts=artifacts,
        decisions=decisions,
        graph=graph,
        bus=bus,
        memory=memory,
        failure_residues=eie_adapter.routed_residues,
        test_records=test_records,
    )
    config = {
        "benchmark": "orchestrator_integration",
        "mode": mode,
        "seed": seed,
        "artifact_ids": [artifact.artifact_id for artifact in artifacts],
        "generations": sorted({int(artifact.created_generation) for artifact in artifacts}),
    }
    return {
        "mode": mode,
        "seed": seed,
        "config": config,
        "artifacts": artifacts,
        "decisions": decisions,
        "graph": graph,
        "bus": bus,
        "memory": memory,
        "eie_adapter": eie_adapter,
        "test_records": test_records,
        "metrics": metrics,
    }


def _metrics(
    *,
    mode: str,
    artifacts: Sequence[Any],
    decisions: Sequence[Any],
    graph: SystemGraph,
    bus: ArtifactBus,
    memory: StructuralMemoryStore,
    failure_residues: Sequence[Any],
    test_records: Sequence[ArtifactEvaluation],
) -> dict[str, Any]:
    accepted = [decision for decision in decisions if decision.accepted]
    rejected = [decision for decision in decisions if not decision.accepted]
    validation_count = sum(1 for items in bus.evaluations.values() for item in items if item.split == "validation")
    hidden_count = sum(1 for items in bus.evaluations.values() for item in items if item.split == "hidden_validation")
    test_signal_rejected = any(
        artifact.uses_test_signal and any(decision.artifact_id == artifact.artifact_id and not decision.accepted for decision in decisions)
        for artifact in artifacts
    )
    no_op_rejected = any(
        str(artifact.payload.get("control_kind")) == "no_op"
        and any(decision.artifact_id == artifact.artifact_id and not decision.accepted for decision in decisions)
        for artifact in artifacts
    )
    random_recorded = any(
        str(artifact.payload.get("control_kind")) == "random"
        and any(record.get("artifact_id") == artifact.artifact_id for record in memory.decision_records)
        for artifact in artifacts
    )
    metrics: dict[str, Any] = {
        "mode": mode,
        "artifact_count": len(artifacts),
        "accepted_artifact_count": len(accepted),
        "rejected_artifact_count": len(rejected),
        "test_signal_artifact_rejected": bool(test_signal_rejected),
        "failure_residue_count": len(failure_residues),
        "eie_routed_residue_count": len(failure_residues),
        "memory_record_count": memory.record_count,
        "graph_node_count": graph.to_dict()["node_count"],
        "graph_edge_count": graph.to_dict()["edge_count"],
        "lineage_record_count": len(memory.accepted_lineage),
        "used_test_signal_for_acceptance": any(decision.uses_test_signal_for_acceptance for decision in decisions),
        "deterministic_replay_passed": False,
        "no_op_control_rejected": bool(no_op_rejected),
        "random_control_recorded": bool(random_recorded),
        "validation_evaluation_count": validation_count,
        "hidden_validation_evaluation_count": hidden_count,
        "held_out_test_record_count": len(test_records),
        "full_validation_passed": False,
        "manifest_path": "",
    }
    return metrics


def _fingerprint(payload: Mapping[str, Any]) -> str:
    metrics = dict(payload["metrics"])
    metrics.pop("deterministic_replay_passed", None)
    metrics.pop("full_validation_passed", None)
    metrics.pop("manifest_path", None)
    summary = {
        "metrics": metrics,
        "decisions": [decision.to_dict() for decision in payload["decisions"]],
        "graph": payload["graph"].to_dict(),
        "artifact_bus": payload["bus"].to_dict(),
        "structural_memory": payload["memory"].to_dict(),
        "failure_residues": [residue.to_dict() for residue in payload["eie_adapter"].routed_residues],
        "non_residue_reasons": list(payload["eie_adapter"].non_residue_reasons),
    }
    return json.dumps(summary, sort_keys=True)


def _full_validation_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics["artifact_count"]) >= 10
        and int(metrics["accepted_artifact_count"]) >= 2
        and int(metrics["rejected_artifact_count"]) >= 3
        and int(metrics["failure_residue_count"]) >= 3
        and int(metrics["eie_routed_residue_count"]) >= 3
        and int(metrics["memory_record_count"]) >= int(metrics["accepted_artifact_count"]) + int(metrics["rejected_artifact_count"])
        and int(metrics["graph_node_count"]) >= 6
        and int(metrics["graph_edge_count"]) >= 6
        and int(metrics["lineage_record_count"]) >= int(metrics["accepted_artifact_count"])
        and int(metrics["validation_evaluation_count"]) >= 3
        and int(metrics["hidden_validation_evaluation_count"]) >= 3
        and bool(metrics["test_signal_artifact_rejected"])
        and not bool(metrics["used_test_signal_for_acceptance"])
        and bool(metrics["deterministic_replay_passed"])
        and bool(metrics["no_op_control_rejected"])
        and bool(metrics["random_control_recorded"])
    )


def run(mode: str, seed: int, output: str | None = None) -> dict[str, Any]:
    primary = _execute(mode, seed)
    replay = _execute(mode, seed)
    deterministic = _fingerprint(primary) == _fingerprint(replay)
    primary["metrics"]["deterministic_replay_passed"] = deterministic
    if mode == "full":
        primary["metrics"]["full_validation_passed"] = _full_validation_passed(primary["metrics"])

    if output is None:
        output = f"results/orchestrator_integration_{mode}_seed{seed}.json"
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.json")
    primary["metrics"]["manifest_path"] = str(manifest_path)

    manifest = build_manifest(
        mode=mode,
        seed=seed,
        config=primary["config"],
        result_metrics=primary["metrics"],
        graph=primary["graph"],
        bus=primary["bus"],
        decisions=primary["decisions"],
        memory_store=primary["memory"],
        failure_residues=primary["eie_adapter"].routed_residues,
        mutation_decisions=primary["eie_adapter"].mutation_decisions,
        non_residue_reasons=primary["eie_adapter"].non_residue_reasons,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "seed": seed,
        "config": primary["config"],
        "metrics": primary["metrics"],
        "manifest_path": str(manifest_path),
        "orchestrator_decisions": [decision.to_dict() for decision in primary["decisions"]],
        "anti_cheat_checks_passed": manifest["anti_cheat_checks_passed"],
    }
    result.update(primary["metrics"])
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator integration benchmark")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.seed, args.output)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"saved={args.output or f'results/orchestrator_integration_{args.mode}_seed{args.seed}.json'}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
