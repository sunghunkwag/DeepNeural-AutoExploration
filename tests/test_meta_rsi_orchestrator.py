import json
from pathlib import Path

from meta_rsi import MetaRSIOrchestrator
from neural_search import ArchitectureEvaluator, NeuralCandidateSandbox


def test_meta_rsi_orchestrator_produces_plan_and_audit(tmp_path):
    orchestrator = MetaRSIOrchestrator(seed=9)
    orchestrator.evaluator = ArchitectureEvaluator(
        seed=9,
        sandbox=NeuralCandidateSandbox(seed=9, train_steps=2, task_count=1, samples_per_split=8),
        improvement_threshold=-999.0,
    )
    audit_path = tmp_path / "audit.json"
    audit = orchestrator.run(
        [
            {
                "task_id": "arc-like-object",
                "benchmark_name": "external_arc_agi_same_shape_code_level_rsi",
                "seed": 44,
                "candidate_id": "candidate_a",
                "evidence_metrics": {"object_binding_error_rate": 0.4},
                "split_used": "validation",
            },
            {
                "task_id": "arc-like-architecture",
                "benchmark_name": "external_arc_agi_same_shape_code_level_rsi",
                "seed": 44,
                "candidate_id": "candidate_b",
                "evidence_metrics": {"parameter_budget_pressure": 0.8},
                "split_used": "hidden_validation",
            },
        ],
        audit_path=audit_path,
    )
    assert audit_path.exists()
    reloaded = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    assert reloaded["experiment_plan"]["selection_split"] == "validation"
    assert reloaded["experiment_plan"]["candidate_module_families"]
    assert reloaded["architecture_decisions"]
    accepted = [item for item in audit["architecture_decisions"] if item["accepted"]]
    assert accepted
    assert all(item["used_heldout_labels"] is False for item in accepted)
    assert audit["anti_cheat"]["accepted_candidates_use_heldout_labels"] is False
