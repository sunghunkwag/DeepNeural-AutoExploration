import pytest

from failure_residue import MissingOperatorInferencer, ResidueExtractor, residues_to_experiment_plan


def test_residue_extraction_refuses_heldout_leakage():
    extractor = ResidueExtractor()
    with pytest.raises(ValueError):
        extractor.extract(
            {
                "task_id": "t",
                "benchmark_name": "bench",
                "seed": 1,
                "failure_type": "object-binding failure",
                "used_heldout_labels": True,
            }
        )
    with pytest.raises(ValueError):
        extractor.extract(
            {
                "task_id": "t",
                "benchmark_name": "bench",
                "seed": 1,
                "failure_type": "object-binding failure",
                "split_used": "test",
            }
        )


def test_residue_inferencer_maps_repeated_failures_to_module_family():
    extractor = ResidueExtractor()
    traces = [
        {
            "task_id": "a",
            "benchmark_name": "arc",
            "seed": 10,
            "candidate_id": "arch_a",
            "evidence_metrics": {"object_binding_error_rate": 0.4},
            "split_used": "validation",
        },
        {
            "task_id": "b",
            "benchmark_name": "arc",
            "seed": 11,
            "candidate_id": "arch_b",
            "evidence_metrics": {"object_binding_error_rate": 0.2},
            "split_used": "hidden_validation",
        },
    ]
    residues = extractor.extract_many(traces)
    hypotheses = MissingOperatorInferencer(min_support=2).infer(residues)
    assert len(hypotheses) == 1
    assert hypotheses[0].failure_type == "object-binding failure"
    assert hypotheses[0].proposed_module_family == "object_binding_adapter"
    plan = residues_to_experiment_plan(residues, hypotheses, seed=5)
    assert plan["selection_split"] == "validation"
    assert plan["uses_heldout_labels"] is False
