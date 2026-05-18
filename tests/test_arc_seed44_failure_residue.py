import json
from pathlib import Path

from failure_residue import ARC_FAILURE_TAXONOMY, ResidueExtractor


def test_arc_seed44_style_failures_classify_into_taxonomy():
    extractor = ResidueExtractor()
    cases = [
        ({"object_binding_error_rate": 0.5}, "object-binding failure"),
        ({"shape_mismatch_rate": 0.3}, "variable-shape failure"),
        ({"color_ambiguity_rate": 0.7}, "color-rule ambiguity"),
        ({"support_pair_count": 1.0}, "insufficient support evidence"),
        ({"primitive_miss_rate": 1.0}, "primitive-library blind spot"),
        ({"hidden_validation_regression": -0.2}, "evaluator-selection failure"),
        ({"parameter_budget_pressure": 0.8}, "architecture-representation bottleneck"),
    ]
    for metrics, expected in cases:
        residue = extractor.extract(
            {
                "task_id": "synthetic-seed44-like",
                "benchmark_name": "external_arc_agi_same_shape_code_level_rsi",
                "seed": 44,
                "candidate_id": "candidate",
                "evidence_metrics": metrics,
                "split_used": "validation",
                "used_heldout_labels": False,
            }
        )
        assert residue.failure_type == expected
        assert residue.missing_representation_hypothesis
        assert residue.proposed_operator_or_module_family


def test_arc_failure_classification_is_not_seed_specific():
    extractor = ResidueExtractor()
    base = {
        "task_id": "synthetic",
        "benchmark_name": "external_arc_agi_same_shape_code_level_rsi",
        "candidate_id": "candidate",
        "evidence_metrics": {"object_binding_error_rate": 0.5},
        "split_used": "validation",
    }
    seed44 = extractor.extract({**base, "seed": 44})
    seed45 = extractor.extract({**base, "seed": 45})
    assert seed44.failure_type == seed45.failure_type == "object-binding failure"


def test_arc_seed44_taxonomy_schema_lists_required_categories():
    schema = json.loads(Path("results_schema/arc_seed44_failure_taxonomy.schema.json").read_text(encoding="utf-8"))
    enum_values = schema["properties"]["failure_type"]["enum"]
    assert sorted(enum_values) == sorted(ARC_FAILURE_TAXONOMY)
    assert schema["properties"]["used_heldout_labels"]["const"] is False
