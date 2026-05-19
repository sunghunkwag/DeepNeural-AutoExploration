from failure_residue import build_residue_lifecycle_sequence_report


def test_residue_lifecycle_tracks_appeared_persisted_resolved_reappeared():
    runs = [
        {"failure_residues": []},
        {"failure_residues": [_residue("object-binding failure", 1)]},
        {"failure_residues": [_residue("object-binding failure", 1), _residue("object-binding failure", 2)]},
        {"failure_residues": []},
        {"failure_residues": [_residue("object-binding failure", 3)]},
    ]
    attribution = {
        "changes": [
            {
                "config_path": "search_strategy_config.mutation_boosts.object_binding_adapter",
                "harmed": True,
                "observed_effect": {
                    "mutation_method": "object_binding_adapter",
                    "target_failure_types": ["object-binding failure"],
                },
            }
        ]
    }

    report = build_residue_lifecycle_sequence_report(runs, [attribution])
    entry = report["residue_lifecycle_report"][0]

    assert entry["status_sequence"] == ["absent", "appeared", "persisted", "resolved", "reappeared"]
    assert entry["unresolved_by_current_module_family"] is True
    assert report["unresolved_residue_types"] == ["object-binding failure"]
    assert report["residue_repair_failure_by_method"]["object_binding_adapter"] == 1


def _residue(failure_type, index):
    return {
        "failure_type": failure_type,
        "task_id": f"object_state_transition-{index}",
        "proposed_operator_or_module_family": "object_binding_adapter",
    }
