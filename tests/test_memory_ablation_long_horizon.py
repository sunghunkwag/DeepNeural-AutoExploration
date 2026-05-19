import json
from pathlib import Path

import benchmarks.memory_ablation_long_horizon as benchmark


def test_memory_guided_and_disabled_modes_both_run(monkeypatch, tmp_path):
    calls = []

    def fake_long(mode, seed, runs, output, *, memory_enabled=True, memory_path_override=None):
        calls.append(memory_enabled)
        trace = [
            {
                "rollback_rate": 0.2 if memory_enabled else 0.5,
                "residue_resolution_rate": 0.7 if memory_enabled else 0.4,
                "hidden_validation_robustness": 0.8,
                "evaluator_overfit_risk": 0.1,
                "transfer_regression_risk": 0.1,
                "meta_decision_quality": 0.6 if memory_enabled else 0.4,
                "process_improvement_delta": 0.2 if memory_enabled else -0.1,
            }
        ]
        payload = {"long_horizon_trace": trace, "manifest_path": str(Path(output).with_suffix(".manifest.json"))}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_external(mode, seed, output=None):
        payload = {"heldout_after_freeze_delta": -0.01, "manifest_path": str(Path(output).with_suffix(".manifest.json"))}
        Path(output).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(benchmark, "run_long_horizon", fake_long)
    monkeypatch.setattr(benchmark, "run_external_adapter", fake_external)

    out = tmp_path / "ablation.json"
    result = benchmark.run("full", [42, 43], 2, str(out))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert calls == [True, False, True, False]
    assert result["conditions"] == ["memory_guided", "memory_disabled"]
    assert result["aggregate_comparison"]["mean_deltas_positive_means_memory_better"]["rollback_rate"] > 0
    assert result["memory_effect"] == "helped"
    assert out.exists()
    assert manifest["anti_cheat_checks_passed"]
