import json

from benchmarks.autonomous_neural_exploration_loop import run as run_autonomous_loop
from neural_search.next_run_config import NextRunConfig


def test_next_run_config_cannot_execute_arbitrary_code(tmp_path):
    path = tmp_path / "next_config.json"
    path.write_text(
        json.dumps(
            {
                "__import__('os').system('echo unsafe')": "ignored",
                "recommended_candidate_count": 99,
                "recommended_generations": 99,
                "recommended_mode": "unsafe",
                "curriculum_strategy_config": {"difficulty": "not_valid"},
            }
        ),
        encoding="utf-8",
    )
    cfg = NextRunConfig.load(path)
    assert cfg.recommended_candidate_count == 6
    assert cfg.recommended_generations == 4
    assert cfg.recommended_mode == "smoke"
    assert cfg.curriculum_strategy_config.difficulty == "hard"


def test_autonomous_loop_accepts_bounded_next_run_config(tmp_path):
    cfg = NextRunConfig.from_dict({"recommended_candidate_count": 1, "recommended_generations": 1})
    config_path = tmp_path / "next.json"
    cfg.save(config_path)
    result = run_autonomous_loop("smoke", 1401, str(tmp_path / "autonomous.json"), str(config_path))
    assert result["applied_next_run_config"] is not None
    assert result["config"]["candidate_count"] == 1
    assert all(decision["used_heldout_labels"] is False for decision in result["architecture_decisions"])

