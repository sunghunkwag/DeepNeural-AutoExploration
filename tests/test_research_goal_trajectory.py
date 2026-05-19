import benchmarks.long_horizon_autonomous_exploration as benchmark

from tests.test_long_horizon_autonomous_exploration import _fake_loop


def test_research_goal_trajectory_is_generated(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "run_autonomous_loop", _fake_loop)
    result = benchmark.run("full", 100, 2, str(tmp_path / "long.json"))
    trajectory = result["research_goal_trajectory"]
    assert len(trajectory) == 2
    assert trajectory[0]["dominant_failure_type"]
    assert "final_research_goal" in result
    assert result["final_research_goal"]["required_evidence"]
