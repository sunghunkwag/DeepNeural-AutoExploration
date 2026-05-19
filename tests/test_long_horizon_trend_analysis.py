from neural_search.long_horizon_trend_analysis import compute_long_horizon_trends


def test_trend_analysis_preserves_negative_trends():
    summaries = [
        {"rollback_rate": 0.1, "residue_resolution_rate": 0.8, "persistent_residue_count": 1.0},
        {"rollback_rate": 0.3, "residue_resolution_rate": 0.5, "persistent_residue_count": 3.0},
    ]
    trends = compute_long_horizon_trends(summaries)
    assert trends["residue_resolution_trend"]["delta"] < 0.0
    assert trends["residue_resolution_trend"]["improved"] is False
    assert trends["rollback_rate_trend"]["delta"] > 0.0
    assert trends["rollback_rate_trend"]["improved"] is False
