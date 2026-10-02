import pytest

from predictive_analytics.core import (
    ForecastPoint,
    acceptance_gate,
    sample,
    validate_features,
)


def test_candidate_beats_baseline_and_is_accepted():
    gate = acceptance_gate(sample())
    assert gate.candidate_mae < gate.baseline_mae
    assert gate.accepted is True


def test_leakage_is_rejected():
    with pytest.raises(ValueError, match="target leakage"):
        validate_features(["price", "future_revenue"])


def test_segment_regression_can_block_acceptance():
    rows = [
        ForecastPoint("A", 100, 80, 99),
        ForecastPoint("A", 100, 80, 99),
        ForecastPoint("B", 100, 99, 110),
    ]
    gate = acceptance_gate(rows, minimum_improvement_pct=0)
    assert gate.worst_segment_regression_pct > 5
    assert gate.accepted is False
