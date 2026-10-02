from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import mean

LEAKAGE_FIELDS = {"future_revenue", "cancelled_after_prediction", "target", "outcome"}


@dataclass(frozen=True)
class ForecastPoint:
    segment: str
    actual: float
    baseline: float
    candidate: float


@dataclass(frozen=True)
class ModelGate:
    baseline_mae: float
    candidate_mae: float
    improvement_pct: float
    worst_segment_regression_pct: float
    accepted: bool


def validate_features(features: Iterable[str]) -> None:
    bad = LEAKAGE_FIELDS.intersection(features)
    if bad:
        raise ValueError(f"target leakage fields detected: {sorted(bad)}")


def mae(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual/predicted must have equal non-zero length")
    return mean(abs(a - p) for a, p in zip(actual, predicted, strict=True))


def acceptance_gate(
    rows: Iterable[ForecastPoint],
    minimum_improvement_pct: float = 5.0,
    max_segment_regression_pct: float = 5.0,
) -> ModelGate:
    if minimum_improvement_pct < 0:
        raise ValueError("minimum improvement must be non-negative")
    if max_segment_regression_pct < 0:
        raise ValueError("maximum segment regression must be non-negative")

    points = list(rows)
    if not points:
        raise ValueError("evaluation set cannot be empty")
    if any(point.actual < 0 or point.baseline < 0 or point.candidate < 0 for point in points):
        raise ValueError("forecast values must be non-negative")

    actual = [point.actual for point in points]
    baseline = [point.baseline for point in points]
    candidate = [point.candidate for point in points]
    baseline_mae = mae(actual, baseline)
    candidate_mae = mae(actual, candidate)
    improvement = (
        (baseline_mae - candidate_mae) / baseline_mae * 100
        if baseline_mae
        else 0.0
    )

    segment_regressions: list[float] = []
    for segment in sorted({point.segment for point in points}):
        scoped = [point for point in points if point.segment == segment]
        baseline_segment_mae = mae(
            [point.actual for point in scoped],
            [point.baseline for point in scoped],
        )
        candidate_segment_mae = mae(
            [point.actual for point in scoped],
            [point.candidate for point in scoped],
        )
        if baseline_segment_mae > 0:
            regression = (
                (candidate_segment_mae - baseline_segment_mae)
                / baseline_segment_mae
                * 100
            )
        else:
            regression = 0.0 if candidate_segment_mae == 0 else 100.0
        segment_regressions.append(regression)

    worst_regression = max(segment_regressions, default=0.0)
    accepted = (
        improvement >= minimum_improvement_pct
        and worst_regression <= max_segment_regression_pct
    )
    return ModelGate(
        baseline_mae,
        candidate_mae,
        improvement,
        worst_regression,
        accepted,
    )


def sample() -> list[ForecastPoint]:
    return [
        ForecastPoint("SMB", 100, 92, 98),
        ForecastPoint("SMB", 120, 108, 117),
        ForecastPoint("SMB", 95, 105, 97),
        ForecastPoint("Enterprise", 140, 125, 136),
        ForecastPoint("Enterprise", 130, 120, 129),
        ForecastPoint("Enterprise", 150, 136, 147),
    ]


def serialise_sample() -> dict[str, object]:
    validate_features(["lag_7_revenue", "price_index", "season"])
    return asdict(acceptance_gate(sample()))
