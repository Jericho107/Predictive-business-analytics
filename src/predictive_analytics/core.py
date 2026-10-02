from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable

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


def acceptance_gate(rows: Iterable[ForecastPoint], minimum_improvement_pct: float = 5.0,
                    max_segment_regression_pct: float = 5.0) -> ModelGate:
    points = list(rows)
    if not points:
        raise ValueError("evaluation set cannot be empty")
    if any(p.actual < 0 or p.baseline < 0 or p.candidate < 0 for p in points):
        raise ValueError("forecast values must be non-negative")
    actual = [p.actual for p in points]
    baseline = [p.baseline for p in points]
    candidate = [p.candidate for p in points]
    baseline_mae = mae(actual, baseline)
    candidate_mae = mae(actual, candidate)
    improvement = ((baseline_mae - candidate_mae) / baseline_mae * 100) if baseline_mae else 0.0

    segment_regressions: list[float] = []
    for segment in sorted({p.segment for p in points}):
        seg = [p for p in points if p.segment == segment]
        b = mae([p.actual for p in seg], [p.baseline for p in seg])
        c = mae([p.actual for p in seg], [p.candidate for p in seg])
        regression = ((c - b) / b * 100) if b else (0.0 if c == 0 else 100.0)
        segment_regressions.append(regression)
    worst_regression = max(segment_regressions, default=0.0)
    accepted = improvement >= minimum_improvement_pct and worst_regression <= max_segment_regression_pct
    return ModelGate(baseline_mae, candidate_mae, improvement, worst_regression, accepted)


def sample() -> list[ForecastPoint]:
    return [
        ForecastPoint("SMB", 100, 92, 98), ForecastPoint("SMB", 120, 108, 117),
        ForecastPoint("SMB", 95, 105, 97), ForecastPoint("Enterprise", 140, 125, 136),
        ForecastPoint("Enterprise", 130, 120, 129), ForecastPoint("Enterprise", 150, 136, 147),
    ]


def serialise_sample() -> dict[str, object]:
    validate_features(["lag_7_revenue", "price_index", "season"])
    return asdict(acceptance_gate(sample()))
