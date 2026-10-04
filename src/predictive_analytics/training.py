from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import ForecastPoint, acceptance_gate, validate_features

FEATURES = (
    "day_index",
    "lag_7",
    "sin_week",
    "cos_week",
    "price_index",
    "promo",
    "enterprise",
)


@dataclass(frozen=True)
class TrainingRow:
    day_index: int
    segment: str
    actual: float
    lag_7: float
    sin_week: float
    cos_week: float
    price_index: float
    promo: int
    enterprise: int


def generate_dataset(seed: int = 42, days: int = 240) -> list[TrainingRow]:
    if days < 220:
        raise ValueError("dataset must cover train, validation and final-test windows")
    rng = random.Random(seed)
    history: dict[str, list[float]] = {"SMB": [], "Enterprise": []}
    rows: list[TrainingRow] = []

    for day in range(days):
        for segment in ("SMB", "Enterprise"):
            base = 100.0 if segment == "SMB" else 150.0
            trend = (0.18 if segment == "SMB" else 0.10) * day
            sin_week = math.sin(2 * math.pi * day / 7)
            cos_week = math.cos(2 * math.pi * day / 7)
            weekly = 12 * sin_week + 5 * cos_week
            promo = int(day % 29 in {0, 1, 2})
            price_index = (
                1.0
                + 0.05 * math.sin(2 * math.pi * day / 31)
                + (0.02 if segment == "Enterprise" else 0.0)
            )
            promo_effect = (18 if segment == "SMB" else 12) * promo
            price_effect = -45 * (price_index - 1)
            actual = (
                base
                + trend
                + weekly
                + promo_effect
                + price_effect
                + rng.gauss(0, 3.5)
            )
            lag_7 = history[segment][day - 7] if day >= 7 else None
            history[segment].append(actual)

            if lag_7 is not None:
                rows.append(
                    TrainingRow(
                        day,
                        segment,
                        actual,
                        lag_7,
                        sin_week,
                        cos_week,
                        price_index,
                        promo,
                        int(segment == "Enterprise"),
                    )
                )
    return rows


def chronological_split(
    rows: list[TrainingRow],
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    train = [row for row in rows if row.day_index < 180]
    validation = [row for row in rows if 180 <= row.day_index < 210]
    test = [row for row in rows if row.day_index >= 210]
    if not train or not validation or not test:
        raise ValueError("chronological split produced an empty partition")
    if max(row.day_index for row in train) >= min(row.day_index for row in validation):
        raise ValueError("train/validation time overlap")
    if max(row.day_index for row in validation) >= min(row.day_index for row in test):
        raise ValueError("validation/test time overlap")
    return train, validation, test


def _features(rows: list[TrainingRow]) -> list[list[float]]:
    return [
        [
            row.day_index,
            row.lag_7,
            row.sin_week,
            row.cos_week,
            row.price_index,
            row.promo,
            row.enterprise,
        ]
        for row in rows
    ]


def _targets(rows: list[TrainingRow]) -> list[float]:
    return [row.actual for row in rows]


def _forecast_points(
    rows: list[TrainingRow],
    candidate: list[float],
) -> list[ForecastPoint]:
    return [
        ForecastPoint(row.segment, row.actual, row.lag_7, prediction)
        for row, prediction in zip(rows, candidate, strict=True)
    ]


def _model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def training_evidence(seed: int = 42) -> dict[str, object]:
    validate_features(FEATURES)
    rows = generate_dataset(seed)
    train, validation, test = chronological_split(rows)

    model = _model()
    model.fit(_features(train), _targets(train))
    validation_prediction = model.predict(_features(validation)).tolist()
    validation_gate = acceptance_gate(_forecast_points(validation, validation_prediction))

    result: dict[str, object] = {
        "seed": seed,
        "features": list(FEATURES),
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_max_day": max(row.day_index for row in train),
            "validation_min_day": min(row.day_index for row in validation),
            "validation_max_day": max(row.day_index for row in validation),
            "test_min_day": min(row.day_index for row in test),
        },
        "validation": asdict(validation_gate),
        "final_test": None,
        "accepted": False,
        "top_coefficients": [],
    }
    if not validation_gate.accepted:
        return result

    development = train + validation
    final_model = _model()
    final_model.fit(_features(development), _targets(development))
    test_prediction = final_model.predict(_features(test)).tolist()
    test_gate = acceptance_gate(_forecast_points(test, test_prediction))

    coefficients = final_model.named_steps["ridge"].coef_.tolist()
    ranked = sorted(
        (
            {"feature": feature, "standardised_coefficient": round(coefficient, 6)}
            for feature, coefficient in zip(FEATURES, coefficients, strict=True)
        ),
        key=lambda item: abs(item["standardised_coefficient"]),
        reverse=True,
    )

    result["final_test"] = asdict(test_gate)
    result["accepted"] = validation_gate.accepted and test_gate.accepted
    result["top_coefficients"] = ranked[:5]
    return result


def multi_seed_validation(seeds: tuple[int, ...] = (42, 107, 314)) -> list[dict[str, object]]:
    return [training_evidence(seed) for seed in seeds]
