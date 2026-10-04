from __future__ import annotations

import json
import sys

from .core import ForecastPoint, acceptance_gate, validate_features
from .reporting import write_model_report
from .training import multi_seed_validation, training_evidence


def smoke() -> int:
    payload = training_evidence(42)
    payload["multi_seed"] = multi_seed_validation()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def report() -> int:
    path = write_model_report("output/predictive_model_report.html")
    print(path.as_posix())
    return 0


def reverse_test() -> int:
    cases: list[dict[str, str]] = []

    try:
        validate_features(["price", "future_revenue"])
    except ValueError as exc:
        cases.append({"case": "target-leakage", "status": "PASS", "error": str(exc)})
    else:
        cases.append({"case": "target-leakage", "status": "FAIL", "error": "corruption accepted"})

    try:
        acceptance_gate([ForecastPoint("X", 10, -1, 9)])
    except ValueError as exc:
        cases.append({"case": "negative-forecast", "status": "PASS", "error": str(exc)})
    else:
        cases.append({"case": "negative-forecast", "status": "FAIL", "error": "corruption accepted"})

    weak = acceptance_gate(
        [
            ForecastPoint("A", 100, 95, 0),
            ForecastPoint("A", 110, 105, 0),
            ForecastPoint("B", 150, 145, 0),
            ForecastPoint("B", 160, 155, 0),
        ]
    )
    cases.append(
        {
            "case": "weak-model-rejection",
            "status": "PASS" if not weak.accepted else "FAIL",
            "error": "weak candidate rejected" if not weak.accepted else "weak candidate accepted",
        }
    )

    robustness = multi_seed_validation()
    cases.append(
        {
            "case": "multi-seed-acceptance",
            "status": "PASS" if all(row["accepted"] for row in robustness) else "FAIL",
            "error": "candidate clears all governed seeds",
        }
    )

    print(json.dumps(cases, indent=2, sort_keys=True))
    return 0 if all(case["status"] == "PASS" for case in cases) else 1


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if command == "smoke":
        return smoke()
    if command == "report":
        return report()
    if command == "reverse-test":
        return reverse_test()
    print("usage: python -m predictive_analytics.cli [smoke|report|reverse-test]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
