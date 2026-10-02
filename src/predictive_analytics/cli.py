from __future__ import annotations

import json
import sys

from .core import ForecastPoint, acceptance_gate, serialise_sample, validate_features


def smoke() -> int:
    print(json.dumps(serialise_sample(), indent=2, sort_keys=True))
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

    print(json.dumps(cases, indent=2, sort_keys=True))
    return 0 if all(case["status"] == "PASS" for case in cases) else 1


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if command == "smoke":
        return smoke()
    if command == "reverse-test":
        return reverse_test()
    print("usage: python -m predictive_analytics.cli [smoke|reverse-test]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
