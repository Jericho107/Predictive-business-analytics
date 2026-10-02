# Predictive Business Analytics

Baseline-first predictive analytics with leakage controls, model acceptance gates, segment regression guardrails, and explicit evidence boundaries.

All data and entities are synthetic. No client result or realised ROI is claimed.

## Management question

Does a candidate model improve enough over a simple baseline to justify operational use without hiding segment-level regressions or target leakage?

## Validation

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m predictive_analytics.cli smoke
python -m predictive_analytics.cli reverse-test
```

**Pretoria BI — Understand · Decide · Act · Measure**
