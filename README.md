<div align="center">

# Predictive Business Analytics

### Baseline-First ML · Chronological Validation · Leakage Control · Segment Guardrails

**Python · scikit-learn · Ridge · Forecasting · CI**

**Pretoria BI — Data · Intelligence · Performance**

</div>

---

## Decision question

> **Does a trained model beat a simple operational baseline strongly enough, consistently enough and safely enough to justify use on unseen future data?**

This repository implements an actual model-development protocol rather than evaluating pre-filled predictions.

## Modelling protocol

```text
DETERMINISTIC SYNTHETIC BUSINESS SERIES
                 ↓
          FEATURE CONTRACT
                 ↓
      CHRONOLOGICAL TRAIN SPLIT
                 ↓
      LAG-7 OPERATIONAL BASELINE
                 ↓
        RIDGE CANDIDATE MODEL
                 ↓
      VALIDATION ACCEPTANCE GATE
                 ↓
          REFIT TRAIN + VALIDATION
                 ↓
         UNTOUCHED FINAL TEST
                 ↓
   OVERALL + SEGMENT PERFORMANCE GATE
                 ↓
      MULTI-SEED ROBUSTNESS CHECK
```

Features include time trend, lag-7 demand, weekly seasonality, price index, promotion state and segment encoding. Target-derived future fields are explicitly blocked.

## Acceptance rules

The candidate must:

- beat the lag-7 baseline by at least 5% MAE on validation;
- avoid >5% MAE regression on any segment;
- pass before final-test results are accessed;
- pass the same performance guardrails on the untouched final test;
- remain accepted across multiple deterministic seeds.

A deliberately weak zero predictor is rejected in the reverse-test suite.

## Explainability

The final linear model exposes ranked standardised coefficients. They describe model association, not causal effect.

## Model report

```bash
python -m predictive_analytics.cli report
```

Produces `output/predictive_model_report.html` with validation, final-test, multi-seed and coefficient evidence.

## Run locally

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m predictive_analytics.cli smoke
python -m predictive_analytics.cli report
python -m predictive_analytics.cli reverse-test
```

All results are synthetic benchmarks. Production use would require real data, drift monitoring, retraining policy, operational latency constraints and post-deployment outcome measurement.

---

**Pretoria BI — Understand · Decide · Act · Measure**
