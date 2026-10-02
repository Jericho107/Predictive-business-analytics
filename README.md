<div align="center">

# Predictive Business Analytics

### Baseline-first predictive analytics with leakage controls and explicit model acceptance gates.

**Pretoria BI — Data · Intelligence · Performance**

</div>

---

## Management question

> **Does a candidate model improve enough over a simple baseline to justify operational use, without hiding segment-level regressions or target leakage?**

**All data and entities are synthetic. No client result or realised ROI is claimed.**

---

## What this repository proves

- Baseline-first evaluation
- Leakage blacklist
- Untouched evaluation contract
- Overall MAE improvement gate
- Segment regression guardrail

The objective is not to inflate a portfolio with screenshots. The repository has an executable happy path and deliberately corrupted states that must be rejected.

## Evidence chain

```text
SIGNAL → CONTRACT → VALIDATION → ANALYSIS → DECISION RULE → ACTION OWNER → FOLLOW-UP
```

## Run locally

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m predictive_analytics.cli smoke
python -m predictive_analytics.cli reverse-test
```

## Repository map

```text
predictive-business-analytics/
├── .github/workflows/ci.yml
├── config/
├── docs/
├── sql/
├── src/predictive_analytics/
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Proof boundary

Implemented evidence is separated from future production claims. See `docs/proof_matrix.md` and `docs/limitations.md`. Thresholds in this synthetic case are examples to demonstrate governance and must be calibrated before real deployment.

---

<div align="center">

**Pretoria BI**  
**Understand · Decide · Act · Measure**

</div>
