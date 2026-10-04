# Validation Matrix

| Claim | Executable evidence | Failure path | Status |
|---|---|---|---|
| Future/target-derived fields are blocked | `core.validate_features` | inject blocked feature | implemented |
| Development/test order is chronological | `training.chronological_split` | overlap assertions | implemented |
| Candidate is compared with an operational baseline | lag-7 ForecastPoint baseline | weak zero predictor | implemented |
| Validation occurs before final-test use | `training.training_evidence` | candidate returns early if validation fails | implemented |
| Overall MAE must improve materially | `core.acceptance_gate` | minimum-improvement threshold | implemented |
| Segment regressions are constrained | per-segment MAE gate | synthetic segment-regression case | implemented |
| Final candidate is actually trained | scikit-learn StandardScaler + Ridge | training tests | implemented |
| Acceptance is robust across seeds | `multi_seed_validation` | three deterministic seeds | implemented |
| Model associations are inspectable | ranked standardised coefficients | report output | implemented |
| Decision report is generated from measured evidence | `reporting.model_report_html` | CI artefact check | implemented |
| Production accuracy / business impact | no production evidence | not applicable | not claimed |

## Review principle

The final test is not used for candidate selection. A candidate must clear the validation gate before it is refit and evaluated on the untouched final period.
