# Limitations

- The training data are deterministic synthetic business series.
- Ridge regression is intentionally transparent; this repository does not claim it is the optimal algorithm for a real production dataset.
- The lag-7 comparator is an operational baseline, not an exhaustive benchmark suite.
- Standardised coefficients describe model association and are not causal explanations.
- The current robustness check uses three deterministic seeds rather than production drift or out-of-time backtesting across multiple real periods.
- Hyperparameter optimisation, model registry, feature store, drift monitoring, retraining orchestration and online inference are outside scope.
- Synthetic MAE improvements are not production forecasting accuracy or realised business value.
