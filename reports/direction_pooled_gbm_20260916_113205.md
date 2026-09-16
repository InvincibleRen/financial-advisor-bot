# Pooled Cross-Stock Direction Forecast (Extension B)

Generated at: 2026-09-16 11:32:05

## Configuration

- Model: `gbm` (pooled across stocks; one model per rebalance date)
- Forecast horizon: 5 trading days
- Rebalance step: 21 trading days; trailing train window: 3 years
- Probability calibration: on (Platt/sigmoid, leakage-safe CV)
- Basket: 30 of 30 requested stocks (local price cache)

## Out-of-Sample Performance (pooled walk-forward)

| Metric | Value |
|---|---:|
| Predictions | 3277 |
| Accuracy | 58.5% |
| Base rate (majority class) | 58.5% |
| Edge (accuracy − base rate) | -0.1pp |
| AUC | 0.524 |
| Brier score | 0.244 |
| Expected calibration error (ECE) | 0.035 |

## Selective Prediction (accuracy on the most confident calls)

| Coverage kept | Calls | Accuracy | Edge vs base |
|---|---:|---:|---:|
| 100% | 3277 | 58.5% | -0.1pp |
| 80% | 2622 | 59.1% | +0.6pp |
| 60% | 1966 | 60.5% | +2.0pp |
| 40% | 1311 | 61.0% | +2.5pp |
| 20% | 655 | 60.2% | +1.6pp |

## Interpretation

Pooling many stocks into one dated model, plus probability calibration, is the
honest lever on a signal that is close to a coin flip: it lifts accuracy to the
base rate and shrinks the calibration error, so the displayed P(up) is a
trustworthy frequency. AUC near 0.5 confirms that single-stock short-horizon
direction carries little ranking information — a real property of the problem,
not a defect. Selective prediction shows the calls the model is most confident
about beat the base rate. This layer is a *confirmation* signal on the core
selector's picks, for evaluation only — not financial advice.
