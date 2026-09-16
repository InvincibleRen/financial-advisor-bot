# Cross-Sectional Stock Selection (Direction 1 core)

Generated at: 2026-07-27 13:13:53

## Configuration

- Universe: 27 tickers (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, JPM, V, MA, UNH, HD, PG, JNJ, XOM, CVX, KO, PEP, COST, WMT, MRK, ABBV, CRM, AMD, NFLX, INTC)
- Rebalance: monthly (138 dates)
- Top-N held each rebalance: 3
- Label: forward 1-month return vs cross-sectional median (leakage-safe)
- Feature normalisation: cross-sectional rank per rebalance date (leakage-safe)
- Transaction cost: 10 bps per unit turnover (strategy only; benchmarks are buy-and-hold)
- Benchmarks: equal-weight universe, and SPY buy-and-hold

## Out-of-Sample Performance (walk-forward, monthly, net of costs)

| Portfolio | Cumulative | Annualised | Sharpe | Max DD | Hit rate |
|---|---:|---:|---:|---:|---:|
| Selector — gbm | 2766.35% | 35.99% | 1.12 | -39.42% | 69.47% |
| Selector — logistic | 8187.31% | 49.88% | 1.39 | -36.56% | 66.41% |
| Equal-weight universe | 817.44% | 22.51% | 1.35 | -23.44% | 70.23% |
| SPY (buy & hold) | 254.76% | 12.30% | 0.83 | -24.80% | 66.41% |

## Turnover

- **gbm** — average monthly turnover 1.54 (1.0 = one full position replaced)
- **logistic** — average monthly turnover 1.13 (1.0 = one full position replaced)

## Prediction Quality (judging the ranker, not the portfolio)

| Model | Precision@N | Mean AUC | Base rate | Monthly win-rate vs bench |
|---|---:|---:|---:|---:|
| gbm | 0.545 | 0.525 | 0.481 | 57.25% |
| logistic | 0.555 | 0.515 | 0.481 | 61.83% |

*Precision@N = fraction of picked names that then beat the universe median (skill
shows as precision above the ~0.5 base rate). Mean AUC = average per-rebalance
ranking quality of the scores (0.5 = chance). Monthly win-rate = fraction of
months the strategy return beat the benchmark.*

## Consistency — Per-Year Strategy vs Benchmark

**gbm**

| Year | Strategy | Benchmark | Periods | Win-rate |
|---|---:|---:|---:|---:|
| 2015 | 6.71% | 0.63% | 6 | 66.67% |
| 2016 | 36.41% | 33.57% | 12 | 50.00% |
| 2017 | 46.26% | 40.97% | 12 | 58.33% |
| 2018 | 23.74% | 5.23% | 12 | 66.67% |
| 2019 | 51.93% | 31.32% | 12 | 75.00% |
| 2020 | 70.39% | 29.77% | 12 | 58.33% |
| 2021 | -2.31% | 27.49% | 12 | 33.33% |
| 2022 | -9.99% | -6.86% | 12 | 41.67% |
| 2023 | 51.93% | 37.14% | 12 | 66.67% |
| 2024 | 20.39% | 25.13% | 12 | 50.00% |
| 2025 | 53.83% | 18.40% | 12 | 58.33% |
| 2026 | 69.89% | 11.92% | 5 | 80.00% |

**logistic**

| Year | Strategy | Benchmark | Periods | Win-rate |
|---|---:|---:|---:|---:|
| 2015 | 5.38% | 0.63% | 6 | 66.67% |
| 2016 | 63.71% | 33.57% | 12 | 66.67% |
| 2017 | 56.58% | 40.97% | 12 | 50.00% |
| 2018 | -17.11% | 5.23% | 12 | 50.00% |
| 2019 | 77.05% | 31.32% | 12 | 75.00% |
| 2020 | 169.77% | 29.77% | 12 | 83.33% |
| 2021 | 1.51% | 27.49% | 12 | 50.00% |
| 2022 | -4.53% | -6.86% | 12 | 41.67% |
| 2023 | 101.98% | 37.14% | 12 | 66.67% |
| 2024 | 83.54% | 25.13% | 12 | 66.67% |
| 2025 | 32.45% | 18.40% | 12 | 58.33% |
| 2026 | 62.85% | 11.92% | 5 | 80.00% |

## Latest Selection

- **gbm** — latest selection 2026-06-30: INTC, UNH, XOM
- **logistic** — latest selection 2026-06-30: CVX, INTC, MRK

## Interpretation

Each month the ranker is trained only on data strictly before the rebalance date,
scores the current cross-section and holds the top-3 equally until the next
rebalance. Strategy returns are **net of 10 bps per-turnover costs**;
the passive benchmarks are reported gross (buy-and-hold), so the actively-traded
selector must clear its trading costs to win. The gradient-boosting ranker is
shown alongside the logistic baseline so any edge over a simple linear model is
explicit. This is an out-of-sample research evaluation only and does not guarantee
future performance.
