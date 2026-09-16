# Walk-Forward Backtest: AAPL

Generated at: 2026-07-18 18:15:35

## Configuration

- Data period requested: 8y
- Train window: 252 trading days
- Test window: 63 trading days
- Step: 63 trading days
- Transaction cost: 0.10% per side (per unit turnover)
- Number of folds: 27
- Benchmark: buy-and-hold over each identical test window

## Per-Fold Results (strategy, net of costs)

| Fold | Test window | Strat return | Bench return | Sharpe | Max DD | Hit rate | Trades |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 1 | 2019-07-19 → 2019-10-16 | 15.57% | 13.96% | 2.24 | -9.25% | 53.97% | 1 |
| 2 | 2019-10-17 → 2020-01-16 | 33.85% | 34.51% | 6.85 | -3.13% | 63.49% | 1 |
| 3 | 2020-01-17 → 2020-04-17 | -25.37% | -10.29% | -1.74 | -27.23% | 38.64% | 2 |
| 4 | 2020-04-20 → 2020-07-17 | 39.00% | 36.25% | 5.20 | -4.80% | 62.90% | 1 |
| 5 | 2020-07-20 → 2020-10-15 | 22.60% | 25.31% | 1.95 | -20.38% | 57.14% | 1 |
| 6 | 2020-10-16 → 2021-01-15 | 6.72% | 5.33% | 0.96 | -8.61% | 47.62% | 1 |
| 7 | 2021-01-19 → 2021-04-19 | 5.38% | 6.06% | 0.82 | -18.72% | 50.79% | 1 |
| 8 | 2021-04-20 → 2021-07-19 | 0.39% | 5.64% | 0.18 | -9.64% | 50.00% | 2 |
| 9 | 2021-07-20 → 2021-10-15 | -1.00% | 1.68% | -0.12 | -11.20% | 52.38% | 1 |
| 10 | 2021-10-18 → 2022-01-14 | 17.98% | 19.49% | 2.77 | -5.50% | 57.14% | 1 |
| 11 | 2022-01-18 → 2022-04-18 | -13.53% | -4.62% | -1.90 | -16.62% | 36.21% | 2 |
| 12 | 2022-04-19 → 2022-07-19 | -6.52% | -8.52% | -2.78 | -6.43% | 16.67% | 1 |
| 13 | 2022-07-20 → 2022-10-17 | -4.32% | -5.69% | -1.05 | -10.83% | 34.62% | 1 |
| 14 | 2022-10-18 → 2023-01-18 | 0.00% | -5.06% | 0.00 | 0.00% | 0.00% | 0 |
| 15 | 2023-01-19 → 2023-04-19 | 11.03% | 23.98% | 2.19 | -6.45% | 54.72% | 1 |
| 16 | 2023-04-20 → 2023-07-20 | 15.77% | 15.21% | 3.51 | -3.04% | 51.61% | 1 |
| 17 | 2023-07-21 → 2023-10-18 | -8.48% | -8.95% | -1.54 | -13.25% | 53.97% | 1 |
| 18 | 2023-10-19 → 2024-01-19 | 2.30% | 8.94% | 0.60 | -8.55% | 47.46% | 2 |
| 19 | 2024-01-22 → 2024-04-19 | -6.16% | -13.87% | -3.13 | -6.69% | 30.00% | 1 |
| 20 | 2024-04-22 → 2024-07-22 | 22.01% | 35.73% | 3.48 | -4.62% | 70.37% | 1 |
| 21 | 2024-07-23 → 2024-10-18 | 4.34% | 4.93% | 0.87 | -7.90% | 58.73% | 1 |
| 22 | 2024-10-21 → 2025-01-22 | -5.44% | -4.75% | -1.02 | -14.05% | 54.84% | 1 |
| 23 | 2025-01-23 → 2025-04-23 | -1.46% | -8.59% | -0.17 | -10.72% | 45.45% | 1 |
| 24 | 2025-04-24 → 2025-07-24 | 0.00% | 4.48% | 0.00 | 0.00% | 0.00% | 0 |
| 25 | 2025-07-25 → 2025-10-22 | 12.58% | 20.91% | 2.26 | -5.42% | 50.94% | 1 |
| 26 | 2025-10-23 → 2026-01-23 | -4.54% | -4.03% | -1.15 | -13.80% | 47.62% | 1 |
| 27 | 2026-01-26 → 2026-04-24 | -2.05% | 9.28% | -0.24 | -12.51% | 49.06% | 2 |

## Aggregated Across Folds (mean of per-fold metrics)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Mean cumulative return (per fold) | 4.84% | 7.31% |
| Mean annualised return | 34.41% | 48.15% |
| Mean Sharpe ratio | 0.70 | 1.23 |
| Mean max drawdown | -9.60% | -12.66% |
| Mean hit rate | 45.79% | 53.51% |

## Overall Out-of-Sample (stitched test windows)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Cumulative return | 182.90% | 427.20% |
| Annualised return | 16.66% | 27.93% |
| Annualised volatility | 24.78% | 31.03% |
| Sharpe ratio | 0.75 | 0.95 |
| Max drawdown | -29.55% | -33.43% |
| Hit rate | 52.01% | 53.51% |
| Days evaluated | 1701 | 1701 |

## Interpretation

This walk-forward evaluation tests the rule-based advisor strategy on a sequence
of unseen windows rather than a single split, reporting standard finance metrics
per fold and aggregated, net of 0.10% per-side costs.
It is for evaluation purposes only and does not guarantee future performance.
