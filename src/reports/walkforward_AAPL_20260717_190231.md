# Walk-Forward Backtest: AAPL

Generated at: 2026-07-17 19:02:31

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
| 1 | 2019-07-18 → 2019-10-15 | 14.31% | 15.72% | 2.07 | -9.25% | 53.97% | 1 |
| 2 | 2019-10-16 → 2020-01-15 | 32.71% | 32.30% | 6.68 | -3.13% | 63.49% | 1 |
| 3 | 2020-01-16 → 2020-04-16 | -23.50% | -7.92% | -1.57 | -26.81% | 40.91% | 2 |
| 4 | 2020-04-17 → 2020-07-16 | 36.39% | 34.67% | 4.82 | -5.10% | 62.90% | 1 |
| 5 | 2020-07-17 → 2020-10-14 | 25.68% | 25.56% | 2.15 | -20.38% | 58.73% | 1 |
| 6 | 2020-10-15 → 2021-01-14 | 6.69% | 6.37% | 0.96 | -9.89% | 47.62% | 1 |
| 7 | 2021-01-15 → 2021-04-16 | 5.42% | 4.07% | 0.82 | -18.72% | 50.79% | 1 |
| 8 | 2021-04-19 → 2021-07-16 | 1.84% | 9.12% | 0.52 | -9.72% | 50.00% | 2 |
| 9 | 2021-07-19 → 2021-10-14 | 0.82% | -1.80% | 0.26 | -11.20% | 52.38% | 1 |
| 10 | 2021-10-15 → 2022-01-13 | 18.76% | 19.78% | 2.87 | -5.50% | 57.14% | 1 |
| 11 | 2022-01-14 → 2022-04-14 | -15.05% | -4.01% | -2.13 | -16.51% | 36.21% | 2 |
| 12 | 2022-04-18 → 2022-07-18 | -5.20% | -11.02% | -2.08 | -6.43% | 28.57% | 1 |
| 13 | 2022-07-19 → 2022-10-14 | -4.32% | -5.91% | -1.05 | -10.83% | 34.62% | 1 |
| 14 | 2022-10-17 → 2023-01-17 | 0.00% | -1.76% | 0.00 | 0.00% | 0.00% | 0 |
| 15 | 2023-01-18 → 2023-04-18 | 10.27% | 22.46% | 2.05 | -6.45% | 53.85% | 1 |
| 16 | 2023-04-19 → 2023-07-19 | 16.27% | 17.20% | 3.63 | -3.04% | 51.61% | 1 |
| 17 | 2023-07-20 → 2023-10-17 | -8.37% | -9.20% | -1.52 | -13.25% | 53.97% | 1 |
| 18 | 2023-10-18 → 2024-01-18 | 0.52% | 6.48% | 0.20 | -8.55% | 45.76% | 2 |
| 19 | 2024-01-19 → 2024-04-18 | -5.02% | -11.45% | -2.40 | -6.69% | 33.33% | 1 |
| 20 | 2024-04-19 → 2024-07-19 | 22.20% | 34.29% | 3.51 | -4.53% | 71.70% | 1 |
| 21 | 2024-07-22 → 2024-10-17 | 3.55% | 3.50% | 0.74 | -7.90% | 58.73% | 1 |
| 22 | 2024-10-18 → 2025-01-21 | -5.35% | -4.10% | -1.00 | -14.05% | 54.84% | 1 |
| 23 | 2025-01-22 → 2025-04-22 | -1.54% | -10.29% | -0.18 | -10.72% | 44.12% | 1 |
| 24 | 2025-04-23 → 2025-07-23 | 0.00% | 7.21% | 0.00 | 0.00% | 0.00% | 0 |
| 25 | 2025-07-24 → 2025-10-21 | 14.46% | 22.70% | 2.59 | -5.42% | 51.92% | 1 |
| 26 | 2025-10-22 → 2026-01-22 | -4.00% | -5.49% | -1.00 | -13.80% | 49.21% | 1 |
| 27 | 2026-01-23 → 2026-04-23 | 1.74% | 10.10% | 0.40 | -12.51% | 50.94% | 2 |

## Aggregated Across Folds (mean of per-fold metrics)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Mean cumulative return (per fold) | 5.16% | 7.36% |
| Mean annualised return | 35.21% | 47.65% |
| Mean Sharpe ratio | 0.79 | 1.23 |
| Mean max drawdown | -9.64% | -12.68% |
| Mean hit rate | 46.57% | 53.57% |

## Overall Out-of-Sample (stitched test windows)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Cumulative return | 210.78% | 437.85% |
| Annualised return | 18.29% | 28.31% |
| Annualised volatility | 24.83% | 31.03% |
| Sharpe ratio | 0.80 | 0.96 |
| Max drawdown | -30.06% | -33.43% |
| Hit rate | 52.32% | 53.57% |
| Days evaluated | 1701 | 1701 |

## Interpretation

This walk-forward evaluation tests the rule-based advisor strategy on a sequence
of unseen windows rather than a single split, reporting standard finance metrics
per fold and aggregated, net of 0.10% per-side costs.
It is for evaluation purposes only and does not guarantee future performance.
