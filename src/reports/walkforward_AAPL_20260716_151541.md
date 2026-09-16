# Walk-Forward Backtest: AAPL

Generated at: 2026-07-16 15:15:41

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
| 1 | 2019-07-17 → 2019-10-14 | 15.88% | 15.34% | 2.26 | -9.25% | 55.56% | 1 |
| 2 | 2019-10-15 → 2020-01-14 | 32.74% | 32.56% | 6.69 | -3.13% | 63.49% | 1 |
| 3 | 2020-01-15 → 2020-04-15 | -23.16% | -9.03% | -1.54 | -26.81% | 40.91% | 2 |
| 4 | 2020-04-16 → 2020-07-15 | 36.21% | 37.43% | 4.80 | -6.39% | 62.90% | 1 |
| 5 | 2020-07-16 → 2020-10-13 | 25.34% | 23.92% | 2.13 | -20.38% | 57.14% | 1 |
| 6 | 2020-10-14 → 2021-01-13 | 7.90% | 8.08% | 1.10 | -10.25% | 47.62% | 1 |
| 7 | 2021-01-14 → 2021-04-15 | 4.23% | 2.76% | 0.68 | -18.72% | 50.79% | 1 |
| 8 | 2021-04-16 → 2021-07-15 | 3.82% | 10.39% | 0.99 | -9.72% | 52.38% | 2 |
| 9 | 2021-07-16 → 2021-10-13 | -3.84% | -5.10% | -0.69 | -11.20% | 50.79% | 1 |
| 10 | 2021-10-14 → 2022-01-12 | 21.98% | 24.57% | 3.35 | -5.50% | 58.73% | 1 |
| 11 | 2022-01-13 → 2022-04-13 | -11.98% | -2.92% | -1.67 | -16.28% | 37.93% | 2 |
| 12 | 2022-04-14 → 2022-07-15 | -5.33% | -11.87% | -2.14 | -6.43% | 25.00% | 1 |
| 13 | 2022-07-18 → 2022-10-13 | -4.32% | -4.78% | -1.05 | -10.83% | 34.62% | 1 |
| 14 | 2022-10-14 → 2023-01-13 | 0.00% | -5.76% | 0.00 | 0.00% | 0.00% | 0 |
| 15 | 2023-01-17 → 2023-04-17 | 9.44% | 22.61% | 1.91 | -6.45% | 52.94% | 1 |
| 16 | 2023-04-18 → 2023-07-18 | 16.26% | 17.25% | 3.63 | -3.04% | 51.61% | 1 |
| 17 | 2023-07-19 → 2023-10-16 | -8.49% | -7.75% | -1.54 | -13.25% | 53.97% | 1 |
| 18 | 2023-10-17 → 2024-01-17 | -3.37% | 2.22% | -0.76 | -8.55% | 44.07% | 2 |
| 19 | 2024-01-18 → 2024-04-17 | -3.54% | -8.04% | -1.55 | -6.69% | 36.36% | 1 |
| 20 | 2024-04-18 → 2024-07-18 | 22.13% | 33.44% | 3.50 | -4.53% | 71.15% | 1 |
| 21 | 2024-07-19 → 2024-10-16 | 3.23% | 3.39% | 0.68 | -7.90% | 57.14% | 1 |
| 22 | 2024-10-17 → 2025-01-17 | -1.03% | -0.78% | -0.12 | -11.88% | 56.45% | 1 |
| 23 | 2025-01-21 → 2025-04-21 | -1.01% | -16.01% | -0.08 | -10.72% | 45.71% | 1 |
| 24 | 2025-04-22 → 2025-07-22 | 0.00% | 11.00% | 0.00 | 0.00% | 0.00% | 0 |
| 25 | 2025-07-23 → 2025-10-20 | 14.23% | 22.31% | 2.55 | -5.42% | 50.98% | 1 |
| 26 | 2025-10-21 → 2026-01-21 | -5.85% | -5.56% | -1.48 | -13.80% | 47.62% | 1 |
| 27 | 2026-01-22 → 2026-04-22 | 1.51% | 10.30% | 0.37 | -12.51% | 49.06% | 2 |

## Aggregated Across Folds (mean of per-fold metrics)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Mean cumulative return (per fold) | 5.30% | 7.41% |
| Mean annualised return | 35.94% | 49.06% |
| Mean Sharpe ratio | 0.82 | 1.27 |
| Mean max drawdown | -9.61% | -12.65% |
| Mean hit rate | 46.48% | 53.52% |

## Overall Out-of-Sample (stitched test windows)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Cumulative return | 222.44% | 434.32% |
| Annualised return | 18.94% | 28.18% |
| Annualised volatility | 24.76% | 31.03% |
| Sharpe ratio | 0.82 | 0.96 |
| Max drawdown | -31.55% | -33.43% |
| Hit rate | 52.16% | 53.51% |
| Days evaluated | 1701 | 1701 |

## Interpretation

This walk-forward evaluation tests the rule-based advisor strategy on a sequence
of unseen windows rather than a single split, reporting standard finance metrics
per fold and aggregated, net of 0.10% per-side costs.
It is for evaluation purposes only and does not guarantee future performance.
