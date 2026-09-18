# Robustness / Self-Honesty Report (core selector)

Generated at: 2026-09-18 18:26:29

A single backtest can fool you three ways; each is checked below.

## 1. Not just luck from many trials — Deflated Sharpe Ratio

| Quantity | Value |
|---|---:|
| Observed Sharpe (per-period) | 0.1734 |
| Best-of-60-trials benchmark SR0 | 0.1240 |
| **Deflated Sharpe (prob. true SR beats the benchmark)** | **0.7170** |

The DSR discounts the Sharpe for having tried ~60 configurations. A value
near 1 means the result is very unlikely to be a lucky best-of-many.

The count of configurations tried is a judgement, so the deflation is also reported across a range of assumptions:

| Assumed trials | Benchmark SR0 | Deflated Sharpe |
|---:|---:|---:|
| 10 | 0.0832 | 0.8523 |
| 30 | 0.1096 | 0.7705 |
| 60 | 0.1240 | 0.7170 |
| 100 | 0.1338 | 0.6773 |

## 2. Not a cherry-picked setting — sensitivity grid

| Top-N | Cost (bps) | Precision@N | CAGR | Sharpe | Max DD |
|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 0.487 | 18.0% | 0.67 | -43.9% |
| 3 | 10 | 0.487 | 15.6% | 0.60 | -45.5% |
| 3 | 20 | 0.487 | 13.1% | 0.54 | -48.5% |
| 5 | 0 | 0.512 | 24.3% | 0.89 | -30.9% |
| 5 | 10 | 0.512 | 21.7% | 0.82 | -31.3% |
| 5 | 20 | 0.512 | 19.3% | 0.75 | -31.7% |
| 10 | 0 | 0.477 | 26.7% | 1.08 | -26.8% |
| 10 | 10 | 0.477 | 24.1% | 1.00 | -27.3% |
| 10 | 20 | 0.477 | 21.6% | 0.91 | -27.9% |

## 3. Not carried by a few months — bootstrap 95% CIs (top-3, 10 bps)

| Metric | Median | 95% CI |
|---|---:|---:|
| Annualised return | 16.0% | [-2.5%, 37.7%] |
| Annualised Sharpe | 0.62 | [0.08, 1.13] |

## Interpretation

The deflated Sharpe is 0.7170 against a best-of-60 benchmark of 0.1240, so on this assumption about the size of the search the observed Sharpe cannot be separated from the best of many attempts. Sharpe stays positive in all 9 top-N / cost cells (range 0.54 to 1.08), so the result is not a single cherry-picked setting. The bootstrap 95% lower bound on the annualised Sharpe is 0.08, which stays above zero. Not every check is passed, and the weaker ones are reported above rather than set aside. Out-of-sample research evaluation only; not financial advice.
