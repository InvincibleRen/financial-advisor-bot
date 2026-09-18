# Robustness / Self-Honesty Report (core selector)

Generated at: 2026-09-18 11:34:12

A single backtest can fool you three ways; each is checked below.

## 1. Not just luck from many trials — Deflated Sharpe Ratio

| Quantity | Value |
|---|---:|
| Observed Sharpe (per-period) | 0.2514 |
| Best-of-30-trials benchmark SR0 | 0.0811 |
| **Deflated Sharpe (prob. true SR beats the benchmark)** | **0.9803** |

The DSR discounts the Sharpe for having tried ~30 configurations. A value
near 1 means the result is very unlikely to be a lucky best-of-many.

## 2. Not a cherry-picked setting — sensitivity grid

| Top-N | Cost (bps) | Precision@N | CAGR | Sharpe | Max DD |
|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 0.500 | 29.4% | 0.93 | -31.0% |
| 3 | 10 | 0.500 | 26.7% | 0.87 | -32.0% |
| 3 | 20 | 0.500 | 24.0% | 0.81 | -33.0% |
| 5 | 0 | 0.508 | 30.2% | 1.06 | -23.3% |
| 5 | 10 | 0.508 | 27.5% | 0.99 | -24.1% |
| 5 | 20 | 0.508 | 24.9% | 0.91 | -24.9% |
| 10 | 0 | 0.511 | 32.2% | 1.23 | -25.8% |
| 10 | 10 | 0.511 | 29.6% | 1.15 | -26.2% |
| 10 | 20 | 0.511 | 27.0% | 1.07 | -26.5% |

## 3. Not carried by a few months — bootstrap 95% CIs (top-3, 10 bps)

| Metric | Median | 95% CI |
|---|---:|---:|
| Annualised return | 26.9% | [7.9%, 49.0%] |
| Annualised Sharpe | 0.88 | [0.39, 1.34] |

## Interpretation

The Sharpe survives deflation for multiple trials, holds across every top-N /
cost cell, and its bootstrap lower bound stays positive — three independent
reasons the edge is unlikely to be an artefact. Out-of-sample research evaluation
only; not financial advice.
