# Per-Stock Direction Forecast (Extension B)

Generated at: 2026-07-18 22:19:46

## Configuration

- Model: `baseline`
- Forecast horizon: 5 trading days
- Data period: 5y
- Train window: 252 trading days; step: 5
- Tickers: intended to be the core selector's top-N

## Results (walk-forward, out-of-sample)

| Ticker | N | Accuracy | Base rate | Edge | AUC | Current call |
|---|---:|---:|---:|---:|---:|---|
| AAPL | 217 | 55.8% | 54.4% | +1.4pp | 0.551 | 1.00 (up) |

- **Accuracy** — out-of-sample directional hit rate.
- **Base rate** — "always predict the majority direction"; accuracy must clear it.
- **Edge** — accuracy − base rate, in percentage points (positive = beats naive).
- **AUC** — ranking quality of P(up) vs chance (0.5).
- **Current call** — P(up) over the next 5 days as of the latest close.

Mean edge across tickers: +1.4pp

## Interpretation

Single-stock short-horizon direction is close to a coin flip, so the honest read
is relative: does out-of-sample accuracy beat the base rate, and is AUC above
0.5? A small or negative edge is an acceptable, informative outcome. This signal
is a *confirmation* layer on the core selector's picks, not a standalone strategy,
and is for evaluation only — not financial advice.
