# `common/` — shared infrastructure

Reusable machinery used by the core and every extension. Already built and unit-tested.

| Module | Responsibility |
|---|---|
| `data/yfinance_provider.py` | Price/quote access via yfinance (single-ticker today; the multi-ticker universe loader in `selection/universe.py` wraps it). |
| `indicators.py` | SMA20/50/200, returns, MACD, ADX (Wilder). Reused directly as technical features. Reused directly as technical features. |
| `metrics.py` | Finance metrics: cumulative/annualised return, volatility, Sharpe, max drawdown, hit-rate, transaction costs. |
| `walkforward.py` | **Model-agnostic** walk-forward engine (consumes a position function). Drives both the legacy rule strategy and the learned cross-sectional ranker. |
| `backtester.py` | Legacy single-ticker rule-based signal + backtest. Retained as a baseline. |
