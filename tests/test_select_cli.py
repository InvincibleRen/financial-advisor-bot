"""Offline integration test for the Phase 2 selection CLI.

The universe loaders are monkeypatched with synthetic in-memory data, so the full
chain — load -> feature matrix -> labels -> walk-forward backtest -> report — runs
end to end without any network access.
"""
import numpy as np
import pandas as pd

from src.cli import select_cli as C
from src.selection import universe as U


def _synthetic_prices(tickers, start=None, end=None, **_):
    """Business-day price paths with distinct per-ticker drifts."""
    idx = pd.date_range("2015-01-01", periods=1500, freq="B")
    drifts = {"AAA": 0.0007, "BBB": 0.0005, "CCC": 0.0002, "DDD": 0.0, "SPY": 0.0004}
    out = {}
    for t in tickers:
        drift = drifts.get(t, 0.0003)
        closes = 100.0 * np.exp(np.cumsum(np.full(len(idx), drift)))
        out[t] = pd.DataFrame({"Close": closes}, index=idx)
    return out


def _synthetic_fundamentals(tickers=None, **_):
    tickers = tickers or ["AAA", "BBB", "CCC", "DDD"]
    frames = []
    for t in tickers:
        periods = pd.date_range("2014-03-31", periods=24, freq="QE")
        frames.append(pd.DataFrame({
            "ticker": t,
            "period_end": periods,
            "available_date": periods + pd.Timedelta(days=60),
            "net_income": 10.0, "total_revenue": 100.0, "total_equity": 200.0,
            "total_debt": 100.0, "shares_outstanding": 1000.0,
            "roe": 0.2, "net_margin": 0.1, "debt_to_equity": 0.5, "earnings_growth_yoy": 0.05,
        }))
    return pd.concat(frames, ignore_index=True)


def test_cli_runs_end_to_end_offline(monkeypatch, capsys):
    monkeypatch.setattr(U, "load_prices", _synthetic_prices)
    monkeypatch.setattr(U, "load_fundamentals", _synthetic_fundamentals)

    # No report written (keeps the test filesystem clean); exercise both rankers.
    C.main(["AAA", "BBB", "CCC", "DDD", "--no-save", "--min-train", "6", "--model", "both"])

    out = capsys.readouterr().out
    assert "Cross-Sectional Selection" in out
    assert "gbm selector" in out
    assert "logistic selector" in out
    assert "equal-wt" in out
