"""Offline tests for the Phase 1 forward-return labelling.

Deterministic synthetic prices; no network. The labels are a *training target*,
so the tests focus on: correct relative-to-median logic, a clean MultiIndex, and
that an incomplete forward window drops the example rather than truncating it.
"""
import numpy as np
import pandas as pd

from src.selection import labels as L


def _prices_from_closes(closes_by_ticker, start="2020-01-01"):
    """Build ``{ticker -> OHLCV frame}`` from lists of daily closes."""
    prices = {}
    for ticker, closes in closes_by_ticker.items():
        idx = pd.date_range(start, periods=len(closes), freq="D")
        prices[ticker] = pd.DataFrame({"Close": closes}, index=idx)
    return prices


def test_labels_beat_median_are_one():
    # Over the horizon: A +50%, B +10%, C -20%. Median forward return = +10% (B).
    # Strictly greater than median -> only A is labelled 1.
    closes = {
        "A": [100.0] * 40 + [150.0] * 40,
        "B": [100.0] * 40 + [110.0] * 40,
        "C": [100.0] * 40 + [80.0] * 40,
    }
    prices = _prices_from_closes(closes)
    t = pd.Timestamp("2020-01-15")  # entry within the flat first leg

    labels = L.make_labels(prices, [t], horizon_months=1)

    assert labels.loc[(t, "A")] == 1
    assert labels.loc[(t, "B")] == 0  # equals the median, not strictly greater
    assert labels.loc[(t, "C")] == 0


def test_label_index_is_named_multiindex():
    prices = _prices_from_closes({"A": [10.0] * 80, "B": [20.0] * 80})
    labels = L.make_labels(prices, [pd.Timestamp("2020-01-10")], horizon_months=1)
    assert list(labels.index.names) == ["rebalance_date", "ticker"]
    assert labels.name == "label"


def test_incomplete_forward_window_is_dropped():
    # 40 days of history; a 1-month horizon from near the end runs past the data,
    # so that rebalance date yields no labels (rather than a truncated return).
    prices = _prices_from_closes({"A": list(np.linspace(100, 140, 40))})
    near_end = pd.Timestamp("2020-02-05")
    labels = L.make_labels(prices, [near_end], horizon_months=1)
    assert (near_end, "A") not in labels.index


def test_entry_before_history_is_dropped():
    prices = _prices_from_closes({"A": [100.0] * 80}, start="2020-06-01")
    too_early = pd.Timestamp("2020-01-01")  # before this ticker's first close
    labels = L.make_labels(prices, [too_early], horizon_months=1)
    assert too_early not in labels.index.get_level_values("rebalance_date")


def test_labels_are_as_of_non_trading_dates():
    # Weekend rebalance date must fall back to the last available close.
    closes = {"A": [100.0] * 30 + [130.0] * 30, "B": [100.0] * 60}
    prices = _prices_from_closes(closes)
    saturday = pd.Timestamp("2020-01-04")  # a Saturday; entry falls in the flat leg
    labels = L.make_labels(prices, [saturday], horizon_months=1)
    # A rallies, B is flat -> A beats the median, B does not.
    assert labels.loc[(saturday, "A")] == 1
    assert labels.loc[(saturday, "B")] == 0
