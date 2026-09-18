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


def test_entry_is_priced_the_trading_day_after_signal():
    # A one-day price spike ON the rebalance date must NOT become the entry price:
    # under T+1 execution the entry is the next trading day's close. Here the spike
    # sits exactly on the signal date; a correct entry ignores it (uses 100, not
    # 1000), so the forward return to the later 130 leg is about +30%, not -87%.
    closes = [100.0] * 20 + [1000.0] + [100.0] * 9 + [130.0] * 40
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    series = {"A": pd.Series(closes, index=idx)}
    signal_date = idx[20]  # the spike day is the rebalance date

    fwd = L._forward_returns_at(series, signal_date, horizon_months=1, execution_lag=1)

    assert fwd["A"] > 0.25  # ~+30% from a 100 entry, not ~-87% from the 1000 spike


def test_default_execution_prices_at_the_signal_close():
    # The default convention (execution_lag=0) transacts at the signal date's own
    # close, so the same spike DOES become the entry price. This documents the
    # simultaneity the T+1 setting exists to remove.
    closes = [100.0] * 20 + [1000.0] + [100.0] * 9 + [130.0] * 40
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    series = {"A": pd.Series(closes, index=idx)}
    signal_date = idx[20]

    fwd = L._forward_returns_at(series, signal_date, horizon_months=1)

    assert fwd["A"] < -0.8  # entered at the 1000 spike, exits near 130


def test_next_trading_day_helper_skips_the_signal_date():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(range(10), index=idx, dtype="float64")
    # Strictly-after semantics: querying an existing date returns the NEXT day.
    assert L._price_next_trading_day(s, idx[3]) == 4.0
    # No trading day after the last date -> None (incomplete forward window).
    assert L._price_next_trading_day(s, idx[-1]) is None
    # A date before the history -> None (not yet trading).
    assert L._price_next_trading_day(s, pd.Timestamp("2019-12-01")) is None
