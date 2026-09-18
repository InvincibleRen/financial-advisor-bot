"""Forward-return labelling for cross-sectional selection (Direction 1 core, Phase 1).

Frames the task as *relative* selection rather than absolute forecasting: a
stock is a positive example if its forward return over the horizon beats the
universe median that period.

Design notes
------------
* **The label is a training target only.** The forward window ``t -> t+horizon``
  is, by definition, unknown at prediction time. It is therefore never exposed as
  a feature (see ``features.py``); it exists purely to supervise the ranker.
* **Relative, not absolute.** Labelling against the *cross-sectional median*
  makes the task market-neutral: in a bull month roughly half the universe is
  still labelled 0, so the model learns *selection* skill rather than simply
  learning "the market went up".
* **As-of pricing.** Rebalance dates need not be trading days, so entry and exit
  prices are read *as-of* (the last close on or before the date). A ticker is
  dropped for a given rebalance when either price is unavailable — in particular
  when ``t + horizon`` runs past the end of that ticker's history, which would
  otherwise silently truncate the forward window.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.common.timeindex import to_naive_index


def make_labels(
    prices: Dict[str, pd.DataFrame],
    rebalance_dates: List[pd.Timestamp],
    horizon_months: int = 1,
) -> pd.Series:
    """Return a binary label Series indexed by ``(rebalance_date, ticker)``.

    ``label = 1`` if the stock's forward return from ``t`` to ``t + horizon`` is
    strictly greater than the cross-sectional median forward return of the
    universe at ``t``; else ``0``.

    Parameters
    ----------
    prices:
        ``{ticker -> daily OHLCV DataFrame}`` (as produced by
        ``universe.load_prices``). Only the ``Close`` column is used.
    rebalance_dates:
        The dates at which the portfolio is re-selected.
    horizon_months:
        Holding period, in calendar months, used to measure the forward return.

    Notes
    -----
    Tickers whose entry or exit price is unavailable at a given rebalance date are
    excluded from that date's cross-section (and therefore from the median).
    """
    closes = _close_series_by_ticker(prices)

    labels: Dict[tuple, int] = {}
    for rebalance_date in pd.to_datetime(list(rebalance_dates)):
        forward_returns = _forward_returns_at(closes, rebalance_date, horizon_months)
        if forward_returns.empty:
            continue
        median = forward_returns.median()
        for ticker, fwd_ret in forward_returns.items():
            labels[(rebalance_date, ticker)] = int(fwd_ret > median)

    return _as_labelled_series(labels)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _close_series_by_ticker(prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Extract a clean, sorted, tz-naive ``Close`` Series per ticker."""
    closes: Dict[str, pd.Series] = {}
    for ticker, frame in prices.items():
        if frame is None or "Close" not in frame.columns or frame.empty:
            continue
        series = frame["Close"].dropna()
        series.index = to_naive_index(series.index, normalize=True)
        closes[ticker] = series.sort_index()
    return closes


def _forward_returns_at(
    closes: Dict[str, pd.Series],
    rebalance_date: pd.Timestamp,
    horizon_months: int,
) -> pd.Series:
    """Forward return per ticker over the **tradeable** window that opens on the
    trading day after ``t`` and closes on the trading day after ``t + horizon``.

    The signal is formed at ``t`` from data up to and including ``t``, but a
    position can only be entered once ``t``'s close is actually observed, so both
    the entry and the exit are priced at the next available close (T+1 execution).
    This removes the point-in-time simultaneity of entering at the very close used
    to form the decision. The example is dropped when either leg has no next
    trading day (an incomplete forward window) or when ``t`` predates the ticker.
    """
    exit_date = rebalance_date + pd.DateOffset(months=horizon_months)

    returns: Dict[str, float] = {}
    for ticker, series in closes.items():
        entry_price = _price_next_trading_day(series, rebalance_date)
        exit_price = _price_next_trading_day(series, exit_date)
        if entry_price is None or exit_price is None or entry_price == 0:
            continue
        returns[ticker] = (exit_price / entry_price) - 1.0

    return pd.Series(returns, dtype="float64")


def _price_next_trading_day(series: pd.Series, when: pd.Timestamp) -> float | None:
    """Close on the first trading day **strictly after** ``when`` (T+1 execution).

    Returns ``None`` when ``when`` predates the ticker's history (the stock was not
    trading at the decision date) or when no trading day follows ``when`` (the
    forward window runs past the available data).
    """
    if when < series.index.min():
        return None
    pos = series.index.searchsorted(when, side="right")
    if pos >= len(series):
        return None
    value = series.iloc[pos]
    return None if pd.isna(value) else float(value)


def _as_labelled_series(labels: Dict[tuple, int]) -> pd.Series:
    """Build the MultiIndex ``(rebalance_date, ticker)`` label Series."""
    if not labels:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["rebalance_date", "ticker"],
        )
        return pd.Series([], index=empty_index, name="label", dtype="int64")

    index = pd.MultiIndex.from_tuples(labels.keys(), names=["rebalance_date", "ticker"])
    return pd.Series(list(labels.values()), index=index, name="label", dtype="int64").sort_index()
