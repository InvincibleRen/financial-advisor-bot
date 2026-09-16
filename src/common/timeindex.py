"""Robust datetime-index normalisation shared across the pipeline.

Real market data (yfinance) is timezone-aware (e.g. America/New_York). Once it is
cached to CSV and read back, a single ticker's index can contain *mixed* UTC
offsets across daylight-saving boundaries (``-05:00`` in winter, ``-04:00`` in
summer). ``pd.to_datetime`` refuses to build a ``datetime64`` array from such a
mixed-offset object array unless told to coerce to UTC — which is exactly the
error that surfaces the first time the selector runs on live data.

Every module that samples prices *as-of* a rebalance date must therefore reduce
indices to a single, comparable, timezone-naive representation. Centralising that
here keeps the rule in one place instead of three subtly different local copies.
"""
from __future__ import annotations

import pandas as pd


def to_naive_index(index, normalize: bool = False) -> pd.DatetimeIndex:
    """Return a timezone-naive ``DatetimeIndex`` from any datetime-like input.

    Parameters
    ----------
    index:
        A ``DatetimeIndex``, an object array of (possibly mixed-offset) timestamps,
        or anything ``pd.to_datetime`` accepts.
    normalize:
        When ``True``, floor to midnight — appropriate for daily price bars, whose
        intraday timestamp is noise and whose only meaningful key is the calendar
        day. This also keeps ``asof`` at a midnight rebalance date matching the
        same day's close rather than silently taking the prior day.

    Notes
    -----
    Mixed offsets are resolved by coercing to UTC (``utc=True``) and then dropping
    the timezone, giving a stable wall-clock value. Naive input is treated as
    already-UTC and returned unchanged (bar an optional ``normalize``).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).tz_localize(None)
    return idx.normalize() if normalize else idx
