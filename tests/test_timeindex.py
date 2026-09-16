"""Tests for the shared timezone-index normaliser (`common/timeindex.py`).

The key regression: a cached yfinance index can contain *mixed* UTC offsets across
a daylight-saving boundary, which plain ``pd.to_datetime`` refuses to convert.
``to_naive_index`` must handle that (and tz-aware / naive inputs) without raising.
"""
import pandas as pd

from src.common.timeindex import to_naive_index


def test_mixed_offset_object_index_does_not_raise():
    # US DST changed on 2021-03-14: before it -05:00, after it -04:00.
    idx = pd.Index(
        [
            pd.Timestamp("2021-03-12 00:00:00-05:00"),
            pd.Timestamp("2021-03-15 00:00:00-04:00"),
        ],
        dtype=object,
    )
    out = to_naive_index(idx)
    assert isinstance(out, pd.DatetimeIndex)
    assert out.tz is None
    assert len(out) == 2


def test_tzaware_index_is_stripped():
    idx = pd.date_range("2022-01-01", periods=3, freq="D", tz="America/New_York")
    out = to_naive_index(idx)
    assert out.tz is None


def test_naive_index_passthrough_values_preserved():
    idx = pd.date_range("2022-01-01", periods=3, freq="D")  # already naive
    out = to_naive_index(idx)
    assert out.tz is None
    assert list(out) == list(idx)


def test_normalize_floors_intraday_time_to_midnight():
    idx = pd.DatetimeIndex(["2022-01-03 14:30:00", "2022-01-04 09:15:00"])
    out = to_naive_index(idx, normalize=True)
    assert (out == pd.DatetimeIndex(["2022-01-03", "2022-01-04"])).all()
