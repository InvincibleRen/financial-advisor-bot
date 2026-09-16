import numpy as np
import pandas as pd

from src.common.indicators import (
    add_adx,
    add_moving_averages,
    add_returns,
)


def test_add_moving_averages_creates_expected_columns():
    data = pd.DataFrame({"Close": list(range(1, 251))})

    result = add_moving_averages(data)

    assert "SMA20" in result.columns
    assert "SMA50" in result.columns
    assert "SMA200" in result.columns
    assert round(result["SMA20"].iloc[-1], 2) == 240.5


def test_add_returns_creates_return_column():
    data = pd.DataFrame({"Close": [100, 110, 121]})

    result = add_returns(data)

    assert "Return" in result.columns
    assert round(result["Return"].iloc[1], 2) == 0.10


# --------------------------------------------------------------------------- #
# ADX                                                                          #
# --------------------------------------------------------------------------- #

def _ohlc_uptrend(n=120, step=1.0):
    close = np.arange(50.0, 50.0 + n * step, step)[:n]
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({"High": high, "Low": low, "Close": close})


def test_adx_strong_uptrend_is_high_and_plus_di_leads():
    result = add_adx(_ohlc_uptrend(), window=14)
    for col in ("ADX14", "plus_DI", "minus_DI"):
        assert col in result.columns
    # A clean, persistent up-trend => strong ADX and +DI clearly above -DI.
    assert result["ADX14"].iloc[-1] > 25
    assert result["plus_DI"].iloc[-1] > result["minus_DI"].iloc[-1]


def test_adx_bounded_and_nan_without_high_low():
    result = add_adx(_ohlc_uptrend(), window=14)
    adx = result["ADX14"].dropna()
    assert (adx >= 0).all() and (adx <= 100).all()

    close_only = add_adx(pd.DataFrame({"Close": [1.0, 2.0, 3.0]}), window=14)
    assert close_only["ADX14"].isna().all()