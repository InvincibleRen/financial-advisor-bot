import pandas as pd

from src.common.indicators import add_macd


def test_macd_columns_and_constant_series():
    data = pd.DataFrame({"Close": [100.0] * 60})
    result = add_macd(data)
    for col in ("MACD", "MACD_signal", "MACD_hist"):
        assert col in result.columns
    # On a flat series MACD and its histogram collapse to zero.
    assert abs(result["MACD"].iloc[-1]) < 1e-9
    assert abs(result["MACD_hist"].iloc[-1]) < 1e-9


def test_macd_histogram_equals_line_minus_signal():
    data = pd.DataFrame({"Close": [float(x) for x in range(1, 80)]})
    result = add_macd(data)
    diff = (result["MACD"] - result["MACD_signal"] - result["MACD_hist"]).abs().max()
    assert diff < 1e-9
