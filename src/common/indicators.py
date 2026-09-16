from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column.")

    result = data.copy()
    result["SMA20"] = result["Close"].rolling(window=20).mean()
    result["SMA50"] = result["Close"].rolling(window=50).mean()
    result["SMA200"] = result["Close"].rolling(window=200).mean()
    return result


def add_returns(data: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column.")

    result = data.copy()
    result["Return"] = result["Close"].pct_change()
    return result


def add_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD line, signal line and histogram (12, 26, 9 by default).

    MACD confirms momentum and the strength of a crossover. A positive and
    rising histogram supports bullish momentum; a negative one supports bearish.
    All three series are causal (exponential moving averages of past closes).
    """
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column.")

    result = data.copy()
    ema_fast = result["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["Close"].ewm(span=slow, adjust=False).mean()

    result["MACD"] = ema_fast - ema_slow
    result["MACD_signal"] = result["MACD"].ewm(span=signal, adjust=False).mean()
    result["MACD_hist"] = result["MACD"] - result["MACD_signal"]
    return result


def add_adx(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add the Average Directional Index (ADX) plus +DI / -DI (Wilder, 14 by default).

    ADX measures **trend strength irrespective of direction**: values above ~25
    indicate a strong (trending) market, below ~20 a weak/ranging one. The
    directional indicators +DI and -DI show *which* side is winning (+DI > -DI is
    an up-trend). ADX is the canonical trend-strength gauge and is most useful as
    a regime filter — a moving-average crossover is far more reliable in a
    high-ADX (trending) market than in a choppy low-ADX one.

    Uses Wilder's smoothing (EWMA with ``alpha = 1/window``) throughout, and is
    fully causal. Requires ``High``, ``Low`` and ``Close``; if any is missing the
    ADX/+DI/-DI columns are filled with NaN rather than raising.
    """
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column.")

    result = data.copy()
    if not {"High", "Low"}.issubset(result.columns):
        result["plus_DI"] = float("nan")
        result["minus_DI"] = float("nan")
        result[f"ADX{window}"] = float("nan")
        return result

    high, low, close = result["High"], result["Low"], result["Close"]
    prev_close = close.shift(1)

    # True Range: the greatest of the three classic spans.
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1.0 / window
    atr = true_range.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr

    di_sum = (plus_di + minus_di).replace(0.0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()

    result["plus_DI"] = plus_di
    result["minus_DI"] = minus_di
    result[f"ADX{window}"] = adx
    return result


def detect_cross_event(data: pd.DataFrame) -> str:
    required_columns = {"SMA50", "SMA200"}

    if not required_columns.issubset(data.columns):
        raise ValueError("Data must contain SMA50 and SMA200 columns.")

    clean_data = data.dropna(subset=["SMA50", "SMA200"])

    if len(clean_data) < 2:
        return "not_enough_data"

    previous = clean_data.iloc[-2]
    latest = clean_data.iloc[-1]

    if previous["SMA50"] <= previous["SMA200"] and latest["SMA50"] > latest["SMA200"]:
        return "golden_cross"

    if previous["SMA50"] >= previous["SMA200"] and latest["SMA50"] < latest["SMA200"]:
        return "death_cross"

    return "none"


def _safe_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def latest_indicator_snapshot(
    historical_data: pd.DataFrame,
    intraday_data: Optional[pd.DataFrame] = None,
    basic: bool = False,
) -> Dict[str, Any]:
    """Build the indicator snapshot used by the advisor.

    With ``basic=True`` only the SMA20/50/200 + crossover indicators are
    produced (MACD and the ADX trend-strength enhancement are omitted). This
    reproduces the report's Chapter 4 prototype exactly, for a scope-faithful
    demo. The default (``basic=False``) adds MACD momentum plus the ADX
    trend-strength reading.
    """
    if basic:
        historical = add_returns(add_moving_averages(historical_data))
    else:
        historical = add_adx(add_macd(add_returns(add_moving_averages(historical_data))))
    latest = historical.iloc[-1]

    cross_event = detect_cross_event(historical)

    latest_close = _safe_float(latest["Close"])
    sma20 = _safe_float(latest["SMA20"])
    sma50 = _safe_float(latest["SMA50"])
    sma200 = _safe_float(latest["SMA200"])

    if basic:
        macd_line = macd_signal = macd_hist = None
        sma_gap_percent = None
        adx = plus_di = minus_di = None
    else:
        macd_line = _safe_float(latest.get("MACD"))
        macd_signal = _safe_float(latest.get("MACD_signal"))
        macd_hist = _safe_float(latest.get("MACD_hist"))

        # SMA gap magnitude (as % of SMA200) measures how strong a crossover is:
        # a wide gap => a strong/decisive cross, a narrow gap => a weak one.
        sma_gap_percent = None
        if sma50 is not None and sma200 not in (None, 0):
            sma_gap_percent = ((sma50 - sma200) / sma200) * 100

        # ADX: trend strength (direction-agnostic). It does not set direction on
        # its own; the advisor uses it to scale how strong the bullish/bearish
        # move is. +DI/-DI say which side currently leads (a cross-check).
        adx = _safe_float(latest.get("ADX14"))
        plus_di = _safe_float(latest.get("plus_DI"))
        minus_di = _safe_float(latest.get("minus_DI"))

    intraday_return_percent = None
    intraday_volatility_percent = None

    if intraday_data is not None and not intraday_data.empty:
        first_price = float(intraday_data["Close"].iloc[0])
        latest_intraday_price = float(intraday_data["Close"].iloc[-1])

        if first_price != 0:
            intraday_return_percent = ((latest_intraday_price / first_price) - 1) * 100

        intraday_returns = intraday_data["Close"].pct_change().dropna()

        if not intraday_returns.empty:
            intraday_volatility_percent = float(intraday_returns.tail(78).std() * 100)

    return {
        "latest_close": latest_close,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "close_above_sma20": latest_close is not None and sma20 is not None and latest_close > sma20,
        "close_above_sma50": latest_close is not None and sma50 is not None and latest_close > sma50,
        "close_above_sma200": latest_close is not None and sma200 is not None and latest_close > sma200,
        "cross_event": cross_event,
        "sma_gap_percent": sma_gap_percent,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "macd_bullish": macd_line is not None and macd_signal is not None and macd_line > macd_signal,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx_trending": adx is not None and adx >= 25,
        "adx_ranging": adx is not None and adx < 20,
        "intraday_return_percent": intraday_return_percent,
        "intraday_volatility_percent": intraday_volatility_percent,
    }
