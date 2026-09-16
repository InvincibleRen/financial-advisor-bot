"""Leakage-safe feature matrix construction (Direction 1 core, Phase 1).

Builds one feature row per ``(rebalance_date, ticker)`` combining technical and
fundamental features. The sentiment feature (Extension A) is merged here when
available, but the module works without it.

Leakage safety (the central invariant)
---------------------------------------
Every value at date ``t`` is computable from information available strictly up to
and including ``t``:

* **Technical** features (momentum, volatility, MACD, ADX) are causal — each uses
  only past and current prices — and are sampled *as-of* ``t`` (the last trading
  day on or before ``t``).
* **Fundamental** features are joined *as-of* each row's ``available_date``
  (period-end + reporting lag, computed in Phase 0), never its ``period_end``.
  Only filings that were already public at ``t`` can be used.
* **Sentiment** (optional) is left-joined on ``(t, ticker)``; the caller is
  responsible for only including headlines published on or before ``t``.

Because construction is strictly as-of, injecting a *future* price spike or a
*future* earnings release leaves every row at or before ``t`` unchanged — the
property the Phase 1 tests assert directly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.indicators import add_adx, add_macd
from src.common.timeindex import to_naive_index

# Fundamental feature columns produced by this module.
FUNDAMENTAL_FEATURES: List[str] = [
    "pe", "pb", "roe", "earnings_growth", "net_margin", "debt_to_equity",
]
# Technical feature columns produced by this module.
TECHNICAL_FEATURES: List[str] = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "reversal_5d",
    "mom_risk_adj", "high_52w",
    "volatility", "macd", "adx",
]
SENTIMENT_FEATURE = "sentiment"

# Trading-day momentum windows as ``(lookback, skip)`` in trading days, where the
# return is measured from ``lookback`` days ago up to ``skip`` days ago (skip=0 is
# a plain trailing return). ``mom_12m`` uses the classic **12-1** construction
# (skip the most recent ~1 month): the well-documented short-term reversal in the
# latest month otherwise contaminates the medium-term momentum signal, so skipping
# it is standard practice (Jegadeesh & Titman) and keeps ``mom_12m`` genuinely
# orthogonal to ``reversal_5d`` / ``mom_1m``. US markets have ~21 trading days per
# calendar month.
_MOMENTUM_WINDOWS = {"mom_1m": (21, 0), "mom_3m": (63, 0), "mom_6m": (126, 0), "mom_12m": (252, 21)}
_VOLATILITY_WINDOW = 21
_REVERSAL_WINDOW = 5
_HIGH_WINDOW = 252  # trailing 1-year window for the 52-week-high proximity feature


def build_feature_matrix(
    prices: Dict[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    sentiment: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a feature matrix indexed by ``(rebalance_date, ticker)``.

    Parameters
    ----------
    prices:
        ``{ticker -> daily OHLCV DataFrame}`` (from ``universe.load_prices``).
    fundamentals:
        Tidy, leakage-aware fundamentals table (from
        ``universe.load_fundamentals``): one row per ``(ticker, period_end)`` with
        an ``available_date`` and the derived ratios.
    rebalance_dates:
        Dates at which features are sampled.
    sentiment:
        Optional Extension-A feature. Accepted either as a frame indexed by
        ``(rebalance_date, ticker)`` with a ``sentiment`` column, or as a tidy
        frame with ``rebalance_date``, ``ticker`` and ``sentiment`` columns.

    Returns
    -------
    A tidy ``MultiIndex`` frame (levels ``rebalance_date``, ``ticker``) with the
    columns ``TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES`` (plus ``sentiment`` when
    provided). Rows are omitted only when a ticker has no price on or before ``t``.
    """
    rebalance_dates = pd.to_datetime(list(rebalance_dates))
    technicals = {t: _technical_frame(f) for t, f in prices.items() if _has_close(f)}
    fundamentals_by_ticker = _augmented_fundamentals_by_ticker(fundamentals)

    rows: Dict[tuple, Dict[str, float]] = {}
    for rebalance_date in rebalance_dates:
        for ticker, tech in technicals.items():
            tech_asof = _row_asof(tech, rebalance_date)
            if tech_asof is None or pd.isna(tech_asof.get("close")):
                continue

            fund_asof = _fundamentals_asof(fundamentals_by_ticker.get(ticker), rebalance_date)
            features = {name: _to_float(tech_asof.get(name)) for name in TECHNICAL_FEATURES}
            features.update(_fundamental_features(fund_asof, float(tech_asof["close"])))
            rows[(rebalance_date, ticker)] = features

    matrix = _rows_to_matrix(rows)
    if sentiment is not None:
        matrix = _merge_sentiment(matrix, sentiment)
    return matrix


# --------------------------------------------------------------------------- #
# Technical features                                                          #
# --------------------------------------------------------------------------- #

def _technical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker daily frame of causal technical features (+ ``close``).

    Indexed by a sorted, tz-naive DatetimeIndex so it can be sampled with
    ``.asof(t)`` at arbitrary (possibly non-trading) rebalance dates.
    """
    # Keep the OHLC columns that are present (High/Low power ADX); a close-only
    # frame still works — ADX simply comes out as NaN.
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    base = frame[keep].copy()
    base.index = to_naive_index(base.index, normalize=True)
    base = base[~base.index.duplicated(keep="last")].sort_index()
    close = base["Close"]

    out = pd.DataFrame(index=base.index)
    for name, (window, skip) in _MOMENTUM_WINDOWS.items():
        # Return from ``window`` days ago to ``skip`` days ago; skip=0 is a plain
        # trailing return, skip>0 gives the 12-1 style skip-a-month momentum.
        out[name] = close.shift(skip) / close.shift(window) - 1.0
    out["volatility"] = close.pct_change().rolling(_VOLATILITY_WINDOW).std()
    # Short-term (1-week) reversal: recent 5-day return. Distinct window from the
    # 21d+ momentum factors, so it carries genuinely orthogonal information a tree
    # cannot recover from the longer-horizon momenta.
    out["reversal_5d"] = close / close.shift(_REVERSAL_WINDOW) - 1.0
    # Risk-adjusted momentum: 12-1 momentum scaled by recent volatility. Dividing by
    # risk favours steadier trends over volatile ones and is empirically a more
    # robust cross-sectional signal than raw momentum. (Scale is irrelevant once the
    # column is cross-sectionally rank-normalised; only the ordering matters.)
    out["mom_risk_adj"] = out["mom_12m"] / out["volatility"].where(out["volatility"] > 0)
    # 52-week-high proximity: last close divided by the trailing 252-day high, in
    # (0, 1]. Nearness to the 1-year high is a well-documented cross-sectional
    # predictor (George & Hwang), distinct from a trailing return.
    out["high_52w"] = close / close.rolling(_HIGH_WINDOW, min_periods=1).max()
    out["macd"] = add_macd(base)["MACD"]
    # ADX (trend strength) is already 0-100 and cross-sectionally comparable.
    out["adx"] = add_adx(base)["ADX14"]
    out["close"] = close
    return out


# --------------------------------------------------------------------------- #
# Fundamental features                                                         #
# --------------------------------------------------------------------------- #

def _augmented_fundamentals_by_ticker(fundamentals: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group fundamentals by ticker and ensure a trailing-12-month net income.

    TTM net income gives a stable earnings base for the price/earnings ratio. When
    the loader already supplies a leakage-safe ``ttm_net_income`` (it computes it
    per frequency, so annual rows carry the annual figure and quarterly rows the
    rolling 4-quarter sum) it is used as-is; only where that is missing (an old
    single-frequency cache, or a synthetic test frame) do we fall back to the
    quarterly 4-period rolling sum. Either way it only ever uses periods at or
    before each row's own ``period_end``.
    """
    if fundamentals is None or fundamentals.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for ticker, group in fundamentals.groupby("ticker"):
        group = group.sort_values("period_end").copy()
        rolling_ttm = group["net_income"].rolling(window=4, min_periods=4).sum()
        if "ttm_net_income" in group.columns:
            group["ttm_net_income"] = group["ttm_net_income"].where(
                group["ttm_net_income"].notna(), rolling_ttm
            )
        else:
            group["ttm_net_income"] = rolling_ttm
        group["available_date"] = to_naive_index(group["available_date"])
        result[str(ticker)] = group.sort_values("available_date").reset_index(drop=True)
    return result


def _fundamentals_asof(group: Optional[pd.DataFrame], when: pd.Timestamp) -> Optional[pd.Series]:
    """Most recent filing whose ``available_date`` is on or before ``when``."""
    if group is None or group.empty:
        return None
    eligible = group[group["available_date"] <= when]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _fundamental_features(fund_row: Optional[pd.Series], price: float) -> Dict[str, float]:
    """Valuation + quality features for one ``(t, ticker)`` cell.

    ``pe`` and ``pb`` need a price, so (per the Phase 0 contract) they are formed
    here rather than in the fundamentals loader. The pure-fundamental ratios
    (``roe``, ``net_margin``, ``debt_to_equity``, earnings growth) are passed
    straight through from Phase 0.
    """
    if fund_row is None:
        return {name: np.nan for name in FUNDAMENTAL_FEATURES}

    shares = _to_float(fund_row.get("shares_outstanding"))
    market_cap = price * shares if _positive(shares) else np.nan

    ttm_net_income = _to_float(fund_row.get("ttm_net_income"))
    total_equity = _to_float(fund_row.get("total_equity"))

    return {
        "pe": market_cap / ttm_net_income if _positive(ttm_net_income) else np.nan,
        "pb": market_cap / total_equity if _positive(total_equity) else np.nan,
        "roe": _to_float(fund_row.get("roe")),
        "earnings_growth": _to_float(fund_row.get("earnings_growth_yoy")),
        "net_margin": _to_float(fund_row.get("net_margin")),
        "debt_to_equity": _to_float(fund_row.get("debt_to_equity")),
    }


# --------------------------------------------------------------------------- #
# Sentiment (Extension A) + assembly                                          #
# --------------------------------------------------------------------------- #

def _merge_sentiment(matrix: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Left-join an optional sentiment column onto the feature matrix."""
    series = _sentiment_series(sentiment)
    matrix[SENTIMENT_FEATURE] = series.reindex(matrix.index)
    return matrix


def _sentiment_series(sentiment: pd.DataFrame) -> pd.Series:
    """Normalise the accepted sentiment shapes into a MultiIndex Series."""
    if isinstance(sentiment.index, pd.MultiIndex) and SENTIMENT_FEATURE in sentiment.columns:
        series = sentiment[SENTIMENT_FEATURE].copy()
        series.index = series.index.set_names(["rebalance_date", "ticker"])
        return series

    tidy = sentiment.copy()
    tidy["rebalance_date"] = pd.to_datetime(tidy["rebalance_date"])
    return tidy.set_index(["rebalance_date", "ticker"])[SENTIMENT_FEATURE]


def _rows_to_matrix(rows: Dict[tuple, Dict[str, float]]) -> pd.DataFrame:
    """Assemble the ordered, MultiIndexed feature matrix from per-cell dicts."""
    columns = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES
    if not rows:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["rebalance_date", "ticker"],
        )
        return pd.DataFrame(columns=columns, index=empty_index, dtype="float64")

    matrix = pd.DataFrame.from_dict(rows, orient="index")[columns]
    matrix.index = pd.MultiIndex.from_tuples(matrix.index, names=["rebalance_date", "ticker"])
    return matrix.sort_index()


# --------------------------------------------------------------------------- #
# Small numeric / datetime utilities                                          #
# --------------------------------------------------------------------------- #

def _row_asof(frame: pd.DataFrame, when: pd.Timestamp) -> Optional[pd.Series]:
    """Row at the last index on or before ``when``, preserving per-column NaNs.

    Unlike ``DataFrame.asof`` (which skips any row containing a NaN, and so would
    discard every date before the longest look-back window has filled), this
    returns the genuine as-of row: a still-warming feature such as 6-month
    momentum is simply left as NaN rather than dropping the whole observation.
    """
    pos = frame.index.searchsorted(when, side="right") - 1
    if pos < 0:
        return None
    return frame.iloc[pos]


def _has_close(frame: Optional[pd.DataFrame]) -> bool:
    return frame is not None and not frame.empty and "Close" in frame.columns


def _to_float(value) -> float:
    """Coerce to float, mapping missing values to NaN."""
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def _positive(value: float) -> bool:
    """True when ``value`` is a finite, strictly positive number."""
    return value is not None and not pd.isna(value) and value > 0


# --------------------------------------------------------------------------- #
# Cross-sectional normalisation (accuracy lever)                              #
# --------------------------------------------------------------------------- #

def cross_sectional_normalize(
    matrix: pd.DataFrame,
    method: str = "rank",
    columns: Optional[List[str]] = None,
    sectors: Optional["pd.Series"] = None,
) -> pd.DataFrame:
    """Standardise each feature *within each rebalance date's cross-section*.

    Cross-sectional selection is a *relative* problem: what matters at date ``t``
    is whether a stock's momentum / value / quality is high or low **versus its
    peers that same month**, not its absolute level (which drifts with the market
    and the macro regime). Normalising each feature within each date strips out
    that time-varying level and hands the ranker a clean, comparable
    cross-section — empirically the single biggest lift to ranking quality
    (mean AUC / precision@N) in this project.

    Sector neutralisation
    ---------------------
    When ``sectors`` (a ``ticker -> sector`` mapping) is supplied, each feature is
    standardised **within each ``(rebalance_date, sector)`` group** instead of the
    whole cross-section. This removes sector bets: the ranker then compares a stock
    against its own-sector peers, so a high-momentum technology name is not simply
    ranked above a utility because technology trended. It is the standard way to
    stop cross-sectional signals from collapsing into a single sector tilt.

    Leakage safety
    --------------
    The transform is computed **independently within each date (and sector)**,
    using only that date's own cross-section — no past or future date is ever
    consulted. It therefore cannot leak information across time and is
    as-of-safe by construction (asserted directly in the Phase 1 tests).

    Parameters
    ----------
    method:
        ``"rank"`` — map each column to its within-group percentile in
        ``[-0.5, 0.5]`` (robust to outliers, monotonic-invariant; recommended and
        the project default). ``"zscore"`` — subtract the within-group mean and
        divide by the within-group std.
    columns:
        Columns to transform; defaults to every column present (technical +
        fundamental + sentiment). NaNs are preserved so a still-warming or
        not-yet-reported feature stays NaN for the model to handle natively.
    sectors:
        Optional ``ticker -> sector`` Series/mapping. When given, normalisation is
        done within ``(date, sector)``; tickers with no mapping fall into a shared
        ``"_UNKNOWN"`` bucket so they are still comparable to one another.
    """
    if matrix.empty:
        return matrix.copy()
    cols = list(matrix.columns) if columns is None else [c for c in columns if c in matrix.columns]
    out = matrix.copy()

    date_level = out.index.get_level_values("rebalance_date")
    if sectors is not None:
        sector_map = pd.Series(sectors)
        ticker_level = out.index.get_level_values("ticker")
        sector_of_row = pd.Index(ticker_level).map(sector_map).fillna("_UNKNOWN")
        keys = [date_level, np.asarray(sector_of_row)]
    else:
        keys = date_level
    grouped = out.groupby(keys)

    if method == "rank":
        for c in cols:
            # Within-group percentile rank in (0, 1], recentred to [-0.5, 0.5];
            # NaNs stay NaN.
            out[c] = grouped[c].rank(pct=True) - 0.5
    elif method == "zscore":
        for c in cols:
            mean = grouped[c].transform("mean")
            std = grouped[c].transform("std")
            out[c] = (out[c] - mean) / std.replace(0.0, np.nan)
    else:
        raise ValueError(f"method must be 'rank' or 'zscore', got {method!r}")
    return out
