"""Data/logic layer for the Streamlit UI (Phase 4).

Deliberately **Streamlit-free**: every function here is pure application logic
that takes an injectable data provider, so it can be unit-tested offline without
a running Streamlit server. The view layer (``views.py``) adds caching and
widgets on top. This keeps the UI thin and the logic testable (Rule 1).

Each function wires together modules that already exist and are tested elsewhere:
the rule-based advisor (core), the direction forecaster (Extension B), the FinBERT
sentiment scorer (Extension A), and the cross-sectional selector (Direction-1 core).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.common.data.yfinance_provider import Quote, YFinanceProvider
from src.common.indicators import add_moving_averages, latest_indicator_snapshot
from src.direction.forecaster import DirectionForecast, DirectionForecaster
from src.explanation.advisor import Recommendation, generate_recommendation

DEFAULT_HISTORY = "2y"

# --------------------------------------------------------------------------- #
# Period presets — friendly label -> (yfinance period, interval, row trim,     #
# and sensible direction-horizon options for that bar frequency).              #
# yfinance has no "3y" period, so 3 years = fetch "5y" daily and keep the last #
# ~756 trading rows. Short ranges use finer intervals so the chart is useful.  #
# --------------------------------------------------------------------------- #

SINGLE_STOCK_PRESETS: Dict[str, dict] = {
    "1 day":    {"period": "1d",  "interval": "5m",  "trim": None, "horizons": [1, 3, 6, 12],  "default_horizon": 6},
    "1 week":   {"period": "5d",  "interval": "30m", "trim": None, "horizons": [1, 3, 6, 13],  "default_horizon": 6},
    "1 month":  {"period": "1mo", "interval": "1d",  "trim": None, "horizons": [1, 2, 3, 5],   "default_horizon": 3},
    "3 months": {"period": "3mo", "interval": "1d",  "trim": None, "horizons": [1, 3, 5, 10],  "default_horizon": 5},
    "6 months": {"period": "6mo", "interval": "1d",  "trim": None, "horizons": [1, 5, 10, 21], "default_horizon": 5},
    "1 year":   {"period": "1y",  "interval": "1d",  "trim": None, "horizons": [1, 5, 10, 21], "default_horizon": 5},
    "3 years":  {"period": "5y",  "interval": "1d",  "trim": 756,  "horizons": [5, 10, 21, 63], "default_horizon": 10},
    "5 years":  {"period": "5y",  "interval": "1d",  "trim": None, "horizons": [5, 10, 21, 63], "default_horizon": 21},
}

# Backtest needs enough daily bars for a walk-forward split, so it starts at 1 month.
BACKTEST_PRESETS: Dict[str, dict] = {
    label: {"period": p["period"], "interval": p["interval"], "trim": p["trim"]}
    for label, p in SINGLE_STOCK_PRESETS.items()
    if label not in ("1 day", "1 week")
}


def default_universe() -> List[str]:
    """The built-in large-cap universe, offered as suggestions in the UI."""
    from src.selection import universe as U

    return list(U.DEFAULT_UNIVERSE)


def _fetch_prices(provider, ticker: str, period: str, interval: str, trim: Optional[int]):
    """Fetch OHLC and optionally keep only the last ``trim`` rows (for '3 years')."""
    data = provider.get_historical_data(ticker, period=period, interval=interval)
    if trim:
        data = data.tail(int(trim))
    return data


# --------------------------------------------------------------------------- #
# 1. Single-stock analysis (core advisor + indicators + Extension-B direction) #
# --------------------------------------------------------------------------- #

@dataclass
class SingleStockAnalysis:
    ticker: str
    quote: Quote
    indicators: Dict
    recommendation: Recommendation
    price_chart: pd.DataFrame                       # Close + SMA20/50/200
    direction: Optional[DirectionForecast] = None
    direction_error: Optional[str] = None


def analyse_single_stock(
    ticker: str,
    period: str = DEFAULT_HISTORY,
    interval: str = "1d",
    horizon_days: int = 5,
    trim_rows: Optional[int] = None,
    provider: Optional[object] = None,
) -> SingleStockAnalysis:
    """Full single-stock view: quote, indicators, recommendation, direction call.

    ``period``/``interval``/``trim_rows`` come from a ``SINGLE_STOCK_PRESETS`` entry
    (the view resolves the friendly label). Moving-average columns that never fill
    on a short/intraday range are dropped from the chart rather than shown empty.
    """
    provider = provider or YFinanceProvider()
    ticker = ticker.upper().strip()

    historical = _fetch_prices(provider, ticker, period, interval, trim_rows)
    try:
        intraday = provider.get_intraday_data(ticker, period="5d", interval="5m")
    except Exception:  # noqa: BLE001 - intraday is optional context
        intraday = None

    quote = provider.get_latest_quote(ticker)
    indicators = latest_indicator_snapshot(historical_data=historical, intraday_data=intraday)
    recommendation = generate_recommendation(quote=quote, indicators=indicators)

    sma = add_moving_averages(historical[["Close"]])
    price_chart = sma[["Close", "SMA20", "SMA50", "SMA200"]].dropna(axis=1, how="all").copy()

    direction: Optional[DirectionForecast] = None
    direction_error: Optional[str] = None
    try:
        # calibrate=True so the displayed P(up) is a trustworthy frequency, not a
        # raw uncalibrated score (leakage-safe CV calibration inside fit).
        forecaster = DirectionForecaster(
            kind="gbm", horizon_days=horizon_days, calibrate=True,
        ).fit(historical, ticker)
        direction = forecaster.predict_direction(historical, ticker)
    except Exception as exc:  # noqa: BLE001 - too little history is a soft failure
        direction_error = str(exc)

    return SingleStockAnalysis(
        ticker=ticker, quote=quote, indicators=indicators, recommendation=recommendation,
        price_chart=price_chart, direction=direction, direction_error=direction_error,
    )


# --------------------------------------------------------------------------- #
# 2. Backtest & evaluation (rule-based walk-forward + Extension-B direction)   #
# --------------------------------------------------------------------------- #

@dataclass
class TickerEvaluation:
    ticker: str
    walkforward: object                             # WalkForwardResult
    direction: object                               # DirectionEvaluation


def evaluate_ticker(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    horizon_days: int = 5,
    train: int = 252,
    test: int = 63,
    cost: float = 0.001,
    trim_rows: Optional[int] = None,
    provider: Optional[object] = None,
) -> TickerEvaluation:
    """Walk-forward rule-based backtest + walk-forward direction evaluation.

    Raises a clear ``ValueError`` when the chosen period is too short to form a
    single train+test walk-forward window, so the UI can show a friendly hint.
    """
    from src.common.walkforward import run_walk_forward
    from src.direction.evaluate import evaluate_direction

    provider = provider or YFinanceProvider()
    ticker = ticker.upper().strip()
    prices = _fetch_prices(provider, ticker, period, interval, trim_rows)

    if len(prices) < train + test + 5:
        raise ValueError(
            f"Only {len(prices)} bars for this period — a walk-forward backtest needs "
            f"more than train+test ({train}+{test}). Pick a longer period or a smaller "
            "train window."
        )

    walkforward = run_walk_forward(
        prices, ticker=ticker, train_size=train, test_size=test, cost_per_turnover=cost,
    )
    direction = evaluate_direction(
        prices, ticker=ticker, kind="gbm", horizon_days=horizon_days, train_size=train,
    )
    return TickerEvaluation(ticker=ticker, walkforward=walkforward, direction=direction)


# --------------------------------------------------------------------------- #
# 3. News sentiment (Extension A, torch-free onnx-local backend)               #
# --------------------------------------------------------------------------- #

@dataclass
class SentimentResult:
    ticker: str
    aggregate: float
    headlines: List[Tuple[pd.Timestamp, str, float, str]] = field(default_factory=list)  # (date, text, score, url)
    n_total: int = 0                       # total headlines fetched in the window
    error: Optional[str] = None


def sentiment_interpretation(score: float) -> Tuple[str, str]:
    """Turn a signed sentiment score in [-1, 1] into (label, plain-English meaning)."""
    if score is None or score != score:    # NaN
        return "Unknown", "No sentiment score could be computed."
    if score >= 0.5:
        return "Strongly positive", "Recent coverage is clearly upbeat about this stock."
    if score >= 0.15:
        return "Positive", "Recent coverage leans favourable on balance."
    if score > -0.15:
        return "Neutral", "Coverage is mixed or neutral — no clear positive or negative tilt."
    if score > -0.5:
        return "Negative", "Recent coverage leans unfavourable on balance."
    return "Strongly negative", "Recent coverage is clearly downbeat about this stock."


def score_ticker_sentiment(
    ticker: str,
    lookback_days: int = 30,
    backend: str = "onnx-local",
    max_headlines: int = 10,
    asof: Optional[pd.Timestamp] = None,
    source: Optional[object] = None,
) -> SentimentResult:
    """Fetch recent headlines and score with FinBERT.

    The aggregate is the mean signed score over **all** headlines in the window;
    the returned ``headlines`` list is the most recent ``max_headlines`` (each with
    its own score and article URL) for display.
    """
    ticker = ticker.upper().strip()
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()

    try:
        if source is None:
            from src.sentiment.news_source import YFinanceNewsSource

            source = YFinanceNewsSource()
        headlines = source.fetch_headlines(ticker, asof, lookback_days)
    except Exception as exc:  # noqa: BLE001 - a news failure must not crash the UI
        return SentimentResult(ticker, float("nan"), [], 0, f"News fetch failed: {exc}")

    if not headlines:
        return SentimentResult(ticker, 0.0, [], 0, None)

    try:
        from src.sentiment.finbert import FinBERTScorer

        scorer = FinBERTScorer(backend=backend)
        aggregate = scorer.score_texts([h.text for h in headlines])   # over all fetched
        recent = sorted(headlines, key=lambda h: h.published_at, reverse=True)[:max_headlines]
        rows = [(h.published_at, h.text, scorer.score_texts([h.text]), h.url) for h in recent]
    except Exception as exc:  # noqa: BLE001 - missing deps / model download issues
        return SentimentResult(ticker, float("nan"), [], len(headlines), f"Scoring failed: {exc}")

    return SentimentResult(ticker, float(aggregate), rows, len(headlines), None)


# --------------------------------------------------------------------------- #
# 4. Top-N recommendations (Direction-1 core selector — heavy)                 #
# --------------------------------------------------------------------------- #

@dataclass
class SelectionResult:
    result: object                                  # SelectionBacktest
    latest_date: Optional[pd.Timestamp]
    latest_picks: List[str]
    universe: List[str]
    n_rebalances: int


def run_universe_selection(
    tickers: Optional[List[str]] = None,
    start: str = "2018-01-01",
    top_n: int = 3,
    model_kind: str = "gbm",
    min_train: int = 6,
    cost: float = 0.001,
    normalize: str = "rank",
) -> SelectionResult:
    """Run the walk-forward selector over a universe and return the latest top-N.

    ``normalize`` applies leakage-safe cross-sectional feature standardisation per
    rebalance date ("rank" by default — the setting that improves ranking AUC /
    precision@N; pass "none" for the raw-level behaviour).
    """
    from src.cli.select_cli import BENCHMARK_TICKER, month_end_rebalances
    from src.selection import universe as U
    from src.selection.features import build_feature_matrix, cross_sectional_normalize
    from src.selection.labels import make_labels
    from src.selection.select import run_selection_backtest

    universe = [t.upper().strip() for t in tickers] if tickers else list(U.DEFAULT_UNIVERSE)
    prices = U.load_prices(universe, start=start, end=None)
    fundamentals = U.load_fundamentals(universe)
    spy = U.load_prices([BENCHMARK_TICKER], start=start, end=None)

    rebalance_dates = month_end_rebalances(prices)
    if len(rebalance_dates) <= min_train + 1:
        raise ValueError("Not enough history for a walk-forward run; use an earlier start date.")

    feature_matrix = build_feature_matrix(prices, fundamentals, rebalance_dates)
    if normalize != "none":
        feature_matrix = cross_sectional_normalize(feature_matrix, method=normalize)
    labels = make_labels(prices, rebalance_dates, horizon_months=1)
    prices_eval = dict(prices)
    prices_eval.update(spy)

    result = run_selection_backtest(
        feature_matrix, labels, prices_eval, rebalance_dates,
        n=top_n, model_kind=model_kind, min_train_dates=min_train, cost_per_turnover=cost,
    )

    latest_date = result.weights.index.max() if not result.weights.empty else None
    latest_picks: List[str] = []
    if latest_date is not None:
        row = result.weights.loc[latest_date]
        latest_picks = list(row[row > 0].index)

    return SelectionResult(
        result=result, latest_date=latest_date, latest_picks=latest_picks,
        universe=universe, n_rebalances=len(rebalance_dates),
    )
