"""Offline tests for the UI data/logic layer (Phase 4).

The Streamlit view layer is intentionally thin; all logic lives in
``src.ui.data_access``, which is Streamlit-free and provider-injectable, so it is
tested here with a fake provider — no network, no Streamlit, no model download.
"""
import numpy as np
import pandas as pd

from src.common.data.yfinance_provider import Quote
from src.ui import data_access as da


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #

def _prices(n=600, seed=0):
    t = np.arange(n)
    close = 100 + 15 * np.sin(2 * np.pi * t / 60) + 0.03 * t \
        + np.random.default_rng(seed).normal(0, 0.5, n)
    return pd.DataFrame({"Close": close}, index=pd.bdate_range("2018-01-01", periods=n))


class _FakeProvider:
    """Minimal stand-in for YFinanceProvider (no network)."""

    def __init__(self, prices, intraday_raises=True):
        self._prices = prices
        self._intraday_raises = intraday_raises

    def get_historical_data(self, ticker, period="2y", interval="1d"):
        return self._prices

    def get_intraday_data(self, ticker, period="5d", interval="5m"):
        if self._intraday_raises:
            raise RuntimeError("intraday unavailable")
        return self._prices.tail(50)

    def get_latest_quote(self, ticker):
        last = float(self._prices["Close"].iloc[-1])
        prev = float(self._prices["Close"].iloc[-2])
        return Quote(ticker=ticker, latest_price=last, previous_close=prev,
                     change=last - prev, change_percent=(last / prev - 1) * 100,
                     timestamp="2020-01-01", source="fake", note="test")


# --------------------------------------------------------------------------- #
# Single-stock analysis                                                        #
# --------------------------------------------------------------------------- #

def test_analyse_single_stock_assembles_all_parts():
    a = da.analyse_single_stock("aapl", period="2y", horizon_days=5,
                                provider=_FakeProvider(_prices()))
    assert a.ticker == "AAPL"                                  # normalised
    assert a.recommendation.signal                             # advisor ran
    assert list(a.price_chart.columns) == ["Close", "SMA20", "SMA50", "SMA200"]
    assert a.direction is not None                             # Ext-B forecast present
    assert 0.0 <= a.direction.prob_up <= 1.0
    assert a.direction.direction in ("up", "down")


def test_single_stock_survives_missing_intraday():
    a = da.analyse_single_stock("MSFT", provider=_FakeProvider(_prices(), intraday_raises=True))
    # Intraday-derived fields are simply absent; the view still assembles.
    assert a.indicators.get("intraday_return_percent") is None
    assert a.recommendation is not None


def test_direction_soft_fails_on_short_history():
    a = da.analyse_single_stock("TSLA", provider=_FakeProvider(_prices(n=30)))
    assert a.direction is None
    assert a.direction_error                                   # populated, not raised


# --------------------------------------------------------------------------- #
# Layering discipline                                                          #
# --------------------------------------------------------------------------- #

def test_data_access_is_streamlit_free():
    """The logic layer must not import streamlit (keeps it testable/headless)."""
    import pathlib

    source = pathlib.Path(da.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source


# --------------------------------------------------------------------------- #
# Period presets + universe suggestions                                        #
# --------------------------------------------------------------------------- #

def test_single_stock_presets_are_well_formed():
    presets = da.SINGLE_STOCK_PRESETS
    assert list(presets)[0] == "1 day" and len(presets) == 8
    for p in presets.values():
        assert {"period", "interval", "trim", "horizons", "default_horizon"} <= set(p)
        assert p["default_horizon"] in p["horizons"]
    # yfinance has no "3y" period -> fetched as 5y daily, trimmed to ~3 years.
    assert presets["3 years"]["period"] == "5y" and presets["3 years"]["trim"] == 756


def test_backtest_presets_exclude_intraday_ranges():
    assert "1 day" not in da.BACKTEST_PRESETS
    assert "1 week" not in da.BACKTEST_PRESETS
    assert "1 month" in da.BACKTEST_PRESETS and "5 years" in da.BACKTEST_PRESETS


def test_default_universe_lists_known_names():
    u = da.default_universe()
    assert "AAPL" in u and "MSFT" in u and len(u) > 10


def test_evaluate_ticker_rejects_too_short_period():
    # ~40 rows can't support a 252+63 walk-forward window -> clear ValueError.
    try:
        da.evaluate_ticker("AAPL", train=252, test=63, provider=_FakeProvider(_prices(n=40)))
        assert False, "expected ValueError for too-short history"
    except ValueError as exc:
        assert "walk-forward" in str(exc)


# --------------------------------------------------------------------------- #
# Sentiment: interpretation + headline plumbing (URLs, count, ordering)       #
# --------------------------------------------------------------------------- #

def test_sentiment_interpretation_bands():
    assert da.sentiment_interpretation(0.8)[0] == "Strongly positive"
    assert da.sentiment_interpretation(0.2)[0] == "Positive"
    assert da.sentiment_interpretation(0.0)[0] == "Neutral"
    assert da.sentiment_interpretation(-0.3)[0] == "Negative"
    assert da.sentiment_interpretation(-0.8)[0] == "Strongly negative"
    assert da.sentiment_interpretation(float("nan"))[0] == "Unknown"


class _FakeSource:
    def __init__(self, heads):
        self._heads = heads

    def fetch_headlines(self, ticker, asof, lookback_days):
        return self._heads


class _FakeScorer:
    def __init__(self, *a, **k):
        pass

    def score_texts(self, texts):
        joined = " ".join(texts).lower()
        return 0.5 if "good" in joined else -0.5 if "bad" in joined else 0.0


def test_score_ticker_sentiment_urls_count_and_limit(monkeypatch):
    import src.sentiment.finbert as fb
    from src.sentiment.news_source import Headline

    monkeypatch.setattr(fb, "FinBERTScorer", _FakeScorer)
    heads = [Headline("AAPL", pd.Timestamp(f"2020-01-0{i + 1}"), f"good news {i}", "yf",
                      f"https://news/{i}") for i in range(5)]
    res = da.score_ticker_sentiment("AAPL", lookback_days=365, max_headlines=3,
                                    asof=pd.Timestamp("2020-02-01"), source=_FakeSource(heads))
    assert res.n_total == 5                       # aggregate over all fetched
    assert len(res.headlines) == 3                # display limited
    dates = [r[0] for r in res.headlines]
    assert dates == sorted(dates, reverse=True)   # most-recent first
    assert all(r[3].startswith("https://") for r in res.headlines)   # URLs plumbed through
    assert res.aggregate == 0.5
