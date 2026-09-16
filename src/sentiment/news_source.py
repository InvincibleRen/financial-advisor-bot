"""Source-agnostic news feed for sentiment (Extension A, Phase 3).

A thin interface so the sentiment collector is not tied to one provider. Free
financial-news headlines are the default source; the X/Twitter API (now paid)
can substitute or supplement behind the same interface.

Leakage safety
--------------
Every source must return **only headlines published on or before ``asof``** (and
no older than ``lookback_days``). Sentiment at a rebalance date must be knowable
at that date, exactly like the price and fundamental features.

Design
------
Two concrete sources ship here, mirroring the injectable-fetcher pattern used in
``selection/universe.py``:

* ``InMemoryNewsSource`` — wraps a fixed list of headlines. Used by the offline
  tests (and any reproducible experiment) so scoring/aggregation is verifiable
  without a network.
* ``YFinanceNewsSource`` — the default live source. **Caveat:** yfinance only
  exposes *recent* headlines, so it supports "as of today" scoring but not a
  historical backtest — the same limitation that affects fundamentals. A richer
  historical news archive can be dropped in behind the same interface later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class Headline:
    ticker: str
    published_at: pd.Timestamp
    text: str
    source: str
    url: str = ""  # link to the article (when the provider supplies one)


class NewsSource(ABC):
    """Abstract headline provider. Implementations must return only headlines
    published on or before ``asof`` to preserve leakage safety."""

    @abstractmethod
    def fetch_headlines(self, ticker: str, asof: pd.Timestamp, lookback_days: int = 7) -> List[Headline]:
        raise NotImplementedError("Phase 3: implement a concrete news source.")


def _within_window(published_at: pd.Timestamp, asof: pd.Timestamp, lookback_days: int) -> bool:
    """True when ``published_at`` lies in ``(asof - lookback_days, asof]``."""
    window_start = asof - pd.Timedelta(days=lookback_days)
    return window_start < published_at <= asof


class InMemoryNewsSource(NewsSource):
    """Serve headlines from a fixed in-memory list (offline / reproducible).

    Ideal for tests and deterministic experiments: construct it with a list of
    ``Headline`` objects and it applies exactly the same leakage-safe window as a
    live source.
    """

    def __init__(self, headlines: List[Headline]) -> None:
        self._headlines = list(headlines)

    def fetch_headlines(self, ticker: str, asof: pd.Timestamp, lookback_days: int = 7) -> List[Headline]:
        asof = pd.Timestamp(asof)
        return [
            h for h in self._headlines
            if h.ticker == ticker and _within_window(pd.Timestamp(h.published_at), asof, lookback_days)
        ]


class YFinanceNewsSource(NewsSource):
    """Default live source using ``yfinance``'s per-ticker news feed.

    Only recent headlines are available from yfinance, so this is suitable for a
    live "sentiment right now" reading, not a multi-year historical backtest. The
    leakage window is still enforced, so results are honest for the dates they can
    actually cover.
    """

    def fetch_headlines(self, ticker: str, asof: pd.Timestamp, lookback_days: int = 7) -> List[Headline]:
        import yfinance as yf  # local import so the module loads without the dep

        asof = pd.Timestamp(asof)
        try:
            raw_items = yf.Ticker(ticker).news or []
        except Exception:  # noqa: BLE001 - a news failure must not abort a run
            return []

        headlines: List[Headline] = []
        for item in raw_items:
            text, published_at, url = _parse_yf_news_item(item)
            if text is None or published_at is None:
                continue
            if _within_window(published_at, asof, lookback_days):
                headlines.append(Headline(ticker, published_at, text, "yfinance", url or ""))
        return headlines


def _parse_yf_news_item(item: dict) -> tuple[Optional[str], Optional[pd.Timestamp], Optional[str]]:
    """Extract (headline text, publish time, article URL) across yfinance schemas.

    Older yfinance returns flat dicts (``title`` + epoch ``providerPublishTime`` +
    ``link``); newer versions nest under ``content`` with an ISO ``pubDate`` and the
    URL under ``canonicalUrl``/``clickThroughUrl``. Handle both.
    """
    content = item.get("content", item)
    text = content.get("title")
    url = _extract_news_url(item, content)

    epoch = item.get("providerPublishTime")
    if epoch is not None:
        return text, pd.to_datetime(epoch, unit="s"), url

    iso = content.get("pubDate") or content.get("displayTime")
    if iso is not None:
        return text, pd.to_datetime(iso).tz_localize(None), url

    return text, None, url


def _extract_news_url(item: dict, content: dict) -> Optional[str]:
    """Pull the article link out of either yfinance news schema."""
    if item.get("link"):                      # old flat schema
        return item["link"]
    for key in ("canonicalUrl", "clickThroughUrl"):   # new nested schema
        value = content.get(key)
        if isinstance(value, dict) and value.get("url"):
            return value["url"]
        if isinstance(value, str) and value:
            return value
    return None
