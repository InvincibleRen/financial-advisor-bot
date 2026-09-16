from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

@dataclass
class Quote:
    ticker: str
    latest_price: float
    previous_close: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    timestamp: str
    source: str = "yfinance"
    note: str = "Latest available market data. This may be delayed depending on the data provider."

class YFinanceProvider:
    # Prototype data provider using yfinance.
    # Later, can add Finnhub, Alpha Vantage, or another provider without
    # changing the main advisor logic.

    def get_intraday_data(
        self,
        ticker: str,
        period: str = "5d",
        interval: str = "5m",
    ) -> pd.DataFrame:
        ticker = ticker.upper().strip()

        data = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )

        if data.empty:
            raise ValueError(f"No intraday data found for ticker: {ticker}")

        data = data.dropna(subset=["Close"])
        return data

    def get_historical_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        ticker = ticker.upper().strip()

        data = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )

        if data.empty:
            raise ValueError(f"No historical data found for ticker: {ticker}")

        data = data.dropna(subset=["Close"])
        return data

    def get_latest_quote(self, ticker: str) -> Quote:
        ticker = ticker.upper().strip()

        intraday = self.get_intraday_data(ticker=ticker, period="5d", interval="5m")
        latest_row = intraday.iloc[-1]
        latest_price = float(latest_row["Close"])

        timestamp = intraday.index[-1]
        timestamp_text = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)

        daily = self.get_historical_data(ticker=ticker, period="10d", interval="1d")

        previous_close = None
        if len(daily) >= 2:
            previous_close = float(daily["Close"].iloc[-2])
        elif len(daily) == 1:
            previous_close = float(daily["Close"].iloc[-1])

        change = None
        change_percent = None

        if previous_close and previous_close != 0:
            change = latest_price - previous_close
            change_percent = (change / previous_close) * 100

        return Quote(
            ticker=ticker,
            latest_price=latest_price,
            previous_close=previous_close,
            change=change,
            change_percent=change_percent,
            timestamp=timestamp_text,
        )