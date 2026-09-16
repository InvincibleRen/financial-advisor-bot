from __future__ import annotations

import argparse
from typing import Optional

from src.explanation.advisor import generate_recommendation
from src.common.data.yfinance_provider import YFinanceProvider
from src.common.indicators import latest_indicator_snapshot


def format_money(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Advisor Bot prototype")
    parser.add_argument("ticker", help="Stock ticker, for example AAPL, MSFT, NVDA")
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Report-faithful prototype mode: SMA20/50/200 + crossover only "
        "(omits the MACD momentum and ADX trend-strength enhancements).",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    provider = YFinanceProvider()

    print(f"\nFetching latest available data for {ticker}...\n")

    quote = provider.get_latest_quote(ticker)
    intraday_data = provider.get_intraday_data(ticker, period="5d", interval="5m")
    historical_data = provider.get_historical_data(ticker, period="2y", interval="1d")

    indicators = latest_indicator_snapshot(
        historical_data=historical_data,
        intraday_data=intraday_data,
        basic=args.basic,
    )

    recommendation = generate_recommendation(
        quote=quote,
        indicators=indicators,
    )

    print("=" * 70)
    print(f"Financial Advisor Bot - Prototype Result for {ticker}")
    print("=" * 70)

    print(f"Latest price:          {format_money(quote.latest_price)}")
    print(f"Previous close:        {format_money(quote.previous_close)}")
    print(f"Change:                {format_money(quote.change)}")
    print(f"Change percent:        {format_percent(quote.change_percent)}")
    print(f"Data timestamp:        {quote.timestamp}")
    print(f"Data source:           {quote.source}")

    print("-" * 70)
    print("Technical Indicators")
    print("-" * 70)

    print(f"SMA20:                 {format_money(indicators.get('sma20'))}")
    print(f"SMA50:                 {format_money(indicators.get('sma50'))}")
    print(f"SMA200:                {format_money(indicators.get('sma200'))}")
    print(f"Cross event:           {indicators.get('cross_event')}")
    if not args.basic:
        print(f"SMA50/200 gap:         {format_percent(indicators.get('sma_gap_percent'))}")
        print(f"MACD line:             {format_number(indicators.get('macd_line'))}")
        print(f"MACD signal:           {format_number(indicators.get('macd_signal'))}")
        print(f"MACD histogram:        {format_number(indicators.get('macd_hist'))}")
        print(f"ADX (14):              {format_number(indicators.get('adx'))}")
    print(f"Intraday return:       {format_percent(indicators.get('intraday_return_percent'))}")
    print(f"Intraday volatility:   {format_percent(indicators.get('intraday_volatility_percent'))}")

    print("-" * 70)
    print("Recommendation")
    print("-" * 70)

    print(f"Signal:                {recommendation.signal}")
    print(f"Risk level:            {recommendation.risk_level}")
    print(f"Rule score:            {recommendation.score}")

    print("\nReasons:")
    for reason in recommendation.reasons:
        print(f"- {reason}")

    if recommendation.warnings:
        print("\nWarnings:")
        for warning in recommendation.warnings:
            print(f"- {warning}")

    print("\nDisclaimer:")
    print(recommendation.disclaimer)

    print("\nData note:")
    print(quote.note)
    print("=" * 70)


if __name__ == "__main__":
    main()