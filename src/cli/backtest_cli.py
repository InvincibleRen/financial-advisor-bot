from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.common.backtester import run_backtest
from src.common.data.yfinance_provider import YFinanceProvider


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def build_markdown_report(result) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trade_returns = ", ".join(f"{value:.2f}%" for value in result.trade_returns_percent)

    if not trade_returns:
        trade_returns = "No completed trades."

    return f"""# Backtest Results: {result.ticker}

Generated at: {generated_at}

## Backtest Period

- Start date: {result.start_date}
- End date: {result.end_date}
- Initial cash: {format_money(result.initial_cash)}

## Performance Summary

| Metric | Value |
|---|---:|
| Final strategy value | {format_money(result.final_strategy_value)} |
| Final buy-and-hold value | {format_money(result.final_buy_hold_value)} |
| Strategy return | {format_percent(result.strategy_return_percent)} |
| Buy-and-hold return | {format_percent(result.buy_hold_return_percent)} |
| Maximum drawdown | {format_percent(result.max_drawdown_percent)} |
| Number of trades | {result.number_of_trades} |
| Win rate | {format_percent(result.win_rate_percent)} |

## Trade Returns

{trade_returns}

## Interpretation

This backtest compares the rule-based advisor strategy against a buy-and-hold baseline.
The strategy combines SMA20, SMA50, SMA200 and moving-average crossover logic with
MACD(12, 26, 9) momentum confirmation and ADX(14) trend-strength amplification.

This result is for evaluation purposes only. It does not prove future performance.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Financial Advisor Bot strategy")
    parser.add_argument("ticker", help="Stock ticker, for example AAPL, MSFT, NVDA")
    parser.add_argument("--period", default="5y", help="Historical period, for example 2y, 5y, 10y")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Initial cash amount")

    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    provider = YFinanceProvider()

    print(f"\nFetching historical data for {ticker}...")
    historical_data = provider.get_historical_data(ticker=ticker, period=args.period, interval="1d")

    print("Running backtest...\n")
    result = run_backtest(
        data=historical_data,
        ticker=ticker,
        initial_cash=args.cash,
    )

    print("=" * 70)
    print(f"Backtest Result for {result.ticker}")
    print("=" * 70)
    print(f"Period:                    {result.start_date} to {result.end_date}")
    print(f"Initial cash:              {format_money(result.initial_cash)}")
    print(f"Final strategy value:      {format_money(result.final_strategy_value)}")
    print(f"Final buy-hold value:      {format_money(result.final_buy_hold_value)}")
    print(f"Strategy return:           {format_percent(result.strategy_return_percent)}")
    print(f"Buy-and-hold return:       {format_percent(result.buy_hold_return_percent)}")
    print(f"Maximum drawdown:          {format_percent(result.max_drawdown_percent)}")
    print(f"Number of trades:          {result.number_of_trades}")
    print(f"Win rate:                  {format_percent(result.win_rate_percent)}")
    print("=" * 70)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = reports_dir / f"backtest_{ticker}_{args.period}_{timestamp}.md"
    report_path.write_text(build_markdown_report(result), encoding="utf-8")

    latest_report_path = reports_dir / "backtest_latest.md"
    latest_report_path.write_text(build_markdown_report(result), encoding="utf-8")

    print(f"\nBacktest report saved to: {report_path}")
    print(f"Latest backtest copy saved to: {latest_report_path}\n")


if __name__ == "__main__":
    main()