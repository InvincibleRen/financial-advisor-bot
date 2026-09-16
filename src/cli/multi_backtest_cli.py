from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.backtester import run_backtest
from src.common.data.yfinance_provider import YFinanceProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests for multiple tickers")
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Stock tickers, for example AAPL MSFT NVDA TSLA SPY",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="Historical period, for example 2y, 5y, 10y",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000.0,
        help="Initial cash amount",
    )

    args = parser.parse_args()

    provider = YFinanceProvider()
    rows = []

    for ticker in args.tickers:
        ticker = ticker.upper().strip()

        print(f"\nRunning backtest for {ticker}...")

        try:
            historical_data = provider.get_historical_data(
                ticker=ticker,
                period=args.period,
                interval="1d",
            )

            result = run_backtest(
                data=historical_data,
                ticker=ticker,
                initial_cash=args.cash,
            )

            rows.append(
                {
                    "Ticker": result.ticker,
                    "Start Date": result.start_date,
                    "End Date": result.end_date,
                    "Initial Cash": result.initial_cash,
                    "Final Strategy Value": result.final_strategy_value,
                    "Final Buy-Hold Value": result.final_buy_hold_value,
                    "Strategy Return %": result.strategy_return_percent,
                    "Buy-Hold Return %": result.buy_hold_return_percent,
                    "Difference %": result.strategy_return_percent - result.buy_hold_return_percent,
                    "Max Drawdown %": result.max_drawdown_percent,
                    "Number of Trades": result.number_of_trades,
                    "Win Rate %": result.win_rate_percent,
                    "Status": "OK",
                }
            )

        except Exception as error:
            rows.append(
                {
                    "Ticker": ticker,
                    "Start Date": None,
                    "End Date": None,
                    "Initial Cash": args.cash,
                    "Final Strategy Value": None,
                    "Final Buy-Hold Value": None,
                    "Strategy Return %": None,
                    "Buy-Hold Return %": None,
                    "Difference %": None,
                    "Max Drawdown %": None,
                    "Number of Trades": None,
                    "Win Rate %": None,
                    "Status": f"ERROR: {error}",
                }
            )

    results = pd.DataFrame(rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = reports_dir / f"backtest_summary_{args.period}_{timestamp}.csv"
    markdown_path = reports_dir / f"backtest_summary_{args.period}_{timestamp}.md"

    results.to_csv(csv_path, index=False)

    markdown = "# Multi-Stock Backtest Summary\n\n"
    markdown += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += f"Period: {args.period}\n\n"
    markdown += results.to_markdown(index=False)
    markdown += "\n\n## Interpretation\n\n"
    markdown += (
        "This table compares the rule-based advisor strategy against a buy-and-hold baseline "
        "across multiple stocks. The 'Difference %' column shows whether the advisor strategy "
        "outperformed or underperformed buy-and-hold for each ticker. These results are for "
        "historical evaluation only and do not guarantee future performance.\n"
    )

    markdown_path.write_text(markdown, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Multi-Stock Backtest Summary")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)

    print(f"\nCSV summary saved to: {csv_path}")
    print(f"Markdown summary saved to: {markdown_path}\n")


if __name__ == "__main__":
    main()