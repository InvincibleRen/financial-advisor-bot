"""Command-line entry point for the walk-forward backtest.

Examples
--------
    python -m src.walkforward_cli AAPL
    python -m src.walkforward_cli AAPL --period 8y --train 252 --test 63 --cost 0.001
    python -m src.walkforward_cli AAPL MSFT SPY --period 8y   # several tickers

Writes a markdown report to ``reports/`` and prints a summary to the terminal.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from src.common.data.yfinance_provider import YFinanceProvider
from src.common.walkforward import WalkForwardResult, run_walk_forward


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _num(value: float) -> str:
    return f"{value:.2f}"


def build_markdown_report(result: WalkForwardResult, period: str) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fold_rows = []
    for f in result.folds:
        fold_rows.append(
            f"| {f.index} | {f.test_start} → {f.test_end} | "
            f"{_pct(f.strategy['cumulative_return'])} | "
            f"{_pct(f.benchmark['cumulative_return'])} | "
            f"{_num(f.strategy['sharpe_ratio'])} | "
            f"{_pct(f.strategy['max_drawdown'])} | "
            f"{_pct(f.strategy['hit_rate'])} | {f.n_trades} |"
        )
    fold_table = "\n".join(fold_rows) if fold_rows else "| — | — | — | — | — | — | — | — |"

    agg = result.aggregated_strategy
    agg_b = result.aggregated_benchmark
    ov = result.overall_strategy
    ov_b = result.overall_benchmark

    return f"""# Walk-Forward Backtest: {result.ticker}

Generated at: {generated_at}

## Configuration

- Data period requested: {period}
- Train window: {result.train_size} trading days
- Test window: {result.test_size} trading days
- Step: {result.step} trading days
- Transaction cost: {_pct(result.cost_per_turnover)} per side (per unit turnover)
- Number of folds: {len(result.folds)}
- Benchmark: buy-and-hold over each identical test window

## Per-Fold Results (strategy, net of costs)

| Fold | Test window | Strat return | Bench return | Sharpe | Max DD | Hit rate | Trades |
|---:|:--|---:|---:|---:|---:|---:|---:|
{fold_table}

## Aggregated Across Folds (mean of per-fold metrics)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Mean cumulative return (per fold) | {_pct(agg['mean_cumulative_return'])} | {_pct(agg_b['mean_cumulative_return'])} |
| Mean annualised return | {_pct(agg['mean_annualised_return'])} | {_pct(agg_b['mean_annualised_return'])} |
| Mean Sharpe ratio | {_num(agg['mean_sharpe_ratio'])} | {_num(agg_b['mean_sharpe_ratio'])} |
| Mean max drawdown | {_pct(agg['mean_max_drawdown'])} | {_pct(agg_b['mean_max_drawdown'])} |
| Mean hit rate | {_pct(agg['mean_hit_rate'])} | {_pct(agg_b['mean_hit_rate'])} |

## Overall Out-of-Sample (stitched test windows)

| Metric | Strategy | Buy-and-hold |
|---|---:|---:|
| Cumulative return | {_pct(ov['cumulative_return'])} | {_pct(ov_b['cumulative_return'])} |
| Annualised return | {_pct(ov['annualised_return'])} | {_pct(ov_b['annualised_return'])} |
| Annualised volatility | {_pct(ov['annualised_volatility'])} | {_pct(ov_b['annualised_volatility'])} |
| Sharpe ratio | {_num(ov['sharpe_ratio'])} | {_num(ov_b['sharpe_ratio'])} |
| Max drawdown | {_pct(ov['max_drawdown'])} | {_pct(ov_b['max_drawdown'])} |
| Hit rate | {_pct(ov['hit_rate'])} | {_pct(ov_b['hit_rate'])} |
| Days evaluated | {ov['n_days']} | {ov_b['n_days']} |

## Interpretation

This walk-forward evaluation tests the rule-based advisor strategy on a sequence
of unseen windows rather than a single split, reporting standard finance metrics
per fold and aggregated, net of {_pct(result.cost_per_turnover)} per-side costs.
It is for evaluation purposes only and does not guarantee future performance.
"""


def print_summary(result: WalkForwardResult) -> None:
    ov = result.overall_strategy
    ov_b = result.overall_benchmark
    print("=" * 70)
    print(f"Walk-Forward Backtest — {result.ticker}  ({len(result.folds)} folds)")
    print("=" * 70)
    print(f"Overall strategy   return: {_pct(ov['cumulative_return']):>10}  "
          f"Sharpe {_num(ov['sharpe_ratio']):>6}  MaxDD {_pct(ov['max_drawdown']):>8}")
    print(f"Overall buy&hold   return: {_pct(ov_b['cumulative_return']):>10}  "
          f"Sharpe {_num(ov_b['sharpe_ratio']):>6}  MaxDD {_pct(ov_b['max_drawdown']):>8}")
    print("=" * 70)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for the advisor strategy")
    parser.add_argument("tickers", nargs="+", help="One or more tickers, e.g. AAPL MSFT SPY")
    parser.add_argument("--period", default="8y", help="History to download (default 8y)")
    parser.add_argument("--train", type=int, default=252, help="Train window in trading days (default 252)")
    parser.add_argument("--test", type=int, default=63, help="Test window in trading days (default 63)")
    parser.add_argument("--step", type=int, default=None, help="Advance step in days (default = test size)")
    parser.add_argument("--cost", type=float, default=0.001, help="Per-side transaction cost (default 0.001 = 10 bps)")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate for Sharpe (default 0)")
    parser.add_argument("--no-save", action="store_true", help="Do not write a markdown report")
    args = parser.parse_args(argv)

    provider = YFinanceProvider()
    # project root is three levels up: src/cli/walkforward_cli.py -> project/
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    for ticker in args.tickers:
        ticker = ticker.upper().strip()
        print(f"\nDownloading {ticker} ({args.period})...")
        data = provider.get_historical_data(ticker, period=args.period, interval="1d")

        result = run_walk_forward(
            data,
            ticker=ticker,
            train_size=args.train,
            test_size=args.test,
            step=args.step,
            cost_per_turnover=args.cost,
            risk_free_rate=args.rf,
        )
        print_summary(result)

        if not args.no_save:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = reports_dir / f"walkforward_{ticker}_{stamp}.md"
            out.write_text(build_markdown_report(result, args.period))
            print(f"Report written to {out}")


if __name__ == "__main__":
    main()
