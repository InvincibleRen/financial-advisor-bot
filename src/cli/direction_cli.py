"""Command-line demo for the per-stock direction forecast (Extension B).

Runs the direction forecaster on the stocks you pass in — intended to be the
**top-N the core selector chose** (`select_cli`) — as a short-term up/down
*confirmation* signal. For each ticker it reports an honest walk-forward
evaluation (out-of-sample directional accuracy vs the naive base rate) and the
**current** call (P(up) over the next `--horizon` days).

Examples
--------
    # Confirm the core's picks (feed the selector's top-N)
    python -m src.cli.direction_cli AAPL MSFT NVDA
    python -m src.cli.direction_cli AAPL --model logistic --horizon 10 --period 8y

Writes a markdown report to ``reports/`` and prints a summary. Single-stock
direction is close to a coin flip, so a small or negative edge is an honest and
acceptable result — that is the point of evaluating it.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.common.data.yfinance_provider import YFinanceProvider
from src.direction.evaluate import DirectionEvaluation, evaluate_direction
from src.direction.forecaster import DirectionForecaster


def _pct(x: float) -> str:
    return "—" if x != x else f"{x * 100:.1f}%"          # x!=x catches NaN


def _auc(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def _run_ticker(provider, ticker, args):
    prices = provider.get_historical_data(ticker, period=args.period, interval="1d")
    evaluation = evaluate_direction(
        prices, ticker=ticker, kind=args.model, horizon_days=args.horizon,
        train_size=args.train, step=args.step, calibrate=args.calibrate,
    )
    forecaster = DirectionForecaster(
        kind=args.model, horizon_days=args.horizon, calibrate=args.calibrate,
    ).fit(prices, ticker)
    forecast = forecaster.predict_direction(prices, ticker)
    return evaluation, forecast


def build_markdown_report(results, args) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for ev, fc in results:
        rows.append(
            f"| {ev.ticker} | {ev.n_predictions} | {_pct(ev.accuracy)} | {_pct(ev.base_rate)} | "
            f"{ev.edge * 100:+.1f}pp | {_auc(ev.auc)} | {ev.brier:.3f} | {ev.ece:.3f} | "
            f"{fc.prob_up:.2f} ({fc.direction}) |"
        )
    table = "\n".join(rows) if rows else "| — | — | — | — | — | — | — | — | — |"
    edges = [ev.edge for ev, _ in results if ev.edge == ev.edge]
    mean_edge = sum(edges) / len(edges) if edges else float("nan")

    return f"""# Per-Stock Direction Forecast (Extension B)

Generated at: {generated_at}

## Configuration

- Model: `{args.model}`
- Forecast horizon: {args.horizon} trading days
- Data period: {args.period}
- Train window: {args.train} trading days; step: {args.step}
- Probability calibration: {"on (Platt/sigmoid, leakage-safe CV)" if args.calibrate else "off"}
- Tickers: intended to be the core selector's top-N

## Results (walk-forward, out-of-sample)

| Ticker | N | Accuracy | Base rate | Edge | AUC | Brier | ECE | Current call |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{table}

- **Accuracy** — out-of-sample directional hit rate.
- **Base rate** — "always predict the majority direction"; accuracy must clear it.
- **Edge** — accuracy − base rate, in percentage points (positive = beats naive).
- **AUC** — ranking quality of P(up) vs chance (0.5).
- **Brier** — mean squared error of P(up) (lower = better probabilities).
- **ECE** — expected calibration error (lower = P(up) closer to the true frequency).
- **Current call** — P(up) over the next {args.horizon} days as of the latest close.

Mean edge across tickers: {mean_edge * 100:+.1f}pp

## Interpretation

Single-stock short-horizon direction is close to a coin flip, so the honest read
is relative: does out-of-sample accuracy beat the base rate, and is AUC above
0.5? A small or negative edge is an acceptable, informative outcome. This signal
is a *confirmation* layer on the core selector's picks, not a standalone strategy,
and is for evaluation only — not financial advice.
"""


def print_summary(results) -> None:
    print("=" * 90)
    print(f"{'Ticker':>7} {'N':>4} {'Acc':>7} {'Base':>7} {'Edge':>8} {'AUC':>6} {'Brier':>6} {'ECE':>6}  Current")
    print("-" * 90)
    for ev, fc in results:
        edge = "—" if ev.edge != ev.edge else f"{ev.edge * 100:+.1f}pp"
        print(f"{ev.ticker:>7} {ev.n_predictions:>4} {_pct(ev.accuracy):>7} {_pct(ev.base_rate):>7} "
              f"{edge:>8} {_auc(ev.auc):>6} {ev.brier:>6.3f} {ev.ece:>6.3f}  {fc.prob_up:.2f} ({fc.direction})")
    print("=" * 90)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Per-stock direction forecast (Extension B)")
    parser.add_argument("tickers", nargs="+", help="Tickers to confirm (the selector's top-N)")
    parser.add_argument("--model", choices=["gbm", "logistic", "baseline"], default="gbm",
                        help="Forecaster backend (default: gbm)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days (default 5)")
    parser.add_argument("--period", default="5y", help="History to download (default 5y)")
    parser.add_argument("--train", type=int, default=252, help="Train window in trading days (default 252)")
    parser.add_argument("--step", type=int, default=5, help="Days between walk-forward predictions (default 5)")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                        help="Disable probability calibration (default: on — makes P(up) a trustworthy frequency)")
    parser.set_defaults(calibrate=True)
    parser.add_argument("--no-save", action="store_true", help="Do not write a markdown report")
    args = parser.parse_args(argv)

    provider = YFinanceProvider()
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    results = []
    for ticker in [t.upper().strip() for t in args.tickers]:
        print(f"Downloading {ticker} ({args.period})...")
        try:
            results.append(_run_ticker(provider, ticker, args))
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not abort the run
            print(f"  {ticker}: skipped ({exc})")

    if not results:
        print("No tickers could be evaluated.")
        return

    print_summary(results)
    if not args.no_save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = reports_dir / f"direction_{args.model}_{stamp}.md"
        out.write_text(build_markdown_report(results, args))
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
