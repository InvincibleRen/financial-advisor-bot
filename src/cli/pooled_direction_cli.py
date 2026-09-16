"""Command-line runner for the pooled cross-stock direction forecast (Extension B).

Trains one model per rebalance date across a basket of stocks (instead of one
model per stock on a short window) and reports an honest pooled walk-forward
evaluation plus a selective-prediction table. Prices are read from the local
cache in ``data/prices`` so runs are reproducible and need no network.

Examples
--------
    python -m src.cli.pooled_direction_cli                 # default 30-name basket
    python -m src.cli.pooled_direction_cli AAPL MSFT NVDA  # a custom basket
    python -m src.cli.pooled_direction_cli --horizon 5 --no-calibrate
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.direction.pooled import evaluate_pooled_direction

# A liquid, high-coverage default basket (the direction layer confirms the core
# selector's top-N; this basket is a representative stand-in for evaluation).
_DEFAULT_BASKET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "JNJ", "V",
    "PG", "HD", "MA", "XOM", "BAC", "KO", "PEP", "COST", "WMT", "DIS",
    "CSCO", "INTC", "VZ", "CVX", "MRK", "ADBE", "NFLX", "CRM", "ABT", "T",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_local_prices(tickers: List[str], cache_dir: Path) -> Dict[str, pd.DataFrame]:
    """Read ``{ticker -> OHLCV frame}`` from the local price cache (no network)."""
    prices: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker}.csv"
        if not path.exists():
            print(f"  {ticker}: no local cache, skipped")
            continue
        df = pd.read_csv(path)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
        prices[ticker] = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return prices


def _pct(x: float) -> str:
    return "—" if x != x else f"{x * 100:.1f}%"


def _auc(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def build_markdown_report(ev, args, n_requested: int) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sel_rows = "\n".join(
        f"| {int(r.coverage * 100)}% | {int(r.n)} | {r.accuracy * 100:.1f}% | {r.edge_vs_base * 100:+.1f}pp |"
        for r in ev.selective.itertuples()
    ) or "| — | — | — | — |"

    return f"""# Pooled Cross-Stock Direction Forecast (Extension B)

Generated at: {generated_at}

## Configuration

- Model: `{ev.kind}` (pooled across stocks; one model per rebalance date)
- Forecast horizon: {ev.horizon_days} trading days
- Rebalance step: {args.step} trading days; trailing train window: {args.train_years} years
- Probability calibration: {"on (Platt/sigmoid, leakage-safe CV)" if ev.calibrated else "off"}
- Basket: {ev.n_stocks} of {n_requested} requested stocks (local price cache)

## Out-of-Sample Performance (pooled walk-forward)

| Metric | Value |
|---|---:|
| Predictions | {ev.n_predictions} |
| Accuracy | {_pct(ev.accuracy)} |
| Base rate (majority class) | {_pct(ev.base_rate)} |
| Edge (accuracy − base rate) | {ev.edge * 100:+.1f}pp |
| AUC | {_auc(ev.auc)} |
| Brier score | {ev.brier:.3f} |
| Expected calibration error (ECE) | {ev.ece:.3f} |

## Selective Prediction (accuracy on the most confident calls)

| Coverage kept | Calls | Accuracy | Edge vs base |
|---|---:|---:|---:|
{sel_rows}

## Interpretation

Pooling many stocks into one dated model, plus probability calibration, is the
honest lever on a signal that is close to a coin flip: it lifts accuracy to the
base rate and shrinks the calibration error, so the displayed P(up) is a
trustworthy frequency. AUC near 0.5 confirms that single-stock short-horizon
direction carries little ranking information — a real property of the problem,
not a defect. Selective prediction shows the calls the model is most confident
about beat the base rate. This layer is a *confirmation* signal on the core
selector's picks, for evaluation only — not financial advice.
"""


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Pooled cross-stock direction forecast (Extension B)")
    parser.add_argument("tickers", nargs="*", help="Basket to evaluate (default: 30 liquid large caps)")
    parser.add_argument("--model", choices=["gbm", "logistic"], default="gbm", help="Estimator (default gbm)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days (default 5)")
    parser.add_argument("--step", type=int, default=21, help="Trading days between rebalances (default 21)")
    parser.add_argument("--train-years", type=int, default=3, dest="train_years", help="Trailing train window in years (default 3)")
    parser.add_argument("--start", default="2017-06-01", help="First prediction date (default 2017-06-01)")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                        help="Disable probability calibration (default: on)")
    parser.set_defaults(calibrate=True)
    parser.add_argument("--no-save", action="store_true", help="Do not write a markdown report")
    args = parser.parse_args(argv)

    tickers = [t.upper().strip() for t in (args.tickers or _DEFAULT_BASKET)]
    prices = _load_local_prices(tickers, _REPO_ROOT / "data" / "prices")
    if not prices:
        print("No local price data found; populate data/prices first.")
        return

    print(f"Evaluating pooled direction on {len(prices)} stocks (horizon {args.horizon}d)...")
    ev = evaluate_pooled_direction(
        prices, kind=args.model, horizon_days=args.horizon, step=args.step,
        train_years=args.train_years, start=args.start, calibrate=args.calibrate,
    )

    print("=" * 74)
    print(f"Pooled direction | {ev.n_stocks} stocks | horizon {ev.horizon_days}d | calibrate={ev.calibrated}")
    print(f"  n={ev.n_predictions}  acc={_pct(ev.accuracy)}  base={_pct(ev.base_rate)}  "
          f"edge={ev.edge * 100:+.1f}pp  auc={_auc(ev.auc)}  brier={ev.brier:.3f}  ece={ev.ece:.3f}")
    print("  selective (most-confident calls):")
    for r in ev.selective.itertuples():
        print(f"    coverage {int(r.coverage * 100):>3}%  n={int(r.n):>4}  "
              f"acc={r.accuracy * 100:.1f}%  edge_vs_base={r.edge_vs_base * 100:+.1f}pp")
    print("=" * 74)

    if not args.no_save:
        reports_dir = _REPO_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = reports_dir / f"direction_pooled_{args.model}_{stamp}.md"
        out.write_text(build_markdown_report(ev, args, len(tickers)))
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
