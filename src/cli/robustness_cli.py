"""Robustness / self-honesty report for the core selector (evaluation layer).

Runs one walk-forward pass, then — reusing those trained folds — evaluates three
standard "did we fool ourselves?" checks:

* **Deflated Sharpe Ratio** — discounts the Sharpe for how many strategy variants
  were tried (guards against a lucky best-of-many result).
* **Sensitivity grid** — top-N x transaction-cost, to show the edge is not a
  cherry-picked single setting.
* **Bootstrap confidence intervals** — a 95% range for annualised return / Sharpe,
  so the reader sees the edge is not carried by a few months.

Example
-------
    python -m src.cli.robustness_cli --model gbm --n-trials 20
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.cli.select_cli import month_end_rebalances
from src.common import metrics
from src.selection import universe as U
from src.selection.features import build_feature_matrix, cross_sectional_normalize
from src.selection.labels import make_labels
from src.selection.robustness import (
    bootstrap_return_metrics,
    deflated_sharpe_ratio,
    sharpe_per_period,
)
from src.selection.select import (
    _FoldRecord,
    _labels_for,
    _walk_forward_records,
    _weights_frame,
    evaluate_portfolio,
    select_top_n,
)

TOP_NS = (3, 5, 10)
COSTS = (0.0, 0.001, 0.002)


def _weights_for_n(records: List[_FoldRecord], n: int) -> pd.DataFrame:
    cells: Dict[tuple, float] = {}
    for rec in records:
        picks = select_top_n(rec.scores, n)
        if not picks:
            continue
        w = 1.0 / len(picks)
        for t in picks:
            cells[(rec.date, t)] = w
    return _weights_frame(cells)


def _precision_at_n(records: List[_FoldRecord], labels: pd.Series, n: int) -> float:
    ps = []
    for rec in records:
        pl = _labels_for(labels, rec.date, select_top_n(rec.scores, n)).dropna()
        if not pl.empty:
            ps.append((pl == 1).mean())
    return float(np.mean(ps)) if ps else float("nan")


def compute_robustness(records, labels, prices, n_trials: int, execution_lag: int = 0,
                       primary_n: int = 3):
    """Return (grid_df, dsr_dict, bootstrap_dict) from already-trained folds.

    ``primary_n`` is the book the deflation and the bootstrap are computed on. It
    must be the same top-N the surrounding evaluation reports, because a deflated
    Sharpe quoted for one book and a headline Sharpe quoted for another describe
    different strategies.
    """
    grid, trial_sharpes = [], []
    for n in TOP_NS:
        w = _weights_for_n(records, n)
        prec = _precision_at_n(records, labels, n)
        for c in COSTS:
            strat = evaluate_portfolio(w, prices, periods_per_year=12, cost_per_turnover=c,
                                       execution_lag=execution_lag)["strategy"]
            m = metrics.compute_metrics(strat, 0.0, 12)
            trial_sharpes.append(sharpe_per_period(strat.to_numpy()))
            grid.append({"N": n, "cost_bps": int(c * 10000), "precision": prec,
                         "CAGR": m["annualised_return"], "Sharpe": m["sharpe_ratio"],
                         "MaxDD": m["max_drawdown"]})
    primary = evaluate_portfolio(_weights_for_n(records, primary_n), prices, periods_per_year=12,
                                 cost_per_turnover=0.001, execution_lag=execution_lag)["strategy"]
    dsr = deflated_sharpe_ratio(primary.to_numpy(), trial_sharpes=trial_sharpes, n_trials=n_trials)
    boot = bootstrap_return_metrics(primary.to_numpy(), periods_per_year=12)
    return pd.DataFrame(grid), dsr, boot


def dsr_across_trial_counts(records, prices, trial_counts, execution_lag: int = 0,
                            primary_n: int = 3):
    """DSR for the primary book under several assumptions about the search size.

    The number of configurations tried is a judgement call, not an observable, and
    the deflation is sensitive to it. Reporting a small ladder of counts is more
    honest than defending one number: the reader sees how the conclusion changes as
    the assumed search widens.
    """
    strat = evaluate_portfolio(_weights_for_n(records, primary_n), prices, periods_per_year=12,
                               cost_per_turnover=0.001, execution_lag=execution_lag)["strategy"]
    trial_sharpes = []
    for n in TOP_NS:
        w = _weights_for_n(records, n)
        for c in COSTS:
            s = evaluate_portfolio(w, prices, periods_per_year=12, cost_per_turnover=c,
                                   execution_lag=execution_lag)["strategy"]
            trial_sharpes.append(sharpe_per_period(s.to_numpy()))
    return [
        deflated_sharpe_ratio(strat.to_numpy(), trial_sharpes=trial_sharpes, n_trials=int(n))
        for n in trial_counts
    ]


def _interpretation(grid: pd.DataFrame, dsr: Dict[str, float], boot) -> str:
    """Describe what these three checks actually returned, pass or fail.

    Earlier revisions hard-coded a favourable reading of the checks, which meant the
    file claimed the edge had survived even on runs where it had not. The wording is
    now derived from the numbers, so a weak result reports itself as weak.
    """
    survives = dsr["dsr"] >= 0.95
    sharpes = grid["Sharpe"].dropna()
    all_positive = bool(len(sharpes)) and bool((sharpes > 0).all())
    lower = boot["ann_sharpe"][1]

    parts = []
    parts.append(
        f"The deflated Sharpe is {dsr['dsr']:.4f} against a best-of-{int(dsr['n_trials'])} "
        f"benchmark of {dsr['sr0']:.4f}, so on this assumption about the size of the search "
        + ("the observed Sharpe is unlikely to be the lucky best of many attempts."
           if survives else
           "the observed Sharpe cannot be separated from the best of many attempts.")
    )
    if all_positive:
        parts.append(
            f"Sharpe stays positive in all {len(sharpes)} top-N / cost cells "
            f"(range {sharpes.min():.2f} to {sharpes.max():.2f}), so the result is not a "
            "single cherry-picked setting."
        )
    else:
        n_neg = int((sharpes <= 0).sum())
        parts.append(
            f"Sharpe is non-positive in {n_neg} of {len(sharpes)} top-N / cost cells "
            f"(range {sharpes.min():.2f} to {sharpes.max():.2f}), so the result depends on "
            "the particular setting chosen."
        )
    parts.append(
        f"The bootstrap 95% lower bound on the annualised Sharpe is {lower:.2f}, "
        + ("which stays above zero." if lower > 0 else
           "which includes zero, so the sample cannot rule out that the edge is noise.")
    )
    verdict = (
        "All three checks are passed."
        if (survives and all_positive and lower > 0)
        else "Not every check is passed, and the weaker ones are reported above rather than "
             "set aside."
    )
    return " ".join(parts) + f" {verdict} Out-of-sample research evaluation only; not financial advice."


def build_report(grid: pd.DataFrame, dsr: Dict[str, float], boot, n_trials: int,
                 dsr_ladder: Optional[List[Dict[str, float]]] = None) -> str:
    ladder = ""
    if dsr_ladder:
        ladder_rows = "\n".join(
            f"| {int(d['n_trials'])} | {d['sr0']:.4f} | {d['dsr']:.4f} |" for d in dsr_ladder
        )
        ladder = (
            "\nThe count of configurations tried is a judgement, so the deflation is also "
            "reported across a range of assumptions:\n\n"
            "| Assumed trials | Benchmark SR0 | Deflated Sharpe |\n|---:|---:|---:|\n"
            f"{ladder_rows}\n"
        )
    rows = "\n".join(
        f"| {int(r.N)} | {int(r.cost_bps)} | {r.precision:.3f} | {r.CAGR*100:.1f}% | "
        f"{r.Sharpe:.2f} | {r.MaxDD*100:.1f}% |"
        for r in grid.itertuples()
    )
    ar, sh = boot["ann_return"], boot["ann_sharpe"]
    return f"""# Robustness / Self-Honesty Report (core selector)

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

A single backtest can fool you three ways; each is checked below.

## 1. Not just luck from many trials — Deflated Sharpe Ratio

| Quantity | Value |
|---|---:|
| Observed Sharpe (per-period) | {dsr['sr']:.4f} |
| Best-of-{int(dsr['n_trials'])}-trials benchmark SR0 | {dsr['sr0']:.4f} |
| **Deflated Sharpe (prob. true SR beats the benchmark)** | **{dsr['dsr']:.4f}** |

The DSR discounts the Sharpe for having tried ~{n_trials} configurations. A value
near 1 means the result is very unlikely to be a lucky best-of-many.
{ladder}
## 2. Not a cherry-picked setting — sensitivity grid

| Top-N | Cost (bps) | Precision@N | CAGR | Sharpe | Max DD |
|---:|---:|---:|---:|---:|---:|
{rows}

## 3. Not carried by a few months — bootstrap 95% CIs (primary book, 10 bps)

| Metric | Median | 95% CI |
|---|---:|---:|
| Annualised return | {ar[0]*100:.1f}% | [{ar[1]*100:.1f}%, {ar[2]*100:.1f}%] |
| Annualised Sharpe | {sh[0]:.2f} | [{sh[1]:.2f}, {sh[2]:.2f}] |

## Interpretation

{_interpretation(grid, dsr, boot)}
"""


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Robustness (DSR + sensitivity grid + bootstrap) for the selector")
    p.add_argument("tickers", nargs="*", default=None)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--model", choices=["gbm", "logistic", "rf", "xgb"], default="rf")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of strategy configurations tried (for the DSR deflation). "
                        "Count the whole search that led to the reported configuration, not "
                        "the final comparison alone; under-counting inflates the DSR.")
    p.add_argument("--trial-ladder", default="10,30,50,80",
                   help="Comma-separated trial counts for the DSR sensitivity ladder "
                        "(empty string disables it)")
    p.add_argument("--label-benchmark", choices=["median", "mean"], default="median",
                   dest="label_benchmark",
                   help="Must match the selection run being tested: the cross-sectional bar a "
                        "stock has to beat to count as a positive example.")
    p.add_argument("--execution-lag", type=int, choices=[0, 1], default=0, dest="execution_lag",
                   help="Must match the selection run being tested: 0 prices at the rebalance "
                        "close (default), 1 at the next trading day's close.")
    p.add_argument("--top-n", type=int, default=3, dest="top_n",
                   help="The book the deflation and bootstrap are computed on. Set it to the "
                        "same top-N the selection run being tested reports (default 3).")
    p.add_argument("--min-train", type=int, default=6)
    p.add_argument("--sector-neutral", action="store_true",
                   help="Normalise within (date, sector); must match the selection run being tested.")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    universe = [t.upper().strip() for t in args.tickers] if args.tickers else U.DEFAULT_UNIVERSE
    print(f"Loading + building features for {len(universe)} tickers...")
    prices = U.load_prices(universe, start=args.start)
    rebalance_dates = month_end_rebalances(prices)
    sectors = U.load_sectors(universe) if args.sector_neutral else None
    fm = cross_sectional_normalize(build_feature_matrix(prices, U.load_fundamentals(universe), rebalance_dates), method="rank", sectors=sectors)
    labels = make_labels(prices, rebalance_dates, horizon_months=1,
                         execution_lag=args.execution_lag,
                         benchmark=args.label_benchmark)

    print(f"Walk-forward ({args.model}), then robustness checks...")
    records = _walk_forward_records(fm, labels, rebalance_dates, 3, args.model, args.min_train)
    grid, dsr, boot = compute_robustness(records, labels, prices, args.n_trials,
                                         execution_lag=args.execution_lag, primary_n=args.top_n)
    counts = [int(x) for x in args.trial_ladder.split(",") if x.strip()]
    ladder = dsr_across_trial_counts(records, prices, counts, execution_lag=args.execution_lag,
                                     primary_n=args.top_n) if counts else None

    print(grid.to_string(index=False))
    print(f"Deflated Sharpe: {dsr['dsr']:.4f}  (SR {dsr['sr']:.3f} vs best-of-{int(dsr['n_trials'])} benchmark {dsr['sr0']:.3f})")
    print(f"Bootstrap ann. Sharpe 95% CI: [{boot['ann_sharpe'][1]:.2f}, {boot['ann_sharpe'][2]:.2f}]")

    if not args.no_save:
        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        out = reports_dir / f"robustness_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        out.write_text(build_report(grid, dsr, boot, args.n_trials, ladder))
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
