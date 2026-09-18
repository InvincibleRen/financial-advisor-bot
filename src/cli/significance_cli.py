"""Statistical-significance report for the selection ranker (evaluation layer).

Runs the label-permutation test and bootstrap confidence intervals on the core
walk-forward selector, so the evaluation chapter can state whether the ranker's
out-of-sample skill is significant rather than a lucky single path.

Examples
--------
    python -m src.cli.significance_cli                       # both models, 100 shuffles
    python -m src.cli.significance_cli --model gbm --permutations 200
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.cli.select_cli import month_end_rebalances
from src.selection import universe as U
from src.selection.features import build_feature_matrix, cross_sectional_normalize
from src.selection.labels import make_labels
from src.selection.select import _walk_forward_records
from src.selection.significance import SignificanceResult, permutation_test


def _histogram(values: np.ndarray, real: float, bins: int = 20, width: int = 40) -> str:
    """A tiny text histogram of the null distribution with the real value marked."""
    counts, edges = np.histogram(values, bins=bins)
    peak = max(int(counts.max()), 1)
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / peak))
        marker = "  <- REAL" if lo <= real < hi else ""
        lines.append(f"  {lo:6.4f}–{hi:6.4f} | {bar}{marker}")
    return "\n".join(lines)


def _verdict(p: float) -> str:
    if p < 0.01:
        return "significant at 1% (strong evidence of real skill)"
    if p < 0.05:
        return "significant at 5% (evidence of real skill)"
    if p < 0.10:
        return "marginal (weak evidence; not significant at 5%)"
    return "not significant (indistinguishable from chance)"


def build_report(results: Dict[str, SignificanceResult], universe: List[str], n_perm: int, top_n: int) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = []
    for model, r in results.items():
        ci = r.auc_ci
        sections.append(f"""### {model}

| Statistic | Real | Null mean | p-value | Verdict |
|---|---:|---:|---:|---|
| Mean AUC | {r.real_mean_auc:.4f} | {r.null_auc.mean():.4f} | {r.p_value_auc:.4f} | {_verdict(r.p_value_auc)} |
| Precision@N | {r.real_precision_at_n:.4f} | {r.null_precision.mean():.4f} | {r.p_value_precision:.4f} | {_verdict(r.p_value_precision)} |

- **Base rate** (label = 1 frequency): {r.base_rate:.4f}
- **Bootstrap 95% CI on per-fold mean AUC**: [{ci[1]:.4f}, {ci[2]:.4f}] (median {ci[0]:.4f}, {len(r.fold_aucs)} folds)

Null distribution of mean AUC over {n_perm} label shuffles:

```
{_histogram(r.null_auc, r.real_mean_auc)}
```
""")
    body = "\n".join(sections)
    return f"""# Statistical Significance of the Selection Ranker

Generated at: {generated_at}

## Method

The walk-forward selector produces one out-of-sample score per (rebalance date,
ticker). Holding those **real scores fixed**, the outcome labels are shuffled
**within each rebalance date's cross-section** and the ranking statistic (mean
AUC, precision@N) is recomputed — {n_perm} times — to build the "no-skill" null
distribution. The p-value is `(1 + #{{null >= real}}) / (1 + {n_perm})`. Because
only labels are permuted (never re-training, never crossing dates), the test is
leakage-free and makes no distributional assumptions. Bootstrap resampling of the
per-fold AUCs adds a 95% confidence interval — the error bar a single backtest
lacks.

## Configuration

- Universe: {len(universe)} tickers
- Top-N: {top_n} · Feature normalisation: cross-sectional rank (leakage-safe)
- Label: forward 1-month return vs cross-sectional median
- Permutations: {n_perm}

## Results

{body}

## Interpretation

A low p-value means random outcomes rarely reproduce the observed ranking quality,
so the edge is unlikely to be luck. A precision@N that is significant even when the
whole-cross-section AUC is only marginal tells you the skill concentrates in the
**top of the ranking** — exactly where the strategy acts. This is an out-of-sample
research evaluation and does not guarantee future performance.
"""


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Permutation + bootstrap significance for the selection ranker")
    p.add_argument("tickers", nargs="*", default=None)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--model", choices=["gbm", "logistic", "rf", "xgb", "both"], default="both")
    p.add_argument("--label-benchmark", choices=["median", "mean"], default="median",
                   dest="label_benchmark",
                   help="Must match the selection run being tested: the cross-sectional bar a "
                        "stock has to beat to count as a positive example.")
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--min-train", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sector-neutral", action="store_true",
                   help="Normalise within (date, sector); must match the selection run being tested.")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    universe = [t.upper().strip() for t in args.tickers] if args.tickers else U.DEFAULT_UNIVERSE
    print(f"Loading + building features for {len(universe)} tickers...")
    prices = U.load_prices(universe, start=args.start)
    fundamentals = U.load_fundamentals(universe)
    rebalance_dates = month_end_rebalances(prices)
    sectors = U.load_sectors(universe) if args.sector_neutral else None
    fm = cross_sectional_normalize(build_feature_matrix(prices, fundamentals, rebalance_dates), method="rank", sectors=sectors)
    labels = make_labels(prices, rebalance_dates, horizon_months=1,
                         benchmark=args.label_benchmark)

    kinds = ["gbm", "logistic"] if args.model == "both" else [args.model]
    results: Dict[str, SignificanceResult] = {}
    for kind in kinds:
        print(f"Walk-forward ({kind}) then {args.permutations} label shuffles...")
        records = _walk_forward_records(fm, labels, rebalance_dates, args.top_n, kind, args.min_train)
        results[kind] = permutation_test(records, labels, n=args.top_n,
                                         n_permutations=args.permutations, seed=args.seed)
        r = results[kind]
        print(f"  {kind}: AUC {r.real_mean_auc:.4f} (p={r.p_value_auc:.4f}), "
              f"precision@N {r.real_precision_at_n:.4f} (p={r.p_value_precision:.4f})")

    if not args.no_save:
        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = reports_dir / f"significance_{args.model}_{stamp}.md"
        out.write_text(build_report(results, list(universe), args.permutations, args.top_n))
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
