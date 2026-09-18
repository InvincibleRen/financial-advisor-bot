"""Command-line entry point for the cross-sectional ML stock selector (Phase 2).

Runs the full Direction-1 core end to end: load the universe's prices and
fundamentals, build the leakage-safe feature matrix and forward-return labels,
then walk-forward select the top-N each month and evaluate the realised portfolio
against an equal-weight universe benchmark and SPY. Both the gradient-boosting
ranker and the logistic baseline are reported so the ML model can be judged
against a simple linear benchmark.

Examples
--------
    python -m src.cli.select_cli                       # default universe, top-3
    python -m src.cli.select_cli --top-n 5 --model gbm
    python -m src.cli.select_cli AAPL MSFT NVDA AMZN GOOGL META --start 2015-01-01

Writes a markdown report to ``reports/`` and prints a summary to the terminal.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.common.timeindex import to_naive_index
from src.selection import universe as U
from src.selection.features import (
    FUNDAMENTAL_FEATURES,
    TECHNICAL_FEATURES,
    build_feature_matrix,
    cross_sectional_normalize,
)
from src.selection.labels import make_labels
from src.selection.select import (
    DEFAULT_COST_PER_TURNOVER,
    SelectionBacktest,
    run_selection_backtest,
)

BENCHMARK_TICKER = "SPY"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _num(value: float) -> str:
    return f"{value:.2f}"


# Below this the fundamental block is judged effectively absent (a data-source
# limitation of yfinance, which only returns a handful of recent quarters).
FUND_COVERAGE_WARN = 0.20


def feature_coverage(feature_matrix: pd.DataFrame) -> Dict[str, float]:
    """Fraction of non-NaN cells per feature column (before normalisation).

    Surfaces how much of each factor actually reached the model. It exists because
    yfinance returns only a few recent quarters of fundamentals, so on a long
    back-test the valuation/quality/growth columns can be almost entirely NaN — a
    "14-factor" model then trains, in practice, on the technical block alone. The
    run must say so rather than imply a full fundamental factor set was used.
    """
    if feature_matrix.empty:
        return {c: 0.0 for c in TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES}
    return {c: float(feature_matrix[c].notna().mean())
            for c in feature_matrix.columns}


def _coverage_lines(coverage: Dict[str, float]) -> List[str]:
    tech = [c for c in TECHNICAL_FEATURES if c in coverage]
    fund = [c for c in FUNDAMENTAL_FEATURES if c in coverage]
    lines = []
    lines.append("  technical: " + ", ".join(f"{c} {coverage[c]*100:.0f}%" for c in tech))
    lines.append("  fundamental: " + ", ".join(f"{c} {coverage[c]*100:.0f}%" for c in fund))
    return lines


def fundamental_coverage_mean(coverage: Dict[str, float]) -> float:
    fund = [coverage[c] for c in FUNDAMENTAL_FEATURES if c in coverage]
    return float(sum(fund) / len(fund)) if fund else 0.0


def month_end_rebalances(
    prices: Dict[str, pd.DataFrame], min_names: int = 10
) -> List[pd.Timestamp]:
    """Month-end rebalance dates spanning the universe's data coverage.

    A broad universe (e.g. the S&P 500) contains recent IPOs and the odd delisted
    name, so the strict *intersection* of every ticker's date range can collapse
    to almost nothing (one 2024 IPO would drag the common start to 2024). Instead
    we span from the earliest to the latest available date and keep only the
    month-ends on which at least ``min_names`` tickers are already public and still
    trading (an as-of "live" cross-section). Tickers that are not yet listed on a
    given date, or already delisted, are simply absent from that cross-section,
    which the leakage-safe feature/label builders handle natively.
    """
    ranges = []
    for frame in prices.values():
        if frame is None or frame.empty:
            continue
        idx = to_naive_index(frame.index, normalize=True)
        ranges.append((idx.min(), idx.max()))
    if not ranges:
        return []
    starts = [s for s, _ in ranges]
    ends = [e for _, e in ranges]
    # Never require more live names than the universe actually holds, so small or
    # curated universes (and the offline tests) still yield rebalance dates; the
    # floor only trims thin months when the universe is genuinely large.
    threshold = max(1, min(min_names, len(ranges)))
    candidates = pd.date_range(min(starts), max(ends), freq="ME")
    return [d for d in candidates
            if sum(1 for s, e in ranges if s <= d <= e) >= threshold]


def _metrics_row(name: str, m: Optional[Dict[str, float]]) -> str:
    if not m:
        return f"| {name} | — | — | — | — | — |"
    return (
        f"| {name} | {_pct(m['cumulative_return'])} | {_pct(m['annualised_return'])} | "
        f"{_num(m['sharpe_ratio'])} | {_pct(m['max_drawdown'])} | {_pct(m['hit_rate'])} |"
    )


def _latest_picks(result: SelectionBacktest) -> str:
    if result.weights.empty:
        return "—"
    last_date = result.weights.index.max()
    row = result.weights.loc[last_date]
    picks = list(row[row > 0].index)
    return f"{last_date.date()}: {', '.join(picks) if picks else '—'}"


def _fmt(value: float, as_pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return _pct(value) if as_pct else f"{value:.3f}"


def _ranking_row(kind: str, result: SelectionBacktest) -> str:
    r = result.ranking_metrics or {}
    return (
        f"| {kind} | {_fmt(r.get('precision_at_n'))} | {_fmt(r.get('mean_auc'))} | "
        f"{_fmt(r.get('base_rate'))} | {_pct(result.monthly_win_rate)} |"
    )


def _ic_ls_row(kind: str, result: SelectionBacktest) -> str:
    ic = result.information_coefficient or {}
    ls = result.long_short_metrics or {}
    ls_ret = ls.get("cumulative_return")
    ls_sharpe = ls.get("sharpe_ratio")
    return (
        f"| {kind} | {_fmt(ic.get('mean_ic'))} | {_fmt(ic.get('icir'))} | "
        f"{_fmt(ic.get('ic_t_stat'))} | "
        f"{_fmt(ls_ret, as_pct=True) if ls_ret is not None else '—'} | "
        f"{_fmt(ls_sharpe)} |"
    )


def _yearly_table(kind: str, result: SelectionBacktest) -> str:
    frame = result.yearly_breakdown
    if frame is None or frame.empty:
        return f"**{kind}** — no per-year data."
    lines = [
        f"**{kind}**",
        "",
        "| Year | Strategy | Benchmark | Periods | Win-rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for year, row in frame.iterrows():
        lines.append(
            f"| {year} | {_pct(row['strategy_return'])} | {_pct(row['benchmark_return'])} | "
            f"{int(row['periods'])} | {_pct(row['win_rate'])} |"
        )
    return "\n".join(lines)


def build_markdown_report(
    results: Dict[str, SelectionBacktest],
    universe: List[str],
    n_rebalances: int,
    top_n: int,
    normalize: str = "rank",
    coverage: Optional[Dict[str, float]] = None,
    horizon: int = 1,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coverage = coverage or {}
    fund_mean = fundamental_coverage_mean(coverage) if coverage else float("nan")
    if coverage:
        tech = [c for c in TECHNICAL_FEATURES if c in coverage]
        fund = [c for c in FUNDAMENTAL_FEATURES if c in coverage]
        coverage_block = (
            "- Feature coverage (non-NaN cells, before normalisation):\n"
            + "    - technical: "
            + ", ".join(f"{c} {coverage[c]*100:.0f}%" for c in tech) + "\n"
            + "    - fundamental: "
            + ", ".join(f"{c} {coverage[c]*100:.0f}%" for c in fund)
        )
        if fund_mean < FUND_COVERAGE_WARN:
            coverage_block += (
                f"\n- **Caveat:** mean fundamental coverage is only {fund_mean*100:.0f}%. "
                "The valuation/quality/growth block is largely NaN over this window "
                "(yfinance returns only a few recent quarters), so this run is in "
                "practice a **technical-factor model**; results must not be read as "
                "evidence about a full fundamental factor set."
            )
    else:
        coverage_block = ""
    any_result = next(iter(results.values()))
    benchmark_row = _metrics_row("Equal-weight universe", any_result.benchmark_metrics)
    spy_row = _metrics_row("SPY (buy & hold)", any_result.spy_metrics)
    cost_bps = any_result.cost_per_turnover * 10000

    model_rows = "\n".join(
        _metrics_row(f"Selector — {kind}", res.strategy_metrics)
        for kind, res in results.items()
    )
    turnover_rows = "\n".join(
        f"- **{kind}** — average monthly turnover {res.avg_turnover:.2f} "
        f"(1.0 = one full position replaced)"
        for kind, res in results.items()
    )
    ranking_rows = "\n".join(_ranking_row(kind, res) for kind, res in results.items())
    ic_ls_rows = "\n".join(_ic_ls_row(kind, res) for kind, res in results.items())
    yearly_section = "\n\n".join(
        _yearly_table(kind, res) for kind, res in results.items()
    )
    picks_rows = "\n".join(
        f"- **{kind}** — latest selection {_latest_picks(res)}"
        for kind, res in results.items()
    )

    return f"""# Cross-Sectional Stock Selection (Direction 1 core)

Generated at: {generated_at}

## Configuration

- Universe: {len(universe)} tickers ({', '.join(universe)})
- Rebalance: monthly ({n_rebalances} dates)
- Top-N held each rebalance: {top_n}
- Label: forward {horizon}-month return vs cross-sectional median (leakage-safe{', with embargo' if horizon > 1 else ''})
- Feature normalisation: cross-sectional {normalize} per rebalance date (leakage-safe)
- Transaction cost: {cost_bps:.0f} bps per unit turnover (strategy only; benchmarks are buy-and-hold)
- Benchmarks: equal-weight universe, and SPY buy-and-hold
{coverage_block}

## Out-of-Sample Performance (walk-forward, monthly, net of costs)

| Portfolio | Cumulative | Annualised | Sharpe | Max DD | Hit rate |
|---|---:|---:|---:|---:|---:|
{model_rows}
{benchmark_row}
{spy_row}

## Turnover

{turnover_rows}

## Prediction Quality (judging the ranker, not the portfolio)

| Model | Precision@N | Mean AUC | Base rate | Monthly win-rate vs bench |
|---|---:|---:|---:|---:|
{ranking_rows}

*Precision@N = fraction of picked names that then beat the universe median (skill
shows as precision above the ~0.5 base rate). Mean AUC = average per-rebalance
ranking quality of the scores (0.5 = chance). Monthly win-rate = fraction of
months the strategy return beat the benchmark.*

## Ranking Skill — Information Coefficient & Long-Short Spread

| Model | Mean IC | ICIR | IC t-stat | L/S cumulative | L/S Sharpe |
|---|---:|---:|---:|---:|---:|
{ic_ls_rows}

*IC = per-rebalance Spearman rank-correlation between the scores and realised
forward returns; mean IC ~ 0 means no monotonic ranking information, and |t-stat|
>= ~2 is the usual bar for a real edge. The long-short book goes long the top
decile and short the bottom decile of the scores each month (gross, market-neutral),
so its spread isolates ranking skill from the market and concentration exposure the
long-only top-N return also carries.*

## Consistency — Per-Year Strategy vs Benchmark

{yearly_section}

## Latest Selection

{picks_rows}

## Interpretation

Each month the ranker is trained only on data strictly before the rebalance date,
scores the current cross-section and holds the top-{top_n} equally until the next
rebalance. Strategy returns are **net of {cost_bps:.0f} bps per-turnover costs**;
the passive benchmarks are reported gross (buy-and-hold), so the actively-traded
selector must clear its trading costs to win. The gradient-boosting ranker is
shown alongside the logistic baseline so any edge over a simple linear model is
explicit. This is an out-of-sample research evaluation only and does not guarantee
future performance.
"""


def print_summary(results: Dict[str, SelectionBacktest]) -> None:
    print("=" * 74)
    print("Cross-Sectional Selection — out-of-sample (monthly walk-forward)")
    print("=" * 74)
    any_result = next(iter(results.values()))
    print(f"(strategy net of {any_result.cost_per_turnover * 10000:.0f} bps/turnover; "
          f"benchmarks gross/buy-and-hold)")
    for kind, res in results.items():
        m = res.strategy_metrics
        print(f"{kind:>9} selector:  return {_pct(m['cumulative_return']):>10}  "
              f"Sharpe {_num(m['sharpe_ratio']):>6}  MaxDD {_pct(m['max_drawdown']):>8}  "
              f"turnover {res.avg_turnover:>4.2f}")
    bench = any_result.benchmark_metrics
    print(f"{'equal-wt':>9} bench:     return {_pct(bench['cumulative_return']):>10}  "
          f"Sharpe {_num(bench['sharpe_ratio']):>6}  MaxDD {_pct(bench['max_drawdown']):>8}")
    print("-" * 74)
    print("Prediction quality (ranker skill) + consistency:")
    for kind, res in results.items():
        r = res.ranking_metrics or {}
        print(f"{kind:>9}:  precision@N {_fmt(r.get('precision_at_n')):>6}  "
              f"AUC {_fmt(r.get('mean_auc')):>6}  (base {_fmt(r.get('base_rate')):>6})  "
              f"monthly-win {_pct(res.monthly_win_rate):>7}")
    print("-" * 74)
    print("Information coefficient (score vs forward return) + long-short spread:")
    for kind, res in results.items():
        ic = res.information_coefficient or {}
        ls = res.long_short_metrics or {}
        print(f"{kind:>9}:  mean IC {_fmt(ic.get('mean_ic')):>6}  "
              f"ICIR {_fmt(ic.get('icir')):>6}  t {_fmt(ic.get('ic_t_stat')):>6}  |  "
              f"L/S return {_pct(ls.get('cumulative_return', float('nan'))):>9}  "
              f"Sharpe {_fmt(ls.get('sharpe_ratio')):>6}")
    print("=" * 74)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional ML stock selector (Direction 1 core)")
    parser.add_argument("tickers", nargs="*", default=None,
                        help="Universe tickers (default: the built-in DEFAULT_UNIVERSE)")
    parser.add_argument("--start", default="2015-01-01", help="History start date (default 2015-01-01)")
    parser.add_argument("--end", default=None, help="History end date (default: today)")
    parser.add_argument("--top-n", type=int, default=3, help="Stocks held each rebalance (default 3)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Label horizon in months: train to predict the N-month-ahead "
                             "relative winners (default 1). Values >1 add a leakage embargo so "
                             "training only uses folds whose label has already resolved.")
    parser.add_argument("--model", choices=["gbm", "logistic", "rf", "xgb", "both", "all"], default="both",
                        help="Ranker to run (default: both, for baseline comparison)")
    parser.add_argument("--min-train", type=int, default=6,
                        help="Minimum prior rebalances before a fold is evaluated (default 6)")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST_PER_TURNOVER,
                        help="Per-turnover transaction cost (default 0.001 = 10 bps); use 0 for gross")
    parser.add_argument("--execution-lag", type=int, choices=[0, 1], default=0,
                        dest="execution_lag",
                        help="Trading days between the signal date and the price at which "
                             "positions transact. 0 (default) prices at the rebalance close, "
                             "the standard monthly-factor convention; 1 prices at the next "
                             "trading day's close, which removes the simultaneity of "
                             "transacting at the very close used to form the signal "
                             "(reported as a robustness check).")
    parser.add_argument("--label-benchmark", choices=["median", "mean"], default="median",
                        dest="label_benchmark",
                        help="Cross-sectional bar a stock must beat to be a positive example. "
                             "'median' (default) follows the factor literature and splits each "
                             "date in half; 'mean' sets the bar at the equal-weight portfolio "
                             "return, aligning the training target with the benchmark the "
                             "strategy is actually judged against.")
    parser.add_argument("--normalize", choices=["rank", "zscore", "none"], default="rank",
                        help="Cross-sectional feature standardisation per rebalance date "
                             "(default: rank — improves ranking AUC/precision; 'none' = raw levels)")
    parser.add_argument("--sector-neutral", action="store_true",
                        help="Standardise features within (date, sector) instead of the whole "
                             "cross-section, so the ranker compares each stock to its sector peers "
                             "rather than making sector bets (fetches sectors via yfinance, cached).")
    parser.add_argument("--no-save", action="store_true", help="Do not write a markdown report")
    args = parser.parse_args(argv)

    universe = [t.upper().strip() for t in args.tickers] if args.tickers else U.DEFAULT_UNIVERSE

    print(f"Loading prices + fundamentals for {len(universe)} tickers...")
    prices = U.load_prices(universe, start=args.start, end=args.end)
    fundamentals = U.load_fundamentals(universe)
    spy = U.load_prices([BENCHMARK_TICKER], start=args.start, end=args.end)

    rebalance_dates = month_end_rebalances(prices)
    if len(rebalance_dates) <= args.min_train + 1:
        raise SystemExit("Not enough history for a walk-forward run; widen --start.")

    print(f"Building feature matrix + labels over {len(rebalance_dates)} monthly rebalances...")
    feature_matrix = build_feature_matrix(prices, fundamentals, rebalance_dates)

    # Measure how much of each factor actually reached the model, before any
    # normalisation. On a long back-test yfinance fundamentals are only a few
    # recent quarters, so the valuation/quality/growth block can be ~all NaN — the
    # run must report that instead of implying a full fundamental model was trained.
    coverage = feature_coverage(feature_matrix)
    print("Feature coverage (non-NaN cells, pre-normalisation):")
    for line in _coverage_lines(coverage):
        print(line)
    fund_mean = fundamental_coverage_mean(coverage)
    if fund_mean < FUND_COVERAGE_WARN:
        print(f"WARNING: mean fundamental coverage {fund_mean*100:.0f}% < "
              f"{FUND_COVERAGE_WARN*100:.0f}%. This run is effectively a "
              "technical-factor model; treat any fundamental conclusions as unsupported.")

    if args.normalize != "none":
        sectors = None
        scope = "per date"
        if args.sector_neutral:
            sectors = U.load_sectors(universe)
            scope = f"per (date, sector); {len(sectors)}/{len(universe)} tickers mapped"
        print(f"Applying cross-sectional {args.normalize} normalisation (leakage-safe, {scope})...")
        feature_matrix = cross_sectional_normalize(
            feature_matrix, method=args.normalize, sectors=sectors
        )
    labels = make_labels(prices, rebalance_dates, horizon_months=args.horizon,
                         execution_lag=args.execution_lag,
                         benchmark=args.label_benchmark)
    if args.horizon > 1:
        print(f"Label horizon = {args.horizon} months; applying a {args.horizon - 1}-month "
              "leakage embargo (training only uses folds whose label has resolved).")

    prices_eval = dict(prices)
    prices_eval.update(spy)  # add SPY for the buy-and-hold benchmark only

    kinds = {"both": ["gbm", "logistic"],
             "all": ["logistic", "gbm", "rf", "xgb"]}.get(args.model, [args.model])
    results: Dict[str, SelectionBacktest] = {}
    for kind in kinds:
        print(f"Running walk-forward selection with the {kind} ranker...")
        results[kind] = run_selection_backtest(
            feature_matrix, labels, prices_eval, rebalance_dates,
            n=args.top_n, model_kind=kind, min_train_dates=args.min_train,
            cost_per_turnover=args.cost, label_horizon_months=args.horizon,
            execution_lag=args.execution_lag,
        )

    print_summary(results)

    if not args.no_save:
        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = reports_dir / f"selection_top{args.top_n}_{stamp}.md"
        out.write_text(build_markdown_report(
            results, universe, len(rebalance_dates), args.top_n, args.normalize,
            coverage=coverage, horizon=args.horizon,
        ))
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
