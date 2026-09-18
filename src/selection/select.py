"""Top-N cross-sectional selection and walk-forward evaluation (Direction 1 core, Phase 2).

This is the point where the project becomes an *ML stock selector*. At each
rebalance date the ranker is trained on every observation strictly in the past,
scores the current cross-section, and the top-N are held (equal-weighted) until
the next rebalance. Realised returns are then compared against two naive
benchmarks — an equal-weight portfolio of the whole universe, and SPY when it is
supplied — using the existing ``common.metrics``.

Why not reuse ``common/walkforward.py``? That engine is *time-series* (one
ticker, a daily 0/1 position). Selection is *cross-sectional* (many tickers, one
choice per rebalance), so it needs its own walk-forward loop — but it deliberately
reports through the same metrics module, so selection and the rule-based strategy
are measured on an identical yardstick.

Leakage safety
--------------
Fold ``t`` trains only on rows with ``rebalance_date < t``. Because the label
horizon equals the rebalance spacing, the most recent training label resolves
exactly at ``t`` and is therefore already known — no future information leaks
into training. Scoring uses only the feature row *at* ``t`` (itself leakage-safe
by construction in ``features.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.common import metrics
from src.common.timeindex import to_naive_index
from src.selection.model import RankerModel

# Cross-sectional rebalancing is monthly, so metrics annualise on 12 periods.
PERIODS_PER_YEAR = 12
# Rolling training-window length, in monthly rebalances. Each walk-forward fold is
# trained on only the most recent ``TRAIN_WINDOW_MONTHS`` rebalances (a fixed
# 1-year look-back) rather than on all prior history (an expanding window). Set to
# ``None`` to restore the original expanding-window behaviour.
TRAIN_WINDOW_MONTHS: Optional[int] = 12
# Conventional benchmark ticker; when present in ``prices`` it is reported too.
BENCHMARK_TICKER = "SPY"
# Default per-turnover transaction cost, matching the rule-based walk-forward
# engine (10 bps per side). A full swap of an equal-weight top-N book has a
# turnover of 2.0, so a full monthly rebalance costs ~2 x this.
DEFAULT_COST_PER_TURNOVER = 0.001


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class SelectionBacktest:
    """Everything the evaluation chapter / report generator needs."""

    model_kind: str
    top_n: int
    used_sentiment: bool
    weights: pd.DataFrame = field(repr=False)
    strategy_returns: pd.Series = field(repr=False)
    benchmark_returns: pd.Series = field(repr=False)
    strategy_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_metrics: Dict[str, float] = field(default_factory=dict)
    spy_metrics: Optional[Dict[str, float]] = None
    cost_per_turnover: float = 0.0
    avg_turnover: float = 0.0
    # Prediction-quality metrics (judge the ranker, not just the portfolio).
    ranking_metrics: Dict[str, float] = field(default_factory=dict)
    # Consistency: fraction of holding periods the strategy beat the benchmark.
    monthly_win_rate: float = 0.0
    # Per-calendar-year strategy vs benchmark breakdown (for the report).
    yearly_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    # Information coefficient: rank-correlation of scores with realised forward
    # returns (the standard cross-sectional skill metric; more sensitive than AUC).
    information_coefficient: Dict[str, float] = field(default_factory=dict)
    # Market-neutral long-short (top-decile minus bottom-decile) diagnostic: it
    # isolates the *ranking* skill from the market/concentration exposure that a
    # long-only top-N book carries.
    long_short_metrics: Dict[str, float] = field(default_factory=dict)
    long_short_returns: pd.Series = field(
        default_factory=lambda: pd.Series(dtype="float64"), repr=False
    )


# --------------------------------------------------------------------------- #
# Selection                                                                   #
# --------------------------------------------------------------------------- #

def select_top_n(scores: pd.Series, n: int = 3) -> List[str]:
    """Return the ``n`` highest-scoring tickers for a single rebalance date."""
    if scores.empty:
        return []
    return list(scores.sort_values(ascending=False).head(n).index)


@dataclass
class _FoldRecord:
    """One walk-forward rebalance: the scored cross-section and the resulting picks."""

    date: pd.Timestamp
    scores: pd.Series
    picks: List[str]


def _walk_forward_records(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    n: int,
    model_kind: str,
    min_train_dates: int,
    train_window: Optional[int] = TRAIN_WINDOW_MONTHS,
    label_horizon_months: int = 1,
) -> List[_FoldRecord]:
    """Run the walk-forward once, returning per-rebalance scores + picks.

    A single training pass feeds both the portfolio weights and the ranking-quality
    metrics, so we never train the folds twice. Dates with fewer than
    ``min_train_dates`` prior rebalances are skipped (too little history).

    ``train_window`` bounds how many of the most recent prior rebalances each fold
    trains on: a fixed rolling look-back (default ``TRAIN_WINDOW_MONTHS`` = 12
    months = 1 year). ``None`` restores the original expanding window (all history).

    Leakage embargo
    ---------------
    A training row at date ``d`` carries a label that only resolves at
    ``d + label_horizon_months``. When the label horizon exceeds the monthly
    rebalance spacing (``label_horizon_months > 1``), the most recent training
    folds have labels that are still in the future at the scoring date ``t`` and
    must be dropped: we keep only ``d`` with ``d + horizon <= t``. For a 1-month
    horizon this excludes nothing (the horizon equals the spacing), preserving the
    original behaviour exactly.
    """
    dates = _usable_dates(feature_matrix, rebalance_dates)
    records: List[_FoldRecord] = []
    for i, rebalance_date in enumerate(dates):
        if i < min_train_dates:
            continue
        start = 0 if train_window is None else max(0, i - train_window)
        window = dates[start:i]
        if label_horizon_months > 1:
            cutoff = rebalance_date  # label must have resolved by the scoring date
            window = [d for d in window
                      if d + pd.DateOffset(months=label_horizon_months) <= cutoff]
        if not window:
            continue
        scores = _train_and_score(feature_matrix, labels, window, rebalance_date, model_kind)
        records.append(_FoldRecord(rebalance_date, scores, select_top_n(scores, n)))
    return records


def _records_to_weights(records: List[_FoldRecord]) -> pd.DataFrame:
    """Equal-weight the picks of each fold into a ``(date x ticker)`` weights frame."""
    weight_cells: Dict[tuple, float] = {}
    for record in records:
        if not record.picks:
            continue
        weight = 1.0 / len(record.picks)
        for ticker in record.picks:
            weight_cells[(record.date, ticker)] = weight
    return _weights_frame(weight_cells)


def build_selection_portfolio(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    n: int = 3,
    model_kind: str = "gbm",
    min_train_dates: int = 3,
    train_window: Optional[int] = TRAIN_WINDOW_MONTHS,
    label_horizon_months: int = 1,
) -> pd.DataFrame:
    """Walk-forward top-N selection -> equal-weight portfolio weights.

    Returns a DataFrame indexed by ``rebalance_date`` with one column per selected
    ticker; each row holds equal weights (``1/n``) on its picks and 0 elsewhere.
    Dates with fewer than ``min_train_dates`` prior rebalances are skipped.
    """
    records = _walk_forward_records(
        feature_matrix, labels, rebalance_dates, n, model_kind, min_train_dates,
        train_window=train_window, label_horizon_months=label_horizon_months,
    )
    return _records_to_weights(records)


def _train_and_score(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    past_dates: List[pd.Timestamp],
    rebalance_date: pd.Timestamp,
    model_kind: str,
) -> pd.Series:
    """Train on strictly-past rows, then score the cross-section at ``rebalance_date``."""
    date_level = feature_matrix.index.get_level_values("rebalance_date")
    train_mask = date_level.isin(past_dates)
    X_train = feature_matrix[train_mask]
    if X_train.empty:
        return pd.Series(dtype="float64")

    model = RankerModel(kind=model_kind).fit(X_train, labels)
    X_now = feature_matrix.loc[rebalance_date]  # rows for this date, indexed by ticker
    return model.predict_scores(X_now)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def evaluate_portfolio(
    weights: pd.DataFrame,
    prices: Dict[str, pd.DataFrame],
    periods_per_year: int = PERIODS_PER_YEAR,
    cost_per_turnover: float = 0.0,
) -> Dict[str, pd.Series]:
    """Turn a weights schedule into realised strategy and benchmark return streams.

    For each holding period ``[t, t_next)`` the strategy return is the weighted
    mean of its picks' realised returns, **net of transaction costs**; the
    benchmark is the equal-weight return of the whole universe over the same
    window. Returns are indexed by the *entry* rebalance date. The final rebalance
    has no closing window and is dropped.

    Transaction costs
    -----------------
    Cost is charged on *turnover* — the sum of absolute changes between the
    previous and the new target weights at each rebalance (the initial buy-in has
    a turnover of 1.0; a full swap of the top-N book is 2.0). This mirrors the
    rule-based walk-forward engine, so the two strategies are cost-comparable. The
    passive equal-weight / SPY benchmarks are treated as buy-and-hold (no periodic
    turnover) and are reported gross, which is the conservative choice — it makes
    the actively-traded strategy clear its costs against a cost-free benchmark.
    """
    closes = _close_series_by_ticker(prices)
    dates = list(weights.index)
    turnover = _turnover_schedule(weights)

    strategy: Dict[pd.Timestamp, float] = {}
    benchmark: Dict[pd.Timestamp, float] = {}
    spy: Dict[pd.Timestamp, float] = {}
    charged_turnover: Dict[pd.Timestamp, float] = {}

    for entry, exit_ in zip(dates[:-1], dates[1:]):
        gross = _weighted_period_return(weights.loc[entry], closes, entry, exit_)
        if gross is not None:
            cost = float(turnover.loc[entry]) * cost_per_turnover
            strategy[entry] = gross - cost
            charged_turnover[entry] = float(turnover.loc[entry])

        universe = [t for t in closes if t != BENCHMARK_TICKER]
        benchmark[entry] = _equal_weight_period_return(universe, closes, entry, exit_)

        if BENCHMARK_TICKER in closes:
            spy_ret = _period_return(closes[BENCHMARK_TICKER], entry, exit_)
            if spy_ret is not None:
                spy[entry] = spy_ret

    out = {
        "strategy": pd.Series(strategy, dtype="float64").sort_index(),
        "benchmark": pd.Series(benchmark, dtype="float64").sort_index(),
        "turnover": pd.Series(charged_turnover, dtype="float64").sort_index(),
    }
    if spy:
        out["spy"] = pd.Series(spy, dtype="float64").sort_index()
    return out


def _turnover_schedule(weights: pd.DataFrame) -> pd.Series:
    """Per-rebalance turnover = sum of absolute target-weight changes.

    The first rebalance compares against an all-cash book, so its turnover is the
    initial buy-in (1.0 for a fully-invested portfolio). Intra-period weight drift
    is ignored — a standard simplification that keeps the cost model transparent.
    """
    if weights.empty:
        return pd.Series(dtype="float64")
    previous = weights.shift(1).fillna(0.0)
    return (weights - previous).abs().sum(axis=1)


def run_selection_backtest(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    prices: Dict[str, pd.DataFrame],
    rebalance_dates: List[pd.Timestamp],
    n: int = 3,
    model_kind: str = "gbm",
    min_train_dates: int = 3,
    periods_per_year: int = PERIODS_PER_YEAR,
    cost_per_turnover: float = DEFAULT_COST_PER_TURNOVER,
    used_sentiment: bool = False,
    train_window: Optional[int] = TRAIN_WINDOW_MONTHS,
    label_horizon_months: int = 1,
) -> SelectionBacktest:
    """End-to-end: walk-forward select, evaluate net of costs, and bundle metrics.

    A single walk-forward pass produces both the portfolio (returns/costs) and the
    ranking-quality metrics (precision@N, AUC) — the folds are never trained twice.
    """
    records = _walk_forward_records(
        feature_matrix, labels, rebalance_dates, n, model_kind, min_train_dates,
        train_window=train_window, label_horizon_months=label_horizon_months,
    )
    weights = _records_to_weights(records)
    streams = evaluate_portfolio(
        weights, prices, periods_per_year=periods_per_year,
        cost_per_turnover=cost_per_turnover,
    )
    avg_turnover = float(streams["turnover"].mean()) if not streams["turnover"].empty else 0.0
    strat, bench = streams["strategy"], streams["benchmark"]

    ls_returns = long_short_returns(records, prices)
    ls_metrics = (
        metrics.compute_metrics(ls_returns, 0.0, periods_per_year)
        if not ls_returns.empty else {}
    )

    return SelectionBacktest(
        model_kind=model_kind,
        top_n=n,
        used_sentiment=used_sentiment,
        weights=weights,
        strategy_returns=strat,
        benchmark_returns=bench,
        strategy_metrics=metrics.compute_metrics(strat, 0.0, periods_per_year),
        benchmark_metrics=metrics.compute_metrics(bench, 0.0, periods_per_year),
        spy_metrics=(
            metrics.compute_metrics(streams["spy"], 0.0, periods_per_year)
            if "spy" in streams else None
        ),
        cost_per_turnover=cost_per_turnover,
        avg_turnover=avg_turnover,
        ranking_metrics=_ranking_metrics_from_records(records, labels),
        monthly_win_rate=_periodic_win_rate(strat, bench),
        yearly_breakdown=yearly_breakdown(strat, bench),
        information_coefficient=information_coefficient(records, prices),
        long_short_metrics=ls_metrics,
        long_short_returns=ls_returns,
    )


# --------------------------------------------------------------------------- #
# Prediction-quality + consistency metrics                                    #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Random-selection null (does the *ranking* earn its keep, or just the        #
# concentration?)                                                             #
# --------------------------------------------------------------------------- #


def _candidates_by_date(
    feature_matrix: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    min_train_dates: int,
) -> List[tuple]:
    """``(date, [tickers scoreable that date])`` for every date the selector traded.

    Mirrors the skipping rule in :func:`_walk_forward_records` so the null draws
    from exactly the cross-sections the real strategy chose from — no model is
    trained, because a random pick does not need one.
    """
    dates = _usable_dates(feature_matrix, rebalance_dates)
    out: List[tuple] = []
    for i, date in enumerate(dates):
        if i < min_train_dates:
            continue
        try:
            tickers = list(feature_matrix.loc[date].index)
        except KeyError:  # pragma: no cover - defensive
            continue
        if tickers:
            out.append((date, tickers))
    return out


def random_selection_null(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,  # noqa: ARG001 - kept for signature symmetry with the real run
    rebalance_dates: List[pd.Timestamp],
    prices: Dict[str, pd.DataFrame],
    n: int = 3,
    min_train_dates: int = 3,
    n_draws: int = 500,
    cost_per_turnover: float = DEFAULT_COST_PER_TURNOVER,
    periods_per_year: int = PERIODS_PER_YEAR,
    seed: int = 0,
) -> Dict[str, object]:
    """Null distribution for "pick ``n`` of the cross-section **at random**, monthly".

    Why this benchmark exists
    -------------------------
    The equal-weight and SPY benchmarks hold the *whole* universe, so beating them
    conflates two different things: the ranking picking good stocks, and a
    concentrated ``n``-stock book simply carrying more idiosyncratic risk. Drawing
    ``n`` tickers at random from the *same* cross-sections, with the *same* costs
    and rebalance dates, isolates the first from the second: the null keeps the
    concentration and throws away only the ranking.

    Returns the per-draw annualised return and Sharpe arrays plus their summary
    quantiles, so the real strategy can be located inside the distribution.
    """
    candidates = _candidates_by_date(feature_matrix, rebalance_dates, min_train_dates)
    rng = np.random.default_rng(seed)
    ann_returns: List[float] = []
    sharpes: List[float] = []

    for _ in range(int(n_draws)):
        cells: Dict[tuple, float] = {}
        for date, tickers in candidates:
            k = min(int(n), len(tickers))
            picks = rng.choice(len(tickers), size=k, replace=False)
            weight = 1.0 / k
            for j in picks:
                cells[(date, tickers[int(j)])] = weight
        weights = _weights_frame(cells)
        streams = evaluate_portfolio(
            weights, prices, periods_per_year=periods_per_year,
            cost_per_turnover=cost_per_turnover,
        )
        r = streams["strategy"].to_numpy(dtype="float64")
        r = r[np.isfinite(r)]
        if r.size < 2:
            continue
        ann_returns.append(float((1.0 + r).prod() ** (periods_per_year / r.size) - 1.0))
        sd = float(r.std(ddof=1))
        sharpes.append(float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0)

    ann = np.asarray(ann_returns, dtype="float64")
    shp = np.asarray(sharpes, dtype="float64")
    return {
        "n_draws": int(ann.size),
        "n": int(n),
        "null_ann_return": ann,
        "null_sharpe": shp,
        "ann_return_quantiles": _quantile_summary(ann),
        "sharpe_quantiles": _quantile_summary(shp),
    }


def _quantile_summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {k: float("nan") for k in ("p5", "p50", "p95", "mean")}
    q5, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
    return {"p5": float(q5), "p50": float(q50), "p95": float(q95), "mean": float(arr.mean())}


def locate_in_null(real_value: float, null_values: np.ndarray) -> Dict[str, float]:
    """Where does the real strategy sit inside the random-selection null?

    ``p_value`` is the one-sided fraction of random draws that match or beat the
    real strategy — the probability of doing this well by concentration alone.
    """
    arr = np.asarray(null_values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(real_value):
        return {"percentile": float("nan"), "p_value": float("nan")}
    beat = float((arr >= real_value).sum())
    return {
        "percentile": float((arr < real_value).mean() * 100.0),
        # +1 smoothing: a permutation-style p-value is never exactly zero.
        "p_value": (beat + 1.0) / (arr.size + 1.0),
    }


def ranking_metrics(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    n: int = 3,
    model_kind: str = "gbm",
    min_train_dates: int = 3,
    train_window: Optional[int] = TRAIN_WINDOW_MONTHS,
    label_horizon_months: int = 1,
) -> Dict[str, float]:
    """Standalone ranking-quality metrics (runs its own walk-forward pass)."""
    records = _walk_forward_records(
        feature_matrix, labels, rebalance_dates, n, model_kind, min_train_dates,
        train_window=train_window, label_horizon_months=label_horizon_months,
    )
    return _ranking_metrics_from_records(records, labels)


def _ranking_metrics_from_records(records: List["_FoldRecord"], labels: pd.Series) -> Dict[str, float]:
    """Judge the ranker itself, not the portfolio.

    * ``precision_at_n`` — of the names actually picked each rebalance, the fraction
      that went on to beat the universe median (label = 1). Compare against
      ``base_rate`` (≈0.5 by construction); above it means genuine selection skill.
    * ``mean_auc`` — average per-rebalance ROC-AUC of the scores against the binary
      labels, over folds where both classes are present. 0.5 = no better than
      chance, 1.0 = perfect ranking.
    """
    from sklearn.metrics import roc_auc_score  # local import (heavy dep is optional)

    precisions: List[float] = []
    aucs: List[float] = []
    label_pool: List[int] = []

    for record in records:
        pick_labels = _labels_for(labels, record.date, record.picks).dropna()
        if not pick_labels.empty:
            precisions.append(float((pick_labels == 1).mean()))

        cross_labels = _labels_for(labels, record.date, list(record.scores.index)).dropna()
        label_pool.extend(int(v) for v in cross_labels.to_numpy())
        if cross_labels.nunique() >= 2:
            aligned_scores = record.scores.reindex(cross_labels.index)
            aucs.append(float(roc_auc_score(cross_labels.to_numpy(), aligned_scores.to_numpy())))

    return {
        "precision_at_n": float(pd.Series(precisions).mean()) if precisions else float("nan"),
        "mean_auc": float(pd.Series(aucs).mean()) if aucs else float("nan"),
        "base_rate": float(pd.Series(label_pool).mean()) if label_pool else float("nan"),
        "n_folds": float(len(records)),
    }


def _labels_for(labels: pd.Series, date: pd.Timestamp, tickers: List[str]) -> pd.Series:
    """Labels for a set of tickers at one rebalance date (NaN where unknown)."""
    if not tickers:
        return pd.Series(dtype="float64")
    try:
        at_date = labels.loc[date]
    except KeyError:
        return pd.Series(index=tickers, dtype="float64")
    return at_date.reindex(tickers).astype("float64")


def _periodic_win_rate(strategy: pd.Series, benchmark: pd.Series) -> float:
    """Fraction of holding periods in which the strategy return beat the benchmark."""
    aligned = pd.concat([strategy, benchmark], axis=1, keys=["s", "b"]).dropna()
    if aligned.empty:
        return 0.0
    return float((aligned["s"] > aligned["b"]).mean())


def yearly_breakdown(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Per-calendar-year strategy vs benchmark: is the edge steady or lumpy?

    Returns a frame indexed by year with compounded strategy/benchmark returns, the
    number of holding periods, and the within-year fraction of periods the strategy
    won — so a reader can see whether one year carries the whole result.
    """
    aligned = pd.concat([strategy, benchmark], axis=1, keys=["strategy", "benchmark"]).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["strategy_return", "benchmark_return", "periods", "win_rate"])

    rows = {}
    for year, group in aligned.groupby(aligned.index.year):
        rows[int(year)] = {
            "strategy_return": float((1.0 + group["strategy"]).prod() - 1.0),
            "benchmark_return": float((1.0 + group["benchmark"]).prod() - 1.0),
            "periods": int(len(group)),
            "win_rate": float((group["strategy"] > group["benchmark"]).mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# --------------------------------------------------------------------------- #
# Information coefficient + long-short spread (isolate the ranking skill)      #
# --------------------------------------------------------------------------- #

def _scored_forward_returns(
    records: List["_FoldRecord"], prices: Dict[str, pd.DataFrame]
) -> List[tuple]:
    """``(scores, realised forward returns)`` per fold, over ``[date, next_date]``.

    The forward window is the holding period actually traded (this date to the next
    rebalance), read as-of from prices exactly as :func:`evaluate_portfolio` does,
    so the diagnostics below measure the same out-of-sample signal the portfolio
    used. The final fold has no closing window and is dropped.
    """
    closes = _close_series_by_ticker(prices)
    dates = [r.date for r in records]
    pairs: List[tuple] = []
    for record, exit_ in zip(records[:-1], dates[1:]):
        returns: Dict[str, float] = {}
        for ticker in record.scores.index:
            if ticker not in closes:
                continue
            period_ret = _period_return(closes[ticker], record.date, exit_)
            if period_ret is not None:
                returns[ticker] = period_ret
        if returns:
            pairs.append((record.scores, pd.Series(returns, dtype="float64")))
    return pairs


def information_coefficient(
    records: List["_FoldRecord"], prices: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """Cross-sectional Information Coefficient of the scores.

    Per rebalance the IC is the Spearman rank-correlation between the scores and
    the realised forward returns; the summary reports the mean IC, its volatility,
    the ICIR (mean / std) and a t-statistic (ICIR x sqrt(n)). IC is the standard,
    more sensitive measure of cross-sectional skill: mean IC ~ 0 means the ranking
    carries no monotonic information, and |t| >= ~2 is the usual bar for "real".
    """
    ics: List[float] = []
    for scores, fwd in _scored_forward_returns(records, prices):
        aligned = pd.concat([scores, fwd], axis=1, keys=["s", "r"]).dropna()
        if len(aligned) >= 3 and aligned["s"].nunique() > 1 and aligned["r"].nunique() > 1:
            ics.append(float(aligned["s"].corr(aligned["r"], method="spearman")))
    if not ics:
        return {"mean_ic": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "ic_t_stat": float("nan"), "n_periods": 0.0}
    arr = np.asarray(ics, dtype="float64")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    icir = mean / std if std and std > 0 else float("nan")
    t_stat = icir * math.sqrt(arr.size) if icir == icir else float("nan")
    return {"mean_ic": mean, "ic_std": std, "icir": icir,
            "ic_t_stat": t_stat, "n_periods": float(arr.size)}


def long_short_returns(
    records: List["_FoldRecord"],
    prices: Dict[str, pd.DataFrame],
    quantile: float = 0.1,
) -> pd.Series:
    """Return stream of an equal-weight top-decile-minus-bottom-decile book.

    Each rebalance goes long the top ``quantile`` of the scored cross-section and
    short the bottom ``quantile`` (at least one name per side), held to the next
    rebalance. The long-minus-short spread is market-neutral, so a positive,
    steady spread is direct evidence of monotonic ranking skill that the long-only
    top-N return (which also carries market beta) cannot isolate. Gross of costs.
    """
    closes = _close_series_by_ticker(prices)
    dates = [r.date for r in records]
    spread: Dict[pd.Timestamp, float] = {}
    for record, exit_ in zip(records[:-1], dates[1:]):
        scores = record.scores.dropna().sort_values(ascending=False)
        if len(scores) < 2:
            continue
        k = max(1, int(math.ceil(len(scores) * quantile)))
        longs = list(scores.index[:k])
        shorts = list(scores.index[-k:])
        long_ret = _equal_weight_period_return(longs, closes, record.date, exit_)
        short_ret = _equal_weight_period_return(shorts, closes, record.date, exit_)
        spread[record.date] = long_ret - short_ret
    return pd.Series(spread, dtype="float64").sort_index()


def run_ablation(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    prices: Dict[str, pd.DataFrame],
    rebalance_dates: List[pd.Timestamp],
    sentiment_column: str = "sentiment",
    **kwargs,
) -> Dict[str, SelectionBacktest]:
    """Extension-A study: run the backtest with and without the sentiment feature.

    Returns ``{"with_sentiment": ..., "without_sentiment": ...}`` so the report
    can quantify sentiment's marginal contribution. If the feature matrix has no
    sentiment column, only the ``without_sentiment`` run is produced.
    """
    results: Dict[str, SelectionBacktest] = {}
    if sentiment_column in feature_matrix.columns:
        results["with_sentiment"] = run_selection_backtest(
            feature_matrix, labels, prices, rebalance_dates, used_sentiment=True, **kwargs
        )
        stripped = feature_matrix.drop(columns=[sentiment_column])
    else:
        stripped = feature_matrix

    results["without_sentiment"] = run_selection_backtest(
        stripped, labels, prices, rebalance_dates, used_sentiment=False, **kwargs
    )
    return results


# --------------------------------------------------------------------------- #
# Return helpers                                                              #
# --------------------------------------------------------------------------- #

def _weighted_period_return(
    row_weights: pd.Series,
    closes: Dict[str, pd.Series],
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
) -> Optional[float]:
    """Weighted realised return of the held names over ``[entry, exit_]``."""
    total = 0.0
    weight_used = 0.0
    for ticker, weight in row_weights.items():
        if weight <= 0 or ticker not in closes:
            continue
        period_ret = _period_return(closes[ticker], entry, exit_)
        if period_ret is None:
            continue
        total += weight * period_ret
        weight_used += weight
    if weight_used == 0.0:
        return None
    # Renormalise in case a pick had no price this window, so weights still sum to 1.
    return total / weight_used


def _equal_weight_period_return(
    tickers: List[str],
    closes: Dict[str, pd.Series],
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
) -> float:
    """Equal-weight realised return across every ticker with a valid window."""
    period_returns = [
        r for t in tickers
        if (r := _period_return(closes[t], entry, exit_)) is not None
    ]
    if not period_returns:
        return 0.0
    return float(sum(period_returns) / len(period_returns))


def _period_return(series: pd.Series, entry: pd.Timestamp, exit_: pd.Timestamp) -> Optional[float]:
    """Realised return between the **tradeable** closes for the holding period.

    A position selected at a rebalance date can only be entered once that date's
    close is known, so both the entry and the exit are priced at the next trading
    day's close (T+1 execution), consistent with the labelling in ``labels.py``.
    """
    entry_price = _price_next_trading_day(series, entry)
    exit_price = _price_next_trading_day(series, exit_)
    if entry_price is None or exit_price is None or entry_price == 0:
        return None
    return (exit_price / entry_price) - 1.0


def _price_next_trading_day(series: pd.Series, when: pd.Timestamp) -> Optional[float]:
    """Close on the first trading day strictly after ``when`` (T+1 execution).

    ``None`` when ``when`` predates the series or no trading day follows it.
    """
    if when < series.index.min():
        return None
    pos = series.index.searchsorted(when, side="right")
    if pos >= len(series):
        return None
    value = series.iloc[pos]
    return None if pd.isna(value) else float(value)


# --------------------------------------------------------------------------- #
# Small structural helpers                                                    #
# --------------------------------------------------------------------------- #

def _usable_dates(feature_matrix: pd.DataFrame, rebalance_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """Sorted rebalance dates that actually have a cross-section in the matrix."""
    available = set(feature_matrix.index.get_level_values("rebalance_date"))
    requested = pd.to_datetime(list(rebalance_dates))
    return sorted(d for d in requested if d in available)


def _weights_frame(weight_cells: Dict[tuple, float]) -> pd.DataFrame:
    """Assemble the ``(date x ticker)`` weights DataFrame from sparse cells."""
    if not weight_cells:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="rebalance_date"))
    series = pd.Series(weight_cells)
    series.index = pd.MultiIndex.from_tuples(series.index, names=["rebalance_date", "ticker"])
    frame = series.unstack(fill_value=0.0)
    frame.index.name = "rebalance_date"
    return frame.sort_index()


def _close_series_by_ticker(prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Clean, sorted, tz-naive ``Close`` Series per ticker."""
    closes: Dict[str, pd.Series] = {}
    for ticker, frame in prices.items():
        if frame is None or "Close" not in frame.columns or frame.empty:
            continue
        series = frame["Close"].dropna()
        series.index = to_naive_index(series.index, normalize=True)
        closes[ticker] = series.sort_index()
    return closes
