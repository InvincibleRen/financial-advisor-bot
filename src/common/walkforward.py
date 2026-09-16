"""Walk-forward evaluation of the advisor strategy.

The single-split backtester in ``backtester.py`` measures the strategy over one
train/test period, which the literature (Bailey et al., 2014) warns is fragile:
a single split can flatter or punish a strategy by luck of the period chosen.
This module implements *walk-forward* evaluation instead. History is divided into
a sequence of consecutive test windows; the strategy is evaluated on each unseen
window in turn, advancing through time. Standard finance metrics are reported
both **per fold** and **aggregated across all folds**, and a single stitched
equity curve gives the overall out-of-sample result.

The engine is deliberately *model-agnostic*. It accepts a ``position_func`` that
maps a price history to a daily target position (1 = invested, 0 = cash). The
rule-based strategy is provided as the default, and the same fold machinery will
drive a learned scikit-learn ranker later without changing this file. Because the
rule-based signals are causal (they use only past/current prices), evaluating
each test window in place introduces no look-ahead bias; transaction costs are
charged on turnover so results reflect realistic trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.common import metrics
from src.common.backtester import calculate_backtest_signals

# A position function maps a price DataFrame to a target-position Series.
PositionFunc = Callable[[pd.DataFrame], pd.Series]


def rule_based_positions(data: pd.DataFrame) -> pd.Series:
    """Default strategy: the project's rule-based signal, as a 0/1 position.

    Reuses ``calculate_backtest_signals`` so the walk-forward test evaluates
    exactly the same SMA/crossover/MACD/ADX logic the live advisor recommends.
    """
    signal_data = calculate_backtest_signals(data)
    return signal_data["target_position"].astype(float)


@dataclass
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    strategy: Dict[str, float]
    benchmark: Dict[str, float]
    n_trades: int


@dataclass
class WalkForwardResult:
    ticker: str
    train_size: int
    test_size: int
    step: int
    cost_per_turnover: float
    folds: List[Fold]
    aggregated_strategy: Dict[str, float]
    aggregated_benchmark: Dict[str, float]
    overall_strategy: Dict[str, float]
    overall_benchmark: Dict[str, float]
    stitched_returns: pd.Series = field(repr=False, default_factory=pd.Series)


def walk_forward_windows(
    n: int,
    train_size: int,
    test_size: int,
    step: Optional[int] = None,
) -> List[Tuple[slice, slice]]:
    """Generate (train_slice, test_slice) index ranges over ``n`` observations.

    Each fold trains on ``train_size`` rows and tests on the following
    ``test_size`` rows, then advances by ``step`` (defaults to ``test_size`` so
    the test windows are contiguous and non-overlapping).
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive.")

    windows: List[Tuple[slice, slice]] = []
    start = 0
    while start + train_size + test_size <= n:
        train_slice = slice(start, start + train_size)
        test_slice = slice(start + train_size, start + train_size + test_size)
        windows.append((train_slice, test_slice))
        start += step
    return windows


def _count_trades(positions: pd.Series) -> int:
    """Number of entries (0 -> 1 transitions) within a position series."""
    prev = positions.shift(1).fillna(0.0)
    return int(((prev == 0.0) & (positions == 1.0)).sum())


def run_walk_forward(
    data: pd.DataFrame,
    ticker: str = "UNKNOWN",
    train_size: int = 252,
    test_size: int = 63,
    step: Optional[int] = None,
    cost_per_turnover: float = 0.001,
    risk_free_rate: float = 0.0,
    position_func: PositionFunc = rule_based_positions,
) -> WalkForwardResult:
    """Run a full walk-forward evaluation on one ticker's price history.

    Defaults: train on ~1 year (252 days), test on ~1 quarter (63 days), 10 bps
    per-side transaction costs. The benchmark on every fold is buy-and-hold over
    the identical test window.
    """
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column.")

    data = data.dropna(subset=["Close"]).copy()

    # Compute causal positions once over the full history; slicing test windows
    # out of a causal series is leakage-free (each day depends only on the past).
    positions_full = position_func(data)
    market_returns_full = data["Close"].pct_change().fillna(0.0)

    windows = walk_forward_windows(len(data), train_size, test_size, step)
    if not windows:
        raise ValueError(
            f"Not enough data for a walk-forward run: need at least "
            f"{train_size + test_size} rows, got {len(data)}."
        )

    folds: List[Fold] = []
    strat_fold_metrics: List[Dict[str, float]] = []
    bench_fold_metrics: List[Dict[str, float]] = []
    stitched: List[pd.Series] = []

    for i, (_train, test) in enumerate(windows):
        test_index = data.index[test]
        test_positions = positions_full.loc[test_index]
        test_market = market_returns_full.loc[test_index]

        strat_returns = metrics.apply_transaction_costs(
            test_positions, test_market, cost_per_turnover
        )
        # Benchmark = always invested (buy-and-hold) over the same window.
        bench_returns = test_market

        strat_metrics = metrics.compute_metrics(strat_returns, risk_free_rate)
        bench_metrics = metrics.compute_metrics(bench_returns, risk_free_rate)

        strat_fold_metrics.append(strat_metrics)
        bench_fold_metrics.append(bench_metrics)
        stitched.append(strat_returns)

        folds.append(
            Fold(
                index=i + 1,
                train_start=str(data.index[_train.start].date()),
                train_end=str(data.index[_train.stop - 1].date()),
                test_start=str(test_index[0].date()),
                test_end=str(test_index[-1].date()),
                strategy=strat_metrics,
                benchmark=bench_metrics,
                n_trades=_count_trades(test_positions),
            )
        )

    stitched_returns = pd.concat(stitched)
    # A stitched benchmark over exactly the same union of test windows.
    stitched_bench = pd.concat(
        [market_returns_full.loc[data.index[test]] for _t, test in windows]
    )

    return WalkForwardResult(
        ticker=ticker,
        train_size=train_size,
        test_size=test_size,
        step=test_size if step is None else step,
        cost_per_turnover=cost_per_turnover,
        folds=folds,
        aggregated_strategy=metrics.aggregate_fold_metrics(strat_fold_metrics),
        aggregated_benchmark=metrics.aggregate_fold_metrics(bench_fold_metrics),
        overall_strategy=metrics.compute_metrics(stitched_returns, risk_free_rate),
        overall_benchmark=metrics.compute_metrics(stitched_bench, risk_free_rate),
        stitched_returns=stitched_returns,
    )
