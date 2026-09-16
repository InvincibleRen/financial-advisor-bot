"""Offline tests for the Phase 2 walk-forward selector (`selection/select.py`).

A deterministic, engineered universe: a stock's (constant) quality drives both an
informative feature and its forward return, so a competent ranker must select the
best names and beat an equal-weight benchmark. No network, no randomness in the
data-generating process.
"""
import warnings

import pandas as pd
import pytest

from src.selection import features as F
from src.selection import select as S

# Quality is fixed per ticker: AAA best ... DDD worst. Monthly return = quality * 5%.
_QUALITY = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": 0.0}
_FEATURE_COLUMNS = F.TECHNICAL_FEATURES + F.FUNDAMENTAL_FEATURES


def _rebalance_dates(n=14):
    return list(pd.date_range("2018-01-31", periods=n, freq="ME"))


def _feature_matrix(dates):
    """One row per (date, ticker); all features 0 except an informative ``roe``."""
    rows = {}
    for date in dates:
        for ticker, quality in _QUALITY.items():
            row = {col: 0.0 for col in _FEATURE_COLUMNS}
            row["roe"] = quality  # the only feature carrying signal
            rows[(date, ticker)] = row
    matrix = pd.DataFrame.from_dict(rows, orient="index")[_FEATURE_COLUMNS]
    matrix.index = pd.MultiIndex.from_tuples(matrix.index, names=["rebalance_date", "ticker"])
    return matrix.sort_index()


def _labels(dates):
    """1 when a ticker's forward return beats the cross-sectional median (7.5%)."""
    cells = {}
    for date in dates:
        for ticker, quality in _QUALITY.items():
            cells[(date, ticker)] = int(quality * 0.05 > 0.075)  # AAA, BBB -> 1
    series = pd.Series(cells, name="label")
    series.index = pd.MultiIndex.from_tuples(series.index, names=["rebalance_date", "ticker"])
    return series.sort_index()


def _prices(dates, include_spy=False):
    """Close paths that realise each ticker's quality-implied monthly return."""
    prices = {}
    for ticker, quality in _QUALITY.items():
        monthly = quality * 0.05
        closes = [100.0 * (1.0 + monthly) ** k for k in range(len(dates))]
        prices[ticker] = pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))
    if include_spy:
        closes = [100.0 * (1.0 + 0.05) ** k for k in range(len(dates))]  # ~5%/mo
        prices["SPY"] = pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))
    return prices


@pytest.fixture(autouse=True)
def _quiet_convergence():
    # Perfectly separable toy data can trip LogisticRegression's convergence warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------- #
# select_top_n                                                                #
# --------------------------------------------------------------------------- #

def test_select_top_n_picks_highest_scores():
    scores = pd.Series({"AAA": 0.9, "BBB": 0.5, "CCC": 0.1})
    assert S.select_top_n(scores, 2) == ["AAA", "BBB"]
    assert S.select_top_n(pd.Series(dtype="float64"), 2) == []


# --------------------------------------------------------------------------- #
# Walk-forward selection                                                      #
# --------------------------------------------------------------------------- #

def test_selector_picks_the_best_ticker():
    dates = _rebalance_dates()
    weights = S.build_selection_portfolio(
        _feature_matrix(dates), _labels(dates), dates, n=1,
        model_kind="logistic", min_train_dates=3,
    )
    # Early dates lack training history and are excluded.
    assert weights.index.min() == dates[3]
    # Every evaluated date puts its full weight on AAA.
    assert (weights["AAA"] == 1.0).all()
    assert weights.drop(columns=["AAA"]).to_numpy().sum() == 0.0


def test_weights_sum_to_one_each_rebalance():
    dates = _rebalance_dates()
    weights = S.build_selection_portfolio(
        _feature_matrix(dates), _labels(dates), dates, n=2, model_kind="logistic",
    )
    row_sums = weights.sum(axis=1)
    assert (abs(row_sums - 1.0) < 1e-9).all()


# --------------------------------------------------------------------------- #
# End-to-end backtest                                                         #
# --------------------------------------------------------------------------- #

def test_strategy_beats_equal_weight_benchmark():
    dates = _rebalance_dates()
    result = S.run_selection_backtest(
        _feature_matrix(dates), _labels(dates), _prices(dates), dates,
        n=1, model_kind="logistic", min_train_dates=3,
    )
    assert set(result.strategy_metrics) >= {"cumulative_return", "sharpe_ratio", "max_drawdown"}
    # Picking the best name must beat holding the whole universe equally.
    assert result.strategy_metrics["cumulative_return"] > result.benchmark_metrics["cumulative_return"]
    assert result.spy_metrics is None  # no SPY supplied


def test_turnover_schedule_counts_weight_changes():
    dates = _rebalance_dates(3)
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.0], "B": [0.5, 0.0, 0.0], "C": [0.0, 0.5, 1.0]},
        index=pd.DatetimeIndex(dates, name="rebalance_date"),
    )
    turnover = S._turnover_schedule(weights)
    # Row 0 buy-in = 1.0; row 1 swaps B->C (half each) = 1.0; row 2 = 1.0.
    assert list(turnover.round(6)) == [1.0, 1.0, 1.0]


def test_transaction_costs_reduce_strategy_returns():
    dates = _rebalance_dates()
    fm, lbl, px = _feature_matrix(dates), _labels(dates), _prices(dates)

    gross = S.run_selection_backtest(
        fm, lbl, px, dates, n=1, model_kind="logistic", min_train_dates=3,
        cost_per_turnover=0.0,
    )
    net = S.run_selection_backtest(
        fm, lbl, px, dates, n=1, model_kind="logistic", min_train_dates=3,
        cost_per_turnover=0.02,
    )
    assert net.cost_per_turnover == 0.02
    assert net.avg_turnover > 0.0
    # Charging costs can only lower (or equal) the realised return.
    assert net.strategy_metrics["cumulative_return"] < gross.strategy_metrics["cumulative_return"]


def test_spy_benchmark_is_reported_when_present():
    dates = _rebalance_dates()
    result = S.run_selection_backtest(
        _feature_matrix(dates), _labels(dates), _prices(dates, include_spy=True), dates,
        n=1, model_kind="logistic",
    )
    assert result.spy_metrics is not None
    assert "cumulative_return" in result.spy_metrics


# --------------------------------------------------------------------------- #
# Prediction-quality + consistency metrics                                    #
# --------------------------------------------------------------------------- #

def test_ranking_metrics_reward_a_skilful_ranker():
    dates = _rebalance_dates()
    rm = S.ranking_metrics(
        _feature_matrix(dates), _labels(dates), dates,
        n=1, model_kind="logistic", min_train_dates=3,
    )
    # roe perfectly ranks the labels, so the ranker should be flawless.
    assert rm["precision_at_n"] == 1.0
    assert rm["mean_auc"] > 0.99
    assert abs(rm["base_rate"] - 0.5) < 1e-9   # 2 of 4 names beat the median
    assert rm["n_folds"] > 0


def test_backtest_bundles_ranking_and_yearly_breakdown():
    dates = _rebalance_dates()
    res = S.run_selection_backtest(
        _feature_matrix(dates), _labels(dates), _prices(dates), dates,
        n=1, model_kind="logistic", min_train_dates=3,
    )
    assert res.ranking_metrics["precision_at_n"] == 1.0
    assert not res.yearly_breakdown.empty
    assert set(res.yearly_breakdown.columns) == {
        "strategy_return", "benchmark_return", "periods", "win_rate"
    }
    assert 0.0 <= res.monthly_win_rate <= 1.0


def test_periodic_win_rate_counts_period_beats():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    strat = pd.Series([0.10, -0.10, 0.20], index=idx)
    bench = pd.Series([0.05, 0.00, 0.30], index=idx)
    # period 1 wins (0.10>0.05), 2 loses, 3 loses -> 1/3.
    assert S._periodic_win_rate(strat, bench) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Sentiment ablation (Extension A hook)                                        #
# --------------------------------------------------------------------------- #

def test_ablation_runs_with_and_without_sentiment():
    dates = _rebalance_dates()
    matrix = _feature_matrix(dates)
    matrix["sentiment"] = 0.0  # present but uninformative here

    results = S.run_ablation(
        matrix, _labels(dates), _prices(dates), dates,
        n=1, model_kind="logistic",
    )
    assert set(results) == {"with_sentiment", "without_sentiment"}
    assert results["with_sentiment"].used_sentiment is True
    assert results["without_sentiment"].used_sentiment is False
