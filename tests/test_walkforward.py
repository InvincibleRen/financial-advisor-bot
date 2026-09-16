import numpy as np
import pandas as pd

from src.common.walkforward import (
    rule_based_positions,
    run_walk_forward,
    walk_forward_windows,
)


def _synthetic_prices(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """A trending, noisy price series long enough for several folds."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    steps = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"Close": close}, index=dates)


def test_windows_are_contiguous_and_nonoverlapping():
    windows = walk_forward_windows(n=1000, train_size=252, test_size=63)
    assert len(windows) > 0
    # Default step == test_size, so each test window starts where the last ended.
    for (_t1, test1), (_t2, test2) in zip(windows, windows[1:]):
        assert test2.start == test1.stop


def test_windows_respect_bounds():
    windows = walk_forward_windows(n=400, train_size=252, test_size=63)
    for train, test in windows:
        assert train.start >= 0
        assert test.stop <= 400


def test_windows_raise_on_bad_sizes():
    try:
        walk_forward_windows(n=100, train_size=0, test_size=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_walk_forward_structure():
    data = _synthetic_prices()
    result = run_walk_forward(data, ticker="TEST", train_size=252, test_size=63)
    assert len(result.folds) >= 3
    # Aggregated and overall metric bundles are populated.
    for key in ("mean_sharpe_ratio", "mean_max_drawdown", "mean_hit_rate"):
        assert key in result.aggregated_strategy
    for key in ("cumulative_return", "sharpe_ratio", "max_drawdown"):
        assert key in result.overall_strategy
    # Stitched series length == folds * test_size.
    assert len(result.stitched_returns) == len(result.folds) * 63


def test_walk_forward_raises_on_insufficient_data():
    data = _synthetic_prices(n=200)
    try:
        run_walk_forward(data, train_size=252, test_size=63)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_costs_reduce_or_equal_returns():
    data = _synthetic_prices(seed=3)
    free = run_walk_forward(data, train_size=252, test_size=63, cost_per_turnover=0.0)
    costed = run_walk_forward(data, train_size=252, test_size=63, cost_per_turnover=0.002)
    # Charging turnover costs cannot improve the overall return.
    assert costed.overall_strategy["cumulative_return"] <= free.overall_strategy["cumulative_return"] + 1e-9


def test_positions_are_binary():
    data = _synthetic_prices(seed=5)
    pos = rule_based_positions(data)
    assert set(pd.unique(pos.dropna())).issubset({0.0, 1.0})
