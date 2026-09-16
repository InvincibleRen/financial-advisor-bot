import numpy as np
import pandas as pd

from src.common import metrics


def test_cumulative_return_compounds():
    r = pd.Series([0.10, 0.10])  # 1.1 * 1.1 = 1.21
    assert round(metrics.cumulative_return(r), 4) == 0.21


def test_annualised_return_on_flat_series_is_zero():
    r = pd.Series([0.0] * 252)
    assert abs(metrics.annualised_return(r)) < 1e-9


def test_annualised_return_matches_geometric():
    # 252 days each +0.1% compounds to (1.001)^252 over exactly one year.
    r = pd.Series([0.001] * 252)
    expected = (1.001 ** 252) - 1.0
    assert abs(metrics.annualised_return(r) - expected) < 1e-9


def test_sharpe_zero_when_no_volatility():
    r = pd.Series([0.01] * 50)  # constant returns => zero std => Sharpe 0
    assert metrics.sharpe_ratio(r) == 0.0


def test_sharpe_positive_for_upward_noisy_series():
    rng = np.random.default_rng(0)
    r = pd.Series(0.0005 + rng.normal(0, 0.001, 500))
    assert metrics.sharpe_ratio(r) > 0


def test_max_drawdown_simple_case():
    # +100% then -50% => back to start; peak was 2.0, trough 1.0 => -50% drawdown.
    r = pd.Series([1.0, -0.5])
    assert abs(metrics.max_drawdown(r) - (-0.5)) < 1e-9


def test_max_drawdown_non_positive():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 300))
    assert metrics.max_drawdown(r) <= 0.0


def test_hit_rate_ignores_zero_days():
    r = pd.Series([0.0, 0.0, 0.01, -0.01, 0.01])  # 3 active, 2 up => 2/3
    assert abs(metrics.hit_rate(r) - (2 / 3)) < 1e-9


def test_transaction_costs_charged_on_turnover():
    market = pd.Series([0.0, 0.0, 0.0], index=pd.RangeIndex(3))
    # Enter on day 1 (0->1). Gross is zero everywhere, so net = -cost on turnover.
    positions = pd.Series([0.0, 1.0, 1.0], index=pd.RangeIndex(3))
    net = metrics.apply_transaction_costs(positions, market, cost_per_turnover=0.001)
    assert abs(net.iloc[1] - (-0.001)) < 1e-12
    assert abs(net.iloc[2]) < 1e-12  # no further turnover, no cost


def test_transaction_costs_no_lookahead():
    # A position taken on day t only earns the market return on day t+1.
    market = pd.Series([0.10, 0.10, 0.10], index=pd.RangeIndex(3))
    positions = pd.Series([1.0, 1.0, 1.0], index=pd.RangeIndex(3))
    net = metrics.apply_transaction_costs(positions, market, cost_per_turnover=0.0)
    assert abs(net.iloc[0]) < 1e-12          # no prior position => no return day 0
    assert abs(net.iloc[1] - 0.10) < 1e-12   # earns day-1 return via day-0 position
