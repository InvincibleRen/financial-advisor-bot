import pandas as pd
import pytest

from src.common.backtester import calculate_backtest_signals, run_backtest


def test_calculate_backtest_signals_creates_expected_columns():
    data = pd.DataFrame({"Close": list(range(1, 251))})

    result = calculate_backtest_signals(data)

    assert "SMA20" in result.columns
    assert "SMA50" in result.columns
    assert "SMA200" in result.columns
    assert "score" in result.columns
    assert "raw_signal" in result.columns
    assert "target_position" in result.columns


def test_run_backtest_returns_summary():
    data = pd.DataFrame({"Close": list(range(1, 301))})

    result = run_backtest(data, ticker="TEST", initial_cash=10_000)

    assert result.ticker == "TEST"
    assert result.initial_cash == 10_000
    assert result.final_strategy_value > 0
    assert result.final_buy_hold_value > 0
    assert isinstance(result.strategy_return_percent, float)
    assert isinstance(result.buy_hold_return_percent, float)


def test_run_backtest_rejects_empty_data():
    data = pd.DataFrame()

    with pytest.raises(ValueError):
        run_backtest(data, ticker="EMPTY")