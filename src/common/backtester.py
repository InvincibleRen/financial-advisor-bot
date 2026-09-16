from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.common.indicators import add_adx, add_macd, add_moving_averages


@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    initial_cash: float
    final_strategy_value: float
    final_buy_hold_value: float
    strategy_return_percent: float
    buy_hold_return_percent: float
    max_drawdown_percent: float
    number_of_trades: int
    win_rate_percent: float
    trade_returns_percent: List[float]
    data: pd.DataFrame = field(repr=False)


def calculate_backtest_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical rule-based signals.

    The strategy is based on:
    - Close above/below SMA20
    - Close above/below SMA50
    - Close above/below SMA200
    - Golden Cross / Death Cross events
    - MACD(12, 26, 9) momentum (line vs signal)
    - ADX(14) trend strength, which amplifies the prevailing bullish/bearish lean

    This mirrors the live signal logic in advisor.generate_recommendation, so the
    backtest evaluates the same enhanced strategy the prototype recommends.

    Output:
    - score: rule-based strength score
    - target_position: 1 means holding the stock, 0 means not holding
    """
    if data.empty:
        raise ValueError("Backtest data is empty.")

    if "Close" not in data.columns:
        raise ValueError("Backtest data must contain a 'Close' column.")

    result = add_adx(add_macd(add_moving_averages(data)))
    result = result.copy()

    result["score"] = 0

    result.loc[result["Close"] > result["SMA20"], "score"] += 1
    result.loc[result["Close"] < result["SMA20"], "score"] -= 1

    result.loc[result["Close"] > result["SMA50"], "score"] += 1
    result.loc[result["Close"] < result["SMA50"], "score"] -= 1

    result.loc[result["Close"] > result["SMA200"], "score"] += 1
    result.loc[result["Close"] < result["SMA200"], "score"] -= 1

    previous_sma50 = result["SMA50"].shift(1)
    previous_sma200 = result["SMA200"].shift(1)

    golden_cross = (previous_sma50 <= previous_sma200) & (result["SMA50"] > result["SMA200"])
    death_cross = (previous_sma50 >= previous_sma200) & (result["SMA50"] < result["SMA200"])

    result.loc[golden_cross, "score"] += 2
    result.loc[death_cross, "score"] -= 2

    # MACD momentum confirmation.
    result.loc[result["MACD"] > result["MACD_signal"], "score"] += 1
    result.loc[result["MACD"] < result["MACD_signal"], "score"] -= 1

    # ADX(14) as a STRENGTH amplifier: it sets no direction of its own, but when
    # the trend is strong (ADX >= 25) it reinforces whichever way the score
    # already leans, and a very strong trend (ADX >= 40) reinforces it more. In a
    # weak / range-bound market (ADX < 20) the lean is halved toward neutral,
    # because trend signals are unreliable there. ADX needs High/Low; on a
    # close-only frame it is NaN and these masks are simply never triggered.
    adx = result.get("ADX14")
    if adx is not None:
        lean = result["score"].clip(lower=-1, upper=1)   # direction the move points
        strong = adx >= 25
        very_strong = adx >= 40
        weak = adx < 20
        result.loc[strong, "score"] += lean[strong]
        result.loc[very_strong, "score"] += lean[very_strong]
        # Dampen conviction when the market is range-bound.
        result.loc[weak, "score"] = (result.loc[weak, "score"] / 2)

    result["score"] = result["score"].round().astype(int)

    result["raw_signal"] = "Hold"
    result.loc[result["score"] >= 3, "raw_signal"] = "Buy"
    result.loc[result["score"] <= -3, "raw_signal"] = "Sell"

    # Convert signals into position:
    # Buy means hold the stock.
    # Sell means exit the stock.
    # Hold means keep the previous position.
    # Use float NaN (not pd.NA) so the column stays numeric and ffill/fillna
    # do not trigger a pandas object-downcasting deprecation warning.
    result["target_position"] = float("nan")
    result.loc[result["raw_signal"] == "Buy", "target_position"] = 1.0
    result.loc[result["raw_signal"] == "Sell", "target_position"] = 0.0

    result["target_position"] = result["target_position"].ffill().fillna(0.0).astype(int)

    return result


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown as a percentage.
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    return float(drawdown.min() * 100)


def calculate_trade_returns(backtest_data: pd.DataFrame) -> List[float]:
    """
    Calculate percentage return for each completed trade.
    A trade starts when target_position changes from 0 to 1.
    A trade ends when target_position changes from 1 to 0.
    """
    trade_returns: List[float] = []

    position = backtest_data["target_position"]
    close = backtest_data["Close"]

    entries = backtest_data.index[(position.shift(1).fillna(0) == 0) & (position == 1)]
    exits = backtest_data.index[(position.shift(1).fillna(0) == 1) & (position == 0)]

    for entry_date in entries:
        possible_exits = exits[exits > entry_date]

        if len(possible_exits) > 0:
            exit_date = possible_exits[0]
        else:
            exit_date = backtest_data.index[-1]

        entry_price = float(close.loc[entry_date])
        exit_price = float(close.loc[exit_date])

        if entry_price > 0:
            trade_return = ((exit_price / entry_price) - 1) * 100
            trade_returns.append(trade_return)

    return trade_returns


def run_backtest(
    data: pd.DataFrame,
    ticker: str = "UNKNOWN",
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    """
    Run backtest for the rule-based advisor strategy.
    """
    signal_data = calculate_backtest_signals(data)

    backtest_data = signal_data.dropna(subset=["SMA200"]).copy()

    if len(backtest_data) < 2:
        raise ValueError("Not enough data for backtesting. At least 200 trading days are required.")

    backtest_data["market_return"] = backtest_data["Close"].pct_change().fillna(0)

    # Use yesterday's position for today's return to avoid look-ahead bias.
    backtest_data["strategy_return"] = (
        backtest_data["target_position"].shift(1).fillna(0) * backtest_data["market_return"]
    )

    backtest_data["strategy_equity"] = initial_cash * (1 + backtest_data["strategy_return"]).cumprod()
    backtest_data["buy_hold_equity"] = initial_cash * (1 + backtest_data["market_return"]).cumprod()

    final_strategy_value = float(backtest_data["strategy_equity"].iloc[-1])
    final_buy_hold_value = float(backtest_data["buy_hold_equity"].iloc[-1])

    strategy_return_percent = ((final_strategy_value / initial_cash) - 1) * 100
    buy_hold_return_percent = ((final_buy_hold_value / initial_cash) - 1) * 100

    max_drawdown_percent = calculate_max_drawdown(backtest_data["strategy_equity"])

    trade_returns_percent = calculate_trade_returns(backtest_data)
    number_of_trades = len(trade_returns_percent)

    if number_of_trades > 0:
        winning_trades = [trade for trade in trade_returns_percent if trade > 0]
        win_rate_percent = (len(winning_trades) / number_of_trades) * 100
    else:
        win_rate_percent = 0.0

    start_date = str(backtest_data.index[0].date()) if hasattr(backtest_data.index[0], "date") else str(backtest_data.index[0])
    end_date = str(backtest_data.index[-1].date()) if hasattr(backtest_data.index[-1], "date") else str(backtest_data.index[-1])

    return BacktestResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        final_strategy_value=final_strategy_value,
        final_buy_hold_value=final_buy_hold_value,
        strategy_return_percent=float(strategy_return_percent),
        buy_hold_return_percent=float(buy_hold_return_percent),
        max_drawdown_percent=float(max_drawdown_percent),
        number_of_trades=number_of_trades,
        win_rate_percent=float(win_rate_percent),
        trade_returns_percent=trade_returns_percent,
        data=backtest_data,
    )