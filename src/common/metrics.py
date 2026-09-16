"""Standard finance performance metrics for strategy evaluation.

These functions turn a series of *daily strategy returns* (simple, not log) into
the headline metrics reported in the project's evaluation chapter: cumulative
return, annualised return, annualised volatility, the Sharpe ratio, the maximum
drawdown, and the hit rate. They also provide the transaction-cost model used by
the walk-forward engine.

All functions are pure and operate on pandas Series / numpy arrays, so they are
straightforward to unit-test on hand-computed examples.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

# US equity markets trade ~252 days a year; used to annualise daily figures.
TRADING_DAYS_PER_YEAR = 252


def equity_curve(daily_returns: pd.Series, initial_cash: float = 1.0) -> pd.Series:
    """Compound a series of simple daily returns into an equity curve."""
    daily_returns = daily_returns.fillna(0.0)
    return initial_cash * (1.0 + daily_returns).cumprod()


def cumulative_return(daily_returns: pd.Series) -> float:
    """Total compounded return over the whole period, as a fraction (0.10 = 10%)."""
    daily_returns = daily_returns.fillna(0.0)
    if daily_returns.empty:
        return 0.0
    return float((1.0 + daily_returns).prod() - 1.0)


def annualised_return(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric (CAGR-style) annualised return derived from daily returns."""
    daily_returns = daily_returns.fillna(0.0)
    n = len(daily_returns)
    if n == 0:
        return 0.0
    total_growth = float((1.0 + daily_returns).prod())
    if total_growth <= 0.0:
        # Total loss of capital: report -100% rather than raising on a fractional power.
        return -1.0
    return total_growth ** (periods_per_year / n) - 1.0


def annualised_volatility(daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised standard deviation of daily returns."""
    daily_returns = daily_returns.fillna(0.0)
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio.

    ``risk_free_rate`` is an *annual* rate; it is converted to a per-period rate
    before subtraction. Returns 0.0 when volatility is zero (a flat curve has no
    risk-adjusted signal to report).
    """
    daily_returns = daily_returns.fillna(0.0)
    if len(daily_returns) < 2:
        return 0.0
    per_period_rf = risk_free_rate / periods_per_year
    excess = daily_returns - per_period_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve, as a negative fraction."""
    daily_returns = daily_returns.fillna(0.0)
    if daily_returns.empty:
        return 0.0
    curve = (1.0 + daily_returns).cumprod()
    running_max = curve.cummax()
    drawdown = (curve / running_max) - 1.0
    return float(drawdown.min())


def hit_rate(daily_returns: pd.Series) -> float:
    """Fraction of *active* days (non-zero return) that were positive.

    Days with exactly zero return (typically flat / out-of-market days) are
    excluded so the hit rate reflects the quality of the days the strategy was
    actually taking risk.
    """
    daily_returns = daily_returns.fillna(0.0)
    active = daily_returns[daily_returns != 0.0]
    if active.empty:
        return 0.0
    return float((active > 0).mean())


def apply_transaction_costs(
    positions: pd.Series,
    market_returns: pd.Series,
    cost_per_turnover: float = 0.001,
) -> pd.Series:
    """Convert positions + market returns into net daily strategy returns.

    * ``positions`` is the target position for each day (1 = fully invested,
      0 = in cash). To avoid look-ahead bias the strategy earns each day's
      market return using the *previous* day's position.
    * ``cost_per_turnover`` is charged on the absolute change in position
      (turnover). A round-trip in and out of a position therefore costs roughly
      ``2 * cost_per_turnover``. The default 0.001 = 10 basis points per side.
    """
    positions = positions.reindex(market_returns.index).ffill().fillna(0.0)
    lagged = positions.shift(1).fillna(0.0)
    gross = lagged * market_returns.fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    costs = turnover * cost_per_turnover
    return gross - costs


def compute_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """Bundle the headline metrics for a return series into one dict."""
    return {
        "cumulative_return": cumulative_return(daily_returns),
        "annualised_return": annualised_return(daily_returns, periods_per_year),
        "annualised_volatility": annualised_volatility(daily_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(daily_returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(daily_returns),
        "hit_rate": hit_rate(daily_returns),
        "n_days": int(len(daily_returns)),
    }


def aggregate_fold_metrics(fold_metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Average per-fold metrics (a simple, unweighted mean across folds)."""
    folds = list(fold_metrics)
    if not folds:
        return {}
    keys = [k for k in folds[0] if k != "n_days"]
    averaged = {f"mean_{k}": float(np.mean([f[k] for f in folds])) for k in keys}
    averaged["n_folds"] = len(folds)
    return averaged
