import pandas as pd

from src.explanation.advisor import _crossover_strength, generate_recommendation
from src.common.data.yfinance_provider import Quote


def _quote(change_percent=0.0):
    return Quote(
        ticker="TEST",
        latest_price=100.0,
        previous_close=100.0,
        change=0.0,
        change_percent=change_percent,
        timestamp="2026-06-18T15:55:00-04:00",
    )


def _base_indicators(**overrides):
    indicators = {
        "close_above_sma20": True,
        "close_above_sma50": True,
        "close_above_sma200": True,
        "cross_event": "none",
        "sma_gap_percent": 3.0,
        "macd_line": 1.0,
        "macd_signal": 0.5,
        "macd_hist": 0.5,
        "macd_bullish": True,
        # ADX defaults: a moderate trend that neither amplifies nor flags risk.
        "adx": 22.0,
        "plus_di": 25.0,
        "minus_di": 15.0,
        "adx_trending": False,
        "adx_ranging": False,
        "intraday_return_percent": 0.2,
        "intraday_volatility_percent": 0.1,
    }
    indicators.update(overrides)
    return indicators


def test_crossover_strength_thresholds():
    assert _crossover_strength(6.0) == "strong"
    assert _crossover_strength(3.0) == "moderate"
    assert _crossover_strength(1.0) == "marginal"
    assert _crossover_strength(None) == "moderate"


def test_strong_golden_cross_scores_higher_than_marginal():
    strong = generate_recommendation(
        _quote(), _base_indicators(cross_event="golden_cross", sma_gap_percent=8.0)
    )
    marginal = generate_recommendation(
        _quote(), _base_indicators(cross_event="golden_cross", sma_gap_percent=0.5)
    )
    assert strong.score > marginal.score
    assert any("strong Golden Cross" in r for r in strong.reasons)
    assert any("marginal Golden Cross" in r for r in marginal.reasons)


def test_strong_adx_amplifies_bullish_signal():
    strong = generate_recommendation(
        _quote(), _base_indicators(adx=30.0, adx_trending=True, plus_di=30.0, minus_di=10.0)
    )
    moderate = generate_recommendation(_quote(), _base_indicators(adx=22.0))
    # A strong trend amplifies the (bullish) lean, so the score is higher.
    assert strong.score > moderate.score
    assert strong.trend_strength == "strong"
    assert any("amplifying the bullish signal" in r for r in strong.reasons)


def test_very_strong_adx_amplifies_more_than_strong():
    strong = generate_recommendation(
        _quote(), _base_indicators(adx=30.0, adx_trending=True)
    )
    very_strong = generate_recommendation(
        _quote(), _base_indicators(adx=45.0, adx_trending=True)
    )
    assert very_strong.score > strong.score
    assert very_strong.trend_strength == "very strong"


def test_weak_adx_flags_range_bound_risk():
    rec = generate_recommendation(
        _quote(), _base_indicators(adx=15.0, adx_ranging=True, adx_trending=False)
    )
    assert rec.trend_strength == "weak"
    assert any("range-bound" in w for w in rec.warnings)


def test_adx_direction_conflict_warns():
    # Bullish trend indicators but ADX's +DI/-DI point down -> flag the conflict.
    rec = generate_recommendation(
        _quote(), _base_indicators(adx=30.0, adx_trending=True, plus_di=10.0, minus_di=30.0)
    )
    assert any("disagrees" in w for w in rec.warnings)


def test_macd_bearish_adds_negative_reason():
    rec = generate_recommendation(
        _quote(),
        _base_indicators(macd_line=-1.0, macd_signal=-0.5, macd_hist=-0.5, macd_bullish=False),
    )
    assert any("bearish momentum" in r for r in rec.reasons)


def test_recommendation_has_disclaimer_and_signal():
    rec = generate_recommendation(_quote(), _base_indicators())
    assert rec.disclaimer
    assert rec.signal in {"Watch / Buy", "Hold / Watch", "Hold", "Sell / Avoid", "Watch with caution"}
