from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.common.data.yfinance_provider import Quote


@dataclass
class Recommendation:
    signal: str
    risk_level: str
    score: int
    reasons: List[str]
    warnings: List[str]
    disclaimer: str
    trend_strength: str = "unknown"   # ADX-based strength of the prevailing move


def _adx_strength(adx) -> str:
    """Classify trend strength from ADX (direction-agnostic).

    ADX only measures how *strong* a move is, not its direction: >=40 is a very
    strong trend, >=25 strong, >=20 moderate/developing, and <20 a weak /
    range-bound market. The advisor uses this to scale how decisive the
    bullish/bearish signal from the other indicators should be.
    """
    if adx is None:
        return "unknown"
    if adx >= 40:
        return "very strong"
    if adx >= 25:
        return "strong"
    if adx >= 20:
        return "moderate"
    return "weak"


def _crossover_strength(sma_gap_percent) -> str:
    """Describe how decisive a crossover is from the SMA50/SMA200 gap size.

    A wide gap means a strong, decisive cross; a narrow gap means a weak,
    marginal one. This lets the explanation reflect signal *magnitude*, not
    only signal type.
    """
    if sma_gap_percent is None:
        return "moderate"
    gap = abs(sma_gap_percent)
    if gap >= 5.0:
        return "strong"
    if gap >= 2.0:
        return "moderate"
    return "marginal"


def generate_recommendation(quote: Quote, indicators: Dict) -> Recommendation:
    score = 0
    reasons: List[str] = []
    warnings: List[str] = []

    if indicators.get("close_above_sma20"):
        score += 1
        reasons.append("The latest price is above the 20-day moving average, suggesting short-term strength.")
    else:
        score -= 1
        reasons.append("The latest price is below the 20-day moving average, suggesting short-term weakness.")

    if indicators.get("close_above_sma50"):
        score += 1
        reasons.append("The latest price is above the 50-day moving average, supporting a positive medium-term trend.")
    else:
        score -= 1
        reasons.append("The latest price is below the 50-day moving average, suggesting medium-term caution.")

    if indicators.get("close_above_sma200"):
        score += 1
        reasons.append("The latest price is above the 200-day moving average, indicating a stronger long-term trend.")
    else:
        score -= 1
        reasons.append("The latest price is below the 200-day moving average, indicating long-term weakness.")

    cross_event = indicators.get("cross_event")
    sma_gap_percent = indicators.get("sma_gap_percent")
    strength = _crossover_strength(sma_gap_percent)
    gap_text = (
        f" The SMA50/SMA200 gap is about {sma_gap_percent:.2f}%, indicating a {strength} signal."
        if sma_gap_percent is not None
        else ""
    )

    if cross_event == "golden_cross":
        # Scale the contribution by how decisive the cross is.
        score += 3 if strength == "strong" else 2 if strength == "moderate" else 1
        if strength == "strong":
            reasons.append(
                "A strong Golden Cross was detected: SMA50 crossed decisively above SMA200." + gap_text
            )
        elif strength == "marginal":
            reasons.append(
                "A marginal Golden Cross was detected: SMA50 only just crossed above SMA200, "
                "so treat this bullish signal with some caution." + gap_text
            )
        else:
            reasons.append("A Golden Cross was detected, where SMA50 crossed above SMA200." + gap_text)
    elif cross_event == "death_cross":
        score -= 3 if strength == "strong" else 2 if strength == "moderate" else 1
        if strength == "strong":
            reasons.append(
                "A strong Death Cross was detected: SMA50 crossed decisively below SMA200." + gap_text
            )
        elif strength == "marginal":
            reasons.append(
                "A marginal Death Cross was detected: SMA50 only just crossed below SMA200, "
                "so this bearish signal is weak." + gap_text
            )
        else:
            reasons.append("A Death Cross was detected, where SMA50 crossed below SMA200." + gap_text)

    # Momentum confirmation via MACD(12, 26, 9).
    macd_line = indicators.get("macd_line")
    macd_hist = indicators.get("macd_hist")
    if macd_line is not None and macd_hist is not None:
        if indicators.get("macd_bullish"):
            score += 1
            reasons.append(
                "MACD is above its signal line with a positive histogram, confirming bullish momentum."
                if macd_hist > 0
                else "MACD is above its signal line, suggesting momentum is turning bullish."
            )
        else:
            score -= 1
            reasons.append(
                "MACD is below its signal line with a negative histogram, confirming bearish momentum."
                if macd_hist < 0
                else "MACD is below its signal line, suggesting momentum is turning bearish."
            )

    # Recent intraday movement — a small same-day nudge to the directional lean.
    intraday_return = indicators.get("intraday_return_percent")
    if intraday_return is not None:
        if intraday_return > 1:
            score += 1
            reasons.append(f"Recent intraday movement is positive at approximately {intraday_return:.2f}%.")
        elif intraday_return < -1:
            score -= 1
            reasons.append(f"Recent intraday movement is negative at approximately {intraday_return:.2f}%.")

    # ADX(14) as a STRENGTH amplifier. ADX sets no direction of its own; instead
    # it co-works with the directional indicators (SMA position, crossover, MACD,
    # intraday) to represent how strong the prevailing bullish/bearish move is. A
    # strong trend amplifies that lean, a very strong one amplifies it more, and a
    # weak / range-bound market dampens it (and raises risk, handled below).
    # +DI/-DI give an independent directional cross-check.
    adx = indicators.get("adx")
    plus_di = indicators.get("plus_di")
    minus_di = indicators.get("minus_di")
    trend_strength = _adx_strength(adx)
    if adx is not None:
        lean = 1 if score > 0 else -1 if score < 0 else 0
        if trend_strength in ("strong", "very strong") and lean != 0:
            boost = 2 if trend_strength == "very strong" else 1
            score += boost * lean
            move = "bullish" if lean > 0 else "bearish"
            reasons.append(
                f"ADX is high at about {adx:.0f} ({trend_strength} trend), amplifying the "
                f"{move} signal — the move has conviction behind it."
            )
            if plus_di is not None and minus_di is not None:
                di_dir = 1 if plus_di > minus_di else -1
                if di_dir != lean:
                    warnings.append(
                        "ADX confirms a strong trend, but its direction (+DI vs -DI) disagrees with "
                        "the other indicators, so the signal is less certain."
                    )
        elif trend_strength == "weak":
            reasons.append(
                f"ADX is low at about {adx:.0f} (weak / range-bound), so the move lacks conviction "
                "and the trend signals are less reliable here."
            )
        elif trend_strength in ("strong", "very strong"):
            # Strong trend, but the bullish and bearish indicators cancelled out
            # (``lean == 0``), so there is no direction to amplify. The score is
            # left alone — but the *description* must still report the reading we
            # actually took. Falling through to the "moderate" wording below would
            # state something false about the data (an ADX of 45 is not moderate).
            reasons.append(
                f"ADX is high at about {adx:.0f} ({trend_strength} trend), but the other "
                "indicators are balanced, so there is no clear direction for it to confirm."
            )
        else:
            reasons.append(
                f"ADX is moderate at about {adx:.0f}, a developing rather than a strong trend."
            )

    risk_score = 0

    if quote.change_percent is not None and abs(quote.change_percent) > 3:
        risk_score += 1
        warnings.append("The stock has moved more than 3% from the previous close, so short-term volatility may be high.")

    intraday_volatility = indicators.get("intraday_volatility_percent")

    if intraday_volatility is not None and intraday_volatility > 0.8:
        risk_score += 1
        warnings.append("Intraday volatility appears high, so the recommendation should be treated carefully.")

    if indicators.get("cross_event") == "not_enough_data":
        risk_score += 1
        warnings.append("There is not enough data to detect long-term moving-average crossover events reliably.")

    if indicators.get("adx_ranging"):
        risk_score += 1
        warnings.append("ADX indicates a weak / range-bound market, so trend-following signals carry extra risk.")

    if risk_score >= 2:
        risk_level = "High"
    elif risk_score == 1:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if score >= 3:
        signal = "Watch / Buy"
    elif score >= 1:
        signal = "Hold / Watch"
    elif score <= -3:
        signal = "Sell / Avoid"
    else:
        signal = "Hold"

    if risk_level == "High" and signal == "Watch / Buy":
        signal = "Watch with caution"

    disclaimer = (
        "This output is for educational and decision-support purposes only. "
        "It is not guaranteed financial advice and should not be used as the only basis for trading decisions."
    )

    return Recommendation(
        signal=signal,
        risk_level=risk_level,
        score=score,
        reasons=reasons,
        warnings=warnings,
        disclaimer=disclaimer,
        trend_strength=trend_strength,
    )