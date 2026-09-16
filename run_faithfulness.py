"""Explanation-faithfulness audit (Layer 5 evidence, no participants needed).

The project's central claim is that every sentence the advisor emits is grounded
in a computed condition and that the system "cannot fabricate a justification".
That claim has never been *measured*. This script measures it.

Method
------
Generate recommendations across the universe at many historical cut-offs, then
for every emitted sentence independently re-derive the claim it makes from the
indicator snapshot and check it holds. The check is deliberately written against
the *indicator values*, not against the advisor's control flow, so it is a real
audit rather than a restatement of the code that produced the sentence.

Two numbers come out:
  * **coverage**  - share of emitted sentences that assert a checkable claim
                    (an unmatched sentence is itself a finding: it means the
                    advisor said something not tied to a verifiable condition)
  * **accuracy**  - share of checkable claims that are true of the data
"""
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass

import pandas as pd

import src.selection.universe as U
from src.common.indicators import latest_indicator_snapshot
from src.common.timeindex import to_naive_index
from src.explanation.advisor import _adx_strength, generate_recommendation

TOL = 0.55  # tolerance for values the sentences round to whole/2-dp numbers


@dataclass
class Quote:
    """Mirror of ``common.data.yfinance_provider.Quote``.

    Redeclared here so the audit runs entirely off the cached CSVs and never
    imports the network-bound provider module.
    """

    ticker: str
    latest_price: float
    previous_close: object
    change: object
    change_percent: object
    timestamp: str
    source: str = "cache"


def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def audit_sentence(s: str, ind: dict):
    """Return (matched, ok, label). ``matched=False`` -> no checkable claim found."""
    close, s20, s50, s200 = (ind.get(k) for k in ("latest_close", "sma20", "sma50", "sma200"))

    for win, sma in (("20", s20), ("50", s50), ("200", s200)):
        if f"{win}-day moving average" in s and "SMA" not in s:
            if close is None or sma is None:
                return True, False, f"sma{win}_missing_value"
            if "above" in s:
                return True, close > sma, f"sma{win}_above"
            if "below" in s:
                return True, close < sma, f"sma{win}_below"

    if "Golden Cross" in s:
        return True, str(ind.get("cross_event", "")).lower().startswith("golden"), "golden_cross"
    if "Death Cross" in s:
        return True, str(ind.get("cross_event", "")).lower().startswith("death"), "death_cross"

    if "SMA50/SMA200 gap" in s:
        stated, actual = _num(r"gap is about ([-\d.]+)%", s), ind.get("sma_gap_percent")
        if stated is None or actual is None:
            return True, False, "gap_missing_value"
        return True, abs(abs(actual) - abs(stated)) <= 0.05, "sma_gap_value"

    if "MACD is above its signal line" in s:
        ml, ms = ind.get("macd_line"), ind.get("macd_signal")
        ok = ml is not None and ms is not None and ml > ms
        if "positive histogram" in s:
            ok = ok and (ind.get("macd_hist") or 0) > 0
        return True, bool(ok), "macd_above"
    if "MACD is below its signal line" in s:
        ml, ms = ind.get("macd_line"), ind.get("macd_signal")
        ok = ml is not None and ms is not None and ml < ms
        if "negative histogram" in s:
            ok = ok and (ind.get("macd_hist") or 0) < 0
        return True, bool(ok), "macd_below"

    if s.startswith("ADX is"):
        stated, adx = _num(r"about (\d+(?:\.\d+)?)", s), ind.get("adx")
        if stated is None or adx is None:
            return True, False, "adx_missing_value"
        value_ok = abs(adx - stated) <= TOL
        strength = _adx_strength(adx)
        if "high at about" in s:
            band_ok = strength in ("strong", "very strong") and f"({strength} trend)" in s
        elif "low at about" in s:
            band_ok = strength == "weak"
        else:
            band_ok = strength == "moderate"
        return True, bool(value_ok and band_ok), "adx_band_and_value"

    if "intraday movement" in s:
        stated, actual = _num(r"approximately ([-\d.]+)%", s), ind.get("intraday_return_percent")
        if stated is None or actual is None:
            return True, False, "intraday_missing_value"
        sign_ok = (actual > 0) == ("positive" in s)
        return True, bool(sign_ok and abs(actual - stated) <= 0.05), "intraday_value"

    return False, False, "UNMATCHED"


def main() -> None:
    universe = U.DEFAULT_UNIVERSE
    prices = U.load_prices(universe, start="2015-01-01", end=None)

    sentences = n_rec = 0
    matched = ok = 0
    by_label = Counter()
    violations = []
    unmatched = Counter()

    for ticker, df in prices.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = to_naive_index(df.index)
        df = df.sort_index()
        idx = pd.DatetimeIndex(df.index)
        month_ends = (pd.Series(idx, index=idx)
                      .groupby([idx.year, idx.month]).max().tolist()[-24:])
        for cutoff in month_ends:
            hist = df.loc[:cutoff]
            if len(hist) < 220:
                continue
            try:
                ind = latest_indicator_snapshot(hist)
            except Exception:
                continue
            close = ind.get("latest_close")
            if close is None:
                continue
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
            quote = Quote(ticker=ticker, latest_price=float(close), previous_close=prev,
                          change=float(close - prev),
                          change_percent=float((close / prev - 1) * 100) if prev else None,
                          timestamp=str(cutoff))
            rec = generate_recommendation(quote, ind)
            n_rec += 1
            for s in rec.reasons:
                sentences += 1
                m, good, label = audit_sentence(s, ind)
                if not m:
                    unmatched[s[:70]] += 1
                    continue
                matched += 1
                by_label[label] += 1
                if good:
                    ok += 1
                elif len(violations) < 25:
                    violations.append({"ticker": ticker, "date": str(cutoff)[:10],
                                       "label": label, "sentence": s})

    out = {
        "n_recommendations": n_rec,
        "n_sentences": sentences,
        "n_checkable": matched,
        "coverage": matched / sentences if sentences else 0.0,
        "n_correct": ok,
        "accuracy": ok / matched if matched else 0.0,
        "claims_by_type": dict(by_label),
        "violations": violations,
        "unmatched_examples": dict(unmatched.most_common(10)),
    }
    with open("faithfulness_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "violations"}, indent=2, default=float))
    print(f"\nviolations recorded: {len(violations)}")
    for v in violations[:10]:
        print(" -", v)


if __name__ == "__main__":
    sys.exit(main())
