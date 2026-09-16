"""Streamlit view layer for the Financial Advisor Bot (Phase 4).

Thin render functions over ``data_access`` (which holds all the testable logic).
Expensive, network-bound calls are wrapped in ``st.cache_data`` so re-runs within
a session are instant (the "on-demand + cache" data strategy).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import data_access as da

_DISCLAIMER = (
    "Educational / decision-support only. Not financial advice; do not use as the "
    "sole basis for trading decisions."
)
_HISTORY_KEY = "ticker_history"
_SUGGESTED = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "SPY"]


# --------------------------------------------------------------------------- #
# Cached wrappers (one place; keep the network on-demand + memoised)          #
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _single(ticker, period, interval, horizon, trim):
    return da.analyse_single_stock(ticker, period=period, interval=interval,
                                   horizon_days=horizon, trim_rows=trim)


@st.cache_data(show_spinner=False)
def _evaluate(ticker, period, interval, horizon, train, test, trim):
    return da.evaluate_ticker(ticker, period=period, interval=interval, horizon_days=horizon,
                              train=train, test=test, trim_rows=trim)


@st.cache_data(show_spinner=False)
def _sentiment(ticker, lookback, backend, max_headlines):
    return da.score_ticker_sentiment(ticker, lookback_days=lookback, backend=backend,
                                     max_headlines=max_headlines)


@st.cache_data(show_spinner=False)
def _selection(tickers, start, top_n, model_kind):
    return da.run_universe_selection(tickers=list(tickers) if tickers else None,
                                     start=start, top_n=top_n, model_kind=model_kind)


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x * 100:.2f}%"


def _record_ticker(ticker: str) -> None:
    """Remember a searched ticker (most-recent-first, capped) for later suggestions."""
    if not ticker:
        return
    hist = st.session_state.setdefault(_HISTORY_KEY, [])
    if ticker in hist:
        hist.remove(ticker)
    hist.insert(0, ticker)
    del hist[8:]


def _ticker_selector(key_prefix: str, default: str = "AAPL") -> str:
    """Ticker chooser: recent searches + common suggestions, plus a free-text box."""
    hist = st.session_state.get(_HISTORY_KEY, [])
    options = ["— type below —"] + list(dict.fromkeys(hist + _SUGGESTED))
    c1, c2 = st.columns(2)
    picked = c1.selectbox("Recent / suggested", options, key=f"{key_prefix}_pick")
    typed_default = "" if picked != "— type below —" else default
    typed = c2.text_input("…or enter a ticker", value=typed_default, key=f"{key_prefix}_text")
    chosen = typed.strip() or (picked if picked != "— type below —" else "")
    return chosen.upper().strip()


# --------------------------------------------------------------------------- #
# View 1 — Single-stock analysis                                              #
# --------------------------------------------------------------------------- #

def render_single_stock() -> None:
    st.header("Single-stock analysis")
    st.caption("Price trend, technical indicators, the rule-based recommendation, "
               "and the Extension-B direction confirmation — for one ticker.")

    col1, col2, col3 = st.columns([2, 1, 1])
    ticker = col1.text_input("Ticker", value="AAPL").upper().strip()
    period_label = col2.selectbox("History range", list(da.SINGLE_STOCK_PRESETS.keys()), index=5)  # 1 year
    preset = da.SINGLE_STOCK_PRESETS[period_label]
    horizon = col3.selectbox("Direction horizon (bars)", preset["horizons"],
                             index=preset["horizons"].index(preset["default_horizon"]))
    if not ticker:
        return

    try:
        with st.spinner(f"Fetching {ticker}…"):
            a = _single(ticker, preset["period"], preset["interval"], int(horizon), preset["trim"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load {ticker} for {period_label}: {exc}")
        return
    _record_ticker(ticker)

    q = a.quote
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest price", f"${q.latest_price:,.2f}" if q.latest_price is not None else "—",
              f"{q.change_percent:+.2f}%" if q.change_percent is not None else None)
    rec = a.recommendation
    adx = a.indicators.get("adx")
    m2.metric("Trend strength (ADX)",
              f"{adx:.0f}" if adx is not None else "—",
              rec.trend_strength if adx is not None else None)
    m3.metric("Cross event", str(a.indicators.get("cross_event", "—")))
    m4.metric("Recommendation", rec.signal, f"risk: {rec.risk_level}")

    st.subheader("Price & moving averages")
    st.line_chart(a.price_chart)
    st.caption(f"Showing **{preset['interval']}** bars over **{period_label}**. Long-window "
               "signals (SMA200) and the direction model need a substantial daily history, so "
               "on short or intraday ranges they may be limited or unavailable.")

    left, right = st.columns(2)
    with left:
        st.subheader("Why (rule-based reasons)")
        for reason in rec.reasons:
            st.markdown(f"- {reason}")
        for warning in rec.warnings:
            st.warning(warning)
    with right:
        st.subheader("Direction confirmation (Ext B)")
        if a.direction is not None:
            d = a.direction
            st.metric(f"P(up) next {d.horizon_days} bars", f"{d.prob_up:.2f}", d.direction)
            st.caption("Single-stock direction is near coin-flip — treat as a weak "
                       "confirmation layer, not a standalone signal.")
        else:
            st.info(f"Direction unavailable for this range: {a.direction_error}")

    st.caption(_DISCLAIMER)


# --------------------------------------------------------------------------- #
# View 2 — Top-N recommendations (core selector, heavy)                       #
# --------------------------------------------------------------------------- #

def render_recommendations() -> None:
    st.header("Top-N recommendations")
    st.markdown(
        "Not sure which stocks to look at? Pick a few from the **built-in large-cap "
        "list** below (or add your own), and the app ranks them with its ML selector "
        "and shows the shortlist it would hold. The ranking is trained only on past "
        "data (walk-forward), so it reflects an out-of-sample process, not hindsight."
    )

    universe_all = da.default_universe()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Your stock list")
        picked = st.multiselect(
            "From the built-in universe",
            options=universe_all,
            default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
            help="Well-known large-cap US stocks — a safe starting point.",
        )
        st.caption("Add your own tickers below — one per row (click the ➕ row to add more):")
        custom_df = st.data_editor(
            pd.DataFrame({"Ticker": pd.Series([], dtype="str")}),
            num_rows="dynamic",
            hide_index=True,
            column_config={"Ticker": st.column_config.TextColumn("Ticker", help="e.g. ORCL")},
            key="rec_custom_list",
        )
        extra = [str(t).strip().upper() for t in custom_df["Ticker"].tolist() if str(t).strip()]
        universe = list(dict.fromkeys(picked + extra))
        st.caption(f"**{len(universe)} selected:** {', '.join(universe) if universe else '—'}")

    with right:
        st.subheader("Settings")
        top_n = st.number_input("How many to shortlist (top-N)", 1, 10, 3)
        model_kind = st.selectbox("Ranker", ["gbm", "logistic"], index=0)
        start = st.select_slider("History start", options=["2015-01-01", "2018-01-01", "2020-01-01"],
                                 value="2018-01-01")
        run = st.button("Run selector")

    if len(universe) < max(int(top_n) + 1, 3):
        st.warning("Pick at least a few stocks (more than the top-N) so there is something to rank.")
        return
    if not run:
        st.info("Build your list on the left, set options on the right, then click **Run selector**. "
                "First run downloads the data, so it takes a moment.")
        return

    try:
        with st.spinner("Loading prices/fundamentals and running walk-forward selection…"):
            sel = _selection(tuple(universe), start, int(top_n), model_kind)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Selector failed: {exc}")
        return

    st.success(f"Latest shortlist ({sel.latest_date.date() if sel.latest_date else '—'}): "
               f"**{', '.join(sel.latest_picks) if sel.latest_picks else '—'}**")

    r = sel.result
    st.subheader("Out-of-sample performance (net of costs)")
    perf = pd.DataFrame({
        f"Selector ({model_kind})": r.strategy_metrics,
        "Equal-weight universe": r.benchmark_metrics,
        "SPY (buy & hold)": r.spy_metrics or {},
    }).T[["cumulative_return", "annualised_return", "sharpe_ratio", "max_drawdown", "hit_rate"]]
    st.dataframe(perf.style.format({
        "cumulative_return": "{:.1%}", "annualised_return": "{:.1%}",
        "sharpe_ratio": "{:.2f}", "max_drawdown": "{:.1%}", "hit_rate": "{:.1%}",
    }))

    rank = r.ranking_metrics or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@N", f"{rank.get('precision_at_n', float('nan')):.2f}")
    c2.metric("Mean AUC", f"{rank.get('mean_auc', float('nan')):.2f}",
              f"base {rank.get('base_rate', float('nan')):.2f}")
    c3.metric("Monthly win-rate", _pct(r.monthly_win_rate))
    st.caption(f"Universe: {len(sel.universe)} tickers · {sel.n_rebalances} monthly rebalances. "
               + _DISCLAIMER)


# --------------------------------------------------------------------------- #
# View 3 — News sentiment (Extension A)                                       #
# --------------------------------------------------------------------------- #

def render_sentiment() -> None:
    st.header("News sentiment (Extension A)")
    st.caption("FinBERT scores recent headlines into a signed sentiment in [-1, 1]. "
               "Uses the torch-free `onnx-local` backend; the first run downloads the model.")

    col1, col2, col3 = st.columns([2, 1, 1])
    ticker = col1.text_input("Ticker", value="AAPL", key="sent_ticker").upper().strip()
    lookback = col2.number_input("Look-back (days)", 1, 90, 30)
    max_headlines = col3.number_input("Headlines to show", 1, 50, 10)
    if not ticker:
        return
    if not st.button("Score headlines"):
        st.info("Click **Score headlines** to fetch and score recent news.")
        return

    with st.spinner("Fetching headlines and scoring with FinBERT (first run downloads the model)…"):
        res = _sentiment(ticker, int(lookback), "onnx-local", int(max_headlines))

    if res.error:
        st.error(res.error)
        return
    if not res.headlines:
        st.info(f"No headlines for {ticker} in the last {int(lookback)} days "
                "(yfinance news is thin — try a wider window).")
        return

    label, meaning = da.sentiment_interpretation(res.aggregate)
    st.metric(f"Aggregate sentiment ({ticker})", f"{res.aggregate:+.3f}", label)
    st.info(f"**{label}.** {meaning}  \nBased on {res.n_total} headline(s) in the last "
            f"{int(lookback)} days.")
    with st.expander("How to read this score"):
        st.markdown(
            "- **+0.50 to +1.00** — strongly positive\n"
            "- **+0.15 to +0.50** — positive\n"
            "- **−0.15 to +0.15** — neutral / mixed\n"
            "- **−0.50 to −0.15** — negative\n"
            "- **−1.00 to −0.50** — strongly negative\n\n"
            "Each headline is scored `P(positive) − P(negative)`; the aggregate is their mean. "
            "This measures the *tone* of news, not a price prediction."
        )

    st.subheader(f"Latest {len(res.headlines)} headline(s)")
    for dt, text, score, url in res.headlines:
        tone, _ = da.sentiment_interpretation(score)
        date = pd.Timestamp(dt).date()
        link = f"[{text}]({url})" if url else text
        st.markdown(f"- **{score:+.2f}** ({tone}) · {date} — {link}")
    st.caption("Score = mean(P(positive) − P(negative)) across headlines. " + _DISCLAIMER)


# --------------------------------------------------------------------------- #
# View 4 — Backtest & evaluation                                              #
# --------------------------------------------------------------------------- #

def render_backtest() -> None:
    st.header("Backtest & evaluation")
    st.caption("Walk-forward rule-based backtest (vs buy-and-hold) plus the "
               "Extension-B directional evaluation (accuracy vs the naive base rate).")

    ticker = _ticker_selector("bt")
    col1, col2, col3 = st.columns(3)
    period_label = col2.selectbox("History range", list(da.BACKTEST_PRESETS.keys()), index=3)  # 1 year
    preset = da.BACKTEST_PRESETS[period_label]
    train = col1.number_input("Train window (days)", 40, 756, 252)
    horizon = col3.number_input("Direction horizon (days)", 1, 30, 5)
    if not ticker:
        st.info("Choose a ticker above.")
        return
    if not st.button("Run evaluation"):
        st.info("Pick a ticker and range, then click **Run evaluation**.")
        return

    try:
        with st.spinner(f"Downloading {ticker} and running walk-forward evaluation…"):
            ev = _evaluate(ticker, preset["period"], preset["interval"], int(horizon),
                           int(train), 63, preset["trim"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Evaluation failed: {exc}")
        return
    _record_ticker(ticker)

    wf = ev.walkforward
    ov, ovb = wf.overall_strategy, wf.overall_benchmark
    st.subheader("Rule-based strategy vs buy-and-hold (out-of-sample)")
    a, b, c = st.columns(3)
    a.metric("Strategy return", _pct(ov["cumulative_return"]),
             f"vs {_pct(ovb['cumulative_return'])} B&H")
    b.metric("Sharpe", f"{ov['sharpe_ratio']:.2f}", f"B&H {ovb['sharpe_ratio']:.2f}")
    c.metric("Max drawdown", _pct(ov["max_drawdown"]), f"B&H {_pct(ovb['max_drawdown'])}")

    d = ev.direction
    st.subheader("Direction forecast evaluation (Ext B)")
    e, f, g = st.columns(3)
    e.metric("Accuracy", _pct(d.accuracy), f"base {_pct(d.base_rate)}")
    f.metric("Edge", f"{d.edge * 100:+.1f}pp")
    g.metric("AUC", "—" if d.auc is None else f"{d.auc:.3f}")
    st.caption(f"{d.n_predictions} out-of-sample predictions. Single-stock direction is "
               "near coin-flip; judge accuracy against the base rate, not 50%. " + _DISCLAIMER)
