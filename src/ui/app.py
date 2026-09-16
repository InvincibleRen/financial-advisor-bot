"""Financial Advisor Bot — Streamlit web interface (Phase 4).

Run from the project root:

    streamlit run src/ui/app.py

A sidebar switches between four views (single-stock analysis, top-N
recommendations, news sentiment, backtest & evaluation). All heavy/logic code
lives in ``data_access.py`` (Streamlit-free, unit-tested); this file only wires
navigation and page config.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file directly, so ensure the project root is importable.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402  (import after sys.path fix)

from src.ui import views  # noqa: E402

_PAGES = {
    "Single-stock analysis": views.render_single_stock,
    "Top-N recommendations": views.render_recommendations,
    "News sentiment (Ext A)": views.render_sentiment,
    "Backtest & evaluation": views.render_backtest,
}


def main() -> None:
    st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈", layout="wide")
    st.sidebar.title("Financial Advisor Bot")
    st.sidebar.caption("Explainable ML stock advisor — CM3070 FYP")
    choice = st.sidebar.radio("View", list(_PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info("Educational / decision-support only. Not financial advice.")
    _PAGES[choice]()


if __name__ == "__main__":
    main()
