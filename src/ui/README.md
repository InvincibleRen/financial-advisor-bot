# `ui/` — Streamlit web interface (Phase 4)

The user-facing front end. It reuses the existing Python modules (rule-based
advisor, cross-sectional selector, FinBERT sentiment, direction forecaster)
behind a four-view Streamlit app.

| File | Responsibility | Status |
|---|---|---|
| `data_access.py` | **Streamlit-free** logic layer: provider-injectable functions (single-stock analysis, ticker evaluation, sentiment scoring, universe selection). Unit-tested offline. | done |
| `views.py` | Thin Streamlit render functions + `st.cache_data` wrappers (on-demand + memoised data). | done |
| `app.py` | Page config + sidebar navigation; wires the four views. | done |

## Views

1. **Single-stock analysis** — price + SMA chart, MACD/ADX/cross indicators, the
   rule-based recommendation with plain-language reasons, and the Extension-B
   direction confirmation. A **history-range** selector (1 day → 5 years) sets the
   chart interval, and the **direction-horizon** options adapt to the chosen range
   (`SINGLE_STOCK_PRESETS`). Long-window signals gracefully degrade on short ranges.
2. **Top-N recommendations** — designed for non-professionals: **pick stocks from
   the built-in large-cap universe** and/or **add your own list**, then the
   Direction-1 selector ranks them walk-forward and shows the shortlist + out-of-
   sample performance and ranking quality. Gated behind a **Run selector** button.
3. **News sentiment (Ext A)** — FinBERT signed sentiment via the torch-free
   `onnx-local` backend. The aggregate score comes with a **plain-English
   interpretation** and a scale legend, each **headline links to its article**, and
   the user chooses **how many headlines to show**.
4. **Backtest & evaluation** — walk-forward rule-based backtest vs buy-and-hold plus
   the Extension-B directional evaluation. History range 1 month → 5 years
   (`BACKTEST_PRESETS`); the ticker box offers **recent searches + suggestions**, and
   too-short ranges get a clear hint instead of an error.

## Run

```bash
pip install -r requirements.txt -r requirements-ui.txt
# for the sentiment view, also: pip install -r requirements-extension-a-onnx-local.txt
streamlit run src/ui/app.py
```

## Design notes

- **Logic vs presentation split (Rule 1).** All application logic lives in
  `data_access.py`, which imports no Streamlit and takes an injectable data
  provider — so it is unit-tested headlessly (`tests/test_ui.py`) with a fake
  provider, no network, no Streamlit, no model download. `views.py`/`app.py` only
  add widgets and caching.
- **On-demand + cached data.** Network fetches are wrapped in `st.cache_data`, so
  re-running a view within a session is instant.
- Portable on x86 macOS / Intel Mac: Streamlit ships x86-macOS wheels and the app
  reuses the core scikit-learn stack (no torch/TF).
