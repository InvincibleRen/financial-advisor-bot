# Financial Advisor Bot

An explainable stock-market advisor bot for non-technical retail investors
(CM3070 Project, CM3020 Artificial Intelligence template). It collects market
data, computes technical indicators (SMA and MACD, with ADX used as a
trend-strength amplifier that scales how strong the bullish/bearish signal is),
produces a categorised recommendation with a plain-language explanation, and
evaluates the strategy with both single-split and walk-forward backtests.

## Requirements

- Python 3.10 or newer
- An internet connection (live market data is fetched from Yahoo Finance via `yfinance`)

## Setup

Run these commands once, from the project folder (`financial-advisor-bot`):

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

For every later session, just re-activate the environment:

```bash
source venv/bin/activate           # Windows: venv\Scripts\activate
```

## Usage

All commands are run from the project root with the virtual environment active.

> **Note on module paths.** The code is organised by function (see *Project
> layout* below), so the command-line tools live under `src.cli.*`. Run them as
> shown.

### 1. Get a live recommendation

Fetch the latest data for a ticker and print the signal, indicators and a
plain-language explanation:

```bash
python -m src.cli.cli AAPL
```

Use `--basic` for the scope-faithful prototype mode (SMA + crossover only, no
RSI/MACD):

```bash
python -m src.cli.cli AAPL --basic
```

### 2. Single-split backtest

Backtest the rule-based strategy against a buy-and-hold baseline for one ticker.
A markdown report is written to `reports/`.

```bash
python -m src.cli.backtest_cli AAPL --period 5y --cash 10000
```

| Option | Meaning | Default |
|---|---|---|
| `--period` | History to download (e.g. `2y`, `5y`, `10y`) | `5y` |
| `--cash` | Starting cash for the simulation | `10000` |

### 3. Multi-stock backtest

Run the single-split backtest across several tickers and produce a comparison
table:

```bash
python -m src.cli.multi_backtest_cli AAPL MSFT NVDA TSLA SPY --period 5y
```

### 4. Walk-forward backtest (recommended evaluation)

Evaluate the strategy on a sequence of unseen windows, reporting standard finance
metrics (cumulative & annualised return, Sharpe ratio, max drawdown, hit rate)
per fold and aggregated, net of transaction costs. A markdown report is written
to `reports/`.

```bash
# One ticker, default settings (train 252 days, test 63 days, 10 bps costs)
python -m src.cli.walkforward_cli AAPL

# Custom windows, longer history, several tickers
python -m src.cli.walkforward_cli AAPL SPY --period 8y --train 252 --test 63 --cost 0.001
```

| Option | Meaning | Default |
|---|---|---|
| `--period` | History to download | `8y` |
| `--train` | Training window, in trading days | `252` (~1 year) |
| `--test` | Test window, in trading days | `63` (~1 quarter) |
| `--step` | Days to advance between folds | test size (contiguous) |
| `--cost` | Transaction cost per side, as a fraction (`0.001` = 10 bps) | `0.001` |
| `--rf` | Annual risk-free rate used for the Sharpe ratio | `0.0` |
| `--no-save` | Print results only; do not write a report | off |

### 5. Cross-sectional ML stock selection (the Direction-1 core)

Run the full ML core end to end: load the universe, build the leakage-safe
feature matrix and forward-return labels, then walk-forward select the top-N each
month and evaluate — **net of transaction costs** — against an equal-weight
universe benchmark and SPY. Both the gradient-boosting ranker and the logistic
baseline are reported. The report also includes **prediction-quality metrics**
(precision@N, mean AUC vs the ~0.5 base rate) that judge the ranker itself, and a
**per-year consistency breakdown** (strategy vs benchmark, with a monthly
win-rate) so you can see whether an edge is steady or driven by a few months. A
markdown report is written to `reports/`.

```bash
# Default universe (built-in ~27 large-caps), top-3, both rankers
python -m src.cli.select_cli

# Custom universe / settings
python -m src.cli.select_cli AAPL MSFT NVDA AMZN GOOGL META --top-n 5 --start 2015-01-01
```

| Option | Meaning | Default |
|---|---|---|
| `tickers` | Universe to select from (positional) | built-in `DEFAULT_UNIVERSE` |
| `--start` / `--end` | History date range | `2015-01-01` / today |
| `--top-n` | Stocks held each rebalance | `3` |
| `--model` | `gbm`, `logistic`, or `both` | `both` |
| `--cost` | Per-turnover transaction cost (`0.001` = 10 bps; `0` for gross) | `0.001` |
| `--min-train` | Prior rebalances required before a fold is scored | `6` |
| `--normalize` | Leakage-safe cross-sectional feature standardisation per rebalance date: `rank`, `zscore`, or `none` | `rank` |
| `--no-save` | Print only; do not write a report | off |

> **Cross-sectional normalisation (`--normalize`).** Selection is a *relative*
> problem, so each feature is standardised **within each rebalance date's own
> cross-section** (default `rank` = within-date percentile). This only ever looks
> at one date's cross-section, so it is leakage-safe by construction, and it lifts
> the ranker's out-of-sample AUC/precision (gbm mean AUC 0.511 → 0.524). Pass
> `--normalize none` for the older raw-level behaviour.

> **Selection pipeline internals.** `universe.py`, `features.py` and `labels.py`
> are library modules (no CLI of their own): they build the cached
> price/fundamentals panel, the leakage-safe feature matrix and the
> forward-return labels that the ranker (`model.py`) consumes via the
> walk-forward selector (`select.py`). All are covered by offline tests.

### 5b. Significance of the ranker (permutation + bootstrap)

Test whether the selector's out-of-sample skill is real rather than luck. Holding
the real scores fixed, the outcome labels are shuffled within each rebalance
date's cross-section (100× by default) to build a "no-skill" null distribution and
a permutation p-value for mean AUC and precision@N; per-fold AUCs are bootstrapped
for a 95% confidence interval. Leakage-free (labels only are permuted; no
re-training). A markdown report is written to `reports/`.

```bash
python -m src.cli.significance_cli                        # both models, 100 shuffles
python -m src.cli.significance_cli --model gbm --permutations 200
```

### 5c. Robustness / self-honesty (Deflated Sharpe + sensitivity + bootstrap)

Check that the selector's edge is not (a) a lucky best-of-many-trials, (b) a
cherry-picked setting, or (c) carried by a few months. Reuses one walk-forward
pass to report the **Deflated Sharpe Ratio** (discounts for the number of configs
tried), a **top-N × cost sensitivity grid**, and **bootstrap 95% CIs** on
annualised return / Sharpe. A markdown report is written to `reports/`.

```bash
python -m src.cli.robustness_cli                         # gbm, deflate for 20 trials
python -m src.cli.robustness_cli --model logistic --n-trials 30
```

### 6. News sentiment (Extension A — optional)

FinBERT scores recent headlines into a sentiment feature that joins the core
feature matrix; its value is proved by an **ablation** (run the selector with and
without the sentiment column). This extension is removable — deleting
`src/sentiment/` leaves the core selector fully working.

Extension A has heavy dependencies kept out of the core install. Pick the install
that matches your machine:

```bash
# Apple Silicon (arm64) / Linux / Windows — default torch backend
pip install -r requirements-extension-a.txt

# x86 macOS / Intel Mac — TORCH-FREE backend (recommended there)
pip install -r requirements-extension-a-onnx-local.txt
```

The scorer backend is selectable, so the same code runs on any machine:

| Backend | Flag | Runs on | Notes |
|---|---|---|---|
| `torch` (default) | `--backend torch` | Apple Silicon / Linux / Windows | Normal `pip install torch`; safetensors avoids the `torch.load` CVE gate. |
| `onnx-local` | `--backend onnx-local` | **x86 macOS / Intel Mac**, + everywhere | Torch-free: loads a pre-exported `model.onnx` on ONNX Runtime + `tokenizers`. |
| `onnx` (optimum) | `--backend onnx` | Where torch ≥ 2.4 exists | Exports via `optimum` (needs torch to trace); not for Intel Mac. |

> **macOS note:** PyTorch's last x86-macOS (Intel Mac) build is 2.2.2 and the
> modern ML stack has moved past it, so neither the `torch` nor the optimum
> `onnx` backend installs on an Intel Mac. Use **`--backend onnx-local`** there —
> it never imports torch. On an **Apple Silicon** Mac, instead create this venv
> from an **arm64** Python (e.g. `/opt/homebrew/bin/python3`), *not* an
> x86/Rosetta one, and the default `torch` backend works with Metal (MPS).

Live "sentiment right now" demo (yfinance exposes only recent news, so this is a
demonstration, not a historical backtest):

```bash
python -m src.cli.sentiment_cli                       # full DEFAULT_UNIVERSE
python -m src.cli.sentiment_cli AAPL MSFT NVDA        # specific tickers
python -m src.cli.sentiment_cli --lookback-days 30    # wider window = more headlines

# x86 macOS / Intel Mac — add the torch-free backend flag:
python -m src.cli.sentiment_cli AAPL MSFT NVDA --backend onnx-local
```

Prints a sentiment score per ticker plus a "market mood" average across all
tickers that had news. yfinance news is thin and recent-only, so a longer
`--lookback-days` (e.g. 30) usually gives each score more headlines to average.

To use sentiment inside the selector, pass the sentiment frame into the feature
matrix and run the ablation:

```python
from src.sentiment.finbert import FinBERTScorer
from src.sentiment.news_source import YFinanceNewsSource
from src.selection.features import build_feature_matrix
from src.selection.select import run_ablation

sent = FinBERTScorer().build_sentiment_feature(YFinanceNewsSource(), tickers, rebalance_dates)
fm = build_feature_matrix(prices, fundamentals, rebalance_dates, sentiment=sent)
results = run_ablation(fm, labels, prices, rebalance_dates)   # {"with_sentiment", "without_sentiment"}
```

### 7. Per-stock direction forecast (Extension B — optional)

A short-term **up/down confirmation** signal for the stocks the core selected,
run *only on the top-N*. It is framework-light (the same scikit-learn stack as
the core, no torch/TF — so it runs on any machine including Intel Macs) and
evaluated honestly: out-of-sample directional accuracy against the naive base
rate (always predict the majority direction), plus AUC. This extension is
removable — deleting `src/direction/` leaves the core selector fully working.

Feed it the selector's top-N picks:

```bash
python -m src.cli.direction_cli AAPL MSFT NVDA          # confirm the core's picks
python -m src.cli.direction_cli AAPL --model logistic --horizon 10 --period 8y
python -m src.cli.direction_cli AAPL --model baseline   # momentum-persistence floor
python -m src.cli.direction_cli AAPL --no-calibrate     # raw (uncalibrated) P(up)
```

For each ticker it prints out-of-sample accuracy, the base rate, the edge
(accuracy − base rate), AUC, **Brier score, calibration error (ECE)**, and the
**current** P(up) over the next `--horizon` days, and writes a markdown report to
`reports/`. Single-stock direction is close to a coin flip, so a small or negative
edge is an honest, acceptable outcome.

> **Probability calibration (`--calibrate`, on by default).** The displayed P(up)
> is calibrated so it reads as a true frequency (a "63%" day really rises ~63% of
> the time). Calibration is fit only on held-out CV folds of each training window
> (leakage-safe) and cuts calibration error sharply (AAPL: ECE 0.17 → 0.06) without
> costing discrimination. Use `--no-calibrate` for the raw score.

### 8. Web interface (Streamlit)

A four-view web app that ties the whole project together: single-stock analysis
(chart + indicators + rule-based recommendation + direction confirmation), the
top-N recommendations from the core selector, the FinBERT news-sentiment panel,
and a backtest/evaluation view. It reuses the core scikit-learn stack (no torch),
so it runs on any machine including Intel Macs.

```bash
pip install -r requirements.txt -r requirements-ui.txt
pip install -r requirements-extension-a-onnx-local.txt
streamlit run src/ui/app.py
```

Data is fetched on demand per ticker and cached within the session, so re-runs are
instant. See `src/ui/README.md` for the view-by-view breakdown.

### 9. Run the tests

```bash
# Whole suite (131 tests)
python -m pytest

# Just the ML-core selection pipeline (Phases 0–2)
python -m pytest tests/test_universe.py tests/test_features.py tests/test_labels.py \
                 tests/test_model.py tests/test_select.py tests/test_select_cli.py

# Extension A (sentiment), fully offline (no model download)
python -m pytest tests/test_sentiment.py tests/test_sentiment_onnx.py

# Extension B (direction forecast), fully offline
python -m pytest tests/test_direction.py

# Web UI logic layer, fully offline (no Streamlit needed)
python -m pytest tests/test_ui.py
```

## Project layout

The code is organised **by function**, matching the project's core-plus-extensions
design (main function ~70%, two removable extensions ~15% each). Each package has
its own `README.md`.

```
src/
  common/                      # shared infrastructure (used by core + all extensions)
    data/yfinance_provider.py  #   market data (quotes, intraday, historical)
    indicators.py              #   SMA, MACD, ADX (trend strength), crossover detection
    metrics.py                 #   finance metrics (Sharpe, drawdown, hit rate, costs)
    walkforward.py             #   model-agnostic walk-forward evaluation engine
    backtester.py              #   single-split rule-based backtest vs buy-and-hold
  selection/                   # ★ CORE — Direction 1: cross-sectional ML stock selection (~70%)
    universe.py                #   fixed universe + multi-ticker/fundamentals loader   [done: Phase 0]
    features.py                #   leakage-safe feature matrix                          [done: Phase 1]
    labels.py                  #   forward-return labels vs universe median            [done: Phase 1]
    model.py                   #   ranker (gradient boosting) + logistic baseline       [done: Phase 2]
    select.py                  #   top-N selection + walk-forward eval + ablation       [done: Phase 2]
  sentiment/                   # Extension A — Direction 3: FinBERT news sentiment (~15%)
    news_source.py             #   leakage-safe headline feed (in-memory + yfinance)    [done: Ext A]
    finbert.py                 #   FinBERT scoring -> per-(date,ticker) sentiment feature [done: Ext A]
  direction/                   # Extension B — Direction 2: per-stock direction forecast (~15%)
    forecaster.py              #   DirectionForecaster (gbm/logistic/baseline) + features/labels [done: Ext B]
    evaluate.py                #   walk-forward directional evaluation (accuracy vs base rate)    [done: Ext B]
  explanation/advisor.py       # rule-based signal + plain-language explanation
  ui/                          # Streamlit web interface (Phase 4)
    data_access.py             #   streamlit-free logic layer (provider-injectable)        [done: Phase 4]
    views.py                   #   four cached Streamlit views                             [done: Phase 4]
    app.py                     #   sidebar nav + page config (streamlit run src/ui/app.py) [done: Phase 4]
  cli/                         # command-line entry points (cli, backtest, multi, walkforward, select, sentiment, direction)
tests/                         # pytest suite
reports/                       # generated backtest reports
```

Build order and full design: see `../../Report Folder/FYP_Project_Design_and_Build_Plan.md`.
Modules marked `[scaffold]` have fixed signatures and phase-tagged `TODO`s. The
ML core (Phases 0–2) is complete: universe loader, feature matrix, labels, the
ranker + baseline, and the walk-forward top-N selector. **Extension A**
(`sentiment/`, FinBERT news sentiment) and **Extension B** (`direction/`,
per-stock direction forecast) are both implemented and offline-tested, and the
**Streamlit `ui/`** (Phase 4) now ties everything together in a four-view web app.
All planned components are in place.

## Disclaimer

This software is for educational and decision-support purposes only. It is not
financial advice and must not be used as the sole basis for trading decisions.

## Project direction

The system is built as a **main function plus two removable extensions**:

1. **Core (~70%) — cross-sectional stock selection.** From a universe of stocks
   with many indicators (PE, sector, earnings, momentum…), rank and select the
   few most likely to be profitable, then tune the model.
2. **Extension B (~15%) — per-stock direction forecast.** Time-series model that
   feeds financial factors for one asset and predicts up/down, run only on the
   selected top-N as a confirmation signal.
3. **Extension A (~15%) — NLP news/sentiment.** FinBERT scores market news and
   sentiment as an added feature, proved via an ablation study.

Full definition, weighting and build sequence:
`../../Report Folder/FYP_Project_Design_and_Build_Plan.md`.
