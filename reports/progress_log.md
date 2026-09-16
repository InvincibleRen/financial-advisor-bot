## Scientific-honesty pass: Deflated Sharpe + sensitivity grid + bootstrap CIs (2026-07-27)

Goal: maximise self-honesty — prove the core selector's edge is not (a) a lucky best-of-many-trials, (b) a cherry-picked setting, or (c) carried by a few months. New module `selection/robustness.py` + `cli/robustness_cli.py`, all formulas on the realised return series (no new model assumptions), reusing one walk-forward pass.

- **Deflated Sharpe Ratio** (Bailey & López de Prado). Probabilistic Sharpe measured against the *expected-max-Sharpe-of-N-trials* benchmark, discounting for skew/kurtosis and the number of configs tried. Result: observed per-period Sharpe 0.323 vs a best-of-20 luck benchmark of 0.080 → **DSR = 0.998**. Even after deflating for ~20 tried configurations, the Sharpe is almost certainly not a multiple-testing artefact.
- **Sensitivity grid** (top-N ∈ {3,5,10} × cost ∈ {0,10,20 bps}), computed from the *same* trained folds (re-weight + re-evaluate only, no retraining). Every one of the 9 cells holds: **Sharpe 1.06–1.46, CAGR 29–38%**. Honest by-product: N=10 (Sharpe ~1.45) is *better risk-adjusted* than N=3 (~1.12) — top-3's raw return comes from concentration.
- **Bootstrap CIs** on the top-3 / 10 bps monthly returns: annualised return median 35.5% (95% CI **[13.6%, 63.2%]**), annualised Sharpe median 1.12 (95% CI **[0.57, 1.73]**). The Sharpe lower bound stays positive — the edge is not one lucky stretch.

All three checks pass, which is the strongest self-honesty evidence in the project so far. Written for a non-specialist reader (each number has a plain-language "陷阱一/二/三" explanation) in `reports/robustness_GENERATED.md`; reproduce with `python -m src.cli.robustness_cli --model gbm --n-trials 20`.

Evidence: `tests/test_robustness.py` (5 offline tests — signal vs noise separation for DSR, PSR bounds, best-of-N benchmark grows with trials, bootstrap brackets the estimate) → **131 passed** total. Honest caveats recorded: the DSR trial count (20) is a conservative estimate, and the bootstrap Sharpe interval is wide (limited monthly sample), both stated in the report.

---

## Real-time step 1: probability calibration for the direction forecast (2026-07-27)

First step toward the real-time goal: the displayed **"P(up)" (涨跌概率) must be a trustworthy frequency**, not a raw uncalibrated score. Added leakage-safe probability calibration to Extension B and the metrics to prove it.

- **Calibration (`forecaster.build_fitted_estimator`).** Optional `calibrate` flag wraps the estimator in `CalibratedClassifierCV` (Platt/`sigmoid` default; `isotonic` available), fitting the calibration map on **held-out CV folds of each training window only** — leakage-safe. Threaded through `DirectionForecaster` and the walk-forward `evaluate_direction`. `direction_cli` gains `--calibrate` (**default on**, `--no-calibrate` to disable); the Streamlit single-stock view now shows a calibrated P(up).
- **Calibration metrics (`evaluate.py`).** Added `brier_score`, `expected_calibration_error` (ECE), and a `reliability_table` (binned mean-predicted vs realised-up-rate — the reliability-diagram diagonal). `DirectionEvaluation` now carries `brier`, `ece`, `calibrated`, `reliability`.

Measured before/after (AAPL, logistic, horizon 5, walk-forward n=545):

| Metric | Raw | Calibrated | Change |
|---|---:|---:|---|
| ECE (calibration error) | 0.1677 | **0.0559** | ↓ 67% |
| Brier score | 0.2848 | **0.2507** | ↓ 12% |
| Accuracy | 0.519 | 0.549 | +3.0pp |
| AUC | 0.483 | 0.528 | +0.045 |

(gbm shows the same direction: ECE 0.285 → 0.074 on a shorter walk-forward.) Calibration is a monotonic re-scaling, so it targets Brier/ECE without costing discrimination — here AUC even rose because the CV-ensemble reduced variance. This is exactly the right lever for a **displayed** probability.

Evidence: `tests/test_direction.py` +4 (Brier/ECE math, reliability binning, calibrated-forecaster valid prob, calibrated-eval metrics well-formed) → **126 passed** total. Report: `reports/calibration_AAPL_compare.md`. Run: `python -m src.cli.direction_cli AAPL --calibrate`.

Note on cost: CV calibration trains k estimators per fold, so it is ~3× slower — fine for the on-demand single-ticker live view, and acceptable in the walk-forward evaluation.

Next real-time steps (not yet done): live data provider beyond delayed yfinance (e.g. Alpaca websocket) behind the existing injectable provider interface; a self-refreshing real-time panel; and as-of discipline for intraday (use the last *completed* bar).

---

## Accuracy pass 2: feature pruning + leaf regularisation (2026-07-27)

Push from AUC 0.524 toward higher pick-quality, driven by a **permutation-importance diagnostic** (train gbm on a 70% time slice, measure the OOS AUC drop when each feature is shuffled) rather than guessing. Findings and what shipped:

- **The 6 fundamental features contribute ≈0 OOS AUC; `volatility` is *negative* (−0.0066)** — the tree overfits it. (This confirms the review's §3.4 worry that the sparse fundamental panel adds little.)
- **`volatility` excluded from the gbm ranker only.** Dropping it lifts gbm; but it *helps* the logistic baseline (a linear risk control), so the exclusion is model-specific — applied inside `RankerModel.fit`/`predict` for `kind="gbm"` via `_GBM_EXCLUDE` (the stored feature names already drive `predict_scores`, so scoring matches). Fundamentals are kept: they're neutral, not harmful, and carry the valuation/quality narrative.
- **Adaptive leaf regularisation.** `min_samples_leaf` tuned to 60 on the walk-forward (OOS AUC/precision peaked at 60; 100 overfit precision back down), but **capped by training size** (`min(60, max(20, n//25))`) so early tiny folds — and the unit tests — can still split.

Levers **tested and rejected** (honest negatives, logged so they aren't re-tried):
- *Multi-seed gbm ensemble* — no effect: `HistGradientBoostingClassifier` has no early-stopping below 10k rows, so it is deterministic across seeds here.
- *Magnitude-weighted training* (sample_weight ∝ |fwd_ret − median|) — **hurt** (AUC 0.515): chasing extreme movers loses the median-crossing decision.
- *Confident-band training* (drop the ambiguous middle 20% at train time) — **hurt** (AUC 0.508): discards too much data.

Net effect (real CLI, `--normalize rank`, min_train 6):

| Model | AUC | precision@N | monthly win-rate |
|---|---:|---:|---:|
| gbm — before (accuracy pass 1) | 0.524 | 0.534 | 54.2% |
| **gbm — after** | 0.525 | **0.545** | **57.3%** |
| logistic (unchanged) | 0.515 | 0.555 | 61.8% |

Honest read: whole-cross-section AUC is near its ceiling for a 27-name large-cap universe, so the tangible gain is **pick quality** — gbm precision@N +1.1pp and monthly win-rate +3pp. Re-running the permutation test on the improved gbm, **precision@N significance strengthened from 5% to 1% (p=0.0099)** and the bootstrap AUC CI lower bound firmed to 0.504. Reaching AUC ≈0.55 would need a larger/broader universe or a richer feature set (logged as future work), not more hyperparameter tuning on this one — chasing it here would risk exactly the overfitting the significance test guards against.

Evidence: full suite **122 passed** (adaptive `min_samples_leaf` keeps the small-data unit tests splitting). Reports refreshed: `selection_top3_RANKNORM_improved.md`, `significance_both_GENERATED.md`.

---

## Significance testing: permutation p-values + bootstrap CIs (2026-07-26)

Goal: answer the question a single backtest cannot — *is the ranker's out-of-sample skill real, or luck?* Added a leakage-free significance layer (`selection/significance.py` + `cli/significance_cli.py`) and evaluated it on the improved (rank-normalised) core.

- **Label-permutation test.** Holding the model's real out-of-sample scores fixed, shuffle the outcome labels **within each rebalance date's cross-section** and recompute mean AUC / precision@N — 100 times — to build the "no-skill" null. p-value = `(1 + #{null ≥ real}) / (1 + n)`. No re-training and no crossing of dates, so it is leakage-free and assumption-free. (A fixed-score permutation is the standard significance test for a rank statistic and, unlike re-training 100× on shuffled labels, is fast enough to run interactively.)
- **Bootstrap CIs.** Resample the per-fold AUCs with replacement (2000×) for a 95% interval — the error bar the single backtest path lacked (addresses the review's "single path, no confidence interval" gap).

Results (top-3, rank-normalised, 100 shuffles):

| Model | Statistic | Real | Null mean | p-value | Verdict |
|---|---|---:|---:|---:|---|
| gbm | Mean AUC | 0.5245 | 0.4997 | **0.0099** | significant at 1% |
| gbm | Precision@N | 0.5344 | 0.4812 | **0.0297** | significant at 5% |
| logistic | Mean AUC | 0.5146 | 0.5006 | 0.0792 | marginal (not sig. at 5%) |
| logistic | Precision@N | 0.5547 | 0.4797 | **0.0099** | significant at 1% |

Bootstrap 95% CI on per-fold mean AUC: gbm [0.5006, 0.5463], logistic [0.4880, 0.5419].

Honest read: the **gbm ranker's AUC edge is significant (p≈0.01)** and **both models' top-N precision is significant** — the skill concentrates at the top of the ranking, exactly where the strategy acts. Logistic's *whole-cross-section* AUC is only marginal (p≈0.08, CI straddles 0.5), a nuance worth stating rather than hiding. This is the strongest single piece of "we did not fool ourselves" evidence in the project.

Evidence: `tests/test_significance.py` (4 offline tests — a signal case comes out significant, a noise case does not, p-value bounded/never-zero, bootstrap brackets the mean). Full suite **122 passed** (118 + 4). Saved report: `reports/significance_both_GENERATED.md`. Run it yourself: `python -m src.cli.significance_cli --permutations 100`.

Remaining higher-ROI accuracy levers (not yet done): ranking/quantile-label objective, permutation-importance noise pruning, and a heavier *re-training* shuffle null for the return series specifically.

---

## Accuracy pass: cross-sectional normalisation + orthogonal factors (2026-07-26)

Goal: raise the core ranker's *predictive* quality (mean AUC / precision@N), the project's real bottleneck — the earlier report showed gbm AUC ≈ 0.511, barely above the 0.5 chance base rate. Two leakage-safe levers, each measured before/after on the cached universe (27 tickers, 138 monthly rebalances, walk-forward, net of 10 bps costs):

- **Cross-sectional normalisation (`features.cross_sectional_normalize`).** Standardises each feature *within each rebalance date's own cross-section* (`rank` percentile in [-0.5, 0.5], default; `zscore` optional). Because it only ever consults a single date's cross-section, it is as-of-safe by construction (asserted by a new test that removing a future date leaves an earlier date's normalised values unchanged). Selection is a *relative* problem, so stripping each month's absolute level is the single biggest lift here. `zscore` was tried and was *worse* for the tree (gbm AUC 0.507), so `rank` is the default.
- **Two orthogonal technical factors.** `mom_12m` (252-day long-horizon momentum, classically the strongest cross-sectional factor) and `reversal_5d` (1-week short-term reversal on a distinct window, so a tree cannot recover it from the 21d+ momenta).

Measured effect (top-3, gbm; `select_cli --normalize rank`):

| Metric | Baseline (raw levels) | + factors + rank-norm | Δ |
|---|---:|---:|---:|
| gbm mean AUC | 0.511 | **0.524** | +1.3pp |
| gbm precision@N | 0.532 | 0.534 | +0.2pp |
| logistic precision@N | 0.540 | 0.555 | +1.5pp |
| logistic monthly win-rate | 56.0% | 61.8% | +5.8pp |

AUC moving from 0.511 → 0.524 is modest but the right direction and, unlike the headline returns, reflects genuine ranking skill above the base rate. `--normalize` defaults to `rank`; pass `--normalize none` to reproduce the old raw-level behaviour. The same normalisation is wired into the UI selector (`data_access.run_universe_selection(normalize="rank")`).

Evidence: full suite **118 passed** (up from 113; +5 tests — 3 for `cross_sectional_normalize` incl. the cross-date leakage guard, plus the two new factor columns flow through the existing shape/robustness tests). Saved report: `reports/selection_top3_RANKNORM_improved.md`.

Reflection: the honest framing is unchanged — the ranker has a small, real edge, and the eye-catching cumulative returns still come mostly from concentrated top-3 factor exposure. Next highest-ROI accuracy levers (not yet done): a `shuffle-label` control + bootstrap CIs for significance, a ranking/quantile-label objective, and permutation-importance noise pruning.

---

## Phase 4 polish: UI usability pass across all four views (2026-07-22)

Four user-requested refinements, all logic added to the Streamlit-free `data_access.py` (still headlessly testable) with `views.py` staying thin:

- **Single-stock analysis** — replaced the 3-option history box with 8 presets (1 day → 5 years) via `SINGLE_STOCK_PRESETS`, each mapping to a suitable yfinance period + interval and its own **direction-horizon** options (bars scale with the range). yfinance has no `3y` period, so "3 years" fetches `5y` daily and keeps the last ~756 rows. All-NaN moving-average columns are dropped from the chart on short/intraday ranges, with a caption that long-window signals may be limited there.
- **Top-N recommendations** — reworked for non-professionals: a **multiselect of the built-in large-cap universe** (sensible 5-name default) plus a **custom-ticker box**, a plain-language explainer, and a guard requiring more names than the top-N.
- **News sentiment (Ext A)** — the aggregate score now shows a **label + plain-English meaning** (`sentiment_interpretation`) and a scale legend; **each headline is a clickable link** (added `url` to `Headline`, extracted from both yfinance news schemas via `_extract_news_url`); and a **"headlines to show"** control (aggregate still computed over all fetched, `n_total` reported).
- **Backtest & evaluation** — history presets 1 month → 5 years (`BACKTEST_PRESETS`, intraday ranges excluded); a **ticker selector with recent-search history + suggestions** (`_ticker_selector` / `_record_ticker` in session state); and a clear "period too short for a walk-forward window" `ValueError` surfaced as a friendly hint rather than a crash.

Evidence: full suite **113 passed** (up from 105; +8 new tests covering presets, `default_universe`, `sentiment_interpretation` bands, headline URL/limit/order plumbing, the too-short-period guard, and both yfinance news schemas). Verified interactively with Streamlit `AppTest` + fakes: switching to the "1 day" preset swaps horizon options to [1,3,6,12]; the sentiment view renders clickable article links; the backtest view runs to metrics — all with no exceptions. READMEs refreshed (Rule 2): `src/ui/README.md` view descriptions updated.

---

## Phase 4: Streamlit web interface — all components tied together (2026-07-18)

Decisions (confirmed with the user): Streamlit app in-repo; all four views built now (single-stock analysis, top-N recommendations, news sentiment, backtest & evaluation); on-demand-per-ticker data with in-session caching.

Completed:
- **`src/ui/data_access.py`** — a deliberately **Streamlit-free** logic layer (Rule 1: logic vs presentation split). Provider-injectable pure functions that wire the already-tested modules: `analyse_single_stock` (advisor + indicators + Ext-B direction), `evaluate_ticker` (walk-forward backtest + direction evaluation), `score_ticker_sentiment` (Ext-A FinBERT via the torch-free `onnx-local` backend), and `run_universe_selection` (Direction-1 selector → latest top-N, reusing `select_cli.month_end_rebalances`).
- **`src/ui/views.py`** — four thin render functions with `st.cache_data` wrappers (the on-demand + memoised data strategy). Single-stock is auto/fast; the heavy selector, sentiment, and backtest views are button-gated so nothing expensive runs on page load.
- **`src/ui/app.py`** — sidebar navigation + page config; adds the project root to `sys.path` (Streamlit runs the file directly) and guards `main()` under `__main__` so importing the module doesn't auto-run.
- **`requirements-ui.txt`** (streamlit, altair) — the UI reuses the core sklearn stack, so no torch; runs on x86 macOS.
- **`tests/test_ui.py`** (4 tests) — exercise the logic layer with a fake provider (no network/Streamlit/model): full single-stock assembly, graceful missing-intraday, direction soft-fail on short history, and a guard that `data_access` imports no Streamlit.

Evidence: full suite **105 passed** (`python -m pytest`), up from 101 (+4). Beyond the committed offline tests, ran a headless render check in the sandbox with Streamlit's `AppTest`: patched the data layer with the fake provider and confirmed all four views render with **no exceptions** (single-stock fully populated; button-gated views show their prompts). READMEs refreshed (Rule 2): new `src/ui/README.md`, root README gains a "Web interface" section + `streamlit run` command + updated layout/status, test count 101→105.

Reflection: the logic/presentation split is what keeps a UI honest and testable — every number the app shows comes from a function that is unit-tested headlessly, so the Streamlit layer is just widgets. With Phase 4 done, all planned components (core + both extensions + UI) are in place.

---

## Extension B: per-stock direction forecast (Direction 2, ~15%) — full vertical slice (2026-07-18)

Decision: the scaffold defaulted to an LSTM/GRU, which needs a deep-learning framework — the exact stack that has no working x86-macOS wheel (the constraint that shaped Extension A). Chosen approach (confirmed with the user): **framework-light, on the same scikit-learn stack as the core ranker**, not a deep net. This is portable, readable (Rule 1), fast, and honest given single-stock direction is near a coin flip. Scope (confirmed): a full vertical slice.

Completed:
- **`direction/forecaster.py`** — rewrote the scaffold into a working `DirectionForecaster(kind="gbm"|"logistic"|"baseline", horizon_days=5)` mirroring `RankerModel`'s model-agnostic `fit`/`predict` surface. Leakage-safe causal features (lagged returns, momentum, volatility, RSI, MACD histogram, distance-from-SMAs — reusing `common/indicators.py`) and forward-direction labels (`sign(Close[t+h]/Close[t]-1)`, NaN tail). Shared estimator helpers (`build_estimator`, `prob_up_from_estimator`, `baseline_prob_up`) so the model and evaluator stay DRY. Handles degenerate single-class windows.
- **`direction/evaluate.py`** — walk-forward directional evaluation with a **`horizon_days` embargo** between training window and prediction date (a row's label is only known *h* days after its feature date, so including the gap would leak). Reports `accuracy`, `base_rate` (naive "always predict the majority direction"), `edge = accuracy − base_rate`, `up_rate`, and ROC-`auc`.
- **`src/cli/direction_cli.py`** — runs the forecaster on the tickers you pass (intended: the selector's top-N), printing per-ticker out-of-sample accuracy / base rate / edge / AUC + the **current** P(up) call, and writing a markdown report to `reports/`. Honest interpretation baked into the report.
- **`tests/test_direction.py`** (10 tests, fully offline/deterministic): feature causality (mutating the last close leaves all earlier rows identical), label definition + NaN tail, forecaster fit/predict contract + prob∈[0,1], invalid-kind rejection, baseline follows momentum, **learnable pattern (sinusoid) beats base rate with AUC>0.9**, **pure noise shows no meaningful edge**, metric well-formedness, and a removability guard (selection package imports no `src.direction`).

Evidence: full suite **101 passed** (`python -m pytest`), up from 91 (+10). Sandbox checks: leakage-safe features confirmed; on a sinusoidal (learnable) series gbm/logistic reach ~0.99–1.00 accuracy vs ~0.50 base rate (edge +0.49, AUC 1.0), while pure noise stays ~chance (edge +0.04) and the momentum baseline fails to beat majority — exactly the honest separation intended. CLI verified end-to-end against a mocked price provider.

READMEs refreshed (Rule 2): new `src/direction/README.md` (implemented status, framework-light rationale, usage), root README gains an Extension-B section + test command and updated layout/status, test count 91→101.

Reflection: framing the result against the base rate (not raw accuracy) is what makes this defensible — on real single stocks the expected finding is a thin or absent edge, and the harness will say so plainly rather than overclaiming.

Live evaluation takeaway (real tickers, 2026-07-18): ran on AAPL/MSFT/NVDA (5y, horizon 5). **No meaningful single-stock edge** — the learned models sit at or slightly below the naive base rate (gbm edge −3.2 / −6.9 / −9.3pp, AUC ~0.47–0.51; logistic AAPL −4.2pp). Notably the **trivial momentum baseline was not outperformed** — on AAPL it scored +1.4pp / AUC 0.551 vs gbm −3.2pp and logistic −4.2pp, i.e. gradient boosting and logistic regression do not earn their complexity on this noisy target. The 8y/horizon-10 AAPL run also illustrates the base-rate-vs-AUC split: AUC 0.544 (faint ranking signal) yet edge −5.8pp because the base rate is 61.4% (a strong upward-drift benchmark). Interpreted honestly, this is the expected outcome for single-stock short-horizon direction and, crucially, the near-chance results are positive evidence the pipeline is leakage-free (no implausibly high accuracy anywhere). This is exactly why Direction 2 carries side weight and is framed as a confirmation layer, not a standalone strategy.

Next: the Streamlit UI (Phase 4) is the last remaining scaffold.

---

## Extension A: `onnx-local` made genuinely torch-free (2026-07-18)

Surfaced by the user running `--backend onnx-local` on their Intel Mac: it worked and produced sensible scores, but the traceback showed `from transformers import AutoConfig, AutoTokenizer` pulling in **`import torch`** (their leftover torch 2.2.2), which then threw a NumPy warning (`_ARRAY_API not found`) from torch 2.2.2's NumPy-1.x build vs the venv's NumPy 2.4.6. So the "torch-free" claim was violated: modern `transformers` (4.57) eagerly imports torch at package load whenever torch is installed.

Fix:
- Rewrote `_load_onnx_local_classifier` to **not import `transformers` at all** — tokenizer now loads via the Rust `tokenizers` lib (`Tokenizer.from_file(tokenizer.json)`, `enable_truncation`/`enable_padding`), and labels are read straight from `config.json` with `json`. Feeds `input_ids`/`attention_mask`/`token_type_ids` (filtered to the model's actual inputs) into ONNX Runtime.
- Dropped `transformers` from `requirements-extension-a-onnx-local.txt` and documented why (with a `pip uninstall torch` tip to silence the leftover-torch NumPy warning).

Evidence: full suite **91 passed**. Additionally verified the *real* rewritten path end-to-end in a sandbox against a hand-built tiny ONNX model + tokenizer + config: correct shape, probabilities sum to 1, labels mapped — and asserted `torch`/`transformers` are **not** in `sys.modules` afterwards. (The real FinBERT download is blocked in the dev sandbox, so the full model still runs only on the user's machine.)

---

## Extension A: completion & polish (2026-07-18)

Decision: after a scope review, Extension A's modelling was judged **complete and honest** as-is. Per an explicit data-scope decision, the news source stays **live-only (yfinance) + documented** — a historical ablation report was *deliberately not built*, because on recent-only headlines the sentiment column is near-empty across the walk-forward and any such report would be misleading. The ablation harness (`select.run_ablation`) remains wired and ready for a historical archive dropped in behind the `NewsSource` interface (future work).

Completed (the two gaps genuinely worth closing, both tied to the new backend):
- **Testable scoring math (Rule 1).** Extracted the softmax + label-mapping into a dependency-free module-level helper `_logits_to_distributions` (and `_softmax_row`) shared by any logit-producing backend; the `onnx-local` `_run` now delegates to it. No numpy needed for the math, so it is unit-testable offline.
- **Offline test coverage for `onnx-local`.** New `tests/test_sentiment_onnx.py` (5 tests): softmax normalisation/ordering vs a hand-computed reference, distribution shape + label mapping, end-to-end plug into `FinBERTScorer`, backend dispatch to the torch-free loader, and lazy single-load of the backend. No model download, no torch/onnx/ORT required.
- **Docs reconciled (Rule 2).** Fixed the stale `src/sentiment/README.md` (it still claimed `onnx` was the default and torch-free) — now a 3-backend table (`torch` default / `onnx-local` torch-free / optimum-`onnx`) with the Intel-Mac guidance; corrected the misleading `_load_onnx_model` docstring; refreshed the root README test count and sentiment test command.

Evidence: full suite **91 passed** (`python -m pytest`), up from 86 — the 5 additions are the new offline `onnx-local` tests. Extension A remains removable-by-design (core still imports without touching `sentiment/`).

Reflection: the contribution stays honest — integration + evaluation of a pre-trained model, portable across machines, with the one real limitation (no historical news archive) documented rather than papered over.

Next: Extension B (`direction/` — per-stock direction forecast on the selected top-N), then the Streamlit UI.

---

## Extension A portability, revisited: torch-free `onnx-local` backend (2026-07-18)

> **Supersedes the entry below.** The earlier "ONNX Runtime backend" entry made
> `optimum`-based `onnx` the *default*. That approach does **not** actually run on
> an Intel Mac — `optimum`'s ONNX export still traces the model with torch and
> needs torch ≥ 2.4, which has no x86-macOS wheel. It was therefore reverted:
> `backend="torch"` is the default again, and a genuinely torch-free path was added.

Problem (restated correctly): on x86 macOS both the `torch` backend (torch caps at 2.2.2) and the `optimum` `onnx` backend (export needs torch ≥ 2.4) are dead ends, so FinBERT could not run on the user's Intel Mac at all.

Completed:
- Added a third backend, **`backend="onnx-local"`** (`_load_onnx_local_classifier` in `sentiment/finbert.py`): loads a *pre-exported* `model.onnx` with plain ONNX Runtime + the `tokenizers` tokenizer, applies softmax, and returns the exact `list[list[{label, score}]]` shape `FinBERTScorer` already consumes. It never imports torch or `optimum`. Defaults to the `jonngan/finbert-onnx` mirror (same weights + `positive/negative/neutral` labels as `ProsusAI/finbert`) when the caller passes the torch-only id.
- Kept the extension modular (Rule 1): the ONNX path is an isolated helper; core stays untouched; torch-free deps live in a new **`requirements-extension-a-onnx-local.txt`** (onnxruntime, tokenizers, transformers, huggingface_hub, numpy — all ship x86-macOS wheels).
- Wired `--backend onnx-local` into `sentiment_cli`; reverted the default backend to `torch`.
- README refreshed (Rule 2): install-per-machine block, a backend comparison table, corrected macOS note, and the `--backend onnx-local` flag on the demo commands.

Evidence: full suite **86 passed** (offline tests use the injected classifier, so no torch/onnx needed). The live model download is blocked by the dev sandbox proxy (HF not allowlisted), so the real FinBERT runs on the user's machine; the ONNX Runtime execution path was verified here against a tiny hand-built ONNX model — real `onnxruntime.run` + softmax produced correctly-shaped, normalised outputs that `FinBERTScorer` aggregates.

Next: complete + polish Extension A (see below).

---

## Extension A portability: ONNX Runtime backend (x86-macOS fix)

> **Note (2026-07-18): superseded — see the entry above.** Making the
> `optimum`/`onnx` backend the default did not solve the Intel-Mac case (its
> export still requires torch ≥ 2.4). The default was reverted to `torch` and a
> torch-free `onnx-local` backend was added instead.

Problem: PyTorch's last macOS-x86 (Intel Mac) build is 2.2.2, but modern transformers refuse to load `.bin` weights on torch < 2.6 (CVE-2025-32434), so FinBERT would not run on the user's Intel Mac.

Completed:
- Made `FinBERTScorer` backend-pluggable (`backend="onnx"` default, `"torch"` alternative). ONNX Runtime runs the same FinBERT with no torch at inference and first-class Intel-Mac support; the torch path now forces safetensors to sidestep the CVE gate where torch ≥ 2.6 exists.
- `_load_finbert_pipeline` builds tokenizer + model per backend; the ONNX loader tries a torch-free load first and falls back to a one-time export (safetensors, no torch.load).
- Added `--backend {onnx,torch}` and `--model` flags to `sentiment_cli`; widened the demo to default to the full universe with a "market mood" average.
- Rewrote `requirements-extension-a.txt` around the ONNX stack (`optimum[onnxruntime]`, `onnxruntime`, `safetensors`).

Evidence: full suite still 86 passed (offline tests use the injected classifier, so no torch/onnx needed); CLI help shows `--backend`, default backend verified `onnx`. README + module README updated (standing rule).

Note: the live ONNX path could not be executed in the dev sandbox (no network to model hub); it is intended for the user's machine.

---

## Evaluation chapter strengthening: ranking quality + consistency

Completed:
- Added **prediction-quality metrics** to `selection/select.py` that judge the ranker itself, not just the portfolio: `precision_at_n` (fraction of picked names that then beat the universe median), `mean_auc` (average per-rebalance ROC-AUC of the scores vs binary labels), and `base_rate` (so precision is read *relative* to chance). Exposed via `ranking_metrics(...)` and on `SelectionBacktest`.
- Added **consistency reporting**: `monthly_win_rate` (fraction of holding periods the strategy beat the benchmark) and `yearly_breakdown(...)` (per-calendar-year strategy vs benchmark return, periods, and within-year win-rate) — so a reader can see whether one year carries the whole result.
- Refactored the walk-forward into a single pass (`_walk_forward_records`) feeding both portfolio weights and ranking metrics — folds are never trained twice.
- The `select_cli` report now has "Prediction Quality" and "Consistency — Per-Year" sections, and the terminal summary prints precision@N / AUC / monthly win-rate.

Evidence:
- Full suite 86 passed (`python -m pytest`), up from 83. New tests: ranking metrics reward a skilful ranker (precision@N 1.0, AUC 1.0, base rate 0.5 on engineered data), backtest bundles ranking + yearly breakdown, periodic win-rate maths.
- README refreshed (standing rule).

Also: attempted the live FinBERT sentiment demo — the sandbox proxy blocks the torch/model download, so the real model runs on the user's machine; verified the sentiment *pipeline* end-to-end here with a transparent stand-in scorer (AAPL/NVDA positive, INTC negative, stale/future headlines correctly excluded).

Reflection:
- These metrics make the honesty concrete: on real data, precision@N only a little above the 0.5 base rate and AUC near 0.5 would say the raw returns are luck/beta, while a steady per-year win-rate would support a genuine edge. This is the backbone of a defensible evaluation chapter.

---

## Extension A: FinBERT news sentiment (Direction 3, ~15%)

Completed:
- Implemented `sentiment/news_source.py`: a leakage-safe `NewsSource` interface with two concrete sources — `InMemoryNewsSource` (offline/tests) and `YFinanceNewsSource` (live). Both return only headlines published on or before the as-of date, within a lookback window.
- Implemented `sentiment/finbert.py`: `FinBERTScorer` maps each headline to a signed score (P(positive) − P(negative)) and aggregates to a per-(rebalance_date, ticker) `sentiment` feature. The scoring backend is injectable (default lazily loads `ProsusAI/finbert`), so tests run fully offline with a fake classifier — no model download.
- Wired to the core: `build_sentiment_feature(...)` output feeds `selection.features.build_feature_matrix(..., sentiment=...)`, and the ablation harness already in `select.run_ablation` measures with/without sentiment.
- Added `src/cli/sentiment_cli.py` (live "sentiment now" demo) and a separate `requirements-extension-a.txt` (transformers, torch) so the core install stays lightweight.

Evidence:
- Full suite 83 passed (`python -m pytest`), up from 77. New `tests/test_sentiment.py` (6 tests): leakage window, signed-score aggregation, feature assembly, future-headline-does-not-leak, and integration with the core feature matrix.
- Removable-by-design verified: `selection/` imports and runs without touching `sentiment/`.
- README + sentiment module README refreshed (standing rule): sentiment CLI, ablation usage, extension-A requirements.

Reflection:
- The contribution here is honest integration + evaluation of a pre-trained model, not training an LLM. The main real-world limitation is data: yfinance only serves recent headlines, so a full historical sentiment backtest needs a deeper news archive — the interface is ready for one.

Next:
- Extension B (`direction/` — per-stock direction forecast on the selected top-N), then the Streamlit UI.

---

## Phase 2 hardening: realistic evaluation + bug fixes

Completed (post-Phase-2 fixes surfaced by running on live data):
- **Transaction costs added to the selection backtest.** `evaluate_portfolio` now charges cost on turnover (sum of absolute target-weight changes per rebalance), matching the rule-based walk-forward engine (default 10 bps/turnover). Strategy returns are net of costs; passive benchmarks stay gross (buy-and-hold). Average turnover is reported. On the full universe top-3, this trims the logistic result from ~13,817% (gross) to ~12,066% (net), and the report/CLI now state costs explicitly. New `--cost` flag.
- **Timezone bug fixed.** Live yfinance indices are tz-aware with mixed DST offsets; centralised into `common/timeindex.py` (`to_naive_index`) used by features/labels/select/universe. Regression test added.
- **Cache windowing bug fixed.** `load_prices` cached by ticker only, so `--start` was silently ignored on repeat runs. Now the cache stores full history but the returned frame is sliced to the requested window. Verified: 2016-start → 126 rebalances, 2019-start → 90.
- **Imputer warning flood silenced** (all-NaN fundamental columns in early folds) via a targeted filter around fit and predict.
- **`walkforward_cli` report path fixed** (was writing to `src/reports/`).

Evidence: full suite 77 passed (`python -m pytest`), up from 68. README refreshed (`--cost`, net-of-cost wording, test count).

Honest read of results: raw returns remain inflated by (a) a survivorship-biased all-mega-cap universe and (b) top-N concentration; the defensible finding is a thin, period-dependent risk-adjusted (Sharpe) edge over the equal-weight benchmark, now measured net of costs.

---

## Phase 2: ML ranker, walk-forward selection and evaluation (ML core complete)

Completed:
- Implemented `selection/model.py`: a model-agnostic `RankerModel` exposing only `fit` / `predict_scores`. `kind="gbm"` uses scikit-learn's `HistGradientBoostingClassifier` (native NaN handling); `kind="logistic"` is a median-impute + standardise + `LogisticRegression` baseline. Handles missing labels and degenerate single-class folds honestly.
- Implemented `selection/select.py`: cross-sectional walk-forward selection — at each monthly rebalance the ranker trains only on strictly-past rows, scores the current cross-section, and holds the top-N equally until the next rebalance. Realised returns are evaluated against an equal-weight universe benchmark and SPY via `common.metrics`. Includes a sentiment ablation hook (`run_ablation`) for Extension A.
- Added `src/cli/select_cli.py`: runs the full pipeline (load → features → labels → walk-forward backtest) for both the GBM ranker and the logistic baseline, prints a summary and writes a markdown report to `reports/`.
- Added `scikit-learn` (+ scipy/joblib/threadpoolctl) to `requirements.txt`.

Evidence:
- Full suite green: 68 passed (`python -m pytest`), up from 52. New tests: `test_model.py` (ranking, NaN handling, single-class folds), `test_select.py` (walk-forward picks, weights, strategy vs benchmark, SPY, ablation), `test_select_cli.py` (offline end-to-end integration).
- README updated with the new `select_cli` operation and refreshed test commands (standing rule).

Reflection:
- This is the "core-complete" milestone: the project is now an ML stock selector, not just a rule-based tool with a clean dataset. The leakage discipline from Phases 0–1 carries straight through — folds never see their own or future data.

Next:
- Extension A (`sentiment/` — FinBERT news sentiment) feeding a `sentiment` feature, evaluated via the ablation already wired here; then Extension B (`direction/`) and the Streamlit UI.

---

## Phase 1: Leakage-safe feature matrix and labels (ML core groundwork)

Completed:
- Implemented `selection/features.py`: one feature row per (rebalance_date, ticker) combining technical features (1/3/6-month momentum, volatility, RSI, MACD) with as-of fundamental features (PE, PB, ROE, earnings growth, net margin, debt-to-equity). Optional sentiment column (Extension A) is left-joined when provided.
- Implemented `selection/labels.py`: binary cross-sectional labels — a stock is positive when its forward return over the horizon beats the universe median that period (relative/market-neutral framing, used as a training target only).
- Enforced the leakage-safety invariant: technical features are sampled as-of the last close on or before t; fundamentals are joined on their `available_date` (period-end + reporting lag), never the period-end.
- Added 12 offline unit tests (`tests/test_features.py`, `tests/test_labels.py`), including a direct leakage test: injecting a future price spike and a future filing leaves every row at/before t unchanged.

Evidence:
- Full suite green: 52 passed (`python -m pytest`), up from 40.
- Modules classified by function under `selection/` (core), separate from `common/` infra and the `sentiment/`/`direction/` extension packages.

Reflection:
- These two modules are the dataset the ML ranker reads from; getting the as-of joins right here is the single most important correctness property before any model is trained.

Next:
- Phase 2: implement `selection/model.py` (gradient-boosting ranker + logistic baseline) and `selection/select.py` (walk-forward top-N selection), then evaluate against SPY / equal-weight.

---

## Stage 2: Backtesting and Evaluation

Completed:
- Created a backtesting module for the rule-based advisor.
- Implemented historical signal generation using SMA20, SMA50, SMA200, and crossover logic.
- Simulated strategy returns using historical stock prices.
- Compared the strategy against a buy-and-hold baseline.
- Calculated evaluation metrics including strategy return, buy-and-hold return, maximum drawdown, number of trades, and win rate.
- Added unit tests for the backtester.

Evidence:
- Ran backtest using AAPL historical data.
- Generated `reports/backtest_results.md`.
- Unit tests passed using `python -m pytest`.

Reflection:
- This stage helps evaluate whether the recommendation logic has historical value.
- The result does not guarantee future performance, but it provides a measurable baseline for comparison with later machine learning models.

Next:
- Improve the backtest by testing multiple stocks.
- Save backtest results for several tickers.
- Begin feature engineering for machine learning.