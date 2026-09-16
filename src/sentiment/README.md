# `sentiment/` — Extension A (Direction 3, ~15%)

News sentiment as an *added feature* to the core ranker. FinBERT scores recent
headlines per stock; the score joins the feature matrix in `selection/features.py`.
Its value is proved by an **ablation study**: run the full walk-forward backtest
with and without the sentiment column and report the difference in each metric.

| Module | Responsibility | Status |
|---|---|---|
| `news_source.py` | Leakage-safe headline feed: `InMemoryNewsSource` (offline/tests) + `YFinanceNewsSource` (live). | done |
| `finbert.py` | `FinBERTScorer` — signed sentiment score per headline, aggregated to a per-(date, ticker) feature. Backend-pluggable (`torch` / `onnx` / `onnx-local`). | done |

**Usage.** `FinBERTScorer.build_sentiment_feature(source, tickers, dates)` returns a
`(rebalance_date, ticker)` frame with a `sentiment` column; pass it to
`selection.features.build_feature_matrix(..., sentiment=...)`, then
`selection.select.run_ablation(...)` reports metrics with and without it.

**Dependencies / backend.** Heavy deps are kept out of the core install. Scoring
is injectable, so the offline tests run with a fake classifier — no model
download, no torch/onnx needed. Three inference backends share one scoring path
(logits → softmax → signed score):

| Backend | Runs on | Install | Notes |
|---|---|---|---|
| `torch` *(default)* | Apple Silicon / Linux / Windows | `requirements-extension-a.txt` | Normal `pip install torch`; forces safetensors to dodge the `torch.load` CVE gate. |
| `onnx-local` | **x86 macOS / Intel Mac**, + everywhere | `requirements-extension-a-onnx-local.txt` | **Torch-free**: loads a *pre-exported* `model.onnx` on ONNX Runtime + `tokenizers`. Never imports torch or `optimum`. |
| `onnx` (optimum) | where torch ≥ 2.4 exists | `+ optimum[onnxruntime]` | Exports via `optimum`, which traces with torch — so **not** an Intel-Mac fix. |

> On an Intel Mac use `--backend onnx-local`. The `torch` and optimum-`onnx`
> backends both need a torch that has no x86-macOS wheel past 2.2.2.

**Removable by design:** deleting this package leaves `selection/` fully working
(it simply trains on the technical + fundamental features only). Verified: the
core imports without touching `sentiment/`.

**Limitation:** yfinance exposes only *recent* headlines, so the live source
supports "sentiment now", not a multi-year historical backtest — a richer news
archive can be dropped in behind the same `NewsSource` interface.
