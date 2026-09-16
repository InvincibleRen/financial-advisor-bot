"""FinBERT sentiment scoring (Extension A, Phase 3).

Wraps the pre-trained FinBERT model to turn headlines into a numeric sentiment
feature per (rebalance_date, ticker). Using a pre-trained model keeps the
project's contribution scoped to *integration and evaluation* of sentiment, not
training a language model from scratch.

Design
------
* **Scoring backend is injectable and framework-flexible.** ``FinBERTScorer``
  takes an optional ``classifier`` callable (``list[str] -> list[list[{label,
  score}]]``, the shape of a 🤗 Transformers ``text-classification`` pipeline with
  ``top_k=None``). The default lazily builds the real FinBERT pipeline on first
  use; tests inject a deterministic fake, so aggregation is verified offline with
  no model download. This mirrors the injectable fetchers in ``universe.py``.
* **Backend (``backend="torch"`` default, ``"onnx"`` / ``"onnx-local"`` optional).**
  The torch path loads FinBERT via safetensors (which sidesteps the ``torch.load``
  security gate, CVE-2025-32434). It runs well on **Apple Silicon (arm64) / Linux /
  Windows** with a normal ``pip install torch``. **x86 macOS (Intel Mac) is a dead
  end for the *torch* stack:** PyTorch stops at 2.2.2 there, and both new
  ``transformers`` (torch ≥ 2.6 gate) and ``optimum`` (needs ``torch.rms_norm``,
  torch ≥ 2.4) require newer torch — so ONNX-via-``optimum`` (the ``"onnx"`` backend,
  which *exports* with torch) does *not* rescue Intel Mac either. **The torch-free
  fix for Intel Mac is ``backend="onnx-local"``:** it loads a *pre-exported*
  ``model.onnx`` with plain ONNX Runtime + the ``tokenizers`` tokenizer and never
  imports torch, so it installs and runs on x86 macOS (``onnxruntime`` ships
  ``x86_64`` macOS wheels). On an Apple Silicon machine you can instead create the
  venv from an **arm64** Python and the default torch backend just works.
* **Signed score.** Each headline maps to ``P(positive) - P(negative) ∈ [-1, 1]``
  (neutral contributes 0); a batch is aggregated by mean. No news → 0.0 (neutral).
* **Removable by design.** ``selection/`` never imports this package, so deleting
  ``sentiment/`` leaves the core ranker fully working (it simply trains on the
  technical + fundamental features only).

Transaction note: the heavy dependencies (``transformers``, ``torch``) live in
``requirements-extension-a.txt``, keeping the core install lightweight.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from src.sentiment.news_source import NewsSource

# A classifier maps a batch of texts to per-text label/score distributions.
Classifier = Callable[[List[str]], List[List[Dict[str, object]]]]

_LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class FinBERTScorer:
    """Score financial text with FinBERT (loaded lazily on first use)."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        classifier: Optional[Classifier] = None,
        backend: str = "torch",
    ) -> None:
        self.model_name = model_name
        self.backend = backend  # "torch" (default), "onnx", or "onnx-local" (torch-free)
        self._classifier = classifier  # injected fake in tests; real pipeline otherwise

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #

    def score_texts(self, texts: List[str]) -> float:
        """Return a single aggregate sentiment score in [-1, 1] for a batch.

        Each text is scored ``P(positive) - P(negative)`` and the batch is
        averaged. An empty batch (or all-blank texts) returns 0.0 (neutral).
        """
        clean = [t for t in texts if isinstance(t, str) and t.strip()]
        if not clean:
            return 0.0
        distributions = self._classify(clean)
        signed = [_signed_score(dist) for dist in distributions]
        return float(sum(signed) / len(signed))

    def _classify(self, texts: List[str]) -> List[List[Dict[str, object]]]:
        """Run the (lazily-loaded) classifier over a batch of texts."""
        if self._classifier is None:
            self._classifier = _load_finbert_pipeline(self.model_name, self.backend)
        return self._classifier(texts)

    # ------------------------------------------------------------------ #
    # Feature assembly                                                   #
    # ------------------------------------------------------------------ #

    def build_sentiment_feature(
        self,
        source: NewsSource,
        tickers: List[str],
        rebalance_dates: List[pd.Timestamp],
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """Return a sentiment feature indexed by ``(rebalance_date, ticker)``.

        The frame has a single ``sentiment`` column and is consumed directly by
        ``selection.features.build_feature_matrix(..., sentiment=<this>)``. For
        each cell the source supplies only leakage-safe headlines (published on or
        before the date, within ``lookback_days``), which are then scored. Cells
        with no headlines are omitted and become NaN once merged.
        """
        cells: Dict[tuple, float] = {}
        for rebalance_date in pd.to_datetime(list(rebalance_dates)):
            for ticker in tickers:
                headlines = source.fetch_headlines(ticker, rebalance_date, lookback_days)
                if not headlines:
                    continue
                cells[(rebalance_date, ticker)] = self.score_texts([h.text for h in headlines])
        return _sentiment_frame(cells)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _softmax_row(logits: Sequence[float]) -> List[float]:
    """Numerically-stable softmax over one logit vector (pure Python)."""
    shift = max(logits)
    exps = [math.exp(float(x) - shift) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _logits_to_distributions(
    logits: Sequence[Sequence[float]],
    id2label: Dict[int, str],
) -> List[List[Dict[str, object]]]:
    """Turn a batch of raw logit rows into the 🤗 ``text-classification`` shape.

    Returns ``list[list[{"label", "score"}]]`` (softmax probabilities per label),
    exactly what ``FinBERTScorer`` consumes. Dependency-free so the ONNX scoring
    math is unit-testable offline, and shared by any logit-producing backend.
    """
    out: List[List[Dict[str, object]]] = []
    for row in logits:
        probs = _softmax_row([float(x) for x in row])
        out.append(
            [{"label": id2label.get(j, str(j)), "score": float(probs[j])}
             for j in range(len(probs))]
        )
    return out


def _signed_score(distribution: List[Dict[str, object]]) -> float:
    """Collapse a label/score distribution into ``P(pos) - P(neg)``."""
    total = 0.0
    for entry in distribution:
        label = str(entry.get("label", "")).lower()
        score = float(entry.get("score", 0.0))
        total += _LABEL_SIGN.get(label, 0.0) * score
    return total


def _sentiment_frame(cells: Dict[tuple, float]) -> pd.DataFrame:
    """Build the ``(rebalance_date, ticker)`` -> sentiment frame."""
    if not cells:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["rebalance_date", "ticker"],
        )
        return pd.DataFrame({"sentiment": []}, index=empty_index, dtype="float64")

    index = pd.MultiIndex.from_tuples(cells.keys(), names=["rebalance_date", "ticker"])
    return pd.DataFrame({"sentiment": list(cells.values())}, index=index).sort_index()


# Pre-exported ONNX FinBERT mirror (same weights/labels as ProsusAI/finbert), used
# by the torch-free ``onnx-local`` backend when the caller passes the torch-only id.
_ONNX_LOCAL_DEFAULT_REPO = "jonngan/finbert-onnx"


def _load_finbert_pipeline(model_name: str, backend: str = "torch") -> Classifier:
    """Lazily construct the real FinBERT text-classification pipeline.

    ``backend="onnx-local"`` is the **torch-free** path (recommended on x86 macOS /
    Intel Mac): it loads a *pre-exported* ``model.onnx`` with plain ONNX Runtime and
    never imports torch or ``optimum``. ``backend="onnx"`` uses ``optimum`` (which
    *exports* with torch, so it needs a torch ≥ 2.4 environment). ``backend="torch"``
    uses the classic PyTorch path, forcing safetensors so it also works on torch
    2.2.x (the CVE-2025-32434 gate only blocks ``torch.load`` of pickled ``.bin``).
    """
    if backend == "onnx-local":
        return _load_onnx_local_classifier(model_name)

    from transformers import AutoTokenizer, pipeline  # local: heavy, optional deps

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_onnx_model(model_name) if backend == "onnx" else _load_torch_model(model_name)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    def _run(texts: List[str]) -> List[List[Dict[str, object]]]:
        return clf(texts)

    return _run


def _load_onnx_local_classifier(model_name: str) -> Classifier:
    """Build a **genuinely torch-free** FinBERT classifier on ONNX Runtime.

    Deliberately avoids importing ``transformers`` — modern ``transformers`` eagerly
    ``import torch`` at package load whenever torch happens to be installed, which
    would defeat the point on an Intel Mac (and drag in a torch/NumPy ABI clash).
    Instead the tokenizer comes from the lightweight ``tokenizers`` library and the
    labels are read straight from ``config.json``. Loads a pre-exported
    ``model.onnx`` (no ``optimum`` export) and returns the 🤗
    ``text-classification(top_k=None)`` shape: ``list[str] -> list[list[{label, score}]]``.
    """
    try:
        import json

        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "The 'onnx-local' backend needs torch-free deps. Install with:\n"
            "    pip install onnxruntime tokenizers huggingface_hub numpy"
        ) from exc

    # The torch-only default id has no model.onnx; use the pre-exported mirror.
    repo = model_name if model_name and model_name != "ProsusAI/finbert" else _ONNX_LOCAL_DEFAULT_REPO

    model_path = hf_hub_download(repo, "model.onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}

    # Tokenizer via the Rust `tokenizers` lib (no torch, no transformers).
    tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding()  # pad to the longest text in each batch

    # Labels straight from config.json (avoids transformers.AutoConfig -> torch).
    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as fh:
        raw_id2label = json.load(fh).get("id2label") or {0: "positive", 1: "negative", 2: "neutral"}
    id2label = {int(k): str(v).lower() for k, v in raw_id2label.items()}

    def _run(texts: List[str]) -> List[List[Dict[str, object]]]:
        encodings = tokenizer.encode_batch(list(texts))
        columns = {
            "input_ids": [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
            "token_type_ids": [e.type_ids for e in encodings],
        }
        feeds = {name: np.asarray(col, dtype=np.int64)
                 for name, col in columns.items() if name in input_names}
        logits = session.run(None, feeds)[0]
        # Shared, dependency-free math (softmax + label mapping) — see tests.
        return _logits_to_distributions(logits, id2label)

    return _run


def _load_onnx_model(model_name: str):
    """Load (or export on first use) an ONNX FinBERT model via ``optimum``.

    Note: the first-use export traces the model with **torch** (needs torch ≥ 2.4),
    so this backend does *not* run on x86 macOS. For a torch-free ONNX path use
    ``backend="onnx-local"`` (:func:`_load_onnx_local_classifier`) instead.
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "ONNX backend requires optimum + onnxruntime. Install with:\n"
            "    pip install 'optimum[onnxruntime]'"
        ) from exc

    # If the repo already ships ONNX weights, this is fully torch-free. Otherwise
    # fall back to a one-time export (uses torch only to trace, never torch.load
    # of .bin because we request safetensors), then runs on ONNX Runtime.
    try:
        return ORTModelForSequenceClassification.from_pretrained(model_name)
    except Exception:  # noqa: BLE001 - broad: many optimum/repo variants
        return ORTModelForSequenceClassification.from_pretrained(
            model_name, export=True, use_safetensors=True
        )


def _load_torch_model(model_name: str):
    """Load FinBERT via PyTorch, using safetensors to avoid the torch.load gate."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
