"""Offline tests for the torch-free ``onnx-local`` sentiment backend.

No model download, no ONNX Runtime, no torch: the scoring *math* (softmax +
label mapping + output shape) lives in a dependency-free helper, and backend
dispatch is checked by monkeypatching the heavy loader. This mirrors the
injectable-classifier discipline in ``test_sentiment.py`` — the real FinBERT
ONNX model only ever runs on the user's machine.
"""
import math

from src.sentiment import finbert
from src.sentiment.finbert import FinBERTScorer, _logits_to_distributions, _softmax_row

_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


# --------------------------------------------------------------------------- #
# Scoring math: softmax + label mapping + output shape                        #
# --------------------------------------------------------------------------- #

def test_softmax_row_normalises_and_orders():
    probs = _softmax_row([2.0, 0.0, -1.0])
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[0] > probs[1] > probs[2]  # order preserved
    # matches a hand-computed reference
    ref = [math.exp(x) for x in (2.0, 0.0, -1.0)]
    ref = [r / sum(ref) for r in ref]
    assert all(abs(a - b) < 1e-12 for a, b in zip(probs, ref))


def test_logits_to_distributions_shape_and_labels():
    dists = _logits_to_distributions([[2.0, 0.0, -1.0], [-1.0, 3.0, 0.0]], _ID2LABEL)
    assert len(dists) == 2
    for row in dists:
        assert [e["label"] for e in row] == ["positive", "negative", "neutral"]
        assert abs(sum(e["score"] for e in row) - 1.0) < 1e-12
    # row 0 leans positive, row 1 leans negative
    assert max(dists[0], key=lambda e: e["score"])["label"] == "positive"
    assert max(dists[1], key=lambda e: e["score"])["label"] == "negative"


def test_scorer_consumes_helper_output_end_to_end():
    """The helper output plugs straight into FinBERTScorer as its classifier."""
    def classifier(texts):
        # strongly-positive logits for every text
        return _logits_to_distributions([[6.0, -6.0, 0.0]] * len(texts), _ID2LABEL)

    score = FinBERTScorer(classifier=classifier).score_texts(["headline"])
    assert score > 0.9  # P(pos) - P(neg) ~ +1


# --------------------------------------------------------------------------- #
# Backend dispatch                                                            #
# --------------------------------------------------------------------------- #

def test_backend_onnx_local_routes_to_torch_free_loader(monkeypatch):
    """``backend='onnx-local'`` must use the torch-free loader, not the pipeline."""
    seen = {}

    def fake_loader(model_name):
        seen["model_name"] = model_name
        return lambda texts: [[{"label": "neutral", "score": 1.0}] for _ in texts]

    monkeypatch.setattr(finbert, "_load_onnx_local_classifier", fake_loader)
    clf = finbert._load_finbert_pipeline("ProsusAI/finbert", backend="onnx-local")
    assert clf(["x", "y"]) == [[{"label": "neutral", "score": 1.0}]] * 2
    assert seen["model_name"] == "ProsusAI/finbert"


def test_scorer_lazy_loads_onnx_local_backend(monkeypatch):
    """A scorer with no injected classifier builds the backend it was asked for."""
    calls = {"n": 0}

    def fake_loader(model_name):
        calls["n"] += 1
        return lambda texts: _logits_to_distributions([[5.0, -5.0, 0.0]] * len(texts), _ID2LABEL)

    monkeypatch.setattr(finbert, "_load_onnx_local_classifier", fake_loader)
    scorer = FinBERTScorer(backend="onnx-local")
    assert scorer.score_texts(["a"]) > 0.9
    scorer.score_texts(["b"])          # second call reuses the loaded backend
    assert calls["n"] == 1             # loaded lazily exactly once
