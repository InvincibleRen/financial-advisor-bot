"""Offline tests for the Phase 2 ranker / baseline (`selection/model.py`).

Small, deterministic, separable data. The contract under test is narrow and
model-agnostic: fit on (features, binary label), then return a per-row
probability, index-aligned, that *ranks* positives above negatives — plus honest
behaviour on NaNs and degenerate single-class folds.
"""
import numpy as np
import pandas as pd
import pytest

from src.selection.model import RankerModel


def _separable(n_per_class=40, with_nan=False):
    """A cleanly separable 2-feature dataset: label follows feature ``a``."""
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(-2, 0.5, n_per_class), rng.normal(2, 0.5, n_per_class)])
    b = rng.normal(0, 1, 2 * n_per_class)  # noise feature
    y = np.array([0] * n_per_class + [1] * n_per_class)
    X = pd.DataFrame({"a": a, "b": b})
    if with_nan:
        X.loc[0, "a"] = np.nan  # a real missing value the model must tolerate
    return X, pd.Series(y, name="label")


@pytest.mark.parametrize("kind", ["gbm", "logistic", "rf", "xgb"])
def test_scores_rank_positives_above_negatives(kind):
    X, y = _separable()
    model = RankerModel(kind=kind, random_state=0).fit(X, y)
    scores = model.predict_scores(X)

    assert list(scores.index) == list(X.index)   # index-aligned
    assert scores.between(0.0, 1.0).all()          # valid probabilities
    # Mean score on the true-positive half exceeds the true-negative half.
    assert scores[y == 1].mean() > scores[y == 0].mean()


@pytest.mark.parametrize("kind", ["gbm", "logistic", "rf", "xgb"])
def test_handles_missing_values(kind):
    X, y = _separable(with_nan=True)
    scores = RankerModel(kind=kind).fit(X, y).predict_scores(X)
    assert not scores.isna().any()


def test_single_class_fold_returns_constant():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
    all_ones = RankerModel(kind="gbm").fit(X, pd.Series([1, 1, 1])).predict_scores(X)
    all_zeros = RankerModel(kind="gbm").fit(X, pd.Series([0, 0, 0])).predict_scores(X)
    assert (all_ones == 1.0).all()
    assert (all_zeros == 0.0).all()


def test_predict_is_column_order_invariant():
    X, y = _separable()
    model = RankerModel(kind="logistic").fit(X, y)
    reordered = X[["b", "a"]]  # same data, columns swapped
    scores = model.predict_scores(reordered)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_missing_labels_are_dropped_in_training():
    X, y = _separable(n_per_class=20)
    y = y.astype("float64")
    y.iloc[:5] = np.nan  # unlabelled rows must be ignored, not crash
    scores = RankerModel(kind="gbm").fit(X, y).predict_scores(X)
    assert len(scores) == len(X)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RankerModel().predict_scores(pd.DataFrame({"a": [1.0]}))


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        RankerModel(kind="banana")


def test_all_nan_column_does_not_emit_imputer_warning(recwarn):
    # An all-NaN feature column (e.g. fundamentals with no history yet) must not
    # flood the console: the benign "Skipping features..." imputer warning is
    # silenced during BOTH fit and predict.
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "empty": [np.nan] * 4})
    y = pd.Series([0, 0, 1, 1])
    model = RankerModel(kind="logistic").fit(X, y)
    model.predict_scores(X)
    assert not any("Skipping features" in str(w.message) for w in recwarn.list)


# --------------------------------------------------------------------------- #
# Model-agnostic surface across every ranker                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["gbm", "logistic", "rf", "xgb"])
def test_every_kind_exposes_the_same_surface(kind):
    """Swapping the ranker must change nothing a caller can observe."""
    X, y = _separable()
    scores = RankerModel(kind=kind, random_state=0).fit(X, y).predict_scores(X)

    assert isinstance(scores, pd.Series)
    assert scores.name == "score"
    assert list(scores.index) == list(X.index)
    assert scores.between(0.0, 1.0).all()


@pytest.mark.parametrize("kind", ["rf", "xgb"])
def test_new_kinds_are_column_order_invariant(kind):
    X, y = _separable()
    model = RankerModel(kind=kind, random_state=0).fit(X, y)
    reordered = X[list(reversed(X.columns))]
    pd.testing.assert_series_equal(model.predict_scores(X), model.predict_scores(reordered))
