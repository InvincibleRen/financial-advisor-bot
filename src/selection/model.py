"""Ranking model + baseline (Direction 1 core, Phase 2).

A gradient-boosting classifier predicts ``P(stock beats the universe median)``.
A logistic-regression baseline is kept for comparison so the evaluation can show
the ML model earns its complexity against a simple linear benchmark.

Design notes
------------
* **Model-agnostic surface.** The walk-forward selector in ``select.py`` only
  ever calls ``fit`` and ``predict_scores``; swapping ``kind`` between ``"gbm"``
  and ``"logistic"`` (or adding a new estimator) changes nothing downstream.
* **Ranking, not thresholding.** ``predict_scores`` returns a continuous
  probability, index-aligned to the input rows, so the selector can rank the
  cross-section and take the top-N. We never hard-classify.
* **Missing-value policy.** Features carry NaNs by design (a momentum window that
  has not filled yet, a stock with no filing so far). Gradient boosting handles
  NaNs natively; the logistic pipeline imputes (median) then standardises, since
  a linear model cannot consume NaNs and is scale-sensitive.
* **Optional dependency.** scikit-learn is imported lazily inside ``fit`` so the
  rest of the project still imports without it installed.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

_VALID_KINDS = ("gbm", "logistic", "rf", "xgb")

# Features the gbm ranker excludes: permutation importance showed ``volatility``
# *degrades* the tree's out-of-sample AUC (the tree overfits it), while the linear
# baseline still benefits from it — so the exclusion is gbm-only, applied inside
# ``fit``/``predict`` and reflected in the stored feature names.
_GBM_EXCLUDE = ("volatility",)


@contextlib.contextmanager
def _silence_empty_feature_warning():
    """Suppress the one benign imputer warning about all-NaN feature columns.

    In early walk-forward folds an entire fundamental column can be all-NaN
    (yfinance has no filing that far back). scikit-learn's ``SimpleImputer`` emits
    "Skipping features without any observed values" during *both* fit and
    transform; over a multi-year monthly backtest that is hundreds of identical
    lines. It carries no information the run cares about, so we silence exactly
    that message (nothing else) around estimator calls.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        yield


@dataclass
class RankerModel:
    """Thin, model-agnostic wrapper around a scikit-learn classifier.

    Parameters
    ----------
    kind:
        ``"gbm"`` -> ``HistGradientBoostingClassifier`` (native NaN support);
        ``"logistic"`` -> median-impute + standardise + ``LogisticRegression``;
        ``"rf"`` -> median-impute + ``RandomForestClassifier`` (bagged trees);
        ``"xgb"`` -> ``XGBClassifier`` with row/column subsampling (native NaN).

        All four expose the same ``fit``/``predict_scores`` surface, so the
        walk-forward selector is indifferent to which is used. Chapter 5 compares
        them on identical folds and selects on out-of-sample stability rather than
        on the best full-period return.
    random_state:
        Seed passed to estimators that accept one, for reproducible folds.
    """

    kind: str = "gbm"
    random_state: int = 0
    _estimator: Optional[object] = field(default=None, repr=False)
    _feature_names: Optional[List[str]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")

    # ------------------------------------------------------------------ #
    # Training / scoring                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankerModel":
        """Fit the chosen estimator on ``(features, binary label)``.

        Rows whose label is missing are dropped. If, after cleaning, only a
        single class remains (common in the earliest, tiny walk-forward folds),
        no estimator is fitted — ``predict_scores`` then returns that class as a
        constant, which is the honest degenerate behaviour.
        """
        X, y = self._align_xy(X, y)
        if self.kind == "gbm":
            X = X.drop(columns=[c for c in _GBM_EXCLUDE if c in X.columns])
        self._feature_names = list(X.columns)

        classes = pd.unique(y)
        if len(classes) < 2:
            # Degenerate fold: remember the single constant outcome.
            self._estimator = _ConstantClassifier(int(classes[0]) if len(classes) else 0)
            return self

        self._estimator = self._build_estimator(len(X))
        with _silence_empty_feature_warning():
            self._estimator.fit(X.to_numpy(dtype="float64"), y.to_numpy(dtype="int64"))
        return self

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """Return ``P(label = 1)`` per row, as a Series aligned to ``X.index``."""
        if self._estimator is None:
            raise RuntimeError("RankerModel.predict_scores called before fit().")

        X = X.reindex(columns=self._feature_names)
        with _silence_empty_feature_warning():
            proba = self._estimator.predict_proba(X.to_numpy(dtype="float64"))
        classes = list(self._estimator.classes_)
        if 1 in classes:
            scores = proba[:, classes.index(1)]
        else:  # the only class seen in training was 0
            scores = np.zeros(len(X), dtype="float64")
        return pd.Series(scores, index=X.index, name="score")

    # ------------------------------------------------------------------ #
    # Estimator construction                                             #
    # ------------------------------------------------------------------ #

    def _build_estimator(self, n_samples: int = 0):
        """Construct the underlying scikit-learn estimator (imported lazily)."""
        if self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier

            # Adaptive leaf regularisation: tuned to 60 on the full universe
            # (walk-forward OOS AUC/precision peak), but capped by the training
            # size so early, tiny folds (and unit tests) can still split.
            min_samples_leaf = min(60, max(20, n_samples // 25))

            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=3,
                max_iter=300,
                l2_regularization=1.0,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
            )

        if self.kind == "rf":
            # Bagged trees. Variance reduction by averaging many de-correlated
            # trees suits a low signal-to-noise cross-section, where boosting can
            # chase noise. The randomisation here (bootstrap resampling plus a
            # random feature subset at each split) is intrinsic to the algorithm
            # rather than a hand-tuned knob, which keeps the researcher degrees of
            # freedom low. Random forests cannot consume NaN, so the sparse
            # fundamentals are median-imputed first.
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            return Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(
                        n_estimators=300, max_depth=6,
                        # Same adaptive leaf rule as the booster: 40 on a full
                        # cross-section, but capped by the training size so early,
                        # tiny folds (and unit tests) can still split.
                        min_samples_leaf=min(40, max(5, n_samples // 25)),
                        n_jobs=1, random_state=self.random_state,
                    )),
                ]
            )

        if self.kind == "xgb":
            # Gradient boosting with stochastic regularisation (row and column
            # subsampling), which the evaluation shows is what separates it from
            # the plain histogram booster. Missing values are handled natively, so
            # no imputation is needed. Imported lazily: xgboost is an optional
            # dependency and the package still imports without it.
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=1, random_state=self.random_state,
                eval_metric="logloss", tree_method="hist",
            )

        # Logistic baseline: impute -> scale -> linear model.
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ]
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _align_xy(X: pd.DataFrame, y: pd.Series):
        """Align X and y on their shared index and drop rows with a missing label."""
        if X.empty:
            raise ValueError("Cannot fit RankerModel on an empty feature matrix.")
        y = y.reindex(X.index)
        keep = y.notna()
        return X.loc[keep], y.loc[keep].astype("int64")


class _ConstantClassifier:
    """Minimal stand-in used for single-class folds.

    Mimics the small slice of the scikit-learn API that ``predict_scores`` needs
    (``classes_`` and ``predict_proba``) so degenerate folds need no special-case
    branching upstream.
    """

    def __init__(self, constant_class: int) -> None:
        self.classes_ = np.array([constant_class])

    def predict_proba(self, X) -> np.ndarray:
        return np.ones((len(X), 1), dtype="float64")
