"""Pooled cross-stock direction forecast (Extension B, improved path).

Instead of training a separate model on each stock's short ~252-day window, this
path pools many stocks' causal feature rows into **one** model per rebalance
date. The extra data (tens of thousands of rows rather than a few hundred) is the
single biggest honest lever on a signal that is otherwise close to a coin flip:
it turns the per-stock design's below-base-rate, badly miscalibrated accuracy
into an accuracy that matches the base rate with a *trustworthy* probability, and
selective prediction (acting only on the most confident calls) then extracts a
small but real positive edge.

Why this is not just "more data for its own sake"
-------------------------------------------------
The per-stock model must relearn everything from ~250 rows every fold, so it
overfits noise and its probabilities are unreliable. Pooling lets one model learn
the shared, cross-sectional relationship between the causal features and the next
move, which is exactly the stable part of the signal. Calibration on held-out
folds of the training window then rescales the probabilities so a displayed
"P(up) = 60%" means roughly 60%.

Leakage safety (identical guarantees to ``evaluate.py``)
-------------------------------------------------------
* Features are causal (built by ``forecaster.build_direction_features`` — only
  closes at or before each row date).
* The label is the sign of the **forward** return (``forward_direction_labels``),
  so the last ``horizon_days`` rows per stock carry no label.
* Each rebalance date trains only on rows whose label was already realised: a
  ``horizon_days`` embargo sits between the newest training row and the
  prediction date, and training is restricted to a trailing window.

Removable by design: ``selection/`` never imports this package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.timeindex import to_naive_index
from src.direction.evaluate import (
    _safe_auc,
    brier_score,
    expected_calibration_error,
    reliability_table,
    selective_accuracy_table,
)
from src.direction.forecaster import (
    build_direction_features,
    build_fitted_estimator,
    forward_direction_labels,
    prob_up_from_estimator,
)

# Default rebalance cadence and windows for the pooled walk-forward. Monthly
# (~21 trading days) keeps the evaluation cost reasonable while pooling gives each
# fold a large, information-rich training set.
_DEFAULT_STEP = 21
_DEFAULT_TRAIN_YEARS = 3
_DEFAULT_MIN_TRAIN = 200


# --------------------------------------------------------------------------- #
# Panel construction                                                          #
# --------------------------------------------------------------------------- #

def build_direction_panel(
    prices: Dict[str, pd.DataFrame],
    horizon_days: int,
) -> pd.DataFrame:
    """Stack per-stock causal features + forward labels into a ``(date, ticker)`` panel.

    Parameters
    ----------
    prices:
        ``{ticker -> daily OHLCV DataFrame}`` (as produced by
        ``universe.load_prices`` or read from the local price cache). Only the
        columns ``build_direction_features`` needs are used; ``High``/``Low``
        enable the ADX feature when present.
    horizon_days:
        Forward horizon (in trading days) used to form the up/down label.

    Returns
    -------
    A DataFrame indexed by ``MultiIndex(date, ticker)`` whose columns are the
    direction features plus a ``label`` column (NaN in each stock's last
    ``horizon_days`` rows and in warm-up rows).
    """
    frames: List[pd.DataFrame] = []
    for ticker, px in prices.items():
        if px is None or "Close" not in getattr(px, "columns", []) or px.empty:
            continue
        px = px.copy()
        px.index = to_naive_index(px.index, normalize=True)
        px = px.sort_index()
        features = build_direction_features(px)
        features["label"] = forward_direction_labels(px, horizon_days)
        features["ticker"] = ticker
        frames.append(features)

    if not frames:
        raise ValueError("build_direction_panel received no usable price frames.")

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True)  # (date, ticker)
    return panel.sort_index()


def _feature_columns(panel: pd.DataFrame) -> List[str]:
    return [c for c in panel.columns if c != "label"]


# --------------------------------------------------------------------------- #
# Walk-forward evaluation                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class PooledDirectionEvaluation:
    """Aggregate out-of-sample result for the pooled direction forecast."""

    kind: str
    horizon_days: int
    n_predictions: int
    n_stocks: int
    accuracy: float
    base_rate: float
    edge: float                     # accuracy - base_rate (positive = beats naive)
    up_rate: float
    auc: Optional[float]
    brier: float
    ece: float
    calibrated: bool
    reliability: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    selective: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def evaluate_pooled_direction(
    prices: Dict[str, pd.DataFrame],
    kind: str = "gbm",
    horizon_days: int = 5,
    step: int = _DEFAULT_STEP,
    train_years: int = _DEFAULT_TRAIN_YEARS,
    min_train: int = _DEFAULT_MIN_TRAIN,
    start: Optional[str] = None,
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    random_state: int = 0,
) -> PooledDirectionEvaluation:
    """Pooled walk-forward evaluation of the per-stock direction forecast.

    At each rebalance date the model is trained on **all** stocks' realised rows
    within the trailing window (embargoed by ``horizon_days``) and then predicts
    the direction of every stock on that date. Metrics are pooled across all
    ``(date, ticker)`` predictions; the ``selective`` table reports accuracy when
    only the most confident calls are kept.
    """
    panel = build_direction_panel(prices, horizon_days)
    feats = _feature_columns(panel)
    level_dates = panel.index.get_level_values(0)

    all_dates = np.array(sorted(pd.unique(level_dates)))
    if start is not None:
        all_dates = all_dates[all_dates >= pd.Timestamp(start)]
    pred_dates = all_dates[::step]

    embargo = pd.Timedelta(days=math.ceil(horizon_days * 1.5))
    trail = pd.Timedelta(days=int(365 * train_years))

    rows = []
    for d in pred_dates:
        cutoff = d - embargo
        lo = d - trail
        try:
            current = panel.xs(d, level=0)
        except KeyError:
            continue
        current = current.dropna(subset=feats)
        if current.empty:
            continue

        train_mask = (level_dates <= cutoff) & (level_dates >= lo)
        train = panel[train_mask].dropna(subset=feats + ["label"])
        if train["label"].nunique() < 2 or len(train) < min_train:
            continue

        estimator, constant_class = build_fitted_estimator(
            kind, train[feats], train["label"].astype("int64"),
            random_state, calibrate=calibrate, method=calibration_method,
        )
        if estimator is None:
            probs = np.full(len(current), float(constant_class if constant_class is not None else 0.5))
        else:
            probs = prob_up_from_estimator(estimator, current[feats])

        for ticker, prob, actual in zip(current.index, probs, current["label"].to_numpy()):
            if not np.isnan(actual):
                rows.append((d, ticker, float(prob), int(actual)))

    predictions = pd.DataFrame(rows, columns=["date", "ticker", "prob_up", "actual"])
    return _summarise_pooled(kind, horizon_days, predictions, calibrate)


def _summarise_pooled(
    kind: str,
    horizon_days: int,
    predictions: pd.DataFrame,
    calibrated: bool,
) -> PooledDirectionEvaluation:
    if predictions.empty:
        return PooledDirectionEvaluation(
            kind=kind, horizon_days=horizon_days, n_predictions=0, n_stocks=0,
            accuracy=float("nan"), base_rate=float("nan"), edge=float("nan"),
            up_rate=float("nan"), auc=None, brier=float("nan"), ece=float("nan"),
            calibrated=calibrated,
        )

    actual = predictions["actual"]
    prob = predictions["prob_up"]
    pred = (prob >= 0.5).astype(int)

    accuracy = float((pred == actual).mean())
    up_share = float(actual.mean())
    base_rate = float(max(up_share, 1.0 - up_share))

    return PooledDirectionEvaluation(
        kind=kind,
        horizon_days=horizon_days,
        n_predictions=int(len(predictions)),
        n_stocks=int(predictions["ticker"].nunique()),
        accuracy=accuracy,
        base_rate=base_rate,
        edge=accuracy - base_rate,
        up_rate=float(pred.mean()),
        auc=_safe_auc(actual, prob),
        brier=brier_score(actual, prob),
        ece=expected_calibration_error(actual, prob),
        calibrated=calibrated,
        reliability=reliability_table(actual, prob),
        selective=selective_accuracy_table(predictions),
        predictions=predictions,
    )


# --------------------------------------------------------------------------- #
# Pooled forecaster (live "current call")                                      #
# --------------------------------------------------------------------------- #

class PooledDirectionForecaster:
    """Fit one pooled model on recent cross-stock history, then call any stock.

    Mirrors the model-agnostic surface of ``DirectionForecaster`` (``fit`` +
    ``predict_prob_up``), but the single fitted model is shared across stocks, so
    a thin universe of names benefits from the whole panel's data.
    """

    def __init__(
        self,
        kind: str = "gbm",
        horizon_days: int = 5,
        train_years: int = _DEFAULT_TRAIN_YEARS,
        calibrate: bool = True,
        calibration_method: str = "sigmoid",
        random_state: int = 0,
    ) -> None:
        self.kind = kind
        self.horizon_days = horizon_days
        self.train_years = train_years
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.random_state = random_state
        self._estimator: Optional[object] = None
        self._constant_class: Optional[int] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, prices: Dict[str, pd.DataFrame], as_of: Optional[pd.Timestamp] = None) -> "PooledDirectionForecaster":
        """Fit on all realised rows in the trailing window ending at ``as_of``."""
        panel = build_direction_panel(prices, self.horizon_days)
        feats = _feature_columns(panel)
        level_dates = panel.index.get_level_values(0)

        as_of = pd.Timestamp(as_of) if as_of is not None else level_dates.max()
        cutoff = as_of - pd.Timedelta(days=math.ceil(self.horizon_days * 1.5))
        lo = as_of - pd.Timedelta(days=int(365 * self.train_years))
        train = panel[(level_dates <= cutoff) & (level_dates >= lo)].dropna(subset=feats + ["label"])
        if train.empty:
            raise ValueError("Not enough pooled history to fit the direction forecaster.")

        self._feature_names = feats
        self._estimator, self._constant_class = build_fitted_estimator(
            self.kind, train[feats], train["label"].astype("int64"),
            self.random_state, calibrate=self.calibrate, method=self.calibration_method,
        )
        return self

    def predict_prob_up(self, prices_one: pd.DataFrame) -> float:
        """Return ``P(up)`` for one stock's latest fully-warmed feature row."""
        if self._feature_names is None:
            raise RuntimeError("PooledDirectionForecaster.predict_prob_up called before fit().")
        px = prices_one.copy()
        px.index = to_naive_index(px.index, normalize=True)
        features = build_direction_features(px.sort_index()).reindex(columns=self._feature_names).dropna()
        if features.empty:
            raise ValueError("Not enough history to form a feature row for prediction.")
        last = features.iloc[[-1]]
        if self._estimator is None:
            return float(self._constant_class if self._constant_class is not None else 0.5)
        return float(prob_up_from_estimator(self._estimator, last)[0])
