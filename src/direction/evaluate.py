"""Walk-forward evaluation of the per-stock direction forecast (Extension B).

Honest evaluation is the whole point of Direction 2: single-stock direction is
near a coin flip, so the question is not "is accuracy high?" but "does the model
beat the naive base rate (always predict the majority class) out of sample?".

Method
------
For a rolling sequence of prediction dates, train only on **strictly past** rows
and predict the direction at the current date, then compare to the realised
outcome. A ``horizon_days`` **embargo** sits between the training window and the
prediction date: a training row's label is the sign of its *forward* return, so
it is only actually known ``horizon_days`` after its feature date — including
rows inside that gap would leak the future into training.

Metrics
-------
* ``accuracy``  — out-of-sample directional hit rate.
* ``base_rate`` — the naive benchmark: share of the majority actual class (what
  "always guess the common direction" would score). Accuracy must clear this.
* ``auc``       — ROC-AUC of ``P(up)`` vs the actual label (vs 0.5 = chance);
  ``None`` when only one class occurs or the backend gives no gradation.
* ``up_rate``   — share of days the model called "up" (a bias sanity check).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.direction.forecaster import (
    baseline_prob_up,
    build_direction_dataset,
    build_fitted_estimator,
    prob_up_from_estimator,
)


@dataclass
class DirectionEvaluation:
    """Aggregate out-of-sample result for one stock's direction forecast."""

    ticker: str
    kind: str
    horizon_days: int
    n_predictions: int
    accuracy: float
    base_rate: float
    up_rate: float
    auc: Optional[float]
    edge: float                      # accuracy - base_rate (positive = beats naive)
    brier: float = float("nan")      # mean (P(up) - actual)^2; lower = better calibrated + sharper
    ece: float = float("nan")        # expected calibration error (|conf - acc| over bins)
    calibrated: bool = False         # whether probability calibration was applied
    reliability: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    selective: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def evaluate_direction(
    prices: pd.DataFrame,
    ticker: str = "",
    kind: str = "gbm",
    horizon_days: int = 5,
    train_size: int = 252,
    step: int = 5,
    min_train: int = 120,
    random_state: int = 0,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
) -> DirectionEvaluation:
    """Roll a train→predict window across history and score the direction calls.

    When ``calibrate`` is set, each fold's estimator is probability-calibrated on
    held-out folds of its own training window (leakage-safe), so the reported Brier
    score and calibration error reflect the numbers a live user would actually see.
    """
    X, y = build_direction_dataset(prices, horizon_days)
    feat = X.dropna()                       # fully-warmed feature rows only
    y = y.reindex(feat.index)
    dates = list(feat.index)
    n = len(dates)

    rows = []
    for i in range(min_train, n, step):
        pred_date = dates[i]
        actual = y.iloc[i]
        if pd.isna(actual):                 # unlabeled tail — no realised outcome
            continue

        x_now = feat.iloc[[i]]
        if kind == "baseline":
            prob = float(baseline_prob_up(x_now)[0])
        else:
            # Embargo `horizon_days`: training labels must be known by pred_date.
            hi = i - horizon_days
            lo = max(0, hi - train_size)
            if hi - lo < min_train:
                continue
            X_tr = feat.iloc[lo:hi]
            y_tr = y.iloc[lo:hi].dropna()
            X_tr = X_tr.loc[y_tr.index]
            est, const = build_fitted_estimator(
                kind, X_tr, y_tr, random_state,
                calibrate=calibrate, method=calibration_method,
            )
            if est is None:                 # degenerate window -> constant call
                prob = float(const) if const is not None else 0.5
            else:
                prob = float(prob_up_from_estimator(est, x_now)[0])

        rows.append((pred_date, prob, int(prob >= 0.5), int(actual)))

    predictions = pd.DataFrame(rows, columns=["date", "prob_up", "pred", "actual"])
    if not predictions.empty:
        predictions = predictions.set_index("date")
    return _summarise(ticker, kind, horizon_days, predictions, calibrate)


def _summarise(ticker, kind, horizon_days, predictions: pd.DataFrame,
               calibrated: bool = False) -> DirectionEvaluation:
    if predictions.empty:
        return DirectionEvaluation(ticker, kind, horizon_days, 0,
                                   float("nan"), float("nan"), float("nan"), None,
                                   float("nan"), calibrated=calibrated)

    actual = predictions["actual"]
    pred = predictions["pred"]
    prob = predictions["prob_up"]
    accuracy = float((pred == actual).mean())
    up_share = float(actual.mean())
    base_rate = float(max(up_share, 1.0 - up_share))
    up_rate = float(pred.mean())
    auc = _safe_auc(actual, prob)
    return DirectionEvaluation(
        ticker=ticker, kind=kind, horizon_days=horizon_days,
        n_predictions=int(len(predictions)), accuracy=accuracy,
        base_rate=base_rate, up_rate=up_rate, auc=auc,
        edge=accuracy - base_rate,
        brier=brier_score(actual, prob),
        ece=expected_calibration_error(actual, prob),
        calibrated=calibrated,
        reliability=reliability_table(actual, prob),
        selective=selective_accuracy_table(predictions),
        predictions=predictions,
    )


# --------------------------------------------------------------------------- #
# Calibration quality (how trustworthy is the displayed probability)          #
# --------------------------------------------------------------------------- #

def brier_score(actual: pd.Series, prob_up: pd.Series) -> float:
    """Mean squared error of the probability: ``mean((P(up) - actual)^2)``.

    Lower is better. Unlike accuracy/AUC it rewards *calibration and sharpness*
    together — a probability that is both correct on average and confident when it
    should be. This is the headline number for a displayed "涨跌概率".
    """
    a = actual.to_numpy(dtype="float64")
    p = prob_up.to_numpy(dtype="float64")
    return float(np.mean((p - a) ** 2)) if len(a) else float("nan")


def reliability_table(actual: pd.Series, prob_up: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions by confidence and compare mean predicted vs realised freq.

    Columns: ``n`` (count in bin), ``mean_pred`` (average P(up)), ``frac_up``
    (actual up-rate). A well-calibrated model has ``mean_pred ≈ frac_up`` in every
    populated bin — the diagonal of a reliability diagram.
    """
    a = actual.to_numpy(dtype="float64")
    p = prob_up.to_numpy(dtype="float64")
    if len(a) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "frac_up"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})", "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()), "frac_up": float(a[m].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(actual: pd.Series, prob_up: pd.Series, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean ``|mean_pred - frac_up|`` over bins.

    0 = perfectly calibrated. This is what calibration is designed to shrink; note
    it can improve even when accuracy/AUC do not, because calibration only rescales
    the probabilities monotonically.
    """
    table = reliability_table(actual, prob_up, n_bins)
    if table.empty:
        return float("nan")
    w = table["n"] / table["n"].sum()
    return float((w * (table["mean_pred"] - table["frac_up"]).abs()).sum())


def selective_accuracy_table(
    predictions: pd.DataFrame,
    coverages=(1.0, 0.8, 0.6, 0.4, 0.2),
) -> pd.DataFrame:
    """Directional accuracy when only the most confident calls are kept.

    Confidence is ``|P(up) - 0.5|``: keeping the top ``coverage`` fraction of the
    most confident predictions and abstaining on the rest is *selective
    prediction*. Because the model is allowed to say "not sure", accuracy on the
    calls it does make can clear the base rate even when the all-in accuracy does
    not — the honest way to show that a near-coin-flip signal is still usable when
    it is confident.

    Expects columns ``prob_up`` and ``actual``. Returns one row per coverage with
    ``coverage``, ``n`` (calls kept), ``accuracy`` and ``edge_vs_base``
    (accuracy minus the majority-class base rate of the full set).
    """
    cols = ["coverage", "n", "accuracy", "edge_vs_base"]
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=cols)

    prob = predictions["prob_up"].to_numpy(dtype="float64")
    actual = predictions["actual"].to_numpy(dtype="float64")
    up_share = actual.mean()
    base_rate = max(up_share, 1.0 - up_share)

    order = np.argsort(-np.abs(prob - 0.5))   # most confident first
    prob, actual = prob[order], actual[order]
    n_total = len(actual)

    rows = []
    for coverage in coverages:
        k = max(1, int(round(n_total * coverage)))
        pred_k = (prob[:k] >= 0.5).astype("float64")
        accuracy = float((pred_k == actual[:k]).mean())
        rows.append({"coverage": float(coverage), "n": int(k),
                     "accuracy": accuracy, "edge_vs_base": accuracy - base_rate})
    return pd.DataFrame(rows, columns=cols)


def _safe_auc(actual: pd.Series, prob_up: pd.Series) -> Optional[float]:
    if actual.nunique() < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(actual.to_numpy(), prob_up.to_numpy()))
    except Exception:  # noqa: BLE001 - AUC is a nice-to-have, never fatal
        return None
