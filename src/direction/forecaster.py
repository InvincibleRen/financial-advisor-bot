"""Per-stock direction forecast (Extension B, Direction 2, ~15%).

Runs *only on the top-N stocks the core selector already chose*, as a short-term
up/down **confirmation** signal — a drill-down on the core's output, not a
competing whole-universe model.

Modelling choice (framework-light, on purpose)
----------------------------------------------
Single-stock daily direction is close to a coin flip, and the target machine is
an **x86 macOS / Intel Mac** where the modern deep-learning stack (PyTorch ≥ 2.4,
etc.) has no working wheel — the same constraint that shaped Extension A. So the
forecaster is built on the **same scikit-learn stack as the core ranker**
(`selection/model.py`) rather than an LSTM/GRU: it is fully portable, readable,
fast, and honest about the weak signal. The `kind` switch keeps it pluggable:

* ``"gbm"``      -> ``HistGradientBoostingClassifier`` (native NaN support),
* ``"logistic"`` -> impute + standardise + ``LogisticRegression`` baseline,
* ``"baseline""`` -> a trivial momentum-persistence rule (no learning), the
  honest floor any learned model must beat.

Leakage safety
--------------
Features at day *t* use only closes up to and including *t* (all rolling / EWMA /
lagged, i.e. causal). The label at *t* is the sign of the **forward** return over
``horizon_days`` (``Close[t+h] > Close[t]``), so the last ``h`` rows have no label.
The walk-forward evaluator (``evaluate.py``) additionally embargoes ``horizon_days``
between the training window and the prediction date, because a training row's
label is only known ``h`` days after its feature date.

Removable by design: ``selection/`` never imports this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.common.indicators import add_adx, add_macd, add_moving_averages

_VALID_KINDS = ("gbm", "logistic", "baseline")

# Longest look-back window used by a feature; rows before it are warm-up (NaN).
_WARMUP = 50


# --------------------------------------------------------------------------- #
# Dataset construction (leakage-safe, shared by the forecaster + evaluator)    #
# --------------------------------------------------------------------------- #

def build_direction_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Causal per-day feature matrix for one stock (index = price dates).

    Every column is computed from closes at or before its row date (lagged
    returns, momentum, volatility, MACD histogram, ADX trend strength, distance
    from SMAs), so a row never sees its own future. Warm-up rows carry NaNs by
    design.
    """
    if "Close" not in prices.columns:
        raise ValueError("prices must contain a 'Close' column.")

    close = prices["Close"].astype("float64")
    daily_ret = close.pct_change()

    feat = pd.DataFrame(index=prices.index)
    for k in (1, 2, 3, 5, 10):
        feat[f"ret_{k}"] = close.pct_change(k)
    feat["mom_10"] = close.pct_change(10)
    feat["mom_20"] = close.pct_change(20)
    feat["vol_10"] = daily_ret.rolling(10).std()
    feat["vol_20"] = daily_ret.rolling(20).std()

    one_col = prices[["Close"]]
    feat["macd_hist"] = add_macd(one_col)["MACD_hist"]

    sma = add_moving_averages(one_col)
    feat["close_sma20"] = close / sma["SMA20"] - 1.0
    feat["close_sma50"] = close / sma["SMA50"] - 1.0

    # ADX (trend strength) is added only when its inputs (High/Low) are present,
    # so a close-only frame keeps the original schema and an *unavailable*
    # indicator is never confused with a warm-up NaN.
    if {"High", "Low"}.issubset(prices.columns):
        feat["adx_14"] = add_adx(prices)["ADX14"]
    return feat


def forward_direction_labels(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Binary up/down label: 1 if ``Close[t+h] > Close[t]`` else 0 (NaN at the tail)."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    close = prices["Close"].astype("float64")
    forward_ret = close.shift(-horizon_days) / close - 1.0
    label = (forward_ret > 0).astype("float64")
    label[forward_ret.isna()] = np.nan  # last h rows have no realised future
    return label.rename("direction")


def build_direction_dataset(prices: pd.DataFrame, horizon_days: int):
    """Return ``(features, labels)`` aligned on the price index."""
    return build_direction_features(prices), forward_direction_labels(prices, horizon_days)


# --------------------------------------------------------------------------- #
# Shared estimator helpers (used by the forecaster and the walk-forward eval)  #
# --------------------------------------------------------------------------- #

def build_estimator(kind: str, random_state: int = 0):
    """Construct the scikit-learn estimator for ``kind`` (imported lazily)."""
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            l2_regularization=1.0,
            random_state=random_state,
        )
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])
    raise ValueError(f"build_estimator does not handle kind={kind!r}")


def prob_up_from_estimator(estimator, X: pd.DataFrame) -> np.ndarray:
    """Extract ``P(up)`` per row from a fitted estimator, handling single-class."""
    proba = estimator.predict_proba(X.to_numpy(dtype="float64"))
    classes = list(estimator.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype="float64")  # only "down" seen in training


def build_fitted_estimator(
    kind: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 0,
    calibrate: bool = False,
    method: str = "sigmoid",
):
    """Fit an estimator on ``(X, y)``, optionally **probability-calibrated**.

    Why calibrate: the raw classifier's ``predict_proba`` ranks well but its
    numbers are often not true frequencies — a displayed "P(up) = 63%" should mean
    roughly 63% of such days actually rise. ``CalibratedClassifierCV`` fixes that by
    fitting the calibration map on **held-out folds of the training window only**
    (Platt/``sigmoid`` by default; ``isotonic`` for larger samples), so it is
    leakage-safe and, being monotonic, leaves AUC essentially unchanged while
    improving Brier score / calibration error.

    Returns ``(estimator, constant_class)``: for a degenerate single-class window
    ``estimator`` is ``None`` and ``constant_class`` carries the only outcome seen.
    """
    classes = pd.unique(y)
    if len(classes) < 2:
        return None, (int(classes[0]) if len(classes) else 0)

    estimator = build_estimator(kind, random_state)
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        # Need at least 2 samples of the minority class per CV fold; cap folds so
        # short training windows still calibrate instead of erroring.
        min_class = int(pd.Series(y).value_counts().min())
        n_folds = max(2, min(3, min_class))
        if min_class >= 2:
            estimator = CalibratedClassifierCV(estimator, method=method, cv=n_folds)
    estimator.fit(X.to_numpy(dtype="float64"), y.to_numpy(dtype="int64"))
    return estimator, None


def baseline_prob_up(features: pd.DataFrame) -> np.ndarray:
    """Momentum-persistence floor: predict 'up' when 10-day momentum is positive."""
    return (features["mom_10"].to_numpy(dtype="float64") > 0).astype("float64")


# --------------------------------------------------------------------------- #
# Forecaster                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class DirectionForecast:
    """One near-term direction call for a single stock."""

    ticker: str
    prob_up: float
    horizon_days: int
    as_of: Optional[pd.Timestamp] = None

    @property
    def direction(self) -> str:
        return "up" if self.prob_up >= 0.5 else "down"


class DirectionForecaster:
    """Predict the near-term up/down direction of a single stock.

    Model-agnostic surface (mirrors ``selection.model.RankerModel``): callers use
    only ``fit`` and ``predict_direction``; changing ``kind`` changes nothing else.
    """

    def __init__(self, kind: str = "gbm", horizon_days: int = 5, random_state: int = 0,
                 calibrate: bool = False, calibration_method: str = "sigmoid") -> None:
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1.")
        self.kind = kind
        self.horizon_days = horizon_days
        self.random_state = random_state
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.ticker: str = ""
        self._estimator: Optional[object] = None
        self._constant_class: Optional[int] = None
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, prices: pd.DataFrame, ticker: str = "") -> "DirectionForecaster":
        """Fit on one stock's price history. ``baseline`` needs no fitting."""
        self.ticker = ticker or self.ticker
        if self.kind == "baseline":
            self._estimator = None
            return self

        X, y = build_direction_dataset(prices, self.horizon_days)
        y = y.dropna()
        X = X.loc[y.index].dropna()          # drop warm-up + unlabeled tail
        y = y.loc[X.index].astype("int64")
        if X.empty:
            raise ValueError("Not enough history to build a direction training set.")

        self._feature_names = list(X.columns)
        self._estimator, self._constant_class = build_fitted_estimator(
            self.kind, X, y, self.random_state,
            calibrate=self.calibrate, method=self.calibration_method,
        )
        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_direction(self, prices: pd.DataFrame, ticker: str = "") -> DirectionForecast:
        """Return the current ``P(up)`` over the next ``horizon_days``."""
        ticker = ticker or self.ticker
        features = build_direction_features(prices).dropna()
        if features.empty:
            raise ValueError("Not enough history to form a feature row for prediction.")
        last = features.iloc[[-1]]
        as_of = last.index[-1]

        if self.kind == "baseline":
            prob = float(baseline_prob_up(last)[0])
        elif self._estimator is None:
            prob = float(self._constant_class if self._constant_class is not None else 0.0)
        else:
            last = last.reindex(columns=self._feature_names)
            prob = float(prob_up_from_estimator(self._estimator, last)[0])

        return DirectionForecast(ticker, prob, self.horizon_days, as_of)
