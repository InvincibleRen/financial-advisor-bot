"""Offline tests for the pooled cross-stock direction forecast (Extension B).

Deterministic, no network, no model download. Synthetic multi-stock panels
exercise the pooled panel builder, the selective-prediction metric, the pooled
walk-forward evaluation (a learnable pattern beats pure noise), and the pooled
forecaster's fit/predict surface.
"""
import numpy as np
import pandas as pd

from src.direction.evaluate import selective_accuracy_table
from src.direction.pooled import (
    PooledDirectionForecaster,
    build_direction_panel,
    evaluate_pooled_direction,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _sine_prices(n=1300, period=40, phase=0.0, seed=1):
    t = np.arange(n)
    close = 100 + 12 * np.sin(2 * np.pi * t / period + phase)
    close = close + 0.05 * np.random.default_rng(seed).normal(0, 1, n)
    return pd.DataFrame({"Close": close}, index=pd.bdate_range("2015-01-01", periods=n))


def _noise_prices(n=1300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    return pd.DataFrame({"Close": close}, index=pd.bdate_range("2015-01-01", periods=n))


def _sine_basket(k=6):
    return {f"S{i}": _sine_prices(phase=i * 0.7, seed=i) for i in range(k)}


def _noise_basket(k=6):
    return {f"N{i}": _noise_prices(seed=100 + i) for i in range(k)}


# --------------------------------------------------------------------------- #
# Panel construction + leakage safety                                          #
# --------------------------------------------------------------------------- #

def test_panel_has_multiindex_and_label():
    panel = build_direction_panel(_sine_basket(3), horizon_days=5)
    assert list(panel.index.names) == ["date", "ticker"]
    assert "label" in panel.columns
    assert panel.index.get_level_values("ticker").nunique() == 3


def test_panel_label_tail_is_unlabeled():
    # Each stock's final `horizon` rows have no realised forward return -> NaN.
    horizon = 5
    panel = build_direction_panel({"S0": _sine_prices(n=300)}, horizon_days=horizon)
    labels = panel.xs("S0", level="ticker")["label"]
    assert labels.iloc[-horizon:].isna().all()
    assert labels.iloc[:-horizon].notna().any()


def test_panel_labels_are_binary():
    panel = build_direction_panel(_sine_basket(2), horizon_days=5)
    values = set(panel["label"].dropna().unique().tolist())
    assert values.issubset({0.0, 1.0})


# --------------------------------------------------------------------------- #
# Selective-prediction metric                                                  #
# --------------------------------------------------------------------------- #

def test_selective_accuracy_empty_returns_empty():
    out = selective_accuracy_table(pd.DataFrame(columns=["prob_up", "actual"]))
    assert out.empty
    assert list(out.columns) == ["coverage", "n", "accuracy", "edge_vs_base"]


def test_selective_accuracy_confident_calls_are_more_accurate():
    # Confident, correct calls at the extremes; wrong, unconfident calls near 0.5.
    prob = [0.95, 0.05, 0.93, 0.07, 0.52, 0.48, 0.51, 0.49]
    actual = [1, 0, 1, 0, 0, 1, 0, 1]  # the four confident ones are right, the rest wrong
    df = pd.DataFrame({"prob_up": prob, "actual": actual})
    table = selective_accuracy_table(df, coverages=(1.0, 0.5))
    acc_full = table.loc[table["coverage"] == 1.0, "accuracy"].iloc[0]
    acc_half = table.loc[table["coverage"] == 0.5, "accuracy"].iloc[0]
    assert acc_half > acc_full
    assert acc_half == 1.0


def test_selective_n_decreases_with_coverage():
    df = pd.DataFrame({"prob_up": np.linspace(0.01, 0.99, 100),
                       "actual": np.random.default_rng(0).integers(0, 2, 100)})
    table = selective_accuracy_table(df, coverages=(1.0, 0.6, 0.2))
    assert list(table["n"]) == sorted(table["n"], reverse=True)


# --------------------------------------------------------------------------- #
# Pooled walk-forward evaluation                                               #
# --------------------------------------------------------------------------- #

def _run(basket, calibrate=True, **kw):
    return evaluate_pooled_direction(
        basket, kind="gbm", horizon_days=5, step=15, train_years=2,
        start="2016-06-01", calibrate=calibrate, **kw,
    )


def test_pooled_evaluation_metrics_are_valid():
    ev = _run(_sine_basket(5))
    assert ev.n_predictions > 0
    assert ev.n_stocks == 5
    assert 0.5 <= ev.base_rate <= 1.0
    assert 0.0 <= ev.brier <= 1.0
    assert ev.ece >= 0.0
    assert ev.auc is None or 0.0 <= ev.auc <= 1.0
    assert not ev.selective.empty


def test_pooled_learnable_beats_noise():
    # A clearly periodic signal must carry more directional information (higher
    # AUC) than a pure random walk, which should sit near chance.
    auc_sine = _run(_sine_basket(6)).auc
    auc_noise = _run(_noise_basket(6)).auc
    assert auc_sine is not None and auc_noise is not None
    assert auc_sine > 0.5
    assert auc_sine > auc_noise


def test_uncalibrated_path_also_runs():
    ev = _run(_sine_basket(3), calibrate=False)
    assert ev.calibrated is False
    assert ev.n_predictions > 0


# --------------------------------------------------------------------------- #
# Pooled forecaster (live current call)                                        #
# --------------------------------------------------------------------------- #

def test_pooled_forecaster_predicts_probability():
    basket = _sine_basket(5)
    forecaster = PooledDirectionForecaster(kind="gbm", horizon_days=5, train_years=2).fit(basket)
    prob = forecaster.predict_prob_up(basket["S0"])
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
