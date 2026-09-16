"""Offline tests for Extension B (per-stock direction forecast).

Deterministic, no network, no model download: synthetic price series exercise
the leakage-safe features, the forward-direction labels, the forecaster's
fit/predict, and the honest walk-forward evaluation (a learnable pattern beats
the base rate; pure noise does not).
"""
import numpy as np
import pandas as pd

from src.direction.evaluate import (
    brier_score,
    evaluate_direction,
    expected_calibration_error,
    reliability_table,
)
from src.direction.forecaster import (
    DirectionForecast,
    DirectionForecaster,
    build_direction_features,
    forward_direction_labels,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _noise_prices(n=800, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    return pd.DataFrame({"Close": close}, index=pd.bdate_range("2016-01-01", periods=n))


def _sine_prices(n=1000, period=40, seed=1):
    t = np.arange(n)
    close = 100 + 12 * np.sin(2 * np.pi * t / period) + 0.05 * np.random.default_rng(seed).normal(0, 1, n)
    return pd.DataFrame({"Close": close}, index=pd.bdate_range("2016-01-01", periods=n))


# --------------------------------------------------------------------------- #
# Features + labels: leakage safety and correctness                           #
# --------------------------------------------------------------------------- #

def test_features_are_causal_no_lookahead():
    prices = _noise_prices(300)
    feats = build_direction_features(prices)
    mutated = prices.copy()
    mutated.iloc[-1, 0] *= 5.0                       # change only the final close
    feats_mutated = build_direction_features(mutated)
    # Every row before the last must be untouched by a future price change.
    pd.testing.assert_frame_equal(feats.iloc[:-1], feats_mutated.iloc[:-1])


def test_forward_labels_match_definition():
    close = pd.Series([10, 11, 12, 9, 9, 13], dtype="float64",
                      index=pd.bdate_range("2020-01-01", periods=6))
    prices = pd.DataFrame({"Close": close})
    y = forward_direction_labels(prices, horizon_days=1)
    # up, up, down, flat(=down), up, then NaN (no future for the last row)
    assert list(y.iloc[:5]) == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert np.isnan(y.iloc[-1])


def test_horizon_labels_have_nan_tail():
    y = forward_direction_labels(_noise_prices(100), horizon_days=5)
    assert y.iloc[-5:].isna().all()
    assert y.iloc[:-5].notna().all()


# --------------------------------------------------------------------------- #
# Forecaster: fit / predict contract                                          #
# --------------------------------------------------------------------------- #

def test_forecaster_predicts_probability_in_unit_interval():
    prices = _noise_prices(400)
    fc = DirectionForecaster(kind="gbm", horizon_days=5).fit(prices, "TEST")
    out = fc.predict_direction(prices, "TEST")
    assert isinstance(out, DirectionForecast)
    assert 0.0 <= out.prob_up <= 1.0
    assert out.direction in ("up", "down")
    assert out.horizon_days == 5
    assert out.as_of == prices.index[-1]


def test_invalid_kind_rejected():
    for bad in ("lstm", "gru", "xgboost"):
        try:
            DirectionForecaster(kind=bad)
            assert False, f"expected ValueError for kind={bad!r}"
        except ValueError:
            pass


def test_baseline_follows_momentum_sign():
    # Steadily rising series -> positive 10-day momentum -> baseline calls "up".
    up = pd.DataFrame({"Close": np.linspace(100, 200, 300)},
                      index=pd.bdate_range("2016-01-01", periods=300))
    fc = DirectionForecaster(kind="baseline", horizon_days=5).fit(up, "UP")
    assert fc.predict_direction(up, "UP").direction == "up"


# --------------------------------------------------------------------------- #
# Walk-forward evaluation: honest signal detection                            #
# --------------------------------------------------------------------------- #

def test_learnable_pattern_beats_base_rate():
    ev = evaluate_direction(_sine_prices(600), "SINE", kind="gbm",
                            horizon_days=5, train_size=200, step=6)
    assert ev.n_predictions > 30
    assert 0.45 <= ev.base_rate <= 0.75          # roughly balanced classes
    assert ev.edge > 0.2                          # model clearly beats naive
    assert ev.auc is not None and ev.auc > 0.9


def test_noise_has_no_meaningful_edge():
    ev = evaluate_direction(_noise_prices(500), "NOISE", kind="gbm",
                            horizon_days=5, train_size=200, step=8)
    assert ev.n_predictions > 20
    assert abs(ev.edge) < 0.15                     # accuracy ~ base rate
    assert 0.0 <= ev.accuracy <= 1.0
    assert ev.base_rate >= 0.5


def test_evaluation_metrics_are_well_formed():
    ev = evaluate_direction(_sine_prices(500), "SINE", kind="logistic",
                            horizon_days=5, train_size=200, step=8)
    cols = set(ev.predictions.columns)
    assert {"prob_up", "pred", "actual"} <= cols
    assert ev.predictions["pred"].isin([0, 1]).all()
    assert (ev.predictions["prob_up"].between(0.0, 1.0)).all()
    assert abs(ev.edge - (ev.accuracy - ev.base_rate)) < 1e-9


# --------------------------------------------------------------------------- #
# Probability calibration (trustworthy "P(up)" for the live display)          #
# --------------------------------------------------------------------------- #

def test_brier_score_matches_definition():
    actual = pd.Series([1, 0, 1, 0])
    prob = pd.Series([0.8, 0.2, 0.6, 0.4])
    # mean of (0.2^2, 0.2^2, 0.4^2, 0.4^2) = (0.04+0.04+0.16+0.16)/4 = 0.10
    assert abs(brier_score(actual, prob) - 0.10) < 1e-9


def test_reliability_table_and_ece():
    # Two clear confidence bins; predictions systematically over-confident.
    actual = pd.Series([1, 1, 0, 0, 0, 0])
    prob = pd.Series([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    table = reliability_table(actual, prob, n_bins=10)
    # High-confidence bin (0.9): mean_pred 0.9 but only 2/3 actually up.
    hi = table[table["bin"] == "[0.9,1.0)"].iloc[0]
    assert hi["n"] == 3 and abs(hi["mean_pred"] - 0.9) < 1e-9 and abs(hi["frac_up"] - 2 / 3) < 1e-9
    # ECE is the count-weighted gap; strictly positive here (mis-calibrated).
    assert expected_calibration_error(actual, prob) > 0.0


def test_calibrated_forecaster_produces_valid_probability():
    prices = _sine_prices(600)
    fc = DirectionForecaster(kind="gbm", horizon_days=5, calibrate=True).fit(prices, "SINE")
    out = fc.predict_direction(prices, "SINE")
    assert 0.0 <= out.prob_up <= 1.0


def test_calibrated_evaluation_reports_calibration_metrics():
    ev = evaluate_direction(_sine_prices(600), "SINE", kind="logistic", horizon_days=5,
                            train_size=200, step=8, calibrate=True)
    assert ev.calibrated is True
    assert 0.0 <= ev.brier <= 1.0
    assert 0.0 <= ev.ece <= 1.0
    assert not ev.reliability.empty


# --------------------------------------------------------------------------- #
# Removable by design: the core must not depend on this extension             #
# --------------------------------------------------------------------------- #

def test_core_selection_does_not_import_direction():
    """Deleting `direction/` must leave the core selector working: it never
    imports the extension. Check the selection package source for such imports."""
    import pathlib
    import re

    import src.selection as selection

    pkg_dir = pathlib.Path(selection.__file__).parent
    import_pat = re.compile(r"^\s*(from|import)\s+src\.direction", re.MULTILINE)
    for path in pkg_dir.glob("*.py"):
        assert not import_pat.search(path.read_text(encoding="utf-8")), \
            f"{path.name} imports src.direction — breaks removability"
