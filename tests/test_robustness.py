"""Offline tests for the robustness / self-honesty statistics.

Deterministic synthetic return series: a genuinely positive-mean series must come
out significant (high DSR, CI above zero); pure noise must not. Also checks the
building blocks (PSR bounds, the best-of-N benchmark growing with the trial count,
bootstrap intervals bracketing the point estimate).
"""
import numpy as np

from src.selection.robustness import (
    bootstrap_return_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_per_period,
)


def _good(seed=0, n=120):
    return np.random.default_rng(seed).normal(0.02, 0.05, n)   # positive mean


def _noise(seed=1, n=120):
    return np.random.default_rng(seed).normal(0.0, 0.05, n)    # zero mean


def test_sharpe_per_period_sign_and_zero():
    assert sharpe_per_period(_good()) > 0
    assert sharpe_per_period([0.01, 0.01, 0.01]) == 0.0        # zero variance -> 0


def test_psr_is_a_probability_and_higher_for_stronger_series():
    p_good = probabilistic_sharpe_ratio(_good())
    p_noise = probabilistic_sharpe_ratio(_noise())
    assert 0.0 <= p_good <= 1.0 and 0.0 <= p_noise <= 1.0
    assert p_good > 0.95            # clearly positive Sharpe
    assert p_good > p_noise


def test_expected_max_sharpe_grows_with_trials():
    v = 0.04
    assert expected_max_sharpe(v, 2) < expected_max_sharpe(v, 20) < expected_max_sharpe(v, 200)
    assert expected_max_sharpe(v, 1) == 0.0        # a single trial has no selection bias
    assert expected_max_sharpe(0.0, 50) == 0.0     # no trial variance -> no bias


def test_deflated_sharpe_separates_signal_from_noise():
    trials = list(np.random.default_rng(3).normal(0.1, 0.15, 20))
    good = deflated_sharpe_ratio(_good(), trial_sharpes=trials, n_trials=20)
    noise = deflated_sharpe_ratio(_noise(), trial_sharpes=trials, n_trials=20)
    assert good["dsr"] > 0.9 and noise["dsr"] < 0.5
    assert good["sr0"] > 0.0            # the best-of-N benchmark is positive


def test_bootstrap_ci_brackets_and_orders():
    boot = bootstrap_return_metrics(_good(), periods_per_year=12, n_boot=1000, seed=0)
    for key in ("ann_return", "ann_sharpe"):
        med, lo, hi = boot[key]
        assert lo < med < hi
        assert lo > 0.0                # a clearly positive series stays positive


# --------------------------------------------------------------------------- #
# Report wording                                                               #
# --------------------------------------------------------------------------- #
# The interpretation paragraph used to be a fixed, favourable sentence, so the
# generated report asserted that the edge had survived even on runs where it had
# not. These two tests pin the wording to the numbers in both directions.

def _grid(sharpes):
    import pandas as pd
    return pd.DataFrame({"Sharpe": list(sharpes)})


def test_interpretation_reports_a_strong_result_as_strong():
    from src.cli.robustness_cli import _interpretation

    text = _interpretation(
        _grid([1.1, 1.2, 0.9]),
        {"dsr": 0.99, "sr0": 0.12, "sr": 0.33, "n_trials": 50.0},
        {"ann_sharpe": (1.0, 0.4, 1.7)},
    )
    assert "unlikely to be the lucky best of many attempts" in text
    assert "All three checks are passed." in text


def test_interpretation_reports_a_weak_result_as_weak():
    from src.cli.robustness_cli import _interpretation

    text = _interpretation(
        _grid([0.4, -0.1, 0.2]),
        {"dsr": 0.31, "sr0": 0.20, "sr": 0.08, "n_trials": 50.0},
        {"ann_sharpe": (0.3, -0.5, 1.1)},
    )
    assert "cannot be separated from the best of many attempts" in text
    assert "non-positive in 1 of 3" in text
    assert "includes zero" in text
    assert "Not every check is passed" in text
