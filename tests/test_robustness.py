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
