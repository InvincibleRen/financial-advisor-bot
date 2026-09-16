"""Offline tests for the permutation + bootstrap significance layer.

Synthetic fold records, no network and no model training: a *signal* case (scores
that rank the labels well) must come out significant, a *noise* case (scores
unrelated to labels) must not, and the bootstrap interval must bracket the mean.
"""
import numpy as np
import pandas as pd

from src.selection.select import _FoldRecord
from src.selection.significance import bootstrap_ci, permutation_test


def _make(records_scores_labels):
    """Build (records, labels) from a list of (date, {ticker: (score, label)})."""
    recs, lab = [], {}
    for date, cell in records_scores_labels:
        scores = pd.Series({t: s for t, (s, _) in cell.items()})
        recs.append(_FoldRecord(pd.Timestamp(date), scores, list(scores.sort_values(ascending=False).head(2).index)))
        for t, (_, y) in cell.items():
            lab[(pd.Timestamp(date), t)] = y
    labels = pd.Series(lab)
    labels.index = pd.MultiIndex.from_tuples(labels.index, names=["rebalance_date", "ticker"])
    return recs, labels


def _signal_folds(n_dates=40, n_names=10, seed=1):
    """Scores that carry real ranking signal: label = 1 for the top-half scores."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_dates):
        scores = rng.normal(size=n_names)
        # top half by score are winners, but with 20% label noise
        order = scores.argsort()[::-1]
        y = np.zeros(n_names, dtype=int)
        y[order[: n_names // 2]] = 1
        flip = rng.random(n_names) < 0.2
        y[flip] = 1 - y[flip]
        cell = {f"T{i}": (float(scores[i]), int(y[i])) for i in range(n_names)}
        out.append((f"2020-{(d % 12) + 1:02d}-{(d // 12) + 1:02d}", cell))
    return _make(out)


def _noise_folds(n_dates=40, n_names=10, seed=2):
    """Scores unrelated to labels: no skill should be detected."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_dates):
        scores = rng.normal(size=n_names)
        y = (rng.random(n_names) < 0.5).astype(int)
        cell = {f"T{i}": (float(scores[i]), int(y[i])) for i in range(n_names)}
        out.append((f"2020-{(d % 12) + 1:02d}-{(d // 12) + 1:02d}", cell))
    return _make(out)


def test_signal_is_significant():
    recs, labels = _signal_folds()
    r = permutation_test(recs, labels, n=2, n_permutations=200, seed=0)
    assert r.real_mean_auc > 0.6            # real skill
    assert r.p_value_auc < 0.05             # detected as significant
    assert r.null_auc.mean() < r.real_mean_auc


def test_noise_is_not_significant():
    recs, labels = _noise_folds()
    r = permutation_test(recs, labels, n=2, n_permutations=200, seed=0)
    assert r.p_value_auc > 0.10             # cannot reject "no skill"


def test_pvalue_is_bounded_and_never_zero():
    recs, labels = _signal_folds()
    r = permutation_test(recs, labels, n=2, n_permutations=50, seed=0)
    assert 1 / (1 + 50) <= r.p_value_auc <= 1.0


def test_bootstrap_ci_brackets_mean():
    values = np.array([0.50, 0.52, 0.55, 0.48, 0.53, 0.51])
    med, lo, hi = bootstrap_ci(values, n_boot=1000, seed=0)
    assert lo <= values.mean() <= hi
    assert lo < med < hi
