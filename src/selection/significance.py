"""Statistical-significance tools for the selection ranker (evaluation layer).

Answers the question a backtest cannot answer on its own: *is the ranker's
out-of-sample skill real, or could random chance produce it?* Two complementary,
leakage-free procedures:

1. **Label-permutation test** (``permutation_test``). Holding the model's real,
   already-computed out-of-sample scores fixed, shuffle the outcome labels within
   each rebalance date's cross-section and recompute the ranking statistic
   (mean AUC, precision@N). Repeating this builds the null distribution of "no
   skill" and yields a p-value: the fraction of shuffles that match or beat the
   real statistic. This is the standard permutation test for a rank statistic and
   needs no re-training, so thousands of shuffles are cheap.

2. **Bootstrap confidence intervals** (``bootstrap_ci``). Resample the per-fold
   AUCs (or the monthly returns) with replacement to report a median and 95%
   interval, showing the edge is not carried by a handful of months.

Both consume the per-fold records the walk-forward already produces, so they add
no model assumptions of their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.selection.select import _FoldRecord, _labels_for, select_top_n


@dataclass
class SignificanceResult:
    """Everything the report needs for one model."""
    real_mean_auc: float
    real_precision_at_n: float
    base_rate: float
    n_permutations: int
    p_value_auc: float
    p_value_precision: float
    null_auc: np.ndarray            # null distribution of mean AUC
    null_precision: np.ndarray      # null distribution of precision@N
    auc_ci: Tuple[float, float, float]        # (median, lo, hi) over folds, bootstrapped
    fold_aucs: np.ndarray


def _per_fold_arrays(records: List[_FoldRecord], labels: pd.Series, n: int):
    """Extract, per fold, the aligned (scores, labels) and the picked tickers.

    Returns a list of dicts so both the real statistic and every permutation reuse
    exactly the same cross-sections (only the labels get shuffled downstream).
    """
    folds = []
    for rec in records:
        cross = _labels_for(labels, rec.date, list(rec.scores.index)).dropna()
        if cross.empty:
            continue
        scores = rec.scores.reindex(cross.index).to_numpy(dtype="float64")
        y = cross.to_numpy(dtype="int64")
        picks = select_top_n(rec.scores, n)
        pick_idx = [list(cross.index).index(t) for t in picks if t in cross.index]
        folds.append({"scores": scores, "y": y, "pick_idx": np.array(pick_idx, dtype=int)})
    return folds


def _auc(scores: np.ndarray, y: np.ndarray) -> Optional[float]:
    """ROC-AUC of scores vs binary y; None when only one class is present."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def _mean_auc_and_precision(folds, y_by_fold) -> Tuple[float, float]:
    """Mean per-fold AUC and precision@N for a given set of per-fold label arrays."""
    aucs, precs = [], []
    for f, y in zip(folds, y_by_fold):
        a = _auc(f["scores"], y)
        if a is not None:
            aucs.append(a)
        if len(f["pick_idx"]):
            precs.append(float((y[f["pick_idx"]] == 1).mean()))
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    mean_prec = float(np.mean(precs)) if precs else float("nan")
    return mean_auc, mean_prec


def permutation_test(
    records: List[_FoldRecord],
    labels: pd.Series,
    n: int = 3,
    n_permutations: int = 100,
    seed: int = 0,
) -> SignificanceResult:
    """Fixed-score label-permutation test + bootstrap CI on the real fold AUCs.

    The real scores are held fixed; only the outcome labels are shuffled *within
    each rebalance date's cross-section* (so each month keeps its own size and
    base rate). The p-value is ``(1 + #{null >= real}) / (1 + n_permutations)``
    — the conventional, unbiased permutation p-value that never returns 0.
    """
    folds = _per_fold_arrays(records, labels, n)
    if not folds:
        raise ValueError("No usable folds for the permutation test.")

    real_y = [f["y"] for f in folds]
    real_auc, real_prec = _mean_auc_and_precision(folds, real_y)
    base_rate = float(np.mean(np.concatenate(real_y)))
    fold_aucs = np.array([a for f in folds if (a := _auc(f["scores"], f["y"])) is not None])

    rng = np.random.default_rng(seed)
    null_auc = np.empty(n_permutations)
    null_prec = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = [rng.permutation(f["y"]) for f in folds]
        null_auc[i], null_prec[i] = _mean_auc_and_precision(folds, shuffled)

    p_auc = (1 + int(np.sum(null_auc >= real_auc))) / (1 + n_permutations)
    p_prec = (1 + int(np.sum(null_prec >= real_prec))) / (1 + n_permutations)
    return SignificanceResult(
        real_mean_auc=real_auc, real_precision_at_n=real_prec, base_rate=base_rate,
        n_permutations=n_permutations, p_value_auc=p_auc, p_value_precision=p_prec,
        null_auc=null_auc, null_precision=null_prec,
        auc_ci=bootstrap_ci(fold_aucs, seed=seed), fold_aucs=fold_aucs,
    )


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap ``(median, lo, hi)`` of the mean of ``values`` at ``1 - alpha``.

    Resamples ``values`` (e.g. per-fold AUCs or monthly returns) with replacement
    to put an interval around the mean — the "error bar" a single backtest lacks.
    """
    values = np.asarray(values, dtype="float64")
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(np.median(means)), float(lo), float(hi))
