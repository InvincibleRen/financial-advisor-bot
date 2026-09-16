"""Robustness / self-honesty statistics for the selection strategy (evaluation layer).

A single backtest number can fool you three ways; this module supplies the
standard antidotes:

1. **"Is it just luck from trying many variants?"** — the **Deflated Sharpe Ratio**
   (Bailey & López de Prado, 2014). We tried several models / top-N / normalisation
   settings; the best of many trials is upward-biased. The DSR discounts the
   observed Sharpe by *how many strategies were tried* and by the return series'
   skew/kurtosis, and returns the probability the true Sharpe still beats zero.

2. **"Does it only work at one cherry-picked setting?"** — handled by the
   sensitivity grid in ``cli/robustness_cli.py`` (this module supplies the Sharpe
   used at each grid point).

3. **"Is the edge carried by a few lucky months?"** — ``bootstrap_return_metrics``
   resamples the monthly returns to put a 95% confidence interval around the
   annualised return and Sharpe.

Everything here is a formula on the realised return series (plus the trial count),
so it adds no new model assumptions and runs instantly.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (via scipy when present, else a good approx)."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:  # pragma: no cover - scipy is a sklearn dependency, normally present
        # Acklam's rational approximation (max abs error ~1.15e-9).
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _clean(returns: Sequence[float]) -> np.ndarray:
    r = np.asarray(list(returns), dtype="float64")
    return r[~np.isnan(r)]


def sharpe_per_period(returns: Sequence[float]) -> float:
    """Non-annualised Sharpe: ``mean / std`` of the return series (0 if degenerate)."""
    r = _clean(returns)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def _skew_kurt(r: np.ndarray) -> Tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis; Gaussian kurtosis = 3."""
    n = r.size
    m = r.mean()
    s = r.std(ddof=0)
    if s == 0 or n < 3:
        return 0.0, 3.0
    skew = float(np.mean(((r - m) / s) ** 3))
    kurt = float(np.mean(((r - m) / s) ** 4))
    return skew, kurt


def probabilistic_sharpe_ratio(returns: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``), adjusting for skew & kurtosis.

    ``sr_benchmark`` is a *per-period* Sharpe. With the default 0 this is
    "probability the strategy's true Sharpe is positive".
    """
    r = _clean(returns)
    if r.size < 3:
        return float("nan")
    sr = sharpe_per_period(r)
    skew, kurt = _skew_kurt(r)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - sr_benchmark) * math.sqrt(r.size - 1) / denom
    return float(_norm_cdf(z))


def expected_max_sharpe(trial_sharpe_variance: float, n_trials: int) -> float:
    """Expected maximum *per-period* Sharpe from ``n_trials`` independent tries.

    This is the null benchmark the Deflated Sharpe must clear: even with no skill,
    the best of many trials has a positive expected Sharpe of roughly this size.
    """
    if n_trials < 2 or trial_sharpe_variance <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    term = (1 - g) * _norm_ppf(1 - 1.0 / n_trials) + g * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(trial_sharpe_variance) * term)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    trial_sharpes: Optional[Sequence[float]] = None,
    n_trials: Optional[int] = None,
) -> Dict[str, float]:
    """Deflated Sharpe Ratio: PSR measured against the best-of-N-trials benchmark.

    Parameters
    ----------
    returns:
        The selected strategy's realised per-period (e.g. monthly) returns.
    trial_sharpes:
        Per-period Sharpe estimates of the *other strategies tried* (e.g. every
        cell of the sensitivity grid). Their variance sets how much luck N trials
        could manufacture. If omitted, a conservative default variance of 1/(T-1)
        (the Sharpe estimator's null variance) is used.
    n_trials:
        Number of strategy configurations tried. Defaults to ``len(trial_sharpes)``.

    Returns ``{sr, sr0, dsr, n_trials}`` where ``dsr`` is the probability the true
    Sharpe beats the best-of-N-trials benchmark — i.e. the honest "is this real?".
    """
    r = _clean(returns)
    sr = sharpe_per_period(r)
    if trial_sharpes is not None and len(list(trial_sharpes)) >= 2:
        ts = np.asarray(list(trial_sharpes), dtype="float64")
        var = float(ts.var(ddof=1))
        n = int(n_trials) if n_trials else int(ts.size)
    else:
        n = int(n_trials) if n_trials else 1
        var = 1.0 / max(1, r.size - 1)
    sr0 = expected_max_sharpe(var, n)
    dsr = probabilistic_sharpe_ratio(r, sr_benchmark=sr0)
    return {"sr": sr, "sr0": sr0, "dsr": dsr, "n_trials": float(n)}


def default_block_size(n: int) -> int:
    """Rule-of-thumb moving-block length for a series of length ``n`` (``n**(1/3)``).

    For the monthly walk-forward series (``n`` ~ 140) this gives a block of about
    5 months, long enough to carry over the persistence induced by overlapping
    feature windows without shrinking the number of distinct blocks too far.
    """
    if n < 4:
        return 1
    return max(2, int(round(n ** (1.0 / 3.0))))


def _block_indices(n: int, n_boot: int, block_size: int, rng) -> np.ndarray:
    """Circular moving-block resample indices, shape ``(n_boot, n)``.

    Blocks wrap around the end of the series so every observation is drawn with
    equal probability (a plain non-circular scheme under-weights both tails).
    """
    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def bootstrap_return_metrics(
    returns: Sequence[float],
    periods_per_year: int = 12,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap ``(median, lo, hi)`` for annualised return and annualised Sharpe.

    Resamples the return series with replacement so the reader sees the *range* the
    edge could plausibly take, not a single point estimate carried by a few months.

    Parameters
    ----------
    block_size:
        Length of the resampled blocks. ``1`` gives the plain i.i.d. bootstrap.
        ``None`` (the default) uses :func:`default_block_size`, a **moving-block**
        bootstrap.

        The i.i.d. scheme assumes each monthly return is independent of its
        neighbours. That assumption does not hold here: the long-window momentum
        features overlap heavily between consecutive rebalances (``mom_12m`` shares
        ~92% of its window with the previous month), so the selector tends to hold
        overlapping baskets and consecutive monthly returns share exposure.
        Resampling whole blocks preserves that local dependence, which widens the
        interval to an honest width instead of an optimistically narrow one.
    """
    r = _clean(returns)
    out: Dict[str, Tuple[float, float, float]] = {}
    if r.size < 2:
        nan3 = (float("nan"), float("nan"), float("nan"))
        return {"ann_return": nan3, "ann_sharpe": nan3}
    if block_size is None:
        block_size = default_block_size(r.size)
    block_size = max(1, min(int(block_size), r.size))
    rng = np.random.default_rng(seed)
    if block_size == 1:
        idx = rng.integers(0, r.size, size=(n_boot, r.size))
    else:
        idx = _block_indices(r.size, n_boot, block_size, rng)
    samples = r[idx]
    ann_ret = (1.0 + samples).prod(axis=1) ** (periods_per_year / r.size) - 1.0
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    ann_sharpe = np.where(std > 0, mean / std * math.sqrt(periods_per_year), 0.0)
    for name, arr in (("ann_return", ann_ret), ("ann_sharpe", ann_sharpe)):
        lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
        out[name] = (float(np.median(arr)), float(lo), float(hi))
    return out
