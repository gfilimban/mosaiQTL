"""
fdr — turning per-gene LOD scores into genome-wide significance calls.

beta_calibrate_pvalue: converts one gene's permutation results into a properly
scaled p-value. this has to happen before genes can be compared to each other.
storey_qvalue: Storey's q-value method for controlling the false discovery rate
across all genes at once, estimating what fraction are truly null (pi0).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

logger = logging.getLogger("mosaiqtl")

__all__ = ["beta_calibrate_pvalue", "storey_qvalue"]

# Grid of tuning values used to estimate pi0, as recommended in Storey (2002).
_LAMBDA_DEFAULT = np.arange(0.05, 1.00, 0.05)


def beta_calibrate_pvalue(
    peak_lod: float,
    null_max_lods: np.ndarray,
    df: int,
) -> tuple[float, float, float]:
    """Smooth out a gene's permutation results and convert its peak LOD to a p-value.

    running only a handful of permutations per gene (e.g. 100) can only ever
    resolve p-values down to about 1/100 — too coarse to compare genes against
    each other genome-wide, and running enough permutations per gene to get
    finer resolution directly would be far too slow. instead, this fits a smooth
    beta distribution to the small set of permutation results and reads the
    p-value off that curve, which extrapolates sensibly into the tail beyond
    what the raw permutation count alone could resolve.

    each permutation's peak LOD is first converted to a p-value using the
    standard chi-square relationship for a likelihood-ratio test. a beta
    distribution is then fit to those permutation p-values, and the real
    (observed) peak LOD's p-value is read off that fitted curve. doing this
    per gene matters because different genes have different nuisance
    parameters (e.g. heritability), so a p-value taken from one shared or
    "representative" gene's null would not be valid for every other gene.

    parameters
    ----------
    peak_lod      : float  the gene's real observed peak LOD in its scan window
    null_max_lods : np.ndarray (n_perms,)  peak LOD from each permutation, run
        with this same gene's own nuisance parameter (heritability for the
        LMM engine, dispersion for the NB-GLMM engine)
    df            : int  degrees of freedom for the likelihood-ratio test
        (number of founder haplotypes minus one, for a haplotype scan)

    returns
    -------
    p_adj : float  calibrated p-value for peak_lod, in [0, 1]
    a_hat : float  fitted beta shape parameter
    b_hat : float  fitted beta shape parameter

    notes
    -----
    LOD is converted to a p-value via: test statistic = 2*ln(10)*LOD, which
    follows a chi-square distribution under the null. feed the p_adj values
    from many genes into storey_qvalue() to get genome-wide significance calls.
    """
    stat_null = null_max_lods * (2.0 * np.log(10.0))
    p_null = np.clip(stats.chi2.sf(stat_null, df), 1e-300, 1.0)

    mean_p = float(p_null.mean())
    var_p = float(p_null.var())
    if var_p > 0.0 and 0.0 < mean_p < 1.0:
        common = max(mean_p * (1.0 - mean_p) / var_p - 1.0, 1e-3)
        a0 = max(mean_p * common, 1e-3)
        b0 = max((1.0 - mean_p) * common, 1e-3)
    else:
        a0, b0 = 1.0, 1.0

    def _negloglik(params: np.ndarray) -> float:
        a, b = params
        if a <= 0.0 or b <= 0.0:
            return 1e10
        return float(-np.sum(stats.beta.logpdf(p_null, a, b)))

    fit = minimize(_negloglik, x0=[a0, b0], method="Nelder-Mead")
    a_hat, b_hat = (float(fit.x[0]), float(fit.x[1])) if fit.success else (a0, b0)

    stat_real = peak_lod * (2.0 * np.log(10.0))
    p_real = float(np.clip(stats.chi2.sf(stat_real, df), 1e-300, 1.0))
    p_adj = float(stats.beta.cdf(p_real, a_hat, b_hat))
    logger.debug(
        "beta_calibrate_pvalue: n_perms=%d  a=%.3f  b=%.3f  p_real=%.3g  p_adj=%.3g",
        len(null_max_lods),
        a_hat,
        b_hat,
        p_real,
        p_adj,
    )
    return p_adj, a_hat, b_hat


def storey_qvalue(
    pvalues: np.ndarray,
    lambda_range: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate genome-wide false discovery rate (Storey 2002 q-value method).

    estimates what fraction of the genes tested are truly null (pi0) from the
    shape of the overall p-value distribution, then uses that fraction to scale
    a standard multiple-testing correction into a q-value per gene. this gives
    more power to call true discoveries than a correction that assumes every
    gene is null, which is the more conservative (but wrong) default assumption.

    parameters
    ----------
    pvalues      : np.ndarray (m,)  one p-value per gene, in [0, 1]; NaN skipped
    lambda_range : np.ndarray or None
        grid of tuning values used to estimate pi0 (default: 0.05 to 0.95 in
        steps of 0.05). needs at least 4 points to fit smoothly; falls back to
        a simpler estimate when fewer than 4 usable values are available.

    returns
    -------
    q_values : np.ndarray (m,)  q-value per gene, in [0, 1]; NaN where input was NaN
    pi0      : float             estimated fraction of genes that are truly null

    notes
    -----
    if essentially all genes are null, pi0 approaches 1 and this reduces to the
    standard (more conservative) correction. if there is real signal in the
    data, pi0 drops below 1 and genes are called significant more readily.
    """
    lam = _LAMBDA_DEFAULT if lambda_range is None else np.asarray(lambda_range)
    pv = np.asarray(pvalues, dtype=float)

    finite_mask = np.isfinite(pv)
    m = int(finite_mask.sum())
    if m == 0:
        return pv.copy(), 1.0

    pv_fin = pv[finite_mask]

    # estimate pi0 at each tuning value, then smooth across the grid with a
    # spline to get one stable estimate (raw per-tuning-value estimates are noisy).
    pi0_hat = np.array([(pv_fin > lv).sum() / (m * (1.0 - lv)) for lv in lam])

    # keep only tuning values where the estimate came out finite and positive
    ok = np.isfinite(pi0_hat) & (pi0_hat > 0)
    if ok.sum() >= 4:
        try:
            # known limitations, not yet addressed:
            # - the smoothing spline used here is a simpler variant than the
            #   flat-curvature-at-the-ends spline this method is meant to use
            # - the smoothing amount is looser than typical default tuning
            # - values above 1 are silently capped below rather than flagged;
            #   they would otherwise signal that the p-values feeding this
            #   function are miscalibrated
            spl = UnivariateSpline(lam[ok], pi0_hat[ok], k=3, s=len(lam[ok]))
            pi0 = float(spl(lam[ok].max()))
        except Exception:
            pi0 = float(pi0_hat[ok].min())
    else:
        # too few usable tuning values to fit a spline — use the smallest estimate instead
        pi0 = float(pi0_hat[ok].min()) if ok.any() else 1.0

    # pi0 is a proportion, so cap it at 1; a fitted value above 1 just means noise
    pi0 = min(pi0, 1.0)
    pi0 = max(pi0, 1e-6)  # keep pi0 above zero so every q-value stays finite
    logger.debug("storey_qvalue: m=%d  pi0=%.4f", m, pi0)

    # standard multiple-testing correction, scaled by pi0.
    order = np.argsort(pv_fin)
    ranks = np.arange(1, m + 1)  # rank of each p-value once sorted, smallest first
    sorted_p = pv_fin[order]

    q_sorted = np.minimum(pi0 * m * sorted_p / ranks, 1.0)

    # q-values must not decrease as the p-value gets larger; enforce that here
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    # put the q-values back in the same gene order as the input
    q_fin = np.empty(m)
    q_fin[order] = q_sorted

    q_out = np.full_like(pv, np.nan)
    q_out[finite_mask] = q_fin
    return q_out, pi0
