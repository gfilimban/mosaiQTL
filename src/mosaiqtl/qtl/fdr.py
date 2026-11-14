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
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import brentq, minimize

logger = logging.getLogger("mosaiqtl")

__all__ = ["beta_calibrate_pvalue", "storey_qvalue", "calibrate_gene_pvalue"]

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


def _smooth_df3_at_max(x: np.ndarray, y: np.ndarray) -> float:
    """Fit a cubic smoothing spline with 3 effective df and evaluate at max(x).

    The smoothing penalty is chosen so that the trace of the spline's linear
    smoother matrix equals 3.

    Args:
        x: Strictly increasing tuning values, at least 5.
        y: Raw pi0 estimate at each tuning value.

    Returns:
        The smoothed pi0 at the largest tuning value.
    """
    n = len(x)
    eye = np.eye(n)

    def _trace_minus_three(log10_pen: float) -> float:
        pen = 10.0 ** log10_pen
        return (
            sum(make_smoothing_spline(x, eye[i], lam=pen)(x[i]) for i in range(n))
            - 3.0
        )

    log10_pen = brentq(_trace_minus_three, -8.0, 4.0)
    return float(make_smoothing_spline(x, y, lam=10.0 ** log10_pen)(x.max()))


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
    if ok.sum() >= 5:
        try:
            # cubic smoothing spline over the tuning grid, penalty chosen so the
            # fit uses 3 effective degrees of freedom, read at the largest
            # tuning value
            pi0 = _smooth_df3_at_max(lam[ok], pi0_hat[ok])
        except (ValueError, np.linalg.LinAlgError):
            pi0 = float(pi0_hat[ok].min())
    else:
        # too few usable tuning values to fit a spline — use the smallest estimate instead
        pi0 = float(pi0_hat[ok].min()) if ok.any() else 1.0

    # pi0 is a proportion, so it is capped at 1. A fitted value above 1 means the
    # p-values pile up near 1 (a null that is too conservative or too narrow),
    # so it is reported before the cap hides it.
    if pi0 > 1.0:
        logger.warning(
            "storey_qvalue: raw pi0=%.4f exceeds 1 (m=%d) — p-values are skewed "
            "toward 1; the calibration null is likely miscalibrated",
            pi0,
            m,
        )
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


def calibrate_gene_pvalue(
    y_gene: np.ndarray,
    genomecache,
    chrom: str,
    peak_lod: float,
    covariates: np.ndarray,
    kinship_decomp: dict,
    kinship_loco: dict,
    n_founders: "int | None" = None,
    n_perms: int = 200,
    gene_pos_mb: "float | None" = None,
    window_mb: "float | None" = None,
    use_rint: bool = True,
    z_mat: "np.ndarray | None" = None,
    seed: int = 1,
    chr_data: "dict | None" = None,
) -> "tuple[float, float, float, float]":
    """calibrate one gene's peak LOD into a Beta-adjusted, genome-comparable p-value.

    a genome-wide threshold treats every gene as though it shares one null
    distribution, but each gene has its own fixed nuisance parameter (h²) and
    therefore its own null LOD distribution. this draws a small parametric null
    specific to this gene — same h² and same cis window used for its real scan —
    then calls beta_calibrate_pvalue() for the Beta-smoothed adjustment. the
    returned p-value is Uniform(0, 1) under the null for this gene specifically,
    so p-values from many genes can be pooled and fed to storey_qvalue() for a
    genome-wide q-value.

    parameters
    ----------
    y_gene         : np.ndarray (n,)  this gene's phenotype vector, already at
                     scan level (strain means if replicate-aware, sample level
                     otherwise).  RINT is applied inside if use_rint=True.
    genomecache    : GenomeCache
    chrom          : str  chromosome to scan (the gene's cis chromosome)
    peak_lod       : float  this gene's real observed peak LOD on chrom
    covariates     : np.ndarray (n, p)  covariate matrix at scan level, with
                     intercept as first column.
    kinship_decomp : dict  {chrom: {'values':, 'vectors':}}  build once
                     genome-wide and reuse across genes.
    kinship_loco   : dict  {chrom: np.ndarray (m, m)}  raw LOCO kinship,
                     required for the parametric null draw.
    n_founders     : int or None  df for the chi² LRT (n_founders - 1); when
                     None, taken from genomecache.n_founders.
    n_perms        : int  permutations for this gene's null (default 200).
    gene_pos_mb    : float or None  gene TSS position in Mb; with window_mb,
                     restricts the null scan to the cis window used for the real scan.
    window_mb      : float or None  half-window in Mb around gene_pos_mb.
    use_rint       : bool  rank-inverse normal transform y_gene before scanning.
    z_mat          : np.ndarray (n, m) or None  sample-to-strain design matrix
                     for n-level scanning; when provided, kinship_decomp must
                     hold eigdecomp of K_n = Z K_m Z^T.
    seed           : int  rng seed for the null draw.
    chr_data       : dict or None  lmm.precompute_chromosome output for chrom built
                     from kinship_decomp[chrom] and z_mat; reuses the rotated
                     genotype array across genes on the same chromosome.

    returns
    -------
    p_adj : float  Beta-calibrated p-value for this gene's peak_lod
    a_hat : float  fitted Beta shape parameter
    b_hat : float  fitted Beta shape parameter
    h2_g  : float  this gene's own fixed h² used for the null draw
    """
    from . import lmm as lmm_module
    from .permutation import draw_parametric_null_batch
    from ..utils import rint as _rint

    y0 = _rint(y_gene) if use_rint else np.asarray(y_gene, dtype=float).copy()

    ev = kinship_decomp[chrom]["values"]
    evc = kinship_decomp[chrom]["vectors"]
    K = kinship_loco[chrom]

    h2_g, _ = lmm_module.estimate_h2_null(evc.T @ y0, evc.T @ covariates, ev, method="ml")

    rng = np.random.default_rng(seed)
    null_mat = draw_parametric_null_batch(y0, covariates, K, evc, ev, h2_g, n_perms, rng)

    loci_mask = None
    if gene_pos_mb is not None and window_mb is not None:
        _, positions_c, _ = genomecache.get_chromosome(chrom)
        loci_mask = np.abs(positions_c - gene_pos_mb) <= window_mb

    null_max_lods = lmm_module.scan_chromosome_batch(
        null_mat,
        genomecache,
        chrom,
        covariates,
        ev,
        evc,
        h2_g,
        z_mat=z_mat,
        loci_mask=loci_mask,
        chr_data=chr_data,
    )

    df = (n_founders if n_founders is not None else genomecache.n_founders) - 1
    p_adj, a_hat, b_hat = beta_calibrate_pvalue(peak_lod, null_max_lods, df)
    logger.debug(
        "calibrate_gene_pvalue: chr%s  n_perms=%d  h2=%.4f  peak_lod=%.3f  p_adj=%.3g",
        chrom,
        n_perms,
        h2_g,
        peak_lod,
        p_adj,
    )
    return p_adj, a_hat, b_hat, h2_g
