"""
mosaiqtl.lmm — linear mixed model engine

implements the eigendecomposition-based fast lmm from
kang et al. (2008, nature genetics) as used in qtl2/miqtl.

model:
  y = X β + Zu + ε
  u ~ N(0, σ²_g K)
  ε ~ N(0, σ²_e I)

after eigendecomposition K = U D U':
  rotate everything by U':
    y* = U'y,  X* = U'X
  the covariance becomes diagonal:
    Var(y*) = σ²_g D + σ²_e I = σ²_e (h²/(1-h²) D + I)
  where h² = σ²_g / (σ²_g + σ²_e)

this reduces the lmm to weighted least squares, solved in O(np²)
instead of O(n³) per locus.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import logging

logger = logging.getLogger('mosaiqtl')


class LMMResult:
    """container for lmm fit results at one locus.

    attributes
    ----------
    lod : float
        lod score = log10(likelihood_ratio / (2 * ln(10)))
    p_value : float
        p-value from f-test
    h2 : float
        heritability estimate under alternative model
    betas : np.ndarray
        estimated fixed effects (haplotype + covariates)
    sigma2_g : float
        genetic variance component
    sigma2_e : float
        residual variance component
    rss_null : float
        residual sum of squares under null
    rss_alt : float
        residual sum of squares under alternative
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def estimate_h2_null(y_rot, x_rot, eigvals, method='reml'):
    """estimate heritability under the null model (no qtl effect).

    optimizes h² by profile likelihood on a grid, then refines
    with brent's method.

    parameters
    ----------
    y_rot : np.ndarray (n,)
        rotated phenotype (U' y)
    x_rot : np.ndarray (n, p)
        rotated covariate matrix (U' X)
    eigvals : np.ndarray (n,)
        eigenvalues of kinship matrix
    method : str
        'reml' or 'ml'

    returns
    -------
    h2 : float
        estimated heritability (0 to 1)
    loglik : float
        log-likelihood at optimal h²
    """
    n = len(y_rot)
    p = x_rot.shape[1] if x_rot.ndim > 1 else 1

    def neg_loglik(h2):
        """negative (restricted) log-likelihood as function of h²."""
        # clamp h2 to valid range
        h2 = np.clip(h2, 1e-6, 1 - 1e-6)

        # weight vector: w_i = h² * d_i + (1 - h²)
        # = (1 - h²) * (h²/(1-h²) * d_i + 1)
        delta = h2 / (1 - h2)
        w = delta * eigvals + 1.0  # weights in rotated space

        # weighted least squares
        # solve (X' W^{-1} X) β = X' W^{-1} y
        w_inv = 1.0 / w
        xw = x_rot * w_inv[:, np.newaxis]  # X * diag(1/w)
        xtw_x = xw.T @ x_rot  # X' W^{-1} X
        xtw_y = xw.T @ y_rot  # X' W^{-1} y

        try:
            beta = np.linalg.solve(xtw_x, xtw_y)
        except np.linalg.LinAlgError:
            return 1e10  # singular — return large value

        # residuals
        resid = y_rot - x_rot @ beta

        # weighted rss: Σ resid² / w
        wrss = np.sum(resid ** 2 * w_inv)

        # sigma² estimate
        if method == 'reml':
            sigma2 = wrss / (n - p)
        else:
            sigma2 = wrss / n

        # log-likelihood
        logdet = np.sum(np.log(w))

        if method == 'reml':
            # reml: add log|X'W^{-1}X| term
            _, logdet_xtx = np.linalg.slogdet(xtw_x)
            ll = -0.5 * (
                (n - p) * np.log(2 * np.pi * sigma2) +
                logdet +
                logdet_xtx +
                (n - p)
            )
        else:
            ll = -0.5 * (
                n * np.log(2 * np.pi * sigma2) +
                logdet +
                n
            )

        return -ll

    # optimize h² via bounded scalar optimization
    result = minimize_scalar(
        neg_loglik,
        bounds=(1e-6, 1 - 1e-6),
        method='bounded',
        options={'xatol': 1e-6}
    )

    h2_opt = result.x
    loglik_opt = -result.fun

    return h2_opt, loglik_opt


def fit_lmm_at_locus(y_rot, h_rot, x_rot, eigvals, h2_null,
                      use_reml=True):
    """fit lmm at a single locus and compute test statistic.

    uses the pre-estimated null h² (from estimate_h2_null) as the
    variance component, following qtl2 convention: h² is estimated
    once under the null and held fixed during the genome scan.

    parameters
    ----------
    y_rot : np.ndarray (n,)
        rotated phenotype
    h_rot : np.ndarray (n, q)
        rotated haplotype design matrix (j-1 founders after dropping one)
    x_rot : np.ndarray (n, p)
        rotated covariate matrix (including intercept)
    eigvals : np.ndarray (n,)
        eigenvalues of kinship matrix
    h2_null : float
        heritability from null model
    use_reml : bool
        use reml (True) or ml (False)

    returns
    -------
    LMMResult
    """
    n = len(y_rot)
    q = h_rot.shape[1]  # number of haplotype effects (j-1)
    p = x_rot.shape[1]  # number of covariate columns

    # weight vector from null h²
    delta = h2_null / max(1 - h2_null, 1e-10)
    w = delta * eigvals + 1.0
    w_inv = 1.0 / w

    # --- null model: y* = X* α ---
    xw = x_rot * w_inv[:, np.newaxis]
    xtw_x = xw.T @ x_rot
    xtw_y = xw.T @ y_rot

    try:
        beta_null = np.linalg.solve(xtw_x, xtw_y)
    except np.linalg.LinAlgError:
        return LMMResult(
            lod=0.0, p_value=1.0, h2=h2_null,
            betas=np.zeros(q + p), sigma2_g=0.0, sigma2_e=0.0,
            rss_null=np.inf, rss_alt=np.inf
        )

    resid_null = y_rot - x_rot @ beta_null
    rss_null = np.sum(resid_null ** 2 * w_inv)

    # --- alt model: y* = [H* X*] [γ; α] ---
    # combine haplotype and covariate matrices
    z_rot = np.hstack([h_rot, x_rot])  # (n, q+p)

    zw = z_rot * w_inv[:, np.newaxis]
    ztw_z = zw.T @ z_rot
    ztw_y = zw.T @ y_rot

    try:
        beta_alt = np.linalg.solve(ztw_z, ztw_y)
    except np.linalg.LinAlgError:
        return LMMResult(
            lod=0.0, p_value=1.0, h2=h2_null,
            betas=np.zeros(q + p), sigma2_g=0.0, sigma2_e=0.0,
            rss_null=rss_null, rss_alt=rss_null
        )

    resid_alt = y_rot - z_rot @ beta_alt
    rss_alt = np.sum(resid_alt ** 2 * w_inv)

    # --- lod score ---
    # LOD = (n/2) * log10(rss_null / rss_alt)
    if use_reml:
        denom = n - p  # null df
    else:
        denom = n

    if rss_alt > 0 and rss_null > rss_alt:
        lod = (denom / 2.0) * np.log10(rss_null / rss_alt)
    else:
        lod = 0.0

    # --- f-test p-value ---
    # F = ((rss_null - rss_alt) / q) / (rss_alt / (n - p - q))
    df1 = q  # number of haplotype parameters tested
    df2 = n - p - q  # residual df under alt

    if df2 > 0 and rss_alt > 0:
        f_stat = ((rss_null - rss_alt) / df1) / (rss_alt / df2)
        f_stat = max(f_stat, 0.0)
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
    else:
        f_stat = 0.0
        p_value = 1.0

    # --- variance components ---
    sigma2_total = rss_alt / max(denom - q, 1)
    sigma2_g = h2_null * sigma2_total
    sigma2_e = (1 - h2_null) * sigma2_total

    return LMMResult(
        lod=lod,
        p_value=p_value,
        h2=h2_null,
        betas=beta_alt[:q],  # just the haplotype effects
        sigma2_g=sigma2_g,
        sigma2_e=sigma2_e,
        rss_null=rss_null,
        rss_alt=rss_alt,
        f_stat=f_stat,
    )


def scan_chromosome(y, genome_cache, chrom, x_covariates,
                    eigvals, eigvecs, h2_null, use_reml=True):
    """scan all loci on one chromosome.

    parameters
    ----------
    y : np.ndarray (n,)
        phenotype vector
    genome_cache : GenomeCache
        haplotype probability data
    chrom : str
        chromosome to scan
    x_covariates : np.ndarray (n, p)
        covariate design matrix (with intercept)
    eigvals : np.ndarray (n,)
        eigenvalues from loco kinship for this chromosome
    eigvecs : np.ndarray (n, n)
        eigenvectors from loco kinship for this chromosome
    h2_null : float
        null heritability for this chromosome
    use_reml : bool
        use reml estimation

    returns
    -------
    dict with keys:
        'locus_ids' : np.ndarray of str
        'positions' : np.ndarray of float
        'lod'       : np.ndarray of float
        'p_value'   : np.ndarray of float
        'betas'     : np.ndarray (n_loci, n_founders-1)
    """
    locus_ids, positions, probs = genome_cache.get_chromosome(chrom)
    n_loci = len(locus_ids)

    # rotate phenotype and covariates once per chromosome
    y_rot = eigvecs.T @ y
    x_rot = eigvecs.T @ x_covariates

    # allocate output arrays
    lod_arr = np.zeros(n_loci)
    pval_arr = np.ones(n_loci)
    n_founders_minus1 = genome_cache.n_founders - 1
    betas_arr = np.zeros((n_loci, n_founders_minus1))

    for l_idx in range(n_loci):
        # get haplotype probs at this locus
        h_l = probs[l_idx]  # (n_subjects, n_founders)

        # drop one founder column (last = wsbj) for identifiability
        h_l_reduced = h_l[:, :n_founders_minus1]

        # rotate haplotype matrix
        h_rot = eigvecs.T @ h_l_reduced

        # fit lmm at this locus
        result = fit_lmm_at_locus(
            y_rot, h_rot, x_rot, eigvals, h2_null, use_reml
        )

        lod_arr[l_idx] = result.lod
        pval_arr[l_idx] = result.p_value
        betas_arr[l_idx] = result.betas

    return {
        'locus_ids': locus_ids,
        'positions': positions,
        'lod': lod_arr,
        'p_value': pval_arr,
        'betas': betas_arr,
    }
