"""
mosaiqtl.lmm — linear mixed model engine

implements the eigendecomposition-based fast LMM from
Kang et al. (2008, Genetics) as used in qtl2/miQTL.

model:
    y = H γ + X α + u + ε
    u ~ N(0, σ²_g K)
    ε ~ N(0, σ²_e I)

after eigendecomposition K = U D U':
    rotate: y* = U'y,  X* = U'X,  H* = U'H
    Var(y*) = σ²_g D + σ²_e I  — diagonal, O(n) weights
    LOD = (n/2) log10(RSS_0 / RSS_1)       [ML, matches miQTL]
    h²   estimated once under null via REML  [matches miQTL use.par='h2.REML']

matches miQTL::lmmbygls behaviour:
  - h² optimized with scipy.optimize.minimize_scalar on [0,1] + grid safeguard
  - brute check: compare optimized fit against h²=0 (miQTL's brute=TRUE default)
  - LOD uses ML formula (n/2 log10 ratio), not REML
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import logging

logger = logging.getLogger('mosaiqtl')


class LMMResult:
    """container for LMM fit results at one locus.

    attributes
    ----------
    lod      : float   LOD = (n/2) log10(RSS_0/RSS_1)
    p_value  : float   F-test p-value
    h2       : float   heritability from null model
    betas    : np.ndarray (J-1,)  founder haplotype effects
    sigma2_g : float   genetic variance component
    sigma2_e : float   residual variance component
    rss_null : float   weighted RSS under null
    rss_alt  : float   weighted RSS under alternative
    f_stat   : float   F statistic
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def estimate_h2_null(y_rot, x_rot, eigvals, method='reml'):
    """estimate heritability under the null model (no QTL effect).

    algorithm (matches miQTL lmmbygls):
      1. evaluate profile (RE)ML on a 20-point h² grid [0.01, 0.99]
      2. refine around the best grid point with Brent's method
      3. brute check: compare against h²=0 — take whichever has
         higher log-likelihood (miQTL's brute=TRUE safeguard)

    parameters
    ----------
    y_rot   : np.ndarray (n,)    U'y — rotated phenotype
    x_rot   : np.ndarray (n, p)  U'X — rotated covariates
    eigvals : np.ndarray (n,)    eigenvalues of kinship matrix
    method  : str  'reml' (default, matches miQTL) or 'ml'

    returns
    -------
    h2      : float   optimal h² ∈ (0, 1)
    loglik  : float   log-likelihood at optimal h²
    """
    n = len(y_rot)
    p = x_rot.shape[1] if x_rot.ndim > 1 else 1

    def neg_loglik(h2):
        """negative profile (RE)ML as a function of h²."""
        h2    = np.clip(h2, 1e-6, 1 - 1e-6)
        delta = h2 / (1 - h2)
        w     = delta * eigvals + 1.0     # diagonal weights
        w_inv = 1.0 / w
        xw    = x_rot * w_inv[:, None]
        try:
            beta  = np.linalg.solve(xw.T @ x_rot, xw.T @ y_rot)
        except np.linalg.LinAlgError:
            return 1e10
        resid = y_rot - x_rot @ beta
        wrss  = float(resid @ (resid * w_inv))
        logdet = float(np.sum(np.log(w)))
        if method == 'reml':
            df     = n - p
            sigma2 = wrss / df
            _, ldet_xtx = np.linalg.slogdet(xw.T @ x_rot)
            ll = -0.5 * (df*(np.log(2*np.pi*sigma2)+1) + logdet + ldet_xtx)
        else:
            sigma2 = wrss / n
            ll     = -0.5 * (n*(np.log(2*np.pi*sigma2)+1) + logdet)
        return -ll

    # ── step 1: grid search ──────────────────────────────────────────────────
    _grid     = np.linspace(0.01, 0.99, 20)
    _nll      = np.array([neg_loglik(h) for h in _grid])
    _best_idx = int(np.argmin(_nll))
    _lo = float(_grid[max(0, _best_idx - 1)])
    _hi = float(_grid[min(len(_grid)-1, _best_idx + 1)])

    # ── step 2: brent refinement ─────────────────────────────────────────────
    res    = minimize_scalar(neg_loglik, bounds=(_lo, _hi),
                             method='bounded', options={'xatol': 1e-8})
    h2_opt = float(np.clip(res.x, 1e-6, 1 - 1e-6))
    ll_opt = -float(res.fun)

    # ── step 3: brute check at h²=0 (miQTL brute=TRUE) ───────────────────────
    ll_zero = -neg_loglik(1e-6)
    if ll_zero > ll_opt:
        h2_opt = 1e-6
        ll_opt = ll_zero

    return h2_opt, ll_opt


def fit_lmm_at_locus(y_rot, h_rot, x_rot, eigvals, h2_null, use_reml=True):
    """fit LMM at a single locus and compute test statistic.

    uses null h² (from estimate_h2_null) as the fixed variance
    component during the genome scan — same convention as miQTL.

    LOD = (n/2) log10(RSS_0 / RSS_1)   [ML formula, n not n-p]

    parameters
    ----------
    y_rot    : np.ndarray (n,)     U'y
    h_rot    : np.ndarray (n, J-1) U'H  — haplotype design (one col dropped)
    x_rot    : np.ndarray (n, p)   U'X
    eigvals  : np.ndarray (n,)     eigenvalues of LOCO kinship
    h2_null  : float               null heritability
    use_reml : bool                (unused for LOD; kept for API compatibility)

    returns
    -------
    LMMResult
    """
    n = len(y_rot)
    q = h_rot.shape[1]
    p = x_rot.shape[1]

    delta = h2_null / max(1 - h2_null, 1e-10)
    w     = delta * eigvals + 1.0
    w_inv = 1.0 / w

    # null model: y* = X* α
    xw    = x_rot * w_inv[:, None]
    try:
        beta_null = np.linalg.solve(xw.T @ x_rot, xw.T @ y_rot)
    except np.linalg.LinAlgError:
        return LMMResult(lod=0.0, p_value=1.0, h2=h2_null, f_stat=0.0,
                         betas=np.zeros(q), sigma2_g=0.0, sigma2_e=0.0,
                         rss_null=np.inf, rss_alt=np.inf)
    resid_null = y_rot - x_rot @ beta_null
    rss_null   = float(resid_null @ (resid_null * w_inv))

    # alternative model: y* = [H* X*] [γ; α]
    z_rot = np.hstack([h_rot, x_rot])
    zw    = z_rot * w_inv[:, None]
    try:
        beta_alt = np.linalg.solve(zw.T @ z_rot, zw.T @ y_rot)
    except np.linalg.LinAlgError:
        return LMMResult(lod=0.0, p_value=1.0, h2=h2_null, f_stat=0.0,
                         betas=np.zeros(q), sigma2_g=0.0, sigma2_e=0.0,
                         rss_null=rss_null, rss_alt=rss_null)
    resid_alt = y_rot - z_rot @ beta_alt
    rss_alt   = float(resid_alt @ (resid_alt * w_inv))

    # LOD: ML formula (n not n-p) — matches miQTL::lmmbygls
    if rss_alt > 0 and rss_null > rss_alt:
        lod = (n / 2.0) * np.log10(rss_null / rss_alt)
    else:
        lod = 0.0

    # F-test p-value
    df2 = n - p - q
    if df2 > 0 and rss_alt > 0:
        f_stat  = ((rss_null - rss_alt) / q) / (rss_alt / df2)
        f_stat  = max(f_stat, 0.0)
        p_value = 1.0 - stats.f.cdf(f_stat, q, df2)
    else:
        f_stat  = 0.0
        p_value = 1.0

    sigma2_total = rss_alt / max(n - p - q, 1)
    return LMMResult(
        lod=lod, p_value=p_value, h2=h2_null, f_stat=f_stat,
        betas=beta_alt[:q],
        sigma2_g=h2_null * sigma2_total,
        sigma2_e=(1 - h2_null) * sigma2_total,
        rss_null=rss_null, rss_alt=rss_alt,
    )


def scan_chromosome(y, genome_cache, chrom, x_covariates,
                    eigvals, eigvecs, h2_null, use_reml=True,
                    z_mat=None):
    """scan all loci on one chromosome — vectorized batch WLS.

    replaces the per-locus python loop with a single tensor computation
    via the frisch-waugh-lovell (fwl) theorem. all loci share the same
    weight vector w and null model, so:
        rss_alt[l] = rss_null - c[l]' (H_perp[l]'H_perp[l])^{-1} c[l]
    where H_perp[l] = M_X H_t[l] (weighted H residualized on X),
    c[l] = H_perp[l]' e0, e0 = null residual.

    equivalent to qtl2::scan_pg_onechr (rcpp), now in numpy.

    parameters
    ----------
    y            : np.ndarray (n,)
    genome_cache : GenomeCache
    chrom        : str
    x_covariates : np.ndarray (n, p)
    eigvals      : np.ndarray (n,)     eigenvalues of n×n kinship
    eigvecs      : np.ndarray (n, n)   eigenvectors
    h2_null      : float
    use_reml     : bool                (unused for lod; kept for api compat)
    z_mat        : np.ndarray (n, m) or None

    returns
    -------
    dict: locus_ids, positions, lod, p_value, betas
    """
    locus_ids, positions, probs = genome_cache.get_chromosome(chrom)
    n_loci = len(locus_ids)
    nf1    = genome_cache.n_founders - 1
    n      = len(y)
    p      = x_covariates.shape[1]
    q      = nf1

    # ── shared weight vector from h² ─────────────────────────────────────────
    delta  = h2_null / max(1.0 - h2_null, 1e-10)
    w_sqrt = 1.0 / np.sqrt(delta * eigvals + 1.0)   # sqrt(w_inv), (n,)

    # ── null wls (once for all loci) ─────────────────────────────────────────
    yt   = (eigvecs.T @ y) * w_sqrt                  # weighted rotated pheno
    xt   = (eigvecs.T @ x_covariates) * w_sqrt[:,None]  # weighted rotated X
    xtwx = xt.T @ xt                                 # (p, p)
    try:
        beta_null = np.linalg.solve(xtwx, xt.T @ yt)
    except np.linalg.LinAlgError:
        return {'locus_ids': locus_ids, 'positions': positions,
                'lod': np.zeros(n_loci), 'p_value': np.ones(n_loci),
                'betas': np.zeros((n_loci, nf1))}
    e0   = yt - xt @ beta_null                       # null residual (n,)
    rss0 = float(e0 @ e0)                            # scalar null rss

    # ── build & rotate haplotype array (n_loci, n, J-1) ─────────────────────
    H = probs[:, :, :nf1]                            # (n_loci, m, J-1)
    if z_mat is not None:
        H = np.einsum('nm,lmj->lnj', z_mat, H, optimize=True)
    # rotate by u': H_rot[l,k,j] = sum_n eigvecs[n,k] * H[l,n,j]
    H_t = np.einsum('nk,lnj->lkj', eigvecs, H, optimize=True)  # (n_loci, n, J-1)
    H_t *= w_sqrt[None, :, None]                     # apply sqrt weights in-place

    # ── residualize H on X via fwl ───────────────────────────────────────────
    # xt_pinv = (X'X)^{-1} X', shape (p, n)
    xt_pinv  = np.linalg.solve(xtwx, xt.T)           # (p, n)
    # proj_coef[l,p,j] = sum_n xt_pinv[p,n] * H_t[l,n,j]
    proj     = np.einsum('pn,lnj->lpj', xt_pinv, H_t, optimize=True)  # (n_loci, p, J-1)
    H_perp   = H_t - np.einsum('np,lpj->lnj', xt, proj, optimize=True)  # (n_loci, n, J-1)

    # ── gram matrices and rhs ────────────────────────────────────────────────
    c_all = np.einsum('lnj,n->lj', H_perp, e0, optimize=True)       # (n_loci, J-1)
    G_all = np.einsum('lnj,lnk->ljk', H_perp, H_perp, optimize=True)  # (n_loci, J-1, J-1)
    G_all += np.eye(q)[None] * 1e-10                 # ridge for singular loci

    # solve G[l] beta_h[l] = c[l] for all loci simultaneously
    beta_h = np.linalg.solve(G_all, c_all)           # (n_loci, J-1)

    # ── rss, lod, p-values ───────────────────────────────────────────────────
    rss_red  = np.einsum('lj,lj->l', c_all, beta_h)  # (n_loci,) rss reduction
    rss1     = np.maximum(rss0 - rss_red, 1e-30)     # (n_loci,) alt rss
    lod_arr  = np.where(rss1 < rss0,
                        (n / 2.0) * np.log10(rss0 / rss1), 0.0)

    df2 = n - p - q
    if df2 > 0:
        f_stat   = np.maximum((rss0 - rss1) / q / (rss1 / df2), 0.0)
        pval_arr = 1.0 - stats.f.cdf(f_stat, q, df2)
    else:
        pval_arr = np.ones(n_loci)

    return {'locus_ids': locus_ids, 'positions': positions,
            'lod': lod_arr, 'p_value': pval_arr, 'betas': beta_h}
