"""
lmm — linear mixed model engine

implements the eigendecomposition-based fast LMM from
Kang et al. (2008, Genetics) as used in qtl2.

model:
    y = H γ + X α + u + ε
    u ~ N(0, σ²_g K)
    ε ~ N(0, σ²_e I)

after eigendecomposition K = U D U':
    rotate: y* = U'y,  X* = U'X,  H* = U'H
    Var(y*) = σ²_g D + σ²_e I  — diagonal, O(n) weights
    LOD = (n/2) log10(RSS_0 / RSS_1)       [ML,     h²   estimated once under null via REML  [
  - h² optimized with scipy.optimize.minimize_scalar on [0,1] + grid safeguard
  - brute check: compare optimized fit against h²=0 (brute-force safeguard)
  - LOD uses ML formula (n/2 log10 ratio), not REML
"""

import gc

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import logging

from ..models.records import LMMResult  # noqa: F401 — re-exported for callers

logger = logging.getLogger('mosaiqtl')

__all__: list[str] = []   # internal engine — no public symbols; import via scan.py


def estimate_h2_null(y_rot, x_rot, eigvals, method='ml'):
    """estimate heritability under the null model (no QTL effect).

    Uses ML (not REML) by default. The LOD score is a likelihood ratio test between
    null (no QTL) and alternative (+ haplotype H) — models with different fixed effects.
    REML likelihoods are not comparable across different fixed-effect structures, so
    the LRT is only valid under ML. This matches R miQTL's use.par='h2' (ML) default.
    REML is available as method='reml' for variance component estimation outside LRT.

    algorithm: 1. evaluate profile ML on a 20-point h² grid [0.01, 0.99]
      2. refine around the best grid point with Brent's method
      3. brute check: compare against h²=0 — take whichever has
         higher log-likelihood (brute-force safeguard)

    parameters
    ----------
    y_rot   : np.ndarray (n,)    U'y — rotated phenotype
    x_rot   : np.ndarray (n, p)  U'X — rotated covariates
    eigvals : np.ndarray (n,)    eigenvalues of kinship matrix
    method  : str  'ml' (default) or 'reml'
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
        w     = h2 * eigvals + (1.0 - h2)   # eigenvalues of V = h²K + (1-h²)I; > 0 always
        w_inv = 1.0 / w
        xw    = x_rot * w_inv[:, None]
        try:
            beta  = np.linalg.solve(xw.T @ x_rot, xw.T @ y_rot)
        except np.linalg.LinAlgError:
            return 1e10
        resid = y_rot - x_rot @ beta
        wrss  = float(resid @ (resid * w_inv))
        if not (wrss > 0):
            # catches wrss≤0 (degenerate phenotype) AND wrss=NaN (solve returned NaN)
            return 1e10
        logdet = float(np.sum(np.log(w)))
        if method == 'reml':
            df     = n - p
            sigma2 = wrss / df
            _, ldet_xtx = np.linalg.slogdet(xw.T @ x_rot)
            ll = -0.5 * (df*(np.log(2*np.pi*sigma2)+1) + logdet + ldet_xtx)
        else:
            sigma2 = wrss / n
            ll     = -0.5 * (n*(np.log(2*np.pi*sigma2)+1) + logdet)
        if not np.isfinite(ll):
            # slogdet returns -inf when xw.T@x_rot is near-singular (large h², large
            # eigvals down-weight covariate rows to near zero) → ll=+inf → nll=-inf
            return 1e10
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

    # ── step 3: brute check at h²=0 — take whichever has higher LL ────────────
    ll_zero = -neg_loglik(1e-6)
    if ll_zero > ll_opt:
        h2_opt = 1e-6
        ll_opt = ll_zero

    return h2_opt, ll_opt


def fit_lmm_at_locus(y_rot, h_rot, x_rot, eigvals, h2_null):
    """fit LMM at a single locus and compute test statistic.

    uses null h² (from estimate_h2_null) as the fixed variance
    component during the genome scan —

    LOD = (n/2) log10(RSS_0 / RSS_1)   [ML formula, n not n-p]

    parameters
    ----------
    y_rot    : np.ndarray (n,)   U'y
    h_rot    : np.ndarray (n, J) U'H  — all J founders; lstsq handles rank J-1
    x_rot    : np.ndarray (n, p) U'X
    eigvals  : np.ndarray (n,)     eigenvalues of LOCO kinship
    h2_null  : float               null heritability

    returns
    -------
    LMMResult
    """
    n = len(y_rot)
    q = h_rot.shape[1]
    p = x_rot.shape[1]

    w     = h2_null * eigvals + (1.0 - h2_null)
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

    # alternative model: y* = [H* X*] [γ; α] — lstsq handles rank-deficient H
    z_rot    = np.hstack([h_rot, x_rot])
    zw       = z_rot * np.sqrt(w_inv)[:, None]
    yw       = y_rot * np.sqrt(w_inv)
    beta_alt = np.linalg.lstsq(zw, yw, rcond=1e-10)[0]
    resid_alt = y_rot - z_rot @ beta_alt
    rss_alt   = float(resid_alt @ (resid_alt * w_inv))

    # LOD: ML formula (n not n-p)
    if rss_alt > 0 and rss_null > rss_alt:
        lod = (n / 2.0) * np.log10(rss_null / rss_alt)
    else:
        lod = 0.0

    # effective rank for correct df (lstsq result may have fewer active directions)
    q_eff = int(np.linalg.matrix_rank(
        np.hstack([h_rot, x_rot]) * np.sqrt(w_inv)[:, None], tol=1e-10)) - p
    q_eff = max(q_eff, 0)
    df2   = n - p - q_eff
    if df2 > 0 and rss_alt > 0 and q_eff > 0:
        f_stat  = ((rss_null - rss_alt) / q_eff) / (rss_alt / df2)
        f_stat  = max(f_stat, 0.0)
        p_value = 1.0 - stats.f.cdf(f_stat, q_eff, df2)
    else:
        f_stat  = 0.0
        p_value = 1.0

    sigma2_total = rss_alt / max(n - p - q_eff, 1)
    return LMMResult(
        lod=lod, p_value=p_value, h2=h2_null, f_stat=f_stat,
        betas=beta_alt[:q],          # first q cols = founder betas (J)
        sigma2_g=h2_null * sigma2_total,
        sigma2_e=(1 - h2_null) * sigma2_total,
        rss_null=rss_null, rss_alt=rss_alt,
    )


def precompute_chromosome(
    genomecache, chrom: str, eigvals: np.ndarray, eigvecs: np.ndarray,
    z_mat: 'np.ndarray | None' = None, model: str = 'additive',
) -> dict:
    """precompute the chromosome-level H rotation shared across all genes.

    the H rotation (and optional Z expansion) does not depend on the gene
    phenotype — only on the genotype cache and kinship eigenvectors.
    calling this once per chromosome and passing the result as ``chr_data``
    to scan_chromosome reduces per-gene cost from O(L·n·m·J) to O(L·n·J).

    parameters
    ----------
    genomecache : GenomeCache
    chrom       : str
    eigvals     : np.ndarray (n,)  eigenvalues of K_n (or K_m if z_mat is None)
    eigvecs     : np.ndarray (n, n)  corresponding eigenvectors
    z_mat       : np.ndarray (n, m) or None
    model       : 'additive' or 'full'

    returns
    -------
    dict with keys: locus_ids, positions, H_t_raw (n_loci, n, J), n_founders
    """
    locus_ids, positions, probs = genomecache.get_chromosome(chrom)
    nf = genomecache.n_founders

    if model == 'full':
        J = nf
        _ii, _jj = np.triu_indices(J)
        probs = probs[:, :, _ii] * probs[:, :, _jj]
        nf = len(_ii)

    H = probs  # (n_loci, m, J)
    if z_mat is not None:
        _z_rot = eigvecs.T @ z_mat                                    # (n, m)
        H_t_raw = np.einsum('ks,lsj->lkj', _z_rot, H, optimize=True)  # (n_loci, n, J)
        del _z_rot
    else:
        H_t_raw = np.einsum('nk,lnj->lkj', eigvecs, H, optimize=True)  # (n_loci, n, J)

    return {
        'locus_ids': locus_ids,
        'positions': positions,
        'H_t_raw':   H_t_raw,
        'n_founders': nf,
    }


def _scan_chromosome_core(
    y: np.ndarray,
    H_t_raw: np.ndarray,
    x_covariates: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    h2_null: float,
    locus_ids: 'list | np.ndarray',
    positions: np.ndarray,
    nf: int,
) -> dict:
    """inner scan kernel — takes pre-rotated (unweighted) H, applies per-gene w_sqrt.

    H_t_raw is chromosome-constant; calling this function skips the O(n_loci·n·m·J)
    rotation einsum that scan_chromosome pays on every call.  w_sqrt (O(n)) and
    H_t = H_t_raw * w_sqrt are O(n_loci·n·J) scalar multiply — cheap per gene.

    parameters
    ----------
    y        : np.ndarray (n,)
    H_t_raw  : np.ndarray (n_loci, n, J)  eigvec-rotated H, NOT yet weighted
    x_covariates : np.ndarray (n, p)
    eigvals  : np.ndarray (n,)
    eigvecs  : np.ndarray (n, n)
    h2_null  : float
    locus_ids, positions : metadata passed through to output dict
    nf       : int  number of founder columns in H_t_raw

    returns
    -------
    dict: locus_ids, positions, lod, p_value, betas (n_loci, nf)
    """
    n_loci = H_t_raw.shape[0]
    n      = len(y)
    p      = x_covariates.shape[1]

    w_sqrt = 1.0 / np.sqrt(h2_null * eigvals + (1.0 - h2_null))  # (n,)

    yt   = (eigvecs.T @ y) * w_sqrt
    xt   = (eigvecs.T @ x_covariates) * w_sqrt[:, None]
    xtwx = xt.T @ xt
    try:
        beta_null = np.linalg.solve(xtwx, xt.T @ yt)
    except np.linalg.LinAlgError:
        return {'locus_ids': locus_ids, 'positions': positions,
                'lod': np.zeros(n_loci), 'p_value': np.ones(n_loci),
                'betas': np.zeros((n_loci, nf))}
    e0   = yt - xt @ beta_null
    rss0 = float(e0 @ e0)

    H_t = H_t_raw * w_sqrt[None, :, None]               # (n_loci, n, J) — new array

    xt_pinv = np.linalg.solve(xtwx, xt.T)               # (p, n)
    proj    = np.einsum('pn,lnj->lpj', xt_pinv, H_t, optimize=True)
    H_perp  = H_t - np.einsum('np,lpj->lnj', xt, proj, optimize=True)

    c_all = np.einsum('lnj,n->lj',   H_perp, e0,     optimize=True)   # (n_loci, J)
    G_all = np.einsum('lnj,lnk->ljk', H_perp, H_perp, optimize=True)  # (n_loci, J, J)

    G_pinv  = np.linalg.pinv(G_all, rcond=1e-10)
    beta_h  = (G_pinv @ c_all[..., None]).squeeze(-1)

    rss_red = np.einsum('lj,lj->l', c_all, beta_h)
    rss1    = np.maximum(rss0 - rss_red, 1e-30)
    with np.errstate(divide='ignore', invalid='ignore'):
        lod_arr = np.where(rss1 < rss0, (n / 2.0) * np.log10(rss0 / rss1), 0.0)

    q_eff   = np.linalg.matrix_rank(G_all)
    df2_arr = np.maximum(n - p - q_eff, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        f_stat   = np.where(
            q_eff > 0,
            np.maximum((rss0 - rss1) / np.maximum(q_eff, 1) / (rss1 / df2_arr), 0.0),
            0.0)
        pval_arr = np.where(df2_arr > 0, 1.0 - stats.f.cdf(f_stat, q_eff, df2_arr), 1.0)

    return {'locus_ids': locus_ids, 'positions': positions,
            'lod': lod_arr, 'p_value': pval_arr, 'betas': beta_h,
            'rss0': rss0, 'rss1': rss1}


def scan_chromosome(
    y: np.ndarray,
    chr_data: dict,
    x_covariates: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    h2_null: float,
) -> dict:
    """scan one chromosome using precomputed chromosome data.

    call precompute_chromosome once per chromosome, then call this once per
    gene.  the O(n_loci·n·m·J) H rotation is paid once; only the cheap
    O(n_loci·n·J) per-gene weight scaling is repeated.

    parameters
    ----------
    y            : np.ndarray (n,)
    chr_data     : dict from precompute_chromosome — keys: locus_ids,
                   positions, H_t_raw (n_loci, n, J), n_founders
    x_covariates : np.ndarray (n, p)
    eigvals      : np.ndarray (n,)
    eigvecs      : np.ndarray (n, n)
    h2_null      : float

    returns
    -------
    dict: locus_ids, positions, lod, p_value, betas (n_loci, nf)
    """
    return _scan_chromosome_core(
        y, chr_data['H_t_raw'], x_covariates, eigvals, eigvecs, h2_null,
        chr_data['locus_ids'], chr_data['positions'], chr_data['n_founders'],
    )


def scan_chromosome_batch(
    Y_mat: np.ndarray,
    genomecache: 'GenomeCache',  # type: ignore[name-defined]  # noqa: F821
    chrom: str,
    x_covariates: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    h2_null: float,
    z_mat: 'np.ndarray | None' = None,
    model: str = 'additive',
    batch_size: int = 500,
    return_peak_positions: bool = False,
) -> 'np.ndarray | tuple[np.ndarray, np.ndarray]':
    """scan one chromosome for a batch of permuted phenotypes.

    precomputes H_perp and G_pinv once (genotype-only); only the null
    residual e0 varies per permutation.  designed for permutation threshold
    estimation with fix_h2=True (constant weight vector across all perms).

    savings vs calling scan_chromosome N_perms times:
      - H_perp einsum: O(n_loci × n × J) computed once, not N_perms times
      - G_pinv batched SVD: O(n_loci × J³) computed once, not N_perms times
      - per-perm work reduces to O(n_loci × J × batch_size)

    parameters
    ----------
    Y_mat      : (n, N_perms)  permuted/null phenotype columns,
                 already rint-transformed and z_mat-reduced if applicable
    genomecache
    chrom
    x_covariates : (n, p)  at scan level (strain-mean if z_mat was applied)
    eigvals    : (n,)
    eigvecs    : (n, n)
    h2_null    : float  from fix_h2 pre-estimation
    z_mat      : np.ndarray (n, m) or None
                 when provided, H is expanded from strain level (m) to sample
                 level (n) via the fused rotation z_rot = eigvecs.T @ z_mat,
                 matching the sample-level model in scan_chromosome.
                 Y_mat, x_covariates, eigvals, eigvecs must all be at sample
                 level (n) when z_mat is supplied.
    model      : 'additive' or 'full'
    batch_size : int  mini-batch size over the N_perms axis (memory control)

    returns
    -------
    np.ndarray (N_perms,)  per-permutation max LOD on this chromosome, or
    tuple (max_lods, peak_positions) when return_peak_positions=True
    """
    locus_ids, positions_c, probs = genomecache.get_chromosome(chrom)
    nf      = genomecache.n_founders
    n       = eigvals.shape[0]
    n_perms = Y_mat.shape[1]

    if model == 'full':
        J = nf
        _ii, _jj = np.triu_indices(J)
        probs = probs[:, :, _ii] * probs[:, :, _jj]
        nf = len(_ii)

    # ── genotype-only precomputation (same for every permutation) ─────────────
    w_sqrt = 1.0 / np.sqrt(h2_null * eigvals + (1.0 - h2_null))  # (n,)

    xt      = (eigvecs.T @ x_covariates) * w_sqrt[:, None]  # (n, p)
    xtwx    = xt.T @ xt                                      # (p, p)
    xt_pinv = np.linalg.solve(xtwx, xt.T)                   # (p, n)

    H = probs                                                # (n_loci, m, J)
    if z_mat is not None:
        # fuse Z expansion + eigvec rotation: same optimisation as scan_chromosome
        _z_rot = eigvecs.T @ z_mat                                   # (n, m)
        H_t    = np.einsum('ks,lsj->lkj', _z_rot, H, optimize=True)  # (n_loci, n, J)
        del _z_rot
    else:
        H_t = np.einsum('nk,lnj->lkj', eigvecs, H, optimize=True)   # (n_loci, n, J)
    H_t *= w_sqrt[None, :, None]

    proj   = np.einsum('pn,lnj->lpj', xt_pinv, H_t, optimize=True)    # (n_loci, p, J)
    H_perp = H_t - np.einsum('np,lpj->lnj', xt, proj, optimize=True)  # (n_loci, n, J)
    del H, H_t, proj
    gc.collect()

    G_all  = np.einsum('lnj,lnk->ljk', H_perp, H_perp, optimize=True)  # (n_loci, J, J)
    G_pinv = np.linalg.pinv(G_all, rcond=1e-10)                         # (n_loci, J, J)
    del G_all
    gc.collect()

    # ── mini-batch loop over permutations ─────────────────────────────────────
    max_lods  = np.full(n_perms, -np.inf)
    peak_idxs = np.zeros(n_perms, dtype=np.intp) if return_peak_positions else None

    for p0 in range(0, n_perms, batch_size):
        p1   = min(p0 + batch_size, n_perms)
        Y_b  = Y_mat[:, p0:p1]                               # (n, b)

        YT   = (eigvecs.T @ Y_b) * w_sqrt[:, None]           # (n, b)
        e0   = YT - xt @ (xt_pinv @ YT)                      # (n, b)
        rss0 = (e0 * e0).sum(axis=0)                          # (b,)

        c    = np.einsum('lnj,nb->ljb', H_perp, e0, optimize=True)    # (n_loci, J, b)
        Gc   = np.einsum('ljk,lkb->ljb', G_pinv, c, optimize=True)    # (n_loci, J, b)
        rss_red = np.einsum('ljb,ljb->lb', c, Gc, optimize=True)      # (n_loci, b)
        del c, Gc

        rss1    = np.maximum(rss0[None, :] - rss_red, 1e-30)          # (n_loci, b)
        with np.errstate(divide='ignore', invalid='ignore'):
            lod_b = np.where(
                rss1 < rss0[None, :],
                (n / 2.0) * np.log10(rss0[None, :] / rss1),
                0.0,
            )                                                          # (n_loci, b)
        max_lods[p0:p1] = lod_b.max(axis=0)
        if peak_idxs is not None:
            peak_idxs[p0:p1] = lod_b.argmax(axis=0)
        del rss_red, rss1, lod_b, e0, YT

    del H_perp, G_pinv
    gc.collect()

    if return_peak_positions:
        return max_lods, positions_c[peak_idxs]
    return max_lods


def scan_chromosome_mi(
    y: np.ndarray,
    genomecache: 'GenomeCache',  # type: ignore[name-defined]  # noqa: F821
    chrom: str,
    x_covariates: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    h2_null: float,
    n_imp: int = 11,
    rng: 'np.random.Generator | None' = None,
    loci_batch: int = 200,
) -> np.ndarray:
    """scan one chromosome under multiple imputation — vectorised over imputations.

    generates n_imp hard diplotype assignments from soft haplotype probs,
    then computes LOD for all imputations in a single vectorised pass and
    returns the per-locus mean.

    savings vs calling scan_chromosome n_imp times:
      - all n_imp one-hot draws generated once per chromosome
      - H_perp / G / G_pinv computed in a single einsum pass over
        the (n_imp, loci_batch, n, J) tensor
      - eliminates n_imp GenomeCache object allocations
      - eliminates n_imp × n_chr scan_chromosome calls

    parameters
    ----------
    y            : (n,)     strain-mean phenotype (already rint-transformed)
    genomecache
    chrom        : str
    x_covariates : (n, p)   strain-mean covariates
    eigvals      : (n,)
    eigvecs      : (n, n)
    h2_null      : float    fixed h² — same for all imputations (fix_h2=True rule)
    n_imp        : int      number of imputations (default 11)
    rng          : Generator or None  — uses default_rng(0) when None
    loci_batch   : int      mini-batch size over loci axis (memory control)

    returns
    -------
    np.ndarray (n_loci,)  mean LOD across n_imp imputations
    """
    if rng is None:
        rng = np.random.default_rng(0)

    locus_ids, positions, probs = genomecache.get_chromosome(chrom)
    n_loci, n_s, n_f = probs.shape
    n = len(y)

    # ── shared weight vector from h² (identical for all imputations) ────────
    w_sqrt = 1.0 / np.sqrt(h2_null * eigvals + (1.0 - h2_null))  # (n,)

    # ── null model (once for all imputations) ────────────────────────────────
    yt   = (eigvecs.T @ y) * w_sqrt                         # (n,)
    xt   = (eigvecs.T @ x_covariates) * w_sqrt[:, None]     # (n, p)
    xtwx = xt.T @ xt                                        # (p, p)
    try:
        beta_null = np.linalg.solve(xtwx, xt.T @ yt)
    except np.linalg.LinAlgError:
        return np.zeros(n_loci)
    e0   = yt - xt @ beta_null                              # (n,)
    rss0 = float(e0 @ e0)
    xt_pinv = np.linalg.solve(xtwx, xt.T)                  # (p, n)

    # ── generate all n_imp one-hot draws at once ─────────────────────────────
    # probs: (n_loci, n_s, n_f) — CDF along founder axis for each (locus, strain)
    cum = probs.cumsum(axis=2)                              # (n_loci, n_s, n_f)
    u   = rng.random((n_imp, n_loci, n_s, 1))              # (n_imp, n_loci, n_s, 1)
    # argmax of CDF > u gives the sampled founder index per (imp, locus, strain)
    idx = (u > cum[None]).sum(axis=3).clip(0, n_f - 1)     # (n_imp, n_loci, n_s)

    # convert to one-hot: imp_h[i,l,s,j] = 1 iff founder j drawn for (i,l,s)
    imp_h = np.zeros((n_imp, n_loci, n_s, n_f), dtype=np.float32)
    _i = np.arange(n_imp)[:, None, None]
    _l = np.arange(n_loci)[None, :, None]
    _s = np.arange(n_s)[None, None, :]
    imp_h[_i, _l, _s, idx] = 1.0
    del u, idx, cum

    # ── vectorised scan over loci mini-batches ───────────────────────────────
    lod_sum = np.zeros(n_loci, dtype=np.float64)

    for b0 in range(0, n_loci, loci_batch):
        b1  = min(b0 + loci_batch, n_loci)
        b   = b1 - b0
        # H_b: (n_imp, b, n_s, n_f) → rotate → (n_imp, b, k, n_f)
        # k is the eigenvector-space coordinate (size n); same as scan_chromosome 'nk,lnj->lkj'
        H_b = imp_h[:, b0:b1, :, :].astype(np.float64)    # (n_imp, b, n_s, n_f)

        # rotate: H_t[i,b,k,j] = sum_n eigvecs[n,k] * H[i,b,n,j]
        H_t = np.einsum('nk,ibnj->ibkj', eigvecs, H_b, optimize=True)  # (n_imp, b, n, n_f)
        H_t *= w_sqrt[None, None, :, None]

        # FWL: H_perp = H_t - xt @ (xt_pinv @ H_t)
        # xt_pinv: (p, k); H_t: (n_imp, b, k, n_f)
        proj   = np.einsum('pk,ibkj->ibpj', xt_pinv, H_t, optimize=True)   # (n_imp, b, p, n_f)
        H_perp = H_t - np.einsum('kp,ibpj->ibkj', xt, proj, optimize=True)  # (n_imp, b, k, n_f)
        del H_b, H_t, proj

        # gram matrices — contract over k (rotated strain space)
        G_b    = np.einsum('ibkj,ibkl->ibjl', H_perp, H_perp, optimize=True)  # (n_imp, b, n_f, n_f)
        G_pinv = np.linalg.pinv(
            G_b.reshape(-1, n_f, n_f), rcond=1e-10
        ).reshape(n_imp, b, n_f, n_f)  # (n_imp, b, n_f, n_f)

        # per-imputation LOD
        c      = np.einsum('ibkj,k->ibj', H_perp, e0, optimize=True)           # (n_imp, b, n_f)
        Gc     = np.einsum('ibjl,ibl->ibj', G_pinv, c, optimize=True)          # (n_imp, b, n_f)
        rss_red = np.einsum('ibj,ibj->ib', c, Gc, optimize=True)              # (n_imp, b)
        del H_perp, G_b, G_pinv, c, Gc

        rss1  = np.maximum(rss0 - rss_red, 1e-30)                             # (n_imp, b)
        with np.errstate(divide='ignore', invalid='ignore'):
            lod_ib = np.where(rss1 < rss0,
                              (n / 2.0) * np.log10(rss0 / rss1), 0.0)        # (n_imp, b)
        lod_sum[b0:b1] += lod_ib.sum(axis=0)                                  # accumulate
        del rss_red, rss1, lod_ib

    del imp_h
    gc.collect()

    return lod_sum / n_imp   # (n_loci,) mean LOD
