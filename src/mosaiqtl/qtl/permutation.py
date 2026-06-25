"""
permutation — permutation thresholds and parametric bootstrap CI

implements significance thresholds for genome-wide scans via permutation
and positional confidence intervals via parametric bootstrap.

  generate.sample.outcomes.matrix()  →  generate_null_outcomes()
  run.threshold.scans()              →  run_threshold_scans()
  get.gev.thresholds()               →  get_gev_thresholds()
  run.positional.scans()             →  run_positional_scans()

design notes:
  - fix_h2=True (default) re-uses h² estimated from original data across all
    permutations, matching the validate_port.py inline loop and saving
    n_perms × n_chrom calls to estimate_h2_null.
  - chrom kwarg restricts threshold scans to a single chromosome (fast single-
    gene mode used in validation); leave None for full-genome threshold.
  - run_positional_scans operates in genotype space (m samples), consistent
    with kinship_decomp which is always (m×m) before Z-expansion.
  - permutation_method='parametric' (default) mirrors miQTL sampling.processes.R
    lines 110-137: draws u~N(0,K·σ²_g), e~N(0,σ²_e·I), rank-permutes y using the
    parametric sample's order. Correct for real LOCO kinship (Phase 2+).
    permutation_method='plain' shuffles y labels — valid only when τ²≈0 (identity
    kinship / OLS); used by Phase 1 validation only.
"""

import gc
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy.stats import genextreme

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'KinshipSingularError',
    'generate_null_outcomes',
    'run_threshold_scans',
    'get_gev_thresholds',
    'run_positional_scans',
]


class KinshipSingularError(ValueError):
    """raised when a LOCO kinship matrix is genuinely non-PSD.

    threshold: min_eig < -(max_eig × n² × machine_epsilon) — n² accounts for
    Cholesky floating-point accumulation (O(n²) ops), not just eigenvalue
    computation noise (O(n×eps)).  n×eps was too tight and missed failures.
    """


# ─────────────────────────────────────────────────────────────────────────────
# null outcome generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_null_outcomes(y, n_perms=1000, method='permutation', seed=1):
    """generate null phenotype columns for threshold estimation.

    for permutation: shuffles phenotype labels (breaks geno-pheno link).
    for bootstrap: resamples with replacement from centred phenotype.


    parameters
    ----------
    y       : np.ndarray (n,)  original phenotype (NaN-free)
    n_perms : int              number of permuted/bootstrapped columns to draw
    method  : str              'permutation' (default) or 'bootstrap'
    seed    : int

    returns
    -------
    np.ndarray (n, n_perms)  null phenotype matrix
    """
    rng = np.random.default_rng(seed)   # reproducible rng

    if method == 'permutation':
        # shuffle rows independently for each column
        return np.column_stack([rng.permutation(y) for _ in range(n_perms)])
    elif method == 'bootstrap':
        y_c = y - y.mean()              # centre before resampling
        return np.column_stack([
            rng.choice(y_c, size=len(y), replace=True)
            for _ in range(n_perms)])
    else:
        raise ValueError(f'method must be permutation or bootstrap, got {method!r}')


# ─────────────────────────────────────────────────────────────────────────────
# parametric null helpers
# ─────────────────────────────────────────────────────────────────────────────

def _null_fit_params(
    y_rot: np.ndarray,
    x_rot: np.ndarray,
    eigvals: np.ndarray,
    h2: float,
    use_reml: bool = True,
) -> tuple[np.ndarray, float, float]:
    """recover beta_hat and variance components at a given h².

    parameters
    ----------
    y_rot   : (n,)    U'y — rotated phenotype
    x_rot   : (n, p)  U'X — rotated covariates
    eigvals : (n,)    eigenvalues of kinship matrix
    h2      : float   heritability
    use_reml: bool

    returns
    -------
    beta      : (p,)  fixed-effect estimates
    sigma2_g  : float genetic variance  (= h2 * sigma2_total)
    sigma2_e  : float residual variance (= (1-h2) * sigma2_total)
    """
    if x_rot.ndim == 1:
        x_rot = x_rot[:, None]
    n, p = len(y_rot), x_rot.shape[1]
    w_inv = 1.0 / (h2 * eigvals + (1.0 - h2))
    xw    = x_rot * w_inv[:, None]
    beta  = np.linalg.solve(xw.T @ x_rot, xw.T @ y_rot)
    resid = y_rot - x_rot @ beta
    wrss  = float(resid @ (resid * w_inv))
    df    = (n - p) if use_reml else n
    sigma2_total = wrss / df
    return beta, h2 * sigma2_total, (1.0 - h2) * sigma2_total


def _draw_kinship_re(
    K: np.ndarray,
    sigma2_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """draw u ~ N(0, K·sigma2_g) with PSD guard.

    raises KinshipSingularError when min eigenvalue of K is more negative than
    max_eig × n² × machine_epsilon — n² accounts for Cholesky O(n²) floating-point
    accumulation, not just eigenvalue computation error (n×eps too tight).
    """
    eigs    = np.linalg.eigvalsh(K)
    min_eig = float(eigs.min())
    psd_tol = float(eigs.max()) * len(K) ** 2 * float(np.finfo(float).eps)
    if min_eig < -psd_tol:
        raise KinshipSingularError(
            f'LOCO kinship not PSD (min_eig={min_eig:.2e}); '
            'skipping chromosome — mirrors R behaviour on singular K'
        )
    return rng.multivariate_normal(np.zeros(len(K)), K * sigma2_g)


def _draw_parametric_batch(
    y: np.ndarray,
    x: np.ndarray,
    K: np.ndarray,
    evc: np.ndarray,
    ev: np.ndarray,
    h2: float,
    use_reml: bool,
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """draw n_perms parametric null phenotypes simultaneously.

    vectorised version of _parametric_null_col: draws all u and e vectors
    at once, then rank-permutes y by each null column's order.
    PSD check done once instead of once per permutation.

    raises KinshipSingularError if K is not PSD.

    returns
    -------
    np.ndarray (n, n_perms)  each column is y rank-permuted by one null draw
    """
    y_rot = evc.T @ y
    x_rot = evc.T @ x
    beta, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, ev, h2, use_reml)

    eigs    = np.linalg.eigvalsh(K)
    min_eig = float(eigs.min())
    psd_tol = float(eigs.max()) * len(K) ** 2 * float(np.finfo(float).eps)
    if min_eig < -psd_tol:
        raise KinshipSingularError(
            f'LOCO kinship not PSD (min_eig={min_eig:.2e}); '
            'skipping chromosome — mirrors R behaviour on singular K'
        )

    # draw all u ~ N(0, K·sigma2_g) at once: (n_perms, n) → (n, n_perms)
    U_mat  = rng.multivariate_normal(
        np.zeros(len(y)), K * sigma2_g, size=n_perms).T
    E_mat  = rng.normal(0.0, np.sqrt(max(sigma2_e, 0.0)),
                        size=(len(y), n_perms))
    Y_null = (x @ beta)[:, None] + U_mat + E_mat           # (n, n_perms)

    # rank-permute y by each column's argsort — vectorised
    return y[np.argsort(Y_null, axis=0)]                    # (n, n_perms)


def _parametric_null_col(
    y: np.ndarray,
    x: np.ndarray,
    K: np.ndarray,
    evc: np.ndarray,
    ev: np.ndarray,
    h2: float,
    use_reml: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """one kinship-informed parametric null column.

    mirrors miQTL sampling.processes.R lines 110-137:
      u ~ N(0, K·sigma2_g)
      e ~ N(0, sigma2_e·I)
      y_null = X@beta_hat + u + e
      return y[argsort(y_null)]   # rank-permute y by y_null's order

    parameters
    ----------
    y    : (m,)    original strain-level phenotype (after RINT if requested)
    x    : (m, p)  strain-level covariates
    K    : (m, m)  raw LOCO kinship for this chromosome
    evc  : (m, m)  eigenvectors of K (from kinship_decomp)
    ev   : (m,)    eigenvalues of K
    h2   : float   heritability under the null
    use_reml : bool
    rng  : np.random.Generator

    returns
    -------
    (m,)  rank-permuted y (same values as y, different order)

    raises
    ------
    KinshipSingularError  if K is not PSD — caller should skip the chromosome
    """
    y_rot = evc.T @ y
    x_rot = evc.T @ x
    beta, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, ev, h2, use_reml)
    u      = _draw_kinship_re(K, sigma2_g, rng)
    e      = rng.normal(0.0, np.sqrt(max(sigma2_e, 0.0)), size=len(y))
    y_null = x @ beta + u + e
    return y[np.argsort(y_null)]


# ─────────────────────────────────────────────────────────────────────────────
# threshold scans
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_scans(null_outcomes, genomecache,
                        phenotype=None,
                        covariates=None, kinship_decomp=None,
                        kinship_loco=None,
                        chrom=None,
                        ref_chrom=None,
                        fix_h2=True,
                        permutation_method='parametric',
                        use_rint=True, use_reml=True,
                        sample_ids=None,
                        z_mat=None,
                        seed=1,
                        batch_size=500):
    """scan null phenotype columns to build the peak-LOD null distribution.


    parameters
    ----------
    null_outcomes  : np.ndarray (n, n_perms)  from generate_null_outcomes().
        in 'parametric' mode the array is used only to determine n_perms;
        its contents are ignored.
    genomecache   : GenomeCache
    phenotype      : np.ndarray (n,) or None
        original phenotype; required when fix_h2=True to pre-estimate h².
        ignored when fix_h2=False (h² re-estimated per permutation).
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
        if None, computed from genomecache (loco).
    kinship_loco   : dict or None  {chrom: np.ndarray (m, m)}
        raw LOCO kinship matrices (not decomposed).  used when
        permutation_method='parametric'; if None, reconstructed from
        kinship_decomp automatically.  ignored for 'plain'.
    chrom          : str or None
        if given, restrict scans to this chromosome only (validation mode).
        if None, scan all chromosomes (full-genome threshold).
    ref_chrom      : str or None
        chromosome whose LOCO kinship and h² are used to draw the null
        phenotype matrix in 'parametric' mode.  all chromosomes are then
        scanned with this single null matrix — this is correct for a
        genome-wide FWE threshold (see notes).  if None, the first PSD
        chromosome in `chrs` is used.  mirrors R's behaviour: miQTL's
        `generate.sample.outcomes.matrix(scan.object)` draws all permuted
        phenotypes using `scan.object$fit0$K`, which is the LOCO kinship
        of the chromosome on which `scan.h2lmm` was called (typically the
        peak chromosome).
    fix_h2         : bool  default True
        True  — estimate h² once from `phenotype` and reuse across perms.
        False — re-estimate h² for each permuted phenotype (slow).
    permutation_method : str  'parametric' (default) or 'plain'
        'parametric' — kinship-informed rank-permutation matching miQTL
                       sampling.processes.R lines 110-137; draws u~N(0,K·σ²_g),
                       rank-permutes y by the null order.  correct default.
        'plain'      — simple shuffle; equivalent only when τ²≈0 (identity
                       kinship / OLS).  used by Phase 1 validation only.
    use_rint       : bool  apply rank-inverse normal transform to phenotype
    use_reml       : bool  REML for h² estimation
    sample_ids     : list of str or None
        replicate sample IDs (length n).  when provided, each permuted
        phenotype is averaged to strain means via the Z-matrix before
        scanning; kinship_decomp must be at strain level (m×m).
        Mutually exclusive with z_mat: use one or the other, not both.
    z_mat          : np.ndarray (n, m) or None
        sample-to-strain design matrix for n-level scanning.  when provided,
        null_outcomes and kinship_decomp must already be at n-level (i.e.
        kinship_decomp holds eigdecomp of K_n = Z K_m Z^T).  z_mat is passed
        through to scan_chromosome_batch so haplotype probs are expanded from
        m cache strains to n samples inside each scan.  this ensures the null
        LOD scale matches a sample-level scan (LOD = (n/2)·log10(rss0/rss1))
        rather than a strain-mean scan (LOD = (m/2)·log10(rss0/rss1)).
        sample_ids must be None when z_mat is provided.
    seed           : int  rng seed for all internal random draws.
        'parametric' — seeds the null-draw rng inside _draw_parametric_batch.
        'plain'      — seed is unused here; reproducibility is controlled by
                       the seed passed to generate_null_outcomes by the caller.
    batch_size     : int  mini-batch size for the fast vectorised path
        (fix_h2=True + plain).  500 balances memory and loop overhead;
        reduce if RAM is tight.

    returns
    -------
    max_lods       : np.ndarray (n_perms,)  per-permutation genome-wide max LOD
    chr_null_lods  : dict {chr: np.ndarray (n_perms,)}
        per-chromosome null max-LOD distributions — already computed during the
        scan at no extra cost. use these to derive calibrated per-gene empirical
        p-values: p_g = mean(chr_null_lods[gene_chr] >= peak_lod_g), which avoids
        the conservative bias of comparing against the genome-wide maximum.

    notes
    -----
    fast path (fix_h2=True + permutation_method='plain'):
        precomputes H_perp and G_pinv once per chromosome then vectorises
        over all permutations in mini-batches via scan_chromosome_batch.
        O(n_perms × n_loci × J) vs O(n_perms × n_loci × J²) for slow path.
    slow path (fix_h2=False or permutation_method='parametric'):
        calls scan_chromosome once per permutation per chromosome.
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if permutation_method not in ('plain', 'parametric'):
        raise ValueError(
            f"permutation_method must be 'plain' or 'parametric', got {permutation_method!r}")

    # build kinship decomp if not supplied
    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    # reconstruct raw kinship matrices from eigendecomposition when not supplied
    if permutation_method == 'parametric' and kinship_loco is None:
        kinship_loco = {
            c: (kinship_decomp[c]['vectors']
                * kinship_decomp[c]['values'][None, :]
                @ kinship_decomp[c]['vectors'].T)
            for c in kinship_decomp
        }

    # chromosomes to scan
    chrs = ([chrom] if chrom is not None
            else [c for c in genomecache.chromosomes if c in kinship_decomp])

    # covariates default to intercept
    n = null_outcomes.shape[0]
    x = (covariates if covariates is not None
         else np.ones((n, 1), dtype=float))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):            # ensure intercept present
        x = np.hstack([np.ones((n, 1)), x])

    # sample_ids path: reduce phenotype/covariates to strain means; scan at m-level.
    # use when no z_mat is supplied (no replicated design or single-level data).
    _z_red: 'np.ndarray | None' = None
    _r_counts: 'np.ndarray | None' = None
    x_scan = x
    if sample_ids is not None and z_mat is None:
        from ..utils import build_z_matrix
        _z_red, _ = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        _r_counts = _z_red.sum(axis=0)                   # (m,) replicates per strain
        x_scan = (_z_red.T @ x) / _r_counts[:, None]     # (m, p) strain-mean covariates

    # z_mat path: expand from m cache strains to n samples at n-level.
    # build K_n = z_mat @ K_m @ z_mat.T per chromosome; use n-level eigendecomps
    # so null LOD scale matches sample-level scan: LOD = (n/2)·log10(rss0/rss1).
    _n_kd: 'dict | None' = None
    if z_mat is not None:
        _n_kd = {}
        for c in chrs:
            ev_m = kinship_decomp[c]['values']
            evc_m = kinship_decomp[c]['vectors']
            K_m = (evc_m * ev_m) @ evc_m.T
            K_n = z_mat @ K_m @ z_mat.T
            ev_n, evc_n = np.linalg.eigh(K_n)
            _n_kd[c] = {'values': ev_n, 'vectors': evc_n, 'K_n': K_n}

    def _ev_evc(c: str) -> 'tuple[np.ndarray, np.ndarray]':
        """return eigendecomp: n-level when z_mat provided, else m-level."""
        if _n_kd is not None:
            return _n_kd[c]['values'], _n_kd[c]['vectors']
        return kinship_decomp[c]['values'], kinship_decomp[c]['vectors']

    def _K_raw(c: str) -> np.ndarray:
        """return raw kinship for parametric draws: K_n when z_mat, else K_m."""
        if _n_kd is not None:
            return _n_kd[c]['K_n']
        return kinship_loco[c]

    # pre-compute per-chrom h² from original phenotype (fix_h2=True mode)
    h2_fixed = {}
    if fix_h2:
        if phenotype is None:
            raise ValueError('phenotype required when fix_h2=True')
        y0 = _rint(phenotype) if use_rint else phenotype.copy()
        if _r_counts is not None:
            y0 = (_z_red.T @ y0) / _r_counts   # strain means (RINT already applied above)
        for c in chrs:
            ev, evc = _ev_evc(c)
            h2_fixed[c], _ = lmm_module.estimate_h2_null(
                evc.T @ y0, evc.T @ x_scan, ev,
                method='reml' if use_reml else 'ml')
    else:
        if phenotype is not None:
            y0 = _rint(phenotype) if use_rint else phenotype.copy()
            if _r_counts is not None:
                y0 = (_z_red.T @ y0) / _r_counts
        else:
            y0 = None

    n_perms  = null_outcomes.shape[1]
    max_lods = np.zeros(n_perms)
    rng      = np.random.default_rng(seed) if permutation_method == 'parametric' else None

    if fix_h2 and permutation_method in ('plain', 'parametric'):
        plain_mat = None
        if permutation_method == 'plain':
            if use_rint:
                plain_mat = np.column_stack([_rint(null_outcomes[:, i])
                                             for i in range(n_perms)])
            else:
                plain_mat = null_outcomes.astype(float)
            if _r_counts is not None:
                plain_mat = (_z_red.T @ plain_mat) / _r_counts[:, None]  # (m, n_perms)

        # parametric: draw null phenotypes ONCE from the first valid chromosome's
        # kinship then reuse across all chromosomes — mirrors R's
        # run.threshold.scans which scans each permuted phenotype genome-wide
        # from a single kinship draw.  per-chromosome independent draws inflate
        # the genome-wide max-LOD by taking the max over 20 independent nulls.
        if permutation_method == 'parametric':
            ref_chrom = None
            for c in chrs:
                try:
                    ev_r, evc_r = _ev_evc(c)
                    null_mat_gw = _draw_parametric_batch(
                        y0, x_scan, _K_raw(c), evc_r, ev_r,
                        h2_fixed[c], use_reml, n_perms, rng)
                    ref_chrom = c
                    break
                except KinshipSingularError as exc:
                    logger.warning('  chr%s  [skip for ref draw] %s', c, exc)
            if ref_chrom is None:
                logger.warning('  all chromosomes singular — returning zeros')
                return max_lods
        else:
            null_mat_gw = plain_mat

        chr_null_lods: dict = {}
        for c in chrs:
            ev, evc = _ev_evc(c)
            h2  = h2_fixed[c]

            chr_max = lmm_module.scan_chromosome_batch(
                null_mat_gw, genomecache, c, x_scan, ev, evc, h2,
                z_mat=z_mat, batch_size=batch_size)
            chr_null_lods[c] = chr_max
            max_lods = np.maximum(max_lods, chr_max)
            logger.info('  [batch] chr%s  max_lod=%.3f', c, float(chr_max.max()))

        return max_lods, chr_null_lods

    # ── slow path: fix_h2=False ───────────────────────────────────────────────
    # precompute H rotation once per chromosome; only phenotype varies per perm
    _chr_data_cache = {c: lmm_module.precompute_chromosome(genomecache, c, *_ev_evc(c))
                       for c in chrs}
    chr_null_lods = {c: np.zeros(n_perms) for c in chrs}
    for p_idx in range(n_perms):
        if (p_idx + 1) % 100 == 0:
            logger.info(f'  threshold scan {p_idx+1}/{n_perms}')

        best = -np.inf

        for c in chrs:
            ev, evc = _ev_evc(c)

            if permutation_method == 'parametric':
                h2 = h2_fixed[c] if fix_h2 else None
                if not fix_h2:
                    h2, _ = lmm_module.estimate_h2_null(
                        evc.T @ y0, evc.T @ x_scan, ev,
                        method='reml' if use_reml else 'ml')
                try:
                    y_p = _parametric_null_col(
                        y0, x_scan, _K_raw(c), evc, ev, h2, use_reml, rng)
                except KinshipSingularError as exc:
                    logger.warning(f'  chr{c}  perm {p_idx}  [skip] {exc}')
                    continue
            else:
                y_p = null_outcomes[:, p_idx]
                if use_rint:
                    y_p = _rint(y_p)
                if _r_counts is not None:
                    y_p = (_z_red.T @ y_p) / _r_counts
                if fix_h2:
                    h2 = h2_fixed[c]
                else:
                    h2, _ = lmm_module.estimate_h2_null(
                        evc.T @ y_p, evc.T @ x_scan, ev,
                        method='reml' if use_reml else 'ml')

            res = lmm_module.scan_chromosome(y_p, _chr_data_cache[c], x_scan, ev, evc, h2)
            chr_lod = float(res['lod'].max())
            chr_null_lods[c][p_idx] = chr_lod
            best = max(best, chr_lod)

        max_lods[p_idx] = best

    return max_lods, chr_null_lods


# ─────────────────────────────────────────────────────────────────────────────
# gev threshold
# ─────────────────────────────────────────────────────────────────────────────

def get_gev_thresholds(max_lods, percentile=0.95):
    """fit a generalised extreme value (GEV) distribution to the
    permutation peak-LOD distribution and return the threshold.

    fitting GEV is more stable than empirical quantiles when n_perms
    is small (100-500).


    parameters
    ----------
    max_lods   : np.ndarray (n_perms,)  per-permutation max LOD scores
    percentile : float  significance percentile (default 0.95 = 5% genome-wide)

    returns
    -------
    float  LOD threshold at the requested percentile
    dict   {'c', 'loc', 'scale', 'threshold', 'percentile',
            'n_perms', 'empirical_quantile'}
    """
    finite = max_lods[np.isfinite(max_lods)]              # drop any inf/nan
    if len(finite) < 10:
        raise ValueError(f'too few finite max-LOD values ({len(finite)}) for GEV fit')

    c, loc, scale = genextreme.fit(finite)                 # scipy GEV fit
    threshold = float(genextreme.ppf(percentile, c, loc=loc, scale=scale))

    params = {
        'c': float(c), 'loc': float(loc), 'scale': float(scale),
        'threshold': threshold, 'percentile': percentile,
        'n_perms': len(finite),
        'empirical_quantile': float(np.quantile(finite, percentile)),
    }
    return threshold, params


# ─────────────────────────────────────────────────────────────────────────────
# positional CI via parametric bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def run_positional_scans(bootstrap_outcomes, genomecache,
                         phenotype,
                         covariates=None, kinship_decomp=None,
                         peak_chrom=None,
                         fix_h2=True,
                         use_rint=True, use_reml=True,
                         ci_alpha=0.05,
                         sample_ids=None):
    """run single-chromosome scans on parametric bootstrap samples to compute
    a positional confidence interval for a QTL.


    parameters
    ----------
    bootstrap_outcomes : np.ndarray (n, n_boot)  from generate_null_outcomes(method='bootstrap')
    genomecache       : GenomeCache
    phenotype          : np.ndarray (n,)  original phenotype (for fix_h2 pre-estimation)
    covariates         : np.ndarray (n, p) or None
    kinship_decomp     : dict or None  {chrom: {'values':, 'vectors':}}
    peak_chrom         : str or None
        chromosome to scan; if None, scans all and takes overall peak position.
    fix_h2             : bool  reuse h² from original data (faster)
    use_rint           : bool
    use_reml           : bool
    ci_alpha           : float  confidence level = 1 - ci_alpha (default 0.05 = 95% CI)
    sample_ids         : list of str or None
        replicate sample IDs (length n).  when provided, each bootstrap
        phenotype is averaged to strain means before scanning; kinship_decomp
        must be at strain level (m×m).

    returns
    -------
    np.ndarray (n_boot,)   bootstrap peak positions in Mb
    dict  {'ci_lo', 'ci_hi', 'ci': (lo, hi), 'n_boot', 'n_valid', 'alpha'}
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    chrs = ([peak_chrom] if peak_chrom is not None
            else [c for c in genomecache.chromosomes if c in kinship_decomp])

    n = bootstrap_outcomes.shape[0]
    x = (covariates if covariates is not None
         else np.ones((n, 1), dtype=float))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    # replicate path: reduce to strain means if sample_ids provided
    z_mat    = None
    r_counts = None
    x_scan   = x
    if sample_ids is not None:
        from ..utils import build_z_matrix
        z_mat, _  = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        r_counts  = z_mat.sum(axis=0)
        x_scan    = (z_mat.T @ x) / r_counts[:, None]

    # pre-compute h² from original phenotype
    h2_fixed = {}
    if fix_h2:
        y0 = _rint(phenotype) if use_rint else phenotype.copy()
        if z_mat is not None:
            y0 = (z_mat.T @ y0) / r_counts
        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']
            h2_fixed[c], _ = lmm_module.estimate_h2_null(
                evc.T @ y0, evc.T @ x_scan, ev,
                method='reml' if use_reml else 'ml')

    n_boot   = bootstrap_outcomes.shape[1]

    # build the full boot matrix at strain level once — shape (n_scan, n_boot)
    Y_boot = bootstrap_outcomes.copy().astype(float)
    if use_rint:
        Y_boot = np.apply_along_axis(_rint, 0, Y_boot)
    if z_mat is not None:
        Y_boot = (z_mat.T @ Y_boot) / r_counts[:, None]

    # accumulate per-boot best (lod, pos) across chromosomes
    best_lods = np.full(n_boot, -np.inf)
    peak_pos  = np.full(n_boot, np.nan)

    for c in chrs:
        ev  = kinship_decomp[c]['values']
        evc = kinship_decomp[c]['vectors']

        if fix_h2:
            h2 = h2_fixed[c]
            chr_max, chr_pos = lmm_module.scan_chromosome_batch(
                Y_boot, genomecache, c, x_scan, ev, evc, h2,
                return_peak_positions=True,
            )
            better = chr_max > best_lods
            best_lods = np.where(better, chr_max, best_lods)
            peak_pos  = np.where(better, chr_pos, peak_pos)
        else:
            # fix_h2=False: re-estimate h² per replicate (slow path, rarely used)
            for b_idx in range(n_boot):
                y_b = Y_boot[:, b_idx]
                h2, _ = lmm_module.estimate_h2_null(
                    evc.T @ y_b, evc.T @ x_scan, ev,
                    method='reml' if use_reml else 'ml')
                cd_b   = lmm_module.precompute_chromosome(genomecache, c, ev, evc)
                res    = lmm_module.scan_chromosome(y_b, cd_b, x_scan, ev, evc, h2)
                peak_i = int(np.argmax(res['lod']))
                if res['lod'][peak_i] > best_lods[b_idx]:
                    best_lods[b_idx] = res['lod'][peak_i]
                    peak_pos[b_idx]  = res['positions'][peak_i]

    valid   = peak_pos[np.isfinite(peak_pos)]
    lo      = float(np.quantile(valid, ci_alpha / 2))
    hi      = float(np.quantile(valid, 1 - ci_alpha / 2))
    ci_info = {
        'ci_lo': lo, 'ci_hi': hi, 'ci': (lo, hi),
        'n_boot': n_boot, 'n_valid': len(valid),
        'alpha': ci_alpha,
    }
    return peak_pos, ci_info


def ci_scan_multi_gene(
    expr_cache: 'dict[str, np.ndarray]',
    gene_chr_map: 'dict[str, str]',
    peak_mb_map: 'dict[str, float]',
    genomecache,
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    sample_ids: 'list[str] | None' = None,
    n_boot: int = 200,
    seed: int = 1,
    use_rint: bool = True,
    output_path: 'Path | None' = None,
) -> pd.DataFrame:
    """bootstrap positional CI for a set of genes, streaming to disk.

    calls generate_null_outcomes + run_positional_scans per gene, streams each
    CI row immediately — no accumulation of bootstrap matrices across genes.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map   : dict {gene_id: str}  chromosome per gene
    peak_mb_map    : dict {gene_id: float}  peak position in Mb (from S4 scan)
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids     : list[str] or None  genotype-level strain IDs
    n_boot         : int  bootstrap replicates per gene (default 200)
    seed           : int  base seed; incremented per gene
    use_rint       : bool
    output_path    : Path or None  stream CSV here; None accumulates in RAM

    returns
    -------
    pd.DataFrame  columns: gene, chr, peak_mb, ci_lo, ci_hi, ci_width_mb
    """
    _first = True
    _buf: 'list[dict]' = []
    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, '')
        peak_mb = peak_mb_map.get(gene_id)
        if not gene_chr or peak_mb is None:
            logger.warning(
                'ci_scan_multi_gene: missing chr/peak_mb for %s — skipped', gene_id)
            continue
        _kd_g = ({gene_chr: kinship_decomp[gene_chr]}
                 if kinship_decomp is not None else None)
        try:
            boot_out = generate_null_outcomes(
                y, n_perms=n_boot, seed=seed + _g, method='bootstrap')
            ci_mbs, ci_info = run_positional_scans(
                boot_out, genomecache,
                phenotype=y,
                covariates=covariates,
                kinship_decomp=_kd_g,
                peak_chrom=gene_chr,
                use_rint=use_rint,
                sample_ids=sample_ids,
            )
            del boot_out, ci_mbs
            row = {
                'gene':        gene_id,
                'chr':         gene_chr,
                'peak_mb':     float(peak_mb),
                'ci_lo':       ci_info['ci_lo'],
                'ci_hi':       ci_info['ci_hi'],
                'ci_width_mb': ci_info['ci_hi'] - ci_info['ci_lo'],
            }
            if output_path is not None:
                pd.DataFrame([row]).to_csv(
                    output_path, mode='a', header=_first, index=False)
                _first = False
            else:
                _buf.append(row)
        except Exception as exc:
            logger.warning('ci_scan_multi_gene: %s failed — %s', gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (pd.read_csv(output_path)
                if output_path.exists() and not _first else pd.DataFrame())
    return pd.DataFrame(_buf) if _buf else pd.DataFrame()
