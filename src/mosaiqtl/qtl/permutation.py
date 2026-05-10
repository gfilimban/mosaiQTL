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
  - permutation_method='plain' (default) shuffles y labels — matches Phase 1
    identity-kinship (τ²≈0) validation.
    permutation_method='parametric' mirrors miQTL sampling.processes.R lines
    110-137: draws u~N(0,K·σ²_g), e~N(0,σ²_e·I), rank-permutes y using the
    parametric sample's order. Required for Phase 2 real LOCO kinship.
"""

import numpy as np
import logging
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
    """raised when a LOCO kinship matrix is not PSD (min eigenvalue < 1e-8)."""


# ─────────────────────────────────────────────────────────────────────────────
# null outcome generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_null_outcomes(y, n_samples=1000, method='permutation', seed=1):
    """generate null phenotype columns for threshold estimation.

    for permutation: shuffles phenotype labels (breaks geno-pheno link).
    for bootstrap: resamples with replacement from centred phenotype.


    parameters
    ----------
    y         : np.ndarray (n,)  original phenotype (NaN-free)
    n_samples : int              number of null columns to draw
    method    : str              'permutation' (default) or 'bootstrap'
    seed      : int

    returns
    -------
    np.ndarray (n, n_samples)  null phenotype matrix
    """
    rng = np.random.default_rng(seed)   # reproducible rng

    if method == 'permutation':
        # shuffle rows independently for each column
        return np.column_stack([rng.permutation(y) for _ in range(n_samples)])
    elif method == 'bootstrap':
        y_c = y - y.mean()              # centre before resampling
        return np.column_stack([
            rng.choice(y_c, size=len(y), replace=True)
            for _ in range(n_samples)])
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
    delta = h2 / (1.0 - h2)
    w_inv = 1.0 / (delta * eigvals + 1.0)
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

    raises KinshipSingularError when min eigenvalue of K < 1e-8,
    mirroring the behaviour of R's mnormt::rmnorm (chol fails on near-singular K).
    """
    min_eig = float(np.linalg.eigvalsh(K).min())
    if min_eig < 1e-8:
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

    min_eig = float(np.linalg.eigvalsh(K).min())
    if min_eig < 1e-8:
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
                        fix_h2=True,
                        permutation_method='parametric',
                        use_rint=True, use_reml=True,
                        sample_ids=None,
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
        raw LOCO kinship matrices (not decomposed).  required when
        permutation_method='parametric'.  ignored for 'plain'.
    chrom          : str or None
        if given, restrict scans to this chromosome only (validation mode).
        if None, scan all chromosomes (full-genome threshold).
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
    seed           : int  rng seed (used only for 'parametric' mode)
    batch_size     : int  mini-batch size for the fast vectorised path
        (fix_h2=True + plain).  500 balances memory and loop overhead;
        reduce if RAM is tight.

    returns
    -------
    np.ndarray (n_perms,)  per-permutation genome-wide max LOD

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
    if permutation_method == 'parametric' and kinship_loco is None:
        raise ValueError("kinship_loco is required when permutation_method='parametric'")

    # build kinship decomp if not supplied
    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

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

    # replicate path: reduce to strain means if sample_ids provided
    z_mat    = None
    r_counts = None
    x_scan   = x
    if sample_ids is not None:
        from ..utils import build_z_matrix
        z_mat, _  = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        r_counts  = z_mat.sum(axis=0)
        x_scan    = (z_mat.T @ x) / r_counts[:, None]

    # pre-compute per-chrom h² from original phenotype (fix_h2=True mode)
    h2_fixed = {}
    if fix_h2:
        if phenotype is None:
            raise ValueError('phenotype required when fix_h2=True')
        y0 = _rint(phenotype) if use_rint else phenotype.copy()
        if z_mat is not None:
            y0 = (z_mat.T @ y0) / r_counts
        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']
            h2_fixed[c], _ = lmm_module.estimate_h2_null(
                evc.T @ y0, evc.T @ x_scan, ev,
                method='reml' if use_reml else 'ml')
    else:
        if phenotype is not None:
            y0 = _rint(phenotype) if use_rint else phenotype.copy()
            if z_mat is not None:
                y0 = (z_mat.T @ y0) / r_counts
        else:
            y0 = None

    n_perms  = null_outcomes.shape[1]
    max_lods = np.zeros(n_perms)
    rng      = np.random.default_rng(seed) if permutation_method == 'parametric' else None

    # ── fast vectorised path: fix_h2=True ────────────────────────────────────
    # precomputes H_perp / G_pinv once per chromosome; only e0 varies per perm.
    # plain:      builds one Y_mat for all chromosomes (no kinship draw per perm)
    # parametric: rebuilds Y_mat per chromosome using that chr's K_loco
    # ~100–500× faster than the slow path on large n_loci.
    if fix_h2 and permutation_method in ('plain', 'parametric'):
        plain_mat = None
        if permutation_method == 'plain':
            # same permuted matrix reused across all chromosomes
            if use_rint:
                plain_mat = np.column_stack([_rint(null_outcomes[:, i])
                                             for i in range(n_perms)])
            else:
                plain_mat = null_outcomes.astype(float)
            if z_mat is not None:
                plain_mat = (z_mat.T @ plain_mat) / r_counts[:, None]

        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']
            h2  = h2_fixed[c]

            if permutation_method == 'parametric':
                # chromosome-specific null draw using K_loco[c]
                try:
                    null_mat_c = _draw_parametric_batch(
                        y0, x_scan, kinship_loco[c], evc, ev,
                        h2, use_reml, n_perms, rng)
                except KinshipSingularError as exc:
                    logger.warning('  chr%s  [skip] %s', c, exc)
                    continue
            else:
                null_mat_c = plain_mat

            chr_max = lmm_module.scan_chromosome_batch(
                null_mat_c, genomecache, c, x_scan, ev, evc, h2,
                batch_size=batch_size)
            max_lods = np.maximum(max_lods, chr_max)
            logger.info('  [batch] chr%s  max_lod=%.3f', c, float(chr_max.max()))

        return max_lods

    # ── slow path: fix_h2=False ───────────────────────────────────────────────
    for p_idx in range(n_perms):
        if (p_idx + 1) % 100 == 0:
            logger.info(f'  threshold scan {p_idx+1}/{n_perms}')

        best = -np.inf

        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']

            if permutation_method == 'parametric':
                # kinship-informed rank-permutation (miQTL sampling.processes.R:110-137)
                h2 = h2_fixed[c] if fix_h2 else None
                if not fix_h2:
                    h2, _ = lmm_module.estimate_h2_null(
                        evc.T @ y0, evc.T @ x_scan, ev,
                        method='reml' if use_reml else 'ml')
                try:
                    y_p = _parametric_null_col(
                        y0, x_scan, kinship_loco[c], evc, ev, h2, use_reml, rng)
                except KinshipSingularError as exc:
                    logger.warning(f'  chr{c}  perm {p_idx}  [skip] {exc}')
                    continue
            else:
                # plain shuffle
                y_p = null_outcomes[:, p_idx]
                if use_rint:
                    y_p = _rint(y_p)
                if z_mat is not None:
                    y_p = (z_mat.T @ y_p) / r_counts
                if fix_h2:
                    h2 = h2_fixed[c]
                else:
                    h2, _ = lmm_module.estimate_h2_null(
                        evc.T @ y_p, evc.T @ x_scan, ev,
                        method='reml' if use_reml else 'ml')

            res  = lmm_module.scan_chromosome(
                y_p, genomecache, c, x_scan, ev, evc, h2, use_reml=use_reml)
            best = max(best, float(res['lod'].max()))

        max_lods[p_idx] = best

    return max_lods


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
    peak_pos = np.full(n_boot, np.nan)

    for b_idx in range(n_boot):
        if (b_idx + 1) % 50 == 0:
            logger.info(f'  bootstrap scan {b_idx+1}/{n_boot}')

        y_b = bootstrap_outcomes[:, b_idx]
        if use_rint:
            y_b = _rint(y_b)
        if z_mat is not None:
            y_b = (z_mat.T @ y_b) / r_counts

        best_lod = -np.inf
        best_pos = np.nan

        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']

            if fix_h2:
                h2 = h2_fixed[c]
            else:
                h2, _ = lmm_module.estimate_h2_null(
                    evc.T @ y_b, evc.T @ x_scan, ev,
                    method='reml' if use_reml else 'ml')

            res    = lmm_module.scan_chromosome(
                y_b, genomecache, c, x_scan, ev, evc, h2, use_reml=use_reml)
            peak_i = int(np.argmax(res['lod']))
            if res['lod'][peak_i] > best_lod:
                best_lod = res['lod'][peak_i]
                best_pos = res['positions'][peak_i]

        peak_pos[b_idx] = best_pos

    valid   = peak_pos[np.isfinite(peak_pos)]
    lo      = float(np.quantile(valid, ci_alpha / 2))
    hi      = float(np.quantile(valid, 1 - ci_alpha / 2))
    ci_info = {
        'ci_lo': lo, 'ci_hi': hi, 'ci': (lo, hi),
        'n_boot': n_boot, 'n_valid': len(valid),
        'alpha': ci_alpha,
    }
    return peak_pos, ci_info
