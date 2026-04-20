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
"""

import numpy as np
import logging
from scipy.stats import genextreme

logger = logging.getLogger('mosaiqtl')


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
# threshold scans
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_scans(null_outcomes, genomecache,
                        phenotype=None,
                        covariates=None, kinship_decomp=None,
                        chrom=None,
                        fix_h2=True,
                        use_rint=True, use_reml=True):
    """scan null phenotype columns to build the peak-LOD null distribution.


    parameters
    ----------
    null_outcomes  : np.ndarray (n, n_perms)  from generate_null_outcomes()
    genomecache   : GenomeCache
    phenotype      : np.ndarray (n,) or None
        original phenotype; required when fix_h2=True to pre-estimate h².
        ignored when fix_h2=False (h² re-estimated per permutation).
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
        if None, computed from genomecache (loco).
    chrom          : str or None
        if given, restrict scans to this chromosome only (validation mode).
        if None, scan all chromosomes (full-genome threshold).
    fix_h2         : bool  default True
        True  — estimate h² once from `phenotype` and reuse across perms
                 (fast; matches validate_port.py inline loop).
        False — re-estimate h² for each permuted phenotype (slow, more correct).
    use_rint       : bool  apply rank-inverse normal transform to phenotype
    use_reml       : bool  REML for h² estimation

    returns
    -------
    np.ndarray (n_perms,)  per-permutation genome-wide max LOD
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

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

    # pre-compute per-chrom h² from original phenotype (fix_h2=True mode)
    h2_fixed = {}
    if fix_h2:
        if phenotype is None:
            raise ValueError('phenotype required when fix_h2=True')
        y0 = _rint(phenotype) if use_rint else phenotype.copy()
        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']
            h2_fixed[c], _ = lmm_module.estimate_h2_null(
                evc.T @ y0, evc.T @ x, ev,
                method='reml' if use_reml else 'ml')

    n_perms  = null_outcomes.shape[1]
    max_lods = np.zeros(n_perms)

    for p_idx in range(n_perms):
        if (p_idx + 1) % 100 == 0:
            logger.info(f'  threshold scan {p_idx+1}/{n_perms}')

        y_p = null_outcomes[:, p_idx]                   # permuted phenotype
        if use_rint:
            y_p = _rint(y_p)

        best = -np.inf
        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']

            if fix_h2:
                h2 = h2_fixed[c]                        # reuse pre-computed h²
            else:
                h2, _ = lmm_module.estimate_h2_null(
                    evc.T @ y_p, evc.T @ x, ev,
                    method='reml' if use_reml else 'ml')

            # scan_chromosome handles y/x rotation internally
            res  = lmm_module.scan_chromosome(
                y_p, genomecache, c, x, ev, evc, h2, use_reml=use_reml)
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
                         ci_alpha=0.05):
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

    # pre-compute h² from original phenotype
    h2_fixed = {}
    if fix_h2:
        y0 = _rint(phenotype) if use_rint else phenotype.copy()
        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']
            h2_fixed[c], _ = lmm_module.estimate_h2_null(
                evc.T @ y0, evc.T @ x, ev,
                method='reml' if use_reml else 'ml')

    n_boot   = bootstrap_outcomes.shape[1]
    peak_pos = np.full(n_boot, np.nan)

    for b_idx in range(n_boot):
        if (b_idx + 1) % 50 == 0:
            logger.info(f'  bootstrap scan {b_idx+1}/{n_boot}')

        y_b = bootstrap_outcomes[:, b_idx]
        if use_rint:
            y_b = _rint(y_b)

        best_lod = -np.inf
        best_pos = np.nan

        for c in chrs:
            ev  = kinship_decomp[c]['values']
            evc = kinship_decomp[c]['vectors']

            if fix_h2:
                h2 = h2_fixed[c]
            else:
                h2, _ = lmm_module.estimate_h2_null(
                    evc.T @ y_b, evc.T @ x, ev,
                    method='reml' if use_reml else 'ml')

            res    = lmm_module.scan_chromosome(
                y_b, genomecache, c, x, ev, evc, h2, use_reml=use_reml)
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
