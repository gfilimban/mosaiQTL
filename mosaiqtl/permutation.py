"""
mosaiqtl.permutation — permutation thresholds and parametric bootstrap CI

implements significance thresholds for genome-wide scans via permutation
and positional confidence intervals via parametric bootstrap.

equivalent to miqtl:
  generate.sample.outcomes.matrix()  →  generate_null_outcomes()
  run.threshold.scans()              →  run_threshold_scans()
  get.gev.thresholds()               →  get_gev_thresholds()
  run.positional.scans()             →  run_positional_scans()
"""

import numpy as np
import logging
from scipy.stats import genextreme

logger = logging.getLogger('mosaiqtl')


def generate_null_outcomes(scan_result, method='permutation',
                           n_samples=1000, seed=1,
                           use_blup=False):
    """generate null phenotype samples for threshold estimation.

    for permutation: shuffles phenotype labels.
    for bootstrap: samples from the null model (no qtl effect) using
    the fitted null distribution from the scan.

    equivalent to miqtl::generate.sample.outcomes.matrix().

    parameters
    ----------
    scan_result : ScanResult
        completed scan (used to extract phenotype from h2_null context)
    method : str  'permutation' (default) or 'bootstrap'
    n_samples : int  number of samples to generate
    seed : int
    use_blup : bool  use blup residuals for bootstrap (ignored for permutation)

    returns
    -------
    np.ndarray (n_subjects, n_samples)  null phenotype matrix
    """
    rng   = np.random.default_rng(seed)
    y     = scan_result._phenotype    # stored by run_threshold_scans caller
    n     = len(y)

    if method == 'permutation':
        # permute sample labels — breaks genotype-phenotype association
        outcomes = np.column_stack([rng.permutation(y) for _ in range(n_samples)])
    elif method == 'bootstrap':
        # resample with replacement from centered phenotype
        y_c = y - y.mean()
        outcomes = np.column_stack([
            rng.choice(y_c, size=n, replace=True) for _ in range(n_samples)])
    else:
        raise ValueError(f'method must be permutation or bootstrap, got {method!r}')

    return outcomes


def run_threshold_scans(null_outcomes, phenotype, genome_cache,
                        covariates=None, kinship_decomp=None,
                        use_rint=True, use_reml=True,
                        scan_seed=1, n_cores=1):
    """scan all null phenotype columns to build the peak lod distribution.

    equivalent to miqtl::run.threshold.scans().

    parameters
    ----------
    null_outcomes : np.ndarray (n_subjects, n_perms)
        null phenotypes from generate_null_outcomes()
    phenotype : np.ndarray (n_subjects,)
        original phenotype (used for alignment only)
    genome_cache : GenomeCache
    covariates : np.ndarray or None
    kinship_decomp : dict or None
    use_rint : bool
    use_reml : bool
    scan_seed : int
    n_cores : int

    returns
    -------
    np.ndarray (n_perms,)  per-permutation genome-wide max lod
    """
    from . import scan as scan_module
    from . import kinship as kin_module

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    n_perms = null_outcomes.shape[1]
    max_lods = np.zeros(n_perms)

    for p_idx in range(n_perms):
        if (p_idx + 1) % 100 == 0:
            logger.info(f'  threshold scan {p_idx+1}/{n_perms}')
        y_perm = null_outcomes[:, p_idx]
        res = scan_module.scan_h2lmm(
            y_perm, genome_cache, covariates,
            kinship_decomp, use_rint, use_reml,
            phenotype_id=f'perm_{p_idx}')
        max_lods[p_idx] = float(res.lod.max())

    return max_lods


def get_gev_thresholds(max_lods, percentile=0.95):
    """fit a generalised extreme value (GEV) distribution to the
    permutation peak-lod distribution and return the threshold.

    fitting GEV is more stable than empirical quantiles when n_perms
    is small (e.g. 100–500), matching miQTL behaviour.

    equivalent to miqtl::get.gev.thresholds().

    parameters
    ----------
    max_lods : np.ndarray (n_perms,)
        per-permutation genome-wide maximum lod scores
    percentile : float  significance percentile (default 0.95 = 5% genome-wide)

    returns
    -------
    float  lod threshold at the requested percentile
    dict   {'c': shape, 'loc': location, 'scale': scale, 'threshold': float}
    """
    # fit gev to the max-lod distribution
    # genextreme parameterisation: shape c, loc, scale
    c, loc, scale = genextreme.fit(max_lods)

    # threshold = inverse cdf at percentile
    threshold = float(genextreme.ppf(percentile, c, loc=loc, scale=scale))

    params = {'c': float(c), 'loc': float(loc), 'scale': float(scale),
              'threshold': threshold, 'percentile': percentile,
              'n_perms': len(max_lods),
              'empirical_quantile': float(np.quantile(max_lods, percentile))}
    return threshold, params


def run_positional_scans(bootstrap_outcomes, phenotype, genome_cache,
                         covariates=None, kinship_decomp=None,
                         peak_chrom=None, peak_locus=None,
                         use_rint=True, use_reml=True,
                         scan_seed=1):
    """run single-chromosome scans on parametric bootstrap samples.

    used to compute the positional CI for a QTL.  for each bootstrap
    outcome, scans the chromosome containing the QTL and records the
    position of the peak locus.

    equivalent to miqtl::run.positional.scans().

    parameters
    ----------
    bootstrap_outcomes : np.ndarray (n_subjects, n_boot)
        bootstrap phenotypes from generate_null_outcomes(method='bootstrap')
    phenotype : np.ndarray (n_subjects,)  original phenotype
    genome_cache : GenomeCache
    covariates : np.ndarray or None
    kinship_decomp : dict or None
    peak_chrom : str  chromosome to scan (if None, scan all)
    peak_locus : str  known peak locus id (used for logging only)
    use_rint : bool
    use_reml : bool
    scan_seed : int

    returns
    -------
    np.ndarray (n_boot,)  bootstrap peak positions in Mb
    dict  {'ci_lo': float, 'ci_hi': float, 'ci_95': (lo, hi)}
    """
    from . import scan as scan_module
    from . import kinship as kin_module

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    n_boot = bootstrap_outcomes.shape[1]
    peak_pos = np.zeros(n_boot)

    chrs_to_scan = ([peak_chrom] if peak_chrom
                    else list(genome_cache.chromosomes))

    for b_idx in range(n_boot):
        if (b_idx + 1) % 50 == 0:
            logger.info(f'  bootstrap scan {b_idx+1}/{n_boot}')
        y_b = bootstrap_outcomes[:, b_idx]
        best_lod = -np.inf; best_pos = np.nan

        for chrom in chrs_to_scan:
            if chrom not in kinship_decomp:
                continue
            eigvals = kinship_decomp[chrom]['values']
            eigvecs = kinship_decomp[chrom]['vectors']
            from . import lmm as lmm_module
            valid   = ~np.isnan(y_b)
            y_v     = y_b[valid]
            if use_rint:
                from .utils import rint as _rint
                y_v = _rint(y_v)
            cov_v   = (covariates[valid] if covariates is not None
                       else np.ones((valid.sum(), 1)))
            if not np.allclose(cov_v[:, 0], 1.0):
                cov_v = np.hstack([np.ones((len(y_v), 1)), cov_v])
            ev_v    = eigvecs[valid]
            y_rot   = ev_v.T @ y_v
            x_rot   = ev_v.T @ cov_v
            h2, _   = lmm_module.estimate_h2_null(y_rot, x_rot, eigvals)
            chr_res = lmm_module.scan_chromosome(
                y_v, genome_cache, chrom, cov_v, eigvals, ev_v, h2, use_reml)
            peak_i  = int(np.argmax(chr_res['lod']))
            if chr_res['lod'][peak_i] > best_lod:
                best_lod = chr_res['lod'][peak_i]
                best_pos = chr_res['positions'][peak_i]

        peak_pos[b_idx] = best_pos

    valid_pos = peak_pos[~np.isnan(peak_pos)]
    ci_lo = float(np.quantile(valid_pos, 0.025))
    ci_hi = float(np.quantile(valid_pos, 0.975))
    ci_info = {'ci_lo': ci_lo, 'ci_hi': ci_hi, 'ci_95': (ci_lo, ci_hi),
               'n_boot': n_boot, 'n_valid': len(valid_pos)}
    return peak_pos, ci_info
