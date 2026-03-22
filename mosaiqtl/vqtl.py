"""
mosaiqtl.vqtl — variance/dispersion qtl mapping

maps genetic effects on expression noise (within-subject
overdispersion) rather than mean expression.

two approaches:
  1. two-stage: estimate per-subject φ̂ via moments or nebula-ln,
     then scan log(φ̂) as phenotype through the lmm haplotype scan.
  2. joint (future): extend nebula to model log(φ_i) = h_i' δ,
     testing δ = 0 via likelihood ratio.

the two-stage approach is computationally cheap (reuses the lmm
scan infrastructure) and provides a good first pass.

biological interpretation:
  a significant dispersion vqtl at a locus means that founder
  haplotype controls how *noisy* a gene's expression is — not
  just how high or low. this could reflect genetic control of:
    - transcriptional bursting kinetics
    - sub-type heterogeneity within cell type
    - stochastic regulatory network stability
"""

import numpy as np
import pandas as pd
import logging
from . import nb as nb_module
from . import scan as scan_module
from .utils import rint

logger = logging.getLogger('mosaiqtl')


class VQTLResult:
    """container for vqtl scan results.

    attributes
    ----------
    scan_result : ScanResult
        genome-wide scan of dispersion phenotype
    phi_estimates : np.ndarray (n_subjects,)
        per-subject overdispersion estimates used as phenotype
    phi_method : str
        estimation method ('moments' or 'nb')
    gene_id : str
        gene name
    mean_scan_result : ScanResult or None
        corresponding mean eqtl scan (for comparison)
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def summary(self):
        """print summary of vqtl results."""
        peak = self.scan_result.peak_locus()
        print(f'vqtl scan for {self.gene_id}:')
        print(f'  method: {self.phi_method}')
        print(f'  n subjects with valid φ: {np.sum(~np.isnan(self.phi_estimates))}')
        print(f'  median φ: {np.nanmedian(self.phi_estimates):.4f}')
        print(f'  peak locus: {peak["locus_id"]} (chr{peak["chromosome"]})')
        print(f'  peak lod: {peak["lod"]:.2f}')
        print(f'  peak p-value: {peak["p_value"]:.2e}')


def estimate_dispersion_per_subject(counts, subject_ids, library_sizes=None,
                                     method='moments', min_cells=20):
    """estimate per-subject cell-level overdispersion.

    for each subject, estimates φ from the distribution of
    single-cell counts for one gene.

    parameters
    ----------
    counts : np.ndarray (n_cells,)
        raw umi counts for one gene
    subject_ids : np.ndarray (n_cells,)
        subject identifier per cell
    library_sizes : np.ndarray (n_cells,) or None
        total umi per cell
    method : str
        'moments' — method-of-moments estimator (fast)
        'mle'     — maximum likelihood nb fit per subject (slower)
    min_cells : int
        minimum cells per subject to compute φ (subjects with
        fewer cells get nan)

    returns
    -------
    subjects : np.ndarray of str
        unique subject identifiers
    phi_hat : np.ndarray (n_subjects,)
        per-subject overdispersion estimates (nan if insufficient cells)
    """
    # precompute sufficient statistics
    suff = nb_module.precompute_sufficient_stats(
        counts, subject_ids, library_sizes
    )

    subjects = suff['subjects']
    n_subjects = len(subjects)
    phi_hat = np.full(n_subjects, np.nan)

    if method == 'moments':
        # φ = (var - mean) / mean²
        for i in range(n_subjects):
            if suff['n_cells'][i] < min_cells:
                continue
            mu = suff['mean_count'][i]
            var = suff['var_count'][i]
            if mu > 0.01:
                phi_hat[i] = max((var - mu) / (mu ** 2), 1e-6)

    elif method == 'mle':
        # fit nb distribution per subject via ml
        from scipy.optimize import minimize_scalar

        # rebuild cell-level data per subject
        subject_map = {}
        for idx, sid in enumerate(subject_ids):
            if sid not in subject_map:
                subject_map[sid] = []
            subject_map[sid].append(idx)

        for i, sid in enumerate(subjects):
            cell_idx = np.array(subject_map[sid])
            if len(cell_idx) < min_cells:
                continue

            y_s = counts[cell_idx].astype(np.float64)
            mu_s = np.mean(y_s)

            if mu_s < 0.01:
                continue

            # nb log-likelihood as function of log(φ)
            def neg_ll(log_phi):
                phi = np.exp(log_phi)
                r = 1.0 / phi  # nb size parameter
                from scipy.special import gammaln
                # nb log-pmf summed over cells
                ll = np.sum(
                    gammaln(y_s + r) - gammaln(r) - gammaln(y_s + 1) +
                    r * np.log(r / (r + mu_s)) +
                    y_s * np.log(mu_s / (r + mu_s) + 1e-300)
                )
                return -ll

            # optimize
            result = minimize_scalar(
                neg_ll,
                bounds=(np.log(1e-4), np.log(100)),
                method='bounded'
            )
            phi_hat[i] = np.exp(result.x)

    else:
        raise ValueError(f'unknown dispersion method: {method}')

    return subjects, phi_hat


def scan_vqtl(counts, subject_ids, genome_cache,
              library_sizes=None, covariates=None,
              kinship_decomp=None, phi_method='moments',
              min_cells=20, use_rint=True, gene_id='unknown',
              also_scan_mean=True):
    """run dispersion vqtl scan for one gene.

    two-stage approach:
      stage 1: estimate per-subject φ̂ from cell-level counts
      stage 2: scan log(φ̂) as phenotype through lmm haplotype scan

    parameters
    ----------
    counts : np.ndarray (n_cells,)
        raw umi counts for one gene
    subject_ids : np.ndarray (n_cells,)
        subject identifier per cell
    genome_cache : GenomeCache
        haplotype probability data
    library_sizes : np.ndarray (n_cells,) or None
    covariates : np.ndarray (n_subjects, p) or None
    kinship_decomp : dict or None
    phi_method : str
        'moments' or 'mle'
    min_cells : int
        minimum cells per subject for φ estimation
    use_rint : bool
        apply rint to log(φ̂) before scanning
    gene_id : str
        gene name
    also_scan_mean : bool
        also run standard mean eqtl scan for comparison

    returns
    -------
    VQTLResult
    """
    # stage 1: estimate per-subject dispersion
    logger.info(f'{gene_id}: estimating per-subject dispersion ({phi_method})')
    subjects, phi_hat = estimate_dispersion_per_subject(
        counts, subject_ids, library_sizes,
        method=phi_method, min_cells=min_cells
    )

    # align subjects with genome cache
    from .utils import align_samples
    idx_phi, idx_gc, shared = align_samples(
        subjects, genome_cache.sample_ids
    )

    if len(shared) < 20:
        logger.warning(
            f'{gene_id}: only {len(shared)} shared subjects, skipping'
        )
        return None

    # extract aligned dispersion values
    phi_aligned = phi_hat[idx_phi]

    # log-transform (φ is positive, log makes it more normal)
    # set nan where φ estimation failed
    log_phi = np.full(len(phi_aligned), np.nan)
    valid_phi = phi_aligned > 0
    log_phi[valid_phi] = np.log(phi_aligned[valid_phi])

    n_valid = np.sum(~np.isnan(log_phi))
    logger.info(
        f'{gene_id}: {n_valid}/{len(shared)} subjects with valid φ, '
        f'median φ = {np.nanmedian(phi_aligned[valid_phi]):.4f}'
    )

    # align covariates if provided
    cov_aligned = None
    if covariates is not None:
        cov_aligned = covariates[idx_gc]

    # stage 2: scan log(φ̂) as phenotype
    logger.info(f'{gene_id}: scanning dispersion phenotype')
    vqtl_scan = scan_module.scan_h2lmm(
        log_phi,
        genome_cache,
        covariates=cov_aligned,
        kinship_decomp=kinship_decomp,
        use_rint=use_rint,
        phenotype_id=f'{gene_id}_vqtl',
    )

    # optionally scan mean expression for comparison
    mean_scan = None
    if also_scan_mean:
        # compute pseudobulk mean as phenotype
        suff = nb_module.precompute_sufficient_stats(
            counts, subject_ids, library_sizes
        )
        mean_expr = suff['mean_count'][idx_phi]

        mean_scan = scan_module.scan_h2lmm(
            mean_expr,
            genome_cache,
            covariates=cov_aligned,
            kinship_decomp=kinship_decomp,
            use_rint=use_rint,
            phenotype_id=f'{gene_id}_mean',
        )

    return VQTLResult(
        scan_result=vqtl_scan,
        phi_estimates=phi_aligned,
        phi_method=phi_method,
        gene_id=gene_id,
        mean_scan_result=mean_scan,
    )


def compare_mean_vs_dispersion(vqtl_result, lod_threshold=3.0):
    """classify genes by their mean and dispersion qtl patterns.

    parameters
    ----------
    vqtl_result : VQTLResult
        result from scan_vqtl with also_scan_mean=True
    lod_threshold : float
        lod score threshold for significance

    returns
    -------
    dict with keys:
        'category' : str
            'both'            — significant mean + dispersion qtl
            'mean_only'       — significant mean qtl only
            'dispersion_only' — significant dispersion qtl only (most novel)
            'neither'         — no significant qtl
        'mean_peak_lod' : float
        'vqtl_peak_lod' : float
        'colocalized' : bool
            whether peak loci are on same chromosome within 5 mb
    """
    vqtl_peak = vqtl_result.scan_result.peak_locus()
    mean_peak = vqtl_result.mean_scan_result.peak_locus()

    has_vqtl = vqtl_peak['lod'] >= lod_threshold
    has_mean = mean_peak['lod'] >= lod_threshold

    if has_vqtl and has_mean:
        category = 'both'
    elif has_mean:
        category = 'mean_only'
    elif has_vqtl:
        category = 'dispersion_only'
    else:
        category = 'neither'

    # check colocalization (same chr, within 5 mb)
    colocalized = (
        vqtl_peak['chromosome'] == mean_peak['chromosome'] and
        abs(vqtl_peak['position'] - mean_peak['position']) < 5e6
    )

    return {
        'category': category,
        'mean_peak_lod': mean_peak['lod'],
        'vqtl_peak_lod': vqtl_peak['lod'],
        'colocalized': colocalized,
        'mean_peak_locus': mean_peak['locus_id'],
        'vqtl_peak_locus': vqtl_peak['locus_id'],
    }
