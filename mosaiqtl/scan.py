"""
mosaiqtl.scan — genome-wide haplotype qtl scan

orchestrates the scan across all chromosomes for a single
phenotype (gene), combining genome cache, kinship, and the
lmm or nebula-ln engine.

equivalent to miqtl::scan.h2lmm() with loco kinship.
"""

import numpy as np
import pandas as pd
import logging
from . import kinship as kin_module
from . import lmm as lmm_module
from .utils import rint, drop_colinear_founders

logger = logging.getLogger('mosaiqtl')


class ScanResult:
    """container for genome-wide scan results.

    attributes
    ----------
    locus_ids : np.ndarray of str
        marker identifiers across all chromosomes
    chromosomes : np.ndarray of str
        chromosome for each locus
    positions : np.ndarray of float
        physical position (bp) for each locus
    lod : np.ndarray of float
        lod scores
    p_value : np.ndarray of float
        p-values
    betas : np.ndarray (n_loci, n_founders-1)
        founder haplotype effects
    h2_null : dict
        {chrom: h2_estimate} under null model per loco chromosome
    phenotype_id : str
        gene/phenotype name
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def peak_locus(self):
        """return the locus with highest lod score."""
        idx = np.argmax(self.lod)
        return {
            'locus_id': self.locus_ids[idx],
            'chromosome': self.chromosomes[idx],
            'position': self.positions[idx],
            'lod': self.lod[idx],
            'p_value': self.p_value[idx],
            'betas': self.betas[idx],
        }

    def to_dataframe(self):
        """convert scan results to pandas dataframe."""
        df = pd.DataFrame({
            'locus_id': self.locus_ids,
            'chromosome': self.chromosomes,
            'position': self.positions,
            'lod': self.lod,
            'p_value': self.p_value,
        })
        # add founder effect columns
        n_founders = self.betas.shape[1]
        for f_idx in range(n_founders):
            df[f'beta_f{f_idx}'] = self.betas[:, f_idx]
        return df


def scan_h2lmm(phenotype, genome_cache, covariates=None,
               kinship_decomp=None, use_rint=True,
               use_reml=True, phenotype_id='unknown'):
    """run haplotype-based genome scan with lmm and loco kinship.

    this is the primary scan function, equivalent to
    miqtl::scan.h2lmm().

    parameters
    ----------
    phenotype : np.ndarray (n_subjects,)
        phenotype values (one gene). nan values are excluded.
    genome_cache : GenomeCache
        haplotype probability data
    covariates : np.ndarray (n_subjects, p) or None
        covariate design matrix. if none, intercept-only model.
    kinship_decomp : dict or None
        {chrom: {'values': eigvals, 'vectors': eigvecs}}
        precomputed loco kinship eigendecompositions.
        if none, computed internally.
    use_rint : bool
        apply rank-inverse normal transformation to phenotype
    use_reml : bool
        use reml (True) or ml (False) for variance estimation
    phenotype_id : str
        name of the phenotype (for logging)

    returns
    -------
    ScanResult
    """
    n = len(phenotype)

    # handle missing values
    valid = ~np.isnan(phenotype)
    if valid.sum() < n:
        logger.info(
            f'{phenotype_id}: excluding {n - valid.sum()} missing values'
        )
    y = phenotype[valid].copy()
    n_valid = len(y)

    # apply rint if requested
    if use_rint:
        y = rint(y)

    # build covariate matrix (with intercept)
    if covariates is not None:
        x = covariates[valid].copy()
        # add intercept if not present
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((n_valid, 1)), x])
    else:
        x = np.ones((n_valid, 1))

    # compute kinship if not provided
    if kinship_decomp is None:
        logger.info(f'{phenotype_id}: computing loco kinship')
        kinship_dict = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kinship_dict)

    # scan each chromosome
    all_locus_ids = []
    all_chroms = []
    all_positions = []
    all_lod = []
    all_pval = []
    all_betas = []
    h2_null_dict = {}

    for chrom in genome_cache.chromosomes:
        if chrom not in kinship_decomp:
            logger.warning(f'no kinship for chr{chrom}, skipping')
            continue

        eigvals = kinship_decomp[chrom]['values']
        eigvecs = kinship_decomp[chrom]['vectors']

        # subset eigvals/eigvecs for valid samples only
        eigvals_v = eigvals  # already aligned
        eigvecs_v = eigvecs[valid][:, :] if valid.sum() < n else eigvecs

        # estimate null h² for this chromosome
        y_rot = eigvecs_v.T @ y
        x_rot = eigvecs_v.T @ x

        h2_null, ll_null = lmm_module.estimate_h2_null(
            y_rot, x_rot, eigvals_v, method='reml' if use_reml else 'ml'
        )
        h2_null_dict[chrom] = h2_null

        logger.debug(
            f'{phenotype_id} chr{chrom}: h2_null = {h2_null:.4f}'
        )

        # scan all loci on this chromosome
        chr_result = lmm_module.scan_chromosome(
            y, genome_cache, chrom, x,
            eigvals_v, eigvecs_v, h2_null, use_reml
        )

        n_loci = len(chr_result['locus_ids'])
        all_locus_ids.append(chr_result['locus_ids'])
        all_chroms.append(np.full(n_loci, chrom))
        all_positions.append(chr_result['positions'])
        all_lod.append(chr_result['lod'])
        all_pval.append(chr_result['p_value'])
        all_betas.append(chr_result['betas'])

    # concatenate across chromosomes
    return ScanResult(
        locus_ids=np.concatenate(all_locus_ids),
        chromosomes=np.concatenate(all_chroms),
        positions=np.concatenate(all_positions),
        lod=np.concatenate(all_lod),
        p_value=np.concatenate(all_pval),
        betas=np.vstack(all_betas),
        h2_null=h2_null_dict,
        phenotype_id=phenotype_id,
    )


def scan_multi_gene(phenotype_matrix, gene_ids, genome_cache,
                    covariates=None, kinship_decomp=None,
                    use_rint=True, n_cores=1):
    """scan multiple genes (columns of phenotype_matrix).

    parameters
    ----------
    phenotype_matrix : np.ndarray (n_subjects, n_genes)
        phenotype matrix (columns are genes)
    gene_ids : list of str
        gene names
    genome_cache : GenomeCache
        haplotype probability data
    covariates : np.ndarray (n_subjects, p) or None
    kinship_decomp : dict or None
    use_rint : bool
    n_cores : int
        number of parallel processes (1 = serial)

    returns
    -------
    dict of {gene_id: ScanResult}
    """
    # precompute kinship once for all genes
    if kinship_decomp is None:
        logger.info('precomputing loco kinship for multi-gene scan')
        kinship_dict = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kinship_dict)

    results = {}

    if n_cores == 1:
        # serial execution
        for g_idx, gene_id in enumerate(gene_ids):
            logger.info(
                f'scanning gene {g_idx + 1}/{len(gene_ids)}: {gene_id}'
            )
            results[gene_id] = scan_h2lmm(
                phenotype_matrix[:, g_idx],
                genome_cache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                use_rint=use_rint,
                phenotype_id=gene_id,
            )
    else:
        # parallel execution via multiprocessing
        from multiprocessing import Pool
        from functools import partial

        def _scan_one_gene(args):
            g_idx, gene_id = args
            return gene_id, scan_h2lmm(
                phenotype_matrix[:, g_idx],
                genome_cache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                use_rint=use_rint,
                phenotype_id=gene_id,
            )

        with Pool(n_cores) as pool:
            for gene_id, scan_result in pool.imap_unordered(
                _scan_one_gene, enumerate(gene_ids)
            ):
                results[gene_id] = scan_result
                logger.info(
                    f'completed: {gene_id} '
                    f'(peak lod = {scan_result.peak_locus()["lod"]:.2f})'
                )

    return results


# ── Phase 2 additions ─────────────────────────────────────────────────────────

def grab_locus_from_scan(scan_result):
    """return the locus id with the highest lod score.

    equivalent to miqtl::grab.locus.from.scan().

    parameters
    ----------
    scan_result : ScanResult

    returns
    -------
    str  locus id at peak
    """
    return scan_result.locus_ids[np.argmax(scan_result.lod)]


def lod_drop_ci(scan_result, chrom, drop=1.5):
    """1.5-lod-drop confidence interval on a single chromosome.

    parameters
    ----------
    scan_result : ScanResult
    chrom : str
    drop : float  lod units to drop from peak (default 1.5)

    returns
    -------
    (lo_mb, hi_mb, peak_mb) floats
    """
    mask    = scan_result.chromosomes == chrom
    pos     = scan_result.positions[mask]
    lod     = scan_result.lod[mask]
    if len(lod) == 0:
        return np.nan, np.nan, np.nan
    peak_i  = np.argmax(lod)
    thresh  = lod[peak_i] - drop
    above   = np.where(lod >= thresh)[0]
    return float(pos[above[0]]), float(pos[above[-1]]), float(pos[peak_i])


def fixef_eqtl_scan(phenotype, genome_cache, covariates=None,
                    use_rint=True, phenotype_id='unknown',
                    just_these_loci=None):
    """fixed-effect only eQTL scan — no kinship, OLS F-test.

    fast scan for screening or stability analysis.
    equivalent to miqtl::fixef.eQTL.scan().

    parameters
    ----------
    phenotype : np.ndarray (n,)
    genome_cache : GenomeCache
    covariates : np.ndarray (n, p) or None
    use_rint : bool
    phenotype_id : str
    just_these_loci : set or None  restrict to these locus ids

    returns
    -------
    ScanResult  (h2_null will be empty dict)
    """
    from scipy import stats as _stats
    from .utils import rint as _rint, drop_colinear_founders as _drop

    valid = ~np.isnan(phenotype)
    y     = phenotype[valid].copy()
    if use_rint:
        y = _rint(y)
    n     = len(y)

    x = (covariates[valid].copy() if covariates is not None
         else np.ones((n, 1), dtype=float))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])
    p = x.shape[1]

    # precompute null residuals once
    xTx_inv = np.linalg.pinv(x.T @ x)
    y_hat0  = x @ (xTx_inv @ x.T @ y)
    e0      = y - y_hat0
    rss0    = float(e0 @ e0)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genome_cache.chromosomes:
        lids, pos, probs = genome_cache.get_chromosome(chrom)
        nf1 = genome_cache.n_founders - 1

        for i_l, lid in enumerate(lids):
            if just_these_loci and lid not in just_these_loci:
                continue
            h   = probs[i_l][valid, :nf1]          # (n, J-1)
            z   = np.hstack([h, x])                 # (n, J-1+p)
            q   = nf1
            try:
                beta = np.linalg.lstsq(z, y, rcond=None)[0]
                e1   = y - z @ beta
                rss1 = float(e1 @ e1)
            except Exception:
                all_lod.append(0.0); all_pval.append(1.0)
                all_betas.append(np.zeros(nf1))
                all_lids.append(lid); all_chrs.append(chrom)
                all_pos.append(pos[i_l]); continue
            lod  = (n / 2.0) * np.log10(max(rss0, 1e-30) / max(rss1, 1e-30))
            lod  = max(lod, 0.0)
            df2  = n - p - q
            if df2 > 0 and rss1 > 0:
                f   = ((rss0 - rss1) / q) / (rss1 / df2)
                pv  = 1.0 - _stats.f.cdf(max(f, 0), q, df2)
            else:
                pv  = 1.0
            all_lod.append(lod); all_pval.append(pv)
            all_betas.append(beta[:nf1])
            all_lids.append(lid); all_chrs.append(chrom)
            all_pos.append(pos[i_l])

    return ScanResult(
        locus_ids   = np.array(all_lids),
        chromosomes = np.array(all_chrs),
        positions   = np.array(all_pos, dtype=float),
        lod         = np.array(all_lod),
        p_value     = np.array(all_pval),
        betas       = np.vstack(all_betas) if all_betas else np.empty((0, genome_cache.n_founders - 1)),
        h2_null     = {},
        phenotype_id = phenotype_id,
    )


def instability_lm_scan(phenotype, genome_cache, covariates=None,
                        use_rint=True, n_samples=100, subsample_prop=0.63,
                        seed=1, phenotype_id='unknown', just_these_loci=None):
    """bootstrap instability scan — fixed-effect only.

    subsamples strains b times, runs fixef_eqtl_scan() on each subsample,
    reports the proportion of bootstraps where each locus is the peak and
    the mean lod across bootstraps.

    equivalent to miqtl::instability.lm.scan().

    parameters
    ----------
    phenotype : np.ndarray (n,)
    genome_cache : GenomeCache
    covariates : np.ndarray (n, p) or None
    use_rint : bool
    n_samples : int  number of bootstrap subsamples
    subsample_prop : float  proportion of strains per subsample (0.63)
    seed : int
    phenotype_id : str
    just_these_loci : set or None

    returns
    -------
    ScanResult  with lod = mean bootstrap lod, p_value = peak proportion
    """
    rng   = np.random.default_rng(seed)
    n     = len(phenotype)
    k     = max(int(n * subsample_prop), 2)

    # collect per-locus lod sums and peak counts across bootstraps
    lod_sum  = None; peak_cnt = None; meta = None

    for b in range(n_samples):
        idx    = rng.choice(n, size=k, replace=False)
        y_b    = phenotype[idx]
        cov_b  = covariates[idx] if covariates is not None else None

        # build a mini cache view for this subsample (reindex probs)
        class _SubCache:
            def __init__(self, cache, idx):
                self._c = cache; self._idx = idx
                self.chromosomes = cache.chromosomes
                self.n_founders  = cache.n_founders
                self.sample_ids  = [cache.sample_ids[i] for i in idx]
            def get_chromosome(self, chrom):
                lids, pos, probs = self._c.get_chromosome(chrom)
                return lids, pos, probs[:, self._idx, :]  # (n_loci, k, J)

        sub = _SubCache(genome_cache, idx)
        res = fixef_eqtl_scan(y_b, sub, cov_b, use_rint=use_rint,
                               phenotype_id=phenotype_id,
                               just_these_loci=just_these_loci)

        if lod_sum is None:
            lod_sum  = np.zeros(len(res.lod))
            peak_cnt = np.zeros(len(res.lod))
            meta     = (res.locus_ids, res.chromosomes, res.positions, res.betas)

        lod_sum += res.lod
        peak_idx = np.argmax(res.lod)
        peak_cnt[peak_idx] += 1

    lod_ids, lod_chrs, lod_pos, lod_betas = meta
    return ScanResult(
        locus_ids   = lod_ids,
        chromosomes = lod_chrs,
        positions   = lod_pos,
        lod         = lod_sum / n_samples,              # mean lod
        p_value     = 1.0 - peak_cnt / n_samples,       # 1 - peak proportion
        betas       = lod_betas,
        h2_null     = {},
        phenotype_id = phenotype_id,
    )


def scan_h2lmm_mi(phenotype, genome_cache, covariates=None,
                  kinship_decomp=None, n_imp=11, seed=1,
                  use_rint=True, use_reml=True, phenotype_id='unknown'):
    """multiple-imputation haplotype scan.

    samples n_imp diplotype states per locus from the haplotype probability
    distribution, runs scan_h2lmm() on each imputed dosage matrix, and
    averages lod scores across imputations.

    equivalent to miqtl::scan.h2lmm(use.multi.impute=TRUE, num.imp=n_imp).

    parameters
    ----------
    phenotype : np.ndarray (n,)
    genome_cache : GenomeCache
    covariates : np.ndarray (n, p) or None
    kinship_decomp : dict or None
    n_imp : int  number of imputations (default 11)
    seed : int
    use_rint : bool
    use_reml : bool
    phenotype_id : str

    returns
    -------
    ScanResult  with lod = mean over n_imp imputations
    """
    from .genome_cache import GenomeCache as _GC

    rng = np.random.default_rng(seed)
    n   = np.sum(~np.isnan(phenotype))

    if kinship_decomp is None:
        kinship_dict   = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kinship_dict)

    lod_accum = None

    for imp in range(n_imp):
        # build imputed cache: at each locus, sample one founder state
        # for each strain from Categorical(p_1...p_J)
        imp_data = {}
        for chrom in genome_cache.chromosomes:
            lids, pos, probs = genome_cache.get_chromosome(chrom)
            n_loci, n_s, n_f = probs.shape
            # sample: (n_loci, n_s) integer founder indices
            cum = probs.cumsum(axis=2)                        # (L, n, J)
            u   = rng.random((n_loci, n_s, 1))               # uniform draws
            idx = (u > cum).sum(axis=2).clip(0, n_f - 1)     # (L, n)
            # convert to one-hot additive dosage (1 for sampled founder)
            imp_probs = np.zeros_like(probs)
            for l in range(n_loci):
                imp_probs[l, np.arange(n_s), idx[l]] = 1.0
            imp_data[chrom] = {'locus_ids': lids, 'positions': pos,
                               'probs': imp_probs}

        imp_cache = _GC(list(genome_cache.sample_ids),
                        list(genome_cache.founders), imp_data)
        res = scan_h2lmm(phenotype, imp_cache, covariates,
                         kinship_decomp, use_rint, use_reml, phenotype_id)
        if lod_accum is None:
            lod_accum = res.lod.copy()
            ref_res   = res
        else:
            lod_accum += res.lod

        logger.debug(f'{phenotype_id} mi {imp+1}/{n_imp} done')

    ref_res.lod = lod_accum / n_imp
    return ref_res
