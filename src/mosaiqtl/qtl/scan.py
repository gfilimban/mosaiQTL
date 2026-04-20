"""
scan — genome-wide haplotype QTL scan

orchestrates the scan across chromosomes for a single phenotype,
combining genome cache, kinship, and the LMM engine.

global design — pheno_id / geno_id split:
    pheno_id : identifiers for each row of phenotype data (n)
    geno_id  : identifiers for rows of the genome cache (m ≤ n)

when n > m (replicated CC strains), a Z matrix (n×m) maps each
phenotype sample to its genotype:
    H_n = Z H_m     kinship expanded to K_n = Z K_m Z^T

when n == m and ids match, Z == I and all results are identical
to the 1:1 case.

"""

import numpy as np
import pandas as pd
import logging

from ..genotype import kinship as kin_module
from . import lmm as lmm_module
from ..utils import rint, build_z_matrix, expand_kinship

logger = logging.getLogger('mosaiqtl')


class ScanResult:
    """container for genome-wide scan output.

    attributes
    ----------
    locus_ids    : np.ndarray (L,) str   marker identifiers
    chromosomes  : np.ndarray (L,) str   chromosome per locus
    positions    : np.ndarray (L,) float physical position (Mb)
    lod          : np.ndarray (L,) float LOD scores
    p_value      : np.ndarray (L,) float F-test p-values
    betas        : np.ndarray (L, J-1)   founder haplotype effects
    h2_null      : dict {chrom: float}   null h² per LOCO chr
    phenotype_id : str
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def peak_locus(self):
        """return dict with peak locus info."""
        idx = np.argmax(self.lod)
        return {'locus_id': self.locus_ids[idx],
                'chromosome': self.chromosomes[idx],
                'position': self.positions[idx],
                'lod': self.lod[idx],
                'p_value': self.p_value[idx],
                'betas': self.betas[idx]}

    def to_dataframe(self):
        """convert to pandas DataFrame."""
        df = pd.DataFrame({'locus_id': self.locus_ids,
                           'chromosome': self.chromosomes,
                           'position': self.positions,
                           'lod': self.lod,
                           'p_value': self.p_value})
        for fi in range(self.betas.shape[1]):
            df[f'beta_f{fi}'] = self.betas[:, fi]
        return df


def _resolve_z(sample_ids, genomecache, pheno_id=None, geno_id=None):
    """build (Z, valid_mask, n_eff) for a given sample set.

    if sample_ids already matches the cache order exactly (n == m, same ids),
    returns Z=None (signals identity to callers — no expansion needed).
    otherwise builds the full Z matrix.

    parameters
    ----------
    sample_ids   : array-like  phenotype-level ids (n)
    genomecache : GenomeCache
    pheno_id     : str or None  column name in sample_ids (if DataFrame)
    geno_id      : str or None  column name for geno lookup (if DataFrame)

    returns
    -------
    z_mat       : np.ndarray (n, m) or None
    valid_mask  : np.ndarray (n,) bool
    """
    cache_ids = list(genomecache.sample_ids)
    sid_list  = list(sample_ids)
    if sid_list == cache_ids:
        return None, np.ones(len(sid_list), dtype=bool)
    z_mat, valid = build_z_matrix(sid_list, cache_ids)
    return z_mat, valid


def scan_h2lmm(phenotype, genomecache, covariates=None,
               kinship_decomp=None, sample_ids=None,
               use_rint=True, use_reml=True, phenotype_id='unknown',
               fixed_h2_dict=None):
    """haplotype-based genome-wide scan with LOCO LMM.

    supports both 1:1 (n == m) and replicated (n > m) designs via
    the Z-matrix expansion. when sample_ids is provided and differs
    from genomecache.sample_ids, Z is built automatically.


    parameters
    ----------
    phenotype       : np.ndarray (n,)    phenotype values; NaN excluded
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    kinship_decomp  : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids      : array-like (n,) or None
        phenotype-level sample identifiers. if None, assumed to match
        genomecache.sample_ids in order (1:1 case).
    use_rint        : bool  apply rank-inverse normal transformation
    use_reml        : bool  REML for h² estimation (ML for LOD)
    phenotype_id    : str

    returns
    -------
    ScanResult
    """
    n = len(phenotype)

    # resolve Z mapping
    sid = (list(sample_ids) if sample_ids is not None
           else list(genomecache.sample_ids))
    z_mat, valid_sid = _resolve_z(sid, genomecache)

    # combine sample validity with phenotype NaN mask
    valid_pheno = ~np.isnan(phenotype)
    valid       = valid_pheno & valid_sid
    if valid.sum() < n:
        logger.info(f'{phenotype_id}: {n - valid.sum()} samples excluded '
                    f'({(~valid_pheno).sum()} NaN pheno, '
                    f'{(~valid_sid).sum()} missing geno)')

    y = phenotype[valid].copy()
    if use_rint:
        y = rint(y)

    # covariates
    if covariates is not None:
        x = covariates[valid].copy()
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((len(y), 1)), x])
    else:
        x = np.ones((len(y), 1))

    # Z matrix for this valid subset
    z_sub = z_mat[valid] if z_mat is not None else None

    # kinship
    if kinship_decomp is None:
        kd_raw  = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []
    h2_null_dict = {}

    # restrict to chromosomes with kinship available; avoids noisy skip messages
    # when kinship_decomp is built for a single chromosome (e.g. --chr 8)
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    for chrom in _chrom_list:

        _ev  = kinship_decomp[chrom]['values']
        _evc = kinship_decomp[chrom]['vectors']

        # if Z exists, expand kinship to sample space
        if z_sub is not None:
            k_m   = _evc @ np.diag(_ev) @ _evc.T   # reconstruct K_m
            k_n   = expand_kinship(k_m, z_sub)       # K_n = Z K_m Z'
            evals, evecs = np.linalg.eigh(k_n)
        else:
            evals, evecs = _ev, _evc[valid] if _evc.shape[0] > len(y) else _evc

        y_rot = evecs.T @ y
        x_rot = evecs.T @ x
        # use fixed h2 when provided (e.g. from scan_h2lmm_mi)
        if fixed_h2_dict is not None and chrom in fixed_h2_dict:
            h2 = fixed_h2_dict[chrom]
        else:
            h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, evals,
                                                 method='reml' if use_reml else 'ml')
        h2_null_dict[chrom] = h2

        chr_res = lmm_module.scan_chromosome(
            y, genomecache, chrom, x, evals, evecs, h2, use_reml, z_sub)

        n_loci = len(chr_res['locus_ids'])
        all_lids.append(chr_res['locus_ids'])
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(chr_res['positions'])
        all_lod.append(chr_res['lod'])
        all_pval.append(chr_res['p_value'])
        all_betas.append(chr_res['betas'])

    return ScanResult(
        locus_ids   = np.concatenate(all_lids),
        chromosomes = np.concatenate(all_chrs),
        positions   = np.concatenate(all_pos),
        lod         = np.concatenate(all_lod),
        p_value     = np.concatenate(all_pval),
        betas       = np.vstack(all_betas),
        h2_null     = h2_null_dict,
        phenotype_id = phenotype_id,
    )


def scan_multi_gene(phenotype_matrix, gene_ids, genomecache,
                    covariates=None, kinship_decomp=None,
                    sample_ids=None, use_rint=True, n_cores=1):
    """scan multiple genes sharing the same genome cache and kinship.

    parameters
    ----------
    phenotype_matrix : np.ndarray (n, G)  columns = genes
    gene_ids         : list of str
    genomecache     : GenomeCache
    covariates       : np.ndarray (n, p) or None
    kinship_decomp   : dict or None
    sample_ids       : array-like (n,) or None
    use_rint         : bool
    n_cores          : int  (1 = serial)

    returns
    -------
    dict {gene_id: ScanResult}
    """
    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    results = {}
    if n_cores == 1:
        for gi, gid in enumerate(gene_ids):
            logger.info(f'scanning {gi+1}/{len(gene_ids)}: {gid}')
            results[gid] = scan_h2lmm(
                phenotype_matrix[:, gi], genomecache, covariates,
                kinship_decomp, sample_ids, use_rint, phenotype_id=gid)
    else:
        from multiprocessing import Pool

        def _one(args):
            gi, gid = args
            return gid, scan_h2lmm(
                phenotype_matrix[:, gi], genomecache, covariates,
                kinship_decomp, sample_ids, use_rint, phenotype_id=gid)
        with Pool(n_cores) as pool:
            for gid, res in pool.imap_unordered(_one, enumerate(gene_ids)):
                results[gid] = res
    return results


def grab_locus_from_scan(scan_result):
    """return the locus id with the highest LOD score.

        """
    return scan_result.locus_ids[np.argmax(scan_result.lod)]


def lod_drop_ci(scan_result, chrom, drop=1.5):
    """1.5-LOD-drop confidence interval on a single chromosome.

    returns (lo_mb, hi_mb, peak_mb).
    """
    mask   = scan_result.chromosomes == chrom
    pos    = scan_result.positions[mask]
    lod    = scan_result.lod[mask]
    if len(lod) == 0:
        return np.nan, np.nan, np.nan
    pi     = np.argmax(lod)
    thresh = lod[pi] - drop
    above  = np.where(lod >= thresh)[0]
    return float(pos[above[0]]), float(pos[above[-1]]), float(pos[pi])


def fixef_eqtl_scan(phenotype, genomecache, covariates=None,
                    sample_ids=None, use_rint=True,
                    phenotype_id='unknown', just_these_loci=None):
    """fixed-effect eQTL scan — OLS, no kinship.


    parameters
    ----------
    phenotype       : np.ndarray (n,)
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    sample_ids      : array-like (n,) or None
    use_rint        : bool
    phenotype_id    : str
    just_these_loci : set or None

    returns
    -------
    ScanResult  (h2_null = {})
    """
    from scipy import stats as _stats

    sid   = (list(sample_ids) if sample_ids is not None
             else list(genomecache.sample_ids))
    z_mat, valid_sid = _resolve_z(sid, genomecache)
    valid = (~np.isnan(phenotype)) & valid_sid
    y     = phenotype[valid].copy()
    if use_rint:
        y = rint(y)
    n     = len(y)
    z_sub = z_mat[valid] if z_mat is not None else None

    x = (covariates[valid].copy() if covariates is not None
         else np.ones((n, 1), dtype=float))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])
    p     = x.shape[1]
    e0    = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    rss0  = float(e0 @ e0)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)
        nf1 = genomecache.n_founders - 1
        for i_l, lid in enumerate(lids):
            if just_these_loci and lid not in just_these_loci:
                continue
            h   = probs[i_l]
            if z_sub is not None:
                h = z_sub @ h
            h   = h[:, :nf1]
            z_  = np.hstack([h, x])
            q   = nf1
            try:
                beta = np.linalg.lstsq(z_, y, rcond=None)[0]
                e1   = y - z_ @ beta
                rss1 = float(e1 @ e1)
            except Exception:
                all_lod.append(0.0); all_pval.append(1.0)
                all_betas.append(np.zeros(nf1)); all_lids.append(lid)
                all_chrs.append(chrom); all_pos.append(pos[i_l]); continue
            lod  = max((n/2.0)*np.log10(max(rss0,1e-30)/max(rss1,1e-30)), 0.0)
            df2  = n - p - q
            if df2 > 0 and rss1 > 0:
                f   = ((rss0-rss1)/q)/(rss1/df2)
                pv  = 1.0 - _stats.f.cdf(max(f,0), q, df2)
            else:
                pv  = 1.0
            all_lod.append(lod); all_pval.append(pv); all_betas.append(beta[:nf1])
            all_lids.append(lid); all_chrs.append(chrom); all_pos.append(pos[i_l])

    nf1_out = genomecache.n_founders - 1
    return ScanResult(
        locus_ids   = np.array(all_lids),
        chromosomes = np.array(all_chrs),
        positions   = np.array(all_pos, dtype=float),
        lod         = np.array(all_lod),
        p_value     = np.array(all_pval),
        betas       = (np.vstack(all_betas) if all_betas
                       else np.empty((0, nf1_out))),
        h2_null     = {},
        phenotype_id = phenotype_id,
    )


def instability_lm_scan(phenotype, genomecache, covariates=None,
                        sample_ids=None, use_rint=True,
                        n_samples=100, subsample_prop=0.63,
                        seed=1, phenotype_id='unknown',
                        just_these_loci=None):
    """bootstrap instability scan — fixed-effect only.

    subsamples strains b times, runs fixef_eqtl_scan() on each subsample,
    reports mean LOD and peak proportion across bootstraps.

        pre-generated sample matrix from generate.simple.sample.outcomes.matrix).
    this implementation handles sampling internally.

    parameters
    ----------
    phenotype       : np.ndarray (n,)
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    sample_ids      : array-like (n,) or None
    use_rint        : bool
    n_samples       : int    number of bootstrap subsamples
    subsample_prop  : float  proportion of samples per subsample (0.63)
    seed            : int
    phenotype_id    : str
    just_these_loci : set or None

    returns
    -------
    ScanResult  lod = mean bootstrap LOD, p_value = 1 − peak proportion
    """
    rng  = np.random.default_rng(seed)
    n    = len(phenotype)
    k    = max(int(n * subsample_prop), 2)

    lod_sum  = None; peak_cnt = None; meta = None

    for _ in range(n_samples):
        idx    = rng.choice(n, size=k, replace=False)
        y_b    = phenotype[idx]
        cov_b  = covariates[idx] if covariates is not None else None
        sid_b  = ([list(sample_ids)[i] for i in idx] if sample_ids is not None
                  else None)

        class _SubCache:
            def __init__(self, cache, idx):
                self._c = cache; self._idx = idx
                self.chromosomes = cache.chromosomes
                self.n_founders  = cache.n_founders
                self.sample_ids  = [cache.sample_ids[i] for i in idx]

            def get_chromosome(self, chrom):
                lids, pos, probs = self._c.get_chromosome(chrom)
                return lids, pos, probs[:, self._idx, :]

        sub = _SubCache(genomecache, idx)
        res = fixef_eqtl_scan(y_b, sub, cov_b, sid_b, use_rint,
                               phenotype_id, just_these_loci)
        if lod_sum is None:
            lod_sum  = np.zeros(len(res.lod))
            peak_cnt = np.zeros(len(res.lod))
            meta     = (res.locus_ids, res.chromosomes,
                        res.positions, res.betas)
        lod_sum += res.lod
        peak_cnt[np.argmax(res.lod)] += 1

    lids, chrs, pos, betas = meta
    return ScanResult(
        locus_ids   = lids, chromosomes = chrs, positions = pos,
        lod         = lod_sum / n_samples,
        p_value     = 1.0 - peak_cnt / n_samples,
        betas       = betas, h2_null = {},
        phenotype_id = phenotype_id,
    )


def scan_h2lmm_mi(phenotype, genomecache, covariates=None,
                  kinship_decomp=None, sample_ids=None,
                  n_imp=11, seed=1, use_rint=True,
                  use_reml=True, phenotype_id='unknown'):
    """multiple-imputation haplotype scan.

    samples n_imp diplotype states per locus and averages LOD scores.

    parameters
    ----------
    phenotype      : np.ndarray (n,)
    genomecache   : GenomeCache
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None
    sample_ids     : array-like (n,) or None
    n_imp          : int   number of imputations (default 11)
    seed           : int
    use_rint       : bool
    use_reml       : bool
    phenotype_id   : str

    returns
    -------
    ScanResult  lod = mean over n_imp imputations
    """
    from .genomecache import GenomeCache as _GC
    rng = np.random.default_rng(seed)

    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # only impute chromosomes that will actually be scanned
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]

    # pre-estimate h2 once from the REAL cache — held fixed across all imputations
    valid_mi = ~np.isnan(phenotype)
    y_base   = phenotype[valid_mi].copy()
    if use_rint:
        from ..utils import rint as _rint_mi
        y_base = _rint_mi(y_base)
    x_base = (covariates[valid_mi].copy() if covariates is not None
              else np.ones((valid_mi.sum(), 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(x_base), 1)), x_base])
    fixed_h2 = {}
    for _c in _chrom_list:
        _ev  = kinship_decomp[_c]['values']
        _evc = kinship_decomp[_c]['vectors'][valid_mi]
        _yr  = _evc.T @ y_base
        _xr  = _evc.T @ x_base
        _h2, _ = lmm_module.estimate_h2_null(_yr, _xr, _ev,
                                              method='reml' if use_reml else 'ml')
        fixed_h2[_c] = _h2

    lod_accum = None; ref_res = None

    for imp in range(n_imp):
        imp_data = {}
        for chrom in _chrom_list:
            lids, pos, probs = genomecache.get_chromosome(chrom)
            n_loci, n_s, n_f = probs.shape
            cum   = probs.cumsum(axis=2)
            u     = rng.random((n_loci, n_s, 1))
            idx   = (u > cum).sum(axis=2).clip(0, n_f - 1)  # (n_loci, n_s)
            imp_p = np.zeros_like(probs)
            # vectorized one-hot: no Python loop over loci
            l_idx = np.arange(n_loci)[:, None]   # (n_loci, 1)
            s_idx = np.arange(n_s)[None, :]       # (1, n_s)
            imp_p[l_idx, s_idx, idx] = 1.0
            imp_data[chrom] = {'locus_ids': lids, 'positions': pos,
                                'probs': imp_p}
        imp_cache = _GC(list(genomecache.sample_ids),
                        list(genomecache.founders), imp_data)
        res = scan_h2lmm(phenotype, imp_cache, covariates,
                         kinship_decomp, sample_ids, use_rint,
                         use_reml, phenotype_id,
                         fixed_h2_dict=fixed_h2)   # fixed from real cache
        if lod_accum is None:
            lod_accum = res.lod.copy(); ref_res = res
        else:
            lod_accum += res.lod
        logger.debug(f'{phenotype_id} MI {imp+1}/{n_imp} done')

    ref_res.lod = lod_accum / n_imp
    return ref_res
