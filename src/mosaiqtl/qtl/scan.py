"""
scan — genome-wide haplotype QTL scan

orchestrates the scan across chromosomes for a single phenotype,
combining genome cache, kinship, and the LMM engine.

global design — pheno_id / geno_id split:
    pheno_id : identifiers for each row of phenotype data (n)
    geno_id  : identifiers for rows of the genome cache (m ≤ n)

when n > m (replicated CC strains), a Z matrix (n×m) maps each
phenotype sample to its strain genotype. phenotype and covariates
are reduced to strain means (m,) before scanning. h² and the scan
both operate at strain level (m×m K_m eigendecomp); H is never
expanded to n.

when n == m and ids match, Z == I and all results are identical
to the 1:1 case.

"""

import numpy as np
import pandas as pd
import logging

from ..genotype import kinship as kin_module
from . import lmm as lmm_module
from ..utils import rint, build_z_matrix

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'ScanResult',
    'scan_h2lmm',
    'scan_h2lmm_mi',
    'scan_multi_gene',
    'grab_locus_from_scan',
    'lod_drop_ci',
    'find_peaks',
    'fixef_eqtl_scan',
    'instability_lm_scan',
]


class ScanResult:
    """container for genome-wide scan output.

    attributes
    ----------
    locus_ids    : np.ndarray (L,) str   marker identifiers
    chromosomes  : np.ndarray (L,) str   chromosome per locus
    positions    : np.ndarray (L,) float physical position (Mb)
    lod          : np.ndarray (L,) float LOD scores
    p_value      : np.ndarray (L,) float F-test p-values
    betas        : np.ndarray (L, J)     founder haplotype effects (sum ≈ 0)
    h2_null      : dict {chrom: float}   null h² per LOCO chr
    phenotype_id : str
    """
    def __init__(self, **kwargs):
        """Set all scan result fields from keyword arguments."""
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
               fixed_h2_dict=None, model='additive',
               scan_window='cis', gene_chr=None,
               gene_pos_mb=None, window_mb=1.0,
               sex=None):
    """haplotype-based genome-wide scan with LOCO LMM.

    supports both 1:1 (n == m) and replicated (n > m) designs.

    replicated design: when sample_ids differs from genomecache.sample_ids,
    Z is built automatically and phenotype/covariates are reduced to strain
    means before scanning. h² is estimated at strain level (m×m K_m
    eigendecomp) and the scan operates entirely in the m-dimensional strain
    space.

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
    model           : str  'additive' (default) or 'full'
        'additive' — 8 haplotype columns per locus (J founders).
        'full'     — J*(J+1)/2 diplotype columns constructed from additive
                     probs via outer product P(fi/fj)=P(fi)*P(fj); valid
                     for CC inbred data where heterozygous probs ≈ 0.
    scan_window     : str  'cis' | 'genome' | 'local'  (default 'cis')
        'cis'    — scan only the chromosome carrying the gene (gene_chr
                   required); equivalent to passing a single-chr kinship dict.
        'genome' — scan all chromosomes present in kinship_decomp.
        'local'  — cis scan restricted to loci within ±window_mb of the
                   gene TSS (gene_chr and gene_pos_mb required).
    gene_chr        : str or None  chromosome of the gene (required for cis/local)
    gene_pos_mb     : float or None  TSS position in Mb (required for local)
    window_mb       : float  half-window in Mb for local mode (default 1.0)

    returns
    -------
    ScanResult  (betas shape (L, J) for additive; (L, J*(J+1)/2) for full)
    """
    if scan_window in ('cis', 'local') and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")
    if scan_window == 'local' and gene_pos_mb is None:
        raise ValueError("scan_window='local' requires gene_pos_mb")
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

    # sex appended after intercept/covariates (X-linked dosage + hormone effects)
    if sex is not None:
        sex_col = np.asarray(sex, dtype=float)[valid].reshape(-1, 1)
        x = np.hstack([x, sex_col])

    # Z matrix for this valid subset
    z_sub = z_mat[valid] if z_mat is not None else None

    # replicate-aware: compute strain means once; scan and h² operate at m-strain level.
    if z_sub is not None:
        r_counts = z_sub.sum(axis=0)             # (m,) replicates per strain
        y        = (z_sub.T @ y) / r_counts      # strain-mean phenotype (m,)
        x        = (z_sub.T @ x) / r_counts[:, None]  # strain-mean covariates (m, p)
        z_sub    = None                           # H stays at m level in scan_chromosome

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
    if scan_window in ('cis', 'local'):
        _gene_chr = str(gene_chr)
        _chrom_list = [c for c in _chrom_list if c == _gene_chr]
    for chrom in _chrom_list:

        _ev  = kinship_decomp[chrom]['values']
        _evc = kinship_decomp[chrom]['vectors']

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
            y, genomecache, chrom, x, evals, evecs, h2, use_reml, z_sub, model)

        if scan_window == 'local':
            mask = np.abs(chr_res['positions'] - gene_pos_mb) <= window_mb
            if not mask.any():
                logger.debug(f'{phenotype_id}: no loci within {window_mb} Mb of '
                             f'{gene_pos_mb:.3f} on {chrom}; skipping')
                continue
            chr_res = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask)
                           else v)
                       for k, v in chr_res.items()}
            chr_res['locus_ids'] = np.asarray(chr_res['locus_ids'])[mask]

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
        model        = model,
    )


def scan_multi_gene(phenotype_matrix, gene_ids, genomecache,
                    covariates=None, kinship_decomp=None,
                    sample_ids=None, use_rint=True, n_cores=1,
                    scan_window='genome'):
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
    scan_window      : str  'genome' | 'cis' | 'local'  (default 'genome')

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
                kinship_decomp, sample_ids, use_rint, phenotype_id=gid,
                scan_window=scan_window)
    else:
        from multiprocessing import Pool

        def _one(args):
            """Worker: scan one gene; returns (gene_id, ScanResult)."""
            gi, gid = args
            return gid, scan_h2lmm(
                phenotype_matrix[:, gi], genomecache, covariates,
                kinship_decomp, sample_ids, use_rint, phenotype_id=gid,
                scan_window=scan_window)
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


def find_peaks(scan_result, threshold, drop=1.5, allow_multiple=False,
               min_lod_gap=1.5):
    """find significant QTL peaks in a genome-wide scan.

    mirrors qtl2::find_peaks(scanone, map, threshold, drop, peakdrop).

    parameters
    ----------
    scan_result   : ScanResult
    threshold     : float  minimum LOD to report a peak
    drop          : float  LOD drop for CI bounds (default 1.5)
    allow_multiple: bool   if True, report secondary peaks per chromosome
        that are above threshold AND at least min_lod_gap LOD below the
        primary peak (i.e. separated local maxima).  default False returns
        only the highest-LOD peak per chromosome.
    min_lod_gap   : float  minimum LOD separation between multiple peaks
        on the same chromosome (only used when allow_multiple=True)

    returns
    -------
    pd.DataFrame  columns: phenotype_id, chr, locus_id, pos_mb, lod,
                            ci_lo_mb, ci_hi_mb
        one row per peak, sorted by chr then pos_mb.
        empty DataFrame (same columns) when no locus exceeds threshold.
    """
    _COLS = ('phenotype_id', 'chr', 'locus_id', 'pos_mb',
             'lod', 'ci_lo_mb', 'ci_hi_mb')
    rows = []

    for chrom in np.unique(scan_result.chromosomes):
        mask = scan_result.chromosomes == chrom
        lod_c  = scan_result.lod[mask]
        pos_c  = scan_result.positions[mask]
        lids_c = scan_result.locus_ids[mask]

        if lod_c.max() < threshold:
            continue

        if not allow_multiple:
            # single best peak per chromosome
            pi = int(np.argmax(lod_c))
            lo, hi, pk = lod_drop_ci(scan_result, chrom, drop)
            rows.append((scan_result.phenotype_id, chrom,
                         lids_c[pi], float(pos_c[pi]),
                         float(lod_c[pi]), lo, hi))
        else:
            # all local maxima above threshold separated by >= min_lod_gap
            # a point is a local max if it is >= both neighbours
            n = len(lod_c)
            for i in range(n):
                if lod_c[i] < threshold:
                    continue
                left_ok  = (i == 0)     or (lod_c[i] >= lod_c[i - 1])
                right_ok = (i == n - 1) or (lod_c[i] >= lod_c[i + 1])
                if not (left_ok and right_ok):
                    continue
                # check separation from any already-accepted peak on this chr
                too_close = any(
                    abs(pos_c[i] - r[3]) < 1e-9   # same position
                    or (lod_c[i] < r[4] - min_lod_gap)  # not sep enough
                    for r in rows if r[1] == chrom
                )
                if too_close:
                    continue
                # build a temporary ScanResult view for this chromosome to
                # get the CI (lod_drop_ci operates on a ScanResult)
                _sr_c = ScanResult(
                    locus_ids=lids_c, chromosomes=np.array([chrom] * n),
                    positions=pos_c, lod=lod_c,
                    p_value=scan_result.p_value[mask],
                    betas=scan_result.betas[mask],
                    h2_null={chrom: scan_result.h2_null.get(chrom, 0.0)},
                    phenotype_id=scan_result.phenotype_id,
                )
                lo, hi, pk = lod_drop_ci(_sr_c, chrom, drop)
                rows.append((scan_result.phenotype_id, chrom,
                             lids_c[i], float(pos_c[i]),
                             float(lod_c[i]), lo, hi))

    return pd.DataFrame(rows, columns=list(_COLS))


def fixef_eqtl_scan(phenotype, genomecache, covariates=None,
                    sample_ids=None, use_rint=True,
                    phenotype_id='unknown', just_these_loci=None,
                    sex=None):
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
    if sex is not None:
        sex_col = np.asarray(sex, dtype=float)[valid].reshape(-1, 1)
        x = np.hstack([x, sex_col])
    p     = x.shape[1]
    e0    = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    rss0  = float(e0 @ e0)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)
        nf = genomecache.n_founders   # J — all founders; lstsq handles rank J-1
        for i_l, lid in enumerate(lids):
            if just_these_loci and lid not in just_these_loci:
                continue
            h   = probs[i_l]          # (n, J) — all founders
            if z_sub is not None:
                h = z_sub @ h
            z_  = np.hstack([h, x])
            try:
                beta = np.linalg.lstsq(z_, y, rcond=1e-10)[0]
                e1   = y - z_ @ beta
                rss1 = float(e1 @ e1)
            except Exception:
                all_lod.append(0.0); all_pval.append(1.0)
                all_betas.append(np.zeros(nf)); all_lids.append(lid)
                all_chrs.append(chrom); all_pos.append(pos[i_l]); continue
            lod  = max((n/2.0)*np.log10(max(rss0,1e-30)/max(rss1,1e-30)), 0.0)
            # effective rank: rank([H, X]) - rank(X) = haplotype df
            q_eff = max(int(np.linalg.matrix_rank(z_, tol=1e-10)) - p, 0)
            df2  = n - p - q_eff
            if df2 > 0 and rss1 > 0 and q_eff > 0:
                f   = ((rss0-rss1)/q_eff)/(rss1/df2)
                pv  = 1.0 - _stats.f.cdf(max(f, 0), q_eff, df2)
            else:
                pv  = 1.0
            all_lod.append(lod); all_pval.append(pv); all_betas.append(beta[:nf])
            all_lids.append(lid); all_chrs.append(chrom); all_pos.append(pos[i_l])

    nf_out = genomecache.n_founders
    return ScanResult(
        locus_ids   = np.array(all_lids),
        chromosomes = np.array(all_chrs),
        positions   = np.array(all_pos, dtype=float),
        lod         = np.array(all_lod),
        p_value     = np.array(all_pval),
        betas       = (np.vstack(all_betas) if all_betas
                       else np.empty((0, nf_out))),
        h2_null     = {},
        phenotype_id = phenotype_id,
    )


def instability_lm_scan(phenotype, genomecache, covariates=None,
                        sample_ids=None, use_rint=True,
                        n_subsamples=100, subsample_prop=0.63,
                        seed=1, phenotype_id='unknown',
                        just_these_loci=None, sex=None):
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
    n_subsamples    : int    number of bootstrap subsamples
    subsample_prop  : float  proportion of strains per subsample (0.63)
    seed            : int
    phenotype_id    : str
    just_these_loci : set or None

    returns
    -------
    ScanResult  lod = mean bootstrap LOD, p_value = 1 − peak proportion
    """
    rng = np.random.default_rng(seed)

    # pre-reduce to strain means once so subsample indices are always in [0, n_strains)
    if sample_ids is not None:
        _z, _vs = _resolve_z(list(sample_ids), genomecache)
        _valid  = (~np.isnan(phenotype)) & _vs
        _z_v    = _z[_valid]; _r = _z_v.sum(axis=0)
        y_scan  = (_z_v.T @ phenotype[_valid]) / _r
        cov_scan = ((_z_v.T @ covariates[_valid]) / _r[:, None]
                    if covariates is not None else None)
        sex_scan = ((_z_v.T @ np.asarray(sex, dtype=float)[_valid]) / _r
                    if sex is not None else None)
    else:
        y_scan   = phenotype
        cov_scan = covariates
        sex_scan = (np.asarray(sex, dtype=float) if sex is not None else None)

    n = len(y_scan)
    k = max(int(n * subsample_prop), 2)

    class _SubCache:
        """Genome cache view restricted to a subset of strain indices."""

        def __init__(self, cache, idx):
            """Wrap cache, storing the index subset for subsetting probs."""
            self._c = cache; self._idx = idx
            self.chromosomes = cache.chromosomes
            self.n_founders  = cache.n_founders
            self.sample_ids  = [cache.sample_ids[i] for i in idx]

        def get_chromosome(self, chrom):
            """Return lids, pos, and subset probs for chrom."""
            lids, pos, probs = self._c.get_chromosome(chrom)
            return lids, pos, probs[:, self._idx, :]

    lod_sum  = None; peak_cnt = None; meta = None

    for _ in range(n_subsamples):
        idx   = rng.choice(n, size=k, replace=False)
        sub   = _SubCache(genomecache, idx)
        y_b   = y_scan[idx]
        cov_b = cov_scan[idx] if cov_scan is not None else None
        sex_b = sex_scan[idx] if sex_scan is not None else None

        res = fixef_eqtl_scan(y_b, sub, cov_b, None, use_rint,
                               phenotype_id, just_these_loci, sex=sex_b)
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
        lod         = lod_sum / n_subsamples,
        p_value     = 1.0 - peak_cnt / n_subsamples,
        betas       = betas, h2_null = {},
        phenotype_id = phenotype_id,
    )


def scan_h2lmm_mi(phenotype, genomecache, covariates=None,
                  kinship_decomp=None, sample_ids=None,
                  n_imp=11, seed=1, use_rint=True,
                  use_reml=True, phenotype_id='unknown',
                  scan_window='cis', gene_chr=None,
                  gene_pos_mb=None, window_mb=1.0,
                  sex=None):
    """multiple-imputation haplotype scan.

    samples n_imp diplotype states per locus and averages LOD scores.

    parameters
    ----------
    phenotype      : np.ndarray (n_samples,)  one value per mouse
    genomecache    : GenomeCache
    covariates     : np.ndarray (n_samples, p) or None
    kinship_decomp : dict or None
    sample_ids     : array-like (n_samples,) or None  genotype-level strain IDs
    n_imp          : int   number of imputations (default 11)
    seed           : int
    use_rint       : bool
    use_reml       : bool
    phenotype_id   : str
    scan_window    : str  'cis' | 'genome' | 'local'  (default 'cis')
    gene_chr       : str or None  chromosome of the gene (required for cis/local)
    gene_pos_mb    : float or None  TSS position in Mb (required for local)
    window_mb      : float  half-window for local mode (default 1.0)

    returns
    -------
    ScanResult  lod = mean over n_imp imputations
    """
    rng = np.random.default_rng(seed)

    if scan_window in ('cis', 'local') and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")
    if scan_window == 'local' and gene_pos_mb is None:
        raise ValueError("scan_window='local' requires gene_pos_mb")

    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # only impute chromosomes that will actually be scanned
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ('cis', 'local'):
        _chrom_list = [c for c in _chrom_list if c == gene_chr]

    # pre-estimate h² once from the REAL cache — held fixed across all imputations.
    # replicate-aware: same strain-mean path as scan_h2lmm.
    _sid_mi    = (list(sample_ids) if sample_ids is not None
                  else list(genomecache.sample_ids))
    _z_mi, _vs = _resolve_z(_sid_mi, genomecache)
    valid_mi   = (~np.isnan(phenotype)) & _vs
    y_base     = phenotype[valid_mi].copy()
    if use_rint:
        from ..utils import rint as _rint_mi
        y_base = _rint_mi(y_base)
    x_base = (covariates[valid_mi].copy() if covariates is not None
              else np.ones((valid_mi.sum(), 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(x_base), 1)), x_base])
    if sex is not None:
        sex_col_mi = np.asarray(sex, dtype=float)[valid_mi].reshape(-1, 1)
        x_base = np.hstack([x_base, sex_col_mi])
    _zsub_mi = _z_mi[valid_mi] if _z_mi is not None else None
    if _zsub_mi is not None:
        _r_mi  = _zsub_mi.sum(axis=0)
        y_base = (_zsub_mi.T @ y_base) / _r_mi
        x_base = (_zsub_mi.T @ x_base) / _r_mi[:, None]
    fixed_h2 = {}
    for _c in _chrom_list:
        _ev  = kinship_decomp[_c]['values']
        _evc = (kinship_decomp[_c]['vectors'] if _zsub_mi is not None
                else kinship_decomp[_c]['vectors'][valid_mi])
        _yr  = _evc.T @ y_base
        _xr  = _evc.T @ x_base
        _h2, _ = lmm_module.estimate_h2_null(_yr, _xr, _ev,
                                              method='reml' if use_reml else 'ml')
        fixed_h2[_c] = _h2

    # ── one pass per chromosome — all n_imp imputations vectorised ──────────
    # scan_chromosome_mi generates n_imp one-hot draws at once and computes
    # H_perp/G/G_pinv in a single batched einsum, eliminating:
    #   - n_imp × GenomeCache object constructions
    #   - n_imp × scan_chromosome calls per chromosome
    all_lids: list[np.ndarray] = []
    all_chrs: list[np.ndarray] = []
    all_pos:  list[np.ndarray] = []
    all_lod:  list[np.ndarray] = []
    # p_value and betas from soft-prob scan on first chromosome (reference only)
    all_pval:  list[np.ndarray] = []
    all_betas: list[np.ndarray] = []
    h2_null_dict: dict[str, float] = {}

    for chrom in _chrom_list:
        ev  = kinship_decomp[chrom]['values']
        evc = kinship_decomp[chrom]['vectors']

        mean_lod = lmm_module.scan_chromosome_mi(
            y_base, genomecache, chrom,
            x_base, ev, evc, fixed_h2[chrom],
            n_imp=n_imp, rng=rng,
        )
        # reference scan on soft probs — for p_value/betas fields only
        ref = lmm_module.scan_chromosome(
            y_base, genomecache, chrom, x_base, ev, evc, fixed_h2[chrom],
        )
        n_loci = len(ref['locus_ids'])
        all_lids.append(ref['locus_ids'])
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(ref['positions'])
        all_lod.append(mean_lod)
        all_pval.append(ref['p_value'])
        all_betas.append(ref['betas'])
        h2_null_dict[chrom] = fixed_h2[chrom]
        logger.debug('%s MI chr%s done (%d loci)', phenotype_id, chrom, len(ref['locus_ids']))

    return ScanResult(
        locus_ids    = np.concatenate(all_lids),
        chromosomes  = np.concatenate(all_chrs),
        positions    = np.concatenate(all_pos),
        lod          = np.concatenate(all_lod),
        p_value      = np.concatenate(all_pval),
        betas        = np.vstack(all_betas),
        h2_null      = h2_null_dict,
        phenotype_id = phenotype_id,
        model        = 'additive',
    )
