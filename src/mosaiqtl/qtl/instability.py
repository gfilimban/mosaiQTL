"""instability — bootstrap instability scans for haplotype QTL mapping.

fixef_eqtl_scan            plain fixed-effect (no kinship) LOD scan.
instability_lm_scan        repeated fixed-effect scan over strain subsamples.
instability_scan_multi_gene  per-gene instability scan over a set of genes, streaming.
"""

import gc
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..io.tables import read_table
from ..utils import rint, _build_gm_idx, _enrich_scan_df
from .scan import ScanResult, _resolve_z

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'fixef_eqtl_scan',
    'instability_lm_scan',
    'instability_scan_multi_gene',
]


def fixef_eqtl_scan(
    phenotype,
    genomecache,
    covariates=None,
    sample_ids=None,
    use_rint=True,
    phenotype_id='unknown',
    just_these_loci=None,
):
    """fixed-effect eQTL scan — plain least-squares regression, no kinship
    correction for relatedness.

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

    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    z_mat, valid_sid = _resolve_z(sid, genomecache)
    valid = (~np.isnan(phenotype)) & valid_sid
    y = phenotype[valid].copy()
    if use_rint:
        y = rint(y)
    n = len(y)
    z_sub = z_mat[valid] if z_mat is not None else None

    x = (
        covariates[valid].copy()
        if covariates is not None
        else np.ones((n, 1), dtype=float)
    )
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])
    p = x.shape[1]
    e0 = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    rss0 = float(e0 @ e0)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)
        nf = (
            genomecache.n_founders
        )  # use all founder columns; lstsq handles the rank deficiency
        for i_l, lid in enumerate(lids):
            if just_these_loci and lid not in just_these_loci:
                continue
            h = probs[i_l]  # haplotype probabilities for all founders at this locus
            if z_sub is not None:
                h = z_sub @ h
            z_ = np.hstack([h, x])
            try:
                beta = np.linalg.lstsq(z_, y, rcond=1e-10)[0]
                e1 = y - z_ @ beta
                rss1 = float(e1 @ e1)
            except np.linalg.LinAlgError:
                all_lod.append(0.0)
                all_pval.append(1.0)
                all_betas.append(np.zeros(nf))
                all_lids.append(lid)
                all_chrs.append(chrom)
                all_pos.append(pos[i_l])
                continue
            lod = max((n / 2.0) * np.log10(max(rss0, 1e-30) / max(rss1, 1e-30)), 0.0)
            # degrees of freedom contributed by the haplotype term alone
            q_eff = max(int(np.linalg.matrix_rank(z_, tol=1e-10)) - p, 0)
            df2 = n - p - q_eff
            if df2 > 0 and rss1 > 0 and q_eff > 0:
                f = ((rss0 - rss1) / q_eff) / (rss1 / df2)
                pv = 1.0 - _stats.f.cdf(max(f, 0), q_eff, df2)
            else:
                pv = 1.0
            all_lod.append(lod)
            all_pval.append(pv)
            all_betas.append(beta[:nf])
            all_lids.append(lid)
            all_chrs.append(chrom)
            all_pos.append(pos[i_l])

    nf_out = genomecache.n_founders
    return ScanResult(
        locus_ids=np.array(all_lids),
        chromosomes=np.array(all_chrs),
        positions=np.array(all_pos, dtype=float),
        lod=np.array(all_lod),
        p_value=np.array(all_pval),
        betas=(np.vstack(all_betas) if all_betas else np.empty((0, nf_out))),
        h2_null={},
        phenotype_id=phenotype_id,
    )


def instability_lm_scan(
    phenotype,
    genomecache,
    covariates=None,
    sample_ids=None,
    use_rint=True,
    n_subsamples=100,
    subsample_prop=0.63,
    seed=1,
    phenotype_id='unknown',
    just_these_loci=None,
):
    """bootstrap instability scan — fixed-effect only.

    repeatedly draws a random subset of strains, scans each subset the same
    way as fixef_eqtl_scan(), and reports the average LOD plus how often
    each locus came out as the peak across all the subsets. a locus that
    is consistently the peak across many random subsamples is a more
    trustworthy signal than one that only peaks once.

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

    logger.info(
        'instability_lm_scan: gene=%s  n_subsamples=%d  subsample_prop=%.2f',
        phenotype_id,
        n_subsamples,
        subsample_prop,
    )

    # reduce to strain means once, up front, so subsample indices always
    # fall in [0, n_strains) rather than the larger replicate-sample range
    if sample_ids is not None:
        _z, _vs = _resolve_z(list(sample_ids), genomecache)
        _valid = (~np.isnan(phenotype)) & _vs
        if _z is not None:
            _z_v = _z[_valid]
            _r = _z_v.sum(axis=0)
            y_scan = (_z_v.T @ phenotype[_valid]) / _r
            cov_scan = (
                (_z_v.T @ covariates[_valid]) / _r[:, None]
                if covariates is not None
                else None
            )
        else:
            y_scan = phenotype[_valid]
            cov_scan = covariates[_valid] if covariates is not None else None
    else:
        y_scan = phenotype
        cov_scan = covariates

    n = len(y_scan)
    k = max(int(n * subsample_prop), 2)

    # only look at chromosomes that actually contain a requested locus, so we
    # don't load the whole genome when just_these_loci restricts to one chromosome
    if just_these_loci:
        _loci_set = set(just_these_loci)
        _active_chroms = [
            c
            for c in genomecache.chromosomes
            if any(lid in _loci_set for lid in genomecache.get_chromosome(c)[0])
        ]
    else:
        _active_chroms = list(genomecache.chromosomes)

    class _SubCache:
        """genome cache view restricted to a subset of strain indices."""

        def __init__(self, cache, idx):
            """wrap the cache, remembering which strain indices to keep."""
            self._c = cache
            self._idx = idx
            self.chromosomes = _active_chroms
            self.n_founders = cache.n_founders
            self.sample_ids = [cache.sample_ids[i] for i in idx]

        def get_chromosome(self, chrom):
            """return locus ids, positions, and probs subset to the chosen strains."""
            lids, pos, probs = self._c.get_chromosome(chrom)
            return lids, pos, probs[:, self._idx, :]

    # load haplotype probabilities for every requested locus just once, up
    # front — much faster than re-reading the same data from disk on every subsample
    _loci_set = set(just_these_loci) if just_these_loci else None
    _lids_chunks, _chrs_chunks, _pos_chunks, _prob_chunks = [], [], [], []
    for chrom in _active_chroms:
        lids_c, pos_c, probs_c = genomecache.get_chromosome(chrom)
        if _loci_set:
            sel = [i for i, lid in enumerate(lids_c) if lid in _loci_set]
            if not sel:
                continue
            lids_c = np.asarray(lids_c)[sel]
            pos_c = pos_c[sel]
            probs_c = probs_c[sel]
        _lids_chunks.append(np.asarray(lids_c))
        _chrs_chunks.append(np.full(len(lids_c), chrom))
        _pos_chunks.append(pos_c)
        _prob_chunks.append(probs_c)

    probs_all = np.concatenate(_prob_chunks, axis=0)  # (L, n_strains, J)
    lids_all = np.concatenate(_lids_chunks)
    chrs_all = np.concatenate(_chrs_chunks)
    pos_all = np.concatenate(_pos_chunks)
    L = len(lids_all)
    J = genomecache.n_founders

    lod_sum = np.zeros(L)
    peak_cnt = np.zeros(L, dtype=int)
    betas_0 = None
    _t_inst = time.perf_counter()

    for _si in range(n_subsamples):
        idx = rng.choice(n, size=k, replace=False)
        y_b = y_scan[idx]
        if use_rint:
            y_b = rint(y_b)

        # build covariate matrix for this subsample
        if cov_scan is not None:
            x_b = cov_scan[idx].copy()
        else:
            x_b = np.ones((k, 1), dtype=float)
        if not np.allclose(x_b[:, 0], 1.0):
            x_b = np.hstack([np.ones((k, 1)), x_b])

        # remove the covariate effect from the phenotype once, up front
        e0_b = y_b - x_b @ np.linalg.lstsq(x_b, y_b, rcond=None)[0]
        rss0_b = float(e0_b @ e0_b)

        # haplotype probabilities for the subsampled strains at every locus
        H_b = probs_all[:, idx, :]

        # remove the same covariate effect from the haplotype probabilities
        Xp_b = np.linalg.pinv(x_b)
        proj = np.einsum('pn,lnj->lpj', Xp_b, H_b)
        H_perp = H_b - np.einsum('np,lpj->lnj', x_b, proj)
        del H_b, proj

        G = np.einsum('lni,lnj->lij', H_perp, H_perp)  # (L, J, J)
        c = np.einsum('lnj,n->lj', H_perp, e0_b)  # (L, J)
        G_pinv = np.linalg.pinv(G, rcond=1e-10)  # (L, J, J)
        rss_red = np.einsum('li,lij,lj->l', c, G_pinv, c)  # (L,)

        rss1 = np.maximum(rss0_b - rss_red, 1e-30)
        with np.errstate(divide='ignore', invalid='ignore'):
            lod_b = np.where(rss1 < rss0_b, (k / 2.0) * np.log10(rss0_b / rss1), 0.0)

        lod_sum += lod_b
        peak_cnt[np.argmax(lod_b)] += 1
        if betas_0 is None:
            betas_0 = np.einsum('lij,lj->li', G_pinv, c)  # (L, J)

        if (_si + 1) % 10 == 0 or _si == n_subsamples - 1:
            logger.debug(
                'instability_lm_scan: %s subsample %d/%d  elapsed=%.1fs',
                phenotype_id,
                _si + 1,
                n_subsamples,
                time.perf_counter() - _t_inst,
            )

    _inst_lod = lod_sum / n_subsamples
    _inst_peak_idx = int(np.argmax(peak_cnt))
    logger.info(
        'instability_lm_scan: %s done — peak_freq=%.3f at locus %s',
        phenotype_id,
        float(peak_cnt[_inst_peak_idx]) / n_subsamples,
        lids_all[_inst_peak_idx],
    )
    return ScanResult(
        locus_ids=lids_all,
        chromosomes=chrs_all,
        positions=pos_all,
        lod=_inst_lod,
        p_value=1.0 - peak_cnt / n_subsamples,
        betas=betas_0 if betas_0 is not None else np.empty((0, J)),
        h2_null={},
        phenotype_id=phenotype_id,
    )


def instability_scan_multi_gene(
    expr_cache: 'dict[str, np.ndarray]',
    gene_chr_map: 'dict[str, str]',
    genomecache,
    covariates: 'np.ndarray | None' = None,
    sample_ids: 'list[str] | None' = None,
    n_subsamples: int = 100,
    subsample_prop: float = 0.63,
    seed: int = 1,
    peak_mb_map: 'dict[str, float] | None' = None,
    window_mb: float = 10.0,
    peak_only: bool = True,
    output_path: 'Path | None' = None,
    gene_meta: 'pd.DataFrame | None' = None,
) -> pd.DataFrame:
    """instability scan over a set of genes, streaming to disk.

    calls instability_lm_scan per gene restricted to each gene's chromosome
    (or a local window when peak_mb_map is provided) and streams each result
    immediately, keeping peak memory bounded.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}
    gene_chr_map   : dict {gene_id: str}
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None
    sample_ids     : list[str] or None  genotype-level strain IDs
    n_subsamples   : int  bootstrap iterations per gene (default 100)
    subsample_prop : float  proportion of strains per subsample (default 0.63)
    seed           : int  base seed; incremented per gene
    peak_mb_map    : dict {gene_id: float} or None — S4 peak positions; when provided
                     scan is restricted to loci within ± window_mb of each gene's peak
    window_mb      : float  half-window in Mb for local locus filter (default 10.0)
    peak_only      : bool   when True (default) output one peak-locus row per gene;
                     when False output all scanned loci
    output_path    : Path or None  stream a TSV here; None accumulates in RAM
    gene_meta      : DataFrame keyed by 'gene_id' (the identity key; carries
                     gene_symbol and any other display columns) — extra
                     columns joined into each row

    returns
    -------
    peak_only=True  — gene_id, chr, locus [+ gene_meta cols], peak_mb, peak_lod
    peak_only=False — gene_id, chr [+ gene_meta cols], locus, pos_mb, lod
    """
    _gm_idx = _build_gm_idx(gene_meta)
    if peak_only:
        _id_cols = ['gene_id', 'chr', 'locus']
        _data_cols = ['peak_mb', 'peak_lod']
    else:
        _id_cols = ['gene_id', 'chr']
        _data_cols = ['locus', 'pos_mb', 'lod']
    _first = True
    _buf: 'list[pd.DataFrame]' = []
    _n_genes_inst = len(expr_cache)
    logger.info(
        'instability_scan_multi_gene: n_genes=%d  n_subsamples=%d',
        _n_genes_inst,
        n_subsamples,
    )
    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, '')
        if not gene_chr:
            logger.warning(
                'instability_scan_multi_gene: no chr for %s — skipped', gene_id
            )
            continue
        logger.debug(
            'instability_scan_multi_gene: [%d/%d] %s', _g + 1, _n_genes_inst, gene_id
        )
        try:
            lids_chr, _, pos_chr = genomecache.get_chromosome(gene_chr)
            peak_mb = peak_mb_map.get(gene_id) if peak_mb_map else None
            if peak_mb is not None:
                loci_set = {
                    lid
                    for lid, p in zip(lids_chr, pos_chr)
                    if abs(p - peak_mb) <= window_mb
                }
            else:
                loci_set = set(lids_chr)
            res = instability_lm_scan(
                y,
                genomecache,
                covariates=covariates,
                sample_ids=sample_ids,
                n_subsamples=n_subsamples,
                subsample_prop=subsample_prop,
                seed=seed + _g,
                phenotype_id=gene_id,
                just_these_loci=loci_set,
            )
            _df = pd.DataFrame(
                {
                    'gene_id': gene_id,
                    'chr': res.chromosomes,
                    'locus': res.locus_ids,
                    'pos_mb': res.positions,
                    'lod': res.lod,
                }
            )
            del res
            if peak_only and not _df.empty:
                _df = _df.loc[[_df['lod'].idxmax()]].rename(
                    columns={'pos_mb': 'peak_mb', 'lod': 'peak_lod'}
                )
            _df = _enrich_scan_df(_df, gene_id, _gm_idx, _id_cols, _data_cols)
            if output_path is not None:
                _df.to_csv(output_path, sep='\t', mode='a', header=_first, index=False)
                _first = False
            else:
                _buf.append(_df)
            del _df
        except Exception as exc:
            logger.warning('instability_scan_multi_gene: %s failed — %s', gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (
            read_table(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()
