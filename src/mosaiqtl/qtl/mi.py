"""mi — multiple-imputation haplotype QTL scans.

scan_h2lmm_mi       draws n_imp hard genotype calls per locus, averages LOD scores.
scan_multi_gene_mi  runs scan_h2lmm_mi over a set of genes, streaming to disk.
"""

import gc
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..genotype import kinship as kin_module
from . import lmm as lmm_module
from ..utils import rint, _build_gm_idx, _enrich_scan_df
from .scan import ScanResult, _resolve_z

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'scan_h2lmm_mi',
    'scan_multi_gene_mi',
]


def scan_h2lmm_mi(
    phenotype,
    genomecache,
    covariates=None,
    kinship_decomp=None,
    sample_ids=None,
    n_imp=11,
    seed=1,
    use_rint=True,
    phenotype_id='unknown',
    scan_window='cis',
    gene_chr=None,
    gene_pos_mb=None,
    window_mb=10.0,
):
    """multiple-imputation haplotype scan.

    at each locus, draws n_imp random hard genotype calls (rather than using
    the soft haplotype probabilities directly) and averages the resulting
    LOD scores. this captures the extra uncertainty that comes from genotype
    calling, rather than treating the probabilities as if they were exact.

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
    phenotype_id   : str
    scan_window    : str  'cis' | 'genome' | 'local'  (default 'cis')
    gene_chr       : str or None  chromosome of the gene (required for cis/local)
    gene_pos_mb    : float or None  TSS position in Mb (required for local)
    window_mb      : float  half-window for local mode (default 10.0)

    returns
    -------
    ScanResult  lod = mean over n_imp imputations
    """
    rng = np.random.default_rng(seed)

    if scan_window in ('cis', 'local') and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")
    if scan_window == 'local' and gene_pos_mb is None:
        raise ValueError("scan_window='local' requires gene_pos_mb")

    logger.info(
        'scan_h2lmm_mi: gene=%s  chr=%s  n_imp=%d  n_samples=%d',
        phenotype_id,
        gene_chr if gene_chr is not None else 'genome',
        n_imp,
        len(phenotype),
    )

    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # only impute chromosomes that will actually be scanned
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ('cis', 'local'):
        _chrom_list = [c for c in _chrom_list if c == str(gene_chr)]

    # estimate heritability once from the real haplotype probabilities and hold
    # it fixed across all imputations; replicate samples are reduced to strain
    # means the same way as in scan_h2lmm.
    _sid_mi = (
        list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    )
    _z_mi, _vs = _resolve_z(_sid_mi, genomecache)
    valid_mi = (~np.isnan(phenotype)) & _vs
    y_base = phenotype[valid_mi].copy()
    x_base = (
        covariates[valid_mi].copy()
        if covariates is not None
        else np.ones((valid_mi.sum(), 1), dtype=float)
    )
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(x_base), 1)), x_base])
    _zsub_mi = _z_mi[valid_mi] if _z_mi is not None else None
    if use_rint:
        y_base = rint(y_base)
    if _zsub_mi is not None:
        _r_mi = _zsub_mi.sum(axis=0)
        y_base = (_zsub_mi.T @ y_base) / _r_mi
        x_base = (_zsub_mi.T @ x_base) / _r_mi[:, None]
    fixed_h2 = {}
    for _c in _chrom_list:
        _ev = kinship_decomp[_c]['values']
        _evc = (
            kinship_decomp[_c]['vectors']
            if _zsub_mi is not None
            else kinship_decomp[_c]['vectors'][valid_mi]
        )
        _yr = _evc.T @ y_base
        _xr = _evc.T @ x_base
        _h2, _ = lmm_module.estimate_h2_null(_yr, _xr, _ev, method='ml')
        fixed_h2[_c] = _h2

    # one pass per chromosome; all n_imp imputations are drawn and scanned
    # together in a single batched computation rather than looping n_imp times
    all_lids: list[np.ndarray] = []
    all_chrs: list[np.ndarray] = []
    all_pos: list[np.ndarray] = []
    all_lod: list[np.ndarray] = []
    # p_value and betas are taken from a reference scan on the soft
    # probabilities (not from the imputations) — used for reporting only
    all_pval: list[np.ndarray] = []
    all_betas: list[np.ndarray] = []
    h2_null_dict: dict[str, float] = {}

    for chrom in _chrom_list:
        ev = kinship_decomp[chrom]['values']
        evc = kinship_decomp[chrom]['vectors']

        mean_lod = lmm_module.scan_chromosome_mi(
            y_base,
            genomecache,
            chrom,
            x_base,
            ev,
            evc,
            fixed_h2[chrom],
            n_imp=n_imp,
            rng=rng,
        )
        # reference scan on the soft probabilities, for p_value/betas fields only
        cd_ref = lmm_module.precompute_chromosome(genomecache, chrom, ev, evc)
        ref = lmm_module.scan_chromosome(
            y_base, cd_ref, x_base, ev, evc, fixed_h2[chrom]
        )
        n_loci = len(ref['locus_ids'])
        all_lids.append(ref['locus_ids'])
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(ref['positions'])
        all_lod.append(mean_lod)
        all_pval.append(ref['p_value'])
        all_betas.append(ref['betas'])
        h2_null_dict[chrom] = fixed_h2[chrom]
        logger.debug(
            'scan_h2lmm_mi: %s chr%s  n_loci=%d',
            phenotype_id,
            chrom,
            len(ref['locus_ids']),
        )

    _mi_lod = np.concatenate(all_lod)
    _mi_peak_idx = int(np.argmax(_mi_lod))
    logger.info(
        'scan_h2lmm_mi: %s done — peak_lod=%.3f (mean over %d imputations)',
        phenotype_id,
        float(_mi_lod[_mi_peak_idx]),
        n_imp,
    )
    return ScanResult(
        locus_ids=np.concatenate(all_lids),
        chromosomes=np.concatenate(all_chrs),
        positions=np.concatenate(all_pos),
        lod=_mi_lod,
        p_value=np.concatenate(all_pval),
        betas=np.vstack(all_betas),
        h2_null=h2_null_dict,
        phenotype_id=phenotype_id,
        model='additive',
    )


def scan_multi_gene_mi(
    expr_cache: 'dict[str, np.ndarray]',
    gene_chr_map: 'dict[str, str]',
    genomecache,
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    sample_ids: 'list[str] | None' = None,
    n_imp: int = 11,
    seed: int = 1,
    use_rint: bool = True,
    peak_mb_map: 'dict[str, float] | None' = None,
    window_mb: float = 10.0,
    peak_only: bool = True,
    output_path: 'Path | None' = None,
    gene_meta: 'pd.DataFrame | None' = None,
) -> pd.DataFrame:
    """multiple-imputation scan over a set of genes, streaming to disk.

    calls scan_h2lmm_mi per gene and writes each result to output_path
    immediately, keeping peak memory proportional to one gene rather than
    all genes simultaneously.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map   : dict {gene_id: str}  chromosome for cis scan
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids     : list[str] or None  genotype-level strain IDs
    n_imp          : int  imputations per gene (default 11)
    seed           : int  base seed; incremented per gene
    use_rint       : bool
    peak_mb_map    : dict {gene_id: float} or None — S4 peak positions; when provided
                     scan uses local window ± window_mb around each gene's peak
    window_mb      : float  half-window in Mb for local scan (default 10.0)
    peak_only      : bool   when True (default) output one peak-locus row per gene;
                     when False output all scanned loci (full per-locus profile)
    output_path    : Path or None  stream CSV here; None accumulates in RAM
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
    _n_genes_mi = len(expr_cache)
    logger.info(
        'scan_multi_gene_mi: n_genes=%d  n_imp=%d  scan_window=%s',
        _n_genes_mi,
        n_imp,
        'local' if peak_mb_map else 'cis',
    )
    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, '')
        if not gene_chr:
            logger.warning('scan_multi_gene_mi: no chr for %s — skipped', gene_id)
            continue
        logger.debug(
            'scan_multi_gene_mi: gene %d/%d (%.1f%%) %s',
            _g + 1,
            _n_genes_mi,
            100.0 * (_g + 1) / _n_genes_mi,
            gene_id,
        )
        peak_mb = peak_mb_map.get(gene_id) if peak_mb_map else None
        scan_kw: 'dict' = dict(
            scan_window='local' if peak_mb is not None else 'cis',
            gene_chr=gene_chr,
        )
        if peak_mb is not None:
            scan_kw['gene_pos_mb'] = peak_mb
            scan_kw['window_mb'] = window_mb
        try:
            res = scan_h2lmm_mi(
                y,
                genomecache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                sample_ids=sample_ids,
                n_imp=n_imp,
                seed=seed + _g,
                use_rint=use_rint,
                phenotype_id=gene_id,
                **scan_kw,
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
                _df.to_csv(output_path, mode='a', header=_first, index=False)
                _first = False
            else:
                _buf.append(_df)
            del _df
        except Exception as exc:
            logger.warning('scan_multi_gene_mi: %s failed — %s', gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()
