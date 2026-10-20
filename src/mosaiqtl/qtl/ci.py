"""ci — positional confidence intervals for QTL peaks.

lod_support_interval  computes a 1.5-LOD drop interval from a LOD profile.
ci_scan_multi_gene    runs a per-gene scan and returns CI bounds for a set of eGenes.

The LOD support interval (Dupuis & Siegmund 1999) is the standard approach in
haplotype-based QTL mapping: a single LMM scan yields the full LOD profile and the
CI is all loci within lod_drop of the peak.  This is both faster and more correct
than phenotypic-bootstrap CI, which destroys the genotype-phenotype pairing and
produces near-chromosome-wide intervals regardless of signal strength.
"""

import gc
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..utils import _build_gm_idx, _enrich_scan_df

logger = logging.getLogger('mosaiqtl')

CI_COLS: list[str] = ['gene_id', 'gene_symbol', 'chr', 'peak_lod', 'peak_mb', 'ci_peak_lod', 'ci_lo', 'ci_hi', 'ci_width_mb']

__all__ = [
    'CI_COLS',
    'lod_support_interval',
    'lod_drop_ci',
    'ci_scan_multi_gene',
]


def lod_support_interval(
    positions: np.ndarray,
    lod: np.ndarray,
    lod_drop: float = 1.5,
) -> 'tuple[float, float]':
    """LOD support interval: all loci within lod_drop of the peak.

    The standard haplotype-QTL CI (Dupuis & Siegmund 1999; R/qtl lodint).
    Width reflects signal sharpness — a steep LOD peak gives a narrow CI;
    a flat or diffuse peak gives a wide one.

    Parameters
    ----------
    positions : np.ndarray (n_loci,)
        Locus positions in Mb.
    lod : np.ndarray (n_loci,)
        LOD scores at each locus.
    lod_drop : float
        Half-width of the interval in LOD units (default 1.5).

    Returns
    -------
    tuple[float, float]
        (ci_lo, ci_hi) in Mb.
    """
    peak_lod = float(np.nanmax(lod))
    above = positions[lod >= peak_lod - lod_drop]
    if len(above) == 0:
        pk = float(positions[int(np.nanargmax(lod))])
        return pk, pk
    return float(above.min()), float(above.max())


def lod_drop_ci(
    scan_result: 'object',
    chrom: str,
    drop: float = 1.5,
) -> 'tuple[float, float, float]':
    """LOD support interval from a ScanResult, returning (ci_lo, ci_hi, peak_mb).

    Convenience wrapper around :func:`lod_support_interval` that extracts arrays
    from a ScanResult object and also returns the peak position.  Returns
    (nan, nan, nan) when the result does not cover ``chrom``.

    Parameters
    ----------
    scan_result : ScanResult
        Output of scan_h2lmm or similar.
    chrom : str
        Chromosome to extract the interval for.
    drop : float
        LOD units below the peak that define the interval (default 1.5).

    Returns
    -------
    tuple[float, float, float]
        (ci_lo, ci_hi, peak_mb) in Mb; NaN when chrom not in scan_result.
    """
    _nan = float('nan')
    if getattr(scan_result, 'chr', None) != chrom:
        return _nan, _nan, _nan
    positions = scan_result.pos_mb
    lod = scan_result.lod
    if positions is None or len(positions) == 0:
        return _nan, _nan, _nan
    peak_idx = int(np.nanargmax(lod))
    peak_mb = float(positions[peak_idx])
    ci_lo, ci_hi = lod_support_interval(positions, lod, lod_drop=drop)
    return ci_lo, ci_hi, peak_mb


def _scan_one_chromosome(
    y: np.ndarray,
    genomecache,
    chrom: str,
    x_covariates: np.ndarray,
    kinship_decomp: dict,
    sample_ids: 'list[str] | None' = None,
    use_rint: bool = True,
) -> 'tuple[np.ndarray, np.ndarray]':
    """Single-gene single-chromosome LMM scan.

    Returns
    -------
    positions : np.ndarray (n_loci,)  in Mb
    lod       : np.ndarray (n_loci,)
    """
    from . import lmm as lmm_module
    from ..utils import rint as _rint, build_z_matrix

    ev = kinship_decomp[chrom]['values']
    evc = kinship_decomp[chrom]['vectors']

    n = y.shape[0]
    x = x_covariates if x_covariates is not None else np.ones((n, 1), dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1), dtype=float), x])

    y0 = _rint(y) if use_rint else y.astype(float).copy()
    x_scan = x
    z_mat = None
    r_counts = None

    if sample_ids is not None:
        z_mat, _ = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        r_counts = z_mat.sum(axis=0)

        # Rare cell types may have zero cells for some strains; dividing by zero
        # produces NaN that propagates through the scan.  Replace zero counts with 1
        # (safe divisor) — those strains' Z.T@y and Z.T@x sums are already 0, so
        # 0/1 = 0 = RINT mean, the correct mean imputation.  Covariates for missing
        # strains are then set to the valid-strain mean with the intercept forced to 1.
        # The genomecache always returns the full m-strain haplotype matrix, so we
        # must keep ev/evc at the original m-strain dimension — never subset them.
        _valid_strains = r_counts > 0
        if not _valid_strains.all():
            r_counts = np.where(_valid_strains, r_counts, 1.0)
        x_scan = (z_mat.T @ x) / r_counts[:, None]
        if not _valid_strains.all():
            x_scan[~_valid_strains] = x_scan[_valid_strains].mean(axis=0)
            x_scan[~_valid_strains, 0] = 1.0
        y0 = (z_mat.T @ y0) / r_counts
        # y0[~_valid_strains] = 0 (RINT mean) since Z.T@y0 = 0 for missing strains

    h2, _ = lmm_module.estimate_h2_null(evc.T @ y0, evc.T @ x_scan, ev, method='ml')
    chr_data = lmm_module.precompute_chromosome(genomecache, chrom, ev, evc, z_mat=None)
    result = lmm_module.scan_chromosome(y0, chr_data, x_scan, ev, evc, h2)
    return result['positions'], result['lod']


def ci_scan_multi_gene(
    expr_cache: 'dict[str, np.ndarray]',
    gene_chr_map: 'dict[str, str]',
    peak_mb_map: 'dict[str, float]',
    genomecache,
    peak_lod_map: 'dict[str, float] | None' = None,
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    sample_ids: 'list[str] | None' = None,
    use_rint: bool = True,
    lod_drop: float = 1.5,
    output_path: 'Path | None' = None,
    gene_meta: 'pd.DataFrame | None' = None,
) -> pd.DataFrame:
    """LOD support interval CI for a set of eGenes, streaming to disk.

    For each gene, runs a single LMM scan on its chromosome and returns the
    1.5-LOD support interval (all loci within lod_drop of the peak).  This is
    the standard approach in haplotype-based QTL mapping and correctly reflects
    signal sharpness — a strong, localised peak gives a narrow CI; a weak or
    diffuse one gives a wide CI.

    Parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map   : dict {gene_id: str}  chromosome per gene
    peak_mb_map    : dict {gene_id: float}  S4 peak position in Mb
    genomecache    : GenomeCache
    peak_lod_map   : dict {gene_id: float} or None  S4 peak LOD per gene; written to
                     the ``peak_lod`` column for direct comparison with ``ci_peak_lod``
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict  {chrom: {'values':, 'vectors':}}  LOCO kinship decomposition
    sample_ids     : list[str] or None  genotype-level strain IDs (for replicate-aware path)
    use_rint       : bool  apply rank-normal transform before scanning
    lod_drop       : float  LOD units below peak that define the interval (default 1.5)
    output_path    : Path or None  stream CSV here row-by-row; None accumulates in RAM
    gene_meta      : DataFrame with 'gene_id' index — extra columns (gene_symbol, …)
                     joined into each output row

    Returns
    -------
    pd.DataFrame  columns per CI_COLS: gene_id, chr [+ gene_meta cols], peak_lod,
                  peak_mb, ci_peak_lod, ci_lo, ci_hi, ci_width_mb
    """
    if kinship_decomp is None:
        raise ValueError('kinship_decomp is required for ci_scan_multi_gene')

    if output_path is not None:
        output_path.unlink(missing_ok=True)

    _gm_idx = _build_gm_idx(gene_meta)
    _id_cols = ['gene_id', 'chr']
    _data_cols = ['peak_lod', 'peak_mb', 'ci_peak_lod', 'ci_lo', 'ci_hi', 'ci_width_mb']
    _first = True
    _buf: 'list[pd.DataFrame]' = []
    _n = len(expr_cache)
    logger.info('ci_scan_multi_gene: n_genes=%d  lod_drop=%.1f', _n, lod_drop)

    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, '')
        peak_mb_s4 = peak_mb_map.get(gene_id)
        if not gene_chr or peak_mb_s4 is None:
            logger.warning('ci_scan_multi_gene: missing chr/peak_mb for %s — skipped', gene_id)
            continue
        logger.debug(
            'ci_scan_multi_gene: gene %d/%d (%.1f%%) %s',
            _g + 1, _n, 100.0 * (_g + 1) / _n, gene_id,
        )
        _kd_g = {gene_chr: kinship_decomp[gene_chr]}
        try:
            positions, lod = _scan_one_chromosome(
                y, genomecache, gene_chr, covariates, _kd_g,
                sample_ids=sample_ids, use_rint=use_rint,
            )
            ci_lo, ci_hi = lod_support_interval(positions, lod, lod_drop=lod_drop)
            _peak_idx = int(np.nanargmax(lod))
            peak_mb_scan = float(positions[_peak_idx])
            ci_peak_lod = float(lod[_peak_idx])
            del positions, lod
            _df = pd.DataFrame([{
                'gene_id': gene_id,
                'chr': gene_chr,
                'peak_lod': peak_lod_map.get(gene_id) if peak_lod_map else None,
                'peak_mb': peak_mb_scan,
                'ci_peak_lod': ci_peak_lod,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'ci_width_mb': ci_hi - ci_lo,
            }])
            _df = _enrich_scan_df(_df, gene_id, _gm_idx, _id_cols, _data_cols)
            if output_path is not None:
                _df.to_csv(output_path, mode='a', header=_first, index=False)
                _first = False
            else:
                _buf.append(_df)
            del _df
        except Exception as exc:
            logger.warning('ci_scan_multi_gene: %s failed — %s', gene_id, exc)

    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()
