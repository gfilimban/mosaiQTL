"""ci — positional confidence intervals for QTL peaks.

lod_support_run       indices of the contiguous 1.5-LOD run around the peak.
lod_support_interval  the same run as positions in Mb.
ci_scan_multi_gene    runs a per-gene scan and returns CI bounds for a set of eGenes.

A single LMM scan yields the full LOD profile and the interval is the contiguous run of
loci within lod_drop of the peak. Unlike a phenotype bootstrap, which breaks the
genotype-phenotype pairing, the width follows the sharpness of the signal.
"""

import gc
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..utils import _build_gm_idx, _enrich_scan_df

logger = logging.getLogger('mosaiqtl')

CI_COLS: list[str] = [
    'gene_id', 'gene_symbol', 'chr', 'peak_lod', 'peak_mb', 'ci_peak_lod',
    'ci_lo', 'ci_hi', 'ci_width_mb', 'ci_truncated',
]

__all__ = [
    'CI_COLS',
    'lod_support_run',
    'lod_support_interval',
    'lod_drop_ci',
    'ci_scan_multi_gene',
]


def lod_support_run(lod: np.ndarray, lod_drop: float = 1.5) -> 'tuple[int, int]':
    r"""Indices of the contiguous run of loci around the peak within ``lod_drop`` of it.

    The run is the maximal block of consecutive loci containing the peak locus
    $\hat\ell = \arg\max_\ell \mathrm{LOD}_\ell$ on which
    $\mathrm{LOD}_\ell \ge \mathrm{LOD}_{\hat\ell} - \mathrm{lod\_drop}$. A locus with a
    non-finite LOD ends the run. Loci elsewhere above the threshold, such as a secondary
    peak, do not extend it.

    Parameters
    ----------
    lod : np.ndarray (n_loci,)
        LOD scores in position order.
    lod_drop : float
        LOD units below the peak that bound the run (default 1.5).

    Returns
    -------
    tuple[int, int]
        (first, last) locus indices of the run, both inclusive.
    """
    peak = int(np.nanargmax(lod))
    thr = lod[peak] - lod_drop
    keep = np.isfinite(lod) & (lod >= thr)
    lo = peak
    while lo > 0 and keep[lo - 1]:
        lo -= 1
    hi = peak
    last = len(lod) - 1
    while hi < last and keep[hi + 1]:
        hi += 1
    return lo, hi


def lod_support_interval(
    positions: np.ndarray,
    lod: np.ndarray,
    lod_drop: float = 1.5,
) -> 'tuple[float, float]':
    """LOD support interval: the contiguous run of loci within lod_drop of the peak.

    A support region is approximately a confidence region for the location of one QTL.
    Width reflects signal sharpness: a steep LOD peak gives a narrow interval and a flat
    or diffuse peak a wide one. The interval always contains the peak.

    Parameters
    ----------
    positions : np.ndarray (n_loci,)
        Locus positions in Mb, in the same order as ``lod``.
    lod : np.ndarray (n_loci,)
        LOD scores at each locus.
    lod_drop : float
        LOD units below the peak that bound the interval (default 1.5).

    Returns
    -------
    tuple[float, float]
        (ci_lo, ci_hi) in Mb.
    """
    lo, hi = lod_support_run(lod, lod_drop=lod_drop)
    return float(positions[lo]), float(positions[hi])


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
    chr_data: dict,
    x: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    use_rint: bool = True,
) -> 'tuple[np.ndarray, np.ndarray]':
    r"""Single-gene LMM scan of one chromosome at observation level.

    $y$ and $x$ hold one row per observation, and ``evals``/``evecs`` decompose the
    observation-level kinship $K_n = Z K_m Z^\top$, so every observation enters the scan
    once and no row is invented for a strain without observations.

    Returns
    -------
    positions : np.ndarray (n_loci,)  in Mb
    lod       : np.ndarray (n_loci,)
    """
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    y0 = _rint(y) if use_rint else y.astype(float).copy()
    h2, _ = lmm_module.estimate_h2_null(evecs.T @ y0, evecs.T @ x, evals, method='ml')
    result = lmm_module.scan_chromosome(y0, chr_data, x, evals, evecs, h2)
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
    gene_pos_mb_map: 'dict[str, float] | None' = None,
    window_mb: 'float | None' = None,
) -> pd.DataFrame:
    r"""LOD support interval CI for a set of eGenes, streaming to disk.

    For each gene, runs a single LMM scan on its chromosome and returns the contiguous
    1.5-LOD support interval around the peak. A strong, localised peak gives a narrow
    interval; a weak or diffuse one gives a wide interval.

    When ``gene_pos_mb_map`` is given, the profile is restricted to the loci with
    $|x_\ell - t_g| \le w$, the loci of the haplotype scan window, before the peak and the
    interval are read. The interval then lies inside the window and contains the peak.
    ``ci_truncated`` is True when the run reaches the first or last locus of the window.
    Without ``gene_pos_mb_map`` the whole chromosome is used and ``ci_truncated`` is False.

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
    gene_meta      : DataFrame with a 'gene_id' column — extra columns (gene_symbol, …)
                     joined into each output row
    gene_pos_mb_map: dict {gene_id: float} or None  strand-aware TSS in Mb; genes absent
                     from the map are skipped
    window_mb      : float or None  scan half-window in Mb; required with gene_pos_mb_map

    Returns
    -------
    pd.DataFrame  columns per CI_COLS: gene_id, chr [+ gene_meta cols], peak_lod,
                  peak_mb, ci_peak_lod, ci_lo, ci_hi, ci_width_mb, ci_truncated
    """
    if kinship_decomp is None:
        raise ValueError('kinship_decomp is required for ci_scan_multi_gene')
    if gene_pos_mb_map is not None and window_mb is None:
        raise ValueError('window_mb is required with gene_pos_mb_map')

    if output_path is not None:
        output_path.unlink(missing_ok=True)

    from . import lmm as lmm_module
    from ..utils import build_z_matrix

    _gm_idx = _build_gm_idx(gene_meta)
    _id_cols = ['gene_id', 'chr']
    _data_cols = ['peak_lod', 'peak_mb', 'ci_peak_lod', 'ci_lo', 'ci_hi', 'ci_width_mb', 'ci_truncated']
    _first = True
    _buf: 'list[pd.DataFrame]' = []
    _n = len(expr_cache)
    logger.info('ci_scan_multi_gene: n_genes=%d  lod_drop=%.1f', _n, lod_drop)

    # observation-level design shared by every gene: one row per sample with a genotype
    if not expr_cache:
        return pd.DataFrame(columns=CI_COLS)
    n_obs = next(iter(expr_cache.values())).shape[0]
    z_sub = None
    valid = np.ones(n_obs, dtype=bool)
    if sample_ids is not None:
        z_mat, valid = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        z_sub = z_mat[valid]
        del z_mat
    x = covariates[valid] if covariates is not None else np.ones((int(valid.sum()), 1))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((x.shape[0], 1)), x])

    # genes grouped by chromosome so that one chromosome is in memory at a time
    genes_by_chr: 'dict[str, list[tuple[str, np.ndarray, float | None]]]' = {}
    for gene_id, y in expr_cache.items():
        gene_chr = gene_chr_map.get(gene_id, '')
        if not gene_chr or peak_mb_map.get(gene_id) is None:
            logger.warning('ci_scan_multi_gene: missing chr/peak_mb for %s — skipped', gene_id)
            continue
        tss_mb = None
        if gene_pos_mb_map is not None:
            tss_mb = gene_pos_mb_map.get(gene_id)
            if tss_mb is None:
                logger.warning('ci_scan_multi_gene: no TSS for %s — skipped', gene_id)
                continue
        if gene_chr not in kinship_decomp:
            logger.warning(
                'ci_scan_multi_gene: no kinship decomposition for chr%s (%s) — skipped', gene_chr, gene_id
            )
            continue
        genes_by_chr.setdefault(gene_chr, []).append((gene_id, y[valid], tss_mb))

    for gene_chr, chr_genes in genes_by_chr.items():
        ev = kinship_decomp[gene_chr]['values']
        evc = kinship_decomp[gene_chr]['vectors']
        if z_sub is not None:
            _K_m = (evc * ev) @ evc.T
            evals, evecs = np.linalg.eigh(z_sub @ _K_m @ z_sub.T)
            del _K_m
        else:
            evals, evecs = ev, evc
        chr_data = lmm_module.precompute_chromosome(genomecache, gene_chr, evals, evecs, z_sub)

        for gene_id, y_g, tss_mb in chr_genes:
            try:
                gene_cd = chr_data
                if tss_mb is not None:
                    gene_cd = lmm_module.window_chromosome_data(chr_data, float(tss_mb), window_mb)
                    if gene_cd is None:
                        logger.warning('ci_scan_multi_gene: no loci in window for %s — skipped', gene_id)
                        continue
                positions, lod = _scan_one_chromosome(y_g, gene_cd, x, evals, evecs, use_rint)
                _i_lo, _i_hi = lod_support_run(lod, lod_drop=lod_drop)
                ci_lo, ci_hi = float(positions[_i_lo]), float(positions[_i_hi])
                _truncated = tss_mb is not None and (_i_lo == 0 or _i_hi == len(lod) - 1)
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
                    'ci_truncated': _truncated,
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
        del chr_data, evals, evecs

    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame(columns=CI_COLS)
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame(columns=CI_COLS)
