"""Convert pseudobulk h5ad to HDF5 + metadata CSV for mosaiQTL / miQTL scan.

Whole-tissue mode (default):
    reads bulk h5ad (one obs per sample)
    writes to --out/:
        counts_lognorm.h5   metadata.csv   exp_gene_info_{tissue}.csv

Cell-type mode (--celltype):
    reads cell-type h5ad (one obs per sample × cell type)
    subsets to the requested cell type
    writes counts_lognorm.h5 AND exp_gene_info_{ct}.csv to --out/celltypes/{ct}/
    metadata.csv is shared from --out/ (same samples for all cell types)

Usage:
    python validation/convert_h5ad.py --tissue gastroc
    python validation/convert_h5ad.py --tissue gastroc --celltype myonuclei
    python validation/convert_h5ad.py --tissue gastroc --list_celltypes
    python validation/convert_h5ad.py --tissue ctxhipp --out validation/ctxhipp_data --h5ad /path/to/ctxhipp.h5ad
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import re

import anndata
import h5py
import numpy as np
import pandas as pd

from mosaiqtl.genotype.founders import STRAIN_ORDER

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

_NON_CC_ORDER: list[str] = list(STRAIN_ORDER)

_DEFAULT_BULK_H5AD = (pathlib.Path.home() / 'Documents/journey/learning/graduate'
                      '/lab/projects/igvf/analysis/qtl/input/adata'
                      '/gastroc_all_pb_bulk.h5ad')
_DEFAULT_CT_H5AD   = (pathlib.Path.home() / 'Documents/journey/learning/graduate'
                      '/lab/projects/igvf/analysis/qtl/input/adata'
                      '/gastroc_all_pb.h5ad')

# global gene universe — generated once from GENCODE vM32 GTF, tissue-agnostic
_GLOBAL_GENE_INFO = pathlib.Path(__file__).parent / 'input' / 'gene_info.vm32.csv'


def _strain_sort_key(genotype: str) -> tuple[int, str]:
    if genotype in _NON_CC_ORDER:
        return (0, str(_NON_CC_ORDER.index(genotype)).zfill(3))
    return (1, genotype)


def _ct_safe(cell_type: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', cell_type.lower()).strip('_')


def _cpm_filter(X: np.ndarray, lib_sizes: np.ndarray, var: pd.DataFrame,
                min_cpm: float = 1.0,
                min_frac: float = 0.2) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (bool mask, report DataFrame) for CPM > min_cpm in >= min_frac of samples.

    lib_sizes: (n_samples,) total UMI per sample from adata.obs['psbulk_counts'].
    """
    n_samples = X.shape[0]
    cpm = X / lib_sizes[:, None] * 1e6
    n_pass = (cpm > min_cpm).sum(axis=0)
    threshold = int(np.ceil(min_frac * n_samples))
    keep = n_pass >= threshold

    X_dense = X if not hasattr(X, 'toarray') else X.toarray()
    report = pd.DataFrame({'gene_name': var.index})
    if 'gene_id' in var.columns:
        report['gene_id'] = var['gene_id'].values
    for col in ('chr', 'start', 'end', 'total_counts', 'mean_counts',
                'n_cells_by_counts', 'pct_dropout_by_counts'):
        if col in var.columns:
            report[col] = var[col].values
    report['n_samples_cpm_gt1'] = n_pass
    report['frac_samples_cpm_gt1'] = (n_pass / n_samples).round(4)
    report['mean_cpm'] = cpm.mean(axis=0).round(4)
    report['median_cpm'] = np.median(cpm, axis=0).round(4)
    report['n_samples_zero'] = (X_dense == 0).sum(axis=0)
    report['frac_samples_zero'] = ((X_dense == 0).sum(axis=0) / n_samples).round(4)
    return keep, report


def _write_counts_h5(adata: anndata.AnnData, sorted_idx: 'pd.Index',
                     out_path: pathlib.Path) -> None:
    sub = adata[sorted_idx]
    X = sub.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    lib_sizes = sub.obs['psbulk_counts'].values.astype(np.float32)
    lognorm   = np.log2(X / lib_sizes[:, None] * 1e6 + 1.0)
    genes   = np.array(adata.var_names, dtype='S')
    samples = adata[sorted_idx].obs['sample_id'].values.astype('S')
    log.info('writing %s', out_path)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('counts', data=lognorm.T, dtype='float32',
                         compression='gzip', compression_opts=4,
                         chunks=(min(1000, adata.n_vars), min(50, len(sorted_idx))))
        f.create_dataset('genes',   data=genes)
        f.create_dataset('samples', data=samples)


def convert(h5ad_path: pathlib.Path, out_dir: pathlib.Path, tissue: str) -> None:
    """Whole-tissue conversion: writes counts_lognorm.h5, metadata.csv, exp_gene_info_{tissue}.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d samples × %d genes', adata.n_obs, adata.n_vars)

    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    X = adata.X if not hasattr(adata.X, 'toarray') else adata.X.toarray()
    lib_sizes = adata.obs['psbulk_counts'].values.astype(np.float32)
    keep, report = _cpm_filter(X.astype(np.float32), lib_sizes, adata.var)
    log.info('  CPM filter: %d / %d genes pass (CPM>1 in ≥20%% of samples)',
             int(keep.sum()), adata.n_vars)
    adata = adata[:, keep].copy()

    _write_counts_h5(adata, obs.index, out_dir / 'counts_lognorm.h5')

    meta_path = out_dir / 'metadata.csv'
    log.info('writing %s', meta_path)
    obs.reset_index(drop=True).to_csv(meta_path, index=False)

    # ── exp_gene_info_{tissue}.csv ────────────────────────────────────────────
    # coordinates + strand from global GENCODE gene universe; CPM stats from filter
    if not _GLOBAL_GENE_INFO.exists():
        log.warning('global gene info not found at %s — skipping coordinate annotation',
                    _GLOBAL_GENE_INFO)
        global_gi: pd.DataFrame = pd.DataFrame(columns=['gene_name', 'gene_id', 'chr',
                                                         'start', 'end', 'strand'])
    else:
        global_gi = pd.read_csv(_GLOBAL_GENE_INFO,
                                 usecols=['gene_name', 'gene_id', 'chr', 'start', 'end', 'strand'])

    cpm_stats_cols = ['gene_name', 'total_counts', 'mean_counts', 'n_cells_by_counts',
                      'pct_dropout_by_counts', 'n_samples_cpm_gt1', 'frac_samples_cpm_gt1',
                      'mean_cpm', 'median_cpm', 'n_samples_zero', 'frac_samples_zero']
    exp_gene_info = (
        pd.DataFrame({'gene_name': adata.var_names})
        .merge(global_gi, on='gene_name', how='left')
        .merge(report[[c for c in cpm_stats_cols if c in report.columns]],
               on='gene_name', how='left')
    )
    n_missing = exp_gene_info['gene_id'].isna().sum()
    if n_missing:
        log.warning('  %d genes not found in global gene_info — chr/start/end/strand will be NaN',
                    n_missing)
    exp_path = out_dir / f'exp_gene_info_{tissue}.csv'
    log.info('writing %s  (%d genes)', exp_path, len(exp_gene_info))
    exp_gene_info.to_csv(exp_path, index=False)

    log.info('done — %d samples, %d genes', adata.n_obs, adata.n_vars)
    for ds, grp in obs.groupby('dataset', sort=True):
        log.info('  dataset=%-12s  %d samples  genotypes: %s',
                 ds, len(grp), ', '.join(sorted(grp['genotype'].unique())))


def convert_celltype(h5ad_path: pathlib.Path, celltype: str,
                     out_root: pathlib.Path, tissue: str) -> None:
    """Cell-type conversion: subsets pb h5ad; writes counts_lognorm.h5 + exp_gene_info_{ct}.csv.

    Gene set is inherited from the whole-tissue exp_gene_info_{tissue}.csv (same passing genes
    for all cell types — absent genes kept as zeros for cross-CT comparability).
    CT-specific CPM stats are computed for the whole-tissue gene set.
    metadata.csv is shared from out_root.parent (same samples, same order).
    """
    ct = _ct_safe(celltype)

    # ── load whole-tissue gene set + coordinates ───────────────────────────────
    wt_exp_path = out_root.parent / f'exp_gene_info_{tissue}.csv'
    if not wt_exp_path.exists():
        raise FileNotFoundError(
            f'whole-tissue exp_gene_info_{tissue}.csv not found at {wt_exp_path} — '
            f'run convert() (whole-tissue mode) first')
    wt_exp = pd.read_csv(wt_exp_path, usecols=['gene_name'])
    passing_genes = set(wt_exp['gene_name'])
    log.info('  whole-tissue gene set: %d genes', len(passing_genes))

    log.info('reading %s', h5ad_path)
    adata_full = anndata.read_h5ad(h5ad_path)

    if 'celltype' not in adata_full.obs.columns:
        raise ValueError(f'obs has no "celltype" column; found: {list(adata_full.obs.columns)}')

    # ── subset to requested cell type ─────────────────────────────────────────
    adata = adata_full[adata_full.obs['celltype'] == celltype].copy()
    if adata.n_obs == 0:
        available = sorted(adata_full.obs['celltype'].unique())
        raise ValueError(f'cell type {celltype!r} not found. available: {available}')

    # ── restrict to whole-tissue passing genes (preserve zeros for absent genes) ──
    ct_genes = [g for g in adata.var_names if g in passing_genes]
    adata = adata[:, ct_genes].copy()
    log.info('  celltype=%r  %d samples × %d genes', celltype, adata.n_obs, adata.n_vars)

    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    ct_dir = out_root / ct
    ct_dir.mkdir(parents=True, exist_ok=True)

    # ── CT-specific CPM stats (no filtering — whole-tissue gene set only) ──────
    X = adata.X if not hasattr(adata.X, 'toarray') else adata.X.toarray()
    X = X.astype(np.float32)
    lib_sizes = adata.obs['psbulk_counts'].values.astype(np.float32)
    cpm = X / lib_sizes[:, None] * 1e6
    n_samples = adata.n_obs
    ct_stats = pd.DataFrame({
        'gene_name':            adata.var_names,
        'n_samples_cpm_gt1':    (cpm > 1).sum(axis=0),
        'frac_samples_cpm_gt1': ((cpm > 1).sum(axis=0) / n_samples).round(4),
        'mean_cpm':             cpm.mean(axis=0).round(4),
        'median_cpm':           np.median(cpm, axis=0).round(4),
        'n_samples_zero':       (X == 0).sum(axis=0),
        'frac_samples_zero':    ((X == 0).sum(axis=0) / n_samples).round(4),
    })

    # ── build exp_gene_info_{ct}.csv with coordinates + CT CPM stats ─────────
    if _GLOBAL_GENE_INFO.exists():
        global_gi = pd.read_csv(_GLOBAL_GENE_INFO,
                                usecols=['gene_name', 'gene_id', 'chr', 'start', 'end', 'strand'])
    else:
        log.warning('global gene info not found at %s — coordinate columns will be NaN',
                    _GLOBAL_GENE_INFO)
        global_gi = pd.DataFrame(columns=['gene_name', 'gene_id', 'chr', 'start', 'end', 'strand'])

    exp_gene_info_ct = (
        pd.DataFrame({'gene_name': adata.var_names})
        .merge(global_gi[['gene_name', 'gene_id', 'chr', 'start', 'end', 'strand']],
               on='gene_name', how='left')
        .merge(ct_stats, on='gene_name', how='left')
    )
    exp_path = ct_dir / f'exp_gene_info_{ct}.csv'
    exp_gene_info_ct.to_csv(exp_path, index=False)
    log.info('writing %s  (%d genes)', exp_path, len(exp_gene_info_ct))

    _write_counts_h5(adata, obs.index, ct_dir / 'counts_lognorm.h5')

    log.info('done — %d samples  %d genes  (metadata.csv shared from %s)',
             adata.n_obs, adata.n_vars, out_root.parent)
    for ds, grp in obs.groupby('dataset', sort=True):
        log.info('  dataset=%-12s  %d samples', ds, len(grp))


def main() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    gastroc_data = repo_root / 'validation' / 'gastroc_data'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--h5ad', type=pathlib.Path, default=None,
                        help='path to input h5ad (defaults depend on mode)')
    parser.add_argument('--out', type=pathlib.Path, default=gastroc_data,
                        help='output root (default: validation/gastroc_data)')
    parser.add_argument('--tissue', type=str, default='gastroc',
                        help='tissue label used in output filenames (default: gastroc)')
    parser.add_argument('--celltype', type=str, default=None,
                        help='cell type to extract (e.g. myonuclei); omit for whole-tissue')
    parser.add_argument('--list_celltypes', action='store_true',
                        help='list available cell types in the cell-type h5ad and exit')
    args = parser.parse_args()

    if args.list_celltypes:
        h5ad = args.h5ad or _DEFAULT_CT_H5AD
        adata = anndata.read_h5ad(h5ad)
        for ct in sorted(adata.obs['celltype'].unique()):
            n = (adata.obs['celltype'] == ct).sum()
            print(f'  {ct!r:50s}  {n} samples')
        return

    if args.celltype is not None:
        h5ad = args.h5ad or _DEFAULT_CT_H5AD
        ct_root = args.out / 'celltypes'
        convert_celltype(h5ad, args.celltype, ct_root, args.tissue)
    else:
        h5ad = args.h5ad or _DEFAULT_BULK_H5AD
        convert(h5ad, args.out, args.tissue)


if __name__ == '__main__':
    main()
