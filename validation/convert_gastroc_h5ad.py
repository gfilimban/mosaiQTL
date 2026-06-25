"""Convert pseudobulk h5ad to HDF5 + metadata CSV for mosaiQTL / miQTL scan.

Whole-tissue mode (default):
    reads gastroc_all_pb_bulk.h5ad (one obs per sample)
    writes to validation/gastroc_data/:
        counts_lognorm.h5   metadata.csv   gene_info.csv

Cell-type mode (--cell_type):
    reads gastroc_all_pb.h5ad (one obs per sample × cell type)
    subsets to the requested cell type
    writes ONLY counts_lognorm.h5 to validation/gastroc_data/celltypes/{cell_type}/
    metadata.csv and gene_info.csv are shared from gastroc_data/ (same samples, same genes)

Usage:
    python validation/convert_gastroc_h5ad.py
    python validation/convert_gastroc_h5ad.py --cell_type myonuclei
    python validation/convert_gastroc_h5ad.py --list_celltypes
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


def _strain_sort_key(genotype: str) -> tuple[int, str]:
    if genotype in _NON_CC_ORDER:
        return (0, str(_NON_CC_ORDER.index(genotype)).zfill(3))
    return (1, genotype)


def _ct_safe(cell_type: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', cell_type.lower()).strip('_')


def _write_counts_h5(adata: anndata.AnnData, sorted_idx: 'pd.Index',
                     out_path: pathlib.Path) -> None:
    X = adata[sorted_idx].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    lib_sizes = X.sum(axis=1, keepdims=True)
    lognorm   = np.log2(X / lib_sizes * 1e6 + 1.0)
    genes   = np.array(adata.var_names, dtype='S')
    samples = adata[sorted_idx].obs['sample_id'].values.astype('S')
    log.info('writing %s', out_path)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('counts', data=lognorm.T, dtype='float32',
                         compression='gzip', compression_opts=4,
                         chunks=(min(1000, adata.n_vars), min(50, len(sorted_idx))))
        f.create_dataset('genes',   data=genes)
        f.create_dataset('samples', data=samples)


def convert(h5ad_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Whole-tissue conversion: writes counts_lognorm.h5, metadata.csv, gene_info.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d samples × %d genes', adata.n_obs, adata.n_vars)

    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    _write_counts_h5(adata, obs.index, out_dir / 'counts_lognorm.h5')

    meta_path = out_dir / 'metadata.csv'
    log.info('writing %s', meta_path)
    obs.reset_index(drop=True).to_csv(meta_path, index=False)

    # ── gene_info.csv ─────────────────────────────────────────────────────────
    var = adata.var.copy().reset_index(names='_var_index')
    if 'gene_id' not in var.columns:
        var = var.rename(columns={'_var_index': 'gene_id'})
    else:
        var = var.drop(columns='_var_index')

    chr_col   = next((c for c in var.columns
                      if c.lower() in ('chromosome', 'chrom', 'chr', 'seqname', 'seqnames')), None)
    start_col = next((c for c in var.columns
                      if c.lower() in ('start', 'chromstart', 'gene_start', 'start_position')), None)
    end_col   = next((c for c in var.columns
                      if c.lower() in ('end', 'chromend', 'gene_end', 'end_position')), None)

    if chr_col is None:
        log.warning('no chromosome column found in adata.var — gene_info.csv will lack chr')
    else:
        log.info('  gene annotations: chr_col=%s  start_col=%s  end_col=%s',
                 chr_col, start_col, end_col)

    gene_info = pd.DataFrame({'gene_id': adata.var_names})
    if chr_col is not None:
        gene_info['chr'] = var[chr_col].astype(str).str.replace(r'^chr', '', regex=True)
    if start_col is not None:
        gene_info['start'] = pd.to_numeric(var[start_col], errors='coerce').astype('Int64')
    if end_col is not None:
        gene_info['end'] = pd.to_numeric(var[end_col], errors='coerce').astype('Int64')

    gene_info_path = out_dir / 'gene_info.csv'
    log.info('writing %s  (%d genes)', gene_info_path, len(gene_info))
    gene_info.to_csv(gene_info_path, index=False)

    log.info('done — %d samples, %d genes', adata.n_obs, adata.n_vars)
    for ds, grp in obs.groupby('dataset', sort=True):
        log.info('  dataset=%-12s  %d samples  genotypes: %s',
                 ds, len(grp), ', '.join(sorted(grp['genotype'].unique())))


def convert_celltype(h5ad_path: pathlib.Path, cell_type: str,
                     out_root: pathlib.Path) -> None:
    """Cell-type conversion: subsets gastroc_all_pb.h5ad and writes counts_lognorm.h5 only.

    metadata.csv and gene_info.csv are shared from out_root/../ (same across all cell types).
    """
    log.info('reading %s', h5ad_path)
    adata_full = anndata.read_h5ad(h5ad_path)

    if 'celltype' not in adata_full.obs.columns:
        raise ValueError(f'obs has no "celltype" column; found: {list(adata_full.obs.columns)}')

    # ── subset to requested cell type ─────────────────────────────────────────
    # gastroc_all_pb.h5ad is already pseudobulked: one obs per (sample_id × celltype)
    adata = adata_full[adata_full.obs['celltype'] == cell_type].copy()
    if adata.n_obs == 0:
        available = sorted(adata_full.obs['celltype'].unique())
        raise ValueError(f'cell type {cell_type!r} not found. available: {available}')
    log.info('  cell_type=%r  %d samples × %d genes', cell_type, adata.n_obs, adata.n_vars)

    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    ct_dir = out_root / _ct_safe(cell_type)
    ct_dir.mkdir(parents=True, exist_ok=True)

    _write_counts_h5(adata, obs.index, ct_dir / 'counts_lognorm.h5')

    log.info('done — %d samples  (metadata.csv / gene_info.csv shared from %s)',
             adata.n_obs, out_root.parent)
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
    parser.add_argument('--cell_type', type=str, default=None,
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

    if args.cell_type is not None:
        h5ad = args.h5ad or _DEFAULT_CT_H5AD
        ct_root = args.out / 'celltypes'
        convert_celltype(h5ad, args.cell_type, ct_root)
    else:
        h5ad = args.h5ad or _DEFAULT_BULK_H5AD
        convert(h5ad, args.out)


if __name__ == '__main__':
    main()
