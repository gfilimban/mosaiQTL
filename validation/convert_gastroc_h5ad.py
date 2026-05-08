"""Convert gastroc_all_pb_bulk.h5ad to HDF5 + metadata CSV for miQTL R scan.

Outputs (validation/gastroc_data/):
    metadata.csv        — sample_id, genotype, sex, dataset (canonical strain order)
    counts_lognorm.h5   — log2(CPM+1) matrix; HDF5 datasets: /counts (genes×samples),
                          /genes, /samples

Usage:
    python validation/convert_gastroc_h5ad.py
    python validation/convert_gastroc_h5ad.py --h5ad /path/to/other.h5ad
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import anndata
import h5py
import numpy as np
import pandas as pd

from mosaiqtl.genotype.founders import FOUNDER_ORDER, STRAIN_ORDER

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ── canonical genotype sort key ───────────────────────────────────────────────
# Founders (A/J→WSB/J) → F1s → CC strains (alphanumeric)
_NON_CC_ORDER: list[str] = list(STRAIN_ORDER)   # 8 founders + 7 F1s

def _strain_sort_key(genotype: str) -> tuple[int, str]:
    if genotype in _NON_CC_ORDER:
        return (0, str(_NON_CC_ORDER.index(genotype)).zfill(3))
    return (1, genotype)   # CC strains: alphanumeric within group


def convert(h5ad_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d samples × %d genes', adata.n_obs, adata.n_vars)

    # ── sort samples by canonical strain order then sample_id ─────────────────
    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    sorted_idx = obs.index   # original h5ad obs_names order after sort

    # ── log2(CPM + 1) normalisation ───────────────────────────────────────────
    # X is raw integer pseudobulk counts (samples × genes)
    X = adata[sorted_idx].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)

    lib_sizes = X.sum(axis=1, keepdims=True)            # (n_samples, 1)
    cpm = X / lib_sizes * 1e6
    lognorm = np.log2(cpm + 1.0)                        # (n_samples, genes)

    genes   = np.array(adata.var_names, dtype='S')      # bytes for HDF5
    samples = obs['sample_id'].values.astype('S')

    # ── write HDF5: /counts (genes × samples), /genes, /samples ──────────────
    h5_path = out_dir / 'counts_lognorm.h5'
    log.info('writing %s', h5_path)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('counts', data=lognorm.T, dtype='float32',
                         compression='gzip', compression_opts=4,
                         chunks=(min(1000, adata.n_vars), min(50, adata.n_obs)))
        f.create_dataset('genes',   data=genes)
        f.create_dataset('samples', data=samples)

    # ── write metadata CSV ────────────────────────────────────────────────────
    meta_path = out_dir / 'metadata.csv'
    log.info('writing %s', meta_path)
    obs.reset_index(drop=True).to_csv(meta_path, index=False)

    # ── write gene annotations CSV ────────────────────────────────────────────
    var = adata.var.copy()
    var = var.reset_index(names='_var_index')
    # use existing gene_id column if present, otherwise fall back to the index
    if 'gene_id' not in var.columns:
        var = var.rename(columns={'_var_index': 'gene_id'})
    else:
        var = var.drop(columns='_var_index')

    # normalise chromosome column — accept common naming variants
    chr_col = next(
        (c for c in var.columns if c.lower() in ('chromosome', 'chrom', 'chr', 'seqname', 'seqnames')),
        None,
    )
    start_col = next(
        (c for c in var.columns if c.lower() in ('start', 'chromstart', 'gene_start', 'start_position')),
        None,
    )
    end_col = next(
        (c for c in var.columns if c.lower() in ('end', 'chromend', 'gene_end', 'end_position')),
        None,
    )

    if chr_col is None:
        log.warning('no chromosome column found in adata.var — gene_info.csv will lack chr; '
                    'available columns: %s', list(var.columns))
    else:
        log.info('  gene annotations: chr_col=%s  start_col=%s  end_col=%s',
                 chr_col, start_col, end_col)

    # key must match adata.var_names — that's what gets written to HDF5 /genes
    # and read back by R as gene column names via make.names()
    gene_info = pd.DataFrame({'gene_id': adata.var_names})
    if chr_col is not None:
        # strip "chr" prefix if present so values match genomecache ("1".."19","X")
        gene_info['chr'] = (var[chr_col].astype(str)
                            .str.replace(r'^chr', '', regex=True))
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


def main() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    default_h5ad = (pathlib.Path.home() / 'Documents/journey/learning/graduate'
                    '/lab/projects/igvf/analysis/qtl/input/adata'
                    '/gastroc_all_pb_bulk.h5ad')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad', type=pathlib.Path, default=default_h5ad)
    parser.add_argument('--out',  type=pathlib.Path,
                        default=repo_root / 'validation' / 'gastroc_data')
    args = parser.parse_args()

    convert(args.h5ad, args.out)


if __name__ == '__main__':
    main()
