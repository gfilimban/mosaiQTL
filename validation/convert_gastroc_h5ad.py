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

    log.info('done — %d samples, %d genes', adata.n_obs, adata.n_vars)
    log.info('canonical order: %d founders, %d F1s, %d CC strains',
             sum(1 for g in obs['genotype'] if g in FOUNDER_ORDER),
             sum(1 for g in obs['genotype'] if g in STRAIN_ORDER and g not in FOUNDER_ORDER),
             sum(1 for g in obs['genotype'] if g.startswith('CC')))


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
