"""plot_ctxhipp_eqtl_summary.py — detected eQTLs vs genes tested per ctxhipp cell type.

Reads, per cell type under validation/ctxhipp_data/nova/celltypes/:
    {ct}/counts_lognorm.h5        genes tested
    {ct}/Lw1/eqtl_summary.csv     detected eQTLs and peak LOD

Writes validation/ctxhipp_data/nova/compare/ctxhipp_eqtl_summary.png (transparent)
and ctxhipp_eqtl_summary_share.png (white).

Usage
-----
    conda run -n bioinfo python validation/tools/plot_ctxhipp_eqtl_summary.py
"""
from __future__ import annotations

import logging
import pathlib
import sys

import h5py
import pandas as pd
from natsort import natsorted

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / 'src'))
from mosaiqtl.viz.celltypes import ct_eqtl_summary

logger = logging.getLogger('mosaiqtl')

_ROOT = pathlib.Path(__file__).parents[2] / 'validation' / 'ctxhipp_data' / 'nova'
_SCAN_TAG = 'Lw1'


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cts_dir = _ROOT / 'celltypes'
    rows: list[dict] = []
    summaries: list[pd.DataFrame] = []
    for ct_dir in natsorted(p for p in cts_dir.iterdir() if p.is_dir() and p.name != 'compare'):
        summary = ct_dir / _SCAN_TAG / 'eqtl_summary.csv'
        h5 = ct_dir / 'counts_lognorm.h5'
        if not (summary.exists() and h5.exists()):
            logger.warning('skipping %s: missing summary or counts', ct_dir.name)
            continue
        df = pd.read_csv(summary)
        with h5py.File(h5, 'r') as f:
            n_tested = int(f['genes'].shape[0])
        rows.append({'celltype': ct_dir.name, 'n_tested': n_tested, 'n_eqtl': len(df)})
        summaries.append(df)
    all_eqtl = pd.concat(summaries, ignore_index=True)
    ct_eqtl_summary(
        counts=pd.DataFrame(rows),
        peak_lods=all_eqtl['peak_lod'].to_numpy(dtype=float),
        out_path=_ROOT / 'compare' / 'ctxhipp_eqtl_summary.png',
        n_distinct_egenes=int(all_eqtl['gene_id'].nunique()),
    )
    logger.info('%d cell types, %d eQTLs', len(rows), len(all_eqtl))


if __name__ == '__main__':
    main()
