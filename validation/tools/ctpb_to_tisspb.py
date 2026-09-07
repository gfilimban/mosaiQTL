"""Derive a sample-level (tisspb) pseudobulk h5ad from an existing celltype-level (ctpb) one.

Summation is associative: ``ctpb`` already holds one (sample, celltype) row per
group, summed from raw counts. Grouping those rows by ``sample_id`` alone and
summing again reproduces exactly what ``pseudobulk_backed(adata, sample_col,
groups_col=None, ...)`` would have produced directly from the raw h5ad —
verified byte-for-byte against ``next``'s existing ctpb/tisspb pair (per-sample
``psbulk_cells`` and summed ``X`` rows match with zero difference). This lets a
generation missing its own tisspb (e.g. ``nova``) get one without re-reading
the >200GB raw h5ad.

Reads ``X`` directly from the HDF5 file via row-index slicing rather than
``anndata.read_h5ad(..., backed='r')`` — backed mode still eagerly loads every
``layers`` entry (``psbulk_props`` here, same shape as ``X``) regardless of
which layer is used; see ``pseudobulk_backed.py`` module docstring for the
confirmed root cause. Only one sample's rows are ever resident at once.
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import anndata
import h5py
import numpy as np
import pandas as pd
from anndata._io.specs import read_elem

log = logging.getLogger('ctpb_to_tisspb')

# obs columns that are meaningless at the tissue level — every sample was
# pseudobulked across all its celltypes, so no single celltype label applies.
_DROP_COLS = ('celltype', 'general_celltype')


def ctpb_to_tisspb(ctpb_path: pathlib.Path) -> anndata.AnnData:
    """Aggregate a celltype-level pseudobulk h5ad into a sample-level one.

    Args:
        ctpb_path: Path to a ``*_ctpb.h5ad`` file (one row per sample x
            celltype, ``X`` holding summed raw counts per decoupler's
            ``mode='sum'`` pseudobulk convention).

    Returns:
        AnnData with one row per sample, ``X`` summed across that sample's
        celltype rows, and ``psbulk_cells``/``psbulk_counts`` re-summed to
        match — structurally identical to calling
        ``pseudobulk_backed(raw_adata, sample_col, groups_col=None, ...)``
        directly on the pre-aggregation raw h5ad.
    """
    with h5py.File(ctpb_path, 'r') as f:
        obs = read_elem(f['obs'])
        var = read_elem(f['var'])
        x_ds = f['X']
        n_vars = x_ds.shape[1]

        if 'sample_id' not in obs.columns:
            raise ValueError(f"'sample_id' column missing from {ctpb_path} obs")

        sample_ids = obs['sample_id'].astype(str)
        samples = sample_ids.drop_duplicates().tolist()
        log.info('%d ctpb rows -> %d samples', obs.shape[0], len(samples))

        tiss_cols = [c for c in obs.columns if c not in _DROP_COLS]
        x_out = np.zeros((len(samples), n_vars), dtype=np.float64)
        records = []

        for i, smp in enumerate(samples):
            row_idx = np.flatnonzero((sample_ids == smp).to_numpy())
            x_out[i, :] = x_ds[row_idx, :].sum(axis=0)

            rep = obs.iloc[row_idx[0]]
            record = {c: rep[c] for c in tiss_cols}
            record['psbulk_cells'] = int(obs['psbulk_cells'].iloc[row_idx].sum())
            record['psbulk_counts'] = int(obs['psbulk_counts'].iloc[row_idx].sum())
            records.append(record)

            if (i + 1) % 100 == 0 or i + 1 == len(samples):
                log.info('aggregated %d/%d samples', i + 1, len(samples))

    # unnamed index: 'sample_id' stays a regular column (matches the existing
    # tisspb convention) — naming the index 'sample_id' too makes anndata's
    # writer reject it as a duplicate-but-divergent key. Building from
    # records (rather than an empty object-dtype frame filled via .at[])
    # lets pandas infer each column's real dtype instead of leaving every
    # column as 'object', which anndata's HDF5 writer cannot serialise.
    new_obs = pd.DataFrame.from_records(records, index=pd.Index(samples))
    new_obs['psbulk_cells'] = new_obs['psbulk_cells'].astype('int32')
    new_obs['psbulk_counts'] = new_obs['psbulk_counts'].astype('int32')

    return anndata.AnnData(X=x_out, obs=new_obs, var=var)


def main() -> None:
    """CLI entry point: derive ``*_all_tisspb.h5ad`` next to an existing ``*_all_ctpb.h5ad``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctpb', required=True, type=pathlib.Path, help='Path to the *_ctpb.h5ad input.')
    parser.add_argument(
        '--out',
        type=pathlib.Path,
        default=None,
        help='Output path. Default: same dir as --ctpb, "_ctpb" replaced with "_tisspb".',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='DEBUG-level logging.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    out_path = args.out or args.ctpb.with_name(args.ctpb.name.replace('_ctpb', '_tisspb'))
    if out_path == args.ctpb:
        raise ValueError(f"could not derive an output name from {args.ctpb} — pass --out explicitly")

    tisspb = ctpb_to_tisspb(args.ctpb)
    log.info('writing %s (%d obs x %d vars)', out_path, tisspb.n_obs, tisspb.n_vars)
    tisspb.write_h5ad(out_path)


if __name__ == '__main__':
    main()
