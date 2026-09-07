"""Memory-efficient pseudobulking of a >200GB annotated single-cell h5ad.

Reproduces the aggregation from the original pseudobulking notebook
(``dc.pp.pseudobulk`` on a raw-counts-style layer, sample-level and
celltype-level, with ``psbulk_counts``/``psbulk_cells`` cast to int32)
without ever materialising the full count matrix in memory.

WHY THE NAIVE APPROACH DOESN'T WORK
    Two separate, confirmed problems, not one:

    (1) ``adata.layers[...]`` ACCESS on a backed AnnData always fully
    materialises into memory regardless of backed status — the source
    data lives in a layer (``denoised_counts``), not ``.X``, so
    ``dc.pp.pseudobulk(..., layer='denoised_counts')`` on a plain backed
    read still loads the entire matrix.

    (2) ``anndata.read_h5ad(path, backed='r')`` itself — BEFORE any
    ``.layers[...]`` access — is the actual OOM cause on this file.
    Confirmed by reading ``anndata._io.h5ad.read_h5ad_backed()``
    directly (anndata 0.13.2): only ``X`` gets deferred, disk-referenced
    construction. Every other top-level slot, ``obsm/varm/obsp/varp/uns
    AND layers``, is read via
    ``d.update({k: read_elem(f[k]) for k in attributes if k in f})`` —
    an unconditional EAGER, fully-recursive read of the whole group,
    independent of ``backed`` and independent of which layer the caller
    will ever touch. On this file that means all three full-size layers
    (``denoised_counts`` ~48GB, ``norm_log1p`` ~48GB, ``raw_counts``
    ~69GB — real sizes from HDF5 metadata inspection) are pulled into
    memory just to open the file, ~165GB before the script's own code
    ever runs — this is why the real SLURM run died right after
    "Opening ... in backed mode" and before the next log line, at both
    64GB and 128GB.

    Explicitly assigning the layer's dataset into ``adata.X`` looks like
    a fix but silently flips ``adata.isbacked`` to False (assignment is
    treated as detaching from the backing file), which breaks decoupler's
    internal backed-mode branch — and doesn't address (2) anyway, since
    the eager ``layers`` read already happened inside ``read_h5ad()``
    before this line could run.

APPROACH
    Never call ``anndata.read_h5ad()``/``read_h5ad_backed()`` at all —
    open the HDF5 file directly with ``h5py`` and read ONLY ``obs`` and
    ``var`` via anndata's own ``read_elem()`` (small: metadata columns,
    not count matrices). ``layers``/``obsp``/``varp``/``obsm``/``uns``
    are never touched, so anndata's eager ``read_elem(f[k])`` over those
    keys never runs. A minimal ``_LazyH5AD`` container (file handle +
    obs + var + shapes) stands in for the AnnData object everywhere this
    module needs one.

    Never touch ``adata.X``. Build a lazy backed sparse dataset directly
    from the layer's HDF5 group with anndata's own ``sparse_dataset()``
    helper. Boolean fancy-indexing on that dataset streams only the rows
    needed per (sample[, group]) partition from disk — confirmed by
    reading ``anndata._core.sparse_dataset.BackedSparseMatrix.
    subset_by_major_axis_mask()``/``get_compressed_vectors_for_slices()``
    directly: a boolean row mask is grouped into contiguous runs and each
    run is read as its own HDF5 slice, never a min-to-max byte range
    spanning unselected rows.

    Within a partition, this does NOT delegate to decoupler's private
    ``_psbulk`` — that function's per-partition line
    (``profile = bX[rows][:, msk_col]; profile.toarray()``) densifies
    every gene column for the partition in one call. For celltype-level
    output that's bounded (partition = one sample × one celltype). For
    sample-level output (``groups_col=None``) a partition is an entire
    sample's cells — on an uneven or large sample this densifies
    ``cells_in_sample × n_vars`` at once, which is the confirmed
    mechanism behind an OOM that "per-partition memory is small" doesn't
    predict. This module instead streams gene columns in chunks of
    ``gene_chunk_size`` within each partition, accumulating sums and
    nonzero-fractions incrementally — peak dense memory per partition is
    bounded by ``cells_in_partition x gene_chunk_size``, never
    ``cells_in_partition x n_vars``.

VERIFIED IDENTICAL TO A FULL IN-MEMORY CALL
    Confirmed byte-for-byte against ``dc.pp.pseudobulk()`` on real (non-zero,
    integer-count) synthetic data: X values, obs row order, psbulk_counts,
    psbulk_cells, AND gene column order all match exactly (decoupler
    2.1.6). Two version-sensitive details this depends on, both re-checked
    against decoupler source directly (not assumed):

    - ``msk_col`` passed as ``(bX, msk_col)`` to ``_psbulk`` must be a
      COLUMNS-TO-KEEP mask (``bX[rows][:, msk_col]``), all-True here for
      "keep every column" — NOT the "empty columns to drop" mask
      ``_validate_backed`` itself returns. ``extract()`` inverts one into
      the other internally (``msk_col = ~msk_col``); this script has to
      do that inversion itself since it never calls ``extract()``.
    - decoupler 2.1.6 added a ``shuffle`` parameter to ``extract()``
      (default True — a seed-0 gene-column permutation, ``_break_ties``);
      ``pseudobulk()`` internally calls ``extract(..., shuffle=False, ...)``,
      so gene column order is NEVER permuted via ``pseudobulk()`` at this
      version, backed or not — the concern below only applied to 2.1.1,
      where ``extract()`` had no ``shuffle`` control and the non-backed
      path always permuted.

    decoupler 2.1.1 (no ``shuffle`` param): the non-backed path always
    permutes gene column order via ``_break_ties``; the backed path
    (and this script) never does. Under 2.1.1 only, gene column order
    would differ from a full in-memory call (values, row order, and row
    counts still match) — original h5ad ``var_names`` order is kept here.
    ``_sample_group_by`` is a private decoupler internal reused directly
    (this module reimplements ``_psbulk``'s partition loop itself rather
    than importing ``_psbulk``/``_validate_mode`` — see APPROACH above);
    a version bump could change its signature or defaults without
    notice — see ``_check_decoupler_version()``.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys

import gc

import anndata
import decoupler
import h5py
import numpy as np
import pandas as pd
from anndata._core.sparse_dataset import sparse_dataset
from anndata._io.specs import read_elem
from decoupler.pp.anndata import _sample_group_by

log = logging.getLogger('pseudobulk_backed')


@dataclasses.dataclass
class _LazyH5AD:
    """Minimal stand-in for a backed AnnData that never reads layers/obsp/etc.

    ``anndata.read_h5ad(path, backed='r')`` eagerly reads the ENTIRE
    ``layers``/``obsp``/``varp``/``obsm``/``uns`` groups into memory as
    part of opening the file (confirmed via
    ``anndata._io.h5ad.read_h5ad_backed()`` source — see module
    docstring); only ``X`` gets real backed/lazy treatment. This class
    is built instead, reading only ``obs``/``var`` (small metadata
    tables), and exposes just what :func:`pseudobulk_backed` needs.

    Attributes:
        file: Open ``h5py.File`` handle (read-only) — layer datasets are
            read lazily from this via ``sparse_dataset()``.
        obs: Full obs DataFrame.
        var: Full var DataFrame.
        n_obs: Number of observations (rows).
        n_vars: Number of variables (genes/columns).
        isbacked: Always True — mirrors the AnnData attribute this
            module's functions check.
    """

    file: h5py.File
    obs: pd.DataFrame
    var: pd.DataFrame
    n_obs: int
    n_vars: int
    isbacked: bool = True


def _open_lazy(path: pathlib.Path) -> _LazyH5AD:
    """Open an h5ad reading only ``obs``/``var`` — never ``layers``/``obsp``/etc.

    Args:
        path: Path to the h5ad file.

    Returns:
        A :class:`_LazyH5AD` wrapping the open file handle plus obs/var.
    """
    f = h5py.File(path, 'r')
    obs = read_elem(f['obs'])
    var = read_elem(f['var'])
    return _LazyH5AD(file=f, obs=obs, var=var, n_obs=obs.shape[0], n_vars=var.shape[0])


_VERIFIED_DECOUPLER_VERSIONS = ('2.1.1', '2.1.6')


def _check_decoupler_version() -> None:
    """Hard-fail if the installed decoupler wasn't verified against this script.

    ``_sample_group_by``/``_psbulk``/``_validate_mode`` are private
    decoupler internals reused directly (not part of decoupler's public
    API), and the correctness of the manually-built ``(bX, msk_col)``
    tuple depends on ``pseudobulk()``'s ``empty=False`` default and
    ``_validate_backed``'s empty-column gating staying exactly as
    verified. A version drift could change either silently — raise
    rather than warn, since a silent mismatch would produce output
    that looks fine but isn't guaranteed correct.

    Raises:
        RuntimeError: If the installed decoupler version was not
            explicitly checked against this script's source-level
            assumptions (see module docstring).
    """
    if decoupler.__version__ not in _VERIFIED_DECOUPLER_VERSIONS:
        raise RuntimeError(
            f'decoupler {decoupler.__version__} installed; this script was verified only against '
            f'{", ".join(_VERIFIED_DECOUPLER_VERSIONS)}. Private function _sample_group_by, and the '
            'partition/new_obs semantics this script reimplements from decoupler\'s _psbulk, may have '
            'changed. Re-verify against the new version\'s source before adding it here.'
        )


def pseudobulk_backed(
    adata: _LazyH5AD,
    sample_col: str,
    groups_col: str | None,
    layer: str,
    gene_chunk_size: int = 2000,
) -> anndata.AnnData:
    """Aggregate a lazily-opened h5ad into pseudobulk profiles without full load.

    Mirrors ``decoupler.pp.pseudobulk(adata, sample_col, groups_col, layer,
    mode='sum')`` byte-for-byte — X values, obs row order/index,
    ``psbulk_counts``/``psbulk_cells``, and gene column order all match
    under decoupler 2.1.6 (see module docstring for the one caveat under
    2.1.1: gene column order only). Reimplements decoupler's private
    ``_psbulk`` partition loop directly (rather than calling it) so each
    partition's gene columns can be streamed in chunks — see module
    docstring for why that matters for the ``groups_col=None`` case.

    Args:
        adata: Lazily-opened h5ad (:func:`_open_lazy`) — never a
            ``anndata.read_h5ad(..., backed='r')`` result, which eagerly
            reads every layer into memory at open time (see module
            docstring).
        sample_col: obs column identifying the sample.
        groups_col: obs column identifying the group (e.g. celltype);
            None aggregates to one profile per sample.
        layer: Name of the layer in ``adata.file['layers']`` to sum.
        gene_chunk_size: Number of gene columns densified at once per
            partition. Peak memory per partition is
            ``cells_in_partition x gene_chunk_size x 8 bytes``; lower
            this if a single sample's cell count is very large.

    Returns:
        AnnData of pseudobulk profiles, one row per (sample[, group]).
    """
    if not adata.isbacked:
        raise ValueError('adata must be a _LazyH5AD from _open_lazy(); got a non-backed object')

    n_vars = adata.n_vars
    bX = sparse_dataset(adata.file['layers'][layer])
    obs, resolved_groups_col, smples, groups, n_rows = _sample_group_by(
        sample=sample_col, group=groups_col, obs=adata.obs.copy(), verbose=True
    )
    new_obs = pd.DataFrame(columns=obs.columns)
    psbulk = np.zeros((n_rows, n_vars))
    props = np.zeros((n_rows, n_vars))
    ncells = np.zeros(n_rows)
    counts = np.zeros(n_rows)

    if resolved_groups_col is None:
        partitions = [(smp, None) for smp in smples]
    else:
        partitions = [(smp, grp) for grp in groups for smp in smples]

    for i, (smp, grp) in enumerate(partitions):
        if grp is None:
            row_mask = (obs[sample_col] == smp).values
            tmp = obs[obs[sample_col] == smp].drop_duplicates().values
            index = smp
        else:
            row_mask = ((obs[sample_col] == smp) & (obs[resolved_groups_col] == grp)).values
            tmp = obs[(obs[sample_col] == smp) & (obs[resolved_groups_col] == grp)].drop_duplicates().values
            if tmp.shape[0] == 0:
                tmp = obs[obs[sample_col] == smp].drop(columns=resolved_groups_col).drop_duplicates()
                tmp = tmp.head(1)
                tmp[resolved_groups_col] = grp
                tmp = tmp[obs.columns].values
            index = f'{smp}_{grp}'
        new_obs.loc[index, :] = tmp

        ncell = int(row_mask.sum())
        ncells[i] = ncell
        if ncell == 0:
            log.info('%s cells=0 counts=0', index)
            continue

        # One backed disk read for every gene column of this partition's
        # rows (cheap: subset_by_major_axis_mask reads only the byte
        # ranges for the selected rows — see module docstring). Still
        # sparse at this point; densification happens per gene-chunk below.
        partition_rows = bX[row_mask, :]
        for start in range(0, n_vars, gene_chunk_size):
            stop = min(start + gene_chunk_size, n_vars)
            chunk = partition_rows[:, start:stop]
            if not isinstance(chunk, np.ndarray):
                chunk = chunk.toarray()
            props[i, start:stop] = chunk.astype(bool).mean(axis=0)
            psbulk[i, start:stop] = chunk.sum(axis=0)
            del chunk
        del partition_rows
        gc.collect()

        count = float(psbulk[i].sum())
        counts[i] = count
        log.info('%s cells=%d counts=%d', index, ncell, count)

    new_obs['psbulk_cells'] = ncells
    new_obs['psbulk_counts'] = counts
    for col in new_obs.columns:
        if not pd.api.types.is_numeric_dtype(new_obs[col]):
            new_obs[col] = new_obs[col].astype('category')

    new_obs['psbulk_counts'] = new_obs['psbulk_counts'].astype('int32')
    new_obs['psbulk_cells'] = new_obs['psbulk_cells'].astype('int32')

    return anndata.AnnData(X=psbulk, obs=new_obs, var=adata.var.copy(), layers={'psbulk_props': props})


def main() -> None:
    """CLI entry point: run both celltype-level and sample-level pseudobulking."""
    parser = argparse.ArgumentParser(
        description='Memory-efficient pseudobulking of a large annotated h5ad via decoupler backed mode.'
    )
    parser.add_argument('--h5ad', required=True, type=pathlib.Path, help='Path to the input annotated h5ad.')
    parser.add_argument(
        '--out', required=True, type=pathlib.Path,
        help='Output directory (tissue data root, e.g. validation/ctxhipp_data). '
        'If --dataset is also given, output is written to {out}/{dataset}.',
    )
    parser.add_argument(
        '--dataset', default=None, choices=['next', 'nova'],
        help='Dataset generation subdirectory appended under --out (default: none — '
        'writes directly under --out).',
    )
    parser.add_argument('--tissue', required=True, help='Tissue prefix for output filenames, e.g. ctxhipp.')
    parser.add_argument('--sample-col', default='sample_id', help='obs column identifying the sample.')
    parser.add_argument('--groups-col', default='celltype', help='obs column identifying the celltype group.')
    parser.add_argument('--layer', default='denoised_counts', help='Layer to sum.')
    parser.add_argument(
        '--gene-chunk-size',
        type=int,
        default=2000,
        help='Gene columns densified at once per partition. Lower this if a single '
        'sample/group has a very large cell count (peak mem per partition is '
        'cells_in_partition x gene_chunk_size x 8 bytes).',
    )
    parser.add_argument(
        '--skip-tissue-level',
        action='store_true',
        help='Skip the sample-level (groups_col=None) output.',
    )
    parser.add_argument(
        '--skip-celltype-level',
        action='store_true',
        help='Skip the celltype-level (groups_col=<--groups-col>) output.',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='DEBUG-level logging.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stdout,
    )
    _check_decoupler_version()

    out_dir = args.out / args.dataset if args.dataset else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    anndata.settings.allow_write_nullable_strings = True

    # NOT anndata.read_h5ad(..., backed='r') — that eagerly reads every
    # layer (obsp/varp/obsm/uns too) into memory at open time regardless
    # of backed mode; see module docstring. _open_lazy reads only obs/var.
    log.info('Opening %s (lazy: obs/var only, layers never read)', args.h5ad)
    adata = _open_lazy(args.h5ad)
    log.info('%d obs x %d vars; layer=%s', adata.n_obs, adata.n_vars, args.layer)

    try:
        if not args.skip_celltype_level:
            log.info(
                'Celltype-level pseudobulk: sample_col=%s groups_col=%s', args.sample_col, args.groups_col
            )
            pdata = pseudobulk_backed(
                adata, args.sample_col, args.groups_col, args.layer, gene_chunk_size=args.gene_chunk_size
            )
            out_path = out_dir / f'{args.tissue}_all_ctpb.h5ad'
            pdata.write_h5ad(out_path)
            log.info('Wrote %s (%d obs x %d vars)', out_path, pdata.n_obs, pdata.n_vars)

        if not args.skip_tissue_level:
            log.info('Sample-level pseudobulk: sample_col=%s groups_col=None', args.sample_col)
            pdata1 = pseudobulk_backed(
                adata, args.sample_col, None, args.layer, gene_chunk_size=args.gene_chunk_size
            )
            out_path = out_dir / f'{args.tissue}_all_tisspb.h5ad'
            pdata1.write_h5ad(out_path)
            log.info('Wrote %s (%d obs x %d vars)', out_path, pdata1.n_obs, pdata1.n_vars)
    finally:
        adata.file.close()


if __name__ == '__main__':
    main()
