"""cache_builder — build a GenomeCache HDF5 from source genotype files.

Setup-time utilities: write haplotype probabilities and kinship to HDF5,
collapse redundant loci, and drive the ``build-cache`` CLI entry point.
None of these functions run at scan time; they are called once during
data preparation.
"""

import os
import logging
import numpy as np

from .genomecache import GenomeCache

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'collapse_genomecache',
    'build',
]


def _save_group_probs(cache, sample_list, filepath):
    """Save one group's haplotype probabilities (e.g. just the CC strains) to HDF5.

    The axis order and file layout match the standard .rds output format so
    results can be compared directly against an independently computed
    reference cache.

    HDF5 layout
    -----------
    /founders           str array  8 founder names
    /sample_ids         str array  sample names in this group
    /chr{c}/probs       float32    (n_strains, n_founders, n_loci)
    /chr{c}/locus_ids   str array  marker names
    /chr{c}/positions   float64    physical positions (Mb)

    Parameters
    ----------
    cache       : GenomeCache  full combined cache (all strains)
    sample_list : list of str  strain names to extract into this group
    filepath    : str          output .h5 path
    """
    import h5py

    sid_to_idx = {sid: i for i, sid in enumerate(cache.sample_ids)}
    grp_idx = [sid_to_idx[s] for s in sample_list if s in sid_to_idx]
    grp_sids = [s for s in sample_list if s in sid_to_idx]

    if not grp_idx:
        logger.warning(f'_save_group_probs: no samples found in cache for {filepath}')
        return

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('founders', data=np.array(cache.founders, dtype='S'))
        f.create_dataset('sample_ids', data=np.array(grp_sids, dtype='S'))

        for chrom in cache.chromosomes:
            lids, pos, probs = cache.get_chromosome(chrom)
            # keep only this group's samples, then reorder axes to
            # (n_grp, n_founders, n_loci) for the output file
            grp_probs = probs[:, grp_idx, :]
            grp_probs = np.transpose(grp_probs, (1, 2, 0)).astype(np.float32)

            grp = f.create_group(f'chr{chrom}')
            grp.create_dataset('locus_ids', data=np.array(lids, dtype='S'))
            grp.create_dataset('positions', data=pos)
            grp.create_dataset(
                'probs', data=grp_probs, compression='gzip', compression_opts=4
            )

    n_loci_total = sum(cache.n_loci(c) for c in cache.chromosomes)
    logger.info(
        f'saved {len(grp_sids)} samples × {len(cache.chromosomes)} chrs '
        f'× {n_loci_total} loci → {filepath}'
    )


def _save_kinship(kinship_dict, sample_ids, filepath):
    """Save LOCO kinship matrices to HDF5.

    HDF5 layout
    -----------
    /sample_ids   str array   sample names
    /chr{c}       float64     (n_strains, n_strains) LOCO kinship matrix

    Parameters
    ----------
    kinship_dict : dict  {chrom: K}  from kinship.calc_kinship(method='loco')
    sample_ids   : array-like of str
    filepath     : str  output .h5 path
    """
    import h5py

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        for key, K in kinship_dict.items():
            f.create_dataset(
                f'chr{key}',
                data=K.astype(np.float64),
                compression='gzip',
                compression_opts=4,
            )

    logger.info(f'saved LOCO kinship ({len(kinship_dict)} chrs) → {filepath}')


def collapse_genomecache(
    cache: 'GenomeCache',
    criterion: str = 'l2_norm',
    tol: float = 0.02,
) -> 'GenomeCache':
    """Reduce a genome cache by removing redundant loci.

    Greedy forward scan per chromosome: keep a locus if its normalized
    L2 distance from the previous kept locus exceeds ``tol``.  The first
    locus on each chromosome is always kept.

    Parameters
    ----------
    cache     : GenomeCache
    criterion : str  only 'l2_norm' supported
    tol       : float
        minimum normalized L2 distance to keep a locus.
        normalized by sqrt(n_strains × n_founders) so that tol=0.02 is
        meaningful regardless of cache dimensions.

    Returns
    -------
    GenomeCache  with fewer loci per chromosome; same sample_ids and founders
    """
    if criterion != 'l2_norm':
        raise ValueError(f'unsupported criterion: {criterion!r}')

    new_data: dict = {}
    for chrom in cache.chromosomes:
        lids, pos, probs = cache.get_chromosome(chrom)
        n_loci, n_strains, n_founders = probs.shape
        norm_factor = np.sqrt(n_strains * n_founders)

        keep = [0]
        ref = probs[0].ravel()
        for i in range(1, n_loci):
            if np.linalg.norm(probs[i].ravel() - ref) / norm_factor >= tol:
                keep.append(i)
                ref = probs[i].ravel()

        idx = np.array(keep)
        new_data[chrom] = {
            'locus_ids': lids[idx],
            'positions': pos[idx],
            'probs': probs[idx],
        }
        logger.info(
            f'collapse {chrom}: {n_loci} -> {len(idx)} loci '
            f'({100*(1-len(idx)/n_loci):.1f}% removed)'
        )

    return GenomeCache(
        sample_ids=cache.sample_ids,
        founders=cache.founders,
        data=new_data,
    )


def build(args=None):
    """CLI entry point: build genome cache from raw GigaMUGA data.

    Writes five output files to --out-dir:

        genomecache.h5      full combined cache (all strains, all chrs)
        probs_cc.h5         inbred strain haplotype probs
        probs_founders.h5   founder haplotype probs
        probs_f1.h5         F1 haplotype probs
        kinship_loco.h5     LOCO kinship matrices (n_strains×n_strains per chr)

    Column lists (--cc-cols, --founder-cols) can be either a comma-separated
    string or a path to a JSON file containing a list of strings.

    F1 crosses (--f1-crosses) must be a JSON file: list of
    [sample_id, parent1_idx, parent2_idx] where indices are 0-based
    positions into --founder-cols.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog='build-cache',
        description='Build a genome cache HDF5 from source genotype data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--gigamuga',
        required=True,
        help='GigaMUGA consensus matrix CSV (markers × strains)',
    )
    parser.add_argument(
        '--gmap-dir',
        required=True,
        help='directory containing gmap.csv (marker, chr, pos_cM)',
    )
    parser.add_argument(
        '--pmap-dir',
        required=True,
        help='directory containing pmap.csv (marker, chr, pos_mb)',
    )
    parser.add_argument(
        '--cc-cols',
        required=True,
        help='comma-sep CC sample names, or path to JSON list',
    )
    parser.add_argument(
        '--founder-cols',
        required=True,
        help='comma-sep founder names in canonical order, or JSON',
    )
    parser.add_argument(
        '--f1-crosses',
        default=None,
        help='JSON file: [[sample_id, p1_idx, p2_idx], ...]',
    )
    parser.add_argument(
        '--crossinfo',
        default=None,
        help='cc_crossinfo.csv (funnel positions per CC strain)',
    )
    parser.add_argument(
        '--out-dir', default='validation/phase2/py_data', help='output directory'
    )
    parser.add_argument(
        '--step-cM',
        type=float,
        default=0.2,
        help='pseudomarker spacing in cM (default 0.2); pass 0 to disable',
    )
    parser.add_argument(
        '--error-prob',
        type=float,
        default=0.002,
        help='GigaMUGA genotyping error probability',
    )
    parser.add_argument(
        '--n-gen',
        type=int,
        default=1,
        help='HMM transition scaling (1 for consensus CC map)',
    )
    parser.add_argument(
        '--collapse', action='store_true', help='collapse redundant loci before saving'
    )
    parser.add_argument('--verbose', action='store_true', help='enable DEBUG logging')

    args = parser.parse_args(args)

    import logging as _logging

    _logging.basicConfig(
        level=_logging.DEBUG if args.verbose else _logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    def _parse_list(val):
        """Parse a comma-separated string or a JSON file path into a list."""
        if val.endswith('.json') and os.path.exists(val):
            with open(val) as fh:
                return json.load(fh)
        return [x.strip() for x in val.split(',')]

    cc_cols = _parse_list(args.cc_cols)
    founder_cols = _parse_list(args.founder_cols)
    f1_crosses = []
    if args.f1_crosses:
        with open(args.f1_crosses) as fh:
            f1_crosses = json.load(fh)  # [[sample_id, pi1, pi2], ...]

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info('=== build-cache ===')
    logger.info(f'  gigamuga : {args.gigamuga}')
    logger.info(f'  gmap_dir : {args.gmap_dir}')
    logger.info(f'  pmap_dir : {args.pmap_dir}')
    logger.info(
        f'  CC strains ({len(cc_cols)}): {cc_cols[:4]}{"..." if len(cc_cols) > 4 else ""}'
    )
    logger.info(f'  founders  ({len(founder_cols)}): {founder_cols}')
    logger.info(f'  F1 crosses: {len(f1_crosses)}')
    logger.info(f'  out_dir  : {args.out_dir}')

    step_cM = args.step_cM if args.step_cM and args.step_cM > 0 else None
    cache = GenomeCache.from_gigamuga(
        gigamuga_path=args.gigamuga,
        gmap_dir=args.gmap_dir,
        pmap_dir=args.pmap_dir,
        cc_cols=cc_cols,
        founder_cols=founder_cols,
        f1_crosses=f1_crosses,
        error_prob=args.error_prob,
        n_gen=args.n_gen,
        crossinfo_path=args.crossinfo,
        step_cM=step_cM,
    )

    if args.collapse:
        logger.info('collapsing redundant loci...')
        cache = cache.collapse()

    # ── save full combined cache ───────────────────────────────────────────────
    full_path = os.path.join(args.out_dir, 'genomecache.h5')
    cache.to_hdf5(full_path)

    # ── save per-group probs ───────────────────────────────────────────────────
    _save_group_probs(cache, cc_cols, os.path.join(args.out_dir, 'probs_cc.h5'))
    _save_group_probs(
        cache, founder_cols, os.path.join(args.out_dir, 'probs_founders.h5')
    )
    if f1_crosses:
        f1_ids = [row[0] for row in f1_crosses]
        _save_group_probs(cache, f1_ids, os.path.join(args.out_dir, 'probs_f1.h5'))

    # ── compute and save LOCO kinship ─────────────────────────────────────────
    from .kinship import calc_kinship

    logger.info('computing LOCO kinship...')
    kinship_loco = calc_kinship(cache, method='loco')
    _save_kinship(
        kinship_loco, cache.sample_ids, os.path.join(args.out_dir, 'kinship_loco.h5')
    )

    logger.info('=== build complete ===')
    logger.info(f'  outputs in {args.out_dir}:')
    for fname in (
        'genomecache.h5',
        'probs_cc.h5',
        'probs_founders.h5',
        'probs_f1.h5',
        'kinship_loco.h5',
    ):
        fp = os.path.join(args.out_dir, fname)
        if os.path.exists(fp):
            size_mb = os.path.getsize(fp) / 1e6
            logger.info(f'    {fname:<22s} {size_mb:6.1f} MB')


if __name__ == '__main__':
    build()
