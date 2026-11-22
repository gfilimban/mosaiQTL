"""Build igvf_cache.h5 from GM_ccf1fo_mm39.csv and the per-chromosome maps in ref/GM/.

Usage: python build_cache.py <GM_ccf1fo_mm39.csv> <output_dir>

Inputs, resolved next to the CSV:
  GM/gmap/GM_gmap{chr}.csv, GM/pmap/GM_pmap{chr}.csv   per-chromosome genetic and physical maps
  GM/cc_cols.json, GM/founder_cols.json                strain column lists
  GM/f1_crosses.json                                   [[id, parent1_idx, parent2_idx], ...]
  cc_crossinfo.csv                                     funnel position of each founder per CC strain (X prior)

The build fails if the cache holds fewer markers than the CSV has rows or a different number of
strains than the three column lists define, so a marker or strain is never dropped silently.
"""
import json
import logging
import sys
from pathlib import Path

import h5py
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ERROR_PROB = 0.002


def main() -> None:
    from mosaiqtl.genotype.genomecache import GenomeCache

    gigamuga_csv = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]) / 'igvf_cache.h5'
    if out_path.exists():
        logger.info('cache already exists, skipping: %s', out_path)
        return

    gm_dir = gigamuga_csv.parent / 'GM'
    crossinfo_csv = gigamuga_csv.parent / 'cc_crossinfo.csv'
    required = [gigamuga_csv, crossinfo_csv, gm_dir / 'gmap', gm_dir / 'pmap'] + [
        gm_dir / name for name in ('cc_cols.json', 'founder_cols.json', 'f1_crosses.json')
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f'required input not found: {path}')

    cc_cols = json.loads((gm_dir / 'cc_cols.json').read_text())
    founder_cols = json.loads((gm_dir / 'founder_cols.json').read_text())
    f1_crosses = json.loads((gm_dir / 'f1_crosses.json').read_text())
    n_markers = len(pd.read_csv(gigamuga_csv, usecols=['marker']))
    n_strains = len(founder_cols) + len(f1_crosses) + len(cc_cols)
    logger.info('building cache: %d markers, %d strains', n_markers, n_strains)

    cache = GenomeCache.from_gigamuga(
        gigamuga_path=str(gigamuga_csv),
        gmap_dir=str(gm_dir / 'gmap'),
        pmap_dir=str(gm_dir / 'pmap'),
        cc_cols=cc_cols,
        founder_cols=founder_cols,
        f1_crosses=f1_crosses,
        error_prob=ERROR_PROB,
        crossinfo_path=str(crossinfo_csv),
        inbred=True,
        step_cM=None,
    )
    cache.to_hdf5(str(out_path))

    with h5py.File(out_path, 'r') as fh:
        groups = [fh[key] for key in fh if key.startswith('chr')]
        cached_markers = sum(grp['locus_ids'].shape[0] for grp in groups)
        cached_strains = groups[0]['sample_ids'].shape[0]
    if cached_markers != n_markers or cached_strains != n_strains:
        out_path.unlink()
        raise ValueError(
            f'cache has {cached_markers} markers and {cached_strains} strains; '
            f'expected {n_markers} and {n_strains}'
        )
    logger.info('wrote %s', out_path)


if __name__ == '__main__':
    main()
