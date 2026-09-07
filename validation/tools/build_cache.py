"""Build the mosaiQTL genome cache (cache.h5) from raw GigaMUGA input.

This is the first step for any new user before running validate_tissue.py
or any scan.  It encodes the CC/DO population parameters for this dataset
(CC founders, F1 crosses, 33 CC strains) and calls the pure-Python HMM
pipeline in genotype/genomecache.py — no R required.

Expected raw inputs (under the sibling qtl/input/ tree):
  GM_ccf1fo_mm39.csv   — GigaMUGA consensus genotype matrix (133,863 markers)
  GM/gmap/             — per-chromosome genetic maps  (GM_gmap{chr}.csv)
  GM/pmap/             — per-chromosome physical maps (GM_pmap{chr}.csv)

Outputs (written to --out, default: data/ inside the repo):
  cache.h5             — GenomeCache HDF5 (read by validate_tissue.py + scans)
  kinship_decomp.npz   — LOCO kinship eigendecomposition (computed on first
                         validate_tissue.py run if absent; not written here)

data/ is gitignored so large derived files never enter the repo.

Usage
-----
  # default: build into data/ (repo root, gitignored)
  conda run -n bioinfo python validation/tools/build_cache.py

  # explicit output directory
  conda run -n bioinfo python validation/tools/build_cache.py --out /path/to/out

  # dry-run: print resolved paths and population parameters, then exit
  conda run -n bioinfo python validation/tools/build_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

log = logging.getLogger(__name__)

# ── CC population parameters ──────────────────────────────────────────────────
# Founders in canonical FOUNDER_ORDER (A→H codes map to these names).
FOUNDER_COLS: list[str] = [
    'AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ',
]

# F1 crosses present in GM_ccf1fo_mm39.csv.
# Format: [sample_id, parent1_idx, parent2_idx] — indices into FOUNDER_COLS.
# B6J (index 1) is always parent 1 (dam) in these crosses.
F1_CROSSES: list[list] = [
    ['B6AF1J',     1, 0],   # B6J × AJ
    ['B6129S1F1J', 1, 2],   # B6J × 129S1J
    ['B6NODF1J',   1, 3],   # B6J × NODJ
    ['B6NZOF1J',   1, 4],   # B6J × NZOJ
    ['B6CASTF1J',  1, 5],   # B6J × CASTJ
    ['B6PWKF1J',   1, 6],   # B6J × PWKJ
    ['B6WSBF1J',   1, 7],   # B6J × WSBJ
]

# CC strains present in GM_ccf1fo_mm39.csv (33 strains).
CC_COLS: list[str] = [
    'CC001', 'CC002', 'CC003', 'CC004', 'CC005', 'CC006', 'CC007', 'CC008',
    'CC009', 'CC010', 'CC011', 'CC012', 'CC013', 'CC015', 'CC017', 'CC018',
    'CC024', 'CC025', 'CC028', 'CC029', 'CC030', 'CC032', 'CC036', 'CC037',
    'CC038', 'CC041', 'CC043', 'CC057', 'CC060', 'CC062', 'CC065', 'CC071',
    'CC074',
]

# HMM parameters — n_gen=1 for the CC consensus map (map already encodes
# cumulative recombination history; r is used directly without scaling).
N_GEN: int = 1
ERROR_PROB: float = 0.002


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent


def _default_input_dir() -> pathlib.Path:
    """Sibling qtl/input/ tree — where GM_ccf1fo_mm39.csv and gmap/pmap live."""
    return _repo_root().parent / 'input'


def _default_out_dir() -> pathlib.Path:
    """Repo-internal data/ dir for derived files (cache.h5, kinship_decomp.npz)."""
    return _repo_root() / 'data'


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--out',
        type=pathlib.Path,
        default=None,
        help='output directory for cache.h5 (default: data/ inside the repo)',
    )
    parser.add_argument(
        '--input-dir',
        type=pathlib.Path,
        default=None,
        help='directory containing GM_ccf1fo_mm39.csv and GM/gmap + GM/pmap '
             '(default: sibling qtl/input/)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print resolved paths and population params, then exit',
    )
    parser.add_argument(
        '--verbose', action='store_true', help='enable DEBUG logging',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    input_dir = args.input_dir or _default_input_dir()
    out_dir   = args.out or _default_out_dir()

    gigamuga  = input_dir / 'GM_ccf1fo_mm39.csv'
    gmap_dir  = input_dir / 'GM' / 'gmap'
    pmap_dir  = input_dir / 'GM' / 'pmap'
    out_h5    = out_dir / 'cache.h5'

    log.info('=== build_cache ===')
    log.info('  input_dir : %s', input_dir)
    log.info('  gigamuga  : %s', gigamuga)
    log.info('  gmap_dir  : %s', gmap_dir)
    log.info('  pmap_dir  : %s', pmap_dir)
    log.info('  out_dir   : %s', out_dir)
    log.info('  out_h5    : %s', out_h5)
    log.info('  founders  : %s', FOUNDER_COLS)
    log.info('  F1 crosses: %d', len(F1_CROSSES))
    log.info('  CC strains: %d', len(CC_COLS))
    log.info('  n_gen=%d  error_prob=%.4f', N_GEN, ERROR_PROB)

    # ── validate inputs ───────────────────────────────────────────────────────
    missing = [p for p in (gigamuga, gmap_dir, pmap_dir) if not p.exists()]
    if missing:
        for p in missing:
            log.error('not found: %s', p)
        log.error(
            'Raw inputs missing — see docs/DATA_SOURCES.md for download instructions.'
        )
        sys.exit(1)

    if args.dry_run:
        log.info('--dry-run: paths OK, exiting without building.')
        return

    # ── build ─────────────────────────────────────────────────────────────────
    # Import here so the script is importable even without the package installed
    # (for --dry-run / --help at least).
    sys.path.insert(0, str(_repo_root() / 'src'))
    from mosaiqtl.genotype.genomecache import GenomeCache

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('running HMM pipeline (133k markers × 48 strains — takes ~15–30 min)...')
    cache = GenomeCache.from_gigamuga(
        gigamuga_path=str(gigamuga),
        gmap_dir=str(gmap_dir),
        pmap_dir=str(pmap_dir),
        founder_cols=FOUNDER_COLS,
        cc_cols=CC_COLS,
        f1_crosses=F1_CROSSES,
        n_gen=N_GEN,
        error_prob=ERROR_PROB,
    )

    log.info('writing %s ...', out_h5)
    cache.to_hdf5(str(out_h5))
    log.info('=== done — %s ===', out_h5)
    log.info('')
    log.info('Next step: run validate_tissue.py — it will compute and cache')
    log.info('kinship_decomp.npz in data/ on the first run.')
    log.info('If stale kinship exists: rm -f data/kinship_decomp.npz')


if __name__ == '__main__':
    main()
