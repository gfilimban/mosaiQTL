"""Rebuild per-chromosome GM_gmap{chr}.csv / GM_pmap{chr}.csv from gm_uwisc_v4.csv.

Source: gm_uwisc_v4.csv only (Layer-0, cited in docs/DATA_SOURCES.md) — same marker
filter as build_gm_ccf1fo.py (unique==TRUE, unmapped==FALSE, valid bp_grcm39,
chr in 1-19/X). Deliberately NOT copied from qtl/input/old/GM/{gmap,pmap}/ — those
files are legacy and out of scope for the Layer-0-only rebuild.

Written to qtl/input/GM/{gmap,pmap} (canonical), matching build_gm_ccf1fo.py's
convention of writing only to qtl/input/. Mirror to qtl/miqtl/input/GM/ separately
if build_genomecache.R needs it there (same pattern used for GM_ccf1fo_mm39.csv).

Output consumed by build_genomecache.R as gmap_dir/pmap_dir (read_map(), pattern
'^GM_(g|p)map[0-9XYM]+\\.csv$'): gmap uses cM_cox (genetic map), pmap uses
bp_grcm39 / 1e6 (physical map in Mb).
"""
import sys

import pandas as pd

IN_DIR = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/input'
OUT_BASE = f'{IN_DIR}/GM'


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    log('loading gm_uwisc_v4.csv...')
    gm = pd.read_csv(f'{IN_DIR}/gm_uwisc_v4.csv', dtype=str)
    valid_chr = [str(i) for i in range(1, 20)] + ['X']
    unique_b = gm['unique'].str.upper() == 'TRUE'
    unmapped_b = gm['unmapped'].str.upper() == 'TRUE'
    gm_f = gm[unique_b & (~unmapped_b) & gm['bp_grcm39'].notna()
              & (gm['bp_grcm39'] != '') & gm['chr'].isin(valid_chr)].copy()
    gm_f['cM_cox'] = gm_f['cM_cox'].astype(float)
    gm_f['bp_grcm39'] = gm_f['bp_grcm39'].astype(float)
    log(f'{len(gm_f)} markers pass filter')

    for chrom in valid_chr:
        sub = gm_f[gm_f['chr'] == chrom]
        if sub.empty:
            continue
        gmap = sub[['marker', 'chr']].copy()
        gmap['pos'] = sub['cM_cox']
        pmap = sub[['marker', 'chr']].copy()
        pmap['pos'] = sub['bp_grcm39'] / 1e6

        gmap_path = f'{OUT_BASE}/gmap/GM_gmap{chrom}.csv'
        pmap_path = f'{OUT_BASE}/pmap/GM_pmap{chrom}.csv'
        with open(gmap_path, 'w') as fh:
            fh.write(f'# Genetic map for GigaMUGA chromosome {chrom} (build GRCm39)\n')
            fh.write(f'# nrow {len(gmap)}\n# ncol 3\n')
            gmap.to_csv(fh, index=False)
        with open(pmap_path, 'w') as fh:
            fh.write(f'# Physical map for GigaMUGA chromosome {chrom} (build GRCm39)\n')
            fh.write(f'# nrow {len(pmap)}\n# ncol 3\n')
            pmap.to_csv(fh, index=False)
        log(f'  chr{chrom}: {len(sub)} markers -> {gmap_path}, {pmap_path}')

    log('DONE')


if __name__ == '__main__':
    main()
