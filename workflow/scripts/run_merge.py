"""Merge per-chr LOD CSVs → genome-wide tables — called by Nextflow MERGE_SCAN process.

Usage: python run_merge.py <lod_csvs_dir> <gene_info.csv> <celltype>
           <scan_tag> <output_dir> [tissue]

Writes: {tag}_lod_genome.csv, {tag}_peak_lod.csv, {tag}_scan_stats.csv
  where tag = {tissue}_{celltype}_{scan_tag}  (tissue omitted when empty)
  Note: {tag}_peak_lod.csv includes beta_{founder} columns from the LMM scan
  so run_finemap.py can enrich egenes_df without a separate allele_effects file.
"""
import sys, json, logging, time
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    lod_dir        = Path(sys.argv[1])
    gene_info_path = Path(sys.argv[2])
    celltype       = sys.argv[3]
    stag           = sys.argv[4]
    out            = Path(sys.argv[5])
    tissue         = sys.argv[6] if len(sys.argv) > 6 else ''

    tag  = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    from mosaiqtl.pipeline import PEAK_LOD_COLS
    from mosaiqtl.utils import chr_sort_key

    gi = pd.read_csv(gene_info_path)

    # ── lod_chr*.csv → {tag}_lod_genome.csv ──────────────────────────────────
    lod_files = sorted(lod_dir.glob('lod_chr*.csv'),
                       key=lambda p: chr_sort_key(p.stem.replace('lod_chr', '')))
    if not lod_files:
        raise FileNotFoundError(f'no lod_chr*.csv in {lod_dir}')

    chrs = [p.stem.replace('lod_chr', '') for p in lod_files]
    log.info('START merge: tag=%s  %d chrs: %s', tag, len(chrs), ' '.join(chrs))
    lod_genome = pd.concat([pd.read_csv(f) for f in lod_files], ignore_index=True)
    lod_genome['chr'] = lod_genome['chr'].astype(str)
    lod_genome.to_csv(out / f'{tag}_lod_genome.csv', index=False)
    log.info('%s_lod_genome.csv: %d loci', tag, len(lod_genome))

    # ── peak_stats_chr*.csv → {tag}_peak_lod.csv ─────────────────────────────
    peak_files = sorted(lod_dir.glob('peak_stats_chr*.csv'),
                        key=lambda p: chr_sort_key(p.stem.replace('peak_stats_chr', '')))
    peak_lod = pd.concat(
        [pd.read_csv(f) for f in peak_files if f.stat().st_size > 0],
        ignore_index=True,
    )
    peak_lod['chr'] = peak_lod['chr'].astype(str)

    gi_cols = ['gene_id']
    if 'gene_symbol' in gi.columns and 'gene_symbol' not in peak_lod.columns:
        gi_cols.append('gene_symbol')
    if 'start' in gi.columns:
        gi_cols.append('start')
    gi_sub = gi[gi_cols].drop_duplicates('gene_id')
    if 'start' in gi_sub.columns:
        gi_sub = gi_sub.assign(start_mb=gi_sub['start'] / 1e6)
    peak_lod = peak_lod.merge(gi_sub, on='gene_id', how='left')

    if 'start_mb' in peak_lod.columns and 'peak_mb' in peak_lod.columns:
        peak_lod['cis_dist_mb'] = peak_lod['peak_mb'] - peak_lod['start_mb']

    # beta_{founder} cols come through from run_lmm_scan_chr.py peak_stats write
    _base = [c for c in PEAK_LOD_COLS if c in peak_lod.columns]
    _beta = [c for c in peak_lod.columns if c.startswith('beta_')]
    peak_lod[_base + _beta].to_csv(out / f'{tag}_peak_lod.csv', index=False)
    log.info('%s_peak_lod.csv: %d genes  %d beta cols', tag, len(peak_lod), len(_beta))

    # ── timing_chr*.json → {tag}_scan_stats.csv ───────────────────────────────

    timing_files = sorted(lod_dir.glob('timing_chr*.json'))
    if timing_files:
        rows = [json.loads(f.read_text()) for f in timing_files]
        pd.DataFrame(rows).to_csv(out / f'{tag}_scan_stats.csv', index=False)

    log.info('DONE merge: tag=%s  %.1fs', tag, time.perf_counter() - t0)


if __name__ == '__main__':
    main()
