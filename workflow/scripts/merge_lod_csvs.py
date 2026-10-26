"""Concatenate per-chromosome LOD CSVs into lod_genome.csv.

Called by the MERGE_RESULTS Nextflow process:
    python merge_lod_csvs.py --celltype <name> --output lod_genome.csv lod_chr*.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def _chr_key(path: str) -> tuple[int, int | str]:
    import re
    m = re.search(r'(\d+)$', Path(path).stem)
    return (0, int(m.group(1))) if m else (1, Path(path).stem)


@click.command()
@click.argument('csvs', nargs=-1, required=True)
@click.option('--celltype', default='', help='Cell type label (for logging).')
@click.option('--output', default='lod_genome.csv', show_default=True,
              help='Output file path.')
def main(csvs: tuple[str, ...], celltype: str, output: str) -> None:
    """Merge per-chromosome LOD CSVs into a single genome-wide file."""
    sorted_csvs = sorted(csvs, key=_chr_key)
    if not sorted_csvs:
        raise FileNotFoundError('no per-chr LOD CSVs provided')

    label = celltype or 'unknown'
    log.info('merging %d chr CSVs for %s', len(sorted_csvs), label)
    merged = pd.concat(
        [pd.read_csv(f) for f in sorted_csvs],
        ignore_index=True,
    )

    merged.to_csv(output, index=False)
    log.info('wrote %s (%d rows)', output, len(merged))

    # peak_lod.csv — one row per gene at its maximum-LOD locus
    # columns renamed to the peak-specific schema used by mosaiqtl estimate
    peak_idx = merged.groupby('gene_id')['lod'].idxmax()
    peak = merged.loc[peak_idx].reset_index(drop=True).rename(columns={
        'locus':  'peak_locus',
        'lod':    'peak_lod',
        'pos_mb': 'peak_mb',
    })
    peak_path = str(output).replace('lod_genome.csv', 'peak_lod.csv')
    peak.to_csv(peak_path, index=False)
    log.info('wrote %s (%d genes)', peak_path, len(peak))

    # scan_stats.csv — collect per-chr timing JSONs emitted by run_lmm_scan_chr.py
    timing_files = sorted(Path('.').glob('timing_chr*.json'), key=lambda p: _chr_key(str(p)))
    scan_stats_rows = []
    for tf in timing_files:
        try:
            t = json.loads(tf.read_text())
            scan_stats_rows.append({
                'section':      'lmm_scan',
                'component':    f'chr{t["chr"]}',
                'n_items':      t['n_genes'],
                'elapsed_s':    t['elapsed_s'],
                'peak_heap_mb': t['peak_heap_mb'],
                'peak_rss_mb':  t['peak_rss_mb'],
            })
        except Exception as exc:
            log.warning('could not read %s: %s', tf, exc)
    stats_path = str(output).replace('lod_genome.csv', 'scan_stats.csv')
    pd.DataFrame(scan_stats_rows).to_csv(stats_path, index=False)
    log.info('wrote %s (%d rows)', stats_path, len(scan_stats_rows))


if __name__ == '__main__':
    main()
