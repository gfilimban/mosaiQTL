"""Concatenate per-chromosome LOD CSVs into lod_genome.csv.

Called by the MERGE_RESULTS Nextflow process:
    python merge_lod_csvs.py --celltype <name> --output lod_genome.csv lod_chr*.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def _chr_key(path: str) -> tuple[int, int | str]:
    stem = Path(path).stem.replace('lod_chr', '')
    return (0, int(stem)) if stem.isdigit() else (1, stem)


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

    # also write peak_lod.csv — one row per gene at its maximum-LOD locus
    peak_idx = merged.groupby('gene_id')['lod'].idxmax()
    peak = merged.loc[peak_idx].reset_index(drop=True)
    peak_path = str(output).replace('lod_genome.csv', 'peak_lod.csv')
    peak.to_csv(peak_path, index=False)
    log.info('wrote %s (%d genes)', peak_path, len(peak))


if __name__ == '__main__':
    main()
