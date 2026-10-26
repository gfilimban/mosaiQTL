"""Permutation threshold via GEV — called by Nextflow PERMUTE process."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


_SW_MAP: dict[str, str] = {'L': 'local', 'C': 'cis', 'G': 'genome'}


@click.command()
@click.option('--cache',       required=True, type=click.Path(exists=True))
@click.option('--phenotypes',  required=True, type=click.Path(exists=True))
@click.option('--kinship',     required=True, type=click.Path(exists=True))
@click.option('--metadata',    required=True, type=click.Path(exists=True))
@click.option('--gene-info',   required=True, type=click.Path(exists=True))
@click.option('--n-perms',     default=1000, show_default=True)
@click.option('--scan-type',   default='L', show_default=True,
              type=click.Choice(['L', 'C', 'G']),
              help='L = local ±window_mb | C = chromosome-wide | G = genome-wide')
@click.option('--window-mb',   default=1.0, show_default=True)
@click.option('--output',      required=True, type=click.Path())
def main(cache: str, phenotypes: str, kinship: str, metadata: str, gene_info: str,
         n_perms: int, scan_type: str, window_mb: float, output: str) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    mosaiqtl_bin = str(Path(sys.executable).parent / 'mosaiqtl')
    cmd = [
        mosaiqtl_bin, 'permute',
        '--cache',       cache,
        '--phenotypes',  phenotypes,
        '--kinship',     kinship,
        '--metadata',    metadata,
        '--gene-info',   gene_info,
        '--n-perms',     str(n_perms),
        '--scan-window', _SW_MAP[scan_type],
        '--window-mb',   str(window_mb),
        '--output',      str(out),
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
