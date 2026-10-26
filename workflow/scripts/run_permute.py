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


@click.command()
@click.option('--cache',       required=True, type=click.Path(exists=True))
@click.option('--counts',      required=True, type=click.Path(exists=True))
@click.option('--metadata',    required=True, type=click.Path(exists=True))
@click.option('--null-fit',    required=True, type=click.Path(exists=True))
@click.option('--gene-info',   required=True, type=click.Path(exists=True))
@click.option('--n-perms',     default=1000, show_default=True)
@click.option('--scan-window', default='local', show_default=True)
@click.option('--window-mb',   default=1.0, show_default=True)
@click.option('--covariates',  default='sex,dataset', show_default=True)
@click.option('--output',      required=True, type=click.Path())
def main(cache: str, counts: str, metadata: str, null_fit: str, gene_info: str,
         n_perms: int, scan_window: str, window_mb: float, covariates: str,
         output: str) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'mosaiqtl', 'permute',
        '--cache',       cache,
        '--counts',      counts,
        '--metadata',    metadata,
        '--null-fit',    null_fit,
        '--gene-info',   gene_info,
        '--n-perms',     str(n_perms),
        '--scan-window', scan_window,
        '--window-mb',   str(window_mb),
        '--covariates',  covariates,
        '--output',      str(out),
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
