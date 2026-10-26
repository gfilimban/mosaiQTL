"""Fit LMM null model — called by Nextflow FIT_NULL process."""

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
@click.option('--cache',      required=True, type=click.Path(exists=True))
@click.option('--counts',     required=True, type=click.Path(exists=True))
@click.option('--metadata',   required=True, type=click.Path(exists=True))
@click.option('--output',     required=True, type=click.Path())
@click.option('--covariates', default='sex,dataset', show_default=True)
def main(cache: str, counts: str, metadata: str, output: str, covariates: str) -> None:
    cmd = [
        sys.executable, '-m', 'mosaiqtl', 'fit-null',
        '--cache',       cache,
        '--counts',      counts,
        '--metadata',    metadata,
        '--output',      output,
        '--covariates',  covariates,
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
