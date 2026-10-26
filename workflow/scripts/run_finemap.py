"""Sanger SNP fine-mapping — called by Nextflow FINEMAP process."""

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
@click.option('--egenes',     required=True, type=click.Path(exists=True))
@click.option('--phenotypes', required=True, type=click.Path(exists=True))
@click.option('--kinship',    required=True, type=click.Path(exists=True))
@click.option('--metadata',   required=True, type=click.Path(exists=True))
@click.option('--cache',      required=True, type=click.Path(exists=True))
@click.option('--alleles',    required=True, type=click.Path(exists=True))
@click.option('--celltype',   required=True)
@click.option('--output-dir', required=True, type=click.Path())
@click.option('--covariates', default='sex,dataset', show_default=True)
def main(egenes: str, phenotypes: str, kinship: str, metadata: str, cache: str,
         alleles: str, celltype: str, output_dir: str, covariates: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path(sys.executable).parent / 'mosaiqtl'), 'finemap',
        '--egenes',      egenes,
        '--phenotypes',  phenotypes,
        '--kinship',     kinship,
        '--metadata',    metadata,
        '--cache',       cache,
        '--alleles',     alleles,
        '--cell-type',   celltype,
        '--covariates',  covariates,
        '--output-dir',  str(out),
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
