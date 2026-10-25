"""Sanger SNP fine-mapping for eGenes — called by Nextflow FINEMAP process."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


@click.command()
@click.option('--egenes',          required=True, type=click.Path(exists=True))
@click.option('--counts',          required=True, type=click.Path(exists=True))
@click.option('--lod',             required=True, type=click.Path(exists=True))
@click.option('--sanger-alleles',  required=True, type=click.Path(exists=True))
@click.option('--celltype',        required=True)
@click.option('--output-dir',      required=True, type=click.Path())
def main(egenes: str, counts: str, lod: str, sanger_alleles: str,
         celltype: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info('finemap %s', celltype)
    cmd = [
        sys.executable, '-m', 'mosaiqtl', 'finemap',
        '--egenes',         egenes,
        '--counts',         counts,
        '--lod',            lod,
        '--sanger-alleles', sanger_alleles,
        '--output-dir',     str(out),
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
