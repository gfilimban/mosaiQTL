"""Submission file writer (eqtl_peak + eqtl_stats) — called by Nextflow REPORT process."""

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
@click.option('--egenes',           required=True, type=click.Path(exists=True))
@click.option('--allele-effects',   required=True, type=click.Path(exists=True))
@click.option('--perm-stats',       required=True, type=click.Path(exists=True))
@click.option('--lod',              required=True, type=click.Path(exists=True))
@click.option('--finemapping',      required=True, type=click.Path(exists=True))
@click.option('--finemapping-peak', required=True, type=click.Path(exists=True))
@click.option('--celltype',         required=True)
@click.option('--output-dir',       required=True, type=click.Path())
def main(egenes: str, allele_effects: str, perm_stats: str, lod: str,
         finemapping: str, finemapping_peak: str,
         celltype: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info('report %s', celltype)
    cmd = [
        sys.executable, '-m', 'mosaiqtl', 'report',
        '--egenes',          egenes,
        '--allele-effects',  allele_effects,
        '--perm-dir',        str(Path(perm_stats).parent),
        '--lod',             lod,
        '--finemapping',     finemapping,
        '--finemapping-peak', finemapping_peak,
        '--output-dir',      str(out),
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
