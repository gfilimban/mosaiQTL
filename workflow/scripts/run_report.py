"""eGene calling + submission files — called by Nextflow REPORT process."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def _git_hash(project_dir: str | None) -> str:
    if not project_dir:
        return 'unknown'
    try:
        result = subprocess.run(
            ['git', '-C', project_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def _pkg_version() -> str:
    try:
        return importlib.metadata.version('mosaiqtl')
    except importlib.metadata.PackageNotFoundError:
        return '0.0.0+unknown'


@click.command()
@click.option('--scan-result',    required=True,  type=click.Path(exists=True))
@click.option('--perm-dir',       required=True,  type=click.Path(exists=True))
@click.option('--allele-effects', required=True,  type=click.Path(exists=True))
@click.option('--output-dir',     required=True,  type=click.Path())
@click.option('--celltype',       default='',     help='Cell type label for metadata.')
@click.option('--project-dir',    default=None,   help='Repo root for git-hash resolution.')
@click.option('--nf-run-name',    default='',     help='Nextflow workflow.runName.')
def main(scan_result: str, perm_dir: str, allele_effects: str, output_dir: str,
         celltype: str, project_dir: str | None, nf_run_name: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'mosaiqtl', 'report',
        '--scan-result',    scan_result,
        '--perm-dir',       perm_dir,
        '--allele-effects', allele_effects,
        '--output-dir',     str(out),
    ]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)

    metadata = {
        'mosaiqtl_version': _pkg_version(),
        'git_hash':         _git_hash(project_dir),
        'celltype':         celltype,
        'nf_run_name':      nf_run_name,
        'timestamp':        datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out / 'run_metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info('metadata written to %s', meta_path)


if __name__ == '__main__':
    main()
