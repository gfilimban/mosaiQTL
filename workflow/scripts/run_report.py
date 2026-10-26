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
@click.option('--scan-stats',     default=None,   type=click.Path(exists=True),
              help='scan_stats.csv from MERGE_RESULTS — copied to output dir.')
@click.option('--output-dir',     required=True,  type=click.Path())
@click.option('--celltype',       default='',     help='Cell type label for metadata.')
@click.option('--project-dir',    default=None,   help='Repo root for git-hash resolution.')
@click.option('--nf-run-name',    default='',     help='Nextflow workflow.runName.')
@click.option('--metadata',       default=None,   type=click.Path(),
              help='Sample metadata CSV — forwarded to mosaiqtl report for scan_info.json.')
@click.option('--phenotypes',     default=None,   type=click.Path(),
              help='counts_lognorm.h5 — forwarded to mosaiqtl report (currently unused).')
def main(scan_result: str, perm_dir: str, allele_effects: str, scan_stats: str | None,
         output_dir: str, celltype: str, project_dir: str | None, nf_run_name: str,
         metadata: str | None, phenotypes: str | None) -> None:
    import shutil
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(Path(sys.executable).parent / 'mosaiqtl'), 'report',
        '--scan-result',    scan_result,
        '--perm-dir',       perm_dir,
        '--allele-effects', allele_effects,
        '--output-dir',     str(out),
        '--celltype',       celltype or 'whole_tissue',
    ]
    if metadata and Path(metadata).exists():
        cmd += ['--metadata', metadata]
    log.info('running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)

    # copy scan_stats.csv (per-chr timing) into the output dir
    if scan_stats and Path(scan_stats).exists():
        dest = out / 'scan_stats.csv'
        if Path(scan_stats).resolve() != dest.resolve():
            shutil.copy2(scan_stats, dest)
        log.info('scan_stats.csv copied to %s', dest)

    run_metadata = {
        'mosaiqtl_version': _pkg_version(),
        'git_hash':         _git_hash(project_dir),
        'celltype':         celltype,
        'nf_run_name':      nf_run_name,
        'timestamp':        datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out / 'run_metadata.json'
    meta_path.write_text(json.dumps(run_metadata, indent=2))
    log.info('run_metadata.json written to %s', meta_path)


if __name__ == '__main__':
    main()
