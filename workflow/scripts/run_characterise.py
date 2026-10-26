#!/usr/bin/env python
"""Run mosaiqtl characterise — compute LOD-drop CI for eGenes.

Reads eqtl_peak.csv (from REPORT), fills ci_lo_bp/ci_hi_bp via LMM rescan,
writes eqtl_peak_ci.csv for downstream FINEMAP.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--egenes',     required=True)
    p.add_argument('--phenotypes', required=True)
    p.add_argument('--kinship',    required=True)
    p.add_argument('--cache',      required=True)
    p.add_argument('--metadata',   required=True)
    p.add_argument('--covariates', default='sex,dataset')
    p.add_argument('--output-dir', default='.')
    args = p.parse_args()

    cmd = [
        sys.executable, '-m', 'mosaiqtl', 'characterise',
        '--egenes',      args.egenes,
        '--phenotypes',  args.phenotypes,
        '--kinship',     args.kinship,
        '--cache',       args.cache,
        '--metadata',    args.metadata,
        '--covariates',  args.covariates,
        '--output-dir',  args.output_dir,
        '--verbose',
    ]
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
