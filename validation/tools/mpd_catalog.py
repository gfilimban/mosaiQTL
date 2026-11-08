"""mpd_catalog.py — download and cache Mouse Phenome Database data for CC/CC8/CC pre panels.

Queries three panel groups:
  CC      — Collaborative Cross inbred strains (panelsym=CC)
  CC pre  — Pre-CC emerging lines + 8 inbred founders (panelsym=CC pre)
  CC8     — 8 inbred founders alone (panelsym=inbred, paneldesc contains 'CC8')

For each project with phenotype data (sector=pheno):
  1. Download measure metadata → measures.csv
  2. Download all strain means   → strain_means/{projsym}.csv

Outputs written to validation/mpd/:
  projects.csv              one row per project (all panels)
  measures.csv              one row per measure (projsym + measure fields)
  strain_means/{projsym}.csv  all strain means for that project

Re-running is safe: existing strain_means/ files are skipped unless --force is given.

Usage
-----
    conda run -n bioinfo python validation/tools/mpd_catalog.py
    conda run -n bioinfo python validation/tools/mpd_catalog.py --force   # re-fetch all
    conda run -n bioinfo python validation/tools/mpd_catalog.py --panels CC8
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
_VALIDATION = pathlib.Path(__file__).parents[1]
_OUT        = _VALIDATION / 'mpd'
_MEANS_DIR  = _OUT / 'strain_means'

# ── MPD API base ──────────────────────────────────────────────────────────────
_BASE = 'https://phenome.jax.org/api'
_HEADERS = {'User-Agent': 'mosaiqtl-catalog/1.0 (gfilimba@uci.edu)'}
_DELAY   = 0.25   # seconds between requests — be polite to JAX servers

# ── CC founder strain names as used in MPD ────────────────────────────────────
# Maps MPD registry name → eQTL effect column suffix (from pipeline.py schema)
FOUNDER_MPD_TO_EFFECT: dict[str, str] = {
    'A/J':           'AJ',
    'C57BL/6J':      'B6J',
    '129S1/SvImJ':   '129S1J',
    'NOD/ShiLtJ':    'NODJ',
    'NZO/HlLtJ':     'NZOJ',
    'CAST/EiJ':      'CASTJ',
    'PWK/PhJ':       'PWKJ',
    'WSB/EiJ':       'WSBJ',
}
FOUNDER_STRAINS = set(FOUNDER_MPD_TO_EFFECT.keys())


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> dict:
    """HTTP GET with retry and politeness delay."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                log.warning('  request failed (%s) — retrying in %ds …', exc, wait)
                time.sleep(wait)
            else:
                raise
    return {}


def _fetch_projects(panel: str, paneldesc_filter: str | None = None) -> list[dict]:
    """Fetch all projects for a given panelsym, optionally filtering by paneldesc."""
    url = f'{_BASE}/projects?panelsym={urllib.parse.quote(panel)}'
    data = _get(url)
    projects: list[dict] = data.get('projects', [])
    if paneldesc_filter:
        projects = [p for p in projects if paneldesc_filter in str(p.get('paneldesc', ''))]
    # tag each row with our logical panel label
    logical = 'CC8' if paneldesc_filter else panel
    for p in projects:
        p['panel_label'] = logical
    return projects


def _fetch_measures(projsym: str) -> list[dict]:
    """Fetch measure metadata for one project."""
    time.sleep(_DELAY)
    url = f'{_BASE}/pheno/measureinfo/{urllib.parse.quote(projsym)}'
    data = _get(url)
    measures = data.get('measures_info', [])
    for m in measures:
        m['projsym'] = projsym
    return measures


def _fetch_strain_means(projsym: str, out_path: pathlib.Path, force: bool) -> bool:
    """Download all strain means for one project.  Returns True if fetched."""
    if out_path.exists() and not force:
        log.debug('  %s: strain means already cached — skipping', projsym)
        return False
    time.sleep(_DELAY)
    url = f'{_BASE}/pheno/strainmeans/{urllib.parse.quote(projsym)}'
    data = _get(url)
    rows = data.get('strainmeans', [])
    if not rows:
        log.warning('  %s: no strain means returned', projsym)
        return False
    df = pd.DataFrame(rows)
    # annotate founder rows
    df['is_founder'] = df['strain'].isin(FOUNDER_STRAINS)
    df['effect_col'] = df['strain'].map(FOUNDER_MPD_TO_EFFECT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    panels: list[str] | None = None,
    out_dir: pathlib.Path = _OUT,
    force: bool = False,
) -> None:
    """Download and cache MPD project + measure + strain-means data.

    Parameters
    ----------
    panels : list[str] | None
        Which logical panels to fetch: any subset of ``['CC', 'CC pre', 'CC8']``.
        Defaults to all three.
    out_dir : pathlib.Path
        Root output directory (default: validation/mpd/).
    force : bool
        If True, re-download strain means even when already cached.
    """
    if panels is None:
        panels = ['CC', 'CC pre', 'CC8']

    means_dir = out_dir / 'strain_means'
    means_dir.mkdir(parents=True, exist_ok=True)

    # ── step 1: collect projects ─────────────────────────────────────────────
    all_projects: list[dict] = []
    for panel in panels:
        if panel == 'CC8':
            log.info('fetching CC8 projects (panelsym=inbred, paneldesc~CC8) …')
            projs = _fetch_projects('inbred', paneldesc_filter='CC8')
        else:
            log.info('fetching %s projects …', panel)
            projs = _fetch_projects(panel)
        log.info('  %d projects', len(projs))
        all_projects.extend(projs)

    proj_df = pd.DataFrame(all_projects)
    # keep only phenotype projects (skip SNP/QTL sectors)
    pheno_df = proj_df[proj_df['mpdsector'].isin(['pheno', 'phenoarchive'])].copy()
    log.info('total pheno projects across panels: %d / %d', len(pheno_df), len(proj_df))

    projects_path = out_dir / 'projects.csv'
    proj_df.to_csv(projects_path, index=False)
    log.info('written → %s', projects_path)

    # ── step 2: measures metadata ─────────────────────────────────────────────
    all_measures: list[dict] = []
    projsyms = pheno_df['projsym'].unique().tolist()
    log.info('downloading measure metadata for %d projects …', len(projsyms))
    for i, projsym in enumerate(projsyms, 1):
        log.info('  [%d/%d] %s', i, len(projsyms), projsym)
        try:
            measures = _fetch_measures(projsym)
            all_measures.extend(measures)
        except Exception as exc:
            log.warning('  %s: measure fetch failed — %s', projsym, exc)

    measures_df = pd.DataFrame(all_measures)
    measures_path = out_dir / 'measures.csv'
    measures_df.to_csv(measures_path, index=False)
    log.info('total measures: %d | written → %s', len(measures_df), measures_path)

    # ── step 3: strain means ──────────────────────────────────────────────────
    log.info('downloading strain means for %d projects …', len(projsyms))
    fetched = skipped = failed = 0
    for i, projsym in enumerate(projsyms, 1):
        out_path = means_dir / f'{projsym}.csv'
        log.info('  [%d/%d] %s', i, len(projsyms), projsym)
        try:
            did_fetch = _fetch_strain_means(projsym, out_path, force=force)
            if did_fetch:
                fetched += 1
            else:
                skipped += 1
        except Exception as exc:
            log.warning('  %s: strain means fetch failed — %s', projsym, exc)
            failed += 1

    log.info(
        'strain means: %d fetched  %d skipped (cached)  %d failed',
        fetched, skipped, failed,
    )
    log.info('done.  output root → %s', out_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Download and cache MPD data for CC/CC8/CC pre panels.'
    )
    p.add_argument(
        '--panels', nargs='+',
        choices=['CC', 'CC pre', 'CC8'],
        default=['CC', 'CC pre', 'CC8'],
        help='Which panels to fetch (default: all three)',
    )
    p.add_argument(
        '--out-dir', type=pathlib.Path, default=_OUT,
        help=f'Output directory (default: {_OUT})',
    )
    p.add_argument(
        '--force', action='store_true',
        help='Re-download strain means even when already cached',
    )
    args = p.parse_args()
    run(panels=args.panels, out_dir=args.out_dir, force=args.force)


if __name__ == '__main__':
    main()
