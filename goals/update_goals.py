"""Update GOALS.md from live output files.

Sources (all optional — missing files skipped silently):
  validation/phase1/compare/{rop_lod,mi_lod,perm_threshold}.csv  → P1-S1/S2/S3
  validation/phase3/py_data/benchmark_summary.csv                 → SPEED-02, MEM-02 (same hw)
  validation/gastroc_data/py_data/scan_stats.csv                  → SPEED-01/05/06, MEM-01
  validation/gastroc_data/r_data/timing.csv                       → miQTL gastroc reference

Usage:
    python goals/update_goals.py [--dry-run]
"""
from __future__ import annotations
import argparse, pathlib, re
from datetime import date
import numpy as np
import pandas as pd

ROOT  = pathlib.Path(__file__).parent.parent
TODAY = date.today().isoformat()

# (goal_id, pipe-column-index-1-based) for every updatable cell
# Phase 1/2 table:  | ID | Section | Threshold | Current | Status |
# SPEED table:      | ID | Desc | Baseline | Target | Current | Delta | Last tested |
# MEM table:        | ID | Desc | Target | Current | Assessment |
COLS: dict[str, tuple[str, int]] = {
    'P1-S1-cur': ('P1-S1', 4), 'P1-S1-sta': ('P1-S1', 5),
    'P1-S2-cur': ('P1-S2', 4), 'P1-S2-sta': ('P1-S2', 5),
    'P1-S3-cur': ('P1-S3', 4), 'P1-S3-sta': ('P1-S3', 5),
    'SPEED-01-cur': ('SPEED-01', 5), 'SPEED-01-dt': ('SPEED-01', 7),
    'SPEED-02-cur': ('SPEED-02', 5), 'SPEED-02-dt': ('SPEED-02', 7),
    'SPEED-05-cur': ('SPEED-05', 5), 'SPEED-05-dt': ('SPEED-05', 7),
    'SPEED-06-cur': ('SPEED-06', 5), 'SPEED-06-dt': ('SPEED-06', 7),
    'MEM-01-cur':   ('MEM-01',   4),
    'MEM-02-cur':   ('MEM-02',   4),
}


def _ok(v: float, thr: float, op: str) -> str:
    passed = v >= thr if op == '>=' else v <= thr if op == '<=' else abs(v) < thr
    return '🟢' if passed else '🔴'


def _patch(lines: list[str], goal_id: str, col: int, value: str) -> bool:
    pat = re.compile(r'^\|\s*' + re.escape(goal_id) + r'\s*\|')
    for i, line in enumerate(lines):
        if pat.match(line):
            cells = line.split('|')
            if col < len(cells):
                cells[col] = f' {value} '
                lines[i] = '|'.join(cells)
            return True
    return False


def collect() -> dict[str, str]:
    """Return {key: string_value} for every cell we can compute."""
    up: dict[str, str] = {}

    # Phase 1 ─────────────────────────────────────────────────────────────────
    cmp = ROOT / 'validation' / 'phase1' / 'compare'
    try:
        df = pd.read_csv(cmp / 'rop_lod.csv').dropna()
        r2 = float(np.corrcoef(df['lod_python'], df['lod_r'])[0, 1] ** 2)
        up['P1-S1-cur'] = f'{r2:.4f}'
        up['P1-S1-sta'] = _ok(r2, 0.990, '>=')
    except Exception: pass
    try:
        df = pd.read_csv(cmp / 'mi_lod.csv').dropna()
        r  = float(np.corrcoef(df['lod_python'], df['lod_r'])[0, 1])
        up['P1-S2-cur'] = f'{r:.4f}'
        up['P1-S2-sta'] = _ok(r, 0.75, '>=')
    except Exception: pass
    try:
        d = float(pd.read_csv(cmp / 'perm_threshold.csv')['delta'].iloc[0])
        up['P1-S3-cur'] = f'{d:.4f}'
        up['P1-S3-sta'] = _ok(d, 0.5, '<')
    except Exception: pass

    # Phase 3 same-hardware benchmark ─────────────────────────────────────────
    try:
        df = (pd.read_csv(ROOT / 'validation/phase3/py_data/benchmark_summary.csv')
                .set_index('section'))
        s4 = df.loc['S4_lmm_scan']
        sx, note = float(s4['speedup']), str(s4['note'])
        up['SPEED-02-cur'] = f'mosaiQTL: {sx:.1f}× vs miQTL (same hw M3 Mac, simulated; {note})'
        up['SPEED-02-dt']  = TODAY
        up['MEM-02-cur']   = f'mosaiQTL: {float(s4["peak_mb"]):.0f} MB (M3 Mac; {note})'
        s6 = df.loc['S6_permutation']
        sx6, note6 = float(s6['speedup']), str(s6['note'])
        up.setdefault('SPEED-05-cur',
                      f'mosaiQTL: {sx6:.0f}× vs miQTL (same hw M3 Mac; {note6})')
        up.setdefault('SPEED-05-dt', TODAY)
    except Exception: pass

    # Gastroc scan_stats (real data) ──────────────────────────────────────────
    try:
        sc = (pd.read_csv(ROOT / 'validation/gastroc_data/py_data/scan_stats.csv')
                .set_index('section'))
        s4   = sc.loc['S4']
        h, n = float(s4['elapsed_s']) / 3600, int(s4['n_items'])
        up['SPEED-01-cur'] = f'mosaiQTL: {h:.2f}h / {n:,} genes / 20 chr (M3 Mac)'
        up['SPEED-01-dt']  = TODAY
        up['SPEED-06-cur'] = f'mosaiQTL: {n/float(s4["elapsed_s"])*60:.0f} genes/min ({n:,} genes, 20 chr, M3 Mac)'
        up['SPEED-06-dt']  = TODAY
        s5_s = float(sc.loc['S5']['elapsed_s'].min())
        up.setdefault('SPEED-05-cur',
                      f'mosaiQTL: {s5_s:.1f}s (1000 perms, genome-wide, n=415, M3 Mac)')
        up.setdefault('SPEED-05-dt', TODAY)
        rss = float(sc[sc.index.str.startswith('S4')]['peak_rss_mb'].max())
        up['MEM-01-cur'] = f'mosaiQTL: {rss:.0f} MB RSS ({n:,} genes, 20 chr, M3 Mac)'
    except Exception: pass

    return up


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    up    = collect()
    gpath = ROOT / 'goals' / 'GOALS.md'
    lines = gpath.read_text().splitlines(keepends=True)

    changed = 0
    for key, (goal_id, col) in COLS.items():
        if key in up:
            _patch(lines, goal_id, col, up[key])
            print(f'  {goal_id}[col {col}] = {up[key][:72]}')
            changed += 1

    if args.dry_run:
        print(f'dry-run: {changed} updates (not written)')
        return
    gpath.write_text(''.join(lines))
    print(f'wrote {gpath}  ({changed} updates)')


if __name__ == '__main__':
    main()
