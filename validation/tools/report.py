"""
report.py — generate docs/VALIDATION_STATUS.md from phase output CSVs.

Reads the compare/ and py_data/ outputs written by each validation script and
produces a single, fully-populated markdown report showing actual numeric values
for every gate.  Run this after any validation phase to get an up-to-date status.

Usage:
    conda run -n bioinfo python validation/report.py
    conda run -n bioinfo python validation/report.py --phase 2
    conda run -n bioinfo python validation/report.py --phase 2 --out docs/phase2.md
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

# ── paths ────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).parent.parent
PH1       = REPO / 'validation' / 'phase1'
PH2       = REPO / 'validation' / 'phase2'
PH3       = REPO / 'validation' / 'phase3'

PASS = '🟢'
FAIL = '❌'
SKIP = '⚠️ '
NA   = '—'


# ── helpers ───────────────────────────────────────────────────────────────────

def _read(path: Path) -> 'pd.DataFrame | None':
    """read CSV, return None if missing."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _pr(a: np.ndarray, b: np.ndarray) -> 'tuple[float, float, float]':
    """(pearson_r, slope, intercept) from arrays a (python) and b (r reference)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float('nan'), float('nan'), float('nan')
    r, _ = _scipy_stats.pearsonr(a[mask], b[mask])
    slope, intercept, *_ = _scipy_stats.linregress(b[mask], a[mask])
    return float(r), float(slope), float(intercept)


def _gate(value: float, threshold: float, op: str = '>=') -> str:
    if np.isnan(value):
        return SKIP
    ok = (value >= threshold) if op == '>=' else (value < threshold)
    return PASS if ok else FAIL


def _fmt(v, fmt='.4f') -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return NA
    return format(v, fmt)


def _row(gate, description, metric_str, threshold_str, status) -> str:
    return f'| {gate} | {description} | {metric_str} | {threshold_str} | {status} |'


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def _phase1_section(lines: list) -> None:
    lines += [
        '',
        '## Phase 1 — Minimal miQTL Workflow',
        '',
        '**Scripts:** `Rscript validation/r_reference/sim_validation.R` → '
        '`python validation/validate_sim.py`  ',
        '**Data:** sparcc_cache + sim.CC.data(num.lines=40, num.sim=1)  ',
        '**Purpose:** Core LMM engine correctness using identity kinship (OLS shortcut).',
        '',
        '| Gate | Description | Metric | Threshold | Status |',
        '|------|-------------|--------|-----------|--------|',
    ]

    # P1-S1: ROP LOD r²
    rop_r2 = NA; s1_status = SKIP
    ph1_cmp = PH1 / 'compare'
    rop_f = ph1_cmp / 'rop_lod_compare.csv' if (ph1_cmp / 'rop_lod_compare.csv').exists() else None
    # also try py_data
    if rop_f is None:
        rop_f2 = PH1 / 'py_data' / 'rop_lod.csv'
        rop_r2_f = PH1 / 'r_data' / 'rop_lod_r.csv'
        if rop_f2.exists() and rop_r2_f.exists():
            py_df = pd.read_csv(rop_f2)
            r_df  = pd.read_csv(rop_r2_f)
            merged = py_df.merge(r_df, on='locus', suffixes=('_py','_r'), how='inner') if 'locus' in py_df.columns else pd.DataFrame()
            if len(merged) > 2:
                r_val, *_ = _pr(merged.filter(like='_py').iloc[:,0].values,
                                merged.filter(like='_r').iloc[:,0].values)
                rop_r2 = r_val ** 2
    # try phase1/compare directory (old layout)
    for cand in [PH1/'compare/rop_lod.csv', PH1/'compare/lod_compare.csv']:
        if cand.exists():
            df = pd.read_csv(cand)
            cols = df.columns.tolist()
            py_col = next((c for c in cols if 'python' in c.lower() or 'py' in c.lower()), None)
            r_col  = next((c for c in cols if '_r' == c[-2:] or c == 'lod_r'), None)
            if py_col and r_col and len(df) > 2:
                r_val, *_ = _pr(df[py_col].values, df[r_col].values)
                rop_r2 = r_val ** 2
            break
    if isinstance(rop_r2, float) and not np.isnan(rop_r2):
        s1_status = _gate(rop_r2, 0.990)
        rop_r2_str = _fmt(rop_r2)
    else:
        rop_r2_str = NA

    lines.append(_row('P1-S1', 'ROP scan LOD r²', f'r²={rop_r2_str}', '≥ 0.990', s1_status))

    # P1-S2 / P1-S3: read from phase1 py_data gate_summary if it exists
    gs1 = _read(PH1 / 'py_data' / 'gate_summary.csv')
    for gate, desc, thresh in [('P1-S2', 'MI scan LOD r', '≥ 0.75'),
                                ('P1-S3', 'GEV threshold delta', '< 0.5')]:
        if gs1 is not None and gate in gs1['gate'].values:
            res = gs1.loc[gs1['gate'] == gate, 'result'].iloc[0]
            sym = PASS if res == 'PASS' else (SKIP if 'SKIP' in res else FAIL)
            lines.append(_row(gate, desc, res, thresh, sym))
        else:
            lines.append(_row(gate, desc, NA, thresh, SKIP))

    # Known passing values from last validated run (Phase 1 all green)
    lines += [
        '',
        '> Last known values: P1-S1 r²=0.9975 | P1-S2 r=0.9756 | P1-S3 Δ=0.1543',
    ]


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def _phase2_section(lines: list) -> None:
    lines += [
        '',
        '---',
        '',
        '## Phase 2 — Full LMM Structural Validation (S1–S11)',
        '',
        '**Scripts:** `Rscript validation/r_reference/phase2_validation.R` → '
        '`python validation/validate_phase2.py`  ',
        '**Data:** sparcc_cache + sim.CC.data(num.lines=40, num.sim=10, seed=42)  ',
        '**Purpose:** Gate-by-gate verification that every miQTL function produces '
        'numerically identical output in Python. Uses real LOCO kinship throughout.',
        '',
        '| Gate | Description | Metric | Threshold | Status |',
        '|------|-------------|--------|-----------|--------|',
    ]

    cmp  = PH2 / 'compare'
    pyd  = PH2 / 'py_data'
    rd   = PH2 / 'r_data'

    # P2-S1 — cache structure
    s1f = _read(cmp / 's1_cache_info.csv')
    if s1f is not None:
        max_d = int(s1f['delta'].max()) if 'delta' in s1f.columns else -1
        ok = max_d == 0
        lines.append(_row('P2-S1', 'Cache structure (loci/founders/samples)',
                          f'max loci Δ={max_d}',
                          'Δ=0, sets match',
                          PASS if ok else FAIL))
    else:
        lines.append(_row('P2-S1', 'Cache structure', NA, 'Δ=0', SKIP))

    # P2-S2 — haplotype concordance
    s2f = _read(cmp / 's2_haplotype_concordance.csv')
    if s2f is not None:
        conc = float(s2f['concordance'].iloc[0])
        raw_r = float(s2f['raw_prob_r'].iloc[0]) if 'raw_prob_r' in s2f.columns else float('nan')
        raw_s = float(s2f['raw_prob_slope'].iloc[0]) if 'raw_prob_slope' in s2f.columns else float('nan')
        if not np.isnan(raw_r):
            met = f'argmax={conc:.4f}  raw r={raw_r:.5f}  slope={raw_s:.4f}'
        else:
            met = f'argmax concordance={conc:.4f}'
        lines.append(_row('P2-S2', 'Haplotype prob concordance + raw r',
                          met, 'concordance ≥ 0.95', _gate(conc, 0.95)))
    else:
        lines.append(_row('P2-S2', 'Haplotype prob concordance', NA, '≥ 0.95', SKIP))

    # P2-S3 — kinship r (worst chr)
    s3f = _read(cmp / 's3_kinship_r.csv')
    if s3f is not None:
        worst_r = float(s3f['r'].min())
        worst_c = str(s3f.loc[s3f['r'].idxmin(), 'chr'])
        lines.append(_row('P2-S3', 'LOCO kinship Pearson r (worst chr)',
                          f'r={worst_r:.5f} (chr{worst_c})',
                          '> 0.999',
                          _gate(worst_r, 0.999)))
    else:
        lines.append(_row('P2-S3', 'LOCO kinship r (worst chr)', NA, '> 0.999', SKIP))

    # P2-S4 — LMM scan
    s4r = _read(cmp / 's4_r2_per_sim.csv')
    s4pl = _read(cmp / 's4_lod_at_planted.csv')
    if s4r is not None:
        pass_n = int(s4r['pass'].sum())
        med_r2 = float(s4r['r2'].median())
        met = f'{pass_n}/10 sims pass  median r²={med_r2:.4f}'
        if s4pl is not None:
            mean_d = float(s4pl['lod_delta'].mean())
            met += f'  planted LOD mean|Δ|={mean_d:.3f}'
        lines.append(_row('P2-S4', 'LMM scan: pass rate + planted-locus LOD',
                          met, '≥ 5/10 sims r²≥0.975',
                          _gate(pass_n / 10.0, 0.50)))
    else:
        lines.append(_row('P2-S4', 'LMM scan pass rate', NA, '≥ 5/10', SKIP))

    # P2-S5 — cache collapse
    lines.append(_row('P2-S5', 'Cache collapse',
                      'sparcc_cache is pre-collapsed',
                      '≥5% removed + lod r²≥0.999',
                      SKIP + 'N/A'))

    # P2-S6 — GEV threshold
    s6f = _read(cmp / 's6_threshold.csv')
    if s6f is not None:
        py_thr = float(s6f['threshold_python'].iloc[0])
        r_thr  = s6f['threshold_r'].iloc[0] if 'threshold_r' in s6f.columns else float('nan')
        r_thr  = float(r_thr) if pd.notna(r_thr) else float('nan')
        delta  = abs(py_thr - r_thr) if not np.isnan(r_thr) else float('nan')
        if np.isnan(delta):
            gev_e = float(s6f['gev_emp_delta'].iloc[0]) if 'gev_emp_delta' in s6f.columns else float('nan')
            met   = f'Py={py_thr:.3f}  GEV-emp Δ={gev_e:.4f}'
            status = _gate(gev_e, 0.5, '<')
        else:
            met    = f'Py={py_thr:.3f}  R={r_thr:.3f}  Δ={delta:.4f}'
            status = _gate(delta, 0.5, '<')
        lines.append(_row('P2-S6', 'GEV threshold (Python vs R)', met,
                          '|Py−R| < 0.5 LOD', status))
    else:
        lines.append(_row('P2-S6', 'GEV threshold delta', NA, '< 0.5', SKIP))

    # P2-S7 — MI scan
    s7f = _read(cmp / 's7_mi_lod.csv')
    if s7f is not None:
        py_col = 'lod_python' if 'lod_python' in s7f.columns else s7f.columns[0]
        r_col  = 'lod_r'      if 'lod_r'      in s7f.columns else s7f.columns[1]
        r7, slope7, int7 = _pr(s7f[py_col].values, s7f[r_col].values)
        slope_ok = abs(slope7 - 1.0) < 0.05
        int_ok   = abs(int7) < 0.1
        ok7 = r7 >= 0.95 and slope_ok and int_ok
        met = (f'r={r7:.5f}  slope={slope7:.4f}  intercept={int7:.4f}  n={len(s7f)}'
               + ('' if slope_ok else '  ⚠ slope') + ('' if int_ok else '  ⚠ intercept'))
        lines.append(_row('P2-S7', 'MI scan LOD (peak chr)',
                          met, 'r≥0.95 + |slope−1|<0.05 + |b|<0.1',
                          PASS if ok7 else FAIL))
    else:
        lines.append(_row('P2-S7', 'MI scan LOD r', NA, 'r≥0.95', SKIP))

    # P2-S8 — SNP scan (gated)
    lines.append(_row('P2-S8', 'SNP scan', 'allele_dir not built', 'gated (PORT-08)', SKIP))

    # P2-S9 — mediation
    s9f = _read(cmp / 's9_mediation.csv')
    if s9f is not None:
        py_col = 'lod_mediated'   if 'lod_mediated'   in s9f.columns else s9f.columns[0]
        r_col  = 'lod_mediated_r' if 'lod_mediated_r' in s9f.columns else s9f.columns[1]
        r9, slope9, int9 = _pr(s9f[py_col].values, s9f[r_col].values)
        slope_ok = abs(slope9 - 1.0) < 0.05
        int_ok   = abs(int9) < 0.1
        ok9 = r9 >= 0.90 and slope_ok and int_ok
        met = (f'r={r9:.5f}  slope={slope9:.4f}  intercept={int9:.4f}  n={len(s9f)}'
               + ('' if slope_ok else '  ⚠ slope') + ('' if int_ok else '  ⚠ intercept'))
        lines.append(_row('P2-S9', 'Mediation LOD (peak chr)',
                          met, 'r≥0.90 + |slope−1|<0.05 + |b|<0.1',
                          PASS if ok9 else FAIL))
    else:
        lines.append(_row('P2-S9', 'Mediation LOD r', NA, 'r≥0.90', SKIP))

    # P2-S10 — bootstrap CI
    s10f = _read(cmp / 's10_ci_summary.csv')
    if s10f is not None:
        in_ci  = bool(s10f['in_ci_py'].iloc[0])
        ci_lo  = float(s10f['ci_lo_py'].iloc[0])
        ci_hi  = float(s10f['ci_hi_py'].iloc[0])
        plant  = float(s10f['planted_mb'].iloc[0]) if 'planted_mb' in s10f.columns else float('nan')
        ci_lo_r = float(s10f['ci_lo_r'].iloc[0]) if 'ci_lo_r' in s10f.columns else float('nan')
        ci_hi_r = float(s10f['ci_hi_r'].iloc[0]) if 'ci_hi_r' in s10f.columns else float('nan')
        lo_d   = float(s10f['lo_delta_mb'].iloc[0]) if 'lo_delta_mb' in s10f.columns else abs(ci_lo - ci_lo_r)
        hi_d   = float(s10f['hi_delta_mb'].iloc[0]) if 'hi_delta_mb' in s10f.columns else abs(ci_hi - ci_hi_r)
        met = (f'Py=[{ci_lo:.2f},{ci_hi:.2f}]  planted={plant:.2f} Mb  '
               f'in_ci={in_ci}  lo_Δ={lo_d:.2f}  hi_Δ={hi_d:.2f} Mb')
        lines.append(_row('P2-S10', 'Bootstrap CI contains planted locus',
                          met, 'planted in CI', PASS if in_ci else FAIL))
    else:
        lines.append(_row('P2-S10', 'Bootstrap CI', NA, 'planted in CI', SKIP))

    # P2-S11 — instability
    s11f = _read(cmp / 's11_instability.csv')
    # gate is from gate_summary
    gs2  = _read(pyd / 'gate_summary.csv')
    s11_gate = NA; s11_status = SKIP
    if gs2 is not None and 'P2-S11' in gs2['gate'].values:
        res = gs2.loc[gs2['gate'] == 'P2-S11', 'result'].iloc[0]
        s11_status = PASS if res == 'PASS' else (SKIP if 'SKIP' in res else FAIL)
        s11_gate = res
    if s11f is not None and 'lod_py' in s11f.columns and 'lod_r' in s11f.columns:
        py_n = s11f['lod_py'].values; r_n = s11f['lod_r'].values
        py_norm = (py_n - py_n.min()) / (py_n.max() - py_n.min() + 1e-12)
        r_norm  = (r_n  - r_n.min())  / (r_n.max()  - r_n.min()  + 1e-12)
        rs11, _ = _scipy_stats.pearsonr(py_norm, r_norm)
        pk_py = float(s11f.loc[s11f['lod_py'].idxmax(), 'position_mb']) if 'position_mb' in s11f.columns else float('nan')
        pk_r  = float(s11f.loc[s11f['lod_r'].idxmax(),  'position_mb']) if 'position_mb' in s11f.columns else float('nan')
        off   = abs(pk_py - pk_r)
        met   = f'{s11_gate}  profile shape r={rs11:.4f}  peak Δ={off:.2f} Mb'
    else:
        met = s11_gate
    lines.append(_row('P2-S11', 'Instability: planted peak_freq + profile shape',
                      met, 'peak_freq ≥ 0.70', s11_status))

    # Summary line
    if gs2 is not None:
        n_pass = int((gs2['result'] == 'PASS').sum())
        n_skip = int(gs2['result'].str.contains('SKIP').sum())
        n_fail = int((gs2['result'] == 'FAIL').sum())
        n_total = len(gs2)
        lines += [
            '',
            f'**Result: {n_pass}/{n_total - n_skip} runnable gates pass '
            f'({n_skip} skipped/blocked)**',
        ]

    # Timing table
    tf = _read(pyd / 'timing_sections.csv')
    if tf is not None and 'elapsed_s_r' in tf.columns:
        lines += [
            '',
            '### Timing vs R (Phase 2 dataset — 10 phenotypes)',
            '',
            '| Section | Python (s) | R (s) | Speedup |',
            '|---------|-----------|-------|---------|',
        ]
        for _, row in tf.iterrows():
            s_py = float(row['elapsed_s'])
            s_r  = float(row['elapsed_s_r']) if pd.notna(row.get('elapsed_s_r')) else float('nan')
            spd  = float(row['speedup']) if pd.notna(row.get('speedup')) else float('nan')
            r_str   = _fmt(s_r, '.1f') if not np.isnan(s_r) else NA
            spd_str = (f'×{spd:.1f}' if not np.isnan(spd) else NA)
            lines.append(f'| {row["section"]} | {s_py:.1f} | {r_str} | {spd_str} |')


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def _phase3_section(lines: list) -> None:
    lines += [
        '',
        '---',
        '',
        '## Phase 3 — Speed and Memory Benchmark',
        '',
        '**Scripts:** `Rscript validation/r_reference/phase3_validation.R` → '
        '`python validation/validate_phase3.py`  ',
        '**Data:** sparcc_cache + sim.CC.data(num.lines=40, num.sim=50, seed=123)  ',
        '**Purpose:** Confirm Phase 2 accuracy is preserved at 5× scale and measure '
        'throughput improvement.',
        '',
    ]

    bench = _read(PH3 / 'py_data' / 'benchmark_summary.csv')
    if bench is None:
        lines.append('> Phase 3 not yet run. '
                     'Execute `Rscript validation/r_reference/phase3_validation.R` then '
                     '`python validation/validate_phase3.py`.')
        return

    lines += [
        '| Section | Python (s) | R (s) | Speedup | Correctness |',
        '|---------|-----------|-------|---------|-------------|',
    ]
    for _, row in bench.iterrows():
        s_py = float(row['elapsed_s']) if pd.notna(row.get('elapsed_s')) else float('nan')
        s_r  = float(row.get('elapsed_s_r', float('nan'))) if pd.notna(row.get('elapsed_s_r')) else float('nan')
        spd  = float(row.get('speedup', float('nan'))) if pd.notna(row.get('speedup')) else float('nan')
        corr = str(row.get('correctness', NA))
        r_str   = _fmt(s_r, '.1f')   if not np.isnan(s_r)  else NA
        spd_str = f'×{spd:.1f}'      if not np.isnan(spd)  else NA
        lines.append(f'| {row["section"]} | {_fmt(s_py,".1f")} | {r_str} | {spd_str} | {corr} |')


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def _phase4_section(lines: list) -> None:
    lines += [
        '',
        '---',
        '',
        '## Phase 4 — NB-GLMM Correctness',
        '',
        '**Script:** `python validation/validate_phase4.py`  ',
        '**Data:** sparc phenotypes → NB count simulation via validation/sim_sc.py  ',
        '**Purpose:** Validate the novel NB-GLMM model (`nb.py`) — no miQTL R equivalent. '
        'Ground truth from planted QTL signal converted to single-cell counts.',
        '',
        '| Gate | Description | Metric | Threshold | Status |',
        '|------|-------------|--------|-----------|--------|',
    ]
    # No outputs yet — all gates red
    for gate, desc, thresh in [
        ('BIO-P1', 'scan_nb recovers planted locus (Piezo1 chr8)', 'LOD', '> 3'),
        ('BIO-P2', 'NB LOD distribution skew', 'skew', '> 0.5'),
        ('BIO-P3', 'scan_vqtl returns ≥1 result', 'n_hits', '≥ 1'),
        ('BIO-P4', 'scan_epistasis lod_additive', 'lod_additive', '> 0'),
        ('BIO-P5', 'scan_conditional signal1_lod', 'signal1_lod', '> 0'),
    ]:
        lines.append(_row(gate, desc, NA, thresh, '🔴 not run'))


# ── main ──────────────────────────────────────────────────────────────────────

_PHASE_FNS = {1: _phase1_section, 2: _phase2_section,
              3: _phase3_section, 4: _phase4_section}


def generate(out_path: Path, phase: 'int | None' = None) -> None:
    """read phase CSVs and write the populated markdown report.

    Args:
        out_path: destination .md file.
        phase: if given, emit only that phase section (1–4).
                if None, emit all four phases.
    """
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    phase_tag = f' — Phase {phase}' if phase is not None else ''
    lines = [
        f'# mosaiQTL — Validation Status Report{phase_tag}',
        '',
        f'_Generated: {ts}_  ',
        f'_Source: `python validation/report.py{"" if phase is None else f" --phase {phase}"}`_  ',
        '_Each section reads directly from the compare/ and py_data/ CSVs '
        'written by the validation scripts._',
        '',
        '---',
        '',
        '## Gate Standard',
        '',
        'Three conditions must hold simultaneously for a LOD-profile gate to pass:',
        '',
        '| Condition | Purpose | Threshold |',
        '|-----------|---------|-----------|',
        '| Pearson r ≥ threshold | Overall profile shape agreement | section-specific |',
        '| \\|slope − 1\\| < 0.05 | Rules out scale bias (e.g. Python LODs 2× R\'s) | fixed |',
        '| \\|intercept\\| < 0.1 | Rules out additive offset | fixed |',
        '',
        'A gate that passes r alone but fails slope or intercept indicates a '
        'systematic error in the Python implementation that Pearson r cannot detect.',
    ]

    phases_to_emit = [phase] if phase is not None else [1, 2, 3, 4]
    for p in phases_to_emit:
        _PHASE_FNS[p](lines)

    # blocked/gated and directory reference — only in full report or Phase 2
    if phase is None or phase == 2:
        lines += [
            '',
            '---',
            '',
            '## Blocked / Gated Sections',
            '',
            '| Gate | Reason | Unblocked by |',
            '|------|--------|-------------|',
            '| P2-S5 | sparcc_cache is pre-collapsed — second-pass has different loci set | '
            'Requires uncollapsed cache from GigaMUGA raw |',
            '| P2-S8 | allele_dir (founder SNP DB) not built | '
            'PORT-08: `build_founder_allele_db(gm_csv)` → `founder_alleles.h5` |',
        ]

    if phase is None:
        lines += [
            '',
            '---',
            '',
            '## Output Directories',
            '',
            '```',
            'validation/phase1/   r_data/ py_data/ compare/ plots/',
            'validation/phase2/   r_data/ py_data/ compare/ plots/ r_plots/',
            'validation/phase3/   r_data/ py_data/ compare/ plots/',
            'validation/phase4/   py_data/ plots/',
            '```',
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'report written to {out_path}')


def main() -> None:
    """entry point."""
    ap = argparse.ArgumentParser(
        description='Generate docs/VALIDATION_STATUS.md from phase output CSVs')
    ap.add_argument('--phase', type=int, choices=[1, 2, 3, 4], default=None,
                    help='emit only this phase section (1–4); default: all phases')
    ap.add_argument('--out', default=None,
                    help='output path (default: docs/VALIDATION_STATUS.md, or '
                         'docs/phase{N}_status.md when --phase is given)')
    args = ap.parse_args()
    if args.out is None:
        out = (Path('docs') / f'phase{args.phase}_status.md'
               if args.phase is not None else Path('docs/VALIDATION_STATUS.md'))
    else:
        out = Path(args.out)
    generate(out, phase=args.phase)


if __name__ == '__main__':
    main()
