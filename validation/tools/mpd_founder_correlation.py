"""mpd_founder_correlation.py — 8-founder correlation triangle: GWAS × eQTL effects × MPD phenotype.

For each eGene in the GWAS–eQTL overlap (gwas_liftover_hits_Lw1.csv), correlates the
8-founder allele effects from eqtl_peak with the 8-founder strain means from the Mouse
Phenome Database (MPD).  Produces an 8-point Pearson + Spearman correlation per
(gene × cell_type × MPD_measure × sex) tuple.

n=8 founders means p-values are low-power (~0.6 critical r for α=0.05, two-tailed).
Direction and effect size are the primary outputs; p-values are informational only.

Triangle rationale
------------------
    GWAS hit (human) → liftOver → mm39 eQTL peak
    eQTL peak     → 8 founder allele effects (effect_AJ … effect_WSBJ)
    Founder strains → MPD phenotype means (same 8 founders)

    If the eQTL drives the GWAS trait, allele effects and phenotype means
    should be correlated across the same 8 founders.

Vectorization
-------------
    Outer loop: n_measures (≈ 3k–5k unique study×varname×sex combos) — one pass.
    Inner: numpy matrix operation over all (gene, CT) pairs at once — no Python loop
    over pairs.  Spearman ranks computed via argsort; no scipy call per pair.
    Pairs with NaN effects fall back to a per-pair scalar path (rare).

Inputs
------
    gwas_liftover_hits:  validation/ctxhipp_data/bioground/gwas_overlap/gwas_liftover_hits_{tag}.csv
    eqtl_peak:           validation/ctxhipp_data/submission/{ct}_{tag}/eqtl_peak_{ct}_{tag}.csv
    strain_means:        validation/mpd/strain_means/*.csv  (is_founder=True rows)

Output
------
    validation/mpd/{tissue}/mpd_founder_corr_{tissue}_{tag}.csv
    validation/mpd/{tissue}/mpd_founder_corr_{tissue}_{tag}_report/
        mpd_corr_summary_{tissue}_{tag}.txt    plain-text run summary + top-N table
        mpd_corr_top_hits_{tissue}_{tag}.csv   top-N rows by |r_pearson|
        mpd_corr_fdr_hits_{tissue}_{tag}.csv   rows with q_pearson < 0.05
        report.png                              3-panel figure: r distribution | Pearson vs Spearman | top-N bar

Columns (main CSV, in order)
-----------------------------
    gene_id, gene_symbol, cell_type,
    study, varname, measnum, sex,
    n_founders,
    r_pearson, p_pearson, r_spearman, p_spearman

Usage
-----
    conda run -n bioinfo python validation/tools/mpd_founder_correlation.py
    conda run -n bioinfo python validation/tools/mpd_founder_correlation.py --tag Lw1
    conda run -n bioinfo python validation/tools/mpd_founder_correlation.py --min-founders 6 --top-n 30
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, rankdata, spearmanr, t as t_dist

# ── repo root (two levels up from validation/tools/) ──────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VAL_DIR   = _REPO_ROOT / 'validation'
_SRC       = _REPO_ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mosaiqtl.genotype.cc import FOUNDER_COLORS  # noqa: E402 (after sys.path fix)

# 8 CC founders in canonical order (matches eqtl_peak effect_* columns)
_FOUNDERS: list[str] = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
_EFFECT_COLS: list[str] = [f'effect_{f}' for f in _FOUNDERS]

log = logging.getLogger(__name__)


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_gwas_genes(gwas_path: Path) -> set[str]:
    """Return the set of gene_ids with a GWAS overlap."""
    df = pd.read_csv(gwas_path, dtype={'gene_id': str})
    genes = set(df['gene_id'].dropna().unique())
    log.info('[load] GWAS liftover: %d unique gene_ids ← %s', len(genes), gwas_path.name)
    return genes


def _load_eqtl_peaks(submission_dir: Path, tag: str, gwas_genes: set[str]) -> pd.DataFrame:
    """Load eqtl_peak CSVs for all {ct}_{tag} subdirs; keep GWAS genes only.

    Returns DataFrame: gene_id, gene_symbol, cell_type, effect_AJ … effect_WSBJ.
    """
    frames: list[pd.DataFrame] = []
    ct_dirs = sorted(d for d in submission_dir.iterdir() if d.name.endswith(f'_{tag}'))
    log.info('[load] scanning %d submission dirs for tag=%s ...', len(ct_dirs), tag)
    for ct_dir in ct_dirs:
        ct = ct_dir.name[: -(len(tag) + 1)]
        peak_csv = ct_dir / f'eqtl_peak_{ct_dir.name}.csv'
        if not peak_csv.exists():
            log.warning('  [skip] %s: eqtl_peak not found', ct_dir.name)
            continue
        df = pd.read_csv(peak_csv, dtype={'gene_id': str})
        present = [c for c in _EFFECT_COLS if c in df.columns]
        if len(present) < 8:
            log.warning('  [warn] %s: only %d/8 founder effect cols', ct, len(present))
        keep = ['gene_id', 'gene_symbol'] + present
        df = df[keep][df['gene_id'].isin(gwas_genes)].copy()
        df.insert(0, 'cell_type', ct)
        frames.append(df)
        log.info('  [CT] %-40s %3d GWAS eGenes', ct, len(df))

    if not frames:
        log.error('[load] no eqtl_peak files found under %s', submission_dir)
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    log.info('[load] eqtl_peak: %d (gene, CT) pairs across %d CTs',
             len(out), out['cell_type'].nunique())
    return out


def _load_mpd_founder_means(strain_means_dir: Path) -> pd.DataFrame:
    """Load all MPD strain-means CSVs; keep is_founder rows.

    Returns DataFrame: study, varname, measnum, sex, effect_col, mean.
    effect_col ∈ {AJ, B6J, 129S1J, NODJ, NZOJ, CASTJ, PWKJ, WSBJ}.
    """
    frames: list[pd.DataFrame] = []
    csvs = sorted(strain_means_dir.glob('*.csv'))
    log.info('[load] MPD strain means: %d study files ...', len(csvs))
    for csv_file in csvs:
        df = pd.read_csv(csv_file, low_memory=False)
        founder_rows = df[df['is_founder'].astype(str).str.lower() == 'true'].copy()
        if founder_rows.empty:
            continue
        founder_rows = founder_rows[
            founder_rows['effect_col'].notna() & (founder_rows['effect_col'] != '')
        ].copy()
        founder_rows['study'] = csv_file.stem
        keep = [c for c in ('study', 'varname', 'measnum', 'sex', 'effect_col', 'mean')
                if c in founder_rows.columns]
        frames.append(founder_rows[keep])

    if not frames:
        log.error('[load] no MPD strain-means files found at %s', strain_means_dir)
        sys.exit(1)

    mpd = pd.concat(frames, ignore_index=True)
    mpd['measnum'] = pd.to_numeric(mpd['measnum'], errors='coerce').astype('Int64')
    mpd['mean'] = pd.to_numeric(mpd['mean'], errors='coerce')
    mpd = mpd[mpd['effect_col'].isin(_FOUNDERS)]
    log.info('[load] MPD: %d founder rows | %d unique varnames | %d studies',
             len(mpd), mpd['varname'].nunique(), mpd['study'].nunique())
    return mpd


# ── vectorised correlation engine ─────────────────────────────────────────────

def _pearson_vec(Y: np.ndarray, x: np.ndarray, df: int) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r and two-tailed p between each row of Y (n_pairs × k) and x (k,).

    Y rows must all be finite (checked by caller).
    """
    x_c = x - x.mean()
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    denom_x = np.sqrt((x_c ** 2).sum())
    denom_Y = np.sqrt((Y_c ** 2).sum(axis=1))          # (n_pairs,)
    safe = (denom_Y > 0) & (denom_x > 0)
    r = np.full(len(Y), np.nan)
    r[safe] = (Y_c[safe] @ x_c) / (denom_x * denom_Y[safe])
    r = np.clip(r, -1.0, 1.0)
    # t = r * sqrt(df / (1 - r²)) — clip r² away from 1 to avoid /0
    t_stat = r * np.sqrt(df / np.maximum(1 - r ** 2, 1e-12))
    p = 2.0 * t_dist.sf(np.abs(t_stat), df)
    return r, p


def _spearman_vec(Y: np.ndarray, x: np.ndarray, df: int) -> tuple[np.ndarray, np.ndarray]:
    """Spearman r and two-tailed p: rank x and rows of Y, then Pearson on ranks.

    Y rows must all be finite (checked by caller).
    """
    # rank each row of Y independently; break ties with average rank
    def _rank_rows(M: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(rankdata, 1, M)

    x_rank = _rank_rows(x[np.newaxis, :])[0]           # (k,)
    Y_rank = _rank_rows(Y)                              # (n_pairs, k)
    return _pearson_vec(Y_rank, x_rank, df)


def _corr_scalar(x_sub: np.ndarray, y_sub: np.ndarray) -> tuple[int, float, float, float, float]:
    """Fallback scalar correlation (used when a row has NaN effects)."""
    n = len(x_sub)
    if n < 4:
        nan = float('nan')
        return n, nan, nan, nan, nan
    r_p, p_p = pearsonr(x_sub, y_sub)
    r_s, p_s = spearmanr(x_sub, y_sub)
    return n, float(r_p), float(p_p), float(r_s), float(p_s)


def _compute_correlations(
    eqtl_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
    min_founders: int,
) -> pd.DataFrame:
    """Vectorised correlation: outer loop over MPD measures, matrix ops over all pairs.

    For each MPD (study, varname, measnum, sex):
      1. Get the k ≤ 8 founders with finite means.
      2. Slice the effect matrix to those k columns.
      3. Split pairs into:
         - fully-finite rows → vectorised Pearson + Spearman (numpy only).
         - rows with NaN effects → scalar fallback (rare).
    """
    t0 = time.perf_counter()

    # effect matrix: (n_pairs, 8) — preserve row order of eqtl_df
    eff_matrix = eqtl_df[_EFFECT_COLS].values.astype(float)  # (n_pairs, 8)
    n_pairs = len(eff_matrix)

    # meta arrays for fast indexing
    gene_ids = eqtl_df['gene_id'].to_numpy()
    gene_syms = eqtl_df['gene_symbol'].to_numpy()
    cell_types = eqtl_df['cell_type'].to_numpy()

    # build measure table: list of (key_tuple, pheno_vec_8)
    log.info('[corr] building MPD measure index ...')
    measures: list[tuple[tuple, np.ndarray]] = []
    # (study, varname, measnum) → list of per-sex pheno vecs, for combined entry below
    _measure_vecs: dict[tuple, list[np.ndarray]] = {}
    for key, grp in mpd_df.groupby(['study', 'varname', 'measnum', 'sex'], sort=False):
        fmap = dict(zip(grp['effect_col'], grp['mean']))
        pheno = np.array([fmap.get(f, np.nan) for f in _FOUNDERS], dtype=float)
        measures.append((key, pheno))
        base_key = key[:3]  # (study, varname, measnum)
        _measure_vecs.setdefault(base_key, []).append(pheno)

    # Add combined-sex entry for any measure that has ≥2 sex strata.
    # Founder means are nanmean'd across sexes: a founder measured in only one
    # sex still contributes; a founder missing in all sexes stays NaN.
    n_combined = 0
    for (study, varname, measnum), vecs in _measure_vecs.items():
        if len(vecs) >= 2:
            combined_pheno = np.nanmean(np.stack(vecs, axis=0), axis=0)
            measures.append(((study, varname, measnum, 'combined'), combined_pheno))
            n_combined += 1
    log.info('[corr] %d sex-stratified + %d combined measures', len(measures) - n_combined, n_combined)

    n_measures = len(measures)
    log.info('[corr] %d (gene, CT) pairs × %d MPD measures — starting vectorised pass ...',
             n_pairs, n_measures)

    records: list[dict] = []
    n_scalar_fallback = 0
    report_every = max(1, n_measures // 20)           # progress ~every 5 %

    for j, ((study, varname, measnum, sex), pheno_vec) in enumerate(measures):

        if j % report_every == 0:
            pct = 100 * j / n_measures
            elapsed = time.perf_counter() - t0
            eta = elapsed / max(j, 1) * (n_measures - j)
            log.info('[corr] measure %d / %d  (%.0f %%)  elapsed %.1fs  ETA %.0fs',
                     j, n_measures, pct, elapsed, eta)

        # which founders have a finite phenotype value?
        mask_m = np.isfinite(pheno_vec)               # (8,) bool
        k = int(mask_m.sum())
        if k < min_founders:
            continue

        x = pheno_vec[mask_m]                         # (k,)
        Y_sub = eff_matrix[:, mask_m]                 # (n_pairs, k)

        # per-pair finite status
        finite_Y = np.isfinite(Y_sub)                 # (n_pairs, k)
        n_per_pair = finite_Y.sum(axis=1)             # (n_pairs,) — count of usable founders
        enough = n_per_pair >= min_founders            # (n_pairs,) bool

        if not enough.any():
            continue

        # ── case A: all k founders finite for this pair → fully vectorised ──
        all_finite = finite_Y.all(axis=1) & enough    # (n_pairs,)
        if all_finite.any():
            idx_A = np.where(all_finite)[0]
            Y_A = Y_sub[idx_A]                        # (nA, k)
            df_val = k - 2
            r_p_A, p_p_A = _pearson_vec(Y_A, x, df_val)
            r_s_A, p_s_A = _spearman_vec(Y_A, x, df_val)
            for ii, gi in enumerate(idx_A):
                records.append({
                    'gene_id':    gene_ids[gi],
                    'gene_symbol': gene_syms[gi],
                    'cell_type':  cell_types[gi],
                    'study':      study,
                    'varname':    varname,
                    'measnum':    measnum,
                    'sex':        sex,
                    'n_founders': k,
                    'r_pearson':  float(r_p_A[ii]),
                    'p_pearson':  float(p_p_A[ii]),
                    'r_spearman': float(r_s_A[ii]),
                    'p_spearman': float(p_s_A[ii]),
                })

        # ── case B: per-pair fallback — some effects are NaN ────────────────
        idx_B = np.where(enough & ~all_finite)[0]
        for gi in idx_B:
            row_mask = finite_Y[gi]                   # (k,) bool
            n_eff = int(row_mask.sum())
            n_scalar_fallback += 1
            n_out, r_p_i, p_p_i, r_s_i, p_s_i = _corr_scalar(
                x[row_mask], Y_sub[gi][row_mask]
            )
            records.append({
                'gene_id':    gene_ids[gi],
                'gene_symbol': gene_syms[gi],
                'cell_type':  cell_types[gi],
                'study':      study,
                'varname':    varname,
                'measnum':    measnum,
                'sex':        sex,
                'n_founders': n_out,
                'r_pearson':  r_p_i,
                'p_pearson':  p_p_i,
                'r_spearman': r_s_i,
                'p_spearman': p_s_i,
            })

    elapsed = time.perf_counter() - t0
    log.info('[corr] done — %d correlations in %.1fs  (%d scalar fallbacks)',
             len(records), elapsed, n_scalar_fallback)
    out = pd.DataFrame(records)
    return out


# ── report generator ──────────────────────────────────────────────────────────
#
# Four output files per run (alongside the usual .txt and .csv):
#   mpd_corr_overview_{tissue}_{tag}.png   — KDE r-distribution + LOO dumbbell
#   mpd_corr_spiders_{tissue}_{tag}.png    — K=6 per-founder radar gallery
#   mpd_corr_parallel_{tissue}_{tag}.png   — parallel coordinates (founders as lines)
#   mpd_corr_sankey_{tissue}_{tag}.html    — interactive plotly Sankey
#
# GWAS trait is gene-level (human GWAS association), not per-founder.
# It enters only via: Sankey right-side nodes and spider subtitle annotations.
# ─────────────────────────────────────────────────────────────────────────────

_SEX_SYM: dict[str, str] = {'m': '♂', 'f': '♀', 'both': '⚥', 'combined': '(m+f)'}

# ── MPD varname formatter ─────────────────────────────────────────────────────
# Rules: split on '_', split CamelCase at UPPER→lower boundary, look up in
# prefix / location tables, keep pure-uppercase acronyms as-is.
# Every mapping is written to the run's summary.txt audit table for review.

_MPD_PREFIX: dict[str, str] = {
    'FAT': 'Fat mass',     'LEAN': 'Lean mass',    'BW': 'Body weight',
    'BMI': 'BMI',          'HDL': 'HDL-C',         'LDL': 'LDL-C',
    'CHOL': 'Cholesterol', 'TG': 'Triglycerides',  'TRIG': 'Triglycerides',
    'GLU': 'Glucose',      'GLC': 'Glucose',        'INS': 'Insulin',
    'HOMA': 'HOMA-IR',     'ACT': 'Activity',       'VO2': 'VO₂',
    'VCO2': 'VCO₂',        'RQ': 'Resp. quotient',  'EE': 'Energy expenditure',
    'FOOD': 'Food intake', 'WATER': 'Water intake', 'BONE': 'Bone mass',
    'BMD': 'Bone mineral density',
    'RBC': 'RBC',          'WBC': 'WBC',            'HGB': 'Haemoglobin',
    'HCT': 'Haematocrit',  'PLT': 'Platelets',
    'ALT': 'ALT',          'AST': 'AST',
    'CREAT': 'Creatinine', 'UREA': 'Urea',
    'PROT': 'Protein',     'ALB': 'Albumin',
}
_MPD_LOCATION: dict[str, str] = {
    'wholebody': 'whole body', 'plasma': 'plasma',   'serum': 'serum',
    'liver': 'liver',          'muscle': 'muscle',   'adipose': 'adipose',
    'brain': 'brain',          'heart': 'heart',     'kidney': 'kidney',
    'lung': 'lung',            'spleen': 'spleen',
}


def _fmt_varname(varname: str) -> str:
    """Heuristic MPD varname → human-readable label.

    Transparent rules applied in order so the audit table is interpretable.
    Extend _MPD_PREFIX / _MPD_LOCATION for any unrecognised tokens.
    """
    parts: list[str] = []
    for seg in varname.split('_'):
        # 'FATmass' → ['FAT', 'mass'] at uppercase-run → lowercase boundary
        tokens = re.sub(r'([A-Z]{2,})([a-z])', r'\1 \2', seg).split()
        for tok in tokens:
            up = tok.upper()
            if up in _MPD_PREFIX:
                parts.append(_MPD_PREFIX[up])
            elif tok.lower() in _MPD_LOCATION:
                parts.append(_MPD_LOCATION[tok.lower()])
            elif re.match(r'^[A-Z]{2,}[0-9]*$', tok):
                parts.append(tok)           # pure-uppercase acronym: keep as-is
            else:
                parts.append(tok.lower())
    result = ' '.join(parts)
    return (result[0].upper() + result[1:]) if result else varname


# ── viz theme & color grammar ─────────────────────────────────────────────────
#
# Single source: Tableau 10 as published by Tableau Inc.
# NOTE: differs from matplotlib's built-in 'tab10' (e.g. tab:blue = #1f77b4
# there vs #4e79a7 here).  Index each semantic role explicitly so provenance
# is readable and the whole report suite shares one palette definition.

_TABLEAU10: tuple[str, ...] = (
    '#4e79a7',   # 0  blue
    '#f28e2b',   # 1  orange
    '#e15759',   # 2  red
    '#76b7b2',   # 3  teal
    '#59a14f',   # 4  green
    '#edc948',   # 5  yellow  (also used in gwas_overlap._PARENT_COLORS)
    '#b07aa1',   # 6  purple
    '#ff9da7',   # 7  pink
    '#9c755f',   # 8  brown
    '#bab0ac',   # 9  grey
)

_C_NEG    = _TABLEAU10[0]   # blue   — negative r
_C_DRIVEN = _TABLEAU10[1]   # orange — key-founder driven (wide LOO range)
_C_POS    = _TABLEAU10[2]   # red    — positive r
_C_GWAS   = _TABLEAU10[3]   # teal   — GWAS trait nodes / labels
_C_STABLE = _TABLEAU10[4]   # green  — robust across founders (narrow LOO range)
_C_MPD    = '#444444'       # dark neutral — MPD trait axis labels (not Tableau)
_C_ZERO   = '#cccccc'       # light grey   — zero / reference lines (not Tableau)


def _setup_viz_theme() -> None:
    """Apply project-consistent seaborn / matplotlib theme for all report figures."""
    sns.set_theme(
        style='ticks',
        context='paper',
        font_scale=1.15,
        rc={
            'figure.facecolor': 'white',
            'axes.facecolor':   'white',
            'axes.grid':        False,
            'font.family':      'sans-serif',
            'font.sans-serif':  ['Carlito', 'Calibri', 'Liberation Sans', 'DejaVu Sans'],
            'xtick.direction':  'out',
            'ytick.direction':  'out',
        },
    )


# ── LOO r helper ──────────────────────────────────────────────────────────────

def _compute_loo_r(
    eff_vec: np.ndarray,
    pheno_vec: np.ndarray,
    r_full: float,
    min_k: int = 4,
) -> tuple[float, float, str]:
    """Leave-one-founder-out r: range + identity of the key-driving founder.

    key_founder = the founder whose removal shifts |r| the most from r_full.
    A wide LOO range signals that one founder is carrying the signal (e.g. NZO
    for fat mass) — this is biological, not a flaw; the key_founder label says WHO.
    """
    loo: list[tuple[float, str]] = []
    for i, fname in enumerate(_FOUNDERS):
        mask = np.ones(8, dtype=bool)
        mask[i] = False
        esub = eff_vec[mask]
        psub = pheno_vec[mask]
        fin  = np.isfinite(esub) & np.isfinite(psub)
        if fin.sum() < min_k:
            continue
        r_i, _ = pearsonr(esub[fin], psub[fin])
        loo.append((float(r_i), fname))
    if not loo:
        return np.nan, np.nan, ''
    r_vals = [r for r, _ in loo]
    deltas = [(abs(abs(r_full) - abs(r)), f) for r, f in loo]
    key_f  = max(deltas, key=lambda d: d[0])[1]
    return float(min(r_vals)), float(max(r_vals)), key_f


# ── founder vector helper ─────────────────────────────────────────────────────

def _get_founder_vecs(
    row: pd.Series,
    eqtl_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (eff_vec, pheno_vec) of length 8 for one top-hit row.

    Handles sex='combined' by nanmean-ing across available sex strata.
    """
    esub = eqtl_df[
        (eqtl_df['gene_id']   == row['gene_id']) &
        (eqtl_df['cell_type'] == row['cell_type'])
    ]
    eff_vec = np.array(
        [float(esub.iloc[0].get(f'effect_{f}', np.nan)) for f in _FOUNDERS]
        if not esub.empty else [np.nan] * 8,
        dtype=float,
    )
    if str(row['sex']) == 'combined':
        subs = mpd_df[
            (mpd_df['study']   == row['study'])   &
            (mpd_df['varname'] == row['varname'])  &
            (mpd_df['measnum'] == row['measnum'])
        ]
        vecs = []
        for _, grp in subs.groupby('sex', sort=False):
            fmap = dict(zip(grp['effect_col'], grp['mean']))
            vecs.append([fmap.get(f, np.nan) for f in _FOUNDERS])
        pheno_vec = (np.nanmean(np.array(vecs, dtype=float), axis=0)
                     if vecs else np.full(8, np.nan))
    else:
        msub = mpd_df[
            (mpd_df['study']   == row['study'])   &
            (mpd_df['varname'] == row['varname'])  &
            (mpd_df['measnum'] == row['measnum'])  &
            (mpd_df['sex']     == row['sex'])
        ]
        fmap      = dict(zip(msub['effect_col'], msub['mean'])) if not msub.empty else {}
        pheno_vec = np.array([fmap.get(f, np.nan) for f in _FOUNDERS], dtype=float)
    return eff_vec, pheno_vec


# ── label helper ──────────────────────────────────────────────────────────────

def _hit_labels(row: pd.Series, gene_traits: dict[str, str]) -> tuple[str, str, str]:
    """Return (gene_ct, mpd_str, gwas_str) for one top hit — used in all figures."""
    sex_s   = _SEX_SYM.get(str(row['sex']), f'({row["sex"]})')
    ct_s    = row['cell_type'].replace('_', ' ')
    gene_ct = f'{row["gene_symbol"]} [{ct_s}]'
    mpd_s   = f'MPD: {_fmt_varname(row["varname"])} {sex_s}'
    trait   = gene_traits.get(str(row['gene_id']), '—')
    gwas_s  = f'GWAS: {trait}'
    return gene_ct, mpd_s, gwas_s


# ── figure 1: KDE r-distribution + LOO dumbbell ───────────────────────────────

def _fig_overview(
    corr_df: pd.DataFrame,
    top: pd.DataFrame,
    eqtl_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
    gene_traits: dict[str, str],
    out_dir: Path,
    tag: str,
    tissue: str,
    has_q: bool,
) -> None:
    """KDE of r distribution (gradient-filled) + LOO dumbbell for top hits."""
    _setup_viz_theme()
    n_bar = min(20, len(top))
    fig, axes = plt.subplots(
        1, 2, figsize=(20, max(7, n_bar * 0.58 + 2)),
        gridspec_kw={'width_ratios': [1, 2.8]},
    )
    fig.suptitle(
        f'MPD–eQTL correlation overview  |  {tissue}  {tag}',
        fontsize=13, fontweight='bold',
    )

    # ── left: gradient-filled KDE ─────────────────────────────────────────────
    ax = axes[0]
    r_vals = corr_df['r_pearson'].dropna().values
    kde   = gaussian_kde(r_vals, bw_method='scott')
    xg    = np.linspace(-1, 1, 300)
    ykde  = kde(xg)
    cmap  = plt.get_cmap('RdBu_r')

    # fill under KDE with a red↔blue gradient by r value
    for i in range(len(xg) - 1):
        c = cmap((xg[i] + 1) / 2)          # map [-1,1] → [0,1] for colormap
        ax.fill_between(xg[i:i+2], 0, ykde[i:i+2], color=c, alpha=0.55, lw=0)
    ax.plot(xg, ykde, color='#222222', lw=1.6, zorder=3)

    # rug: small ticks at each observation (subsample if huge)
    r_rug = r_vals if len(r_vals) <= 5000 else np.random.default_rng(42).choice(
        r_vals, 5000, replace=False)
    ax.plot(r_rug, np.full_like(r_rug, -ykde.max() * 0.05),
            '|', color='#555555', alpha=0.25, ms=3.5, zorder=4)

    for xv, ls in [(-0.6, ':'), (0.0, '--'), (0.6, ':')]:
        ax.axvline(xv, color=_C_ZERO, lw=0.9, ls=ls, zorder=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(bottom=-ykde.max() * 0.09)
    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('r distribution', fontsize=12)
    sns.despine(ax=ax)

    # ── right: LOO dumbbell ───────────────────────────────────────────────────
    ax = axes[1]
    bar_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    dumbbell_rows: list[dict] = []
    for _, row in top.head(n_bar).iterrows():
        eff_vec, pheno_vec = _get_founder_vecs(row, eqtl_df, mpd_df)
        r_min, r_max, key_f = _compute_loo_r(eff_vec, pheno_vec, float(row['r_pearson']))
        gene_ct, mpd_s, gwas_s = _hit_labels(row, gene_traits)
        dumbbell_rows.append({
            'label':       f'{gene_ct}\n{mpd_s}\n{gwas_s}',
            'r_full':      float(row['r_pearson']),
            'r_min':       r_min,
            'r_max':       r_max,
            'key_founder': key_f,
        })

    for i, d in enumerate(dumbbell_rows):
        rfull = d['r_full']
        r0, r1 = d['r_min'], d['r_max']
        span  = abs(float(r1) - float(r0)) if (np.isfinite(r0) and np.isfinite(r1)) else 0.0
        c_line = _C_DRIVEN if span > 0.4 else _C_STABLE
        c_dot  = _C_POS if rfull > 0 else _C_NEG

        # LOO range line + end ticks
        if np.isfinite(r0) and np.isfinite(r1):
            ax.plot([r0, r1], [i, i], color=c_line, lw=2.8, alpha=0.5,
                    solid_capstyle='round', zorder=2)
            ax.plot([r0, r1], [i, i], '|', color=c_line, ms=7, mew=2, zorder=2)
        # r_full filled dot
        ax.scatter(rfull, i, color=c_dot, s=95, zorder=5,
                   edgecolors='white', linewidths=0.9)
        # key-founder label (right margin, coloured by FOUNDER_COLORS)
        kf = d['key_founder']
        if kf:
            ax.text(1.015, i, kf, va='center', ha='left', fontsize=7.5,
                    color=FOUNDER_COLORS.get(kf, '#888888'), fontweight='bold',
                    transform=bar_trans)

    ax.axvline(0, color=_C_ZERO, lw=0.8, ls='--', zorder=1)
    ax.set_yticks(range(n_bar))
    ax.set_yticklabels([d['label'] for d in dumbbell_rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Pearson r   (● = r_full  ── = leave-one-out range)', fontsize=10)
    ax.set_title(
        f'Top {n_bar}  |  ● = r_full  '
        '  orange line = key-founder driven, green = robust',
        fontsize=9.5,
    )
    ax.set_xlim(-1.12, 1.12)

    ax.legend(handles=[
        mpatches.Patch(color=_C_DRIVEN, label='Key-founder driven  (LOO span > 0.4)'),
        mpatches.Patch(color=_C_STABLE, label='Robust across founders  (LOO span ≤ 0.4)'),
        mpatches.Patch(color=_C_POS,    label='r > 0'),
        mpatches.Patch(color=_C_NEG,    label='r < 0'),
    ], fontsize=8, frameon=False, loc='lower right')
    sns.despine(ax=ax)

    fig.tight_layout()
    path = out_dir / f'mpd_corr_overview_{tissue}_{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info('[report] %s written', path.name)


# ── figure 2: spider / radar gallery ──────────────────────────────────────────

def _norm_0_1(v: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1]; returns zeros if degenerate."""
    fin = v[np.isfinite(v)]
    if len(fin) < 2 or fin.max() == fin.min():
        return np.zeros_like(v, dtype=float)
    return (v - fin.min()) / (fin.max() - fin.min())


def _fig_spiders(
    top: pd.DataFrame,
    eqtl_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
    gene_traits: dict[str, str],
    out_dir: Path,
    tag: str,
    tissue: str,
    k: int = 6,
) -> None:
    """K=6 radar gallery: per-founder eQTL vs MPD overlap reveals who drives the signal.

    Each subplot shows two normalised polygons on 8 axes (one per founder):
      - eQTL allele effect polygon (filled, colored by r sign)
      - MPD founder mean polygon  (dashed outline + light fill, neutral)
    Overlap = positive contribution from that founder.  A single large spike on
    both polygons (e.g. NZO for fat mass) is the biologically driven signal.
    Founder axis labels are coloured by FOUNDER_COLORS.
    """
    _setup_viz_theme()
    k = min(k, len(top))
    ncols = 3
    nrows = (k + ncols - 1) // ncols
    fig   = plt.figure(figsize=(5.8 * ncols, 5.5 * nrows))
    fig.suptitle(
        f'Per-founder contribution — top {k} hits  |  {tissue}  {tag}',
        fontsize=13, fontweight='bold',
    )

    angles_base = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles      = np.concatenate([angles_base, [angles_base[0]]])  # close polygon

    for idx, (_, row) in enumerate(top.head(k).iterrows()):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='polar')
        eff_vec, pheno_vec = _get_founder_vecs(row, eqtl_df, mpd_df)
        eff_n   = _norm_0_1(eff_vec)
        pheno_n = _norm_0_1(pheno_vec)

        # close the polygons; replace NaN with 0 so the shape stays closed
        eff_p   = np.nan_to_num(np.concatenate([eff_n,   [eff_n[0]]]),   nan=0.0)
        pheno_p = np.nan_to_num(np.concatenate([pheno_n, [pheno_n[0]]]), nan=0.0)

        r_val    = float(row['r_pearson'])
        col_eqtl = _C_POS if r_val >= 0 else _C_NEG

        ax.plot(angles, eff_p,   color=col_eqtl, lw=2.0, zorder=3)
        ax.fill(angles, eff_p,   color=col_eqtl, alpha=0.22, zorder=2)
        ax.plot(angles, pheno_p, color=_C_MPD,   lw=1.4, ls='--', zorder=3)
        ax.fill(angles, pheno_p, color=_C_MPD,   alpha=0.10, zorder=2)

        # founder axis labels coloured by FOUNDER_COLORS
        ax.set_thetagrids(np.degrees(angles_base), labels=[])
        for ang, fname in zip(angles_base, _FOUNDERS):
            ax.text(ang, 1.28, fname,
                    ha='center', va='center', fontsize=8,
                    color=FOUNDER_COLORS.get(fname, '#888888'),
                    fontweight='bold')

        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([])
        ax.tick_params(pad=0)
        ax.spines['polar'].set_visible(False)
        ax.grid(color='#e0e0e0', lw=0.5)

        gene_ct, mpd_s, gwas_s = _hit_labels(row, gene_traits)
        q_s = f'  q={row["q_pearson"]:.1e}' if 'q_pearson' in row.index else ''
        ax.set_title(
            f'{gene_ct}   r={r_val:+.2f}{q_s}',
            fontsize=9, fontweight='bold', pad=32,
        )
        ax.text(0.5, -0.10, mpd_s,  ha='center', va='top', fontsize=8,
                color=_C_MPD,  transform=ax.transAxes)
        ax.text(0.5, -0.18, gwas_s, ha='center', va='top', fontsize=8,
                color=_C_GWAS, style='italic', transform=ax.transAxes)

    fig.legend(
        handles=[
            mpatches.Patch(color=_C_POS,  alpha=0.4, label='eQTL effect (r > 0)'),
            mpatches.Patch(color=_C_NEG,  alpha=0.4, label='eQTL effect (r < 0)'),
            mpatches.Patch(color=_C_MPD,  alpha=0.2, label='MPD founder mean'),
        ],
        fontsize=9, frameon=False, loc='lower center', ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = out_dir / f'mpd_corr_spiders_{tissue}_{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info('[report] %s written', path.name)


# ── figure 3: parallel coordinates ────────────────────────────────────────────

def _fig_parallel_coords(
    top: pd.DataFrame,
    eqtl_df: pd.DataFrame,
    mpd_df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    tissue: str,
) -> None:
    """Parallel coordinates: 8 founders as lines across [eQTL effect, top-3 MPD traits].

    Shows how each founder's eQTL allele effect co-varies with multiple MPD phenotypes
    simultaneously.  Parallel lines = correlated; crossing lines = anti-correlated.
    A single line that diverges while others stay flat = that founder drives the signal.
    GWAS trait is not an axis (it is not per-founder); it appears via the Sankey only.
    """
    _setup_viz_theme()
    best     = top.iloc[0]
    eff_vec, _ = _get_founder_vecs(best, eqtl_df, mpd_df)

    # top-3 MPD traits by |r| for the best-hit gene (any sex)
    gene_top = (
        top[top['gene_id'] == best['gene_id']]
        .assign(_ar=lambda d: d['r_pearson'].abs())
        .sort_values('_ar', ascending=False)
        .drop(columns='_ar')
        .head(3)
    )
    if len(gene_top) == 0:
        gene_top = top.head(3)

    axis_labels: list[str] = [f'eQTL\n{best["gene_symbol"]}']
    axis_data:   list[np.ndarray] = [eff_vec.copy()]
    for _, hr in gene_top.iterrows():
        _, pv = _get_founder_vecs(hr, eqtl_df, mpd_df)
        sex_s = _SEX_SYM.get(str(hr['sex']), f'({hr["sex"]})')
        axis_labels.append(f'{_fmt_varname(hr["varname"])}\n{sex_s}')
        axis_data.append(pv.copy())

    n_axes    = len(axis_data)
    axis_norm = [_norm_0_1(col) for col in axis_data]

    fig, ax = plt.subplots(figsize=(3.8 * n_axes, 6))
    fig.suptitle(
        f'Parallel coordinates — founders across eQTL × MPD traits\n'
        f'{tissue}  {tag}  |  gene: {best["gene_symbol"]}',
        fontsize=12, fontweight='bold',
    )

    x_pos = np.arange(n_axes)
    # vertical axis lines
    for xi in x_pos:
        ax.axvline(xi, color='#d0d0d0', lw=1.0, zorder=1)
        raw = axis_data[xi]
        fin = raw[np.isfinite(raw)]
        if len(fin):
            ax.text(xi, -0.07, f'{fin.min():.2g}', ha='center', va='top',
                    fontsize=7.5, color='#999999', transform=ax.get_xaxis_transform())
            ax.text(xi,  1.04, f'{fin.max():.2g}', ha='center', va='bottom',
                    fontsize=7.5, color='#999999', transform=ax.get_xaxis_transform())

    for fi, fname in enumerate(_FOUNDERS):
        vals = np.array([an[fi] for an in axis_norm])
        if not np.any(np.isfinite(vals)):
            continue
        col = FOUNDER_COLORS.get(fname, '#aaaaaa')
        ax.plot(x_pos, vals, color=col, lw=2.4, alpha=0.88,
                solid_capstyle='round', zorder=3)
        ax.scatter(x_pos, vals, color=col, s=75, zorder=4,
                   edgecolors='white', linewidths=0.8)
        last_fin = next((j for j in range(n_axes - 1, -1, -1)
                         if np.isfinite(vals[j])), None)
        if last_fin is not None:
            ax.text(last_fin + 0.07, vals[last_fin], f' {fname}',
                    va='center', fontsize=8.5, color=col, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(axis_labels, fontsize=9.5, ha='center')
    ax.set_xlim(-0.35, n_axes - 0.35)
    ax.set_ylim(-0.18, 1.15)
    ax.set_yticks([])
    ax.set_ylabel('Normalised value  (0–1 per axis)', fontsize=10)
    sns.despine(ax=ax, left=True, bottom=False)

    fig.tight_layout()
    path = out_dir / f'mpd_corr_parallel_{tissue}_{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info('[report] %s written', path.name)


# ── figure 4: interactive Sankey (plotly) ─────────────────────────────────────

def _fig_sankey(
    top: pd.DataFrame,
    gene_traits: dict[str, str],
    out_dir: Path,
    tag: str,
    tissue: str,
) -> None:
    """Interactive Sankey: eQTL gene → MPD trait → GWAS trait.

    Flow width = |r_pearson|.  Warm flows = positive r; cool = negative.
    GWAS traits are the right-side nodes — the triangle closure check is visual:
    does the flow from a gene reach a GWAS node that is biologically related to
    the MPD trait it passes through?
    Saved as an HTML file (plotly CDN-linked, no offline dep).
    """
    try:
        import plotly.graph_objects as go                # noqa: PLC0415
    except ImportError:
        log.warning('[report] plotly not available — skipping Sankey (pip install plotly)')
        return

    gene_syms = list(top['gene_symbol'].unique())
    mpd_set   = sorted({_fmt_varname(v) for v in top['varname']})
    gwas_set  = sorted({g for g in
                        (gene_traits.get(str(gid), '') for gid in top['gene_id'].unique())
                        if g and g != '—'})

    all_labels = gene_syms + mpd_set + gwas_set
    idx        = {lbl: i for i, lbl in enumerate(all_labels)}

    sources, targets, values, link_cols = [], [], [], []

    for _, row in top.iterrows():
        gsym  = row['gene_symbol']
        mpd_l = _fmt_varname(row['varname'])
        gwas_l = gene_traits.get(str(row['gene_id']), '—')
        r     = float(row['r_pearson'])
        col   = 'rgba(225,87,89,0.40)' if r > 0 else 'rgba(78,121,167,0.40)'
        absr  = abs(r)

        if gsym in idx and mpd_l in idx:
            sources.append(idx[gsym]);  targets.append(idx[mpd_l])
            values.append(absr);        link_cols.append(col)
        if gwas_l and gwas_l != '—' and mpd_l in idx and gwas_l in idx:
            sources.append(idx[mpd_l]); targets.append(idx[gwas_l])
            values.append(absr);        link_cols.append(col)

    node_x = (
        [0.01] * len(gene_syms) +
        [0.50] * len(mpd_set) +
        [0.99] * len(gwas_set)
    )
    node_colors = (
        [_C_POS] * len(gene_syms) +
        ['#aaaaaa'] * len(mpd_set) +
        [_C_GWAS] * len(gwas_set)
    )

    fig = go.Figure(data=[go.Sankey(
        arrangement='freeform',
        node=dict(
            pad=14, thickness=18,
            line=dict(color='white', width=0.3),
            label=all_labels, x=node_x, color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color=link_cols),
    )])
    n_max = max(len(gene_syms), len(mpd_set), len(gwas_set), 1)
    fig.update_layout(
        title=dict(
            text=(f'eQTL gene → MPD trait → GWAS trait  |  {tissue}  {tag}<br>'
                  f'<sup>Flow width = |r_Pearson|  ·  warm = r > 0  ·  cool = r < 0'
                  f'  ·  teal nodes = GWAS traits</sup>'),
            font=dict(size=14),
        ),
        font=dict(family='Carlito, Calibri, sans-serif', size=11),
        paper_bgcolor='white',
        height=max(500, 30 * n_max),
    )
    path = out_dir / f'mpd_corr_sankey_{tissue}_{tag}.html'
    fig.write_html(str(path), include_plotlyjs='cdn')
    log.info('[report] %s written', path.name)


# ── orchestrator ──────────────────────────────────────────────────────────────

def _write_report(
    corr_df: pd.DataFrame,
    out_dir: Path,
    tag: str,
    tissue: str,
    mpd_df: pd.DataFrame,
    eqtl_df: pd.DataFrame,
    gwas_liftover_df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Write the full MPD–eQTL correlation report into out_dir.

    Outputs:
      mpd_corr_summary_{tissue}_{tag}.txt   — stats + varname audit table
      mpd_corr_top_hits_{tissue}_{tag}.csv  — top N by |r|
      mpd_corr_fdr_hits_{tissue}_{tag}.csv  — FDR-significant hits (if any)
      mpd_corr_overview_{tissue}_{tag}.png  — KDE + LOO dumbbell
      mpd_corr_spiders_{tissue}_{tag}.png   — K=6 founder radar gallery
      mpd_corr_parallel_{tissue}_{tag}.png  — parallel coordinates
      mpd_corr_sankey_{tissue}_{tag}.html   — interactive Sankey
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('[report] writing to %s ...', out_dir)

    r_vals   = corr_df['r_pearson'].dropna()
    r_s_vals = corr_df['r_spearman'].dropna()
    n_total  = len(corr_df)
    n_strong = int((r_vals.abs() >= 0.6).sum())
    n_pos    = int((r_vals >= 0.6).sum())
    n_neg    = int((r_vals <= -0.6).sum())
    has_q    = 'q_pearson' in corr_df.columns
    n_fdr    = int((corr_df['q_pearson'] < 0.05).sum()) if has_q else 0

    # ── gene → GWAS trait(s) ─────────────────────────────────────────────────
    gene_traits: dict[str, str] = {}
    if 'MAPPED_TRAIT' in gwas_liftover_df.columns and 'gene_id' in gwas_liftover_df.columns:
        for gid, grp in gwas_liftover_df.groupby('gene_id', sort=False):
            traits = grp['MAPPED_TRAIT'].dropna().unique()
            parts  = [(str(t)[:40] + '…' if len(str(t)) > 40 else str(t))
                      for t in traits[:2]]
            gene_traits[str(gid)] = '; '.join(parts)

    # ── top hits ──────────────────────────────────────────────────────────────
    top = (
        corr_df
        .assign(_ar=corr_df['r_pearson'].abs())
        .nlargest(top_n, '_ar')
        .drop(columns='_ar')
        .reset_index(drop=True)
    )

    # ── varname audit ─────────────────────────────────────────────────────────
    fmt_audit = [(v, _fmt_varname(v))
                 for v in sorted(corr_df['varname'].unique())]

    # ── text summary ──────────────────────────────────────────────────────────
    lines = [
        f'mpd_founder_correlation — {tissue}  tag={tag}',
        '=' * 62,
        f'Total triples:               {n_total:>10,}',
        f'Unique genes:                {corr_df["gene_id"].nunique():>10,}',
        f'Unique cell types:           {corr_df["cell_type"].nunique():>10,}',
        f'Unique MPD varnames:         {corr_df["varname"].nunique():>10,}',
        f'|r_Pearson| ≥ 0.6:          {n_strong:>10,}  '
        f'({100 * n_strong / max(n_total, 1):.1f}%)',
        f'  r ≥ +0.6:                 {n_pos:>10,}',
        f'  r ≤ −0.6:                 {n_neg:>10,}',
        (f'FDR q<0.05 (per-measure BH):{n_fdr:>10,}'
         if has_q else '  (no q_pearson — re-run to add FDR)'),
        '',
        f'Pearson r   mean={r_vals.mean():.3f}  '
        f'median={r_vals.median():.3f}  sd={r_vals.std():.3f}',
        f'Spearman r  mean={r_s_vals.mean():.3f}  '
        f'median={r_s_vals.median():.3f}  sd={r_s_vals.std():.3f}',
        '',
        f'Top {top_n} by |r_Pearson|:',
        '-' * 62,
    ]
    for _, row in top.iterrows():
        q_s     = f'  q={row["q_pearson"]:.2e}' if has_q else ''
        trait_s = gene_traits.get(str(row['gene_id']), '')
        lines.append(
            f'  {row["gene_symbol"]:<18}  {row["cell_type"]:<28}  '
            f'{row["varname"]:<22}  {row["sex"]}  '
            f'r={row["r_pearson"]:+.3f}  rs={row["r_spearman"]:+.3f}  '
            f'n={row["n_founders"]}{q_s}'
            + (f'  GWAS: {trait_s}' if trait_s else '')
        )

    if has_q and n_fdr > 0:
        sig_df = (
            corr_df[corr_df['q_pearson'] < 0.05]
            .assign(_ar=lambda d: d['r_pearson'].abs())
            .sort_values('_ar', ascending=False)
            .drop(columns='_ar')
            .reset_index(drop=True)
        )
        lines += ['', f'FDR-significant (q<0.05) — {len(sig_df)} total:', '-' * 62]
        for _, row in sig_df.head(top_n).iterrows():
            trait_s = gene_traits.get(str(row['gene_id']), '')
            lines.append(
                f'  {row["gene_symbol"]:<18}  {row["cell_type"]:<28}  '
                f'{row["varname"]:<22}  {row["sex"]}  '
                f'r={row["r_pearson"]:+.3f}  q={row["q_pearson"]:.2e}  '
                f'n={row["n_founders"]}'
                + (f'  GWAS: {trait_s}' if trait_s else '')
            )
        fdr_name = f'mpd_corr_fdr_hits_{tissue}_{tag}.csv'
        sig_df.to_csv(out_dir / fdr_name, index=False)
        log.info('[report] %s written (%d rows)', fdr_name, len(sig_df))

    lines += [
        '',
        'Varname formatting audit — review these mappings for accuracy:',
        '  (extend _MPD_PREFIX / _MPD_LOCATION in the script for any wrong ones)',
        '-' * 62,
    ]
    for raw, fmt in fmt_audit:
        lines.append(f'  {raw:<32} →  {fmt}')

    (out_dir / f'mpd_corr_summary_{tissue}_{tag}.txt').write_text(
        '\n'.join(lines) + '\n'
    )
    for ln in lines:
        log.info('%s', ln)

    top.to_csv(out_dir / f'mpd_corr_top_hits_{tissue}_{tag}.csv', index=False)
    log.info('[report] mpd_corr_top_hits_%s_%s.csv written (%d rows)',
             tissue, tag, len(top))

    # ── figures ───────────────────────────────────────────────────────────────
    _fig_overview(corr_df, top, eqtl_df, mpd_df, gene_traits,
                  out_dir, tag, tissue, has_q)
    _fig_spiders(top, eqtl_df, mpd_df, gene_traits, out_dir, tag, tissue, k=6)
    _fig_parallel_coords(top, eqtl_df, mpd_df, out_dir, tag, tissue)
    _fig_sankey(top, gene_traits, out_dir, tag, tissue)

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='8-founder correlation: GWAS eGenes × eQTL effects × MPD phenotypes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--tag', required=True,
                   help='Scan tag used in submission dir names (e.g. Lw1).')
    p.add_argument('--tissue', required=True,
                   help='Tissue identifier (e.g. ctxhipp).  Used in liftover filename and output naming.')
    p.add_argument('--gwas-csv', type=Path,
                   help=(
                       'Path to gwas_liftover_hits_{tissue}_{tag}.csv.  '
                       'Default: validation/gwas_overlap/gwas_liftover_hits_{tissue}_{tag}.csv'
                   ))
    p.add_argument('--submission-dir', type=Path,
                   help='Path to submission/.  Default: validation/{tissue}_data/submission/')
    p.add_argument('--strain-means-dir', type=Path,
                   help='Path to MPD strain_means/.  Default: validation/mpd/strain_means/')
    p.add_argument('--out', type=Path,
                   help=(
                       'Output CSV path.  '
                       'Default: validation/mpd/{tissue}/mpd_founder_corr_{tissue}_{tag}.csv'
                   ))
    p.add_argument('--min-founders', type=int, default=6,
                   help='Minimum shared founders with finite data required (default: 6).')
    p.add_argument('--min-abs-r', type=float, default=0.0,
                   help=(
                       'Drop rows with |r_pearson| < this threshold before writing '
                       'the output CSV.  Reduces file size dramatically: 0.5 keeps '
                       '~5%% of rows; 0.707 (critical r at n=8, α=0.05) keeps only '
                       'nominally significant hits.  Default: 0.0 (keep all).'
                   ))
    p.add_argument('--top-n', type=int, default=20,
                   help='Number of top hits in report bar chart (default: 20).')
    p.add_argument('--no-report', action='store_true',
                   help='Skip generating the PNG/text report.')
    p.add_argument('--report-only', action='store_true',
                   help=(
                       'Skip correlation compute; read existing results CSV and '
                       'regenerate report figures only.  Requires the CSV written '
                       'by a previous run (--out path).  '
                       'Reference inputs (--submission-dir, --strain-means-dir, '
                       '--gwas-csv) are still loaded for figure data.'
                   ))
    p.add_argument('--log-level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s',
        datefmt='%H:%M:%S',
    )

    tag    = args.tag
    tissue = args.tissue
    gwas_csv: Path = (
        args.gwas_csv
        or _VAL_DIR / 'gwas_overlap' / f'gwas_liftover_hits_{tissue}_{tag}.csv'
    )
    submission_dir: Path = (
        args.submission_dir or _VAL_DIR / f'{tissue}_data' / 'submission'
    )
    strain_means_dir: Path = args.strain_means_dir or _VAL_DIR / 'mpd' / 'strain_means'
    _mpd_root = _VAL_DIR / 'mpd' / tissue
    out_path: Path = (
        args.out
        or _mpd_root / f'mpd_founder_corr_{tissue}_{tag}.csv'
    )
    report_dir = out_path.parent   # all report files share the tissue dir; names carry tissue+tag

    # ── validate inputs ──────────────────────────────────────────────────────
    for p_check, label in [
        (gwas_csv, 'GWAS liftover CSV'),
        (submission_dir, 'submission/'),
        (strain_means_dir, 'MPD strain_means/'),
    ]:
        if not p_check.exists():
            log.error('[init] %s not found: %s', label, p_check)
            sys.exit(1)

    log.info('=' * 60)
    log.info('mpd_founder_correlation  tissue=%s  tag=%s  min_founders=%d',
             tissue, tag, args.min_founders)
    log.info('  mode           : %s', 'REPORT-ONLY (skipping correlation compute)'
             if args.report_only else 'full run')
    log.info('  gwas_csv       : %s', gwas_csv)
    log.info('  submission_dir : %s', submission_dir)
    log.info('  strain_means   : %s', strain_means_dir)
    log.info('  output         : %s', out_path)
    if not args.report_only:
        log.info('  min_abs_r      : %s',
                 args.min_abs_r if args.min_abs_r > 0 else 'none (keep all)')
    log.info('=' * 60)

    # ── load reference frames ─────────────────────────────────────────────────
    # Always loaded: figures call _get_founder_vecs() which queries eqtl_df and
    # mpd_df at draw time.  gwas_liftover_df supplies MAPPED_TRAIT for Sankey nodes.
    log.info('[load] reading GWAS liftover CSV ...')
    gwas_liftover_df = pd.read_csv(gwas_csv, dtype={'CHR_ID': str})
    gwas_genes: set[str] = set(gwas_liftover_df['gene_id'].dropna().unique())
    log.info('[load] liftover CSV: %d rows, %d unique genes', len(gwas_liftover_df), len(gwas_genes))

    log.info('[load] reading eQTL peak files ...')
    eqtl_df = _load_eqtl_peaks(submission_dir, tag, gwas_genes)

    log.info('[load] reading MPD founder means ...')
    mpd_df = _load_mpd_founder_means(strain_means_dir)

    log.info('[load] reference frames ready')

    if args.report_only:
        # ── report-only: skip compute, read existing CSV ──────────────────────
        if not out_path.exists():
            log.error('[report-only] results CSV not found: %s', out_path)
            sys.exit(1)
        corr_df = pd.read_csv(out_path)
        log.info('[report-only] loaded %d rows from %s', len(corr_df), out_path)
        _write_report(
            corr_df, report_dir, tag, args.tissue,
            mpd_df=mpd_df,
            eqtl_df=eqtl_df,
            gwas_liftover_df=gwas_liftover_df,
            top_n=args.top_n,
        )
    else:
        # ── full run: correlate → FDR → filter → write → report ──────────────
        log.info('[compute] running correlation engine  (min_founders=%d)', args.min_founders)
        corr_df = _compute_correlations(eqtl_df, mpd_df, min_founders=args.min_founders)

        if corr_df.empty:
            log.warning('[out] no correlations — check inputs and --min-founders')
            sys.exit(0)

        corr_df = (
            corr_df
            .assign(_abs_r=corr_df['r_pearson'].abs())
            .sort_values('_abs_r', ascending=False)
            .drop(columns='_abs_r')
            .reset_index(drop=True)
        )

        # per-measure BH-FDR: stratify by (study, varname, measnum, sex) so the
        # denominator is the number of (gene, CT) pairs tested for that phenotype,
        # not the global test count.  Computed before the |r| filter so m isn't
        # artificially deflated by the effect-size gate.
        measure_cols = ['study', 'varname', 'measnum', 'sex']
        measure_sizes = corr_df.groupby(measure_cols, sort=False).size()
        q_vals = np.ones(len(corr_df))
        corr_df = corr_df.reset_index(drop=True)
        for key, grp in corr_df.groupby(measure_cols, sort=False):
            idx   = grp.index
            m_bh  = int(measure_sizes[key])
            pvals = corr_df.loc[idx, 'p_pearson'].to_numpy(dtype=float)
            order = np.argsort(pvals)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, m_bh + 1)[:len(idx)]
            raw_q = np.clip(pvals * m_bh / ranks, 0.0, 1.0)
            raw_q[order] = np.minimum.accumulate(raw_q[order][::-1])[::-1]
            q_vals[idx] = raw_q
        corr_df['q_pearson'] = q_vals
        n_sig = int((corr_df['q_pearson'] < 0.05).sum())
        log.info('[fdr] per-measure BH-FDR complete — %d / %d rows q<0.05',
                 n_sig, len(corr_df))

        if args.min_abs_r > 0:
            n_before = len(corr_df)
            corr_df = corr_df[corr_df['r_pearson'].abs() >= args.min_abs_r].reset_index(drop=True)
            log.info('[out] min_abs_r=%.3f: kept %d / %d rows  (%.1f%%)',
                     args.min_abs_r, len(corr_df), n_before,
                     100 * len(corr_df) / max(n_before, 1))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(out_path, index=False)
        log.info('[out] results CSV: %s  (%d rows)', out_path, len(corr_df))

        if not args.no_report:
            _write_report(
                corr_df, report_dir, tag, args.tissue,
                mpd_df=mpd_df,
                eqtl_df=eqtl_df,
                gwas_liftover_df=gwas_liftover_df,
                top_n=args.top_n,
            )

    log.info('[done] mpd_founder_correlation complete.')


if __name__ == '__main__':
    main()
