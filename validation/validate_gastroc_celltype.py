"""
validate_gastroc_celltype.py — per-cell-type QTL cross-comparison

Compares per-cell-type QTL results to each other and to the whole-tissue scan.
Thin caller: data loading + section logic + one _cmp_* call per figure + savefig.
No inline plot logic.

This script does NOT run scans.  It reads outputs produced by running
validate_gastroc.py --cell_type <name> once per cell type, which produces
per-cell-type outputs in:

    gastroc_data/py_data/<cell_type>/
        s4_peak_lod.csv          (gene, chr, start_mb, near_tss_mb, peak_locus, peak_mb,
                                   cis_dist_mb, peak_lod, h2, n_valid, rss0, rss1, r2_peak,
                                   sigma2_e, tau2)
        s4_allele_effects.csv    (gene, chr, start_mb, peak_locus, peak_mb, peak_lod, h2,
                                   beta_AJ, beta_B6J, beta_129S1J, beta_NODJ, beta_NZOJ,
                                   beta_CASTJ, beta_PWKJ, beta_WSBJ)
        eqtl_summary.csv         (eGene table)

Whole-tissue results live in gastroc_data/py_data/ (no subdirectory):
    s4_peak_lod.csv
    s4_allele_effects.csv
    s4_lod.csv
    s5_threshold.csv

Output plots: gastroc_data/plots_celltype/

Usage:
  python validation/validate_gastroc_celltype.py \\
      --celltype myonuclei FAPs endothelial macrophages \\
      [--data_dir validation/gastroc_data] \\
      [--sections 1,2,3] \\
      [--rep_gene Piezo1]

  # auto-discover cell types from py_data/ subdirectories:
  python validation/validate_gastroc_celltype.py

Sections:
  S1  Cell type vs whole tissue — LOD scatter + density summary
  S2  Cell type vs whole tissue — allele effects scatter
  S3  LOD profile overlay for a representative gene (--rep_gene)
  S4  Cell type vs cell type pairwise LOD scatter + lower-triangle grid
  S5  Cell type vs cell type pairwise allele effects scatter
  S6  LOD distribution per cell type vs whole tissue
  S7  Cis-distance violin + proximal-fraction curves across all sources
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Carlito'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mosaiqtl.viz.compare import (
    lod_scatter as _cmp_scatter,
    lod_density_comparison as _cmp_lod_density,
    allele_effects_comparison as _cmp_allele_effects,
    lod_profile_overlay as _cmp_lod_overlay,
    cis_distance_comparison as _cmp_cis_distance,
)
from mosaiqtl.genotype.founders import (
    FOUNDER_COLORS,
    FOUNDER_ORDER,
    COLOR_NEUTRAL_DARK,
    COLOR_CC_FALLBACK,
)

log = logging.getLogger(__name__)

# Canonical beta column names — display names match s4_allele_effects.csv
_BETA_COLS: list[str] = [f'beta_{f}' for f in FOUNDER_ORDER]
_FOUNDERS_DISPLAY: list[str] = FOUNDER_ORDER  # AJ, B6J, 129S1J, …

# Pin myonuclei to NOD/J blue (same as compare.py default series-A color)
_MYONUCLEI_COLOR: str = FOUNDER_COLORS['NODJ']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ct(ct: str) -> str:
    """Return a filesystem-safe cell-type token (spaces → underscores)."""
    return ct.replace(' ', '_')


# Color palette
# ─────────────────────────────────────────────────────────────────────────────

def _build_ct_palette(cell_types: list[str]) -> dict[str, str]:
    """Assign a distinct FOUNDER_COLORS value to each cell type.

    myonuclei is always pinned to FOUNDER_COLORS['NODJ'].
    Remaining cell types are assigned by sorted alphabetical order cycling
    through the remaining FOUNDER_COLORS values.
    """
    all_colors = list(FOUNDER_COLORS.values())
    remaining_colors = [c for c in all_colors if c != _MYONUCLEI_COLOR]
    palette: dict[str, str] = {}
    remaining_cts = sorted(ct for ct in cell_types if ct != 'myonuclei')
    for i, ct in enumerate(remaining_cts):
        palette[ct] = remaining_colors[i % len(remaining_colors)]
    if 'myonuclei' in cell_types:
        palette['myonuclei'] = _MYONUCLEI_COLOR
    return palette


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_peak_lod(path: Path, suffix: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load s4_peak_lod.csv; rename peak_lod → peak_lod_<suffix> when supplied.

    Returns None when the file is absent or unreadable.
    """
    if not path.exists():
        log.warning('missing: %s', path)
        return None
    try:
        df = pd.read_csv(path)
        if suffix and 'peak_lod' in df.columns:
            df = df.rename(columns={'peak_lod': f'peak_lod_{suffix}'})
        return df
    except Exception as exc:
        log.warning('could not read %s: %s', path, exc)
        return None


def _load_allele_effects(path: Path) -> Optional[pd.DataFrame]:
    """Load s4_allele_effects.csv.  Returns None on error."""
    if not path.exists():
        log.warning('missing: %s', path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning('could not read %s: %s', path, exc)
        return None


def _load_gene_lod(path: Path, gene: str) -> Optional[pd.DataFrame]:
    """Load a LOD profile CSV and filter to a single gene name.

    Accepts both streaming s4_lod.csv (large) and small per-gene rop_lod.csv.
    Returns None when the file is absent, unreadable, or the gene has no rows.
    """
    if not path.exists():
        log.debug('not found: %s', path)
        return None
    try:
        # Stream in 1M-row chunks to handle large s4_lod.csv
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=1_000_000):
            if 'gene' in chunk.columns:
                hit = chunk[chunk['gene'] == gene]
                if not hit.empty:
                    chunks.append(hit.copy())
        if not chunks:
            return None
        return pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        log.warning('could not read %s: %s', path, exc)
        return None


def _load_threshold(py_data: Path) -> Optional[float]:
    """Read genome-wide FWER threshold from whole-tissue s5_threshold.csv.

    Always call with whole_py_data — this threshold is used on both axes of
    every scatter comparison regardless of which cell types are being compared.
    """
    path = py_data / 's5_threshold.csv'
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if 'chr' in df.columns:
            row = df[df['chr'] == 'genome_wide']
            if not row.empty:
                return float(row['threshold'].iloc[0])
        if 'threshold' in df.columns:
            return float(df['threshold'].iloc[0])
        return None
    except Exception as exc:
        log.warning('could not read threshold from %s: %s', path, exc)
        return None


def _discover_cell_types(celltypes_dir: Path, input_version: str) -> list[str]:
    """Auto-discover cell type names under celltypes/<name>/<input_version>/py_data/."""
    cts: list[str] = []
    if not celltypes_dir.exists():
        return cts
    for sub in sorted(celltypes_dir.iterdir()):
        if sub.is_dir() and sub.name != 'compare':
            candidate = sub / input_version / 'py_data' / 's4_peak_lod.csv'
            if candidate.exists():
                cts.append(sub.name)
    return cts


# ─────────────────────────────────────────────────────────────────────────────
# Pair ordering
# ─────────────────────────────────────────────────────────────────────────────

def _pair_order(ct_a: str, ct_b: str) -> tuple[str, str]:
    """Determine x (first) vs y (second) axis assignment for a cell-type pair.

    myonuclei is always the x-axis.  Otherwise, alphabetical order determines
    which goes on x.
    """
    if ct_a == 'myonuclei':
        return ct_a, ct_b
    if ct_b == 'myonuclei':
        return ct_b, ct_a
    return (ct_a, ct_b) if ct_a <= ct_b else (ct_b, ct_a)


# ─────────────────────────────────────────────────────────────────────────────
# Section helpers
# ─────────────────────────────────────────────────────────────────────────────

def _melt_betas(
    eff_df: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    """Melt allele-effects DataFrame to long form with columns [gene, beta_col, value_name]."""
    id_cols = ['gene'] if 'gene' in eff_df.columns else []
    beta_present = [c for c in _BETA_COLS if c in eff_df.columns]
    if not beta_present:
        return pd.DataFrame(columns=id_cols + ['beta_col', value_name])
    return eff_df[id_cols + beta_present].melt(
        id_vars=id_cols,
        value_vars=beta_present,
        var_name='beta_col',
        value_name=value_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Cell type vs whole tissue (LOD scatter + density summary)
# ─────────────────────────────────────────────────────────────────────────────

def _s1_ct_vs_whole(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    whole_py_data: Path,
    ct_out_dirs: dict[str, Path],
    compare_dir: Path,
    palette: dict[str, str],
    threshold: Optional[float],
    rep_gene: Optional[str] = None,
) -> None:
    """Per-cell-type LOD scatter vs whole tissue; density summary grid."""
    whole_df = _load_peak_lod(whole_py_data / 's4_peak_lod.csv', suffix='whole')
    if whole_df is None:
        log.warning('S1: whole-tissue s4_peak_lod.csv not found — skipping section')
        return

    # Individual per-cell-type scatter plots
    for ct in cell_types:
        try:
            ct_df = _load_peak_lod(ct_dirs[ct] / 's4_peak_lod.csv', suffix=ct)
            if ct_df is None:
                continue
            merged = whole_df[['gene', 'peak_lod_whole']].merge(
                ct_df[['gene', f'peak_lod_{ct}']], on='gene', how='inner'
            )
            if merged.empty:
                log.warning('S1: no shared genes for cell type %s', ct)
                continue
            log.debug('S1 %s vs whole: %d shared genes', ct, len(merged))
            _cmp_scatter(
                merged,
                x_col='peak_lod_whole',
                y_col=f'peak_lod_{ct}',
                x_label='Whole tissue peak LOD',
                y_label=f'{ct} peak LOD',
                threshold_x=threshold,
                threshold_y=threshold,
                priority_genes={rep_gene: COLOR_NEUTRAL_DARK} if rep_gene else None,
                save_path=ct_out_dirs[ct] / f's1_{_safe_ct(ct)}_vs_whole_lod.png',
            )
            plt.close('all')
            # Export quadrant gene lists for GO analysis
            if threshold is not None:
                _cols = ['gene', 'peak_lod_whole', f'peak_lod_{ct}']
                _w = merged['peak_lod_whole']
                _c = merged[f'peak_lod_{ct}']
                _quadrants = {
                    's1_shared_egenes_vs_whole.csv':
                        merged[(_w >= threshold) & (_c >= threshold)].sort_values(
                            f'peak_lod_{ct}', ascending=False),
                    's1_ct_specific_egenes.csv':
                        merged[(_w < threshold) & (_c >= threshold)].sort_values(
                            f'peak_lod_{ct}', ascending=False),
                    's1_whole_specific_egenes.csv':
                        merged[(_w >= threshold) & (_c < threshold)].sort_values(
                            'peak_lod_whole', ascending=False),
                }
                for fname, df_q in _quadrants.items():
                    out_csv = ct_dirs[ct] / fname
                    df_q[_cols].to_csv(out_csv, index=False)
                    log.info('S1 %s: %d genes → %s', ct, len(df_q), out_csv)
        except Exception as exc:
            log.warning('S1 scatter %s: %s', ct, exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Cell type vs whole tissue (allele effects scatter)
# ─────────────────────────────────────────────────────────────────────────────

def _s2_effects_ct_vs_whole(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    whole_py_data: Path,
    ct_out_dirs: dict[str, Path],
) -> None:
    """Per-cell-type allele effects scatter vs whole tissue."""
    whole_eff = _load_allele_effects(whole_py_data / 's4_allele_effects.csv')
    if whole_eff is None:
        log.warning('S2: whole-tissue s4_allele_effects.csv not found — skipping section')
        return

    whole_long = _melt_betas(whole_eff, 'beta_whole')

    for ct in cell_types:
        try:
            ct_eff = _load_allele_effects(ct_dirs[ct] / 's4_allele_effects.csv')
            if ct_eff is None:
                continue
            ct_long = _melt_betas(ct_eff, f'beta_{ct}')
            if ct_long.empty:
                log.warning('S2: no beta columns for %s', ct)
                continue

            merge_cols = ['gene', 'beta_col'] if 'gene' in whole_long.columns else ['beta_col']
            merged = whole_long[merge_cols + ['beta_whole']].merge(
                ct_long[merge_cols + [f'beta_{ct}']],
                on=merge_cols,
                how='inner',
            )
            if merged.empty:
                log.warning('S2: no overlapping data for %s', ct)
                continue

            founders_present = [
                f for f in _FOUNDERS_DISPLAY
                if f'beta_{f}' in merged['beta_col'].values
            ]
            log.debug('S2 %s vs whole effects: %d gene×founder pairs', ct, len(merged))
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            _cmp_allele_effects(
                betas_a=merged[f'beta_{ct}'].values,
                betas_b=merged['beta_whole'].values,
                founders=founders_present,
                label_a=ct,
                label_b='Whole',
                ax=ax,
            )
            fig.tight_layout()
            save_path = ct_out_dirs[ct] / f's2_{_safe_ct(ct)}_vs_whole_effects.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.debug('S2: saved %s', save_path)
        except Exception as exc:
            log.warning('S2 %s: %s', ct, exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# S3 — LOD profile overlay for a representative gene
# ─────────────────────────────────────────────────────────────────────────────

def _s3_lod_overlay(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    whole_py_data: Path,
    compare_dir: Path,
    palette: dict[str, str],
    threshold: Optional[float],
    rep_gene: str,
) -> None:
    """LOD profile overlay for rep_gene: whole tissue + up to one cell type per call.

    lod_profile_overlay supports exactly two series (A and B).  When more than
    two traces are available, the first pair (whole + first cell type) is plotted
    in the main figure, and each remaining cell type is saved as a separate figure
    paired against whole tissue.
    """
    # Determine chromosome of rep_gene from whole-tissue peak_lod
    whole_peak = _load_peak_lod(whole_py_data / 's4_peak_lod.csv')
    peak_chr: Optional[str] = None
    if whole_peak is not None and 'gene' in whole_peak.columns:
        row = whole_peak[whole_peak['gene'] == rep_gene]
        if not row.empty:
            peak_chr = str(row['chr'].iloc[0])
    if peak_chr is None:
        log.warning('S3: %s not found in whole-tissue peak_lod — chr unknown', rep_gene)

    def _get_trace(lod_path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
        df = _load_gene_lod(lod_path, rep_gene)
        if df is None or df.empty:
            return None
        if peak_chr is not None and 'chr' in df.columns:
            df = df[df['chr'].astype(str) == peak_chr]
        if df.empty:
            return None
        df = df.sort_values('pos_mb')
        return df['pos_mb'].values, df['lod'].values

    # Whole-tissue trace
    whole_trace = _get_trace(whole_py_data / 's4_lod.csv')

    # Cell-type traces
    ct_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ct in cell_types:
        try:
            tr = _get_trace(ct_dirs[ct] / 's4_lod.csv')
            if tr is not None:
                ct_traces[ct] = tr
            else:
                log.warning('S3: no LOD profile for gene=%s ct=%s', rep_gene, ct)
        except Exception as exc:
            log.warning('S3 trace load %s: %s', ct, exc)

    if whole_trace is None and not ct_traces:
        log.warning('S3: no LOD profile data available for %s — skipping', rep_gene)
        return

    # Plot whole + first cell type in the main figure; remaining as paired figures.
    pairs: list[tuple[str, Optional[tuple], str, Optional[tuple]]] = []
    ct_list = list(ct_traces.keys())

    if ct_list:
        first_ct = ct_list[0]
        pairs.append(('Whole', whole_trace, first_ct, ct_traces[first_ct]))
        for ct in ct_list[1:]:
            pairs.append(('Whole', whole_trace, ct, ct_traces[ct]))
    elif whole_trace is not None:
        # Only whole-tissue available
        pairs.append(('Whole', whole_trace, '', None))

    for idx, (label_a, trace_a, label_b, trace_b) in enumerate(pairs):
        try:
            fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
            pos_a: np.ndarray = trace_a[0] if trace_a is not None else np.array([])
            lod_a: np.ndarray = trace_a[1] if trace_a is not None else np.array([])
            pos_b: Optional[np.ndarray] = trace_b[0] if trace_b is not None else None
            lod_b: Optional[np.ndarray] = trace_b[1] if trace_b is not None else None
            color_a = COLOR_NEUTRAL_DARK if label_a == 'Whole' else palette.get(label_a, COLOR_CC_FALLBACK)
            color_b = palette.get(label_b, COLOR_CC_FALLBACK) if label_b else COLOR_CC_FALLBACK
            _cmp_lod_overlay(
                pos_a=pos_a,
                lod_a=lod_a,
                pos_b=pos_b,
                lod_b=lod_b,
                label_a=label_a,
                label_b=label_b if label_b else 'B',
                color_a=color_a,
                color_b=color_b,
                chrom=peak_chr or '',
                gene=rep_gene,
                fwer_thr=threshold,
                ax=ax,
            )
            fig.tight_layout()
            if idx == 0:
                save_name = f's3_lod_overlay_{rep_gene}.png'
            else:
                safe_label = label_b.replace(' ', '_')
                save_name = f's3_lod_overlay_{rep_gene}_{safe_label}.png'
            save_path = compare_dir / save_name
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.debug('S3: saved %s', save_path)
        except Exception as exc:
            log.warning('S3 pair %s/%s: %s', label_a, label_b, exc, exc_info=True)

    log.info('S3: saved %d LOD overlay figure(s) for %s', len(pairs), rep_gene)


# ─────────────────────────────────────────────────────────────────────────────
# S4 — Cell type vs cell type pairwise LOD scatter
# ─────────────────────────────────────────────────────────────────────────────

def _s4_ct_pairwise_lod(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    ct_out_dirs: dict[str, Path],
    compare_dir: Path,
    threshold: Optional[float],
    rep_gene: Optional[str] = None,
) -> None:
    """Pairwise cell-type LOD scatter; lower-triangle summary grid."""
    # Pre-load all tables once
    ct_frames: dict[str, Optional[pd.DataFrame]] = {
        ct: _load_peak_lod(ct_dirs[ct] / 's4_peak_lod.csv', suffix=ct)
        for ct in cell_types
    }

    pairs: list[tuple[str, str]] = []
    for i, ct_a in enumerate(cell_types):
        for ct_b in cell_types[i + 1:]:
            pairs.append(_pair_order(ct_a, ct_b))

    for x_ct, y_ct in pairs:
        try:
            df_x = ct_frames.get(x_ct)
            df_y = ct_frames.get(y_ct)
            if df_x is None or df_y is None:
                log.warning('S4: missing data for %s / %s', x_ct, y_ct)
                continue
            merged = df_x[['gene', f'peak_lod_{x_ct}']].merge(
                df_y[['gene', f'peak_lod_{y_ct}']], on='gene', how='inner'
            )
            if merged.empty:
                log.warning('S4: no shared genes for %s vs %s', x_ct, y_ct)
                continue
            log.debug('S4 %s vs %s: %d shared genes', x_ct, y_ct, len(merged))
            _cmp_scatter(
                merged,
                x_col=f'peak_lod_{x_ct}',
                y_col=f'peak_lod_{y_ct}',
                x_label=f'{x_ct} peak LOD',
                y_label=f'{y_ct} peak LOD',
                threshold_x=threshold,
                threshold_y=threshold,
                priority_genes={rep_gene: COLOR_NEUTRAL_DARK} if rep_gene else None,
                save_path=compare_dir / f's4_{_safe_ct(x_ct)}_vs_{_safe_ct(y_ct)}_lod.png',
            )
            plt.close('all')
            # Export quadrant gene lists for GO analysis
            if threshold is not None:
                _cols = ['gene', f'peak_lod_{x_ct}', f'peak_lod_{y_ct}']
                _x = merged[f'peak_lod_{x_ct}']
                _y = merged[f'peak_lod_{y_ct}']
                _quadrants_s4 = {
                    (x_ct, f's4_shared_egenes_vs_{_safe_ct(y_ct)}.csv'):
                        merged[(_x >= threshold) & (_y >= threshold)].sort_values(
                            f'peak_lod_{x_ct}', ascending=False),
                    (x_ct, f's4_specific_egenes_vs_{_safe_ct(y_ct)}.csv'):
                        merged[(_x >= threshold) & (_y < threshold)].sort_values(
                            f'peak_lod_{x_ct}', ascending=False),
                    (y_ct, f's4_specific_egenes_vs_{_safe_ct(x_ct)}.csv'):
                        merged[(_x < threshold) & (_y >= threshold)].sort_values(
                            f'peak_lod_{y_ct}', ascending=False),
                }
                for (owner_ct, fname), df_q in _quadrants_s4.items():
                    out_csv = ct_dirs[owner_ct] / fname
                    df_q[_cols].to_csv(out_csv, index=False)
                    log.info('S4 %s/%s: %d genes → %s', x_ct, y_ct, len(df_q), out_csv)
        except Exception as exc:
            log.warning('S4 pair %s/%s: %s', x_ct, y_ct, exc, exc_info=True)

    # Summary lower-triangle grid — uses density overlays (lod_scatter has no ax=)
    n = len(cell_types)
    if n < 2:
        return
    try:
        fig = plt.figure(figsize=(4 * n, 4 * n), dpi=300)
        gs = GridSpec(n, n, figure=fig, hspace=0.45, wspace=0.45)
        for i, ct_y in enumerate(cell_types):
            for j, ct_x in enumerate(cell_types):
                if j >= i:
                    continue
                try:
                    x_ct, y_ct = _pair_order(ct_x, ct_y)
                    df_x = ct_frames.get(x_ct)
                    df_y = ct_frames.get(y_ct)
                    if df_x is None or df_y is None:
                        continue
                    merged = df_x[['gene', f'peak_lod_{x_ct}']].merge(
                        df_y[['gene', f'peak_lod_{y_ct}']], on='gene', how='inner'
                    )
                    if merged.empty:
                        continue
                    ax = fig.add_subplot(gs[i, j])
                    _cmp_lod_density(
                        lods_a=merged[f'peak_lod_{x_ct}'].values,
                        lods_b=merged[f'peak_lod_{y_ct}'].values,
                        label_a=x_ct,
                        label_b=y_ct,
                        threshold=threshold,
                        ax=ax,
                    )
                    ax.set_title(f'{x_ct} vs {y_ct}', fontsize=9, fontweight='bold')
                except Exception as exc:
                    log.warning('S4 grid [%d,%d]: %s', i, j, exc)

        fig.suptitle('Pairwise cell-type LOD distributions', fontsize=13,
                     fontweight='bold', y=1.01)
        grid_path = compare_dir / 's4_pairwise_lod_grid.png'
        fig.savefig(grid_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
        log.info('S4: saved pairwise grid %s', grid_path)
    except Exception as exc:
        log.warning('S4 pairwise grid failed: %s', exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# S5 — Cell type vs cell type pairwise allele effects scatter
# ─────────────────────────────────────────────────────────────────────────────

def _s5_effects_pairwise(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    ct_out_dirs: dict[str, Path],
) -> None:
    """Pairwise cell-type allele effects scatter."""
    ct_effects: dict[str, Optional[pd.DataFrame]] = {
        ct: _load_allele_effects(ct_dirs[ct] / 's4_allele_effects.csv')
        for ct in cell_types
    }

    pairs: list[tuple[str, str]] = []
    for i, ct_a in enumerate(cell_types):
        for ct_b in cell_types[i + 1:]:
            pairs.append(_pair_order(ct_a, ct_b))

    for x_ct, y_ct in pairs:
        try:
            eff_x = ct_effects.get(x_ct)
            eff_y = ct_effects.get(y_ct)
            if eff_x is None or eff_y is None:
                log.warning('S5: missing effects for %s / %s', x_ct, y_ct)
                continue

            long_x = _melt_betas(eff_x, f'beta_{x_ct}')
            long_y = _melt_betas(eff_y, f'beta_{y_ct}')
            if long_x.empty or long_y.empty:
                log.warning('S5: no beta columns for %s / %s', x_ct, y_ct)
                continue

            merge_cols = (
                ['gene', 'beta_col']
                if 'gene' in long_x.columns and 'gene' in long_y.columns
                else ['beta_col']
            )
            merged = long_x[merge_cols + [f'beta_{x_ct}']].merge(
                long_y[merge_cols + [f'beta_{y_ct}']],
                on=merge_cols,
                how='inner',
            )
            if merged.empty:
                log.warning('S5: no overlapping effects for %s / %s', x_ct, y_ct)
                continue

            founders_present = [
                f for f in _FOUNDERS_DISPLAY
                if f'beta_{f}' in merged.get('beta_col', pd.Series(dtype=str)).values
            ]
            log.debug('S5 %s vs %s effects: %d pairs', x_ct, y_ct, len(merged))
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            _cmp_allele_effects(
                betas_a=merged[f'beta_{x_ct}'].values,
                betas_b=merged[f'beta_{y_ct}'].values,
                founders=founders_present,
                label_a=x_ct,
                label_b=y_ct,
                ax=ax,
            )
            fig.tight_layout()
            save_path = ct_out_dirs[x_ct] / f's5_{_safe_ct(x_ct)}_vs_{_safe_ct(y_ct)}_effects.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.debug('S5: saved %s', save_path)
        except Exception as exc:
            log.warning('S5 pair %s/%s: %s', x_ct, y_ct, exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# S6 — LOD distribution per cell type vs whole tissue
# ─────────────────────────────────────────────────────────────────────────────

def _s6_lod_distributions(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    whole_py_data: Path,
    compare_dir: Path,
    palette: dict[str, str],
    threshold: Optional[float],
) -> None:
    """Per-cell-type LOD density vs whole tissue; multi-panel summary."""
    whole_df = _load_peak_lod(whole_py_data / 's4_peak_lod.csv', suffix='whole')
    whole_lods: Optional[np.ndarray] = (
        whole_df['peak_lod_whole'].values
        if whole_df is not None and 'peak_lod_whole' in whole_df.columns
        else None
    )

    # Individual density plots
    for ct in cell_types:
        try:
            ct_df = _load_peak_lod(ct_dirs[ct] / 's4_peak_lod.csv', suffix=ct)
            if ct_df is None:
                continue
            ct_col = f'peak_lod_{ct}'
            if ct_col not in ct_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
            _cmp_lod_density(
                lods_a=whole_lods,
                lods_b=ct_df[ct_col].values,
                label_a='Whole',
                label_b=ct,
                color_a=COLOR_NEUTRAL_DARK,
                color_b=palette.get(ct, COLOR_CC_FALLBACK),
                threshold=threshold,
                ax=ax,
            )
            ax.set_title(f'{ct} vs Whole — peak LOD distribution',
                         fontsize=11, fontweight='bold')
            fig.tight_layout()
            save_path = compare_dir / f's6_{_safe_ct(ct)}_vs_whole_lod_density.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.debug('S6: saved %s', save_path)
        except Exception as exc:
            log.warning('S6 %s: %s', ct, exc, exc_info=True)

    # Multi-panel summary
    n_ct = len(cell_types)
    if n_ct < 1:
        return
    try:
        ncols = min(3, n_ct)
        nrows = (n_ct + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 4 * nrows), dpi=300)
        axes_flat: list[plt.Axes] = (
            [axes] if n_ct == 1 else list(np.array(axes).ravel())
        )
        for ax in axes_flat:
            ax.set_visible(False)

        for i, ct in enumerate(cell_types):
            try:
                ct_df = _load_peak_lod(ct_dirs[ct] / 's4_peak_lod.csv', suffix=ct)
                if ct_df is None:
                    continue
                ct_col = f'peak_lod_{ct}'
                if ct_col not in ct_df.columns:
                    continue
                ax = axes_flat[i]
                ax.set_visible(True)
                _cmp_lod_density(
                    lods_a=whole_lods,
                    lods_b=ct_df[ct_col].values,
                    label_a='Whole',
                    label_b=ct,
                    color_a=COLOR_NEUTRAL_DARK,
                    color_b=palette.get(ct, COLOR_CC_FALLBACK),
                    threshold=threshold,
                    ax=ax,
                )
                ax.set_title(ct, fontsize=10, fontweight='bold')
            except Exception as exc:
                log.warning('S6 panel %s: %s', ct, exc)

        fig.suptitle('Peak LOD distributions — each cell type vs whole tissue',
                     fontsize=13, fontweight='bold', y=1.01)
        fig.tight_layout()
        summary_path = compare_dir / 's6_lod_distributions.png'
        fig.savefig(summary_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
        log.info('S6: saved summary %s', summary_path)
    except Exception as exc:
        log.warning('S6 summary figure failed: %s', exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# S7 — Cis-distance distribution comparison across cell types + whole tissue
# ─────────────────────────────────────────────────────────────────────────────

def _s7_cis_distance_comparison(
    cell_types: list[str],
    ct_dirs: dict[str, Path],
    whole_py_data: Path,
    compare_dir: Path,
    palette: dict[str, str],
) -> None:
    """Signed cis-distance violin + proximal-fraction curves across sources."""
    sources: dict[str, tuple] = {}

    def _tss_col(df: pd.DataFrame) -> str:
        return 'near_tss_mb' if 'near_tss_mb' in df.columns else 'start_mb'

    w_path = whole_py_data / 's4_peak_lod.csv'
    if w_path.exists():
        _wdf = pd.read_csv(w_path)
        _wtss = _tss_col(_wdf)
        if {_wtss, 'peak_mb', 'peak_lod'}.issubset(_wdf.columns):
            _wdf = _wdf.dropna(subset=[_wtss, 'peak_mb', 'peak_lod'])
            sources['whole'] = (
                _wdf['peak_lod'].values,
                _wdf[_wtss].values,
                _wdf['peak_mb'].values,
            )

    for ct in cell_types:
        ct_path = ct_dirs[ct] / 's4_peak_lod.csv'
        if not ct_path.exists():
            continue
        _cdf = pd.read_csv(ct_path)
        _ctss = _tss_col(_cdf)
        if not {_ctss, 'peak_mb', 'peak_lod'}.issubset(_cdf.columns):
            continue
        _cdf = _cdf.dropna(subset=[_ctss, 'peak_mb', 'peak_lod'])
        sources[ct] = (
            _cdf['peak_lod'].values,
            _cdf[_ctss].values,
            _cdf['peak_mb'].values,
        )

    if len(sources) < 2:
        log.warning('S7: fewer than 2 sources with cis-distance data — skipping')
        return

    pal = {**{'whole': COLOR_CC_FALLBACK}, **palette}
    fig = _cmp_cis_distance(sources, palette=pal, zoom_mb=1.0)
    out = compare_dir / 's7_cis_distance_comparison.png'
    fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    log.info('S7: %s  (%d sources)', out, len(sources))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for per-cell-type QTL cross-comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description=(
            'Per-cell-type QTL cross-comparison (mosaiQTL). '
            'Reads s4_peak_lod.csv / s4_allele_effects.csv from py_data/<cell_type>/ '
            'and produces comparison plots in plots_celltype/.'
        ),
    )
    parser.add_argument(
        '--celltype',
        nargs='+',
        default=None,
        metavar='CT',
        dest='celltype',
        help='Cell type names to include. If omitted, auto-discovered from '
             'celltypes/<name>/<input_version>/py_data/.',
    )
    parser.add_argument(
        '--data_dir',
        default='validation/gastroc_data',
        help='Path to gastroc_data/ (default: validation/gastroc_data)',
    )
    parser.add_argument(
        '--sections',
        default=None,
        help='Comma-separated section numbers to run, e.g. 1,2,3 (default: all)',
    )
    parser.add_argument(
        '--rep_gene',
        default='Piezo1',
        help='Representative gene for LOD profile overlay in S3 (default: Piezo1)',
    )
    parser.add_argument(
        '--input_version',
        default='v0.4.0',
        help='Version subdirectory to read per-cell-type py_data from '
             '(default: v0.4.0)',
    )
    parser.add_argument(
        '--output_version',
        default='v0.4.7',
        help='Version subdirectory to write per-cell-type plots to '
             '(default: v0.4.7)',
    )
    parser.add_argument(
        '--plots-only',
        action='store_true',
        help='Regenerate all plots from existing py_data/ without re-running any '
             'scans. Runs all sections (overrides --sections); missing input files '
             'are skipped silently.',
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    celltypes_dir = data_dir / 'celltypes'
    whole_py_data = data_dir / 'py_data'       # whole-tissue outputs (unchanged)
    input_ver: str = args.input_version
    output_ver: str = args.output_version

    # Parse sections
    # --plots-only forces all sections and silences missing-file warnings
    plots_only: bool = args.plots_only
    sections: Optional[set[int]] = None
    if plots_only:
        sections = None  # run all
        log.info('--plots-only: running all sections from existing py_data/')
    elif args.sections:
        try:
            sections = {int(s.strip()) for s in args.sections.split(',')}
        except ValueError as exc:
            log.error('invalid --sections value: %s', exc)
            sys.exit(1)

    # Discover or validate cell types
    if args.celltype:
        cell_types = args.celltype
    else:
        cell_types = _discover_cell_types(celltypes_dir, input_ver)
        if cell_types:
            log.info('auto-discovered cell types: %s', cell_types)
        else:
            log.error(
                'no cell-type directories with s4_peak_lod.csv found under '
                '%s/<name>/%s/py_data/; pass --cell_types explicitly',
                celltypes_dir, input_ver,
            )
            sys.exit(1)

    # Build per-cell-type path dicts
    plots_celltypes_dir = data_dir / 'plots' / 'celltypes'
    ct_dirs: dict[str, Path] = {
        ct: celltypes_dir / _safe_ct(ct) / input_ver / 'py_data'
        for ct in cell_types
    }
    ct_out_dirs: dict[str, Path] = {
        ct: plots_celltypes_dir / _safe_ct(ct) / output_ver
        for ct in cell_types
    }
    # Shared comparison output (figures spanning all cell types)
    compare_dir = plots_celltypes_dir / 'compare' / output_ver
    compare_dir.mkdir(parents=True, exist_ok=True)
    for d in ct_out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log.info('cell types    : %s', cell_types)
    log.info('input version : %s', input_ver)
    log.info('output version: %s', output_ver)
    log.info('compare dir   : %s', compare_dir)

    palette = _build_ct_palette(cell_types)
    threshold = _load_threshold(whole_py_data)
    if threshold is not None:
        log.info('FWER threshold (whole-tissue genome_wide): %.4f', threshold)
    else:
        log.info('s5_threshold.csv not found — threshold lines omitted from plots')

    rep_gene: str = args.rep_gene

    # ── Section 1 ──────────────────────────────────────────────────────────────
    if not sections or 1 in sections:
        log.info('=== S1: cell type vs whole tissue (LOD scatter) ===')
        try:
            _s1_ct_vs_whole(
                cell_types, ct_dirs, whole_py_data,
                ct_out_dirs, compare_dir, palette, threshold,
                rep_gene=rep_gene,
            )
        except Exception as exc:
            log.error('S1 failed: %s', exc, exc_info=True)

    # ── Section 2 ──────────────────────────────────────────────────────────────
    if not sections or 2 in sections:
        log.info('=== S2: cell type vs whole tissue (allele effects) ===')
        try:
            _s2_effects_ct_vs_whole(
                cell_types, ct_dirs, whole_py_data, ct_out_dirs,
            )
        except Exception as exc:
            log.error('S2 failed: %s', exc, exc_info=True)

    # ── Section 3 ──────────────────────────────────────────────────────────────
    if not sections or 3 in sections:
        log.info('=== S3: LOD profile overlay (%s) ===', rep_gene)
        try:
            _s3_lod_overlay(
                cell_types, ct_dirs, whole_py_data,
                compare_dir, palette, threshold, rep_gene,
            )
        except Exception as exc:
            log.error('S3 failed: %s', exc, exc_info=True)

    # ── Section 4 ──────────────────────────────────────────────────────────────
    if not sections or 4 in sections:
        log.info('=== S4: cell type vs cell type pairwise LOD scatter ===')
        try:
            _s4_ct_pairwise_lod(
                cell_types, ct_dirs, ct_out_dirs, compare_dir, threshold,
                rep_gene=rep_gene,
            )
        except Exception as exc:
            log.error('S4 failed: %s', exc, exc_info=True)

    # ── Section 5 ──────────────────────────────────────────────────────────────
    if not sections or 5 in sections:
        log.info('=== S5: cell type vs cell type pairwise allele effects ===')
        try:
            _s5_effects_pairwise(cell_types, ct_dirs, ct_out_dirs)
        except Exception as exc:
            log.error('S5 failed: %s', exc, exc_info=True)

    # ── Section 6 ──────────────────────────────────────────────────────────────
    if not sections or 6 in sections:
        log.info('=== S6: LOD distribution per cell type ===')
        try:
            _s6_lod_distributions(
                cell_types, ct_dirs, whole_py_data,
                compare_dir, palette, threshold,
            )
        except Exception as exc:
            log.error('S6 failed: %s', exc, exc_info=True)

    # ── Section 7 ──────────────────────────────────────────────────────────────
    if not sections or 7 in sections:
        log.info('=== S7: cis-distance comparison across cell types ===')
        try:
            _s7_cis_distance_comparison(
                cell_types, ct_dirs, whole_py_data, compare_dir, palette,
            )
        except Exception as exc:
            log.error('S7 failed: %s', exc, exc_info=True)

    log.info('validate_gastroc_celltype.py complete — outputs: %s', compare_dir)


if __name__ == '__main__':
    main()
