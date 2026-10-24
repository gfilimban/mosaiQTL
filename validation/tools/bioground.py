#!/usr/bin/env python
"""Biological grounding of mosaiQTL matching/novel/cistrans eGene sets.

Produces two analyses (GWAS overlap moved to gwas_overlap.py):

    1. Founder-effect architecture
       Per gene, compute the founder-effect range (max_beta − min_beta) at the
       peak locus from allele_effects_<CT>_<tag>.csv.  Compares matching vs
       novel distributions with violin + strip plots per cell type.

    2. Cell-type specificity
       For each significant eGene, count how many cell types it appears in
       across all eqtl_summary.csv files.  Plots ECDF + histogram comparing
       matching vs novel breadth.  CT-specific (breadth=1) vs constitutive
       (breadth≥10) fractions are annotated.

Outputs (all → validation/ctxhipp_data/bioground/):
    founder_effects/   per-CT violin PNGs + summary CSV
    ct_specificity/    ECDF PNG + breadth CSV

Usage
-----
    conda run -n bioinfo python validation/tools/bioground.py
    conda run -n bioinfo python validation/tools/bioground.py --tag Lw1
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as _fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── path wiring so we can import mosaiqtl and genotype_palette ──────────────
_TOOLS = pathlib.Path(__file__).parent
_VALIDATION = _TOOLS.parent
_REPO = _VALIDATION.parent
_MOSAIQTL_SRC = _REPO / 'src'
if str(_MOSAIQTL_SRC) not in sys.path:
    sys.path.insert(0, str(_MOSAIQTL_SRC))

# 8cube scripts for palette helpers (optional — graceful fallback)
_8CUBE_SCRIPTS = (_REPO.parent / '8cube_manuscript' / 'scripts')
if _8CUBE_SCRIPTS.exists() and str(_8CUBE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_8CUBE_SCRIPTS))

try:
    from mosaiqtl.genotype.founders import FOUNDER_ORDER, FOUNDER_COLORS
except ImportError:
    FOUNDER_ORDER = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
    FOUNDER_COLORS = {
        'AJ': '#F4C245', 'B6J': '#C0BFBF', '129S1J': '#DA9CC1', 'NODJ': '#4F6EAF',
        'NZOJ': '#52A5DB', 'CASTJ': '#55AF5B', 'PWKJ': '#D83026', 'WSBJ': '#683C91',
    }

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)
# suppress matplotlib's noisy per-glyph Calibri fallback warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def _lighten(hex_color: str, factor: float = 0.50) -> str:
    """Blend *hex_color* toward white by *factor* (0 = unchanged, 1 = white).

    Used to derive the 'novel' hue from a founder's canonical color so both
    violins in a per-founder subplot share the same color family.
    """
    import matplotlib.colors as _mc
    r, g, b, _ = _mc.to_rgba(hex_color)
    r2 = r + (1.0 - r) * factor
    g2 = g + (1.0 - g) * factor
    b2 = b + (1.0 - b) * factor
    return _mc.to_hex((r2, g2, b2))


# ── constants ────────────────────────────────────────────────────────────────
BETA_COLS = [f'beta_{f}' for f in FOUNDER_ORDER]

CAT_COLORS: dict[str, str] = {
    'matching': '#2166ac',
    'novel':    '#e08214',
}

# Carlito (Calibri-metric open substitute shipped with many Linux distros)
_FONT_PREF = ['Carlito', 'Calibri', 'DejaVu Sans']
for _fp in _FONT_PREF:
    if any(_fp.lower() in f.name.lower() for f in _fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fp
        break

# EBI FTP — ontology-annotated version gives MAPPED_TRAIT (EFO controlled
# vocabulary) so disease bucketing is exact rather than free-text heuristic.
GWAS_CATALOG_URL = (
    'https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/'
    'gwas-catalog-associations_ontology-annotated-full.zip'
)

# EFO trait→parent mapping is loaded at runtime from gwas-efo-trait-mappings.tsv
# (downloaded from GWAS Catalog; lives in bioground/).  This replaces keyword
# guessing — the file gives the official EFO parent term (e.g. "Nervous system
# disease") for each trait term so categorisation is exact and hierarchy-aware.
_EFO_MAPPINGS_FILE = 'gwas-efo-trait-mappings.tsv'   # relative to bioground dir

SCAN_TAGS = ('C', 'Lw1')

# ── helpers ──────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, out_path: pathlib.Path, dpi: int = 300) -> None:
    """Save figure twice: transparent original + opaque *_share.png for direct sharing."""
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=True)
    share_path = out_path.with_name(out_path.stem + '_share' + out_path.suffix)
    fig.savefig(share_path, dpi=dpi, bbox_inches='tight',
                transparent=False, facecolor='white')

def _nova_dir(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / 'nova' / 'celltypes'


def _load_cistrans_partitions(data_dir: pathlib.Path, tag: str) -> dict[str, pd.DataFrame]:
    """Load matching and novel per-CT CSVs written by cistrans_overlap.py."""
    partitions: dict[str, list[pd.DataFrame]] = {'matching': [], 'novel': []}
    for cat in ('matching', 'novel'):
        pattern = f'*/cistrans_{cat}_*_{tag}.csv'
        for p in sorted((data_dir / 'cistrans_comp').glob(pattern)):
            df = pd.read_csv(p)
            df['category'] = cat
            partitions[cat].append(df)
    result = {}
    for cat, frames in partitions.items():
        if frames:
            result[cat] = pd.concat(frames, ignore_index=True)
        else:
            log.warning('No %s CSVs found for tag=%s', cat, tag)
            result[cat] = pd.DataFrame()
    return result


def _load_allele_effects(nova: pathlib.Path) -> pd.DataFrame:
    """Concatenate all allele_effects CSVs across CTs and tags."""
    records: list[pd.DataFrame] = []
    for p in sorted(nova.glob('*/*/core/allele_effects_*.csv')):
        parts = p.parts
        ct   = parts[-4]
        tag  = parts[-3]
        if tag not in SCAN_TAGS:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            log.warning('skip %s: %s', p, exc)
            continue
        df['celltype'] = ct
        df['scan_tag'] = tag
        records.append(df)
    if not records:
        raise FileNotFoundError(f'No allele_effects CSVs found under {nova}')
    ae = pd.concat(records, ignore_index=True)
    # compute per-gene-per-CT-per-tag effect range
    ae['effect_range'] = ae[BETA_COLS].max(axis=1) - ae[BETA_COLS].min(axis=1)
    return ae


def _load_all_egenes(nova: pathlib.Path) -> pd.DataFrame:
    """Load all eqtl_summary.csv files (significant eGenes only)."""
    records: list[pd.DataFrame] = []
    for p in sorted(nova.glob('*/*/eqtl_summary.csv')):
        parts = p.parts
        ct   = parts[-3]
        tag  = parts[-2]
        if tag not in SCAN_TAGS:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            log.warning('skip %s: %s', p, exc)
            continue
        df['celltype'] = ct
        df['scan_tag'] = tag
        records.append(df)
    if not records:
        raise FileNotFoundError(f'No eqtl_summary.csv files found under {nova}')
    return pd.concat(records, ignore_index=True)


def _apply_font_and_despine(fig: plt.Figure) -> None:
    for ax in fig.axes:
        sns.despine(ax=ax)
        ax.tick_params(labelsize=11)


# ── Section 1: Founder-effect architecture ───────────────────────────────────

def founder_effect_architecture(
    partitions: dict[str, pd.DataFrame],
    ae: pd.DataFrame,
    out_dir: pathlib.Path,
    tag: str,
) -> None:
    """Compare per-founder effect range between matching and novel eGenes."""
    log.info('[1] Founder-effect architecture (tag=%s) …', tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge effect range onto each partition
    # ae is per (gene_symbol, celltype, scan_tag) — take the max range when
    # a gene has multiple peak_locus entries (shouldn't happen in eqtl_summary)
    ae_key = ae[ae['scan_tag'] == tag][
        ['gene_symbol', 'celltype', 'scan_tag', 'effect_range']
    ].groupby(['gene_symbol', 'celltype'], as_index=False)['effect_range'].max()

    frames: list[pd.DataFrame] = []
    for cat, df in partitions.items():
        if df.empty:
            continue
        sub = df[df['scan_tag'] == tag].copy()
        merged = sub.merge(ae_key, on=['gene_symbol', 'celltype'], how='left')
        merged['category'] = cat
        frames.append(merged)

    if not frames:
        log.warning('No data for founder effect plot (tag=%s)', tag)
        return

    plot_df = pd.concat(frames, ignore_index=True).dropna(subset=['effect_range'])

    # ── aggregate summary CSV ────────────────────────────────────────────────
    summary = (
        plot_df
        .groupby(['celltype', 'category'])['effect_range']
        .agg(n='count', median='median', mean='mean', q25=lambda x: x.quantile(0.25),
             q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    summary.to_csv(out_dir / f'founder_effects_summary_{tag}.csv', index=False)
    log.info('  saved founder_effects_summary_%s.csv', tag)

    # ── plot: per-CT violin ──────────────────────────────────────────────────
    celltypes = sorted(plot_df['celltype'].unique())
    n_ct = len(celltypes)
    fig, axes = plt.subplots(
        1, n_ct, figsize=(max(12, n_ct * 1.6), 4.5), sharey=True
    )
    axes = [axes] if n_ct == 1 else list(axes)

    for ax, ct in zip(axes, celltypes):
        sub = plot_df[plot_df['celltype'] == ct]
        sns.violinplot(
            data=sub, x='category', y='effect_range', hue='category',
            order=['matching', 'novel'],
            palette=CAT_COLORS,
            inner='box', linewidth=0.8, cut=0, legend=False, ax=ax,
        )
        ax.set_title(ct.replace('_', ' '), fontsize=10, pad=4)
        ax.set_xlabel('')
        if ax is axes[0]:
            ax.set_ylabel('founder effect range\n(max − min β)', fontsize=12)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=10)

    fig.suptitle(
        f'Founder-effect range: matching vs novel eGenes  [{tag}]',
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _apply_font_and_despine(fig)
    out_path = out_dir / f'founder_effects_violin_{tag}.png'
    _save_fig(fig, out_path)
    plt.close(fig)
    log.info('  saved %s  +  _share', out_path.name)

    # ── plot: pooled distribution (all CTs together) ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.violinplot(
        data=plot_df, x='category', y='effect_range', hue='category',
        order=['matching', 'novel'],
        palette=CAT_COLORS,
        inner='box', linewidth=0.8, cut=0, legend=False, ax=ax2,
    )
    ax2.set_xlabel('')
    ax2.set_ylabel('founder effect range  (max − min β)', fontsize=13)
    ax2.set_title(f'Pooled across all CTs  [{tag}]', fontsize=13)
    # annotate medians
    for i, cat in enumerate(['matching', 'novel']):
        med = plot_df.loc[plot_df['category'] == cat, 'effect_range'].median()
        n   = (plot_df['category'] == cat).sum()
        ax2.text(i, ax2.get_ylim()[1] * 0.97, f'n={n:,}\nmed={med:.3f}',
                 ha='center', va='top', fontsize=10)
    _apply_font_and_despine(fig2)
    sns.despine(ax=ax2)
    fig2.tight_layout()
    out_path2 = out_dir / f'founder_effects_pooled_{tag}.png'
    _save_fig(fig2, out_path2)
    plt.close(fig2)
    log.info('  saved %s  +  _share', out_path2.name)

    # ── per-founder beta distribution (matching vs novel) ────────────────────
    founder_rows: list[dict] = []
    for cat, df in partitions.items():
        if df.empty:
            continue
        sub = df[df['scan_tag'] == tag][['gene_symbol', 'celltype']].merge(
            ae[ae['scan_tag'] == tag][['gene_symbol', 'celltype'] + BETA_COLS],
            on=['gene_symbol', 'celltype'], how='inner',
        )
        for col in BETA_COLS:
            founder = col.replace('beta_', '')
            founder_rows.append({
                'category': cat,
                'founder':  founder,
                'beta':     sub[col].values,
            })

    if founder_rows:
        long_rows = []
        for r in founder_rows:
            for v in r['beta']:
                long_rows.append({'category': r['category'], 'founder': r['founder'], 'beta': v})
        founder_long = pd.DataFrame(long_rows)

        n_founders = len(FOUNDER_ORDER)
        fig3, axes3 = plt.subplots(
            1, n_founders, figsize=(n_founders * 1.5, 3.5), sharey=True
        )
        for ax, f in zip(axes3, FOUNDER_ORDER):
            sub = founder_long[founder_long['founder'] == f]
            base_color = FOUNDER_COLORS.get(f, '#888888')
            # matching = canonical founder color; novel = lightened version
            founder_palette = {
                'matching': base_color,
                'novel':    _lighten(base_color, factor=0.50),
            }
            sns.violinplot(
                data=sub, x='category', y='beta', hue='category',
                order=['matching', 'novel'],
                palette=founder_palette,
                inner=None, linewidth=0.6, cut=0, legend=False, ax=ax,
            )
            ax.axhline(0, color='#999', lw=0.6, ls='--')
            ax.set_title(f, fontsize=11, color='#222222',
                         fontweight='bold', pad=3)
            ax.set_xlabel('')
            ax.set_ylabel('β' if ax is axes3[0] else '', fontsize=13)
            ax.tick_params(axis='x', labelsize=10)
        fig3.suptitle(
            f'Per-founder β at peak locus  [{tag}]', fontsize=13, y=1.03,
        )
        fig3.tight_layout()
        _apply_font_and_despine(fig3)
        out_path3 = out_dir / f'founder_effects_per_founder_{tag}.png'
        _save_fig(fig3, out_path3)
        plt.close(fig3)
        log.info('  saved %s  +  _share', out_path3.name)


# ── Section 2: Cell-type specificity ─────────────────────────────────────────

def cell_type_specificity(
    partitions: dict[str, pd.DataFrame],
    all_egenes: pd.DataFrame,
    out_dir: pathlib.Path,
    tag: str,
) -> None:
    """Count how many CTs each gene appears in; compare matching vs novel."""
    log.info('[2] Cell-type specificity (tag=%s) …', tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CT breadth: for each gene, count distinct CTs (across both tags is also
    # valid, but here we use the requested tag for consistency with partitions)
    egenes_tag = all_egenes[all_egenes['scan_tag'] == tag]
    breadth = (
        egenes_tag.groupby('gene_symbol')['celltype']
        .nunique()
        .reset_index(name='n_celltypes')
    )

    frames: list[pd.DataFrame] = []
    for cat, df in partitions.items():
        if df.empty:
            continue
        sub = df[df['scan_tag'] == tag][['gene_symbol', 'celltype']].drop_duplicates()
        merged = sub[['gene_symbol']].drop_duplicates().merge(breadth, on='gene_symbol', how='left')
        merged['n_celltypes'] = merged['n_celltypes'].fillna(1).astype(int)
        merged['category'] = cat
        frames.append(merged)

    if not frames:
        log.warning('No data for CT specificity plot (tag=%s)', tag)
        return

    plot_df = pd.concat(frames, ignore_index=True)

    # save breadth CSV
    breadth_out = plot_df.copy()
    breadth_out.to_csv(out_dir / f'ct_breadth_{tag}.csv', index=False)
    log.info('  saved ct_breadth_%s.csv', tag)

    # ── CT-specific fraction (log) ────────────────────────────────────────────
    total_cts = egenes_tag['celltype'].nunique()
    for cat, color in CAT_COLORS.items():
        sub = plot_df.loc[plot_df['category'] == cat, 'n_celltypes']
        if sub.empty:
            continue
        frac1  = (sub == 1).mean()
        frac10 = (sub >= max(10, total_cts - 2)).mean()
        log.info('  %s  CT-specific(1)=%.1f%%  constitutive(≥%d)=%.1f%%',
                 cat, frac1 * 100, max(10, total_cts - 2), frac10 * 100)

    # ── breadth bins ─────────────────────────────────────────────────────────
    max_b = int(plot_df['n_celltypes'].max())

    def _breadth_bin(n: int) -> str:
        if n == 1:
            return '1'
        if n <= 3:
            return '2–3'
        if n <= 6:
            return '4–6'
        return '7+'

    bin_order = ['1', '2–3', '4–6', '7+']
    plot_df = plot_df.copy()
    plot_df['breadth_bin'] = plot_df['n_celltypes'].apply(_breadth_bin)

    # alpha levels for breadth bins (darkest = CT-specific)
    bin_alphas = {'1': 1.0, '2–3': 0.60, '4–6': 0.35, '7+': 0.18}

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (ax_bar, ax_ecdf) = plt.subplots(
        1, 2,
        figsize=(13, 4),
        gridspec_kw={'width_ratios': [1, 1.1]},
    )

    # ── left: stacked proportion barh ────────────────────────────────────────
    import matplotlib.colors as _mc

    cat_list = [c for c in ('matching', 'novel') if c in plot_df['category'].unique()]
    for y_pos, cat in enumerate(cat_list):
        sub   = plot_df[plot_df['category'] == cat]
        n_tot = len(sub)
        left  = 0.0
        base  = _mc.to_rgba(CAT_COLORS[cat])
        for b in bin_order:
            prop  = (sub['breadth_bin'] == b).sum() / n_tot
            rgba  = (*base[:3], bin_alphas[b])
            ax_bar.barh(y_pos, prop, left=left, color=rgba, height=0.55,
                        edgecolor='white', linewidth=0.4)
            # annotate fraction inside segment when it is wide enough
            if prop >= 0.07:
                txt_color = 'white' if bin_alphas[b] > 0.55 else '#333'
                ax_bar.text(left + prop / 2, y_pos, f'{prop:.0%}',
                            ha='center', va='center', fontsize=10,
                            color=txt_color, fontweight='bold' if b == '1' else 'normal')
            left += prop
        # label CT-specific fraction prominently at the right edge
        frac1 = (sub['n_celltypes'] == 1).mean()
        ax_bar.text(1.01, y_pos, f'{frac1:.0%} CT-specific',
                    va='center', ha='left', fontsize=11,
                    color=CAT_COLORS[cat], fontweight='bold')

    ax_bar.set_yticks(range(len(cat_list)))
    ax_bar.set_yticklabels([f'{c}  (n={len(plot_df[plot_df["category"]==c]):,})'
                            for c in cat_list], fontsize=11)
    ax_bar.set_xlabel('proportion of eGenes', fontsize=13)
    ax_bar.set_title(f'CT breadth distribution  [{tag}]', fontsize=13)
    ax_bar.set_xlim(0, 1)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax_bar.invert_yaxis()   # matching on top

    # legend for breadth bins
    from matplotlib.patches import Patch as _Patch
    base_rgba = _mc.to_rgba(CAT_COLORS['matching'])
    legend_handles = [
        _Patch(facecolor=(*base_rgba[:3], bin_alphas[b]),
               edgecolor='white', label=b if b != '1' else '1  (CT-specific)')
        for b in bin_order
    ]
    ax_bar.legend(handles=legend_handles, title='# cell types',
                  fontsize=10, title_fontsize=11,
                  loc='lower right', frameon=False)

    # ── right: ECDF ──────────────────────────────────────────────────────────
    ecdf_y_at1: dict[str, float] = {}
    for cat, color in CAT_COLORS.items():
        sub = plot_df.loc[plot_df['category'] == cat, 'n_celltypes'].dropna()
        if sub.empty:
            continue
        sorted_vals = np.sort(sub.values)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        sns.lineplot(x=sorted_vals, y=ecdf_y, color=color, linewidth=2.0,
                     label=cat, ax=ax_ecdf, errorbar=None)
        frac1 = float((sub <= 1).mean())
        ecdf_y_at1[cat] = frac1
        # mark the step at breadth=1 with a filled dot
        ax_ecdf.scatter([1], [frac1], color=color, s=45, zorder=6, clip_on=False)

    # vertical reference at x=1 (CT-specific boundary)
    ax_ecdf.axvline(1, color='#ccc', lw=1.0, ls='--', zorder=0)

    # label each step just above the dot — stagger to avoid overlap
    cats_sorted = sorted(
        ecdf_y_at1.items(), key=lambda kv: kv[1], reverse=True
    )
    placed_y: list[float] = []
    for cat, y in cats_sorted:
        color = CAT_COLORS[cat]
        # nudge label up if it would collide with an already-placed one
        label_y = y + 0.03
        for prev in placed_y:
            if abs(label_y - prev) < 0.07:
                label_y = prev + 0.07
        ax_ecdf.text(
            1.0, label_y,
            f' {y:.0%} of {cat}\n genes in 1 CT only',
            fontsize=10, color=color, va='bottom', ha='left',
            fontweight='bold',
        )
        placed_y.append(label_y)

    ax_ecdf.set_xlabel('# cell types  (eGene breadth)', fontsize=13)
    ax_ecdf.set_ylabel('cumulative fraction of eGenes', fontsize=13)
    ax_ecdf.set_title(f'CT breadth ECDF  [{tag}]', fontsize=13)
    ax_ecdf.set_xlim(0.8, max_b + 0.3)
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # legend outside axes at bottom so it never overlaps the curves
    ax_ecdf.legend(fontsize=11, frameon=False,
                   loc='upper center', bbox_to_anchor=(0.5, -0.14),
                   ncol=2)

    fig.tight_layout(rect=[0, 0.06, 0.97, 1])  # bottom room for legend
    _apply_font_and_despine(fig)
    out_path = out_dir / f'ct_specificity_{tag}.png'
    _save_fig(fig, out_path)
    plt.close(fig)
    log.info('  saved %s  +  %s', out_path.name, out_path.stem + '_share.png')


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Biological grounding: founder effects + CT specificity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--data', default='validation/ctxhipp_data',
                    help='ctxhipp data root.')
    ap.add_argument('--tag', default=None,
                    help='Restrict to one scan tag (C or Lw1).  Default: both.')
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data)
    nova     = _nova_dir(data_dir)
    out_root = data_dir / 'bioground'

    tags = [args.tag] if args.tag else list(SCAN_TAGS)

    log.info('mosaiQTL bioground — data: %s', data_dir)
    log.info('  nova dir:   %s', nova)
    log.info('  output:     %s', out_root)
    log.info('  scan tags:  %s', tags)

    # ── shared loads ─────────────────────────────────────────────────────────
    log.info('loading allele_effects …')
    ae = _load_allele_effects(nova)
    log.info('  %d allele_effect rows across %d CTs',
             len(ae), ae['celltype'].nunique())

    log.info('loading eqtl_summary (significant eGenes) …')
    all_egenes = _load_all_egenes(nova)
    log.info('  %d eGene rows, %d CTs, %d distinct genes',
             len(all_egenes), all_egenes['celltype'].nunique(),
             all_egenes['gene_symbol'].nunique())

    for tag in tags:
        log.info('═══ tag: %s ═══', tag)

        partitions = _load_cistrans_partitions(data_dir, tag)
        for cat, df in partitions.items():
            log.info('  %s: %d rows', cat, len(df))

        # 1. founder effects
        founder_effect_architecture(
            partitions, ae,
            out_root / 'founder_effects',
            tag,
        )

        # 2. CT specificity
        cell_type_specificity(
            partitions, all_egenes,
            out_root / 'ct_specificity',
            tag,
        )

    log.info('done — outputs in %s', out_root)


if __name__ == '__main__':
    main()
