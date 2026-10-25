"""plot_founder_egene_effects.py — visualize per-founder eGene effects for B6J and CASTJ.

Produces five figures (A–E) saved to:
    validation/ctxhipp_data/egene_effects/plots/

Usage
-----
    conda run -n bioinfo python validation/tools/plot_founder_egene_effects.py
    conda run -n bioinfo python validation/tools/plot_founder_egene_effects.py --tag C
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / 'src'))
from mosaiqtl.genotype.cc import FOUNDER_COLORS, strip_j

# ── constants ─────────────────────────────────────────────────────────────────
_BASE       = pathlib.Path(__file__).parents[2] / 'validation' / 'ctxhipp_data' / 'egene_effects'
_BY_FOUNDER = _BASE / 'by_founder'
_OUT        = _BASE / 'plots'

_FOUNDERS = ['B6J', 'CASTJ']
_TAGS     = ['Lw1', 'C']

_CHR_ORDER = [str(c) for c in range(1, 20)] + ['X']
_CHR_KEY   = {c: i for i, c in enumerate(_CHR_ORDER)}

# shared axis label for all per-gene effect columns
_EFFECT_LABEL  = 'haplotype effect at peak locus (β, log-norm expression)'
_RANGE_LABEL   = 'allelic expression range (Δβ, log-norm expression)'


def _fcolor(founder: str) -> str:
    return FOUNDER_COLORS.get(founder, '#888888')


def _load(founder: str, tag: str) -> pd.DataFrame | None:
    p = _BY_FOUNDER / f'egene_effects_{founder}_{tag}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['chr'] = df['chr'].astype(str)
    return df


def _savefig(fig: plt.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'  saved → {path}')


# ── A: effect-size distributions ─────────────────────────────────────────────

def plot_A(tag: str) -> None:
    """Proportion histogram + KDE of haplotype effect at peak locus, one curve per founder."""
    frames: dict[str, pd.DataFrame] = {}
    for f in _FOUNDERS:
        df = _load(f, tag)
        if df is not None:
            frames[f] = df

    if not frames:
        print(f'  [A] no data for tag={tag}')
        return

    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

    for founder, df in frames.items():
        color = _fcolor(founder)
        vals  = df['focal_beta'].dropna()
        sns.histplot(
            vals, ax=ax,
            color=color, label=f'{strip_j(founder)}  (n={len(vals):,})',
            stat='count', bins=60,
            fill=True, alpha=0.35, kde=True,
            line_kws={'linewidth': 1.8},
            edgecolor='none',
        )
        sns.rugplot(
            vals, ax=ax,
            color=color, height=0.03, alpha=0.12, linewidth=0.4,
        )

    ax.axvline(0, color='#888888', linewidth=0.8, linestyle=':')
    ax.set_xlabel(_EFFECT_LABEL, fontsize=10)
    ax.set_ylabel('number of eGenes', fontsize=10)
    ax.set_title(f'Distribution of haplotype effects at peak locus  ·  {tag}', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    _savefig(fig, _OUT / f'effect_dist_B6J_CASTJ_{tag}.png')


# ── B: genomic lollipop ───────────────────────────────────────────────────────

def plot_B(tag: str = 'Lw1') -> None:
    """Haplotype effect at peak locus vs concatenated genomic position, lollipop style."""
    frames: dict[str, pd.DataFrame] = {}
    for f in _FOUNDERS:
        df = _load(f, tag)
        if df is not None:
            frames[f] = df

    if not frames:
        print(f'  [B] no data for tag={tag}')
        return

    all_chrs = sorted(
        set(c for df in frames.values() for c in df['chr'].unique()),
        key=lambda c: _CHR_KEY.get(c, 99),
    )
    all_data = pd.concat(list(frames.values()))
    chr_max  = all_data.groupby('chr')['peak_mb'].max().reindex(all_chrs, fill_value=0)
    gap      = 10.0
    offsets: dict[str, float] = {}
    cursor = 0.0
    for c in all_chrs:
        offsets[c] = cursor
        cursor += float(chr_max.get(c, 0)) + gap

    _chr_colors    = ['#bbbbbb' if i % 2 == 0 else '#888888' for i, _ in enumerate(all_chrs)]
    _chr_color_map = {c: _chr_colors[i] for i, c in enumerate(all_chrs)}

    fig, axes = plt.subplots(
        len(frames), 1,
        figsize=(18, 3.5 * len(frames)), dpi=300,
        sharex=False,
    )
    if len(frames) == 1:
        axes = [axes]

    xtick_pos = [offsets[c] + float(chr_max.get(c, 0)) / 2 for c in all_chrs]
    xtick_lab = [f'chr{c}' for c in all_chrs]

    for ax, (founder, df) in zip(axes, frames.items()):
        if len(df) > 30_000:
            df = df.sample(5000, random_state=42)
        df = df.copy()
        df['x']   = df.apply(lambda r: offsets.get(str(r['chr']), 0) + r['peak_mb'], axis=1)
        df['_cc'] = df['chr'].map(_chr_color_map)

        ax.axhline(0, color='#444444', linewidth=0.6, zorder=1)
        ax.vlines(df['x'], 0, df['focal_beta'],
                  colors=df['_cc'].tolist(), linewidth=0.5, alpha=0.6, zorder=2)
        sns.scatterplot(
            x=df['x'], y=df['focal_beta'],
            color=_fcolor(founder), s=4, alpha=0.7,
            edgecolors='none', zorder=3, ax=ax, legend=False,
        )
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lab, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel(_EFFECT_LABEL, fontsize=9)
        ax.set_title(strip_j(founder), fontsize=9, pad=3)
        sns.despine(ax=ax)

    fig.suptitle(f'Genome-wide distribution of eGene haplotype effects  ·  {tag}',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _savefig(fig, _OUT / f'genomic_lollipop_B6J_CASTJ_{tag}.png')


# ── C: cell-type eGene count heatmap ─────────────────────────────────────────

def plot_C(tag: str) -> None:
    """Count of eGenes per cell type × founder (top + bottom combined)."""
    dfs = {f: _load(f, tag) for f in _FOUNDERS}
    dfs = {f: d for f, d in dfs.items() if d is not None}
    if not dfs:
        print(f'  [C] no data for tag={tag}')
        return

    rows = []
    for founder, df in dfs.items():
        counts = df.groupby('celltype').size().reset_index(name='n_egenes')
        counts['founder'] = strip_j(founder)
        rows.append(counts)

    combined = pd.concat(rows, ignore_index=True)
    col_order = [strip_j(f) for f in _FOUNDERS]
    pivot = combined.pivot_table(
        index='celltype', columns='founder', values='n_egenes', aggfunc='sum', fill_value=0,
    ).reindex(columns=col_order, fill_value=0)

    pivot = pivot.reindex(natsorted(pivot.index))
    pivot.index = [ct.replace('_', ' ') for ct in pivot.index]

    fig_h = max(5, len(pivot) * 0.55)
    fig, ax = plt.subplots(figsize=(6, fig_h), dpi=300)
    sns.heatmap(
        pivot, annot=True, fmt='d',
        cmap='YlOrRd',
        linewidths=0.4, linecolor='#eeeeee',
        ax=ax,
        cbar_kws={'shrink': 0.6, 'label': 'number of eGenes'},
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'eGenes per cell type and founder haplotype  ·  {tag}', fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    _savefig(fig, _OUT / f'celltype_heatmap_B6J_CASTJ_{tag}.png')


# ── D: LOD vs effect — 2D density ────────────────────────────────────────────

def plot_D(tag: str = 'Lw1') -> None:
    """2D density contour of peak LOD vs haplotype effect, one panel per founder."""
    dfs = {f: _load(f, tag) for f in _FOUNDERS}
    dfs = {f: d for f, d in dfs.items() if d is not None}
    if not dfs:
        print(f'  [D] no data for tag={tag}')
        return

    fig, axes = plt.subplots(
        1, len(dfs), figsize=(7 * len(dfs), 5), dpi=300, sharey=True,
    )
    if len(dfs) == 1:
        axes = [axes]

    for ax, (founder, df) in zip(axes, dfs.items()):
        df = df.dropna(subset=['peak_lod', 'focal_beta'])

        # filled 2D density
        sns.kdeplot(
            x=df['peak_lod'], y=df['focal_beta'],
            fill=True, thresh=0.05, levels=8,
            cmap='Blues', ax=ax,
        )

        # sparse scatter overlay for individual points
        samp = df.sample(min(300, len(df)), random_state=42)
        sns.scatterplot(
            x=samp['peak_lod'], y=samp['focal_beta'],
            color=_fcolor(founder), s=6, alpha=0.3,
            edgecolors='none', ax=ax, legend=False,
        )

        ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='--')
        ax.set_xlabel('peak LOD score', fontsize=9)
        ax.set_ylabel(_EFFECT_LABEL, fontsize=9)
        ax.set_title(strip_j(founder), fontsize=10)
        sns.despine(ax=ax)

    fig.suptitle(f'QTL strength vs haplotype effect  ·  {tag}', fontsize=10)
    fig.tight_layout()
    _savefig(fig, _OUT / f'lod_effect_density_B6J_CASTJ_{tag}.png')


# ── E: allelic expression range per cell type ────────────────────────────────

def plot_E(tag: str = 'Lw1') -> None:
    """Violin of allelic expression range (focal − contrast beta) per cell type."""
    dfs = {f: _load(f, tag) for f in _FOUNDERS}
    dfs = {f: d for f, d in dfs.items() if d is not None}
    if not dfs:
        print(f'  [E] no data for tag={tag}')
        return

    fig, axes = plt.subplots(
        1, len(dfs), figsize=(10 * len(dfs), 5), dpi=300, sharey=True,
    )
    if len(dfs) == 1:
        axes = [axes]

    for ax, (founder, df) in zip(axes, dfs.items()):
        df = df.dropna(subset=['focal_beta', 'contrast_beta']).copy()
        df['allelic_range'] = df['focal_beta'] - df['contrast_beta']

        # sort cell types by median allelic_range descending
        ct_order = (
            df.groupby('celltype')['allelic_range']
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        df['ct_label'] = df['celltype'].str.replace('_', ' ')
        ct_order_labels = [ct.replace('_', ' ') for ct in ct_order]

        sns.violinplot(
            x='ct_label', y='allelic_range', data=df,
            order=ct_order_labels,
            color=_fcolor(founder),
            inner='quartile', linewidth=0.8,
            ax=ax,
        )
        sns.stripplot(
            x='ct_label', y='allelic_range', data=df,
            order=ct_order_labels,
            color='#333333', size=2, alpha=0.2, jitter=True,
            ax=ax,
        )

        ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='--')
        ax.set_xticklabels(ct_order_labels, rotation=40, ha='right', fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel(_RANGE_LABEL, fontsize=9)
        ax.set_title(strip_j(founder), fontsize=10)
        sns.despine(ax=ax)

    fig.suptitle(
        f'Allelic expression range across founder haplotypes  ·  {tag}',
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, _OUT / f'allelic_range_B6J_CASTJ_{tag}.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='Plot B6J/CASTJ eGene effect figures (A–E).'
    )
    p.add_argument('--tag', default='Lw1',
                   help='Primary scan tag (default: Lw1); A and C also run for C if files exist')
    args = p.parse_args()
    tag = args.tag

    _OUT.mkdir(parents=True, exist_ok=True)
    print(f'output dir: {_OUT}')

    # A — both tags if data present
    for t in [tag] + [t for t in _TAGS if t != tag]:
        if any((_BY_FOUNDER / f'egene_effects_{f}_{t}.csv').exists() for f in _FOUNDERS):
            print(f'\n[A] effect distributions  tag={t}')
            plot_A(t)

    # B — primary tag only
    print(f'\n[B] genomic lollipop  tag={tag}')
    plot_B(tag)

    # C — both tags if data present
    for t in [tag] + [t for t in _TAGS if t != tag]:
        if any((_BY_FOUNDER / f'egene_effects_{f}_{t}.csv').exists() for f in _FOUNDERS):
            print(f'\n[C] celltype mean-effect heatmap  tag={t}')
            plot_C(t)

    # D — primary tag only
    print(f'\n[D] LOD vs effect density  tag={tag}')
    plot_D(tag)

    # E — primary tag only
    print(f'\n[E] allelic range per celltype  tag={tag}')
    plot_E(tag)

    print('\ndone.')


if __name__ == '__main__':
    main()
