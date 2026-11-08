#!/usr/bin/env python
"""GO / pathway enrichment for mosaiQTL cistrans gene sets via gprofiler2.

Runs gprofiler2 on matching and novel eGene sets per cell type / scan tag,
using the ctxhipp expressed gene universe as background.  Style follows the
established mosaiqtl_go_ct.ipynb notebook conventions.

Usage
-----
    conda run -n bioinfo python validation/tools/go_enrich.py
    conda run -n bioinfo python validation/tools/go_enrich.py --celltype microglial_cell
    conda run -n bioinfo python validation/tools/go_enrich.py --all-celltypes

Outputs → validation/ctxhipp_data/bioground/go/<celltype>/
    go_results_<cat>_<tag>.csv      gprofiler2 hits (FDR < 0.05, filtered)
    go_dotplot_<tag>.png            side-by-side dot-plot: matching vs novel
    go_barplot_<cat>_<tag>.png      Enrichr-style horizontal bar chart per cat
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gprofiler import GProfiler

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ── paths ─────────────────────────────────────────────────────────────────────
_TOOLS    = pathlib.Path(__file__).parent
_DATA     = _TOOLS.parent / 'ctxhipp_data'
_CISTRANS = _DATA / 'cistrans_comp'
_UNIVERSE = _DATA / 'nova' / 'exp_gene_info_ctxhipp.csv'
_OUT_ROOT = _DATA / 'bioground' / 'go'

SCAN_TAGS = ('C', 'Lw1')
CATS      = ('matching', 'novel')

CAT_COLORS = {'matching': '#2166ac', 'novel': '#e08214'}

_GO_PALETTE = {
    'GO:BP': '#66c2a5',
    'GO:MF': '#8073ac',
    'GO:CC': '#fc8d62',
    'KEGG':  '#e78ac3',
    'REAC':  '#a6d854',
}

# ── font (Carlito → Calibri → DejaVu fallback) ────────────────────────────────
_FONT_PREF = ['Carlito', 'Calibri', 'DejaVu Sans']
for _fp in _FONT_PREF:
    if any(_fp.lower() in f.name.lower() for f in _fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fp
        break


def _save_fig(fig: plt.Figure, out_path: pathlib.Path, dpi: int = 300) -> None:
    """Save transparent original + opaque *_share.png for direct sharing."""
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=True)
    share = out_path.with_name(out_path.stem + '_share' + out_path.suffix)
    fig.savefig(share, dpi=dpi, bbox_inches='tight',
                transparent=False, facecolor='white')


def _apply_carlito(ax: plt.Axes) -> None:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily(['Carlito', 'Calibri', 'DejaVu Sans'])


# ── gprofiler2 helpers ────────────────────────────────────────────────────────

def _load_background(universe_path: pathlib.Path) -> list[str]:
    u = pd.read_csv(universe_path)
    return u['gene_symbol'].dropna().unique().tolist()


def _run_gprofiler(
    query: list[str],
    background: list[str],
    organism: str = 'mmusculus',
    min_intersection: int = 5,
    max_term_size: int = 500,
) -> pd.DataFrame:
    """Run gprofiler2 with custom background; return filtered significant hits."""
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(
        organism=organism,
        query=query,
        background=background,
        sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
        significance_threshold_method='fdr',
        user_threshold=0.05,
        domain_scope='custom',
        all_results=False,
        no_evidences=False,
    )
    if not isinstance(res, pd.DataFrame) or res.empty:
        return pd.DataFrame()
    # filter for meaningful terms (same as notebook)
    res = res[
        (res['intersection_size'] >= min_intersection) &
        (res['term_size'] <= max_term_size)
    ].sort_values('p_value').reset_index(drop=True)
    return res


def _collapse_terms(df: pd.DataFrame, jaccard_threshold: float = 0.5) -> pd.DataFrame:
    """Remove redundant GO terms using intersection/term size as Jaccard proxy."""
    df = df.sort_values('p_value').reset_index(drop=True)
    keep, dropped = [], set()
    for i, row_i in df.iterrows():
        if i in dropped:
            continue
        keep.append(i)
        for j, row_j in df.iloc[i + 1:].iterrows():
            if j in dropped:
                continue
            est_j = (
                min(row_i['intersection_size'], row_j['intersection_size']) /
                max(row_i['term_size'], row_j['term_size'], 1)
            )
            if est_j >= jaccard_threshold:
                dropped.add(j)
    return df.loc[keep].reset_index(drop=True)


# ── plot functions ────────────────────────────────────────────────────────────

def plot_go_barplot(
    results: pd.DataFrame,
    label: str,
    cat: str,
    top_n: int = 15,
    outdir: pathlib.Path | None = None,
) -> plt.Figure:
    """Enrichr-style horizontal bar chart for one category (notebook style)."""
    if results.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f'{label}: no significant terms', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#888')
        sns.despine(ax=ax)
        return fig

    # most significant first (top of plot); no reversal
    df = _collapse_terms(results).head(top_n).copy().reset_index(drop=True)
    df['log10p']    = -np.log10(df['p_value'].clip(lower=1e-300))
    df['overlap']   = df['intersection_size'].astype(str) + '/' + df['term_size'].astype(str)
    df['term_short'] = df['name'].str.split('(GO', regex=False).str[0].str.strip()
    df['Term'] = [
        textwrap.fill(f'{n} [{ov}]', width=52)
        for n, ov in zip(df['term_short'], df['overlap'])
    ]

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.48 * len(df) + 1.5)), dpi=300, facecolor='white'
    )
    sns.barplot(
        data=df, x='log10p', y='Term', hue='source',
        palette=_GO_PALETTE, errorbar=None, dodge=False, ax=ax,
    )
    for i, (val, p) in enumerate(zip(df['log10p'], df['p_value'])):
        ax.text(val + 0.06, i, f'p={p:.1e}', va='center', fontsize=10, color='#555')

    # pad x-axis right so p-value annotations don't crowd the legend;
    # bbox_inches='tight' in _save_fig captures the outside-right legend automatically
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)

    ax.set_xlabel('−log₁₀(adj p-value)', fontsize=13)
    ax.set_ylabel('GO Term', fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(f'{label}  [{cat}]  —  GO enrichment (top {top_n})', fontsize=12)
    ax.legend(title='', fontsize=10, frameon=False, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    _apply_carlito(ax)
    sns.despine(ax=ax)
    fig.tight_layout()

    if outdir is not None:
        out = outdir / f'go_barplot_{cat}_{label}.png'
        _save_fig(fig, out)
        log.info('  saved %s  +  _share', out.name)

    return fig


def plot_go_dotplot(
    results_by_cat: dict[str, pd.DataFrame],
    label: str,
    top_n: int = 20,
    outdir: pathlib.Path | None = None,
) -> plt.Figure:
    """Side-by-side dot-plot: matching (Blues) vs novel (Oranges)."""
    # gather top terms
    term_frames = []
    for cat, df in results_by_cat.items():
        if df.empty:
            continue
        sub = _collapse_terms(df).head(top_n).copy()
        sub['category'] = cat
        sub['neg_log10_p'] = -np.log10(sub['p_value'].clip(lower=1e-300))
        term_frames.append(sub[['name', 'source', 'category',
                                 'neg_log10_p', 'precision',
                                 'intersection_size', 'term_size']])

    if not term_frames:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, 'no significant GO terms', ha='center',
                va='center', transform=ax.transAxes, fontsize=10, color='#888')
        sns.despine(ax=ax)
        return fig

    combined = pd.concat(term_frames, ignore_index=True)
    term_order = (
        combined.groupby('name')['neg_log10_p']
        .max().sort_values(ascending=False).head(top_n).index.tolist()
    )
    combined = combined[combined['name'].isin(term_order)].copy()
    combined['name'] = pd.Categorical(
        combined['name'], categories=list(reversed(term_order)), ordered=True,
    )

    n_panels = sum(1 for df in results_by_cat.values() if not df.empty)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(6.5 * n_panels, max(4, len(term_order) * 0.35 + 2)),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for cat in CATS:
        df = results_by_cat.get(cat, pd.DataFrame())
        if df.empty:
            continue
        ax = axes[ax_idx]
        sub = combined[combined['category'] == cat].copy()
        if sub.empty:
            ax.set_title(f'{cat}  (no hits)', fontsize=9)
            sns.despine(ax=ax)
            ax_idx += 1
            continue

        cmap = 'Blues' if cat == 'matching' else 'Oranges'
        s = np.clip(sub['intersection_size'].values, 1, 200) * 4
        sc = ax.scatter(
            sub['neg_log10_p'].values, sub['name'].values,
            c=sub['precision'].values, cmap=cmap, s=s,
            alpha=0.85, edgecolors='none', vmin=0, vmax=1,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('precision', fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlabel('−log₁₀(FDR)', fontsize=9)
        ax.set_title(f'{cat}  (n={len(sub)})', fontsize=9,
                     color=CAT_COLORS[cat], fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=7)
        sns.despine(ax=ax)
        _apply_carlito(ax)
        ax_idx += 1

    fig.suptitle(f'GO enrichment — {label.replace("_"," ")}', fontsize=10, y=1.01)
    fig.tight_layout()

    if outdir is not None:
        out = outdir / f'go_dotplot_{label}.png'
        _save_fig(fig, out)
        log.info('  saved %s  +  _share', out.name)

    return fig


# ── per-celltype runner ───────────────────────────────────────────────────────

def run_celltype(
    celltype: str,
    tags: list[str],
    background: list[str],
    out_root: pathlib.Path,
) -> None:
    ct_dir = _CISTRANS / celltype
    if not ct_dir.exists():
        log.warning('no cistrans dir for %s — skip', celltype)
        return

    out_ct = out_root / celltype
    out_ct.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        log.info('── %s  [%s] ──', celltype, tag)
        results_by_cat: dict[str, pd.DataFrame] = {}

        for cat in CATS:
            csv = ct_dir / f'cistrans_{cat}_{celltype}_{tag}.csv'
            if not csv.exists():
                log.warning('  missing %s', csv.name)
                results_by_cat[cat] = pd.DataFrame()
                continue

            genes_df = pd.read_csv(csv)
            query = genes_df['gene_symbol'].dropna().unique().tolist()
            log.info('  %s: %d genes → gprofiler2 (background=%d) …',
                     cat, len(query), len(background))

            if not query:
                results_by_cat[cat] = pd.DataFrame()
                continue

            try:
                res = _run_gprofiler(query, background)
            except Exception as exc:
                log.warning('  gprofiler2 failed for %s/%s: %s', cat, tag, exc)
                results_by_cat[cat] = pd.DataFrame()
                continue

            log.info('  %s: %d significant terms', cat, len(res))
            if not res.empty:
                csv_out = out_ct / f'go_results_{cat}_{tag}.csv'
                res.to_csv(csv_out, index=False)
                log.info('  saved %s', csv_out.name)

                # per-cat barplot
                plot_go_barplot(res, label=tag, cat=cat, top_n=15, outdir=out_ct)
                plt.close('all')

            results_by_cat[cat] = res

        # side-by-side dot-plot
        plot_go_dotplot(results_by_cat, label=tag, top_n=20, outdir=out_ct)
        plt.close('all')


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='GO enrichment for cistrans matching/novel eGene sets.'
    )
    ap.add_argument('--celltype', default='microglial_cell',
                    help='CT subdirectory name (default: microglial_cell)')
    ap.add_argument('--all-celltypes', action='store_true',
                    help='Run for all CTs with cistrans CSVs.')
    ap.add_argument('--tag', default=None,
                    help='Scan tag (C or Lw1). Default: both.')
    ap.add_argument('--data', default='validation/ctxhipp_data',
                    help='ctxhipp data root')
    ap.add_argument('--universe', default=None,
                    help='Path to universe CSV (default: <data>/nova/exp_gene_info_ctxhipp.csv)')
    args = ap.parse_args()

    data_root    = pathlib.Path(args.data)
    cistrans_dir = data_root / 'cistrans_comp'
    out_root     = data_root / 'bioground' / 'go'
    tags = [args.tag] if args.tag else list(SCAN_TAGS)

    # load whole-tissue pseudobulk universe once, shared across all CTs
    univ_path = pathlib.Path(args.universe) if args.universe else (
        data_root / 'nova' / 'exp_gene_info_ctxhipp.csv'
    )
    if not univ_path.exists():
        log.error('universe CSV not found: %s', univ_path)
        return
    background = _load_background(univ_path)
    log.info('  gene universe: %d symbols from %s', len(background), univ_path.name)

    if args.all_celltypes:
        celltypes = sorted(p.name for p in cistrans_dir.iterdir()
                           if p.is_dir() and not p.name.startswith('.'))
    else:
        celltypes = [args.celltype]

    for ct in celltypes:
        run_celltype(ct, tags, background, out_root)

    log.info('done — outputs in %s', out_root)


if __name__ == '__main__':
    main()
