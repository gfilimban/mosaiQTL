#!/usr/bin/env python
"""GWAS enrichment: intersect matching/novel/cistrans eGene sets with liftover hits.

Reads the positional overlap file produced by gwas_liftover_overlap.py
(``gwas_liftover_hits_{tissue}_{tag}.csv``) and runs EFO trait enrichment
(Fisher's exact, BH-FDR) for each cell type × cistrans category
(matching / novel / cistrans).  Then renders a dot-plot bubble matrix of
EFO terms × cell types for each category.

This script does NOT download the GWAS Catalog — the liftover file already
encodes which human GWAS SNPs fall within each eGene's QTL confidence interval
(lifted mm39→hg38).  Only the EFO parent-term mapping TSV is needed as an
additional input.

Prerequisites
-------------
1. ``gwas_liftover_overlap.py --tissue {tissue} --tag {tag}``
   → ``validation/gwas_overlap/gwas_liftover_hits_{tissue}_{tag}.csv``
2. ``cistrans_overlap.py``
   → ``validation/ctxhipp_data/cistrans_comp/{ct}/cistrans_matching_{ct}_{tag}.csv``
   → ``validation/ctxhipp_data/cistrans_comp/{ct}/cistrans_novel_{ct}_{tag}.csv``
   → ``validation/ctxhipp_data/cistrans_comp/cistrans_cistrans_egenes.csv``
3. ``gwas-efo-trait-mappings.tsv`` (see --efo-map)
   Download from https://www.ebi.ac.uk/gwas/api/search/downloads/trait_mappings

Outputs (all under ``--out-dir``, default ``validation/gwas_overlap/``)
------------------------------------------------------------------------
Per cell type × category
    {ct}/gwas_genes_{cat}_{tissue}_{tag}.csv     — eGenes with GWAS hits
    {ct}/gwas_enrich_{cat}_{tissue}_{tag}.csv    — Fisher's exact enrichment table
    {ct}/gwas_{cat}_{tissue}_{tag}.png           — enrichment bar chart (top 15)

Top-level
    gwas_summary_{tissue}_{tag}.csv              — #enriched traits per CT × category
    gwas_overlap_{tissue}_{tag}.png              — summary bar chart across CTs
    gwas_dotplot_matching_{tissue}_{tag}.png     — bubble matrix: matching eGenes
    gwas_dotplot_novel_{tissue}_{tag}.png        — bubble matrix: novel eGenes
    gwas_dotplot_cistrans_{tissue}_{tag}.png     — bubble matrix: cistrans eGenes

Usage
-----
    # typical run (paths resolved from repo root)
    conda run -n bioinfo python validation/tools/gwas_overlap.py \\
        --tissue ctxhipp --tag Lw1

    # explicit overrides
    conda run -n bioinfo python validation/tools/gwas_overlap.py \\
        --liftover-csv  validation/gwas_overlap/gwas_liftover_hits_ctxhipp_Lw1.csv \\
        --cistrans-dir  validation/ctxhipp_data/cistrans_comp \\
        --universe-csv  validation/ctxhipp_data/nova/exp_gene_info_ctxhipp.csv \\
        --efo-map       validation/gwas_overlap/gwas-efo-trait-mappings.tsv \\
        --out-dir       validation/gwas_overlap \\
        --tissue ctxhipp --tag Lw1
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import textwrap

import matplotlib
import matplotlib.transforms as mtransforms
matplotlib.use('Agg')
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact

# ── path wiring ───────────────────────────────────────────────────────────────
_TOOLS      = pathlib.Path(__file__).parent
_VALIDATION = _TOOLS.parent
_REPO       = _VALIDATION.parent
_SRC        = _REPO / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ── constants ─────────────────────────────────────────────────────────────────
_EFO_MAPPINGS_FILE = 'gwas-efo-trait-mappings.tsv'

# Three cistrans categories analysed here (discordant omitted — too small / ambiguous)
_CATEGORIES = ('matching', 'novel', 'cistrans')

CAT_COLORS: dict[str, str] = {
    'matching':  '#2166ac',
    'novel':     '#e08214',
    'cistrans':  '#4dac26',
    'discordant':'#d01c8b',
}

_PARENT_COLORS: dict[str, str] = {
    'Neurological disorder':            '#4e79a7',
    'Cancer':                           '#f28e2b',
    'Cardiovascular disease':           '#e15759',
    'Immune system disorder':           '#76b7b2',
    'Metabolic disorder':               '#59a14f',
    'Digestive system disorder':        '#edc948',
    'Lipid or lipoprotein measurement': '#b07aa1',
    'Cardiovascular measurement':       '#ff9da7',
    'Body measurement':                 '#9c755f',
    'Hematological measurement':        '#00acc1',
    'Inflammatory measurement':         '#d37295',
    'Biological process':               '#a0cbe8',
}
_DEFAULT_COLOR = '#aaaaaa'

_FONT_PREF = ['Carlito', 'Calibri', 'DejaVu Sans']

# ── shared utilities ──────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, out_path: pathlib.Path, dpi: int = 300) -> None:
    """Save figure in two variants.

    Plain ``.png``   — transparent background for embedding in slides.
    ``_share.png``   — white (opaque) background for sharing with others.

    The transparent save must be explicit even when the figure was created with
    ``facecolor='none'``, because ``savefig`` inherits the figure facecolor only
    for the axes patches — the figure background still needs ``transparent=True``
    to fully suppress it.  The opaque save overrides with ``facecolor='white'``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, facecolor='none', transparent=True,
                bbox_inches='tight')
    share_path = out_path.with_name(out_path.stem + '_share' + out_path.suffix)
    fig.savefig(str(share_path), dpi=dpi, facecolor='white', bbox_inches='tight')


def _apply_font(fig: plt.Figure) -> None:
    """Apply preferred font to all text in figure, if available."""
    available = {f.name for f in _fm.fontManager.ttflist}
    font = next((f for f in _FONT_PREF if f in available), None)
    if not font:
        return
    for ax in fig.axes:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels() + ax.get_yticklabels()
        ):
            item.set_fontfamily(font)


def _safe_ct(ct: str) -> str:
    """Filesystem-safe cell-type name: spaces → underscores."""
    return ct.replace(' ', '_')


# ── data loaders ─────────────────────────────────────────────────────────────

def _load_cistrans_partitions(
    cistrans_dir: pathlib.Path,
    tag: str,
) -> dict[str, pd.DataFrame]:
    """Load matching, novel, and cistrans per-CT eGene tables.

    - matching and novel: read from per-CT subdirectory files written by
      cistrans_overlap.py (``{ct}/cistrans_{cat}_{ct}_{tag}.csv``).
    - cistrans: read from the global ``cistrans_cistrans_egenes.csv`` file
      (all CTs combined), then filter by scan_tag.  The global file has both
      ``celltype`` and ``scan_tag`` columns.

    Returns dict with keys 'matching', 'novel', 'cistrans'; each value is a
    concatenated DataFrame with a 'category' column added.
    """
    result: dict[str, pd.DataFrame] = {}

    # matching and novel — per-CT files
    for cat in ('matching', 'novel'):
        pattern = f'*/cistrans_{cat}_*_{tag}.csv'
        frames: list[pd.DataFrame] = []
        for p in sorted(cistrans_dir.glob(pattern)):
            df = pd.read_csv(p)
            df['category'] = cat
            frames.append(df)
        if frames:
            result[cat] = pd.concat(frames, ignore_index=True)
        else:
            log.warning('No %s CSVs found for tag=%s in %s', cat, tag, cistrans_dir)
            result[cat] = pd.DataFrame()

    # cistrans — global file, filter by tag
    global_file = cistrans_dir / 'cistrans_cistrans_egenes.csv'
    if global_file.exists():
        ct_df = pd.read_csv(global_file)
        if 'scan_tag' in ct_df.columns:
            ct_df = ct_df[ct_df['scan_tag'] == tag].copy()
        ct_df['category'] = 'cistrans'
        result['cistrans'] = ct_df
        log.info('cistrans: %d rows (tag=%s)', len(ct_df), tag)
    else:
        log.warning('cistrans_cistrans_egenes.csv not found: %s', global_file)
        result['cistrans'] = pd.DataFrame()

    return result


def _load_efo_parent_map(efo_map_path: pathlib.Path) -> dict[str, str]:
    """Load EFO term → parent-term mapping from gwas-efo-trait-mappings.tsv.

    Download once from:
    https://www.ebi.ac.uk/gwas/api/search/downloads/trait_mappings

    Returns dict keyed by lowercase EFO term label → parent category string.
    Falls back to empty dict on missing or malformed file.
    """
    if not efo_map_path.exists():
        log.warning(
            'EFO mappings file not found: %s\n'
            '  Download from https://www.ebi.ac.uk/gwas/api/search/downloads/trait_mappings\n'
            '  All trait terms will be treated as unmapped.',
            efo_map_path,
        )
        return {}
    df = pd.read_csv(efo_map_path, sep='\t')
    efo_col    = 'EFO term'
    parent_col = 'Parent term'
    if efo_col not in df.columns or parent_col not in df.columns:
        log.warning('Unexpected EFO mappings columns: %s', df.columns.tolist())
        return {}
    mapping = {
        str(row[efo_col]).strip().lower(): str(row[parent_col]).strip()
        for _, row in df.iterrows()
        if pd.notna(row[efo_col]) and pd.notna(row[parent_col])
    }
    log.info('Loaded EFO parent map: %d terms', len(mapping))
    return mapping


def _trait_category(mapped_trait: str, efo_map: dict[str, str]) -> str | None:
    """Map a MAPPED_TRAIT EFO label to its official parent-term category.

    Returns None when no EFO parent mapping exists, or when the resolved parent
    is an EBI catch-all bucket ('Other measurement', 'Other trait', 'NR', etc.).
    Callers skip None entries.
    """
    _skip_exact    = {'nr'}
    _skip_prefixes = ('other ',)
    for term in mapped_trait.split('|'):
        parent = efo_map.get(term.strip().lower())
        if not parent:
            continue
        pl = parent.lower()
        if pl in _skip_exact:
            continue
        if any(pl.startswith(pfx) for pfx in _skip_prefixes):
            continue
        return parent
    return None


# ── core analysis functions ───────────────────────────────────────────────────

def gwas_overlap(
    partitions: dict[str, pd.DataFrame],
    liftover_csv: pathlib.Path,
    out_dir: pathlib.Path,
    tag: str,
    tissue: str,
    efo_map_path: pathlib.Path | None = None,
    universe_csv: pathlib.Path | None = None,
    min_bar_terms: int = 3,
) -> None:
    """Intersect matching/novel/cistrans gene sets with liftover GWAS hits.

    For each cell type × category runs Fisher's exact enrichment per EFO trait
    (over-representation vs expressed gene universe), applies BH-FDR, and writes:
      - per-CT gene CSV  ({ct}/gwas_genes_{cat}_{tissue}_{tag}.csv)
      - per-CT enrichment CSV ({ct}/gwas_enrich_{cat}_{tissue}_{tag}.csv)
      - per-CT enrichment bar-chart PNG ({ct}/gwas_{cat}_{tissue}_{tag}.png)
      - top-level summary CSV  (gwas_summary_{tissue}_{tag}.csv)
      - top-level summary bar chart (gwas_overlap_{tissue}_{tag}.png)

    Parameters
    ----------
    partitions :
        Dict from _load_cistrans_partitions(); keys 'matching', 'novel', 'cistrans'.
    liftover_csv :
        ``gwas_liftover_hits_{tissue}_{tag}.csv`` produced by
        ``gwas_liftover_overlap.py``.  Each row is one (eGene × GWAS hit) pair
        whose SNP falls within the QTL CI lifted mm39→hg38.
    out_dir :
        Output root; per-CT subdirs are created automatically.
    tag :
        Scan tag (e.g. 'Lw1').
    tissue :
        Tissue identifier included in all output filenames (e.g. 'ctxhipp').
    efo_map_path :
        Path to ``gwas-efo-trait-mappings.tsv``.
        Defaults to ``out_dir/gwas-efo-trait-mappings.tsv``.
    universe_csv :
        CSV with a 'gene_symbol' column listing all expressed genes for
        the Fisher background.  Defaults to
        ``validation/{tissue}_data/nova/exp_gene_info_{tissue}.csv``.
    """
    log.info('[gwas_overlap] tag=%s  tissue=%s', tag, tissue)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── require liftover hits file ────────────────────────────────────────────
    if not liftover_csv.exists() or liftover_csv.stat().st_size == 0:
        log.error(
            'Liftover hits file not found or empty: %s\n'
            '  Run gwas_liftover_overlap.py --tissue %s --tag %s first.',
            liftover_csv, tissue, tag,
        )
        return

    # ── EFO parent-term map ───────────────────────────────────────────────────
    # out_dir is the tissue subdirectory (e.g. gwas_overlap/ctxhipp/); the EFO
    # map lives one level up in the shared gwas_overlap/ root.
    _efo_path = efo_map_path or (out_dir.parent / _EFO_MAPPINGS_FILE)
    efo_map = _load_efo_parent_map(_efo_path)

    # ── build trait→gene map from liftover positional hits ───────────────────
    # Each row is one (eGene × GWAS association) whose SNP physically falls
    # within the QTL CI lifted mm39→hg38.  No gene-name string matching.
    lo_df = pd.read_csv(liftover_csv)
    trait_col = 'MAPPED_TRAIT' if 'MAPPED_TRAIT' in lo_df.columns else 'DISEASE/TRAIT'
    log.info('  liftover hits: %d rows  (trait col = %s)', len(lo_df), trait_col)

    trait_gene_map:   dict[str, set[str]] = {}
    trait_parent_map: dict[str, str]      = {}
    gwas_all_genes:   set[str]            = set()

    for _, row in lo_df.iterrows():
        gene = str(row.get('gene_symbol', '')).strip().upper()
        if not gene:
            continue
        gwas_all_genes.add(gene)
        for trait_term in str(row.get(trait_col, '')).split('|'):
            trait_term = trait_term.strip()
            if not trait_term:
                continue
            parent = _trait_category(trait_term, efo_map)
            if parent is None:
                continue
            trait_gene_map.setdefault(trait_term, set()).add(gene)
            trait_parent_map[trait_term] = parent

    log.info('  %d eGenes with GWAS hits, %d specific EFO traits',
             len(gwas_all_genes), len(trait_gene_map))

    # ── expression universe (Fisher background) ───────────────────────────────
    _univ_default = (
        _VALIDATION / f'{tissue}_data' / 'nova' / f'exp_gene_info_{tissue}.csv'
    )
    _univ_path = universe_csv or _univ_default
    if _univ_path.exists():
        univ_df = pd.read_csv(_univ_path, usecols=['gene_symbol'])
        universe_set: set[str] = set(univ_df['gene_symbol'].dropna().str.upper())
        log.info('  expression universe: %d expressed genes', len(universe_set))
    else:
        log.warning(
            '  universe CSV not found (%s) — using GWAS gene set as background', _univ_path,
        )
        universe_set = gwas_all_genes.copy()
    N_univ = len(universe_set)
    trait_gene_univ: dict[str, set[str]] = {
        trait: genes & universe_set
        for trait, genes in trait_gene_map.items()
        if genes & universe_set
    }
    log.info('  %d EFO traits with ≥1 expressed gene in universe', len(trait_gene_univ))

    # ── collect CTs from partitions for this tag ──────────────────────────────
    all_celltypes: list[str] = sorted({
        ct
        for df in partitions.values()
        if not df.empty
        for ct in df[df['scan_tag'] == tag]['celltype'].unique()
    })

    summary_rows: list[dict] = []

    for ct in all_celltypes:
        ct_dir = out_dir / _safe_ct(ct)
        ct_dir.mkdir(parents=True, exist_ok=True)

        for cat in _CATEGORIES:
            df = partitions.get(cat, pd.DataFrame())
            if df.empty:
                continue
            sub = df[(df['scan_tag'] == tag) & (df['celltype'] == ct)]
            if sub.empty:
                continue

            gene_symbols = set(sub['gene_symbol'].dropna().str.upper())
            # Restrict to the expressed universe for all downstream statistics.
            # Genes outside the universe cannot appear in any trait_gene_univ
            # entry; including them in n_ct inflates the "in test set, not in
            # trait" Fisher cell and biases the odds ratio differently per CT
            # (CTs with many non-universe eGenes look artificially depleted).
            gene_symbols_univ = gene_symbols & universe_set
            overlap            = gene_symbols & gwas_all_genes   # raw coverage diagnostic
            frac               = len(overlap) / len(gene_symbols) if gene_symbols else 0.0
            log.info('  %-30s  [%s]  %-10s  %d/%d  (%.1f%%)',
                     ct, tag, cat, len(overlap), len(gene_symbols), frac * 100)

            _summary_row: dict = {
                'celltype':        ct,
                'category':        cat,
                'tag':             tag,
                'tissue':          tissue,
                'n_egenes':        len(gene_symbols),
                'n_egenes_univ':   len(gene_symbols_univ),
                'n_gwas_overlap':  len(overlap),
                'frac_overlap':    frac,
                'n_enriched_cats': 0,
            }

            # ── per-CT gene CSV ───────────────────────────────────────────────
            gene_rows = [
                {'gene_symbol': g, 'celltype': ct, 'category': cat}
                for g in sorted(overlap)
            ]
            pd.DataFrame(gene_rows).to_csv(
                ct_dir / f'gwas_genes_{cat}_{tissue}_{tag}.csv', index=False,
            )

            # ── EFO trait enrichment (Fisher's exact, BH-FDR) ────────────────
            # n_ct is the number of eGenes in the universe — the only genes
            # that can contribute to k.  Using the full gene_symbols set would
            # inflate n_ct - k for CTs with many non-universe eGenes.
            n_ct = len(gene_symbols_univ)
            if n_ct == 0:
                summary_rows.append(_summary_row)
                continue

            enrich_rows: list[dict] = []
            for trait, trait_genes in trait_gene_univ.items():
                k    = len(gene_symbols_univ & trait_genes)
                K_bg = len(trait_genes)
                if k == 0:
                    continue
                tbl = [
                    [k,         n_ct - k],
                    [K_bg - k,  N_univ - n_ct - K_bg + k],
                ]
                _, pval = fisher_exact(tbl, alternative='greater')
                enrich_rows.append({
                    'efo_trait': trait,
                    'parent':    trait_parent_map.get(trait, ''),
                    'k':         k,
                    'n_egenes':  n_ct,
                    'K_bg':      K_bg,
                    'N_univ':    N_univ,
                    'p_value':   pval,
                })

            if not enrich_rows:
                summary_rows.append(_summary_row)
                continue

            enrich_df = (
                pd.DataFrame(enrich_rows)
                .sort_values('p_value')
                .reset_index(drop=True)
            )
            # BH-FDR: m must be the TOTAL number of traits in the universe,
            # not just the subset with k>0 for this CT.  Using len(enrich_df)
            # would give smaller CTs lighter multiple-testing burden and make
            # them appear spuriously enriched compared to larger CTs that test
            # more traits.
            m_bh = len(trait_gene_univ)
            enrich_df['rank']          = np.arange(1, len(enrich_df) + 1)
            enrich_df['fdr']           = (enrich_df['p_value'] * m_bh / enrich_df['rank']).clip(upper=1.0)
            enrich_df['fdr']           = enrich_df['fdr'].iloc[::-1].cummin().iloc[::-1]
            enrich_df['neg_log10_fdr'] = -np.log10(enrich_df['fdr'].clip(lower=1e-300))
            enrich_df['overlap_label'] = (
                enrich_df['k'].astype(str) + '/' + enrich_df['K_bg'].astype(str)
            )
            enrich_df['Term'] = [
                textwrap.fill(f'{t} [{ov}]', width=52)
                for t, ov in zip(enrich_df['efo_trait'], enrich_df['overlap_label'])
            ]
            enrich_df.to_csv(
                ct_dir / f'gwas_enrich_{cat}_{tissue}_{tag}.csv', index=False,
            )

            sig = enrich_df[enrich_df['fdr'] < 0.05].copy()
            _summary_row['n_enriched_cats'] = len(sig)
            summary_rows.append(_summary_row)

            if sig.empty:
                log.info('    %-30s [%s] %s — no significant EFO traits', ct, tag, cat)
                continue

            sig = sig.sort_values('neg_log10_fdr', ascending=False).head(15).reset_index(drop=True)
            n_shown = len(sig)

            fig, ax = plt.subplots(
                figsize=(14, max(4, 0.38 * n_shown + 1.5)), dpi=300, facecolor='none',
            )
            palette = {p: _PARENT_COLORS.get(p, _DEFAULT_COLOR) for p in sig['parent'].unique()}
            sns.barplot(
                data=sig, x='neg_log10_fdr', y='Term', hue='parent',
                palette=palette, errorbar=None, dodge=False, ax=ax, width=0.35,
            )
            for i, (val, fdr_val) in enumerate(zip(sig['neg_log10_fdr'], sig['fdr'])):
                ax.text(val + 0.06, i, f'p={fdr_val:.1e}',
                        va='center', fontsize=10, color='#555')
            ax.axvline(-np.log10(0.05), color='#aaa', lw=1.0, ls='--')
            ax.set_xlabel('−log₁₀(FDR)', fontsize=13)
            ax.set_ylabel('EFO Term', fontsize=13)
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=11)
            ax.set_title(
                f'{ct.replace("_", " ")}  [{tissue} {tag}]  —  {cat} eQTLs  '
                f'(top {n_shown} enriched EFO traits)',
                fontsize=12,
            )
            leg = ax.legend(
                title='EFO Category', fontsize=10, title_fontsize=10,
                frameon=False, loc='lower right',
            )
            leg.get_title().set_fontweight('bold')
            ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
            sns.despine(ax=ax)
            fig.tight_layout()
            out_path = ct_dir / f'gwas_{cat}_{tissue}_{tag}.png'
            _save_fig(fig, out_path, dpi=300)
            plt.close(fig)
            log.info('    saved %s  (%d sig. traits)', out_path.name, n_shown)

    # ── summary bar chart: #enriched traits per CT × category ────────────────
    if not summary_rows:
        log.warning('No summary rows — check liftover CSV and cistrans partitions')
        return

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / f'gwas_summary_{tissue}_{tag}.csv', index=False)

    # Keep only CTs whose best category has at least min_bar_terms significant
    # traits.  CTs with n_enriched_cats=1 on an axis that goes to 20+ are
    # visually indistinguishable from zero and add clutter.
    ct_any_hit = set(
        sum_df.loc[sum_df['n_enriched_cats'] >= min_bar_terms, 'celltype'].unique()
    )
    ct_order  = sorted(ct for ct in sum_df['celltype'].unique() if ct in ct_any_hit)
    n_ct_plot = len(ct_order)
    cats_in_data = [c for c in _CATEGORIES if c in sum_df['category'].unique()]
    n_cats    = len(cats_in_data)
    width     = 0.25
    offsets   = np.linspace(-(n_cats - 1) / 2 * width, (n_cats - 1) / 2 * width, n_cats)

    fig, ax = plt.subplots(figsize=(max(13, n_ct_plot * 1.2), 5.5))
    x = np.arange(n_ct_plot)
    for offset, cat in zip(offsets, cats_in_data):
        sub_s = sum_df[sum_df['category'] == cat].set_index('celltype')
        vals  = [
            int(sub_s.loc[ct, 'n_enriched_cats']) if ct in sub_s.index else 0
            for ct in ct_order
        ]
        # Use np.nan for zero-hit (CT, category) pairs so ax.bar leaves those
        # slots blank — cleaner than a flush-against-zero bar that misleads the
        # eye into thinking the category was tested but returned nothing.
        vals_plot = [v if v > 0 else np.nan for v in vals]
        ax.bar(x + offset, vals_plot, width=width,
               color=CAT_COLORS.get(cat, _DEFAULT_COLOR), label=cat, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace('_', ' ') for c in ct_order],
        rotation=35, ha='right', fontsize=13,
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylabel('# enriched EFO traits (FDR < 0.05)', fontsize=15)
    ax.set_title(
        f'GWAS trait enrichment — {tissue}  [{tag}]\n'
        f"Fisher's exact vs expressed gene universe, BH-FDR < 0.05",
        fontsize=15,
    )
    ax.legend(fontsize=13, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    out_path = out_dir / f'gwas_overlap_{tissue}_{tag}.png'
    _save_fig(fig, out_path, dpi=150)
    plt.close(fig)
    log.info('Saved summary bar chart: %s', out_path.name)


def gwas_dotplot(
    out_dir: pathlib.Path,
    tag: str,
    tissue: str,
    cat: str,
    top_n_terms: int = 30,
    fdr_thresh: float = 0.05,
    min_ct_terms: int = 3,
) -> None:
    """Dot plot: EFO terms × cell types; dot size = −log₁₀(FDR).

    Reads per-CT ``gwas_enrich_{cat}_{tissue}_{tag}.csv`` files written by
    ``gwas_overlap()``.  Bubble matrix where:
      - rows  = top-N EFO terms grouped by parent category (color-coded)
      - cols  = cell types with ≥ min_ct_terms significant terms
      - size  = −log₁₀(FDR), power-1.8 scale for visual disparity
      - grey dots = non-significant (FDR ≥ threshold)
    """
    log.info('[gwas_dotplot] cat=%s  tissue=%s  tag=%s', cat, tissue, tag)

    # ── read per-CT enrichment CSVs ───────────────────────────────────────────
    ct_frames: dict[str, pd.DataFrame] = {}
    for ct_dir in sorted(out_dir.iterdir()):
        if not ct_dir.is_dir():
            continue
        csv = ct_dir / f'gwas_enrich_{cat}_{tissue}_{tag}.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        ct_frames[ct_dir.name.replace('_', ' ')] = df

    if not ct_frames:
        log.warning(
            'gwas_dotplot: no enrichment CSVs found in %s for cat=%s tissue=%s tag=%s',
            out_dir, cat, tissue, tag,
        )
        return

    # ── filter to CTs with ≥ min_ct_terms significant hits ───────────────────
    all_cts = sorted(
        ct for ct, df in ct_frames.items()
        if int((df['fdr'] < fdr_thresh).sum()) >= min_ct_terms
    )
    if not all_cts:
        log.warning('gwas_dotplot [%s]: no CTs with ≥%d significant terms', cat, min_ct_terms)
        return
    log.info('  %d CTs with ≥%d sig terms', len(all_cts), min_ct_terms)

    # ── union of top-N significant terms ─────────────────────────────────────
    term_best:   dict[str, float] = {}
    term_parent: dict[str, str]   = {}
    for ct in all_cts:
        df  = ct_frames[ct]
        sig = df[df['fdr'] < fdr_thresh]
        for _, row in sig.iterrows():
            t = str(row['efo_trait'])
            v = float(row['neg_log10_fdr'])
            if v > term_best.get(t, 0.0):
                term_best[t] = v
            if t not in term_parent:
                term_parent[t] = str(row.get('parent', ''))

    if not term_best:
        log.warning('gwas_dotplot [%s]: no significant terms found', cat)
        return

    top_terms = sorted(term_best, key=lambda t: -term_best[t])[:top_n_terms]

    # ── build value matrix ────────────────────────────────────────────────────
    rows_list: list[dict] = []
    for term in top_terms:
        row: dict = {
            'term':   term,
            'parent': term_parent.get(term, ''),
            '_best':  term_best[term],
        }
        for ct in all_cts:
            df    = ct_frames[ct]
            match = df[df['efo_trait'] == term]
            row[ct] = (
                float(match.iloc[0]['neg_log10_fdr'])
                if not match.empty and float(match.iloc[0]['fdr']) < fdr_thresh
                else 0.0
            )
        rows_list.append(row)

    mat = pd.DataFrame(rows_list).set_index('term')
    mat = mat.sort_values(['parent', '_best'], ascending=[True, False])
    mat = mat.drop(columns=['_best'])

    # ── size function ─────────────────────────────────────────────────────────
    thresh_line = -np.log10(fdr_thresh)
    max_val     = mat[all_cts].values.max()
    _MAX_AREA   = 600.0

    def _dot_size(v: float) -> float:
        if v < thresh_line or max_val <= 0:
            return 0.0
        return (v / max_val) ** 1.8 * _MAX_AREA

    # ── category block coordinates ────────────────────────────────────────────
    terms_list  = list(mat.index)
    parents_seq = [str(mat.loc[t, 'parent']) for t in terms_list]
    blocks: list[tuple[str, int, int]] = []
    cur_parent = parents_seq[0]
    cur_start  = 0
    for ri in range(1, len(terms_list)):
        if parents_seq[ri] != cur_parent:
            blocks.append((cur_parent, cur_start, ri - 1))
            cur_parent = parents_seq[ri]
            cur_start  = ri
    blocks.append((cur_parent, cur_start, len(terms_list) - 1))

    # ── plot ─────────────────────────────────────────────────────────────────
    n_terms = len(mat)
    n_cts   = len(all_cts)
    fig, ax = plt.subplots(
        figsize=(max(13, n_cts * 1.3 + 6), max(9, n_terms * 0.62 + 2)),
        dpi=300, facecolor='none',
    )

    for ci, ct in enumerate(all_cts):
        for ri, term in enumerate(terms_list):
            val    = float(mat.loc[term, ct])
            parent = str(mat.loc[term, 'parent'])
            color  = _PARENT_COLORS.get(parent, _DEFAULT_COLOR)
            if val >= thresh_line:
                ax.scatter(ci, ri, s=_dot_size(val), color=color,
                           alpha=0.85, zorder=3, linewidths=0)
            else:
                ax.scatter(ci, ri, s=15, color='#d0d0d0',
                           alpha=0.45, zorder=2, linewidths=0)

    # ── category bars + labels ────────────────────────────────────────────────
    # Use a blended transform: x in axes fraction, y in data coords.
    # BAR_X=1.02 means the bar is always 2% of the axes width past the right
    # edge — a constant visual gap regardless of how many CTs are shown.
    bar_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    BAR_X  = 1.02   # bar line position  (axes fraction)
    TEXT_X = 1.05   # label text anchor  (axes fraction)
    for parent, start, end in blocks:
        bar_color = _PARENT_COLORS.get(parent, _DEFAULT_COLOR)
        if start > 0:
            ax.axhline(start - 0.5, color='#bbbbbb', linewidth=0.8,
                       zorder=1, linestyle='--')
        ax.plot(
            [BAR_X, BAR_X],
            [start - 0.4, end + 0.4],
            color=bar_color, linewidth=4, clip_on=False,
            solid_capstyle='round', transform=bar_trans,
        )
        words        = parent.split()
        target_width = max(max(len(w) for w in words), len(parent) // 2 + 1)
        lines        = textwrap.wrap(parent, width=target_width)[:3]
        ax.text(
            TEXT_X, (start + end) / 2.0, '\n'.join(lines),
            ha='left', va='center', fontsize=11, color='black',
            fontweight='bold', clip_on=False, transform=bar_trans,
        )

    ax.set_xticks(range(n_cts))
    ax.set_xticklabels(
        [c.replace('_', ' ') for c in all_cts],
        rotation=40, ha='right', fontsize=16,
    )
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels(
        [textwrap.fill(t, width=45) for t in terms_list], fontsize=12,
    )
    ax.set_xlim(-0.6, n_cts - 0.4)
    ax.set_ylim(-0.6, n_terms - 0.4)
    ax.set_title(
        f'GWAS trait enrichment — {cat} eGenes  [{tissue} {tag}]',
        fontsize=18,
    )

    # Bubble-size legend: placed below the axes so it does not compete with the
    # category-bar annotations on the right.  ncol=len(ref_vals) keeps it
    # horizontal and compact.
    ref_vals = [v for v in (2, 4, 6, 8, 10) if v <= max_val + 0.5]
    size_handles = [
        plt.scatter([], [], s=_dot_size(v), color='#555555',
                    alpha=0.85, label=str(int(v)))
        for v in ref_vals
    ]
    # Legend: placed in axes coordinates (bbox_to_anchor in axes fraction) so
    # the gap from the x-tick label area is proportional to axes height — not
    # figure height.  The previous fig.transFigure approach fixed the legend at
    # y=-0.01 of figure coords while bottom=0.18 reserved 18 % of figure height
    # for tick labels; for tall plots (many terms) this produced a 1-3 inch
    # whitespace gap.  Using ax.legend() with a moderate negative-y anchor keeps
    # the legend snug below the tick labels on plots of any height.
    # bbox_inches='tight' in _save_fig always captures the legend even when it
    # falls outside the axes bounding box.
    sns.despine(ax=ax)
    fig.subplots_adjust(right=0.80)   # right margin only — for category-bar labels
    leg = ax.legend(
        handles=size_handles,
        title=f'−log₁₀(FDR)  (FDR < {fdr_thresh})',
        fontsize=12, title_fontsize=12,
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.14),  # axes fraction: just below the rotated tick labels
        ncol=len(ref_vals),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    leg.get_title().set_fontweight('bold')
    out_path = out_dir / f'gwas_dotplot_{cat}_{tissue}_{tag}.png'
    _save_fig(fig, out_path, dpi=300)
    plt.close(fig)
    log.info('Saved %s  (%d terms × %d CTs)', out_path.name, n_terms, n_cts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            'GWAS overlap + EFO enrichment for mosaiQTL eGene sets '
            '(matching / novel / cistrans).'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '--tissue', required=True,
        help='Tissue identifier, e.g. ctxhipp.  Included in all output filenames.',
    )
    ap.add_argument(
        '--tag', required=True,
        help='Scan tag, e.g. Lw1.',
    )
    ap.add_argument(
        '--liftover-csv', default=None,
        help=(
            'Path to gwas_liftover_hits_{tissue}_{tag}.csv.  '
            'Default: {out_dir}/gwas_liftover_hits_{tissue}_{tag}.csv'
        ),
    )
    ap.add_argument(
        '--cistrans-dir', default=None,
        help=(
            'Directory containing cistrans CSVs from cistrans_overlap.py.  '
            'Default: validation/{tissue}_data/cistrans_comp'
        ),
    )
    ap.add_argument(
        '--universe-csv', default=None,
        help=(
            'CSV with gene_symbol column (expressed gene universe for Fisher).  '
            'Default: validation/{tissue}_data/nova/exp_gene_info_{tissue}.csv'
        ),
    )
    ap.add_argument(
        '--efo-map', default=None,
        help=(
            'Path to gwas-efo-trait-mappings.tsv.  '
            'Default: {out_dir}/gwas-efo-trait-mappings.tsv'
        ),
    )
    ap.add_argument(
        '--out-dir', default=None,
        help=(
            'Output root directory.  '
            'Default: validation/gwas_overlap/{tissue}/ '
            '(tissue subdir keeps runs for different tissues isolated).'
        ),
    )
    ap.add_argument(
        '--top-n-terms', type=int, default=30,
        help='Maximum EFO terms shown per dot plot.',
    )
    ap.add_argument(
        '--min-ct-terms', type=int, default=3,
        help='Minimum significant terms a CT must have to appear in the dot plot.',
    )
    ap.add_argument(
        '--no-dotplot', action='store_true',
        help='Skip dot-plot generation.',
    )
    args = ap.parse_args()

    # Default output dir includes tissue so different tissues never clobber each
    # other's outputs.  The liftover CSV lives one level up (the shared
    # gwas_overlap/ root) so its default is NOT derived from out_dir.
    _gwas_root = _VALIDATION / 'gwas_overlap'
    out_dir = (
        pathlib.Path(args.out_dir) if args.out_dir
        else _gwas_root / args.tissue
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    liftover_csv = (
        pathlib.Path(args.liftover_csv) if args.liftover_csv
        else _gwas_root / f'gwas_liftover_hits_{args.tissue}_{args.tag}.csv'
    )
    cistrans_dir = (
        pathlib.Path(args.cistrans_dir) if args.cistrans_dir
        else _VALIDATION / f'{args.tissue}_data' / 'cistrans_comp'
    )
    universe_csv = pathlib.Path(args.universe_csv) if args.universe_csv else None
    efo_map_path = pathlib.Path(args.efo_map) if args.efo_map else None

    log.info('gwas_overlap.py  tissue=%s  tag=%s', args.tissue, args.tag)
    log.info('  liftover CSV : %s', liftover_csv)
    log.info('  cistrans dir : %s', cistrans_dir)
    log.info('  output dir   : %s', out_dir)

    partitions = _load_cistrans_partitions(cistrans_dir, args.tag)
    for cat, df in partitions.items():
        log.info('  %s: %d rows', cat, len(df))

    gwas_overlap(
        partitions, liftover_csv, out_dir, args.tag, args.tissue,
        efo_map_path=efo_map_path,
        universe_csv=universe_csv,
        min_bar_terms=args.min_ct_terms,
    )

    if not args.no_dotplot:
        for cat in _CATEGORIES:
            gwas_dotplot(
                out_dir, args.tag, args.tissue,
                cat=cat,
                top_n_terms=args.top_n_terms,
                min_ct_terms=args.min_ct_terms,
            )

    log.info('Done — outputs in %s', out_dir)


if __name__ == '__main__':
    main()
