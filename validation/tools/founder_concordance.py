"""founder_concordance.py — external concordance of mosaiQTL Lw1 founder effects.

For every cell type x non-reference founder strain, compares the mosaiQTL founder
contrast at each gene's Lw1 peak, ``effect_B6J - effect_<strain>``, against:

  f1_hyb  F1 allele-specific expression, Hyblog2FC (cis only)
  f1_par  parental expression difference, Parlog2FC (cis + trans)
  de      8cube founder-vs-founder DE logFC, ``B6J_vs_<strain>`` (cis + trans)

All three external signals are oriented B6J minus strain. Orientation was calibrated
against the data (Parlog2FC vs. 8cube B6J_vs_AJ logFC, rho = -0.85 vs. AJ - B6J) and is
re-checked per strain at run time.

mosaiQTL effects are not on a log2FC scale, so tests use sign agreement and Spearman
rank correlation only. The peak CSV already holds only genes clearing the genome-wide
LOD gate (the ``genome`` row of eqtl_stats), so every row is an eGene at that gate; LOD
quartiles give a threshold-free dose-response. The 8cube DE table is a joint fit that
also contains CC samples, so the
``de`` source is exploratory; the binomial p-values treat genes as independent and are
informational only.

Usage:
    conda run -n bioinfo python validation/tools/founder_concordance.py
"""

import json
import logging
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted
from scipy import stats

from mosaiqtl.genotype.founders import FOUNDER_ORDER

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO.parents[1]
SUBMISSION = REPO / 'validation' / 'ctxhipp_data' / 'submission'
OUT_DIR = REPO / 'validation' / 'ctxhipp_data' / 'founder_concordance'
F1_CSV = ANALYSIS / 'input' / 'gene_lists' / 'cistrans_genes.csv'
ANNOT_JSON = ANALYSIS / 'preprocessing' / 'input' / 'annotations' / 'ctxhipp_all.json'
DE_DIR = ANALYSIS / '8cube_manuscript' / 'output' / 'degs' / 'CortexHippocampus'

SCAN_TAG = 'Lw1'
TISSUE = 'CortexHippocampus'
REF = 'B6J'
STRAINS = [f for f in FOUNDER_ORDER if f != REF]
F1_STRAIN = {
    'A_J': 'AJ',
    '129S1_SvImJ': '129S1J',
    'NOD_ShiLtJ': 'NODJ',
    'NZO_HlLtJ': 'NZOJ',
    'CAST_EiJ': 'CASTJ',
    'PWK_PhJ': 'PWKJ',
    'WSB_EiJ': 'WSBJ',
}
FDR_ALPHA = 0.05
MIN_N = 30
DE_CHUNK = 500_000
LOD_BINS = ['lod_q1', 'lod_q2', 'lod_q3', 'lod_q4']
REG_CATS = ['cis', 'cis+trans', 'cisxtrans', 'trans', 'conserved']

# source -> (value column, significance column or None)
SOURCES = {
    'f1_hyb': ('f1_hyb', 'f1_fdr_cis'),
    'f1_par': ('f1_par', None),
    'de': ('de_logfc', 'de_fdr'),
}

for _font in ('Carlito', 'Calibri', 'DejaVu Sans'):
    if _font in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        mpl.rcParams['font.family'] = _font
        break


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '_', s.lower())


def load_subtype_map() -> dict[str, str]:
    """F1 subtype name -> normalised cell type name."""
    with open(ANNOT_JSON) as fh:
        annot = json.load(fh)
    return {
        _norm(k): _norm(v)
        for k, v in annot['subtype_to_celltype'].items()
        if v != 'low quality'
    }


def load_mosaiqtl(celltype: str) -> pd.DataFrame:
    """Per-gene Lw1 peak (genes clearing the genome-wide LOD gate) with one B6J-minus-strain
    contrast per strain."""
    d = SUBMISSION / f'{celltype}_{SCAN_TAG}'
    peak = pd.read_csv(d / f'eqtl_peak_{celltype}_{SCAN_TAG}.csv')
    stats_df = pd.read_csv(d / f'eqtl_stats_{celltype}_{SCAN_TAG}.csv')
    genome = stats_df[stats_df['chr'].astype(str) == 'genome'].iloc[0]
    log.info(
        '%s: %d peak genes (genome row: %d eGenes of %d scanned, LOD >= %.2f; min peak LOD %.2f)',
        celltype,
        len(peak),
        genome['n_egenes'],
        genome['n_genes_scanned'],
        genome['lod_threshold'],
        peak['lod'].min(),
    )
    peak['chr'] = peak['chr'].astype(str)
    peak = peak[~peak['gene_symbol'].duplicated(keep=False)]
    for s in STRAINS:
        peak[f'mq_{s}'] = peak[f'effect_{REF}'] - peak[f'effect_{s}']
    cols = ['gene_symbol', 'chr', 'lod'] + [f'mq_{s}' for s in STRAINS]
    return peak[cols]


def load_f1(subtype_map: dict[str, str]) -> pd.DataFrame:
    """F1 hybrid table, long by gene x cell type x strain, unique symbols only."""
    w = pd.read_csv(
        F1_CSV,
        usecols=[
            'gene',
            'subtype',
            'tissue',
            'strain',
            'Parlog2FC',
            'Hyblog2FC',
            'fdr_cis',
            'reg_assignment',
        ],
    )
    w = w[w['tissue'] == TISSUE].copy()
    w['celltype'] = w['subtype'].map(lambda s: subtype_map.get(_norm(s)))
    w['strain'] = w['strain'].map(F1_STRAIN)
    w = w.dropna(subset=['celltype', 'strain'])
    w = w.rename(
        columns={
            'gene': 'gene_symbol',
            'Parlog2FC': 'f1_par',
            'Hyblog2FC': 'f1_hyb',
            'fdr_cis': 'f1_fdr_cis',
            'reg_assignment': 'f1_reg',
        }
    )
    key = ['celltype', 'strain', 'gene_symbol']
    w = w[~w.duplicated(key, keep=False)]
    return w[key + ['f1_par', 'f1_hyb', 'f1_fdr_cis', 'f1_reg']]


def load_de(celltype: str) -> pd.DataFrame:
    """8cube founder DE rows B6J_vs_<strain> for one cell type (logFC = B6J - strain)."""
    files = {
        _norm(p.stem.removeprefix('results_de_analysis_')): p
        for p in DE_DIR.glob('results_de_analysis_*.csv')
    }
    if celltype not in files:
        return pd.DataFrame(columns=['strain', 'gene_symbol', 'de_logfc', 'de_fdr'])
    # The table stores one direction per pair; flip to B6J - strain where needed.
    wanted = {f'{REF}_vs_{s}': (s, 1.0) for s in STRAINS}
    wanted.update({f'{s}_vs_{REF}': (s, -1.0) for s in STRAINS})
    parts = []
    for chunk in pd.read_csv(
        files[celltype],
        usecols=['gene_name', 'comparison', 'logFC', 'FDR'],
        chunksize=DE_CHUNK,
    ):
        parts.append(chunk[chunk['comparison'].isin(wanted)])
    de = pd.concat(parts)
    de['strain'] = de['comparison'].map(lambda c: wanted[c][0])
    de['logFC'] = de['logFC'] * de['comparison'].map(lambda c: wanted[c][1])
    de = de.rename(
        columns={'gene_name': 'gene_symbol', 'logFC': 'de_logfc', 'FDR': 'de_fdr'}
    )
    de = de[~de.duplicated(['strain', 'gene_symbol'], keep=False)]
    return de[['strain', 'gene_symbol', 'de_logfc', 'de_fdr']]


def build_long(celltype: str, f1: pd.DataFrame) -> pd.DataFrame:
    """One row per gene x strain with mosaiQTL, F1 and 8cube DE values."""
    mq = load_mosaiqtl(celltype)
    long = mq.melt(
        id_vars=['gene_symbol', 'chr', 'lod'],
        value_vars=[f'mq_{s}' for s in STRAINS],
        var_name='strain',
        value_name='mq',
    )
    long['strain'] = long['strain'].str.removeprefix('mq_')
    w = f1[f1['celltype'] == celltype].drop(columns='celltype')
    long = long.merge(w, on=['strain', 'gene_symbol'], how='left')
    long = long.merge(load_de(celltype), on=['strain', 'gene_symbol'], how='left')
    long.insert(0, 'celltype', celltype)
    return long


def concordance(df: pd.DataFrame, a: str, b: str) -> dict | None:
    """Sign agreement and Spearman rho between columns a and b; None if n < MIN_N."""
    x = df[[a, b]].dropna()
    x = x[(x[a] != 0) & (x[b] != 0)]
    n = len(x)
    if n < MIN_N:
        return None
    agree = int((np.sign(x[a]) == np.sign(x[b])).sum())
    return {
        'n': n,
        'sign_agree': agree / n,
        'rho': stats.spearmanr(x[a], x[b]).statistic,
        'p_binom': stats.binomtest(agree, n, 0.5).pvalue,
    }


def summarise(df: pd.DataFrame, lod_edges: np.ndarray) -> list[dict]:
    """Concordance of mosaiQTL against every source within every stratum."""
    lod_bin = pd.cut(df['lod'], lod_edges, labels=LOD_BINS, include_lowest=True)
    rows = []
    for source, (val_col, sig_col) in SOURCES.items():
        sets = {'all': df}
        if sig_col:
            sets['source_sig'] = df[df[sig_col] < FDR_ALPHA]
        for lb in LOD_BINS:
            sets[lb] = df[lod_bin == lb]
        for cat in REG_CATS:
            sets[f'reg_{cat}'] = df[df['f1_reg'] == cat]
        for stratum, sub in sets.items():
            res = concordance(sub, 'mq', val_col)
            if res:
                rows.append({'source': source, 'stratum': stratum, **res})
    return rows


def summarise_all(long: pd.DataFrame) -> pd.DataFrame:
    """Summaries per cell type x strain, plus pooled ('ALL') margins."""
    lod_edges = np.quantile(long['lod'], [0, 0.25, 0.5, 0.75, 1])
    out = []
    groupings = [
        (['celltype', 'strain'], None),
        (['strain'], 'celltype'),
        (['celltype'], 'strain'),
        ([], 'both'),
    ]
    for keys, pooled in groupings:
        groups = long.groupby(keys) if keys else [((), long)]
        for key, df in groups:
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(keys, key))
            if pooled in ('celltype', 'both'):
                labels['celltype'] = 'ALL'
            if pooled in ('strain', 'both'):
                labels['strain'] = 'ALL'
            for row in summarise(df, lod_edges):
                out.append({**labels, **row})
    return pd.DataFrame(out)[
        ['celltype', 'strain', 'source', 'stratum', 'n', 'sign_agree', 'rho', 'p_binom']
    ]


def check_orientation(long: pd.DataFrame) -> None:
    """F1 parental and 8cube DE must agree in sign for every strain."""
    for strain, df in long.groupby('strain'):
        res = concordance(df, 'f1_par', 'de_logfc')
        if res is None:
            log.warning('orientation check %s: too few genes', strain)
        elif res['rho'] <= 0:
            log.warning(
                'orientation check %s: rho=%.2f <= 0, sign convention suspect',
                strain,
                res['rho'],
            )
        else:
            log.info('orientation check %s: rho=%.2f (n=%d)', strain, res['rho'], res['n'])


def _save(fig: plt.Figure, stem: str) -> None:
    """Transparent PNG for slides plus an opaque _share PNG."""
    plain = OUT_DIR / f'{stem}.png'
    fig.savefig(plain, dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(
        plain.with_name(f'{stem}_share.png'),
        dpi=300,
        transparent=False,
        facecolor='white',
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_rho_heatmaps(summary: pd.DataFrame, stratum: str = 'all') -> None:
    """Celltype x strain Spearman rho, one panel per source."""
    sub = summary[
        (summary['stratum'] == stratum)
        & (summary['celltype'] != 'ALL')
        & (summary['strain'] != 'ALL')
    ]
    cts = natsorted(sub['celltype'].unique())
    fig, axes = plt.subplots(
        1, len(SOURCES), figsize=(4.6 * len(SOURCES), 0.45 * len(cts) + 2), sharey=True
    )
    for ax, source in zip(axes, SOURCES):
        piv = (
            sub[sub['source'] == source]
            .pivot(index='celltype', columns='strain', values='rho')
            .reindex(index=cts, columns=STRAINS)
        )
        sns.heatmap(
            piv,
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0,
            cmap='vlag',
            annot=True,
            fmt='.2f',
            annot_kws={'size': 7},
            cbar=ax is axes[-1],
            cbar_kws={'label': 'Spearman rho'},
        )
        ax.set_title(source)
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.suptitle(f'mosaiQTL {SCAN_TAG} vs. external founder signal ({stratum} genes)')
    _save(fig, f'rho_heatmap_{stratum}')


def plot_lod_dose_response(summary: pd.DataFrame) -> None:
    """Sign agreement across pooled LOD quartiles, one line per source."""
    sub = summary[
        (summary['celltype'] == 'ALL')
        & (summary['strain'] == 'ALL')
        & (summary['stratum'].isin(LOD_BINS))
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(
        data=sub, x='stratum', y='sign_agree', hue='source', marker='o', ax=ax
    )
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('mosaiQTL peak LOD quartile (low to high)')
    ax.set_ylabel('sign agreement')
    sns.despine(ax=ax)
    _save(fig, 'sign_agreement_by_lod')


def plot_reg_concordance(summary: pd.DataFrame) -> None:
    """Pooled sign agreement by F1 regulatory category, one bar per source."""
    sub = summary[
        (summary['celltype'] == 'ALL')
        & (summary['strain'] == 'ALL')
        & (summary['stratum'].isin([f'reg_{c}' for c in REG_CATS]))
    ].copy()
    sub['category'] = sub['stratum'].str.removeprefix('reg_')
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(
        data=sub, x='category', y='sign_agree', hue='source', order=REG_CATS, ax=ax
    )
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('F1 regulatory category (reg_assignment)')
    ax.set_ylabel('sign agreement with mosaiQTL')
    sns.despine(ax=ax)
    _save(fig, 'sign_agreement_by_reg_assignment')


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s',
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1 = load_f1(load_subtype_map())
    celltypes = natsorted(
        p.name.removesuffix(f'_{SCAN_TAG}') for p in SUBMISSION.glob(f'*_{SCAN_TAG}')
    )
    long = pd.concat([build_long(ct, f1) for ct in celltypes], ignore_index=True)
    log.info('%d gene x strain x celltype rows', len(long))
    long.to_csv(OUT_DIR / 'concordance_long.csv.gz', index=False)

    check_orientation(long)
    summary = summarise_all(long)
    summary.to_csv(OUT_DIR / 'concordance_summary.csv', index=False)
    plot_rho_heatmaps(summary, 'all')
    plot_lod_dose_response(summary)
    plot_reg_concordance(summary)
    log.info('wrote outputs to %s', OUT_DIR)


if __name__ == '__main__':
    main()
