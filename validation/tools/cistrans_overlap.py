"""cistrans_overlap.py — compare mosaiQTL eGenes against F1 ASE cis-gene list.

Usage:
    conda run -n bioinfo python validation/tools/cistrans_overlap.py
"""

import re
import json
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ── font ──────────────────────────────────────────────────────────────────────
for _font in ('Carlito', 'Calibri', 'DejaVu Sans'):
    if _font in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        mpl.rcParams['font.family'] = _font
        break

# ── paths ─────────────────────────────────────────────────────────────────────
data_path       = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis'
nova_dir        = f'{data_path}/qtl/mosaiqtl/validation/ctxhipp_data/nova/celltypes'
cistrans_csv    = f'{data_path}/input/gene_lists/cistrans_genes.csv'
annotation_json = f'{data_path}/preprocessing/input/annotations/ctxhipp_all.json'
output_path     = f'{data_path}/qtl/mosaiqtl/validation/ctxhipp_data/cistrans_comp'

pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
SCAN_TAGS = ['C', 'Lw1']

FOCUS_CELLTYPES = [
    'glutamatergic_neuron',
    'gabaergic_interneuron',
    'astrocyte',
    'microglial_cell',
    'oligodendrocyte',
    'dentate_gyrus_granule_cell',
]

EXCLUDE_CELLTYPES = {'cajal_retzius_cell'}

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '_', s.lower())

with open(annotation_json) as _f:
    _annot = json.load(_f)

SUBTYPE_TO_CELLTYPE: dict[str, str] = {
    _norm(k): _norm(v)
    for k, v in _annot['subtype_to_celltype'].items()
    if v != 'low quality'
}

STRAIN_ALIAS: dict[str, str] = {
    'A_J':         'AJ',
    '129S1_SvImJ': '129S1J',
    'NOD_ShiLtJ':  'NODJ',
    'NZO_HlLtJ':   'NZOJ',
    'CAST_EiJ':    'CASTJ',
    'PWK_PhJ':     'PWKJ',
    'WSB_EiJ':     'WSBJ',
}

CIS_RANK = {'cis': 0, 'cisxtrans': 1, 'cis+trans': 2, 'trans': 3, 'conserved': 4}

CAT_COLORS = {
    'matching':   '#2166ac',   # cis-only in F1, confirmed cis in mosaiQTL
    'cistrans':   '#4dac26',   # F1 shows cis+trans or cisxtrans — cis component confirmed
    'novel':      '#e08214',   # not in F1 list at all
    'discordant': '#d01c8b',   # in F1 list but trans or conserved — genuinely contradictory
    'missed':     '#b2b2b2',
}

# ── load data ─────────────────────────────────────────────────────────────────
print('loading cistrans_genes.csv …')
ct_raw = pd.read_csv(
    cistrans_csv,
    usecols=['gene', 'subtype', 'tissue', 'strain',
             'reg_assignment', 'fdr_cis', 'fdr_trans', 'cis_prop_reordered'],
)
ct = ct_raw[ct_raw['tissue'] == 'CortexHippocampus'].copy()
print(f'{len(ct):,} rows  |  {ct["gene"].nunique():,} genes')

ct['celltype_norm'] = ct['subtype'].map(lambda s: SUBTYPE_TO_CELLTYPE.get(_norm(s)))
ct['strain_norm']   = ct['strain'].map(STRAIN_ALIAS)
ct = ct[
    ct['celltype_norm'].notna() &
    ct['strain_norm'].notna() &
    ~ct['celltype_norm'].isin(EXCLUDE_CELLTYPES)
]
print(f'celltypes: {sorted(ct["celltype_norm"].unique())}')
print(f'strains:   {sorted(ct["strain_norm"].unique())}')

# aggregate across strains — consensus reg_assignment, min FDR, mean cis_prop
def _consensus(series):
    counts = series.value_counts()
    ties   = counts[counts == counts.iloc[0]].index.tolist()
    return min(ties, key=lambda x: CIS_RANK.get(x, 99))

ct_agg = ct.groupby(['gene', 'celltype_norm']).agg(
    reg_assignment = ('reg_assignment',    _consensus),
    fdr_cis_min    = ('fdr_cis',           'min'),
    fdr_trans_min  = ('fdr_trans',          'min'),
    cis_prop_mean  = ('cis_prop_reordered', 'mean'),
    n_strains      = ('strain_norm',        'nunique'),
).reset_index()
ct_agg['is_f1_cis'] = ct_agg['fdr_cis_min'] < 0.05
print(ct_agg['reg_assignment'].value_counts().to_string())

print('loading eqtl_summary files (significant eGenes only) …')
records = []
for p in sorted(pathlib.Path(nova_dir).glob('*/*/eqtl_summary.csv')):
    celltype = p.parts[-3]
    tag      = p.parts[-2]
    if tag in SCAN_TAGS:
        df = pd.read_csv(p)
        df['celltype'] = celltype
        df['scan_tag'] = tag
        records.append(df)

peaks = pd.concat(records, ignore_index=True)
print(f'{len(peaks):,} significant eGenes  |  {peaks["celltype"].nunique()} cell types  '
      f'|  tags: {sorted(peaks["scan_tag"].unique())}')

# ── merge ─────────────────────────────────────────────────────────────────────
merged = peaks.merge(
    ct_agg.rename(columns={'gene': 'gene_symbol', 'celltype_norm': 'celltype'}),
    on=['gene_symbol', 'celltype'],
    how='left',
)
merged['in_f1_list'] = merged['reg_assignment'].notna()

# matching   = pure cis only: F1 confirms cis (is_f1_cis) AND reg_assignment == 'cis'
# cistrans   = in F1 list AND reg_assignment in ('cis+trans', 'cisxtrans') — cis component
#              confirmed but a trans component also exists; mosaiQTL partially right
# discordant = in F1 list AND reg_assignment in ('trans', 'conserved') — genuinely
#              contradictory: F1 finds no cis signal or purely trans regulation
# novel      = not in F1 list at all
_pure_cis    = (merged['in_f1_list']
                & (merged['is_f1_cis'] == True)
                & (merged['reg_assignment'] == 'cis'))
_is_cistrans = (merged['in_f1_list']
                & merged['reg_assignment'].isin(['cis+trans', 'cisxtrans']))
_is_discord  = merged['in_f1_list'] & ~_pure_cis & ~_is_cistrans

matching   = merged[_pure_cis].copy()
cistrans   = merged[_is_cistrans].copy()
discordant = merged[_is_discord].copy()
novel      = merged[~merged['in_f1_list']].copy()

assert len(matching) + len(cistrans) + len(discordant) + len(novel) == len(merged), \
    'categories not a partition'
print(f'total   {len(merged):,}  |  matching {len(matching):,}  '
      f'|  cistrans {len(cistrans):,}  |  novel {len(novel):,}  '
      f'|  discordant {len(discordant):,}')

# cistrans_pr_*.png — removed pending redesign

# ── overlap: breakdown stacked bar ────────────────────────────────────────────
for tag in SCAN_TAGS:
    sub = merged[merged['scan_tag'] == tag].copy()
    cat_data = []
    for ct_name, grp in sub.groupby('celltype'):
        f1_cis_set = set(ct_agg.loc[
            (ct_agg['celltype_norm'] == ct_name) & ct_agg['is_f1_cis'], 'gene'])
        cat_data.append({
            'celltype':   ct_name,
            'matching':   int(_pure_cis[grp.index].sum()),
            'cistrans':   int(_is_cistrans[grp.index].sum()),
            'novel':      int((~grp['in_f1_list']).sum()),
            'discordant': int(_is_discord[grp.index].sum()),
            'missed':     len(f1_cis_set - set(grp['gene_symbol'])),
        })
    cat_df = pd.DataFrame(cat_data).set_index('celltype')
    total = cat_df['matching'] + cat_df['cistrans'] + cat_df['novel'] + cat_df['discordant']
    cat_df = cat_df.loc[total.sort_values(ascending=True).index]  # ascending so barh shows highest at top

    BREAKDOWN_CATS = ['matching', 'cistrans', 'novel', 'discordant']
    fig, ax = plt.subplots(figsize=(13, max(4, len(cat_df) * 0.5)), dpi=300)
    lefts = np.zeros(len(cat_df))
    for cat in BREAKDOWN_CATS:
        color = CAT_COLORS[cat]
        vals = cat_df[cat].values
        ax.barh(range(len(cat_df)), vals, left=lefts,
                color=color, edgecolor='none', height=0.35, label=cat)
        lefts += vals
    ax.set_yticks(range(len(cat_df)))
    ax.set_yticklabels([c.replace('_', ' ') for c in cat_df.index], fontsize=13)
    ax.set_xlabel('number of eGenes', fontsize=14)
    ax.tick_params(axis='x', labelsize=13)
    ax.set_title(f'[{tag}]  overlap breakdown by cell type', fontsize=15)
    ax.legend(loc='lower right', frameon=False, fontsize=13)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(f'{output_path}/cistrans_breakdown_{tag}.png', dpi=300,
                bbox_inches='tight', transparent=True)
    plt.close()
    print(f'saved cistrans_breakdown_{tag}.png')

# ── validation: concordance scatter — mosaiQTL peak LOD vs F1 cis_prop_mean ───
# JointGrid scatter coloured by peak_lod per partition (validated=Blues,
# discordant=Oranges), with overlaid KDE marginals — same idiom as compare.py
# lod_scatter but showing two orthogonal datasets on one set of axes.
#   X (mosaiQTL) : peak_lod         — how strong is our scan signal?
#   Y (F1 ASE)   : cis_prop_mean    — what fraction of ASE variance is cis in F1?
# Novel genes have no F1 data (cis_prop_mean = NaN) so are excluded.
import matplotlib.cm as _mpl_cm
import matplotlib.colors as _mpl_colors

sns.set_style('ticks')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Carlito', 'Calibri', 'DejaVu Sans'],
    'font.size': 11,
})

_v_sub  = matching.dropna(subset=['peak_lod', 'cis_prop_mean'])
_ct_sub = cistrans.dropna(subset=['peak_lod', 'cis_prop_mean'])
_d_sub  = discordant.dropna(subset=['peak_lod', 'cis_prop_mean'])

_lod_min = min(s['peak_lod'].min() for s in [_v_sub, _ct_sub, _d_sub] if len(s))
_lod_max = max(s['peak_lod'].max() for s in [_v_sub, _ct_sub, _d_sub] if len(s))

def _norm_lod(s: 'pd.Series') -> np.ndarray:
    return ((s - _lod_min) / (_lod_max - _lod_min + 1e-9)).clip(0, 1).values

_v_colors  = _mpl_cm.Blues  (0.35 + 0.5 * _norm_lod(_v_sub['peak_lod']))   # mid→dark blue
_ct_colors = _mpl_cm.Greens (0.35 + 0.5 * _norm_lod(_ct_sub['peak_lod']))  # mid→dark green
_d_colors  = _mpl_cm.Oranges(0.35 + 0.5 * _norm_lod(_d_sub['peak_lod']))   # mid→dark orange

_all_sub = pd.concat([_v_sub, _ct_sub, _d_sub])
_x_lim   = (_all_sub['peak_lod'].min() * 0.95, _all_sub['peak_lod'].max() * 1.02)

g = sns.JointGrid(data=_all_sub, x='peak_lod', y='cis_prop_mean', height=6)

# matching scatter (Blues gradient)
g.ax_joint.scatter(
    _v_sub['peak_lod'].values, _v_sub['cis_prop_mean'].values,
    c=_v_colors, s=5, alpha=0.35, linewidths=0, rasterized=True,
    label=f'matching  (n={len(_v_sub):,})',
)
# cistrans scatter (Greens gradient)
g.ax_joint.scatter(
    _ct_sub['peak_lod'].values, _ct_sub['cis_prop_mean'].values,
    c=_ct_colors, s=5, alpha=0.35, linewidths=0, rasterized=True,
    label=f'cistrans  (n={len(_ct_sub):,})',
)
# discordant scatter (Oranges gradient)
g.ax_joint.scatter(
    _d_sub['peak_lod'].values, _d_sub['cis_prop_mean'].values,
    c=_d_colors, s=5, alpha=0.35, linewidths=0, rasterized=True,
    label=f'discordant  (n={len(_d_sub):,})',
)

# regression lines per partition
for sub, col in [(_v_sub, '#2166ac'), (_ct_sub, '#4dac26'), (_d_sub, '#d6600d')]:
    if len(sub) > 1:
        _m, _b = np.polyfit(sub['peak_lod'], sub['cis_prop_mean'], 1)
        _xr = np.linspace(*_x_lim, 200)
        g.ax_joint.plot(_xr, _m * _xr + _b, color=col, lw=1.4, alpha=0.9)

# reference line at cis_prop = 0
g.ax_joint.axhline(0, color='#888888', lw=0.8, ls='--', alpha=0.6)

# correlation annotations
_rv,  _pv  = sp_stats.pearsonr(_v_sub['peak_lod'],  _v_sub['cis_prop_mean'])
_rct, _pct = sp_stats.pearsonr(_ct_sub['peak_lod'], _ct_sub['cis_prop_mean']) if len(_ct_sub) > 1 else (float('nan'), float('nan'))
_rd,  _pd  = sp_stats.pearsonr(_d_sub['peak_lod'],  _d_sub['cis_prop_mean'])
_box_kw = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, linewidth=0)
g.ax_joint.text(0.03, 0.97,
    f'matching:    r = {_rv:.3f}  (p = {_pv:.1e})\n'
    f'cistrans:    r = {_rct:.3f}  (p = {_pct:.1e})\n'
    f'discordant:  r = {_rd:.3f}  (p = {_pd:.1e})',
    transform=g.ax_joint.transAxes, va='top', ha='left',
    fontsize=9, bbox=_box_kw)

# KDE marginals — one curve per partition
sns.kdeplot(data=_v_sub,  x='peak_lod',      ax=g.ax_marg_x,
            fill=True, color='#2166ac', alpha=0.35, linewidth=0.8)
sns.kdeplot(data=_ct_sub, x='peak_lod',      ax=g.ax_marg_x,
            fill=True, color='#4dac26', alpha=0.35, linewidth=0.8)
sns.kdeplot(data=_d_sub,  x='peak_lod',      ax=g.ax_marg_x,
            fill=True, color='#d6600d', alpha=0.35, linewidth=0.8)
sns.kdeplot(data=_v_sub,  y='cis_prop_mean', ax=g.ax_marg_y,
            fill=True, color='#2166ac', alpha=0.35, linewidth=0.8)
sns.kdeplot(data=_ct_sub, y='cis_prop_mean', ax=g.ax_marg_y,
            fill=True, color='#4dac26', alpha=0.35, linewidth=0.8)
sns.kdeplot(data=_d_sub,  y='cis_prop_mean', ax=g.ax_marg_y,
            fill=True, color='#d6600d', alpha=0.35, linewidth=0.8)

# cis_prop_mean is bounded [-1, 1] by construction; fix the y-axis explicitly
# so the JointGrid doesn't auto-range to KDE tails outside the data range
g.ax_joint.set_ylim(-1.05, 1.05)
g.ax_joint.set_yticks([-1, -0.5, 0, 0.5, 1])
g.ax_joint.set_xlabel('mosaiQTL peak LOD', fontsize=11)
g.ax_joint.set_ylabel('F1 cis_prop_mean  (cis fraction of ASE variance, −1 to 1)', fontsize=10)
g.ax_joint.legend(loc='lower right', frameon=True, fontsize=8,
                  framealpha=0.85, edgecolor='none')
sns.despine(ax=g.ax_joint)
sns.despine(ax=g.ax_marg_x, left=True)
sns.despine(ax=g.ax_marg_y, bottom=True)
g.figure.suptitle('concordance: mosaiQTL LOD  ×  F1 ASE cis proportion', y=1.01)
g.figure.savefig(f'{output_path}/cistrans_concordance.png', dpi=300,
                 bbox_inches='tight', transparent=True)
plt.close('all')
print('saved cistrans_concordance.png')

# ── validation: LOD by category (C and Lw1) ───────────────────────────────────
cat_plot_data = pd.concat([
    matching  .assign(category='matching'),
    cistrans  .assign(category='cistrans'),
    novel     .assign(category='novel'),
    discordant.assign(category='discordant'),
], ignore_index=True)

order   = ['matching', 'cistrans', 'novel', 'discordant']
# Distinct hues — blue for genome-wide (C), orange for local-window (Lw1)
tag_pal = {'C': '#2166ac', 'Lw1': '#e08214'}

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
sns.boxplot(
    data=cat_plot_data, x='category', y='peak_lod',
    hue='scan_tag', hue_order=['C', 'Lw1'],
    order=order, palette=tag_pal,
    width=0.55, fliersize=2, flierprops=dict(alpha=0.2),
    linewidth=0.8, ax=ax,
)
# Per-tag n= within each category label so the shared y-axis stays but counts
# are unambiguous: "validated\nC n=2,672 · Lw1 n=14,590"
ax.set_xticks(range(len(order)))
ax.set_xticklabels([
    '{}\nC n={:,}  ·  Lw1 n={:,}'.format(
        cat,
        int(((cat_plot_data['category'] == cat) & (cat_plot_data['scan_tag'] == 'C')).sum()),
        int(((cat_plot_data['category'] == cat) & (cat_plot_data['scan_tag'] == 'Lw1')).sum()),
    )
    for cat in order
], fontsize=8)
ax.set_xlabel('')
ax.set_ylabel('peak LOD')
ax.set_title('LOD distribution by overlap category  [C=chromosome, Lw1=local-window]')
legend = ax.get_legend()
legend.set_title('scan')
legend.set_frame_on(False)
for t in legend.get_texts():
    t.set_fontsize(8)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(f'{output_path}/cistrans_lod_by_category.png', dpi=300,
            bbox_inches='tight', transparent=True)
plt.close()
print('saved cistrans_lod_by_category.png')

# ── bin composition: sub-label breakdown within each category ─────────────────
# Each bin (matching / cistrans / novel / discordant) is subdivided by its
# constituent F1 reg_assignment values (or 'novel' for genes absent from F1).
# Sub-label colours are distinct within each bin to allow easy reading.
_SUBLABEL_COLORS = {
    'cis':       '#2166ac',   # matching — only one sub-label
    'cis+trans': '#74c476',   # cistrans bright green
    'cisxtrans': '#006d2c',   # cistrans dark green
    'novel':     '#e08214',   # not in F1 list at all
    'trans':     '#d01c8b',   # discordant pink
    'conserved': '#8c096b',   # discordant dark magenta
}

_BIN_ORDER    = ['matching', 'cistrans', 'novel', 'discordant']
_BIN_SUBLABELS = {
    'matching':   ['cis'],
    'cistrans':   ['cis+trans', 'cisxtrans'],
    'novel':      ['novel'],
    'discordant': ['trans', 'conserved'],
}

fig, axes = plt.subplots(len(SCAN_TAGS), 1,
                         figsize=(8, 2.8 * len(SCAN_TAGS)), dpi=300,
                         sharex=False)
if len(SCAN_TAGS) == 1:
    axes = [axes]

for ax, tag in zip(axes, SCAN_TAGS):
    sub = cat_plot_data[cat_plot_data['scan_tag'] == tag].copy()
    sub['sublabel'] = sub.apply(
        lambda r: r['reg_assignment'] if pd.notna(r['reg_assignment']) else 'novel',
        axis=1,
    )

    lefts  = np.zeros(len(_BIN_ORDER))
    totals = np.zeros(len(_BIN_ORDER))
    bin_idx = {b: i for i, b in enumerate(_BIN_ORDER)}

    for bin_name in _BIN_ORDER:
        for sl in _BIN_SUBLABELS[bin_name]:
            mask  = (sub['category'] == bin_name) & (sub['sublabel'] == sl)
            count = int(mask.sum())
            if count == 0:
                continue
            bi = bin_idx[bin_name]
            ax.barh(
                bi, count, left=lefts[bi], height=0.5,
                color=_SUBLABEL_COLORS.get(sl, '#aaaaaa'),
                edgecolor='white', linewidth=0.6,
                label=sl if tag == SCAN_TAGS[0] else '_nolegend_',
            )
            # sub-label count inside segment when wide enough
            seg_cx = lefts[bi] + count / 2
            if count >= 60:
                ax.text(seg_cx, bi, f'{count:,}',
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            lefts[bi]  += count
            totals[bi] += count

    # total to the right of each bar
    x_max = totals.max()
    for bi, bin_name in enumerate(_BIN_ORDER):
        ax.text(totals[bi] + x_max * 0.01, bi, f'{int(totals[bi]):,}',
                ha='left', va='center', fontsize=9, color='#333333')

    ax.set_yticks(range(len(_BIN_ORDER)))
    ax.set_yticklabels(_BIN_ORDER, fontsize=11)
    ax.set_xlabel('number of eGenes', fontsize=11)
    ax.set_title(f'[{tag}]  eGene bin composition', fontsize=12)
    ax.set_xlim(0, x_max * 1.18)   # room for right-side labels
    ax.set_ylim(-0.6, len(_BIN_ORDER) - 0.4)
    sns.despine(ax=ax)

# shared legend
_sl_order = ['cis', 'cis+trans', 'cisxtrans', 'novel', 'trans', 'conserved']
handles = [plt.Rectangle((0, 0), 1, 1, color=_SUBLABEL_COLORS[sl]) for sl in _sl_order]
fig.legend(handles, _sl_order, title='F1 reg_assignment',
           loc='center left', bbox_to_anchor=(1.0, 0.5),
           frameon=False, fontsize=9, title_fontsize=9, ncol=1)
fig.suptitle('eGene bin composition by F1 regulatory assignment', fontsize=13)
plt.tight_layout()
fig.subplots_adjust(right=0.82)
plt.savefig(f'{output_path}/cistrans_bin_composition.png', dpi=300,
            bbox_inches='tight', transparent=True)
plt.close()
print('saved cistrans_bin_composition.png')

# breakdown of what's actually inside each in-F1 partition:
#   cistrans  — cis+trans or cisxtrans: mosaiQTL partially right, trans component also present
#   discordant — trans or conserved: reg_assignment breakdown below
#     trans     — F1 shows differential expression but regulated from a trans locus
#     conserved — F1 shows NO allelic imbalance (uniformly expressed); not a cis-eQTL candidate
print('cistrans reg_assignment counts:')
print(cistrans['reg_assignment'].value_counts().to_string())
print('discordant reg_assignment counts:')
print(discordant['reg_assignment'].value_counts().to_string())

# ── discordant: trans-eQTL candidates (C scan, F1 trans, |dist| > 1 Mb) ───────
# discordant now = trans + conserved only; filter to trans for the candidate list
trans_cands = discordant[
    (discordant['scan_tag'] == 'C') &
    (discordant['reg_assignment'] == 'trans') &
    (discordant['cis_dist_mb'].abs() > 1)
].copy()

print(f'{len(trans_cands):,} trans-eQTL candidates  '
      f'({trans_cands["celltype"].nunique()} cell types)')

# Two panels:
#   left  — per-cell-type bar chart of trans candidate counts (sorted descending)
#   right — LOD distribution per cell type as a violin (same sort order, sharey=False)
_tc_ct = (
    trans_cands.groupby('celltype')
               .size()
               .reset_index(name='n')
               .sort_values('n', ascending=False)
)
_ct_order = _tc_ct['celltype'].tolist()   # high → low, used on both panels

fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, len(_ct_order) * 0.5)),
                         dpi=300, constrained_layout=True)

# ── left: count bars ──────────────────────────────────────────────────────────
ax = axes[0]
sns.barplot(data=_tc_ct, y='celltype', x='n',
            order=_ct_order, color=CAT_COLORS['discordant'],
            alpha=0.85, errorbar=None, orient='h', ax=ax)
for bar, (_, row) in zip(ax.patches, _tc_ct.set_index('celltype').loc[_ct_order].iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(int(row['n'])), va='center', fontsize=8, color='#444')
ax.set_yticklabels([c.replace('_', ' ') for c in _ct_order], fontsize=9)
ax.set_xlabel('trans-eQTL candidates', fontsize=9)
ax.set_ylabel('')
ax.set_title('count per cell type', fontsize=9)
sns.despine(ax=ax)

# ── right: LOD violin per cell type ───────────────────────────────────────────
ax = axes[1]
sns.violinplot(data=trans_cands, y='celltype', x='peak_lod',
               order=_ct_order, color=CAT_COLORS['discordant'],
               inner='box', linewidth=0.7, cut=0,
               density_norm='width', orient='h', ax=ax)
ax.set_yticklabels([c.replace('_', ' ') for c in _ct_order], fontsize=9)
ax.set_xlabel('peak LOD', fontsize=9)
ax.set_ylabel('')
ax.set_title('LOD distribution', fontsize=9)
sns.despine(ax=ax)

fig.suptitle('trans-eQTL candidates  [C scan, discordant + F1 trans, |dist| > 1 Mb]',
             fontsize=10)
plt.savefig(f'{output_path}/cistrans_trans_candidates.png', dpi=300,
            bbox_inches='tight', transparent=True)
plt.close()
print('saved cistrans_trans_candidates.png')

# ── sensitivity table ─────────────────────────────────────────────────────────
sens = []
for (ct_name, tag), grp in merged.groupby(['celltype', 'scan_tag']):
    f1c = set(ct_agg.loc[(ct_agg['celltype_norm'] == ct_name) & ct_agg['is_f1_cis'], 'gene'])
    our = set(grp['gene_symbol'])
    sens.append(dict(
        celltype=ct_name, scan_tag=tag,
        n_f1_cis=len(f1c), n_egenes=len(our),
        n_missed=len(f1c - our),
        recall=round(len(f1c & our) / len(f1c), 3) if f1c else None,
    ))
sens_df = pd.DataFrame(sens).sort_values(['scan_tag', 'recall'])
print(sens_df.to_string(index=False))

# ── save CSVs ─────────────────────────────────────────────────────────────────
merged.to_csv(    f'{output_path}/cistrans_overlap_merged.csv',    index=False)
cistrans.to_csv(  f'{output_path}/cistrans_cistrans_egenes.csv',   index=False)
discordant.to_csv(f'{output_path}/cistrans_discordant_egenes.csv', index=False)

# Per-celltype per-scan-tag CSVs in celltype subdirectories.
# Only write for CTs that have F1 ASE equivalents (appear in ct_agg).
#   cistrans_comp/<celltype>/cistrans_validated_{ct}_{tag}.csv
#   cistrans_comp/<celltype>/cistrans_novel_{ct}_{tag}.csv
_out = pathlib.Path(output_path)
_f1_cts = set(ct_agg['celltype_norm'].unique())
_n_val = _n_nov = 0

for (ct_name, tag), grp in matching.groupby(['celltype', 'scan_tag']):
    if ct_name not in _f1_cts:
        continue
    subdir = _out / ct_name
    subdir.mkdir(parents=True, exist_ok=True)
    grp.to_csv(subdir / f'cistrans_matching_{ct_name}_{tag}.csv', index=False)
    _n_val += 1

for (ct_name, tag), grp in novel.groupby(['celltype', 'scan_tag']):
    if ct_name not in _f1_cts:
        continue
    subdir = _out / ct_name
    subdir.mkdir(parents=True, exist_ok=True)
    grp.to_csv(subdir / f'cistrans_novel_{ct_name}_{tag}.csv', index=False)
    _n_nov += 1

print(f'saved {_n_val} matching + {_n_nov} novel per-CT per-tag files '
      f'in celltype subdirs (F1-matched CTs only)')
print('done')
