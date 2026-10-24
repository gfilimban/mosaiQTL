"""Regenerate ctxhipp_celltype_multipanel.png — notebook code, bigger fonts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scanpy as sc
from pathlib import Path

for _font in ('Carlito', 'Calibri', 'DejaVu Sans'):
    if _font in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        mpl.rcParams['font.family'] = _font
        break

DATA = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis'
INPUT_H5AD = f'{DATA}/input/adata/next/ctxhipp_all_annotated.h5ad'
OUT = Path(f'{DATA}/qtl/output/ctxhipp/all')
OUT.mkdir(parents=True, exist_ok=True)

FOUNDER_ORDER = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
F1_ORDER = ['B6AF1J', 'B6129S1F1J', 'B6NODF1J', 'B6NZOF1J', 'B6CASTF1J', 'B6PWKF1J', 'B6WSBF1J']
FOUNDER_COLORS = {'AJ':'#F4C245','B6J':'#C0BFBF','129S1J':'#DA9CC1','NODJ':'#4F6EAF','NZOJ':'#52A5DB','CASTJ':'#55AF5B','PWKJ':'#D83026','WSBJ':'#683C91'}
F1_COLORS = {'B6AF1J':'#DAC082','B6129S1F1J':'#CDADC0','B6CASTF1J':'#8AB78D','B6NODF1J':'#8796B7','B6NZOF1J':'#89B2CD','B6PWKF1J':'#CC7772','B6WSBF1J':'#947DA8'}
CC_COLORS = {'CC001':'#E86B73','CC002':'#208954','CC003':'#9C88DE','CC004':'#AB5F00','CC005':'#00ADC0','CC006':'#A75384','CC007':'#87A723','CC008':'#4873BD','CC009':'#D97B5E','CC010':'#009070','CC011':'#B380CF','CC012':'#907000','CC013':'#00A8E3','CC015':'#AF516A','CC017':'#67AC64','CC018':'#6665CC','CC024':'#D18342','CC025':'#008A8C','CC028':'#CF70C1','CC029':'#797A00','CC030':'#549EE1','CC032':'#BE4742','CC036':'#3BB082','CC037':'#8161B2','CC038':'#CB8A00','CC041':'#0085A3','CC043':'#D3769F','CC057':'#538500','CC060':'#7794E6','CC062':'#AF592E','CC065':'#00B6A2','CC071':'#955AA1','CC074':'#AD982D'}
SEX_COLORS = {'F':'hotpink','M':'dodgerblue','Female':'hotpink','Male':'dodgerblue'}
CC_FALLBACK = '#888888'


def _blend_hex(a, b, t=0.5):
    def to_rgb(h):
        h = h.lstrip('#')
        return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    ra, ga, ba = to_rgb(a)
    rb, gb, bb = to_rgb(b)
    return '#{:02X}{:02X}{:02X}'.format(
        round((ra + t*(rb-ra))*255),
        round((ga + t*(gb-ga))*255),
        round((ba + t*(bb-ba))*255),
    )


def _cc_color(strain, cc_ancestry=None):
    all_canonical = {**FOUNDER_COLORS, **F1_COLORS}
    if strain in all_canonical:
        return all_canonical[strain]
    if strain in CC_COLORS:
        return CC_COLORS[strain]
    digits = ''.join(filter(str.isdigit, strain))
    num = int(digits) if digits else hash(strain) % 1000
    i1 = num % 8
    i2 = (num * 3 + 2) % 8
    if i1 == i2:
        i2 = (i2 + 1) % 8
    return _blend_hex(FOUNDER_COLORS[FOUNDER_ORDER[i1]],
                      FOUNDER_COLORS[FOUNDER_ORDER[i2]],
                      (num % 97) / 97.0)


def _order_columns(df):
    canonical = FOUNDER_ORDER + F1_ORDER
    present = [c for c in canonical if c in df.columns]
    cc_cols = sorted(c for c in df.columns if c not in set(canonical) and str(c).startswith('CC'))
    other = [c for c in df.columns if c not in set(canonical) and not str(c).startswith('CC')]
    return df[present + cc_cols + other]


def _auto_colors(columns, var_key, cc_ancestry=None):
    all_canonical = {**FOUNDER_COLORS, **F1_COLORS, **CC_COLORS}
    if var_key.lower() in ('sex', 'gender'):
        return [SEX_COLORS.get(c, CC_FALLBACK) for c in columns]
    if any(str(c).startswith('CC') for c in columns):
        return [all_canonical.get(c, _cc_color(c, cc_ancestry)) for c in columns]
    if any(c in all_canonical for c in columns):
        return [all_canonical.get(c, CC_FALLBACK) for c in columns]
    return sns.color_palette('husl', len(columns))


_DATASET_MAP   = {'founder': 'Founder', 'f1': 'F1', 'cc1': 'CC1', 'cc2': 'CC2', 'cc3': 'CC3'}
_DATASET_ORDER = ['Founder', 'F1', 'CC1', 'CC2', 'CC3']


def stacked_barplot_proportions_ax(
    obs_df, cluster_key, var_key, ax,
    custom_order=None,
    show_ylabel=True,
    show_annotations=True,
    legend_position='bottom',
    vline=None,
    cc_ancestry=None,
):
    # Capitalize dataset values and impose canonical order
    if var_key == 'dataset':
        obs_df = obs_df.copy()
        obs_df[var_key] = obs_df[var_key].map(
            lambda x: _DATASET_MAP.get(str(x).lower(), x)
        )

    grouped = obs_df.groupby([cluster_key, var_key]).size().unstack(fill_value=0)
    counts  = obs_df.groupby(cluster_key).size()

    order = list(custom_order) if custom_order is not None \
        else counts.sort_values(ascending=False).index.tolist()

    grouped     = grouped.loc[order]
    counts      = counts.loc[order]
    proportions = grouped.div(grouped.sum(axis=1), axis=0)

    if var_key == 'dataset':
        ordered_cols = [c for c in _DATASET_ORDER if c in proportions.columns]
        ordered_cols += [c for c in proportions.columns if c not in _DATASET_ORDER]
        proportions = proportions[ordered_cols]
    else:
        proportions = _order_columns(proportions)

    proportions = proportions.iloc[::-1]
    counts      = counts.iloc[::-1]

    colors = _auto_colors(proportions.columns, var_key, cc_ancestry)

    lefts = np.zeros(len(proportions))
    for col, color in zip(proportions.columns, colors):
        ax.barh(
            range(len(proportions)),
            proportions[col].values,
            left=lefts,
            color=color,
            edgecolor='none',
            height=0.8,
            label=col,
        )
        lefts += proportions[col].values

    ax.set_yticks(range(len(proportions)))

    n_clusters = len(proportions)
    # font sizes — bumped from notebook originals
    ann_fs   = max(12, min(16, int(480 / n_clusters)))
    ytick_fs = max(12, min(16, int(480 / n_clusters)))

    if show_annotations:
        ax.set_xlim(0, 1.18)
        for i, txt in enumerate(counts):
            ax.text(1.02, i, str(txt), fontsize=ann_fs, va='center')
    else:
        ax.set_xlim(0, 1.0)

    if vline is not None:
        ax.axvline(vline, color='#444444', linestyle='--', linewidth=1.5)

    ax.tick_params(axis='x', labelsize=13)
    ax.set_xlabel('Proportion', fontsize=14)

    if show_ylabel:
        ax.set_ylabel(cluster_key, fontsize=14)
        ax.tick_params(axis='y', labelsize=ytick_fs, length=4)
        ax.set_yticklabels(proportions.index)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    sns.despine(ax=ax)

    ax.set_ylim(-0.5, n_clusters - 0.5)

    handles, labels = ax.get_legend_handles_labels()

    # ncol=6 for genotype/zt (many items); 2 for everything else
    ncol = 6 if var_key in ('genotype', 'zt') else 2

    # Legend below axes — anchor top-left of legend to bottom-left of axes.
    # bbox_to_anchor in axes coords: (0, -0.05) places it just below.
    # Subplot widths stay equal because constrained_layout is OFF in the caller
    # and fig.subplots_adjust(bottom=...) reserves fixed space for legends.
    # 4-tuple bbox_to_anchor: (x0, y0, width, height) in axes coords.
    # width=1.0 clamps the legend to exactly the axes width; mode='expand'
    # distributes items across that width so nothing overflows into neighbours.
    if legend_position == 'left':
        ax.legend(handles, labels, title=var_key,
                  bbox_to_anchor=(0, -0.05, 1.0, 0), loc='upper left', mode='expand',
                  fontsize=11, ncol=2, frameon=False, title_fontsize=12)
    else:
        ax.legend(handles, labels, title=var_key,
                  bbox_to_anchor=(0, -0.05, 1.0, 0), loc='upper left', mode='expand',
                  fontsize=11, ncol=ncol, frameon=False, title_fontsize=12)

    ax.grid(False)
    return ax


def stacked_barplot_multipanel(obs_df, cluster_key, panels, figsize=(11, 7), cc_ancestry=None):
    counts       = obs_df.groupby(cluster_key).size()
    shared_order = counts.sort_values(ascending=False).index.tolist()

    n          = len(panels)
    n_clusters = obs_df[cluster_key].nunique()
    h          = max(figsize[1], n_clusters * 0.35)
    # No constrained_layout — let each subplot get equal width.
    # Fixed bottom margin reserves space for the per-panel legends below axes.
    fig, axes  = plt.subplots(1, n, figsize=(figsize[0], h), dpi=300)
    fig.subplots_adjust(wspace=0.05, bottom=0.35)
    if n == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        stacked_barplot_proportions_ax(
            obs_df, cluster_key, panel['var_key'], ax,
            custom_order=shared_order,
            show_ylabel=panel.get('show_ylabel', False),
            show_annotations=panel.get('show_annotations', False),
            legend_position=panel.get('legend_position', 'bottom'),
            vline=panel.get('vline'),
            cc_ancestry=cc_ancestry,
        )

    return fig, axes


print('loading h5ad …')
adata = sc.read_h5ad(INPUT_H5AD, backed='r')

fig, axes = stacked_barplot_multipanel(
    adata.obs,
    cluster_key='celltype',
    panels=[
        dict(var_key='genotype', show_ylabel=True,  show_annotations=False),
        dict(var_key='dataset',  show_ylabel=False, show_annotations=False),
        dict(var_key='zt',       show_ylabel=False, show_annotations=False),
        dict(var_key='sex',      show_ylabel=False, show_annotations=True),
    ],
    figsize=(20, 8),
)

out = OUT / 'ctxhipp_celltype_multipanel.png'
fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
print(f'saved → {out}')
