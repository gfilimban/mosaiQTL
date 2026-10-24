"""split_egene_effects_by_founder.py — per-founder eGene effect files.

Reads egene_effects_all.csv and writes one concise CSV per founder × scan_tag
containing only eGenes where that founder has the highest (role=top) or
lowest (role=bottom) beta across all 8 founders.

Output columns (11 total):
    gene_symbol   — gene name
    celltype      — cell type
    chr           — chromosome (natsorted)
    peak_mb       — peak locus position
    cis_dist_mb   — distance from TSS to peak
    peak_lod      — LOD score at peak
    h2            — heritability estimate
    role          — 'top' (max beta) or 'bottom' (min beta)
    focal_beta    — this founder's beta
    contrast_founder — founder at the opposite extreme
    contrast_beta    — that founder's beta

Output:
    validation/ctxhipp_data/egene_effects/by_founder/
        egene_effects_PWKJ_Lw1.csv  …

Usage:
    conda run -n bioinfo python validation/tools/split_egene_effects_by_founder.py
"""
import pathlib
import numpy as np
import pandas as pd
from natsort import natsorted

IN_CSV  = pathlib.Path('validation/ctxhipp_data/egene_effects/egene_effects_all.csv')
OUT_DIR = pathlib.Path('validation/ctxhipp_data/egene_effects/by_founder')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOUNDERS  = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
BETA_COLS = [f'beta_{f}' for f in FOUNDERS]
OUT_COLS  = [
    'gene_id', 'gene_symbol', 'celltype', 'chr', 'peak_mb', 'cis_dist_mb',
    'peak_lod', 'h2', 'role', 'focal_beta', 'contrast_founder', 'contrast_beta',
]

df = pd.read_csv(IN_CSV, dtype={'chr': str})
betas = df[BETA_COLS].values.astype(float)

top_idx    = np.argmax(betas, axis=1)
bottom_idx = np.argmin(betas, axis=1)

df['_top_founder']    = [FOUNDERS[i] for i in top_idx]
df['_top_beta']       = betas[np.arange(len(df)), top_idx]
df['_bottom_founder'] = [FOUNDERS[i] for i in bottom_idx]
df['_bottom_beta']    = betas[np.arange(len(df)), bottom_idx]

for tag in natsorted(df['scan_tag'].unique()):
    tag_df = df[df['scan_tag'] == tag]

    for founder in FOUNDERS:
        rows = []

        # eGenes where this founder drives expression highest
        top_mask = tag_df['_top_founder'] == founder
        if top_mask.any():
            sub = tag_df[top_mask].copy()
            sub['role']             = 'top'
            sub['focal_beta']       = sub['_top_beta']
            sub['contrast_founder'] = sub['_bottom_founder']
            sub['contrast_beta']    = sub['_bottom_beta']
            rows.append(sub)

        # eGenes where this founder drives expression lowest
        bot_mask = tag_df['_bottom_founder'] == founder
        if bot_mask.any():
            sub = tag_df[bot_mask].copy()
            sub['role']             = 'bottom'
            sub['focal_beta']       = sub['_bottom_beta']
            sub['contrast_founder'] = sub['_top_founder']
            sub['contrast_beta']    = sub['_top_beta']
            rows.append(sub)

        if not rows:
            continue

        out = pd.concat(rows, ignore_index=True)

        # natural chr sort, then peak_mb within chr
        out['_chr_cat'] = pd.Categorical(
            out['chr'],
            categories=natsorted(out['chr'].unique()),
            ordered=True,
        )
        out = (out
               .sort_values(['role', '_chr_cat', 'peak_mb'])
               .drop(columns='_chr_cat')
               [OUT_COLS])

        out_f = OUT_DIR / f'egene_effects_{founder}_{tag}.csv'
        out.to_csv(out_f, index=False)
        print(f'  {founder} / {tag}: {len(out):,} eGenes '
              f'({top_mask.sum()} top, {bot_mask.sum()} bottom) → {out_f.name}')

print('done')
