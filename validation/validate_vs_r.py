# validate_vs_r.py
# compare mosaiqtl scan output against r miqtl lod scores
#
# reads pre-existing files produced by run_validation.R:
#   validation/r_data/lod_r.csv         — r miqtl lod scores (all genes × loci)
#   validation/r_data/pheno_48_used.csv — exact pheno_48 r used
#   validation/r_data/allele_effects.csv — founder effects at peak per gene (optional)
#   [miqtl]/output/genomecache/         — happy genome cache
#   [miqtl]/input/pmap.csv              — physical map
#
# workflow:
#   1. groups genes by chromosome (as in lod_r.csv)
#   2. per chromosome: compute loco kinship, eigendecompose, scan all genes on that chr
#   3. merges python lods with r lods per locus, reports pearson r and max|Δlod|
#   4. generates qc plots: lod scatter, lod residuals, founder effects comparison
#
# pass threshold: pearson r > 0.999 and max|Δlod| < 0.01 per gene
#
# usage:
#   python validation/validate_vs_r.py [--n_genes N]
#   --n_genes: cap genes per chromosome for quick testing (default: all)

import numpy as np
import pandas as pd
import os, sys, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module
from mosaiqtl.utils import rint

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger('validate')

# ── paths ─────────────────────────────────────────────────────────────────────
# miqtl_dir: where real pipeline outputs live (pheno_48, kinship, genomecache)
# pkg_dir:   mosaiqtl repo root
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

cache_dir = os.path.join(miqtl_dir, 'output', 'genomecache')
pmap_path = os.path.join(miqtl_dir, 'input',  'pmap.csv')
val_dir   = os.path.join(pkg_dir,   'validation', 'r_data')
plot_dir  = os.path.join(pkg_dir,   'validation', 'qc_plots')

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_genes', default=None, type=int,
                    help='cap genes per chromosome (default: all)')
parser.add_argument('--n_chrs',  default=None, type=int,
                    help='cap number of chromosomes (default: all)')
args = parser.parse_args()

# ── load r reference outputs ──────────────────────────────────────────────────
lod_r_path = os.path.join(val_dir, 'lod_r.csv')
if not os.path.exists(lod_r_path):
    log.error(f'lod_r.csv not found at {lod_r_path}')
    log.error('run first: Rscript validation/r_reference/run_validation.R')
    sys.exit(1)

pheno_path = os.path.join(val_dir, 'pheno_48_used.csv')
if not os.path.exists(pheno_path):
    log.error(f'pheno_48_used.csv not found at {pheno_path}')
    sys.exit(1)

lod_r = pd.read_csv(lod_r_path)
lod_r['chr'] = lod_r['chr'].astype(str)
chromosomes  = sorted(lod_r['chr'].unique(),
                      key=lambda c: int(c) if c.isdigit() else 99)
if args.n_chrs:
    chromosomes = chromosomes[:args.n_chrs]
n_genes_total = lod_r['gene_id'].nunique()
log.info(f'r reference: {lod_r.shape[0]} rows | '
         f'{n_genes_total} genes | chrs: {chromosomes}')

# pheno_48 — r writes rownames as unnamed first col; subject.name may already exist
pheno_48 = pd.read_csv(pheno_path, index_col=0)
if 'subject.name' not in pheno_48.columns:
    pheno_48.index.name = 'subject.name'
    pheno_48 = pheno_48.reset_index()
else:
    pheno_48 = pheno_48.reset_index(drop=True)
log.info(f'pheno_48: {pheno_48.shape[0]} strains × {pheno_48.shape[1]} cols')

# optional allele effects from r
ae_path   = os.path.join(val_dir, 'allele_effects.csv')
ae_r      = pd.read_csv(ae_path) if os.path.exists(ae_path) else None

# ── load genome cache ─────────────────────────────────────────────────────────
if not os.path.isdir(cache_dir):
    log.error(f'genome cache not found: {cache_dir}')
    sys.exit(1)

log.info(f'loading genome cache from {cache_dir}')
cache = GenomeCache.from_happy_dir(cache_dir, model='additive')
log.info(f'cache: {len(cache.chromosomes)} chr | '
         f'{len(cache.sample_ids)} strains | {cache.n_founders} founders')

# ── load pmap ─────────────────────────────────────────────────────────────────
pmap = pd.read_csv(pmap_path)
if 'marker' in pmap.columns:
    pmap = pmap.rename(columns={'marker': 'locus', 'pos': 'pos_mb'})
pmap['locus'] = pmap['locus'].str.replace('.RData', '', regex=False)
pmap['chr']   = pmap['chr'].astype(str)
log.info(f'pmap: {len(pmap)} loci')

# ── align pheno_48 to cache strain order ──────────────────────────────────────
cache_strains = list(cache.sample_ids)
pheno_48 = pheno_48.set_index('subject.name').reindex(cache_strains)

# covariate columns — batch dummy-encoded + pc1..pc10 (same as scan_eqtl.R)
pc_cols       = [f'pc{i}' for i in range(1, 11)]
batch_dummies = pd.get_dummies(pheno_48['batch'], prefix='batch', drop_first=True)
covar_base    = pd.concat([
    pd.Series(np.ones(len(pheno_48)), index=pheno_48.index, name='intercept'),
    batch_dummies.astype(float),
    pheno_48[pc_cols].astype(float),
], axis=1).values   # (48, p)

pheno_48 = pheno_48.reset_index()   # back to dataframe with subject.name column
log.info(f'covariate matrix: {covar_base.shape}')

# ── scan loop — chromosome by chromosome ─────────────────────────────────────
all_pass    = True
comparisons = []   # list of merged dfs for saving
gene_stats  = []   # list of per-gene summary rows for qc plot

for chrom in chromosomes:
    if chrom not in cache.chromosomes:
        log.warning(f'chr{chrom} not in genome cache — skipping')
        continue

    chr_genes = lod_r[lod_r['chr'] == chrom]['gene_id'].unique().tolist()
    if args.n_genes:
        chr_genes = chr_genes[:args.n_genes]

    log.info(f'\n[chr{chrom}] {len(chr_genes)} genes')

    # loco kinship eigendecomposition for this chromosome (once per chr)
    kinship_dict  = kin_module.calc_kinship(cache, method='loco')
    k_chr         = kinship_dict[chrom]
    eigvals, eigvecs = np.linalg.eigh(k_chr)

    # rotate covariates once per chromosome
    covar_rot = eigvecs.T @ covar_base   # (48, p)

    for gene in chr_genes:
        if gene not in pheno_48.columns:
            log.warning(f'  {gene}: not in pheno_48 — skipping')
            continue

        y = pheno_48.set_index('subject.name').reindex(cache_strains)[gene].values.astype(float)
        if np.all(np.isnan(y)):
            log.warning(f'  {gene}: all NaN — skipping')
            continue

        y_rot = eigvecs.T @ y

        # estimate null h² once per gene (same as scan_h2lmm)
        h2_null, _ = lmm_module.estimate_h2_null(y_rot, covar_rot, eigvals)

        # get the exact loci r scanned for this gene on this chromosome
        r_gene = (lod_r[(lod_r['gene_id'] == gene) & (lod_r['chr'] == chrom)]
                  [['locus', 'lod', 'pos_mb']]
                  .rename(columns={'lod': 'lod_r'}))
        if r_gene.empty:
            continue

        lod_py = []
        for locus in r_gene['locus']:
            try:
                # get locus index in cache
                locus_ids = cache._data[chrom]['locus_ids']
                l_idx = np.where(locus_ids == locus)[0]
                if len(l_idx) == 0:
                    lod_py.append(np.nan)
                    continue
                h_l = cache.get_locus(chrom, int(l_idx[0]))   # (48, J)
                h_rot = eigvecs.T @ h_l[:, :-1]               # drop last founder
                result = lmm_module.fit_lmm_at_locus(
                    y_rot, h_rot, covar_rot, eigvals, h2_null)
                lod_py.append(result.lod)
            except Exception as e:
                log.debug(f'    locus {locus}: {e}')
                lod_py.append(np.nan)

        r_gene = r_gene.copy()
        r_gene['lod_py'] = lod_py
        merged = r_gene.dropna(subset=['lod_r', 'lod_py'])

        if len(merged) < 3:
            log.warning(f'  {gene}: only {len(merged)} valid loci — skipping')
            all_pass = False
            continue

        r_corr = float(np.corrcoef(merged['lod_r'], merged['lod_py'])[0, 1])
        max_d  = float((merged['lod_r'] - merged['lod_py']).abs().max())
        passed = r_corr > 0.999 and max_d < 0.01
        status = 'PASS' if passed else 'FAIL'

        log.info(f'  {status}  {gene:25s}  r={r_corr:.6f}  '
                 f'max|Δlod|={max_d:.6f}  n={len(merged)}')

        if not passed:
            all_pass = False

        merged['gene_id'] = gene
        merged['chr']     = chrom
        comparisons.append(merged)
        gene_stats.append({'gene_id': gene, 'chr': chrom,
                           'r': r_corr, 'max_delta_lod': max_d,
                           'n_loci': len(merged), 'passed': passed})

# ── save comparison table ─────────────────────────────────────────────────────
os.makedirs(val_dir, exist_ok=True)

if comparisons:
    comp_df  = pd.concat(comparisons, ignore_index=True)
    stat_df  = pd.DataFrame(gene_stats)
    comp_df.to_csv(os.path.join(val_dir, 'lod_comparison.csv'), index=False)
    stat_df.to_csv(os.path.join(val_dir, 'gene_summary.csv'),   index=False)
    log.info(f'\nsaved: lod_comparison.csv ({len(comp_df)} rows) | '
             f'gene_summary.csv ({len(stat_df)} genes)')
else:
    log.error('no comparison data — all genes skipped')
    sys.exit(1)

# ── qc plots ──────────────────────────────────────────────────────────────────
os.makedirs(plot_dir, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # font and style
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Carlito', 'DejaVu Sans'],
        'font.size': 11,
    })

    founder_colors = {
        'AJ':     '#F4C245', 'B6J':    '#C0BFBF', '129S1J': '#DA9CC1',
        'NODJ':   '#4F6EAF', 'NZOJ':   '#52A5DB', 'CASTJ':  '#55AF5B',
        'PWKJ':   '#D83026', 'WSBJ':   '#683C91',
    }

    # ── plot 1: lod correlation (python vs r) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(comp_df['lod_r'], comp_df['lod_py'],
               s=8, alpha=0.4, color='#4F6EAF', edgecolors='none',
               rasterized=True)
    lim = max(comp_df['lod_r'].max(), comp_df['lod_py'].max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#D83026', lw=1, label='$y = x$')
    r_all   = float(np.corrcoef(comp_df['lod_r'], comp_df['lod_py'])[0, 1])
    max_all = float((comp_df['lod_r'] - comp_df['lod_py']).abs().max())
    ax.text(0.05, 0.95,
            f'$r$ = {r_all:.6f}\nmax $|\\Delta|$ = {max_all:.4f}\n'
            f'$n$ = {len(comp_df)} loci',
            transform=ax.transAxes, fontsize=10, va='top')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('miQTL LOD (R)'); ax.set_ylabel('mosaiQTL LOD (Python)')
    ax.set_title('LOD cross-language comparison')
    ax.legend(fontsize=9); sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lod_correlation.png'), dpi=150)
    plt.close()
    log.info('saved: lod_correlation.png')

    # ── plot 2: lod residuals (python − r) by chromosome ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    palette = ['#4F6EAF', '#C0BFBF']
    chr_list = comp_df['chr'].unique()
    chr_order = sorted(chr_list, key=lambda c: int(c) if c.isdigit() else 99)
    colors = [palette[chr_order.index(c) % 2] for c in comp_df['chr']]
    diff   = comp_df['lod_py'] - comp_df['lod_r']
    ax.scatter(range(len(diff)), diff, c=colors, s=6, alpha=0.5,
               edgecolors='none', rasterized=True)
    ax.axhline(0, color='#D83026', lw=0.8, linestyle='--')
    ax.text(0.98, 0.95,
            f'mean $\\Delta$ = {diff.mean():.4f}\n'
            f'sd $\\Delta$ = {diff.std():.4f}\n'
            f'max $|\\Delta|$ = {diff.abs().max():.4f}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right')
    # chr boundary lines
    offset = 0
    for c in chr_order:
        n = (comp_df['chr'] == c).sum()
        if offset > 0:
            ax.axvline(offset, color='white', lw=0.5, alpha=0.5)
        ax.text(offset + n / 2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.02,
                f'{c}', ha='center', fontsize=7, color='#444')
        offset += n
    ax.set_xlim(0, len(diff))
    ax.set_xlabel('locus index'); ax.set_ylabel('$\\Delta$ LOD (Py $-$ R)')
    ax.set_title('LOD residuals by chromosome')
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lod_residuals.png'), dpi=150)
    plt.close()
    log.info('saved: lod_residuals.png')

    # ── plot 3: per-gene summary (r vs pass/fail) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # r distribution
    ax = axes[0]
    sns.histplot(stat_df['r'], bins=30, color='#4F6EAF', ax=ax)
    ax.axvline(0.999, color='#D83026', lw=1, linestyle='--', label='threshold')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('Pearson $r$ (mosaiQTL vs miQTL LOD)')
    ax.set_ylabel('genes'); ax.set_title('LOD correlation per gene')
    ax.legend(fontsize=9); sns.despine(ax=ax)

    # max|Δlod| distribution
    ax = axes[1]
    sns.histplot(stat_df['max_delta_lod'], bins=30, color='#683C91', ax=ax)
    ax.axvline(0.01, color='#D83026', lw=1, linestyle='--', label='threshold')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('max $|\\Delta$ LOD$|$')
    ax.set_ylabel('genes'); ax.set_title('Max LOD difference per gene')
    ax.legend(fontsize=9); sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'gene_summary.png'), dpi=150)
    plt.close()
    log.info('saved: gene_summary.png')

    # ── plot 4: founder effects comparison at peak locus (if ae_r available) ──
    if ae_r is not None and not stat_df.empty:
        # pick the gene with highest r-side LOD as the showcase
        best_gene = (lod_r.groupby('gene_id')['lod'].max()
                     .idxmax())
        best_chr  = lod_r[lod_r['gene_id'] == best_gene]['chr'].iloc[0]

        # python allele effects at the peak locus for this gene
        best_comp = comp_df[comp_df['gene_id'] == best_gene]
        if not best_comp.empty:
            peak_locus = best_comp.loc[best_comp['lod_r'].idxmax(), 'locus']

            r_ae = ae_r[(ae_r['gene_id'] == best_gene) &
                        (ae_r['locus'] == peak_locus)]

            if not r_ae.empty:
                founders  = r_ae['founder'].tolist()
                r_effects = r_ae['effect'].values
                colors_ae = [founder_colors.get(f, '#888888') for f in founders]

                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(len(founders))
                ax.bar(x, r_effects, color=colors_ae,
                       edgecolor='white', lw=0.5, label='miQTL (R)')
                ax.axhline(0, color='#444', lw=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(founders, rotation=45, ha='right', fontsize=10)
                ax.set_ylabel('founder effect ($\\hat{\\beta}_f$)')
                ax.set_title(f'founder effects at peak — {best_gene} '
                             f'chr{best_chr} ({peak_locus})')
                ax.legend(fontsize=9); sns.despine()
                plt.tight_layout()
                fig.savefig(os.path.join(plot_dir, 'founder_effects.png'), dpi=150)
                plt.close()
                log.info('saved: founder_effects.png')

    log.info(f'\nall qc plots saved to {plot_dir}/')

except Exception as e:
    log.warning(f'qc plot generation failed: {e}')

# ── final verdict ─────────────────────────────────────────────────────────────
n_pass = sum(g['passed'] for g in gene_stats)
n_fail = len(gene_stats) - n_pass

print()
print('=' * 60)
if all_pass:
    print(f'  ALL {n_pass} GENES PASS  (r > 0.999, max|Δlod| < 0.01)')
else:
    print(f'  {n_pass} PASS / {n_fail} FAIL — see log above')
print('=' * 60)

sys.exit(0 if all_pass else 1)
