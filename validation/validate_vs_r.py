# validate_vs_r.py
# cross-language validation: mosaiqtl (Python) vs miQTL (R)
#
# sections:
#   1. data integrity — compare pheno_48, pmap, and cache haplotype probs
#      between what Python loads and what R exported.
#   2. lod comparison — scan same loci, compare lod per gene.
#      two-tier pass:
#        strict  — r > 0.999 and max|Δlod| < 0.01
#        lenient — r > 0.990 and rel_error < 5% of peak R lod
#   3. qc plots:
#        lod_correlation.png   — scatter Python vs R lod, all loci
#        lod_residuals.png     — Δlod by chromosome
#        pvalue_qq.png         — qq of Python and R p-values with λ_GC
#        founder_effects.png   — per-founder β at peak locus (Python vs R)
#        gene_summary.png      — per-gene r and rel_error distributions
#
# outputs:
#   validation/r_data/lod_comparison.csv
#   validation/r_data/gene_summary.csv
#   validation/r_data/gene_results.csv   ← primary inspection file
#   validation/qc_plots/*.png
#
# prerequisites (run first):
#   Rscript validation/r_reference/run_validation.R [chr] [n_genes]
#
# usage:
#   python validation/validate_vs_r.py                    # all chrs, all genes
#   python validation/validate_vs_r.py --chr 8            # chromosome 8 only
#   python validation/validate_vs_r.py --chr 1,8,10       # specific chromosomes
#   python validation/validate_vs_r.py --chr 8 --n_genes 20  # quick test

import numpy as np
import pandas as pd
import os, sys, argparse, logging
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger('validate')

# ── paths ──────────────────────────────────────────────────────────────────────
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

cache_dir = os.path.join(miqtl_dir, 'output', 'genomecache')
pmap_path = os.path.join(miqtl_dir, 'input',  'pmap.csv')
val_dir   = os.path.join(pkg_dir,   'validation', 'r_data')
plot_dir  = os.path.join(pkg_dir,   'validation', 'qc_plots')
os.makedirs(val_dir,  exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

founder_colors = {
    'AJ':     '#F4C245', 'B6J':    '#C0BFBF', '129S1J': '#DA9CC1',
    'NODJ':   '#4F6EAF', 'NZOJ':   '#52A5DB', 'CASTJ':  '#55AF5B',
    'PWKJ':   '#D83026', 'WSBJ':   '#683C91',
}

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='mosaiqtl vs miQTL cross-language validation')
parser.add_argument('--chr',     default=None,
                    help='chromosome(s), comma-separated (default: all). '
                         'e.g. --chr 8  or  --chr 1,8,10')
parser.add_argument('--n_genes', default=None, type=int,
                    help='cap genes per chromosome for quick testing (default: all)')
args = parser.parse_args()

# ── check prerequisites ────────────────────────────────────────────────────────
for path, label in [
    (os.path.join(val_dir, 'lod_r.csv'),        'lod_r.csv'),
    (os.path.join(val_dir, 'pheno_48_used.csv'), 'pheno_48_used.csv'),
    (cache_dir,                                   'genomecache/'),
    (pmap_path,                                   'pmap.csv'),
]:
    if not os.path.exists(path):
        log.error(f'{label} not found at {path}')
        if 'lod_r' in label or 'pheno_48' in label:
            log.error('run first: Rscript validation/r_reference/run_validation.R')
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 1 — DATA INTEGRITY')
log.info('='*60)

integrity_pass = True

# ── 1a. pheno_48 ───────────────────────────────────────────────────────────────
r_pheno = pd.read_csv(os.path.join(val_dir, 'pheno_48_used.csv'), index_col=0)
if 'subject.name' not in r_pheno.columns:
    r_pheno.index.name = 'subject.name'
    r_pheno = r_pheno.reset_index()
else:
    r_pheno = r_pheno.reset_index(drop=True)

r_strains = r_pheno['subject.name'].tolist()
meta = {'subject.name', 'batch'} | {f'pc{i}' for i in range(1, 11)}
gene_cols_r = [c for c in r_pheno.columns if c not in meta]
n_na = r_pheno[gene_cols_r].isna().sum().sum()
log.info(f'pheno_48 (R): {r_pheno.shape[0]} strains × {r_pheno.shape[1]} cols | '
         f'{len(gene_cols_r)} genes | {n_na} NaN values')
log.info(f'  strain order (first 5): {r_strains[:5]}')

# ── 1b. pmap ───────────────────────────────────────────────────────────────────
pmap = pd.read_csv(pmap_path)
if 'marker' in pmap.columns:
    pmap = pmap.rename(columns={'marker': 'locus', 'pos': 'pos_mb'})
pmap['locus'] = pmap['locus'].str.replace('.RData', '', regex=False)
pmap['chr']   = pmap['chr'].astype(str)

lod_r = pd.read_csv(os.path.join(val_dir, 'lod_r.csv'))
lod_r['chr'] = lod_r['chr'].astype(str)
r_loci  = set(lod_r['locus'].unique())
py_loci = set(pmap['locus'].unique())
missing_from_py = r_loci - py_loci

log.info(f'\npmap: Python={len(pmap)} loci | R lod_r={len(r_loci)} loci | '
         f'shared={len(r_loci & py_loci)} | '
         f'in R but not Python pmap: {len(missing_from_py)}')
if missing_from_py:
    log.warning(f'  {len(missing_from_py)} R loci missing from Python pmap')
    integrity_pass = False

# position agreement for shared loci
pmap_idx = pmap.set_index('locus')
r_pos    = lod_r.drop_duplicates('locus').set_index('locus')['pos_mb']
shared   = list((r_loci & py_loci) & set(r_pos.index))
if shared:
    pos_diff = (pmap_idx.loc[shared, 'pos_mb'] - r_pos.loc[shared]).abs()
    log.info(f'  position agreement (shared loci): max|Δpos| = {pos_diff.max():.6f} Mb')
    if pos_diff.max() > 0.001:
        log.warning('  position mismatch > 1 kb — pmap versions may differ')
        integrity_pass = False

# ── 1c. genome cache spot-check ────────────────────────────────────────────────
log.info(f'\nloading genome cache from {cache_dir}')
cache = GenomeCache.from_happy_dir(cache_dir, model='additive')
log.info(f'  chromosomes: {cache.chromosomes}')
log.info(f'  strains (first 5): {list(cache.sample_ids[:5])}')
log.info(f'  founders: {cache.founders}')

cache_strains = list(cache.sample_ids)
if set(cache_strains) != set(r_strains):
    missing = set(r_strains) - set(cache_strains)
    log.warning(f'  {len(missing)} R strains not in cache: {list(missing)[:5]}')
    integrity_pass = False
elif cache_strains != r_strains:
    log.warning('  cache strain ORDER differs from R pheno_48 — will reindex')
else:
    log.info('  strain order: matches R pheno_48 exactly ✓')

# haplotype row-sum spot-check
spot_chr   = cache.chromosomes[0]
spot_loci  = cache._data[spot_chr]['locus_ids']
spot_locus = spot_loci[len(spot_loci) // 2]
h_spot     = cache.get_locus(spot_chr, int(np.where(spot_loci == spot_locus)[0][0]))
row_sums   = h_spot.sum(axis=1)
log.info(f'\ncache spot-check (chr{spot_chr}, {spot_locus}): '
         f'shape={h_spot.shape}  row_sum=[{row_sums.min():.6f}, {row_sums.max():.6f}]')
if not np.allclose(row_sums, 1.0, atol=1e-4):
    log.warning('  haplotype probs do not sum to 1 — cache may be corrupted')
    integrity_pass = False
else:
    log.info('  haplotype probs sum to 1.0 ✓')

log.info(f'\ndata integrity: {"PASS ✓" if integrity_pass else "ISSUES FOUND — check warnings above"}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 2 — LOD COMPARISON')
log.info('='*60)

# chromosomes to scan
all_chrs = sorted(lod_r['chr'].unique(), key=lambda c: int(c) if c.isdigit() else 99)
if args.chr:
    requested   = [c.strip() for c in args.chr.split(',')]
    missing_c   = [c for c in requested if c not in all_chrs]
    if missing_c:
        log.warning(f'chromosomes not in lod_r.csv: {missing_c}')
    chromosomes = [c for c in requested if c in all_chrs]
else:
    chromosomes = all_chrs

log.info(f'chromosomes: {chromosomes}')
log.info(f'genes in scope: {lod_r[lod_r["chr"].isin(chromosomes)]["gene_id"].nunique()}')

# align pheno_48 to cache strain order + build covariate matrix
pheno_48      = r_pheno.set_index('subject.name').reindex(cache_strains)
pc_cols       = [f'pc{i}' for i in range(1, 11)]
batch_dummies = pd.get_dummies(pheno_48['batch'], prefix='batch', drop_first=True)
covar_base    = pd.concat([
    pd.Series(np.ones(len(pheno_48)), index=pheno_48.index, name='intercept'),
    batch_dummies.astype(float),
    pheno_48[pc_cols].astype(float),
], axis=1).values   # (48, p)
pheno_48 = pheno_48.reset_index()

# load optional allele effects from R
ae_path = os.path.join(val_dir, 'allele_effects.csv')
ae_r    = pd.read_csv(ae_path) if os.path.exists(ae_path) else None

all_pass    = True
comparisons = []
gene_stats  = []
py_betas    = {}   # gene → (chrom, peak_locus, founders, betas)

for chrom in chromosomes:
    if chrom not in cache.chromosomes:
        log.warning(f'chr{chrom} not in genome cache — skipping')
        continue

    chr_genes = lod_r[lod_r['chr'] == chrom]['gene_id'].unique().tolist()
    if args.n_genes:
        chr_genes = chr_genes[:args.n_genes]

    log.info(f'\n[chr{chrom}] {len(chr_genes)} genes')

    # loco kinship eigendecomposition — once per chromosome
    kinship_dict     = kin_module.calc_kinship(cache, method='loco')
    k_chr            = kinship_dict[chrom]
    eigvals, eigvecs = np.linalg.eigh(k_chr)
    covar_rot        = eigvecs.T @ covar_base   # (48, p)

    for gene in chr_genes:
        if gene not in pheno_48.columns:
            log.warning(f'  {gene}: not in pheno_48 — skipping')
            continue

        y = (pheno_48.set_index('subject.name')
                      .reindex(cache_strains)[gene]
                      .values.astype(float))
        if np.all(np.isnan(y)):
            log.warning(f'  {gene}: all NaN — skipping')
            continue

        y_rot    = eigvecs.T @ y
        h2_null, _ = lmm_module.estimate_h2_null(y_rot, covar_rot, eigvals)

        r_gene = (lod_r[(lod_r['gene_id'] == gene) & (lod_r['chr'] == chrom)]
                  [['locus', 'lod', 'pos_mb']]
                  .rename(columns={'lod': 'lod_r'}))
        if r_gene.empty:
            continue

        lod_py    = []
        betas_py  = []
        locus_ids = cache._data[chrom]['locus_ids']

        for locus in r_gene['locus']:
            idx = np.where(locus_ids == locus)[0]
            if len(idx) == 0:
                lod_py.append(np.nan); betas_py.append(None); continue
            h_l    = cache.get_locus(chrom, int(idx[0]))
            h_rot  = eigvecs.T @ h_l[:, :-1]
            result = lmm_module.fit_lmm_at_locus(
                y_rot, h_rot, covar_rot, eigvals, h2_null)
            lod_py.append(result.lod)
            betas_py.append(result.betas)

        r_gene           = r_gene.copy()
        r_gene['lod_py'] = lod_py
        merged           = r_gene.dropna(subset=['lod_r', 'lod_py'])

        if len(merged) < 3:
            log.warning(f'  {gene}: only {len(merged)} valid loci — skipping')
            all_pass = False
            continue

        r_corr      = float(np.corrcoef(merged['lod_r'], merged['lod_py'])[0, 1])
        max_d       = float((merged['lod_r'] - merged['lod_py']).abs().max())
        peak_lod_r  = float(merged['lod_r'].max())
        peak_lod_py = float(merged['lod_py'].max())
        peak_idx_r  = merged['lod_r'].idxmax()
        lead_locus  = merged.loc[peak_idx_r, 'locus']
        delta_peak  = float(abs(merged.loc[peak_idx_r, 'lod_r'] -
                                merged.loc[peak_idx_r, 'lod_py']))
        rel_err     = max_d / max(peak_lod_r, 0.01)

        # two-tier pass:
        #   strict  — near-exact agreement (r > 0.999, max|Δ| < 0.01)
        #   lenient — shape correct, small relative offset from h² boundary
        #             differences (r > 0.990, rel_err < 5%)
        strict  = r_corr > 0.999 and max_d < 0.01
        lenient = r_corr > 0.990 and rel_err < 0.05
        passed  = strict or lenient
        tier    = 'strict' if strict else ('lenient' if lenient else 'fail')
        status  = 'PASS' if passed else 'FAIL'

        log.info(f'  {status}({tier:<7})  {gene:25s}  r={r_corr:.6f}  '
                 f'max|Δlod|={max_d:.6f}  rel={rel_err:.3f}  n={len(merged)}')

        if not passed:
            all_pass = False

        merged['gene_id'] = gene
        merged['chr']     = chrom
        comparisons.append(merged)
        gene_stats.append({
            'gene_id':       gene,
            'chr':           chrom,
            'status':        status,
            'tier':          tier,
            'r':             r_corr,
            'max_delta_lod': max_d,
            'rel_error':     rel_err,
            'peak_lod_r':    peak_lod_r,
            'peak_lod_py':   peak_lod_py,
            'delta_at_peak': delta_peak,
            'lead_locus':    lead_locus,
            'n_loci':        len(merged),
            'h2_null':       h2_null,
            'passed':        passed,
        })

        # store betas at peak locus for founder effects plot
        locus_list = r_gene['locus'].tolist()
        if lead_locus in locus_list:
            bi = locus_list.index(lead_locus)
            if betas_py[bi] is not None:
                py_betas[gene] = (chrom, lead_locus,
                                  cache.founders[:-1], betas_py[bi])

# ── save ───────────────────────────────────────────────────────────────────────
if not comparisons:
    log.error('no comparison data — all genes skipped')
    sys.exit(1)

comp_df = pd.concat(comparisons, ignore_index=True)
stat_df = pd.DataFrame(gene_stats)
comp_df.to_csv(os.path.join(val_dir, 'lod_comparison.csv'), index=False)
stat_df.to_csv(os.path.join(val_dir, 'gene_summary.csv'),   index=False)

# gene_results.csv — primary inspection file, sorted by worst failures first
gene_results_cols = [
    'gene_id', 'chr', 'status', 'tier',
    'peak_lod_r', 'peak_lod_py', 'delta_at_peak',
    'max_delta_lod', 'rel_error', 'r',
    'lead_locus', 'n_loci', 'h2_null',
]
gene_results = stat_df[gene_results_cols].sort_values(
    ['status', 'rel_error'], ascending=[True, False])
gene_results.to_csv(os.path.join(val_dir, 'gene_results.csv'), index=False)

log.info(f'\nsaved: lod_comparison.csv ({len(comp_df)} rows)')
log.info(f'        gene_summary.csv    ({len(stat_df)} genes)')
log.info(f'        gene_results.csv    ({len(stat_df)} genes — primary inspection file)')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 3 — QC PLOTS')
log.info('='*60)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Carlito', 'DejaVu Sans', 'Arial'],
        'font.size':       11,
        'axes.titlesize':  13,
        'axes.labelsize':  12,
    })
    chr_order   = sorted(comp_df['chr'].unique(),
                         key=lambda c: int(c) if c.isdigit() else 99)
    chr_palette = ['#4F6EAF', '#C0BFBF']

    def chr_colors(series):
        return [chr_palette[chr_order.index(c) % 2] for c in series]

    # ── plot 1: lod correlation ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(comp_df['lod_r'], comp_df['lod_py'],
               c=chr_colors(comp_df['chr']), s=6, alpha=0.4,
               edgecolors='none', rasterized=True)
    lim = max(comp_df['lod_r'].max(), comp_df['lod_py'].max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#D83026', lw=1, label='$y=x$')
    r_all = float(np.corrcoef(comp_df['lod_r'], comp_df['lod_py'])[0, 1])
    dif   = comp_df['lod_r'] - comp_df['lod_py']
    ax.text(0.05, 0.95,
            f'$r$ = {r_all:.6f}\nmean $\\Delta$ = {dif.mean():.4f}\n'
            f'max $|\\Delta|$ = {dif.abs().max():.4f}\n$n$ = {len(comp_df)} loci',
            transform=ax.transAxes, fontsize=9, va='top')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('miQTL LOD (R)'); ax.set_ylabel('mosaiQTL LOD (Python)')
    ax.set_title('LOD cross-language comparison')
    ax.legend(fontsize=9); sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lod_correlation.png'), dpi=150)
    plt.close(); log.info('saved: lod_correlation.png')

    # ── plot 2: lod residuals by chromosome ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    diff   = comp_df['lod_py'] - comp_df['lod_r']
    colors = chr_colors(comp_df['chr'])
    ax.scatter(range(len(diff)), diff, c=colors, s=5, alpha=0.4,
               edgecolors='none', rasterized=True)
    ax.axhline(0, color='#D83026', lw=0.8, linestyle='--')
    ax.text(0.99, 0.97,
            f'mean $\\Delta$ = {diff.mean():.4f}\n'
            f'sd = {diff.std():.4f}\nmax $|\\Delta|$ = {diff.abs().max():.4f}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right')
    offset = 0
    for c in chr_order:
        n_c = (comp_df['chr'] == c).sum()
        if offset > 0:
            ax.axvline(offset, color='white', lw=0.5)
        ax.text(offset + n_c / 2, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else -0.02,
                f'chr{c}', ha='center', fontsize=7, color='#444')
        offset += n_c
    ax.set_xlim(0, len(diff))
    ax.set_xlabel('locus index (sorted by chromosome)')
    ax.set_ylabel('$\\Delta$LOD (Python $-$ R)')
    ax.set_title('LOD residuals by chromosome')
    sns.despine(); plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lod_residuals.png'), dpi=150)
    plt.close(); log.info('saved: lod_residuals.png')

    # ── plot 3: p-value QQ + λ_GC comparison ─────────────────────────────────
    n_sub  = len(cache_strains)
    q_hap  = cache.n_founders - 1
    p_cov  = covar_base.shape[1]
    df1    = q_hap

    chi2_py = (2 * comp_df['lod_py'] * np.log(10)).clip(lower=0)
    pvals_py = scipy_stats.chi2.sf(chi2_py, df=df1)

    r_merged = comp_df[['locus', 'chr', 'gene_id', 'lod_r']].merge(
        lod_r[['locus', 'chr', 'gene_id', 'p_value']],
        on=['locus', 'chr', 'gene_id'], how='left')
    pvals_r = r_merged['p_value'].values

    median_chi2_df1 = scipy_stats.chi2.ppf(0.5, df=df1)
    chi2_r = (2 * comp_df['lod_r'] * np.log(10)).clip(lower=0)
    lam_py = np.nanmedian(chi2_py) / median_chi2_df1
    lam_r  = np.nanmedian(chi2_r)  / median_chi2_df1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pv, lam, title, color in [
        (axes[0], pvals_py, lam_py, 'mosaiQTL (Python)', '#4F6EAF'),
        (axes[1], pvals_r,  lam_r,  'miQTL (R)',         '#55AF5B'),
    ]:
        valid = pv[np.isfinite(pv) & (pv > 0)]
        obs   = -np.log10(np.sort(valid))
        exp   = -np.log10(np.linspace(1/len(obs), 1, len(obs))[::-1])
        ax.scatter(exp, obs, s=5, alpha=0.4, color=color,
                   edgecolors='none', rasterized=True)
        m = max(exp.max(), obs.max()) * 1.05
        ax.plot([0, m], [0, m], '--', color='#D83026', lw=1, label='$y=x$')
        ax.text(0.05, 0.95, f'$\\lambda_{{GC}}$ = {lam:.3f}',
                transform=ax.transAxes, fontsize=10, va='top')
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_xlabel('expected $-\\log_{{10}}(p)$')
        ax.set_ylabel('observed $-\\log_{{10}}(p)$')
        ax.set_title(f'QQ — {title}')
        ax.legend(fontsize=9); sns.despine(ax=ax)

    plt.suptitle(f'$\\lambda_{{GC}}$ comparison: Python={lam_py:.3f}  R={lam_r:.3f}',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'pvalue_qq.png'), dpi=150, bbox_inches='tight')
    plt.close(); log.info(f'saved: pvalue_qq.png  (lambda_py={lam_py:.3f}  lambda_r={lam_r:.3f})')

    # ── plot 4: founder effects — Python vs R at peak locus ──────────────────
    if ae_r is not None and py_betas:
        best_gene = (lod_r[lod_r['chr'].isin(chromosomes)]
                     .groupby('gene_id')['lod'].max().idxmax())
        if best_gene in py_betas:
            chrom_b, peak_loc, f_names, py_eff = py_betas[best_gene]
            r_ae = ae_r[(ae_r['gene_id'] == best_gene) &
                        (ae_r['locus'] == peak_loc)]
            fig, ax = plt.subplots(figsize=(9, 4))
            x      = np.arange(len(f_names))
            width  = 0.35
            colors_f = [founder_colors.get(f, '#888888') for f in f_names]
            ax.bar(x - width/2, py_eff, width, color=colors_f,
                   edgecolor='white', lw=0.5, label='mosaiQTL (Python)')
            if not r_ae.empty:
                r_eff = [float(r_ae.loc[r_ae['founder'] == f, 'effect'].iloc[0])
                         if f in r_ae['founder'].values else 0.0
                         for f in f_names]
                ax.bar(x + width/2, r_eff, width, color=colors_f,
                       edgecolor='#333', lw=0.8, alpha=0.55, label='miQTL (R)')
                corr_ae = float(np.corrcoef(py_eff, r_eff)[0, 1])
                ax.text(0.02, 0.97, f'$r$ = {corr_ae:.4f}',
                        transform=ax.transAxes, fontsize=10, va='top')
            ax.axhline(0, color='#444', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(f_names, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('founder effect ($\\hat{\\beta}_f$)')
            ax.set_title(f'founder effects — {best_gene}  chr{chrom_b}  {peak_loc}')
            ax.legend(fontsize=9); sns.despine()
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'founder_effects.png'), dpi=150)
            plt.close(); log.info(f'saved: founder_effects.png  (gene={best_gene})')
    else:
        log.info('founder_effects.png: no allele_effects.csv or no betas — skipping')

    # ── plot 5: per-gene summary distributions ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    tier_colors = {'strict': '#4F6EAF', 'lenient': '#55AF5B', 'fail': '#D83026'}
    for tier_val, grp in stat_df.groupby('tier'):
        sns.histplot(grp['r'], bins=30, color=tier_colors.get(tier_val, '#888'),
                     ax=ax, edgecolor='white', label=tier_val, alpha=0.75)
    ax.axvline(0.999, color='#D83026', lw=1, linestyle='--')
    ax.axvline(0.990, color='#55AF5B', lw=1, linestyle=':')
    ax.set_xlim(left=min(stat_df['r'].min() - 0.05, -0.1)); ax.set_ylim(bottom=0)
    ax.set_xlabel('Pearson $r$ (Python vs R LOD)')
    ax.set_ylabel('genes')
    n_s = (stat_df['tier'] == 'strict').sum()
    n_l = (stat_df['tier'] == 'lenient').sum()
    n_f = (stat_df['tier'] == 'fail').sum()
    ax.set_title(f'LOD correlation per gene\nstrict={n_s}  lenient={n_l}  fail={n_f}')
    ax.legend(fontsize=9); sns.despine(ax=ax)

    ax = axes[1]
    sns.histplot(stat_df['rel_error'], bins=40, color='#683C91', ax=ax, edgecolor='white')
    ax.axvline(0.05, color='#D83026', lw=1.2, linestyle='--', label='lenient threshold 5%')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel('relative error: max|delta LOD| / peak LOD (R)')
    ax.set_ylabel('genes')
    ax.set_title(f'Relative LOD error per gene\n(n={len(stat_df)} genes)')
    ax.legend(fontsize=9); sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'gene_summary.png'), dpi=150)
    plt.close(); log.info('saved: gene_summary.png')

    log.info(f'\nall plots saved to {plot_dir}/')

except Exception as e:
    import traceback
    log.warning(f'plot generation failed:\n{traceback.format_exc()}')

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
n_strict  = sum(g['tier'] == 'strict'  for g in gene_stats)
n_lenient = sum(g['tier'] == 'lenient' for g in gene_stats)
n_fail    = sum(g['tier'] == 'fail'    for g in gene_stats)
n_pass    = n_strict + n_lenient

print()
print('=' * 60)
print(f'  data integrity:  {"PASS" if integrity_pass else "ISSUES — see log"}')
print(f'  LOD comparison:  {n_pass} PASS  '
      f'({n_strict} strict + {n_lenient} lenient)  /  {n_fail} FAIL')
if gene_stats:
    print(f'  median r:        {np.median([g["r"] for g in gene_stats]):.6f}')
    print(f'  median rel_err:  {np.median([g["rel_error"] for g in gene_stats]):.4f}')
print(f'  strict:   r > 0.999 and max|Δlod| < 0.01')
print(f'  lenient:  r > 0.990 and rel_error < 5% of peak LOD')
print(f'  outputs:  validation/r_data/gene_results.csv')
print('=' * 60)

sys.exit(0 if (all_pass and integrity_pass) else 1)
