# validate_vs_r.py
# full end-to-end cross-language validation: mosaiqtl (Python) vs miQTL (R)
#
# Python builds everything from scratch from the same raw inputs R used:
#   probs_cc.rds + probs_founder.rds + probs_f1.rds + pmap.csv
#       → GenomeCache.from_rds()          (Python-native prob arrays)
#       → kin_module.calc_kinship()       (Python-native LOCO kinship)
#       → lmm_module.estimate_h2_null()   (Python REML)
#       → lmm_module.fit_lmm_at_locus()   (Python LMM, Kang 2008)
#
# three comparison layers:
#   1. haplotype probs   — Python cache vs R HAPPY cache spot-check
#   2. kinship matrices  — Python LOCO K vs R kinship_loco.rds
#   3. LOD scores        — Python scan vs R lod_r.csv (from run_validation.R)
#
# qc plots (all Python vs R):
#   haplotype_probs.png   — spot locus heatmap Python vs R
#   kinship_heatmap.png   — Python kinship with founder/F1/CC block structure
#   kinship_diff.png      — Python minus R kinship at one chromosome
#   lod_correlation.png   — scatter Python vs R lod, all loci
#   lod_residuals.png     — delta lod by chromosome
#   pvalue_qq.png         — side-by-side QQ with lambda_GC
#   founder_effects.png   — per-founder beta at peak locus
#   gene_summary.png      — per-gene r and rel_error distributions
#
# prerequisites:
#   Rscript validation/r_reference/run_validation.R [chr] [n_genes]
#
# usage:
#   python validation/validate_vs_r.py                    # all chrs, all genes
#   python validation/validate_vs_r.py --chr 8            # chr 8 only
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
# miqtl_dir: original pipeline outputs (raw rds files, HAPPY cache, kinship)
# pkg_dir:   mosaiqtl repo root
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

# raw prob files (produced by founders_f1_genomecache.ipynb)
probs_cc_path      = os.path.join(miqtl_dir, 'output', 'probs_cc.rds')
probs_founder_path = os.path.join(miqtl_dir, 'output', 'probs_founder.rds')
probs_f1_path      = os.path.join(miqtl_dir, 'output', 'probs_f1.rds')
# reference outputs from R pipeline
kinship_r_path = os.path.join(miqtl_dir, 'output', 'kinship_loco.rds')
cache_r_dir    = os.path.join(miqtl_dir, 'output', 'genomecache')
pmap_path      = os.path.join(miqtl_dir, 'input',  'pmap.csv')
# validation outputs from run_validation.R
val_dir  = os.path.join(pkg_dir, 'validation', 'r_data')
plot_dir = os.path.join(pkg_dir, 'validation', 'qc_plots')
os.makedirs(val_dir,  exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

founder_colors = {
    'AJ':     '#F4C245', 'B6J':    '#C0BFBF', '129S1J': '#DA9CC1',
    'NODJ':   '#4F6EAF', 'NZOJ':   '#52A5DB', 'CASTJ':  '#55AF5B',
    'PWKJ':   '#D83026', 'WSBJ':   '#683C91',
}

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='mosaiqtl vs miQTL full end-to-end validation')
parser.add_argument('--chr',     default=None,
                    help='chromosome(s), comma-separated (default: all). '
                         'e.g. --chr 8  or  --chr 1,8,10')
parser.add_argument('--n_genes', default=None, type=int,
                    help='cap genes per chromosome for quick testing (default: all)')
args = parser.parse_args()

# ── check prerequisites ────────────────────────────────────────────────────────
required = [
    (probs_cc_path,      'probs_cc.rds'),
    (probs_founder_path, 'probs_founder.rds'),
    (probs_f1_path,      'probs_f1.rds'),
    (kinship_r_path,     'kinship_loco.rds'),
    (pmap_path,          'pmap.csv'),
    (os.path.join(val_dir, 'lod_r.csv'),         'lod_r.csv'),
    (os.path.join(val_dir, 'pheno_48_used.csv'),  'pheno_48_used.csv'),
]
for path, label in required:
    if not os.path.exists(path):
        log.error(f'{label} not found at {path}')
        if 'lod_r' in label or 'pheno_48' in label:
            log.error('run first: Rscript validation/r_reference/run_validation.R')
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BUILD PYTHON GENOME CACHE FROM SCRATCH
# reads raw rds files directly, not R's HAPPY cache output
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 1 — BUILD GENOME CACHE FROM SCRATCH')
log.info('='*60)
log.info('source: probs_cc.rds + probs_founder.rds + probs_f1.rds')
log.info('method: GenomeCache.from_rds() — mirrors notebook steps 5-7')

cache_py = GenomeCache.from_rds(
    probs_cc_path, probs_founder_path, probs_f1_path,
    pmap_path, founders=CC_FOUNDERS
)
log.info(f'Python cache: {len(cache_py.chromosomes)} chr | '
         f'{len(cache_py.sample_ids)} strains | {cache_py.n_founders} founders')
log.info(f'  strains (first 5): {list(cache_py.sample_ids[:5])}')
log.info(f'  founders: {cache_py.founders}')

# ─ 1a. spot-check haplotype probs: Python vs R HAPPY cache ───────────────────
# compare Python-built probs against the probs R stored in the HAPPY cache
# for one locus on the first available chromosome
cache_r_ok = os.path.isdir(cache_r_dir)
hap_match  = True

if cache_r_ok:
    log.info('\nspot-checking haplotype probs: Python vs R HAPPY cache')
    cache_r = GenomeCache.from_happy_dir(cache_r_dir, model='additive')

    spot_chr   = cache_py.chromosomes[0]
    spot_loci  = cache_py._data[spot_chr]['locus_ids']
    spot_locus = spot_loci[len(spot_loci) // 2]   # middle locus

    py_h = cache_py.get_locus(
        spot_chr, int(np.where(spot_loci == spot_locus)[0][0]))

    r_loci = cache_r._data[spot_chr]['locus_ids']
    r_idx  = np.where(r_loci == spot_locus)[0]
    if len(r_idx):
        r_h = cache_r.get_locus(spot_chr, int(r_idx[0]))
        # align strain order between Python and R caches
        py_strains = list(cache_py.sample_ids)
        r_strains  = list(cache_r.sample_ids)
        shared     = [s for s in py_strains if s in r_strains]
        py_idx_s   = [py_strains.index(s) for s in shared]
        r_idx_s    = [r_strains.index(s)  for s in shared]
        max_diff   = np.max(np.abs(py_h[py_idx_s] - r_h[r_idx_s]))
        row_sums   = py_h.sum(axis=1)
        log.info(f'  chr{spot_chr} locus {spot_locus}: shape={py_h.shape}')
        log.info(f'  row sums: [{row_sums.min():.6f}, {row_sums.max():.6f}]')
        log.info(f'  max|Δprob| vs R HAPPY cache: {max_diff:.8f}')
        if max_diff > 1e-4:
            log.warning('  haplotype prob mismatch > 1e-4 vs R HAPPY cache')
            hap_match = False
        else:
            log.info('  haplotype probs match R HAPPY cache ✓')
    else:
        log.warning(f'  locus {spot_locus} not found in R HAPPY cache — skipping spot-check')
else:
    log.info('  R HAPPY cache not found — skipping prob comparison')

# ─ 1b. verify row sums = 1 across all chromosomes ────────────────────────────
bad_chrs = []
for chrom in cache_py.chromosomes:
    lids = cache_py._data[chrom]['locus_ids']
    mid  = len(lids) // 2
    h    = cache_py.get_locus(chrom, mid)
    if not np.allclose(h.sum(axis=1), 1.0, atol=1e-4):
        bad_chrs.append(chrom)
if bad_chrs:
    log.warning(f'  row sums != 1 on chromosomes: {bad_chrs}')
    hap_match = False
else:
    log.info(f'  row sums = 1.0 on all {len(cache_py.chromosomes)} chromosomes ✓')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPUTE LOCO KINSHIP FROM SCRATCH AND COMPARE TO R
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 2 — KINSHIP COMPARISON')
log.info('='*60)
log.info('Python: calc_kinship(cache_py, method=loco)')
log.info('R ref:  kinship_loco.rds')

kinship_py = kin_module.calc_kinship(cache_py, method='loco')
log.info(f'Python LOCO kinship: {len(kinship_py)} chromosomes')

# load R kinship via rpy2
import rpy2.robjects as ro
ro.r(f'k_list <- readRDS("{kinship_r_path}")')
chr_names_r = list(ro.r['names'](ro.r['k_list']))
kinship_r   = {}
for c in chr_names_r:
    kinship_r[c] = np.array(ro.r[f'k_list[["{c}"]]'], dtype=float)
r_strains_k = list(ro.r['rownames'](ro.r[f'k_list[["{chr_names_r[0]}"]]']))
log.info(f'R LOCO kinship: {len(kinship_r)} chromosomes | '
         f'{len(r_strains_k)} strains')

# compare kinship matrices on shared chromosomes + shared strains
kinship_pass = True
kinship_diffs = {}
shared_chrs = [c for c in cache_py.chromosomes if c in kinship_r]
py_strains_k = list(cache_py.sample_ids)
shared_s     = [s for s in py_strains_k if s in r_strains_k]
py_idx_k     = np.array([py_strains_k.index(s) for s in shared_s])
r_idx_k      = np.array([r_strains_k.index(s)  for s in shared_s])

for chrom in shared_chrs:
    k_py = kinship_py[chrom][np.ix_(py_idx_k, py_idx_k)]
    k_r  = kinship_r[chrom][np.ix_(r_idx_k,  r_idx_k)]
    diff = np.abs(k_py - k_r)
    max_d = float(diff.max())
    corr  = float(np.corrcoef(k_py.ravel(), k_r.ravel())[0, 1])
    kinship_diffs[chrom] = max_d
    status = '✓' if max_d < 1e-4 else '✗'
    log.info(f'  chr{chrom}: max|ΔK| = {max_d:.8f}  r = {corr:.8f}  {status}')
    if max_d > 1e-4:
        kinship_pass = False

log.info(f'kinship comparison: {"PASS ✓" if kinship_pass else "MISMATCH — check above"}')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 3 — LOD COMPARISON')
log.info('='*60)

# load R reference lod scores and pheno_48
lod_r = pd.read_csv(os.path.join(val_dir, 'lod_r.csv'))
lod_r['chr'] = lod_r['chr'].astype(str)

r_pheno = pd.read_csv(os.path.join(val_dir, 'pheno_48_used.csv'), index_col=0)
if 'subject.name' not in r_pheno.columns:
    r_pheno.index.name = 'subject.name'
    r_pheno = r_pheno.reset_index()
else:
    r_pheno = r_pheno.reset_index(drop=True)

# chromosomes to scan
all_chrs = sorted(lod_r['chr'].unique(), key=lambda c: int(c) if c.isdigit() else 99)
if args.chr:
    requested   = [c.strip() for c in args.chr.split(',')]
    chromosomes = [c for c in requested if c in all_chrs]
    missing_c   = [c for c in requested if c not in all_chrs]
    if missing_c:
        log.warning(f'chromosomes not in lod_r.csv: {missing_c}')
else:
    chromosomes = all_chrs

log.info(f'chromosomes: {chromosomes}')
log.info(f'genes in scope: {lod_r[lod_r["chr"].isin(chromosomes)]["gene_id"].nunique()}')

# align pheno_48 to Python cache strain order + build covariate matrix
cache_strains = list(cache_py.sample_ids)
pheno_48      = r_pheno.set_index('subject.name').reindex(cache_strains)
pc_cols       = [f'pc{i}' for i in range(1, 11)]
batch_dummies = pd.get_dummies(pheno_48['batch'], prefix='batch', drop_first=True)
covar_base    = pd.concat([
    pd.Series(np.ones(len(pheno_48)), index=pheno_48.index, name='intercept'),
    batch_dummies.astype(float),
    pheno_48[pc_cols].astype(float),
], axis=1).values
pheno_48 = pheno_48.reset_index()

ae_path = os.path.join(val_dir, 'allele_effects.csv')
ae_r    = pd.read_csv(ae_path) if os.path.exists(ae_path) else None

all_pass    = True
comparisons = []
gene_stats  = []
py_betas    = {}

for chrom in chromosomes:
    if chrom not in cache_py.chromosomes:
        log.warning(f'chr{chrom} not in Python cache — skipping')
        continue

    chr_genes = lod_r[lod_r['chr'] == chrom]['gene_id'].unique().tolist()
    if args.n_genes:
        chr_genes = chr_genes[:args.n_genes]

    log.info(f'\n[chr{chrom}] {len(chr_genes)} genes')

    # use Python-computed kinship for this chromosome
    k_chr            = kinship_py[chrom]
    eigvals, eigvecs = np.linalg.eigh(k_chr)
    covar_rot        = eigvecs.T @ covar_base

    for gene in chr_genes:
        if gene not in pheno_48.columns:
            log.warning(f'  {gene}: not in pheno_48 — skipping')
            continue

        y = (pheno_48.set_index('subject.name')
                      .reindex(cache_strains)[gene]
                      .values.astype(float))
        if np.all(np.isnan(y)):
            continue

        y_rot      = eigvecs.T @ y
        h2_null, _ = lmm_module.estimate_h2_null(y_rot, covar_rot, eigvals)

        r_gene = (lod_r[(lod_r['gene_id'] == gene) & (lod_r['chr'] == chrom)]
                  [['locus', 'lod', 'pos_mb']]
                  .rename(columns={'lod': 'lod_r'}))
        if r_gene.empty:
            continue

        lod_py    = []
        betas_py  = []
        locus_ids = cache_py._data[chrom]['locus_ids']

        for locus in r_gene['locus']:
            idx = np.where(locus_ids == locus)[0]
            if len(idx) == 0:
                lod_py.append(np.nan); betas_py.append(None); continue
            h_l    = cache_py.get_locus(chrom, int(idx[0]))
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

        locus_list = r_gene['locus'].tolist()
        if lead_locus in locus_list:
            bi = locus_list.index(lead_locus)
            if betas_py[bi] is not None:
                py_betas[gene] = (chrom, lead_locus,
                                  cache_py.founders[:-1], betas_py[bi])

# ── save ───────────────────────────────────────────────────────────────────────
if not comparisons:
    log.error('no comparison data — all genes skipped')
    sys.exit(1)

comp_df = pd.concat(comparisons, ignore_index=True)
stat_df = pd.DataFrame(gene_stats)
comp_df.to_csv(os.path.join(val_dir, 'lod_comparison.csv'), index=False)
stat_df.to_csv(os.path.join(val_dir, 'gene_summary.csv'),   index=False)

gene_results_cols = ['gene_id', 'chr', 'status', 'tier',
                     'peak_lod_r', 'peak_lod_py', 'delta_at_peak',
                     'max_delta_lod', 'rel_error', 'r',
                     'lead_locus', 'n_loci', 'h2_null']
stat_df[gene_results_cols].sort_values(
    ['status', 'rel_error'], ascending=[True, False]
).to_csv(os.path.join(val_dir, 'gene_results.csv'), index=False)

log.info(f'\nsaved: lod_comparison.csv ({len(comp_df)} rows) | '
         f'gene_summary.csv | gene_results.csv ({len(stat_df)} genes)')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
log.info('\n' + '='*60)
log.info('SECTION 4 — QC PLOTS')
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

    def chr_colors(s):
        return [chr_palette[chr_order.index(c) % 2] for c in s]

    # ── plot 1: haplotype probs spot-check ────────────────────────────────────
    if cache_r_ok and hap_match is not None:
        spot_chr   = cache_py.chromosomes[0]
        spot_loci  = cache_py._data[spot_chr]['locus_ids']
        spot_locus = spot_loci[len(spot_loci) // 2]
        py_h = cache_py.get_locus(
            spot_chr, int(np.where(spot_loci == spot_locus)[0][0]))

        r_loci = cache_r._data[spot_chr]['locus_ids']
        r_idx  = np.where(r_loci == spot_locus)[0]
        if len(r_idx):
            r_h = cache_r.get_locus(spot_chr, int(r_idx[0]))

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, mat, title in [
                (axes[0], py_h, 'Python (from_rds)'),
                (axes[1], r_h,  'R (from_happy_dir)'),
                (axes[2], py_h - r_h, 'Difference (Py − R)'),
            ]:
                cmap = 'YlOrRd' if 'Diff' not in title else 'RdBu_r'
                vmax = 1.0 if 'Diff' not in title else None
                vmin = 0.0 if 'Diff' not in title else None
                sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                            xticklabels=cache_py.founders,
                            yticklabels=False, cbar=True)
                ax.set_title(f'{title}\nchr{spot_chr} · {spot_locus}')
                ax.set_xlabel('founder')
                ax.set_ylabel('strain (48)')
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'haplotype_probs.png'), dpi=150)
            plt.close(); log.info('saved: haplotype_probs.png')

    # ── plot 2: Python kinship heatmap (one chromosome) ───────────────────────
    spot_chr_k = chromosomes[0] if chromosomes else cache_py.chromosomes[0]
    if spot_chr_k in kinship_py:
        k_py_plot = kinship_py[spot_chr_k][np.ix_(py_idx_k, py_idx_k)]
        k_r_plot  = kinship_r[spot_chr_k][np.ix_(r_idx_k,  r_idx_k)] \
                    if spot_chr_k in kinship_r else None

        n_cols = 3 if k_r_plot is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        ax_list = [axes] if n_cols == 1 else list(axes)

        sns.heatmap(k_py_plot, ax=ax_list[0], cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'kinship', 'shrink': 0.7})
        ax_list[0].set_title(f'Python LOCO kinship\nchr{spot_chr_k}')

        if k_r_plot is not None:
            sns.heatmap(k_r_plot, ax=ax_list[1], cmap='YlOrRd', vmin=0, vmax=1,
                        xticklabels=False, yticklabels=False,
                        cbar_kws={'label': 'kinship', 'shrink': 0.7})
            ax_list[1].set_title(f'R LOCO kinship\nchr{spot_chr_k}')

            diff_k = k_py_plot - k_r_plot
            abs_max = max(abs(diff_k.min()), abs(diff_k.max()))
            sns.heatmap(diff_k, ax=ax_list[2], cmap='RdBu_r',
                        vmin=-abs_max, vmax=abs_max,
                        xticklabels=False, yticklabels=False,
                        cbar_kws={'label': 'Py − R', 'shrink': 0.7})
            ax_list[2].set_title(f'Kinship difference\nmax|ΔK|={kinship_diffs.get(spot_chr_k, 0):.2e}')

        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'kinship_heatmap.png'), dpi=150)
        plt.close(); log.info('saved: kinship_heatmap.png')

    # ── plot 3: lod correlation ────────────────────────────────────────────────
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

    # ── plot 4: lod residuals by chromosome ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    diff   = comp_df['lod_py'] - comp_df['lod_r']
    ax.scatter(range(len(diff)), diff, c=chr_colors(comp_df['chr']),
               s=5, alpha=0.4, edgecolors='none', rasterized=True)
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
        ax.text(offset + n_c / 2, -0.02, f'chr{c}',
                ha='center', fontsize=7, color='#444', transform=ax.get_xaxis_transform())
        offset += n_c
    ax.set_xlim(0, len(diff))
    ax.set_xlabel('locus index'); ax.set_ylabel('$\\Delta$LOD (Python $-$ R)')
    ax.set_title('LOD residuals by chromosome')
    sns.despine(); plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lod_residuals.png'), dpi=150)
    plt.close(); log.info('saved: lod_residuals.png')

    # ── plot 5: p-value QQ + λ_GC comparison ─────────────────────────────────
    df1      = cache_py.n_founders - 1
    chi2_py  = (2 * comp_df['lod_py'] * np.log(10)).clip(lower=0)
    pvals_py = scipy_stats.chi2.sf(chi2_py, df=df1)
    r_merged = comp_df[['locus', 'chr', 'gene_id']].merge(
        lod_r[['locus', 'chr', 'gene_id', 'p_value']],
        on=['locus', 'chr', 'gene_id'], how='left')
    pvals_r = r_merged['p_value'].values
    med_chi2 = scipy_stats.chi2.ppf(0.5, df=df1)
    lam_py   = np.nanmedian((2 * comp_df['lod_py'] * np.log(10)).clip(0)) / med_chi2
    lam_r    = np.nanmedian((2 * comp_df['lod_r']  * np.log(10)).clip(0)) / med_chi2

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
    plt.suptitle(f'$\\lambda_{{GC}}$: Python={lam_py:.3f}  R={lam_r:.3f}',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'pvalue_qq.png'), dpi=150, bbox_inches='tight')
    plt.close(); log.info(f'saved: pvalue_qq.png  (lam_py={lam_py:.3f}  lam_r={lam_r:.3f})')

    # ── plot 6: founder effects ────────────────────────────────────────────────
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

    # ── plot 7: per-gene summary ───────────────────────────────────────────────
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

except Exception:
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
print(f'  haplotype probs: {"PASS ✓" if hap_match else "MISMATCH"}')
print(f'  kinship:         {"PASS ✓" if kinship_pass else "MISMATCH"}')
print(f'  LOD comparison:  {n_pass} PASS  '
      f'({n_strict} strict + {n_lenient} lenient)  /  {n_fail} FAIL')
if gene_stats:
    print(f'  median r:        {np.median([g["r"] for g in gene_stats]):.6f}')
    print(f'  median rel_err:  {np.median([g["rel_error"] for g in gene_stats]):.4f}')
print(f'  strict:   r > 0.999 and max|Δlod| < 0.01')
print(f'  lenient:  r > 0.990 and rel_error < 5% of peak LOD')
print(f'  outputs:  validation/r_data/gene_results.csv')
print('=' * 60)

sys.exit(0 if (hap_match and kinship_pass and all_pass) else 1)
