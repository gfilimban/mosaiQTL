# validate_vs_r.py
# full end-to-end cross-language validation: mosaiqtl (Python) vs miQTL (R)
#
# each section writes a csv report + plots, then a final qc_summary.csv
#
# sections:
#   1. haplotype probs  — from_gigamuga() (pure Python HMM) vs from_happy_dir()
#   2. kinship          — calc_kinship(loco) vs kinship_loco.rds
#   3. lod scores       — fit_lmm_at_locus() vs lod_r.csv
#
# csv outputs (validation/r_data/):
#   haplotype_report.csv  — per-locus max|Δprob| and row-sum check
#   kinship_report.csv    — per-chromosome max|ΔK| and r
#   gene_results.csv      — per-gene status, tier, lod stats (primary)
#   lod_comparison.csv    — per-locus lod_r vs lod_py
#   gene_summary.csv      — per-gene stats (full)
#   qc_summary.csv        — one-row-per-section pass/fail summary
#
# plot outputs (validation/qc_plots/):
#   haplotype_probs.png   kinship_heatmap.png
#   lod_correlation.png   lod_residuals.png
#   pvalue_qq.png         founder_effects.png   gene_summary.png
#
# usage:
#   python validation/validate_vs_r.py                  # all chrs, all genes
#   python validation/validate_vs_r.py --chr 8          # one chromosome
#   python validation/validate_vs_r.py --chr 8 --n_genes 20

import numpy as np
import pandas as pd
import os, sys, argparse, logging
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger('validate')

# ── paths ──────────────────────────────────────────────────────────────────────
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

# ── genotype inputs (fully Python-native, no R) ───────────────────────────────
GM_CSV     = os.path.join(miqtl_dir, 'input',         'GM_ccfo_mm39.csv')
GM_GMAP    = os.path.join(miqtl_dir, 'input', 'GM',   'gmap')   # dir of GM_gmap*.csv
GM_PMAP    = os.path.join(miqtl_dir, 'input', 'GM',   'pmap')   # dir of GM_pmap*.csv

# CC strain columns (33 strains)
CC_COLS = ['CC001','CC002','CC003','CC004','CC005','CC006','CC007','CC008',
           'CC009','CC010','CC011','CC012','CC013','CC015','CC017','CC018',
           'CC024','CC025','CC028','CC029','CC030','CC032','CC036','CC037',
           'CC038','CC041','CC043','CC057','CC060','CC062','CC065','CC071','CC074']

# founders in standard qtl2 order
FOUNDER_COLS = ['AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ']

# F1 crosses: (sample_id, sire_founder_idx, dam_B6J_idx=1)
# B6J is always the dam (index 1 in FOUNDER_COLS)
F1_CROSSES = [
    ('B6AF1J',     0, 1),   # AJ × B6J
    ('B6129S1F1J', 2, 1),   # 129S1J × B6J
    ('B6NODF1J',   3, 1),   # NODJ × B6J
    ('B6NZOF1J',   4, 1),   # NZOJ × B6J
    ('B6CASTF1J',  5, 1),   # CASTJ × B6J
    ('B6PWKF1J',   6, 1),   # PWKJ × B6J
    ('B6WSBF1J',   7, 1),   # WSBJ × B6J
]

paths = {
    'kinship_r': os.path.join(miqtl_dir, 'output', 'kinship_loco.rds'),
    'cache_r':   os.path.join(miqtl_dir, 'output', 'genomecache'),
    'pmap':      os.path.join(miqtl_dir, 'input',  'pmap.csv'),
    'lod_r':     os.path.join(pkg_dir, 'validation', 'r_data', 'lod_r.csv'),
    'pheno_48':  os.path.join(pkg_dir, 'validation', 'r_data', 'pheno_48_used.csv'),
    'ae_r':      os.path.join(pkg_dir, 'validation', 'r_data', 'allele_effects.csv'),
}
val_dir  = os.path.join(pkg_dir, 'validation', 'r_data')
plot_dir = os.path.join(pkg_dir, 'validation', 'qc_plots')
os.makedirs(val_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

FOUNDER_COLORS = {
    'AJ':'#F4C245','B6J':'#C0BFBF','129S1J':'#DA9CC1','NODJ':'#4F6EAF',
    'NZOJ':'#52A5DB','CASTJ':'#55AF5B','PWKJ':'#D83026','WSBJ':'#683C91',
}

# ── args ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--chr',     default=None, help='comma-separated chromosomes (default: all)')
p.add_argument('--n_genes', default=None, type=int, help='genes per chr cap (default: all)')
args = p.parse_args()

# ── check prerequisites ────────────────────────────────────────────────────────
for key, path in paths.items():
    if key in ('ae_r', 'cache_r'):
        continue   # optional
    if not os.path.exists(path):
        log.error(f'{key} not found: {path}')
        if key in ('lod_r', 'pheno_48'):
            log.error('run: Rscript validation/r_reference/run_validation.R')
        sys.exit(1)

for label, path in [('GM_CSV', GM_CSV), ('GM_GMAP', GM_GMAP), ('GM_PMAP', GM_PMAP)]:
    if not os.path.exists(path):
        log.error(f'{label} not found: {path}'); sys.exit(1)

# ── helpers ────────────────────────────────────────────────────────────────────
def save_csv(df, name):
    path = os.path.join(val_dir, name)
    df.to_csv(path, index=False)
    log.info(f'  saved: {name} ({len(df)} rows)')
    return path

def save_plot(fig, name):
    path = os.path.join(plot_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt; plt.close(fig)
    log.info(f'  saved: {name}')

def section(title):
    log.info('\n' + '='*60 + f'\n{title}\n' + '='*60)

# ── plotting setup (done once) ─────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'sans-serif',
                     'font.sans-serif':['Carlito','DejaVu Sans'],
                     'font.size':11})

qc_rows = []   # collects one dict per section for qc_summary.csv

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HAPLOTYPE PROBS (from_gigamuga vs from_happy_dir)
# ═══════════════════════════════════════════════════════════════════════════════
section('SECTION 1 — HAPLOTYPE PROBS')
# build cache entirely in Python from raw GigaMUGA — no R required
cache_py = GenomeCache.from_gigamuga(
    gigamuga_path = GM_CSV,
    gmap_dir      = GM_GMAP,
    pmap_dir      = GM_PMAP,
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)
log.info(f'Python cache: {len(cache_py.chromosomes)} chr | '
         f'{len(cache_py.sample_ids)} strains | {cache_py.n_founders} founders')

hap_rows = []
cache_r  = GenomeCache.from_happy_dir(paths['cache_r'], model='additive') \
           if os.path.isdir(paths['cache_r']) else None

py_strains = list(cache_py.sample_ids)
r_strains  = list(cache_r.sample_ids) if cache_r else []
shared_s   = [s for s in py_strains if s in r_strains]
py_idx_s   = [py_strains.index(s) for s in shared_s]
r_idx_s    = [r_strains.index(s)  for s in shared_s]

for chrom in cache_py.chromosomes:
    lids = cache_py._data[chrom]['locus_ids']
    mid  = len(lids) // 2
    h_py = cache_py.get_locus(chrom, mid)
    row_sum_ok = bool(np.allclose(h_py.sum(axis=1), 1.0, atol=1e-4))
    max_diff   = np.nan
    if cache_r and chrom in cache_r._data:
        r_lids = cache_r._data[chrom]['locus_ids']
        r_idx  = np.where(r_lids == lids[mid])[0]
        if len(r_idx):
            h_r      = cache_r.get_locus(chrom, int(r_idx[0]))
            max_diff = float(np.abs(h_py[py_idx_s] - h_r[r_idx_s]).max())
    hap_rows.append({'chr': chrom, 'locus': lids[mid],
                     'row_sum_ok': row_sum_ok, 'max_delta_prob': max_diff})

hap_df  = pd.DataFrame(hap_rows)
save_csv(hap_df, 'haplotype_report.csv')

hap_pass = bool(hap_df['row_sum_ok'].all() and
                (hap_df['max_delta_prob'].dropna() < 1e-4).all())
log.info(f'haplotype probs: {"PASS ✓" if hap_pass else "MISMATCH"}')
qc_rows.append({'section':'haplotype_probs', 'pass':hap_pass,
                'detail': f'max|Δprob|={hap_df["max_delta_prob"].max():.2e}'})

# plot: 3-panel heatmap for first chromosome
if cache_r:
    chrom0 = cache_py.chromosomes[0]
    lids0  = cache_py._data[chrom0]['locus_ids']
    mid0   = len(lids0) // 2
    h_py0  = cache_py.get_locus(chrom0, mid0)
    r_l0   = cache_r._data[chrom0]['locus_ids']
    ri0    = np.where(r_l0 == lids0[mid0])[0]
    if len(ri0):
        h_r0  = cache_r.get_locus(chrom0, int(ri0[0]))
        diff0 = h_py0[py_idx_s] - h_r0[r_idx_s]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, mat, title, cmap, vmin, vmax in [
            (axes[0], h_py0,  'Python (from_gigamuga)',    'YlOrRd', 0, 1),
            (axes[1], h_r0,   'R (from_happy_dir)',   'YlOrRd', 0, 1),
            (axes[2], diff0,  'Difference (Py − R)',  'RdBu_r', None, None),
        ]:
            sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                        xticklabels=cache_py.founders, yticklabels=False,
                        cbar_kws={'shrink':0.7})
            ax.set_title(f'{title}\nchr{chrom0} · {lids0[mid0]}')
        plt.tight_layout(); save_plot(fig, 'haplotype_probs.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOCO KINSHIP (Python vs R)
# ═══════════════════════════════════════════════════════════════════════════════
section('SECTION 2 — KINSHIP')

kinship_py = kin_module.calc_kinship(cache_py, method='loco')
log.info(f'Python LOCO kinship: {len(kinship_py)} chromosomes')

import rpy2.robjects as ro
ro.r(f'k_list <- readRDS("{paths["kinship_r"]}")')
chr_names_r  = list(ro.r('names(k_list)'))
kinship_r    = {c: np.array(ro.r(f'k_list[["{c}"]]'), dtype=float) for c in chr_names_r}
r_strains_k  = list(ro.r('rownames(k_list[[1]])'))
py_strains_k = list(cache_py.sample_ids)
shared_k     = [s for s in py_strains_k if s in r_strains_k]
py_idx_k     = np.array([py_strains_k.index(s) for s in shared_k])
r_idx_k      = np.array([r_strains_k.index(s)  for s in shared_k])

kin_rows = []
for chrom in [c for c in cache_py.chromosomes if c in kinship_r]:
    k_py  = kinship_py[chrom][np.ix_(py_idx_k, py_idx_k)]
    k_r   = kinship_r[chrom][np.ix_(r_idx_k, r_idx_k)]
    max_d = float(np.abs(k_py - k_r).max())
    corr  = float(np.corrcoef(k_py.ravel(), k_r.ravel())[0, 1])
    kin_rows.append({'chr': chrom, 'max_delta_K': max_d, 'r': corr,
                     'pass': max_d < 1e-4})
    log.info(f'  chr{chrom}: max|ΔK|={max_d:.2e}  r={corr:.8f}  '
             f'{"✓" if max_d < 1e-4 else "✗"}')

kin_df   = pd.DataFrame(kin_rows)
save_csv(kin_df, 'kinship_report.csv')
kin_pass = bool(kin_df['pass'].all())
log.info(f'kinship: {"PASS ✓" if kin_pass else "MISMATCH"}')
qc_rows.append({'section':'kinship', 'pass':kin_pass,
                'detail': f'max|ΔK|={kin_df["max_delta_K"].max():.2e}'})

# plot: 3-panel kinship heatmap for first chromosome
chrom_k = kin_df['chr'].iloc[0]
if chrom_k in kinship_py and chrom_k in kinship_r:
    k_py0 = kinship_py[chrom_k][np.ix_(py_idx_k, py_idx_k)]
    k_r0  = kinship_r[chrom_k][np.ix_(r_idx_k, r_idx_k)]
    dk    = k_py0 - k_r0
    ab    = max(abs(dk.min()), abs(dk.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], k_py0, f'Python LOCO\nchr{chrom_k}',      'YlOrRd', 0, 1),
        (axes[1], k_r0,  f'R LOCO\nchr{chrom_k}',           'YlOrRd', 0, 1),
        (axes[2], dk,    f'Difference\nmax|ΔK|={ab:.2e}',   'RdBu_r', -ab, ab),
    ]:
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={'shrink':0.7, 'label': title.split('\n')[0]})
        ax.set_title(title)
    plt.tight_layout(); save_plot(fig, 'kinship_heatmap.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
section('SECTION 3 — LOD COMPARISON')

lod_r_df = pd.read_csv(paths['lod_r']); lod_r_df['chr'] = lod_r_df['chr'].astype(str)
ae_r     = pd.read_csv(paths['ae_r']) if os.path.exists(paths['ae_r']) else None

pheno_raw = pd.read_csv(paths['pheno_48'], index_col=0)
if 'subject.name' not in pheno_raw.columns:
    pheno_raw.index.name = 'subject.name'; pheno_raw = pheno_raw.reset_index()
else:
    pheno_raw = pheno_raw.reset_index(drop=True)

all_chrs = sorted(lod_r_df['chr'].unique(), key=lambda c: int(c) if c.isdigit() else 99)
chromosomes = ([c.strip() for c in args.chr.split(',') if c.strip() in all_chrs]
               if args.chr else all_chrs)
log.info(f'chromosomes: {chromosomes}  |  '
         f'genes: {lod_r_df[lod_r_df["chr"].isin(chromosomes)]["gene_id"].nunique()}')

# align pheno to cache strain order + build covariate matrix once
cache_strains = list(cache_py.sample_ids)
pheno_48      = pheno_raw.set_index('subject.name').reindex(cache_strains)
pc_cols       = [f'pc{i}' for i in range(1, 11)]
batch_dummies = pd.get_dummies(pheno_48['batch'], prefix='batch', drop_first=True)
covar_base    = np.hstack([np.ones((len(pheno_48), 1)),
                           batch_dummies.astype(float).values,
                           pheno_48[pc_cols].astype(float).values])   # (48, p)
pheno_48 = pheno_48.reset_index()

all_pass    = True
comparisons = []
gene_stats  = []
py_betas    = {}

for chrom in chromosomes:
    if chrom not in cache_py.chromosomes:
        log.warning(f'chr{chrom} not in cache — skipping'); continue

    chr_genes = lod_r_df[lod_r_df['chr'] == chrom]['gene_id'].unique().tolist()
    if args.n_genes:
        chr_genes = chr_genes[:args.n_genes]
    log.info(f'\n[chr{chrom}] {len(chr_genes)} genes')

    k_chr            = kinship_py[chrom]
    eigvals, eigvecs = np.linalg.eigh(k_chr)
    covar_rot        = eigvecs.T @ covar_base

    locus_ids = cache_py._data[chrom]['locus_ids']

    for gene in chr_genes:
        if gene not in pheno_48.columns: continue
        y = pheno_48.set_index('subject.name').reindex(cache_strains)[gene].values.astype(float)
        if np.all(np.isnan(y)): continue

        y_rot      = eigvecs.T @ y
        h2, _      = lmm_module.estimate_h2_null(y_rot, covar_rot, eigvals)

        r_gene = (lod_r_df[(lod_r_df['gene_id'] == gene) & (lod_r_df['chr'] == chrom)]
                  [['locus', 'lod', 'pos_mb']].rename(columns={'lod': 'lod_r'}))
        if r_gene.empty: continue

        lod_py, betas_py = [], []
        for locus in r_gene['locus']:
            idx = np.where(locus_ids == locus)[0]
            if not len(idx):
                lod_py.append(np.nan); betas_py.append(None); continue
            h_l    = cache_py.get_locus(chrom, int(idx[0]))
            h_rot  = eigvecs.T @ h_l[:, :-1]
            res    = lmm_module.fit_lmm_at_locus(y_rot, h_rot, covar_rot, eigvals, h2)
            lod_py.append(res.lod); betas_py.append(res.betas)

        r_gene = r_gene.copy(); r_gene['lod_py'] = lod_py
        merged = r_gene.dropna(subset=['lod_r', 'lod_py'])
        if len(merged) < 3:
            log.warning(f'  {gene}: {len(merged)} loci — skipping'); all_pass = False; continue

        r_corr     = float(np.corrcoef(merged['lod_r'], merged['lod_py'])[0, 1])
        max_d      = float((merged['lod_r'] - merged['lod_py']).abs().max())
        peak_lod_r = float(merged['lod_r'].max())
        peak_lod_py= float(merged['lod_py'].max())
        peak_row   = merged.loc[merged['lod_r'].idxmax()]
        lead_locus = peak_row['locus']
        delta_peak = float(abs(peak_row['lod_r'] - peak_row['lod_py']))
        rel_err    = max_d / max(peak_lod_r, 0.01)

        strict  = r_corr > 0.999 and max_d < 0.01
        lenient = r_corr > 0.990 and rel_err < 0.05
        passed  = strict or lenient
        tier    = 'strict' if strict else ('lenient' if lenient else 'fail')

        log.info(f'  {"PASS" if passed else "FAIL"}({tier:<7})  {gene:25s}  '
                 f'r={r_corr:.6f}  max|Δlod|={max_d:.4f}  rel={rel_err:.3f}  '
                 f'n={len(merged)}')
        if not passed: all_pass = False

        merged['gene_id'] = gene; merged['chr'] = chrom
        comparisons.append(merged)
        gene_stats.append({'gene_id':gene,'chr':chrom,'status':'PASS' if passed else 'FAIL',
                           'tier':tier,'r':r_corr,'max_delta_lod':max_d,'rel_error':rel_err,
                           'peak_lod_r':peak_lod_r,'peak_lod_py':peak_lod_py,
                           'delta_at_peak':delta_peak,'lead_locus':lead_locus,
                           'n_loci':len(merged),'h2_null':h2,'passed':passed})

        ll = r_gene['locus'].tolist()
        if lead_locus in ll and betas_py[ll.index(lead_locus)] is not None:
            py_betas[gene] = (chrom, lead_locus,
                              cache_py.founders[:-1], betas_py[ll.index(lead_locus)])

if not comparisons:
    log.error('no comparison data'); sys.exit(1)

comp_df = pd.concat(comparisons, ignore_index=True)
stat_df = pd.DataFrame(gene_stats)
save_csv(comp_df, 'lod_comparison.csv')
save_csv(stat_df, 'gene_summary.csv')
save_csv(stat_df[['gene_id','chr','status','tier','peak_lod_r','peak_lod_py',
                  'delta_at_peak','max_delta_lod','rel_error','r',
                  'lead_locus','n_loci','h2_null']]
         .sort_values(['status','rel_error'], ascending=[True,False]),
         'gene_results.csv')

n_strict = (stat_df['tier']=='strict').sum()
n_lenient= (stat_df['tier']=='lenient').sum()
n_fail   = (stat_df['tier']=='fail').sum()
lod_pass = all_pass
qc_rows.append({'section':'lod_scores','pass':lod_pass,
                'detail': f'{n_strict} strict + {n_lenient} lenient / {n_fail} fail | '
                          f'median_r={stat_df["r"].median():.4f}'})

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QC PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
section('SECTION 4 — QC PLOTS')

chr_order   = sorted(comp_df['chr'].unique(), key=lambda c: int(c) if c.isdigit() else 99)
chr_pal     = ['#4F6EAF', '#C0BFBF']
chr_colors  = lambda s: [chr_pal[chr_order.index(c) % 2] for c in s]

# lod correlation
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(comp_df['lod_r'], comp_df['lod_py'], c=chr_colors(comp_df['chr']),
           s=6, alpha=0.4, edgecolors='none', rasterized=True)
lim = max(comp_df['lod_r'].max(), comp_df['lod_py'].max()) * 1.05
ax.plot([0, lim], [0, lim], '--', color='#D83026', lw=1, label='$y=x$')
dif = comp_df['lod_r'] - comp_df['lod_py']
r_all = float(np.corrcoef(comp_df['lod_r'], comp_df['lod_py'])[0, 1])
ax.text(0.05, 0.95, f'$r$={r_all:.6f}\nmean Δ={dif.mean():.4f}\nmax|Δ|={dif.abs().max():.4f}',
        transform=ax.transAxes, fontsize=9, va='top')
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='miQTL LOD (R)', ylabel='mosaiQTL LOD (Python)',
       title='LOD comparison')
ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
save_plot(fig, 'lod_correlation.png')

# lod residuals
fig, ax = plt.subplots(figsize=(12, 3))
diff = comp_df['lod_py'] - comp_df['lod_r']
ax.scatter(range(len(diff)), diff, c=chr_colors(comp_df['chr']),
           s=5, alpha=0.4, edgecolors='none', rasterized=True)
ax.axhline(0, color='#D83026', lw=0.8, ls='--')
ax.text(0.99, 0.97, f'mean={diff.mean():.4f}  sd={diff.std():.4f}  max|Δ|={diff.abs().max():.4f}',
        transform=ax.transAxes, fontsize=9, va='top', ha='right')
offset = 0
for c in chr_order:
    n_c = (comp_df['chr'] == c).sum()
    if offset: ax.axvline(offset, color='white', lw=0.5)
    ax.text(offset + n_c/2, 0, f'chr{c}', ha='center', fontsize=7, color='#444',
            transform=ax.get_xaxis_transform())
    offset += n_c
ax.set(xlim=(0,len(diff)), xlabel='locus index', ylabel='ΔLOD (Python − R)',
       title='LOD residuals by chromosome')
sns.despine(); plt.tight_layout(); save_plot(fig, 'lod_residuals.png')

# p-value QQ
df1 = cache_py.n_founders - 1
chi2_py  = (2 * comp_df['lod_py'] * np.log(10)).clip(lower=0)
pvals_py = sp_stats.chi2.sf(chi2_py, df=df1)
pvals_r  = (comp_df[['locus','chr','gene_id']]
            .merge(lod_r_df[['locus','chr','gene_id','p_value']],
                   on=['locus','chr','gene_id'], how='left')['p_value'].values)
med_chi2 = sp_stats.chi2.ppf(0.5, df=df1)
lam_py   = np.nanmedian(chi2_py) / med_chi2
lam_r    = np.nanmedian((2 * comp_df['lod_r'] * np.log(10)).clip(0)) / med_chi2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, pv, lam, title, color in [
    (axes[0], pvals_py, lam_py, 'mosaiQTL (Python)', '#4F6EAF'),
    (axes[1], pvals_r,  lam_r,  'miQTL (R)',         '#55AF5B'),
]:
    valid = pv[np.isfinite(pv) & (pv > 0)]
    obs = -np.log10(np.sort(valid))
    exp = -np.log10(np.linspace(1/len(obs), 1, len(obs))[::-1])
    m   = max(exp.max(), obs.max()) * 1.05
    ax.scatter(exp, obs, s=5, alpha=0.4, color=color, edgecolors='none', rasterized=True)
    ax.plot([0, m], [0, m], '--', color='#D83026', lw=1)
    ax.text(0.05, 0.95, f'λ_GC = {lam:.3f}', transform=ax.transAxes, fontsize=10, va='top')
    ax.set(xlim=(0, m), ylim=(0, m), title=f'QQ — {title}',
           xlabel='expected −log₁₀(p)', ylabel='observed −log₁₀(p)')
    sns.despine(ax=ax)
plt.suptitle(f'λ_GC: Python={lam_py:.3f}  R={lam_r:.3f}', fontsize=11, y=1.01)
plt.tight_layout(); save_plot(fig, 'pvalue_qq.png')

# founder effects at peak gene
if ae_r is not None and py_betas:
    best  = lod_r_df[lod_r_df['chr'].isin(chromosomes)].groupby('gene_id')['lod'].max().idxmax()
    if best in py_betas:
        chrom_b, peak_loc, f_names, py_eff = py_betas[best]
        r_ae = ae_r[(ae_r['gene_id'] == best) & (ae_r['locus'] == peak_loc)]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(f_names)); w = 0.35
        cf = [FOUNDER_COLORS.get(f, '#888') for f in f_names]
        ax.bar(x - w/2, py_eff, w, color=cf, edgecolor='white', lw=0.5, label='Python')
        if not r_ae.empty:
            r_eff = [float(r_ae.loc[r_ae['founder']==f,'effect'].iloc[0])
                     if f in r_ae['founder'].values else 0.0 for f in f_names]
            ax.bar(x + w/2, r_eff, w, color=cf, edgecolor='#333', lw=0.8,
                   alpha=0.55, label='R')
            ax.text(0.02, 0.97, f'r={np.corrcoef(py_eff,r_eff)[0,1]:.4f}',
                    transform=ax.transAxes, fontsize=10, va='top')
        ax.axhline(0, color='#444', lw=0.5)
        ax.set(xticks=x, title=f'founder effects — {best}  chr{chrom_b}  {peak_loc}',
               ylabel='β̂_f')
        ax.set_xticklabels(f_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
        save_plot(fig, 'founder_effects.png')

# gene summary
tier_colors = {'strict':'#4F6EAF','lenient':'#55AF5B','fail':'#D83026'}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
for tv, grp in stat_df.groupby('tier'):
    sns.histplot(grp['r'], bins=30, color=tier_colors.get(tv,'#888'),
                 ax=ax, edgecolor='white', label=tv, alpha=0.75)
ax.axvline(0.999, color='#D83026', lw=1, ls='--')
ax.axvline(0.990, color='#55AF5B', lw=1, ls=':')
ax.set(xlim=(min(stat_df['r'].min()-0.05,-0.1),1.01), ylim=(0,None),
       xlabel='Pearson r (Python vs R LOD)', ylabel='genes',
       title=f'per-gene r\nstrict={n_strict} lenient={n_lenient} fail={n_fail}')
ax.legend(fontsize=9); sns.despine(ax=ax)

ax = axes[1]
sns.histplot(stat_df['rel_error'], bins=40, color='#683C91', ax=ax, edgecolor='white')
ax.axvline(0.05, color='#D83026', lw=1.2, ls='--', label='5% threshold')
ax.set(xlim=(0,None), ylim=(0,None),
       xlabel='relative error: max|Δlod| / peak lod (R)', ylabel='genes',
       title=f'relative LOD error (n={len(stat_df)})')
ax.legend(fontsize=9); sns.despine(ax=ax)
plt.tight_layout(); save_plot(fig, 'gene_summary.png')

# ═══════════════════════════════════════════════════════════════════════════════
# QC SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
qc_df = pd.DataFrame(qc_rows)
save_csv(qc_df, 'qc_summary.csv')

print()
print('=' * 60)
for row in qc_rows:
    print(f'  {"PASS" if row["pass"] else "FAIL"}  {row["section"]:<20}  {row["detail"]}')
print(f'  pass threshold: strict (r>0.999, max|Δ|<0.01) or lenient (r>0.990, rel<5%)')
print(f'  outputs: {val_dir}/')
print('=' * 60)

sys.exit(0 if all(r['pass'] for r in qc_rows) else 1)
