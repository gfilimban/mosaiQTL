# validate_port.py — complete mosaiQTL validation against miQTL R reference
#
# for every section: runs the Python implementation, reads the R reference
# CSV from port_validation/r_data/, compares, and saves QC plots.
#
# sections mirror port_validation.R exactly:
#   s1.  cache info
#   s2.  haplotype probs
#   s3.  kinship loco
#   s4.  lmm scan (rop)
#   s5.  cache collapse
#   s6.  permutation thresholds
#   s7.  multiple imputation
#   s8.  snp scan
#   s9.  mediation
#   s10. positional ci
#   s11. instability scan
#
# usage:
#   python validation/validate_port.py
#   python validation/validate_port.py --chr 8 --n_genes 50
#   python validation/validate_port.py --skip-phase1

import numpy as np
import pandas as pd
import os, sys, argparse, logging, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module
from mosaiqtl.utils import rint as _rint, build_z_matrix
from scipy import stats as sp_stats

# ── args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--chr',         default=None)
ap.add_argument('--n_genes',     default=None, type=int)
ap.add_argument('--skip-phase1', action='store_true')
args = ap.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

R_DATA  = os.path.join(pkg_dir, 'validation', 'port_validation', 'r_data')
PY_DATA = os.path.join(pkg_dir, 'validation', 'port_validation', 'py_data')
PLOTS   = os.path.join(pkg_dir, 'validation', 'port_validation', 'qc_plots')
LOGS    = os.path.join(pkg_dir, 'validation', 'port_validation', 'logs')
for d in (R_DATA, PY_DATA, PLOTS, LOGS): os.makedirs(d, exist_ok=True)

# GigaMUGA inputs
GM_CSV  = os.path.join(miqtl_dir, 'input', 'GM_ccfo_mm39.csv')
GM_GMAP = os.path.join(miqtl_dir, 'input', 'GM', 'gmap')
GM_PMAP = os.path.join(miqtl_dir, 'input', 'GM', 'pmap')
SNP_CSV = os.path.join(miqtl_dir, 'input', 'snp_genos.csv')

CC_COLS      = ['CC001','CC002','CC003','CC004','CC005','CC006','CC007','CC008',
                'CC009','CC010','CC011','CC012','CC013','CC015','CC017','CC018',
                'CC024','CC025','CC028','CC029','CC030','CC032','CC036','CC037',
                'CC038','CC041','CC043','CC057','CC060','CC062','CC065','CC071','CC074']
FOUNDER_COLS = ['AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ']
F1_CROSSES   = [('B6AF1J',0,1),('B6129S1F1J',2,1),('B6NODF1J',3,1),
                ('B6NZOF1J',4,1),('B6CASTF1J',5,1),('B6PWKF1J',6,1),
                ('B6WSBF1J',7,1)]

FOUNDER_COLORS = {
    'AJ':'#F4C245','B6J':'#C0BFBF','129S1J':'#DA9CC1','NODJ':'#4F6EAF',
    'NZOJ':'#52A5DB','CASTJ':'#55AF5B','PWKJ':'#D83026','WSBJ':'#683C91'}
CHR_PAL = ['#4F6EAF','#888780']

# ── logging ───────────────────────────────────────────────────────────────────
_ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_log = os.path.join(LOGS, f'validate_{_ts}.log')
log  = logging.getLogger('mosaiqtl_val')
log.setLevel(logging.DEBUG)
_fh  = logging.FileHandler(_log)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
log.addHandler(_fh)
_sh  = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.WARNING)
_sh.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(_sh)

def hi(msg): print(msg, flush=True)

# ── plotting ──────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'sans-serif',
                     'font.sans-serif':['Carlito','DejaVu Sans'],
                     'font.size':11})

_chr_sort = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)

# ── helpers ───────────────────────────────────────────────────────────────────
qc_rows = []

def save_csv(df, name):
    p = os.path.join(PY_DATA, name)
    df.to_csv(p, index=False)
    log.debug(f'saved {name} ({len(df)} rows)')
    return p

def save_plot(fig, name):
    p = os.path.join(PLOTS, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.debug(f'saved {name}')

def load_r(name):
    p = os.path.join(R_DATA, name)
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    if 'chr' in df.columns: df['chr'] = df['chr'].astype(str)
    return df

def section(n, title):
    hi(f'\n▶ S{n} — {title}')
    log.info(f'=== S{n} {title} ===')

def skip(key, reason):
    hi(f'  SKIP: {reason}')
    log.warning(f'skip {key}: {reason}')
    qc_rows.append({'section':key,'pass':None,'detail':f'skipped: {reason}'})

def record(key, passed, detail):
    st = 'PASS' if passed else 'FAIL'
    hi(f'  {st}: {detail}')
    log.info(f'{key}: {st} — {detail}')
    qc_rows.append({'section':key,'pass':passed,'detail':detail})

def scatter_qc(ax, x, y, xlabel, ylabel, title):
    """standard scatter with y=x line and r annotation."""
    lim = max(float(np.nanmax(np.abs(np.concatenate([x,y])))), 0.1) * 1.05
    ax.scatter(x, y, s=3, alpha=1.0, color='#4F6EAF', edgecolors='none')
    ax.plot([0,lim],[0,lim],'--',color='#D83026',lw=1)
    r = float(np.corrcoef(x,y)[0,1]) if len(x)>3 else np.nan
    ax.text(0.05,0.95,f'$r$={r:.4f}',transform=ax.transAxes,fontsize=9,va='top')
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel=xlabel,ylabel=ylabel,title=title)
    sns.despine(ax=ax)
    return r

# ── build Python cache ────────────────────────────────────────────────────────
hi('\n● building Python genome cache...')
cache_py = GenomeCache.from_gigamuga(
    gigamuga_path=GM_CSV, gmap_dir=GM_GMAP, pmap_dir=GM_PMAP,
    cc_cols=CC_COLS, founder_cols=FOUNDER_COLS, f1_crosses=F1_CROSSES)
chr_order  = sorted(cache_py.chromosomes, key=_chr_sort)
chr_colors = lambda s: [CHR_PAL[chr_order.index(c)%2]
                        if c in chr_order else '#aaa' for c in s]
chromosomes = ([c.strip() for c in args.chr.split(',')]
               if args.chr else chr_order)
hi(f'  cache: {len(cache_py.chromosomes)} chr | {len(cache_py.sample_ids)} strains')

# build LOCO kinship
kinship_py = kin_module.calc_kinship(cache_py, method='loco')
hi(f'  kinship: {len(kinship_py)} LOCO matrices')

# load phenotype + covariates
_pheno_csv = os.path.join(R_DATA, 'pheno_48_used.csv')
pheno_strain = None; covar_base = None
if os.path.exists(_pheno_csv):
    _raw = pd.read_csv(_pheno_csv, index_col=0)
    if 'subject.name' not in _raw.columns:
        _raw = _raw.reset_index().rename(columns={'index':'subject.name'})
    _raw = _raw.drop_duplicates('subject.name').reset_index(drop=True)
    _cs  = list(cache_py.sample_ids)
    _ps  = _raw.set_index('subject.name').reindex(_cs)
    _cp  = [np.ones((len(_ps),1))]
    if 'batch' in _ps.columns and _ps['batch'].notna().any():
        _cp.append(pd.get_dummies(_ps['batch'],prefix='batch',
                                   drop_first=True).astype(float).values)
    _pcs = [c for c in [f'pc{i}' for i in range(1,11)] if c in _ps.columns]
    if _pcs: _cp.append(_ps[_pcs].fillna(0).astype(float).values)
    pheno_strain = _ps.reset_index()
    covar_base   = np.hstack(_cp)
    hi(f'  pheno: {len(pheno_strain)} strains  covar: {covar_base.shape[1]} cols')
else:
    hi(f'  WARN: pheno_48_used.csv not found — run port_validation.R s4 first')

def _get_y(gene):
    if pheno_strain is None or gene not in pheno_strain.columns: return None
    return (pheno_strain.drop_duplicates('subject.name')
            .set_index('subject.name').reindex(list(cache_py.sample_ids))
            [gene].values.astype(float))

# ══════════════════════════════════════════════════════════════════════════════
# S1 — CACHE INFO
# ══════════════════════════════════════════════════════════════════════════════
section(1, 'CACHE INFO')
r_info = load_r('cache_info_r.csv')
r_fnd  = load_r('founders_r.csv')
r_sid  = load_r('sample_ids_r.csv')

rows_info = []
for chrom in chromosomes:
    if chrom not in cache_py._data: continue
    lids, pos, _ = cache_py.get_chromosome(chrom)
    n_py = len(lids)
    n_r  = (int(r_info.loc[r_info['chr']==chrom,'n_loci'].iloc[0])
            if r_info is not None and (r_info['chr']==chrom).any() else np.nan)
    rows_info.append({'chr':chrom,'n_loci_py':n_py,'n_loci_r':n_r,
                      'min_mb_py':float(pos.min()),'max_mb_py':float(pos.max())})
    log.info(f'  chr{chrom}: py={n_py}  r={n_r}')
info_df = pd.DataFrame(rows_info)
save_csv(info_df,'cache_info_py.csv')

py_founders = list(cache_py.founders)
py_sids     = list(cache_py.sample_ids)
r_founders  = r_fnd['founder'].tolist() if r_fnd is not None else []
r_sids      = r_sid['sample_id'].tolist() if r_sid is not None else []

fnd_match   = set(py_founders) == set(r_founders) if r_founders else None
sid_match   = set(py_sids) == set(r_sids)         if r_sids    else None
loci_match  = bool(info_df['n_loci_py'].equals(info_df['n_loci_r'].fillna(-1).astype(int))) if r_info is not None else None

if r_info is not None:
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    ax = axes[0]
    x = np.arange(len(info_df)); w = 0.35
    ax.bar(x-w/2, info_df['n_loci_py'], w, color='#4F6EAF', edgecolor='none',
           label='mosaiQTL (Python)')
    ax.bar(x+w/2, info_df['n_loci_r'].fillna(0), w, color='#55AF5B',
           edgecolor='none', alpha=0.85, label='miQTL (R)')
    ax.set(xticks=x, xticklabels=[f'chr{c}' for c in info_df['chr']],
           xlabel='chromosome', ylabel='loci', title='loci per chromosome')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); sns.despine(ax=ax)
    ax = axes[1]
    ax.axis('off')
    tbl = [['', 'Python', 'R'],
           ['n_founders', str(len(py_founders)), str(len(r_founders))],
           ['n_samples',  str(len(py_sids)),     str(len(r_sids))],
           ['founders match', '—', str(fnd_match)],
           ['samples match',  '—', str(sid_match)]]
    t = ax.table(tbl, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(10)
    ax.set_title('cache metadata', fontsize=11)
    plt.tight_layout(); save_plot(fig,'s1_cache_info.png')

record('cache_info', fnd_match is not False and sid_match is not False,
       f'founders_match={fnd_match}  samples_match={sid_match}  '
       f'loci_match={loci_match}')

# ══════════════════════════════════════════════════════════════════════════════
# S2 — HAPLOTYPE PROBS
# ══════════════════════════════════════════════════════════════════════════════
section(2, 'HAPLOTYPE PROBS')
r_hap = load_r('haplotype_r.csv')
rows_hap = []
for chrom in chromosomes:
    if chrom not in cache_py._data: continue
    lids, pos, probs = cache_py.get_chromosome(chrom)
    n_loci = len(lids)
    idx    = np.round(np.linspace(0, n_loci-1, 5)).astype(int)
    for i in idx:
        rs = probs[i].sum(axis=1)
        rows_hap.append({'chr':chrom,'locus':lids[i],'pos_mb':float(pos[i]),
                         'mean_rowsum':float(rs.mean()),'min_rowsum':float(rs.min()),
                         'max_rowsum':float(rs.max()),
                         'mean_max_prob':float(probs[i].max(axis=1).mean())})
hap_df   = pd.DataFrame(rows_hap)
save_csv(hap_df, 'haplotype_py.csv')
rsum_ok  = bool(np.allclose(hap_df['mean_rowsum'], 1.0, atol=1e-4))
max_dev  = float((hap_df['mean_rowsum']-1.0).abs().max())

if r_hap is not None:
    mg = hap_df.merge(r_hap[['locus','mean_rowsum']].rename(
        columns={'mean_rowsum':'rowsum_r'}), on='locus', how='inner')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    ax = axes[0]
    mg_v = mg.dropna(subset=['rowsum_r','mean_rowsum'])
    if len(mg_v) > 1:
        scatter_qc(ax, mg_v['rowsum_r'], mg_v['mean_rowsum'],
                   'row sum (R)', 'row sum (Python)', 'haplotype row sums')
    else:
        ax.text(0.5,0.5,'insufficient overlap', ha='center', transform=ax.transAxes)
    ax = axes[1]
    ax.bar(range(len(hap_df)), hap_df['mean_rowsum']-1.0,
           color=chr_colors(hap_df['chr'].tolist()), edgecolor='none')
    ax.axhline(0, color='#D83026', lw=0.8, ls='--')
    ax.set(xlabel='locus index', ylabel='mean row sum − 1',
           title=f'row sum deviation  max={max_dev:.2e}')
    sns.despine(ax=ax)
    plt.tight_layout(); save_plot(fig, 's2_haplotype_probs.png')

record('haplotype_probs', rsum_ok,
       f'row_sums_ok={rsum_ok}  max_dev={max_dev:.2e}')

# ══════════════════════════════════════════════════════════════════════════════
# S3 — KINSHIP LOCO
# ══════════════════════════════════════════════════════════════════════════════
section(3, 'KINSHIP LOCO')
r_kin = load_r('kinship_r.csv')
rows_kin = []
for chrom in chromosomes:
    if chrom not in kinship_py: continue
    k = kinship_py[chrom]
    rows_kin.append({'chr':chrom,'n':k.shape[0],
                     'mean_diag':float(np.diag(k).mean()),
                     'mean_offdiag':float(k[np.tril_indices(k.shape[0],-1)].mean()),
                     'min_val':float(k.min()),'max_val':float(k.max())})
kin_sum_df = pd.DataFrame(rows_kin)
save_csv(kin_sum_df, 'kinship_py.csv')

kin_cors = []
for chrom in chromosomes:
    r_mat_path = os.path.join(R_DATA, f'kinship_mat_r_chr{chrom}.csv')
    if not os.path.exists(r_mat_path) or chrom not in kinship_py: continue
    k_r_df = pd.read_csv(r_mat_path)
    sids_r = k_r_df['sample_id'].tolist()
    k_r    = k_r_df.drop(columns='sample_id').values.astype(float)
    sids_py = list(cache_py.sample_ids)
    shared  = [s for s in sids_py if s in sids_r]
    if len(shared) < 2: continue
    pi = np.array([sids_py.index(s) for s in shared])
    ri = np.array([sids_r.index(s)  for s in shared])
    kp = kinship_py[chrom][np.ix_(pi,pi)]
    kr = k_r[np.ix_(ri,ri)]
    r  = float(np.corrcoef(kp.ravel(), kr.ravel())[0,1])
    md = float(np.abs(kp-kr).max())
    ok = r > 0.999 and md < 0.05
    kin_cors.append({'chr':chrom,'r':r,'max_delta':md,'pass':ok})
    log.info(f'  kinship chr{chrom}: r={r:.6f}  max|ΔK|={md:.2e}  {"✓" if ok else "✗"}')

if kin_cors:
    kc_df = pd.DataFrame(kin_cors)
    save_csv(kc_df, 'kinship_comparison.csv')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    ax = axes[0]
    ax.scatter(range(len(kc_df)), kc_df['r'], s=20, color='#4F6EAF', edgecolors='none')
    ax.axhline(0.999, color='#D83026', lw=1, ls='--', label='r=0.999')
    ax.set(xlabel='chromosome index', ylabel='kinship Pearson r',
           title=f'kinship r  mean={kc_df["r"].mean():.4f}', ylim=(0.98,1.001))
    ax.legend(fontsize=9); sns.despine(ax=ax)
    ax = axes[1]
    ax.scatter(range(len(kc_df)), kc_df['max_delta'], s=20,
               color='#D83026', edgecolors='none')
    ax.set(xlabel='chromosome index', ylabel='max|ΔK|',
           title=f'kinship max deviation  mean={kc_df["max_delta"].mean():.2e}')
    sns.despine(ax=ax)
    plt.tight_layout(); save_plot(fig,'s3_kinship.png')
    kin_pass = bool(kc_df['pass'].all())
    record('kinship', kin_pass,
           f'all_pass={kin_pass}  '
           f'mean_r={kc_df["r"].mean():.4f}  '
           f'max|ΔK|={kc_df["max_delta"].max():.2e}')
else:
    skip('kinship','no per-chr kinship matrices — run port_validation.R s3')

# ══════════════════════════════════════════════════════════════════════════════
# S4 — LMM SCAN
# ══════════════════════════════════════════════════════════════════════════════
section(4, 'LMM SCAN (ROP)')
r_lod = load_r('lod_r.csv')
r_ae  = load_r('allele_effects_r.csv')
if r_lod is None or pheno_strain is None:
    skip('lmm_scan', 'lod_r.csv or pheno missing — run port_validation.R s4')
else:
    comparisons=[]; gene_stats=[]; py_betas={}
    _cs = list(cache_py.sample_ids)

    for chrom in chromosomes:
        if chrom not in cache_py.chromosomes: continue
        genes = r_lod[r_lod['chr']==chrom]['gene_id'].unique().tolist()
        if args.n_genes: genes = genes[:args.n_genes]
        if not genes: continue
        hi(f'  chr{chrom}: {len(genes)} genes')
        k_chr = kinship_py[chrom]
        ev, evc = np.linalg.eigh(k_chr)
        x_rot   = evc.T @ covar_base
        lids    = cache_py._data[chrom]['locus_ids']

        for gene in genes:
            y = _get_y(gene)
            if y is None or np.all(np.isnan(y)): continue
            y_rint = _rint(y)
            y_rot  = evc.T @ y_rint
            h2, _  = lmm_module.estimate_h2_null(y_rot, x_rot, ev)
            rg = r_lod[(r_lod['gene_id']==gene)&(r_lod['chr']==chrom)
                       ][['locus','lod','pos_mb']].rename(columns={'lod':'lod_r'})
            if rg.empty: continue
            lod_py=[]; betas_py=[]
            for locus in rg['locus']:
                idx = np.where(lids==locus)[0]
                if not len(idx): lod_py.append(np.nan); betas_py.append(None); continue
                h_l   = cache_py.get_locus(chrom, int(idx[0]))
                h_rot = evc.T @ h_l[:,:-1]
                res   = lmm_module.fit_lmm_at_locus(y_rot, h_rot, x_rot, ev, h2)
                lod_py.append(res.lod); betas_py.append(res.betas)
            rg = rg.copy(); rg['lod_py'] = lod_py
            mg = rg.dropna(subset=['lod_r','lod_py'])
            if len(mg) < 3: continue
            pk_r   = float(mg['lod_r'].max())
            pk_py  = float(mg['lod_py'].max())
            py_peak= mg.loc[mg['lod_py'].idxmax(),'locus']
            r_at_peak = (float(mg.loc[mg['locus']==py_peak,'lod_r'].iloc[0])
                         if (mg['locus']==py_peak).any() else 0.0)
            pm     = r_at_peak >= 0.99*pk_r
            rel    = float((mg['lod_py']-mg['lod_r']).abs().max()) / max(pk_r,0.01)
            rc     = float(np.corrcoef(mg['lod_r'],mg['lod_py'])[0,1])
            tier   = 'strict' if pm and rel<0.10 else ('lenient' if pm and rel<0.20 else 'fail')
            mg = mg.copy(); mg['gene_id']=gene; mg['chr']=chrom
            comparisons.append(mg)
            gene_stats.append({'gene_id':gene,'chr':chrom,'tier':tier,
                                'r':rc,'rel_error':rel,'peak_match':pm,
                                'peak_lod_r':pk_r,'peak_lod_py':pk_py,'h2_null':h2})
            ll = rg['locus'].tolist()
            if py_peak in ll and betas_py[ll.index(py_peak)] is not None:
                py_betas[gene] = (chrom,py_peak,cache_py.founders[:-1],
                                  betas_py[ll.index(py_peak)])

    if comparisons:
        comp_df = pd.concat(comparisons, ignore_index=True)
        stat_df = pd.DataFrame(gene_stats)
        save_csv(comp_df, 'lod_comparison.csv')
        save_csv(stat_df, 'gene_summary.csv')
        n_s = (stat_df['tier']=='strict').sum()
        n_l = (stat_df['tier']=='lenient').sum()
        n_f = (stat_df['tier']=='fail').sum()

        fig, axes = plt.subplots(1,3,figsize=(16,4))
        # scatter
        scatter_qc(axes[0], comp_df['lod_r'], comp_df['lod_py'],
                   'LOD (miQTL R)', 'LOD (mosaiQTL Python)', 'LOD comparison')
        # residuals
        diff = comp_df['lod_py']-comp_df['lod_r']
        axes[1].scatter(range(len(diff)), diff,
                        c=chr_colors(comp_df['chr'].tolist()),
                        s=2, alpha=1.0, edgecolors='none')
        axes[1].axhline(0,color='#D83026',lw=0.8,ls='--')
        axes[1].text(0.99,0.97,f'mean={diff.mean():.3f}  sd={diff.std():.3f}',
                     transform=axes[1].transAxes,fontsize=8,va='top',ha='right')
        axes[1].set(xlabel='locus index',ylabel='ΔLOD (py−R)',title='LOD residuals')
        sns.despine(ax=axes[1])
        # tier summary
        tc = {'fail':'#D83026','lenient':'#55AF5B','strict':'#4F6EAF'}
        axes[2].bar(['fail','lenient','strict'],[n_f,n_l,n_s],
                    color=[tc['fail'],tc['lenient'],tc['strict']],edgecolor='none')
        axes[2].set(xlabel='tier',ylabel='genes',
                    title=f'gene tiers  strict={n_s}  lenient={n_l}  fail={n_f}')
        sns.despine(ax=axes[2])
        plt.tight_layout(); save_plot(fig,'s4_lmm_scan.png')

        # founder effects
        if r_ae is not None and py_betas:
            best = next((g for g in
                         stat_df[stat_df['gene_id'].isin(r_ae['gene_id'].unique())]
                         .sort_values('peak_lod_r',ascending=False)['gene_id']
                         if g in py_betas), None)
            if best:
                chrom_b,peak_loc,f_names,py_eff = py_betas[best]
                r_sub = r_ae[(r_ae['gene_id']==best)&(r_ae['locus']==peak_loc)]
                fig,ax = plt.subplots(figsize=(9,4))
                x = np.arange(len(f_names)); w=0.35
                cf = [FOUNDER_COLORS.get(f,'#888') for f in f_names]
                ax.bar(x-w/2,py_eff,w,color=cf,edgecolor='white',lw=0.5,
                       label='mosaiQTL (Python)')
                if not r_sub.empty:
                    r_eff=[float(r_sub.loc[r_sub['founder']==f,'effect'].iloc[0])
                           if f in r_sub['founder'].values else 0.0 for f in f_names]
                    ax.bar(x+w/2,r_eff,w,color=cf,hatch='///',edgecolor='#333',
                           lw=0.6,alpha=0.7,label='miQTL (R)')
                    _r2=(float(np.corrcoef(py_eff,r_eff)[0,1])
                         if np.std(py_eff)>0 and np.std(r_eff)>0 else np.nan)
                    ax.text(0.02,0.97,f'r={_r2:.4f}' if np.isfinite(_r2) else 'r=n/a',
                            transform=ax.transAxes,fontsize=10,va='top')
                ax.axhline(0,color='#444',lw=0.5)
                ax.set(xticks=x,xticklabels=f_names,ylabel='effect size',
                       title=f'founder effects — {best}  chr{chrom_b}  {peak_loc}')
                ax.set_xticklabels(f_names,rotation=45,ha='right',fontsize=10)
                ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
                save_plot(fig,'s4_founder_effects.png')

        lod_pass = (n_s+n_l)/max(len(stat_df),1) >= 0.50
        record('lmm_scan', lod_pass,
               f'strict={n_s}  lenient={n_l}  fail={n_f}  '
               f'median_r={stat_df["r"].median():.4f}')
    else:
        skip('lmm_scan','no locus overlap between R and Python')

# ══════════════════════════════════════════════════════════════════════════════
# S5 — CACHE COLLAPSE
# ══════════════════════════════════════════════════════════════════════════════
section(5, 'CACHE COLLAPSE')
r_coll = load_r('collapsed_cache_summary.csv')
if not hasattr(GenomeCache,'collapse'):
    skip('cache_collapse','GenomeCache.collapse() not implemented')
else:
    hi('  collapsing Python cache (l2.norm tol=0.02)...')
    cache_coll = cache_py.collapse(criterion='l2.norm', tol=0.02)
    rows_c = []
    for chrom in chr_order:
        n_full = len(cache_py._data[chrom]['locus_ids'])
        n_coll = len(cache_coll._data[chrom]['locus_ids'])
        pct_py = 100*(n_full-n_coll)/n_full
        n_r = n_r_orig = np.nan
        if r_coll is not None:
            rc = r_coll[r_coll['chr']==chrom]
            if not rc.empty:
                n_r_orig = int(rc['n_loci_orig'].iloc[0])
                n_r      = int(rc['n_loci_coll'].iloc[0])
        pct_r = 100*(n_r_orig-n_r)/n_r_orig if not np.isnan(n_r) else np.nan
        lids,pos,probs = cache_coll.get_chromosome(chrom)
        rsum_ok = np.allclose(probs.sum(axis=2),1.0,atol=1e-4)
        rows_c.append({'chr':chrom,'n_full':n_full,'n_py_collapsed':n_coll,
                       'n_r_collapsed':n_r,'pct_py':pct_py,'pct_r':pct_r,
                       'row_sums_ok':rsum_ok})
    coll_df = pd.DataFrame(rows_c)
    save_csv(coll_df,'collapse_comparison.csv')

    k_coll = kin_module.calc_kinship(cache_coll, method='loco')
    k_cors = [float(np.corrcoef(kinship_py[c].ravel(),k_coll[c].ravel())[0,1])
              for c in chr_order if c in kinship_py and c in k_coll]
    mean_k_r  = float(np.mean(k_cors)) if k_cors else 0.0
    mean_py   = float(coll_df['pct_py'].mean())
    mean_r    = float(coll_df['pct_r'].mean(skipna=True)) if not coll_df['pct_r'].isna().all() else np.nan
    rows_ok   = bool(coll_df['row_sums_ok'].all())
    coll_pass = mean_py >= 5.0 and rows_ok and mean_k_r > 0.99

    # LOD QC: scan same genes with full vs collapsed cache
    if pheno_strain is not None and r_lod is not None:
        _chrs_s5 = [c for c in chromosomes if c in cache_py._data][:2]
        lf=[]; lc=[]
        for _chrom in _chrs_s5:
            _genes = [g for g in r_lod[r_lod['chr']==_chrom]['gene_id'].unique()[:2]
                      if g in pheno_strain.columns]
            for _gene in _genes:
                _y = _get_y(_gene)
                if _y is None: continue
                _k = kinship_py[_chrom]; _ev,_evc = np.linalg.eigh(_k)
                _h2,_ = lmm_module.estimate_h2_null(_evc.T@_rint(_y),
                                                     _evc.T@covar_base,_ev)
                for _ca,_ls in [(cache_py,lf),(cache_coll,lc)]:
                    _cr = lmm_module.scan_chromosome(
                        _rint(_y),_ca,_chrom,covar_base,_ev,_evc,_h2)
                    _ls.append(pd.DataFrame({'locus':_cr['locus_ids'],
                                              'lod':_cr['lod'],'gene_id':_gene}))
        r_fc = np.nan
        if lf and lc:
            df_fc = pd.concat(lf).merge(pd.concat(lc),on=['locus','gene_id'],
                                        suffixes=('_full','_coll'))
            if len(df_fc)>3:
                r_fc = float(np.corrcoef(df_fc['lod_full'],df_fc['lod_coll'])[0,1])

    fig,axes = plt.subplots(1,3,figsize=(16,4))
    ax=axes[0]
    x=np.arange(len(coll_df));w=0.35
    ax.bar(x-w/2,coll_df['pct_py'],w,color='#4F6EAF',edgecolor='none',
           label='mosaiQTL (Python)')
    ax.bar(x+w/2,coll_df['pct_r'].fillna(0),w,color='#55AF5B',edgecolor='none',
           alpha=0.85,label='miQTL (R)')
    ax.set(xticks=x,xticklabels=[f'chr{c}' for c in coll_df['chr']],
           xlabel='chromosome',ylabel='loci reduced (%)',
           title=f'collapse l2.norm tol=0.02\npy={mean_py:.1f}%  r={mean_r:.1f}%')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right',fontsize=8)
    ax.legend(fontsize=9);sns.despine(ax=ax)
    ax=axes[1]
    ax.scatter(range(len(k_cors)),k_cors,s=20,color='#4F6EAF',edgecolors='none')
    ax.axhline(0.99,color='#D83026',lw=1,ls='--',label='r=0.99')
    ax.set(xlabel='chr index',ylabel='kinship r (full vs collapsed)',
           title=f'kinship stability\nmean r={mean_k_r:.4f}',ylim=(0.9,1.005))
    ax.legend(fontsize=9);sns.despine(ax=ax)
    ax=axes[2]
    if lf and lc and not np.isnan(r_fc):
        scatter_qc(ax,df_fc['lod_full'],df_fc['lod_coll'],
                   'LOD (full)','LOD (collapsed)',
                   f'LOD preservation\nr={r_fc:.4f}')
    else:
        ax.text(0.5,0.5,'LOD QC n/a',ha='center',transform=ax.transAxes)
    plt.tight_layout();save_plot(fig,'s5_cache_collapse.png')
    record('cache_collapse',coll_pass,
           f'py={mean_py:.1f}%  r={mean_r:.1f}%  '
           f'kinship_r={mean_k_r:.4f}  '
           f'lod_preservation={r_fc:.4f if not np.isnan(r_fc) else "n/a"}  '
           f'row_sums={rows_ok}')

# ══════════════════════════════════════════════════════════════════════════════
# S6 — PERMUTATION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
section(6, 'PERMUTATION THRESHOLDS')
r_thresh = load_r('thresholds_r.csv')
r_perms  = load_r('perm_lods_r.csv')
if r_thresh is None or pheno_strain is None:
    skip('permutation','thresholds_r.csv missing — run port_validation.R s6')
else:
    from mosaiqtl import permutation as perm_mod
    from scipy.stats import genextreme as _gev
    thresh_gene = str(r_thresh['gene'].iloc[0])
    thresh_chr  = str(r_thresh['chr'].iloc[0])
    thresh_r_v  = float(r_thresh['lod_threshold'].iloc[0])
    n_p         = int(r_thresh['n_perms'].iloc[0])
    y_t  = _get_y(thresh_gene)
    if y_t is None:
        skip('permutation',f'{thresh_gene} not in pheno')
    else:
        k_t   = kinship_py[thresh_chr]
        ev,evc= np.linalg.eigh(k_t)
        valid = ~np.isnan(y_t)
        y_v   = _rint(y_t[valid]); x_v=covar_base[valid]; evc_v=evc[valid]
        h2,_  = lmm_module.estimate_h2_null(evc_v.T@y_v,evc_v.T@x_v,ev)
        rng   = np.random.default_rng(42)
        max_lods = []
        hi(f'  {n_p} permutations — {thresh_gene} chr{thresh_chr}')
        for _ in range(n_p):
            yp  = rng.permutation(y_v)
            cr  = lmm_module.scan_chromosome(yp,cache_py,thresh_chr,x_v,ev,evc_v,h2)
            max_lods.append(float(cr['lod'].max()))
        max_lods = np.array(max_lods)
        thresh_py,gev_p = perm_mod.get_gev_thresholds(max_lods,0.95)
        delta = abs(thresh_py-thresh_r_v)
        perm_pass = delta < 0.5
        save_csv(pd.DataFrame({'perm_lod':max_lods}),'perm_lods_py.csv')
        save_csv(pd.DataFrame([{'gene':thresh_gene,'chr':thresh_chr,
                                 'n_perms':n_p,'threshold_py':thresh_py,
                                 'threshold_r':thresh_r_v,'delta':delta}]),
                 'threshold_comparison.csv')

        fig,axes=plt.subplots(1,2,figsize=(11,4))
        ax=axes[0]
        sns.histplot(max_lods,bins=30,color='#4F6EAF',ax=ax,
                     edgecolor='none',stat='density',label='py permutations')
        if r_perms is not None:
            sns.histplot(r_perms['perm_lod'],bins=30,color='#55AF5B',ax=ax,
                         edgecolor='none',stat='density',alpha=0.5,label='r permutations')
        xf=np.linspace(max_lods.min(),max_lods.max()*1.1,200)
        ax.plot(xf,_gev.pdf(xf,gev_p['c'],loc=gev_p['loc'],scale=gev_p['scale']),
                color='#C0BFBF',lw=2,label='GEV fit (py)')
        ax.axvline(thresh_py,color='#4F6EAF',lw=1.5,ls='--',
                   label=f'py 95%={thresh_py:.2f}')
        ax.axvline(thresh_r_v,color='#D83026',lw=1.5,ls=':',
                   label=f'R 95%={thresh_r_v:.2f}')
        ax.set(xlabel='max LOD',ylabel='density',
               title=f'permutation thresholds\n{thresh_gene} chr{thresh_chr}')
        ax.legend(fontsize=8);sns.despine(ax=ax)
        ax=axes[1]
        if r_perms is not None and len(r_perms)>3:
            scatter_qc(ax,np.sort(r_perms['perm_lod'].values),
                       np.sort(max_lods[:len(r_perms)])[:len(r_perms)],
                       'R perm max-LOD (sorted)',
                       'Python perm max-LOD (sorted)',
                       f'QQ permutation  Δthr={delta:.3f}')
        else:
            ax.text(0.5,0.5,'no R perm lods',ha='center',transform=ax.transAxes)
        plt.tight_layout();save_plot(fig,'s6_permutation.png')
        record('permutation',perm_pass,
               f'py={thresh_py:.3f}  R={thresh_r_v:.3f}  Δ={delta:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# S7 — MULTIPLE IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
section(7, 'MULTIPLE IMPUTATION')
r_mi = load_r('mi_lod_r.csv')
if r_mi is None or pheno_strain is None:
    skip('multi_imputation','mi_lod_r.csv missing — run port_validation.R s7')
else:
    from mosaiqtl.scan import scan_h2lmm_mi as _mi
    mi_comparisons=[]
    for chrom in chromosomes:
        genes_mi = r_mi[r_mi['chr']==chrom]['gene_id'].unique().tolist()
        if args.n_genes: genes_mi=genes_mi[:args.n_genes]
        if not genes_mi: continue
        kd_mi = kin_module.decomp_kinship({chrom:kinship_py[chrom]})
        for gene in genes_mi:
            y_mi = _get_y(gene)
            if y_mi is None: continue
            res_mi = _mi(y_mi,cache_py,covar_base,kd_mi,n_imp=11,
                         seed=1,phenotype_id=gene)
            py_mi  = pd.DataFrame({'locus':res_mi.locus_ids,'lod_py':res_mi.lod,
                                   'chr':res_mi.chromosomes})
            py_mi  = py_mi[py_mi['chr']==chrom]
            rg_mi  = r_mi[(r_mi['gene_id']==gene)&(r_mi['chr']==chrom)
                          ][['locus','lod']].rename(columns={'lod':'lod_r'})
            mg_mi  = py_mi.merge(rg_mi,on='locus',how='inner')
            if len(mg_mi)<3: continue
            mg_mi['gene_id']=gene; mg_mi['chr']=chrom
            mi_comparisons.append(mg_mi)
    if mi_comparisons:
        mi_comp = pd.concat(mi_comparisons,ignore_index=True)
        save_csv(mi_comp,'mi_lod_comparison.csv')
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        scatter_qc(axes[0],mi_comp['lod_r'],mi_comp['lod_py'],
                   'LOD MI (miQTL R)','LOD MI (mosaiQTL Python)','MI LOD scatter')
        # profile for first gene
        first_g = mi_comp['gene_id'].iloc[0]
        sub = mi_comp[mi_comp['gene_id']==first_g]
        axes[1].plot(sub['lod_r'].values,color='#D83026',lw=1.2,alpha=0.85,label='miQTL (R)')
        axes[1].plot(sub['lod_py'].values,color='#4F6EAF',lw=1.2,alpha=0.85,label='mosaiQTL (Python)')
        axes[1].set(xlabel='locus index',ylabel='LOD',
                    title=f'MI LOD profile — {first_g}')
        axes[1].legend(fontsize=9);sns.despine(ax=axes[1])
        plt.tight_layout();save_plot(fig,'s7_mi_scan.png')
        r_mi_all = float(np.corrcoef(mi_comp['lod_r'],mi_comp['lod_py'])[0,1])
        record('multi_imputation',r_mi_all>0.95,
               f'r={r_mi_all:.4f}  n_genes={mi_comp["gene_id"].nunique()}')
    else:
        skip('multi_imputation','no locus overlap')

# ══════════════════════════════════════════════════════════════════════════════
# S8 — SNP SCAN
# ══════════════════════════════════════════════════════════════════════════════
section(8, 'SNP SCAN')
r_snp = load_r('snp_lod_r.csv')
if not os.path.exists(SNP_CSV):
    skip('snp_scan',
         f'snp_genos.csv not found at {SNP_CSV}\n'
         '         prepare: cols=[chr,pos_mb,AJ,B6J,129S1J,NODJ,NZOJ,CASTJ,PWKJ,WSBJ]')
elif pheno_strain is None:
    skip('snp_scan','pheno not loaded')
else:
    from mosaiqtl.snp_scan import imputed_snp_scan_h2lmm as _snp_scan
    snp_genos = pd.read_csv(SNP_CSV, index_col=0)
    snp_genos['chr'] = snp_genos['chr'].astype(str)
    snp_comparisons=[]
    for chrom in chromosomes:
        genes_snp = (r_snp[r_snp['chr']==chrom]['gene_id'].unique().tolist()
                     if r_snp is not None else
                     [g for g in list(pheno_strain.columns)[:3]
                      if g not in ('subject.name','batch')])
        if args.n_genes: genes_snp=genes_snp[:args.n_genes]
        kd_snp = kin_module.decomp_kinship({chrom:kinship_py[chrom]})
        for gene in genes_snp:
            y_snp = _get_y(gene)
            if y_snp is None: continue
            res_snp = _snp_scan(y_snp,cache_py,snp_genos,covar_base,
                                 kd_snp,phenotype_id=gene)
            if res_snp.empty: continue
            if r_snp is not None:
                rg_s = r_snp[(r_snp['gene_id']==gene)&(r_snp['chr']==chrom)
                              ][['snp','lod']].rename(columns={'lod':'lod_r'})
                mg_s = res_snp.rename(columns={'lod':'lod_py'}).merge(
                    rg_s,on='snp',how='inner')
                if len(mg_s)>=3:
                    mg_s['gene_id']=gene; snp_comparisons.append(mg_s)
    if snp_comparisons:
        snp_comp = pd.concat(snp_comparisons,ignore_index=True)
        save_csv(snp_comp,'snp_lod_comparison.csv')
        fig,ax=plt.subplots(figsize=(5,5))
        r_snp_all = scatter_qc(ax,snp_comp['lod_r'],snp_comp['lod_py'],
                                'SNP LOD (miQTL R)','SNP LOD (mosaiQTL Python)',
                                f'SNP scan  r={np.corrcoef(snp_comp["lod_r"],snp_comp["lod_py"])[0,1]:.4f}')
        plt.tight_layout();save_plot(fig,'s8_snp_scan.png')
        record('snp_scan',r_snp_all>0.90,
               f'r={r_snp_all:.4f}  n_snps={snp_comp["snp"].nunique()}')
    else:
        # python-only run (no R reference)
        save_csv(res_snp if 'res_snp' in dir() else pd.DataFrame(),'snp_lod_py.csv')
        record('snp_scan',None,'python-only — no R reference')

# ══════════════════════════════════════════════════════════════════════════════
# S9 — MEDIATION
# ══════════════════════════════════════════════════════════════════════════════
section(9, 'MEDIATION')
r_med = load_r('mediation_r.csv')
if r_med is None or pheno_strain is None:
    skip('mediation','mediation_r.csv missing — run port_validation.R s9')
else:
    from mosaiqtl import mediation as med_mod
    med_gene = str(r_med['gene_id'].iloc[0])
    med_med  = str(r_med['mediator'].iloc[0])
    med_chr  = str(r_med['chr'].iloc[0])
    yo = _get_y(med_gene); ym = _get_y(med_med)
    if yo is None or ym is None:
        skip('mediation',f'{med_gene} or {med_med} not in pheno')
    else:
        kd_m = kin_module.decomp_kinship({med_chr:kinship_py[med_chr]})
        py_med = med_mod.mediation_scan_qr(yo,ym,cache_py,covar_base,kd_m,
                                            outcome_id=med_gene,mediator_id=med_med)
        pm_c = py_med[py_med['chr']==med_chr]
        rg_m = r_med[['locus','lod']].rename(columns={'lod':'lod_r'})
        mg_m = pm_c.rename(columns={'lod':'lod_py'}).merge(rg_m,on='locus',how='inner')
        if len(mg_m)>=3:
            save_csv(mg_m,'mediation_comparison.csv')
            r_med_v = float(np.corrcoef(mg_m['lod_r'],mg_m['lod_py'])[0,1])
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            scatter_qc(axes[0],mg_m['lod_r'],mg_m['lod_py'],
                       'mediation LOD (R)','mediation LOD (Python)',
                       f'mediation  r={r_med_v:.4f}')
            axes[1].plot(pm_c['pos_mb'],pm_c['mediation_lod'] if 'mediation_lod' in pm_c else pm_c['lod_py'],
                         color='#D83026',lw=1.2,label='mediation LOD')
            axes[1].set(xlabel='position (Mb)',ylabel='base LOD − mediated LOD',
                        title=f'mediation profile  chr{med_chr}')
            axes[1].legend(fontsize=9);sns.despine(ax=axes[1])
            plt.tight_layout();save_plot(fig,'s9_mediation.png')
            record('mediation',r_med_v>0.90,f'r={r_med_v:.4f}')
        else:
            skip('mediation',f'insufficient locus overlap ({len(mg_m)})')

# ══════════════════════════════════════════════════════════════════════════════
# S10 — POSITIONAL CI
# ══════════════════════════════════════════════════════════════════════════════
section(10, 'POSITIONAL CI')
r_ci   = load_r('ci_r.csv')
r_boot = load_r('boot_pos_r.csv')
if r_ci is None or pheno_strain is None:
    skip('positional_ci','ci_r.csv missing — run port_validation.R s10')
else:
    from mosaiqtl.permutation import run_positional_scans as _rps
    from mosaiqtl.scan import scan_h2lmm as _sh
    ci_gene = str(r_ci['gene_id'].iloc[0])
    ci_chr  = str(r_ci['chr'].iloc[0])
    ci_lo_r = float(r_ci['ci_lo_mb'].iloc[0])
    ci_hi_r = float(r_ci['ci_hi_mb'].iloc[0])
    n_boot  = int(r_ci['n_boot'].iloc[0]) if 'n_boot' in r_ci else 100
    y_ci = _get_y(ci_gene)
    if y_ci is None:
        skip('positional_ci',f'{ci_gene} not in pheno')
    else:
        kd_ci = kin_module.decomp_kinship({ci_chr:kinship_py[ci_chr]})
        ci_res = _sh(y_ci,cache_py,covar_base,kd_ci,phenotype_id=ci_gene)
        rng_ci = np.random.default_rng(1)
        valid_ci = ~np.isnan(y_ci)
        y_ci_v = _rint(y_ci[valid_ci]); y_ci_v -= y_ci_v.mean()
        boot_oc = np.column_stack([rng_ci.choice(y_ci_v,size=len(y_ci_v),
                                                   replace=True)
                                    for _ in range(n_boot)])
        boot_pos,ci_info = _rps(boot_oc,y_ci,cache_py,covar_base,kd_ci,
                                 peak_chrom=ci_chr)
        ci_lo_py = ci_info['ci_lo']; ci_hi_py = ci_info['ci_hi']
        overlap  = ci_lo_py<=ci_hi_r and ci_lo_r<=ci_hi_py
        save_csv(pd.DataFrame([{'gene_id':ci_gene,'chr':ci_chr,
                                 'ci_lo_py':ci_lo_py,'ci_hi_py':ci_hi_py,
                                 'ci_lo_r':ci_lo_r,'ci_hi_r':ci_hi_r,
                                 'overlap':overlap}]),'ci_comparison.csv')
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        vp = boot_pos[~np.isnan(boot_pos)] if hasattr(boot_pos,'__len__') else []
        ax=axes[0]
        if len(vp):
            sns.histplot(vp,bins=30,color='#4F6EAF',ax=ax,
                         edgecolor='none',label='py bootstrap peaks')
        if r_boot is not None:
            sns.histplot(r_boot['boot_pos'].dropna(),bins=30,color='#55AF5B',
                         ax=ax,edgecolor='none',alpha=0.5,label='R bootstrap peaks')
        ax.axvline(ci_lo_py,color='#4F6EAF',lw=1.2,ls='--')
        ax.axvline(ci_hi_py,color='#4F6EAF',lw=1.2,ls='--',
                   label=f'py=[{ci_lo_py:.2f},{ci_hi_py:.2f}]')
        ax.axvline(ci_lo_r,color='#D83026',lw=1.2,ls=':')
        ax.axvline(ci_hi_r,color='#D83026',lw=1.2,ls=':',
                   label=f'R=[{ci_lo_r:.2f},{ci_hi_r:.2f}]')
        ax.set(xlabel='peak position (Mb)',ylabel='bootstraps',
               title=f'positional CI — {ci_gene} chr{ci_chr}')
        ax.legend(fontsize=8);sns.despine(ax=ax)
        ax=axes[1]
        ax.scatter(ci_lo_py,0,s=60,color='#4F6EAF',zorder=5)
        ax.scatter(ci_hi_py,0,s=60,color='#4F6EAF',zorder=5)
        ax.hlines(0,ci_lo_py,ci_hi_py,colors='#4F6EAF',lw=4,alpha=0.7,label='Python CI')
        ax.hlines(0.05,ci_lo_r,ci_hi_r,colors='#D83026',lw=4,alpha=0.7,label='R CI')
        ax.set(xlabel='Mb',yticks=[],xlim=(min(ci_lo_py,ci_lo_r)-0.5,
                                           max(ci_hi_py,ci_hi_r)+0.5),
               title=f'CI comparison  overlap={overlap}')
        ax.legend(fontsize=9);sns.despine(ax=ax)
        plt.tight_layout();save_plot(fig,'s10_positional_ci.png')
        record('positional_ci',overlap,
               f'py=[{ci_lo_py:.2f},{ci_hi_py:.2f}]  '
               f'R=[{ci_lo_r:.2f},{ci_hi_r:.2f}]  overlap={overlap}')

# ══════════════════════════════════════════════════════════════════════════════
# S11 — INSTABILITY SCAN
# ══════════════════════════════════════════════════════════════════════════════
section(11, 'INSTABILITY SCAN')
r_fx = load_r('fixef_r.csv')
if r_fx is None or pheno_strain is None:
    skip('instability','fixef_r.csv missing — run port_validation.R s11')
else:
    from mosaiqtl.scan import instability_lm_scan as _inst
    fx_gene = str(r_fx['gene_id'].iloc[0])
    fx_chr  = str(r_fx['chr'].iloc[0])
    y_fx = _get_y(fx_gene)
    if y_fx is None:
        skip('instability',f'{fx_gene} not in pheno')
    else:
        loci_r = set(r_fx[r_fx['gene_id']==fx_gene]['locus'].tolist())
        inst_py = _inst(y_fx,cache_py,covar_base,n_samples=100,seed=1,
                        phenotype_id=fx_gene,just_these_loci=loci_r)
        py_inst = pd.DataFrame({'locus':inst_py.locus_ids,'lod_py':inst_py.lod,
                                 'chr':inst_py.chromosomes})
        py_inst = py_inst[py_inst['chr']==fx_chr]
        mg_fx   = py_inst.merge(r_fx[r_fx['gene_id']==fx_gene][['locus','lod']]
                                  .rename(columns={'lod':'lod_r'}),
                                 on='locus',how='inner')
        if len(mg_fx)>=3:
            save_csv(mg_fx,'fixef_comparison.csv')
            r_fx_v = float(np.corrcoef(mg_fx['lod_r'],mg_fx['lod_py'])[0,1])
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            scatter_qc(axes[0],mg_fx['lod_r'],mg_fx['lod_py'],
                       'instability LOD (R)','instability LOD (Python)',
                       f'instability scan  r={r_fx_v:.4f}')
            axes[1].plot(mg_fx['lod_r'].values,color='#D83026',lw=1.2,
                         alpha=0.85,label='miQTL (R)')
            axes[1].plot(mg_fx['lod_py'].values,color='#4F6EAF',lw=1.2,
                         alpha=0.85,label='mosaiQTL (Python)')
            axes[1].set(xlabel='locus index',ylabel='mean bootstrap LOD',
                        title=f'instability profile — {fx_gene} chr{fx_chr}')
            axes[1].legend(fontsize=9);sns.despine(ax=axes[1])
            plt.tight_layout();save_plot(fig,'s11_instability.png')
            record('instability',r_fx_v>0.90,f'r={r_fx_v:.4f}  n_loci={len(mg_fx)}')
        else:
            skip('instability',f'insufficient locus overlap ({len(mg_fx)})')

# ══════════════════════════════════════════════════════════════════════════════
# QC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(os.path.join(PY_DATA,'qc_summary.csv'),index=False)
hi('\n' + '='*65)
hi(f'{"section":<25} {"status":<6} detail')
hi('─'*65)
for row in qc_rows:
    st = 'PASS' if row['pass'] is True else ('SKIP' if row['pass'] is None else 'FAIL')
    hi(f'  {row["section"]:<23} {st:<6} {row["detail"]}')
hi(f'  outputs: {PY_DATA}/')
hi(f'  plots:   {PLOTS}/')
hi(f'  log:     {_log}')
hi('='*65)
impl = [r for r in qc_rows if r['pass'] is not None]
passed = [r for r in impl if r['pass'] is True]
sys.exit(0 if not impl or len(passed)==len(impl) else 1)
