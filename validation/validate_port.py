# validate_port.py
# complete mosaiQTL validation — Phase 1 (LMM core) + Phase 2 (full port)
#
# design:
#   - all outputs go to port_validation/r_data/ and port_validation/qc_plots/
#   - sections 1-3 re-run Phase 1 validation reading from lmm_validation/r_data/
#     as read-only R reference (produced by run_validation.R)
#   - sections 4-10 activate when their R reference CSV exists
#     (produced by port_validation.R) — gracefully skip otherwise
#   - global Z-matrix design: phenotype sample_ids != geno_ids supported
#
# usage:
#   python validation/validate_port.py
#   python validation/validate_port.py --chr 8 --n_genes 20
#   python validation/validate_port.py --skip-phase1

import numpy as np
import pandas as pd
import os, sys, argparse, logging, importlib
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module
from mosaiqtl.utils import rint as _rint, build_z_matrix

import datetime as _dt

# ── logging: detailed → file, highlights only → terminal ─────────────────────
_ts       = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'validation', 'port_validation', 'logs',
                         f'validate_port_{_ts}.log')
os.makedirs(os.path.dirname(_log_file), exist_ok=True)

log = logging.getLogger('validate_port')
log.setLevel(logging.DEBUG)

_fh = logging.FileHandler(_log_file)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
log.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.WARNING)   # terminal: warnings + errors only
_sh.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(_sh)

def hi(msg):
    """print a highlight line to terminal (always visible)."""
    print(msg, flush=True)

# ── args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--chr',         default=None)
p.add_argument('--n_genes',     default=None, type=int)
p.add_argument('--skip-phase1', action='store_true')
args = p.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

GM_CSV  = os.path.join(miqtl_dir, 'input',       'GM_ccfo_mm39.csv')
GM_GMAP = os.path.join(miqtl_dir, 'input', 'GM', 'gmap')
GM_PMAP = os.path.join(miqtl_dir, 'input', 'GM', 'pmap')

LMM_R   = os.path.join(pkg_dir, 'validation', 'lmm_validation',  'r_data')
PORT_R  = os.path.join(pkg_dir, 'validation', 'port_validation', 'r_data')

paths = {
    'kinship_r':    os.path.join(miqtl_dir, 'output', 'kinship_loco.rds'),
    'cache_r':      os.path.join(miqtl_dir, 'output', 'genomecache'),
    'pmap':         os.path.join(miqtl_dir, 'input',  'pmap.csv'),
    'lod_r':        os.path.join(LMM_R, 'lod_r.csv'),
    'pheno_strain':     os.path.join(LMM_R, 'pheno_48_used.csv'),
    'ae_r':         os.path.join(LMM_R, 'allele_effects.csv'),
    'pmap_r':       os.path.join(LMM_R, 'pmap_r.csv'),
    'collapsed_r':  os.path.join(PORT_R, 'collapsed_cache_summary.csv'),
    'thresholds_r': os.path.join(PORT_R, 'thresholds_r.csv'),
    'mi_lod_r':     os.path.join(PORT_R, 'mi_lod_r.csv'),
    'snp_lod_r':    os.path.join(PORT_R, 'snp_lod_r.csv'),
    'mediation_r':  os.path.join(PORT_R, 'mediation_r.csv'),
    'ci_r':         os.path.join(PORT_R, 'ci_r.csv'),
    'fixef_r':      os.path.join(PORT_R, 'fixef_r.csv'),
}

val_dir  = os.path.join(pkg_dir, 'validation', 'port_validation', 'r_data')
plot_dir = os.path.join(pkg_dir, 'validation', 'port_validation', 'qc_plots')
os.makedirs(val_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

# ── plotting ──────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'sans-serif',
                     'font.sans-serif':['Carlito','DejaVu Sans'],
                     'font.size':11})

FOUNDER_COLORS = {
    'AJ':'#F4C245','B6J':'#C0BFBF','129S1J':'#DA9CC1','NODJ':'#4F6EAF',
    'NZOJ':'#52A5DB','CASTJ':'#55AF5B','PWKJ':'#D83026','WSBJ':'#683C91',
}
CHR_PAL = ['#4F6EAF','#888780']

CC_COLS = ['CC001','CC002','CC003','CC004','CC005','CC006','CC007','CC008',
           'CC009','CC010','CC011','CC012','CC013','CC015','CC017','CC018',
           'CC024','CC025','CC028','CC029','CC030','CC032','CC036','CC037',
           'CC038','CC041','CC043','CC057','CC060','CC062','CC065','CC071','CC074']
FOUNDER_COLS = ['AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ']
F1_CROSSES   = [('B6AF1J',0,1),('B6129S1F1J',2,1),('B6NODF1J',3,1),
                ('B6NZOF1J',4,1),('B6CASTF1J',5,1),('B6PWKF1J',6,1),
                ('B6WSBF1J',7,1)]

_chr_sort = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)
qc_rows   = []

def save_csv(df, name):
    path = os.path.join(val_dir, name)
    df.to_csv(path, index=False)
    log.debug(f'  saved {name} ({len(df)} rows)')
    return path

def save_plot(fig, name):
    path = os.path.join(plot_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.debug(f'  saved {name}')

def section(title):
    msg = f'\n{"="*60}\n{title}\n{"="*60}'
    log.debug(msg)
    hi(f'\n▶ {title}')

def skip(name, reason):
    log.warning(f'{name}: SKIPPED — {reason}')
    qc_rows.append({'section': name, 'pass': None, 'detail': f'skipped: {reason}'})

def has_fn(mod, fn):
    try:
        return hasattr(importlib.import_module(f'mosaiqtl.{mod}'), fn)
    except ImportError:
        return False

# ── build Python cache (shared) ───────────────────────────────────────────────
log.info('building genome cache from raw GigaMUGA (R-free)...')
cache_py = GenomeCache.from_gigamuga(
    gigamuga_path = GM_CSV,
    gmap_dir      = GM_GMAP,
    pmap_dir      = GM_PMAP,
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)
log.info(f'cache: {len(cache_py.chromosomes)} chr | {len(cache_py.sample_ids)} strains')

# build LOCO kinship (48×48 genotype space)
kinship_py = kin_module.calc_kinship(cache_py, method='loco')
log.info(f'kinship: {len(kinship_py)} LOCO matrices')

chr_order  = sorted(cache_py.chromosomes, key=_chr_sort)
chr_colors = lambda s: [CHR_PAL[chr_order.index(c) % 2]
                        if c in chr_order else '#aaa' for c in s]

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SECTIONS 1-3  (LMM core, same as validate_vs_r.py)
# ══════════════════════════════════════════════════════════════════════════════

if not args.skip_phase1:

    # ── SECTION 1 — haplotype probs ───────────────────────────────────────────
    section('SECTION 1 — HAPLOTYPE PROBS')
    try:
        cache_r = (GenomeCache.from_happy_dir(paths['cache_r'], model='additive')
                   if os.path.isdir(paths['cache_r']) else None)
    except Exception as e:
        cache_r = None; log.warning(f'R cache unavailable: {e}')

    hap_rows = []
    py_s = list(cache_py.sample_ids)
    r_s  = list(cache_r.sample_ids) if cache_r else []
    shr  = [s for s in py_s if s in r_s]
    py_i = [py_s.index(s) for s in shr]
    r_i  = [r_s.index(s)  for s in shr]
    f_py = [f.upper() for f in cache_py.founders]
    f_r  = [f.upper() for f in cache_r.founders] if cache_r else []
    _MAP = {'AJ':['A','AJ'],'B6J':['B','B6J'],'129S1J':['C','129S1J'],
            'NODJ':['D','NODJ'],'NZOJ':['E','NZOJ'],'CASTJ':['F','CASTJ'],
            'PWKJ':['G','PWKJ'],'WSBJ':['H','WSBJ']}
    f_ri = []
    for f in f_py:
        c = [i for i,rf in enumerate(f_r)
             if rf in [x.upper() for x in _MAP.get(f,[])]]
        if not c: c = [i for i,rf in enumerate(f_r) if rf==f]
        f_ri.append(c[0] if c else None)
    fok = all(x is not None for x in f_ri)
    r_sc = 1.0
    if cache_r:
        _c0 = cache_r.chromosomes[0]
        _h  = cache_r._data[_c0]['probs'][len(cache_r._data[_c0]['probs'])//2]
        if float(_h.max()) > 1.5: r_sc = 2.0

    for chrom in chr_order:
        lids = cache_py._data[chrom]['locus_ids']
        n_loci = len(lids)
        diffs = []; rok = True
        n_r = (len(cache_r._data[chrom]['locus_ids'])
               if cache_r and chrom in cache_r._data else np.nan)
        if cache_r and chrom in cache_r._data:
            r_lids = cache_r._data[chrom]['locus_ids']
            r_pos  = cache_r._data[chrom]['positions']
            py_pos = cache_py._data[chrom]['positions']
            shr_l  = [l for l in lids if l in set(r_lids)]
            if len(shr_l) >= 5:
                step = len(shr_l)//5
                pairs = [(int(np.where(lids==l)[0][0]),
                          int(np.where(r_lids==l)[0][0]))
                         for l in shr_l[::step][:5]]
            else:
                py_si = [int(n_loci*f) for f in (0.1,0.3,0.5,0.7,0.9)]
                pairs = [(pi, int(np.argmin(np.abs(r_pos - py_pos[pi]))))
                         for pi in py_si]
            for pyi, ri in pairs:
                h = cache_py.get_locus(chrom, pyi)
                if not np.allclose(h.sum(axis=1), 1.0, atol=1e-4): rok = False
                hr = cache_r.get_locus(chrom, ri)
                hr_c = (hr[:,f_ri] if fok else hr) / r_sc
                diffs.append(float(np.abs(h[py_i] - hr_c[r_i]).max()))
        hap_rows.append({'chr':chrom,'n_loci_py':n_loci,'n_loci_r':n_r,
                         'n_shared':len(diffs),'row_sum_ok':rok,
                         'median_delta_prob': float(np.median(diffs)) if diffs else np.nan})

    hap_df = pd.DataFrame(hap_rows)
    save_csv(hap_df, 'haplotype_report.csv')
    hap_ok = bool(hap_df['row_sum_ok'].all())
    qc_rows.append({'section':'haplotype_probs','pass':hap_ok,
                    'detail':f'row_sums_ok={hap_ok}  '
                             f'median|Δprob|={hap_df["median_delta_prob"].max():.2e} (informational)'})

    # ── SECTION 2 — kinship ───────────────────────────────────────────────────
    section('SECTION 2 — KINSHIP')
    try:
        import rpy2.robjects as ro
        ro.r(f'k_list <- readRDS("{paths["kinship_r"]}")')
        kr     = {c: np.array(ro.r(f'k_list[["{c}"]]'), dtype=float)
                  for c in list(ro.r('names(k_list)'))}
        r_sk   = list(ro.r('rownames(k_list[[1]])'))
        py_sk  = list(cache_py.sample_ids)
        shr_k  = [s for s in py_sk if s in r_sk]
        py_ik  = np.array([py_sk.index(s) for s in shr_k])
        r_ik   = np.array([r_sk.index(s)  for s in shr_k])
        kin_rows = []
        for chrom in sorted([c for c in chr_order if c in kr], key=_chr_sort):
            kp = kinship_py[chrom][np.ix_(py_ik, py_ik)]
            kr_c = kr[chrom][np.ix_(r_ik, r_ik)]
            md = float(np.abs(kp - kr_c).max())
            r  = float(np.corrcoef(kp.ravel(), kr_c.ravel())[0,1])
            ok = r > 0.999 and md < 0.05
            kin_rows.append({'chr':chrom,'max_delta_K':md,'r':r,'pass':ok})
            log.info(f'  chr{chrom}: max|ΔK|={md:.2e}  r={r:.6f}  {"✓" if ok else "✗"}')
        kin_df = pd.DataFrame(kin_rows)
        save_csv(kin_df, 'kinship_report.csv')
        kin_pass = bool(kin_df['pass'].all())
        qc_rows.append({'section':'kinship','pass':kin_pass,
                        'detail':f'max|ΔK|={kin_df["max_delta_K"].max():.2e}'})
    except Exception as e:
        log.warning(f'kinship comparison skipped: {e}')
        qc_rows.append({'section':'kinship','pass':None,'detail':f'skipped: {e}'})

    # ── SECTION 3 — LMM scan ──────────────────────────────────────────────────
    section('SECTION 3 — LMM SCAN')
    if not os.path.exists(paths['lod_r']):
        skip('lmm_scan', 'lod_r.csv missing — run validate_vs_r.py or run_validation.R first')
    else:
        lod_r_df = pd.read_csv(paths['lod_r']); lod_r_df['chr'] = lod_r_df['chr'].astype(str)
        ae_r     = pd.read_csv(paths['ae_r']) if os.path.exists(paths['ae_r']) else None

        _raw = pd.read_csv(paths['pheno_strain'])
        if _raw.columns[0] != 'subject.name':
            _raw = pd.read_csv(paths['pheno_strain'], index_col=0).reset_index()
        pheno_strain_raw = _raw.drop_duplicates('subject.name', keep='first').reset_index(drop=True)

        all_chrs    = sorted(lod_r_df['chr'].unique(), key=_chr_sort)
        chromosomes = ([c.strip() for c in args.chr.split(',')
                        if c.strip() in all_chrs]
                       if args.chr else all_chrs)
        cache_s   = list(cache_py.sample_ids)
        pheno_strain  = pheno_strain_raw.set_index('subject.name').reindex(cache_s)

        # build covariate matrix (intercept + batch + 10 PCs)
        covar_parts = [np.ones((len(pheno_strain),1))]
        if 'batch' in pheno_strain.columns and pheno_strain['batch'].notna().any():
            bd = pd.get_dummies(pheno_strain['batch'], prefix='batch', drop_first=True)
            covar_parts.append(bd.astype(float).values)
        pcs = [c for c in [f'pc{i}' for i in range(1,11)] if c in pheno_strain.columns]
        if pcs: covar_parts.append(pheno_strain[pcs].fillna(0).astype(float).values)
        covar_base = np.hstack(covar_parts)
        pheno_strain   = pheno_strain.reset_index()

        comparisons=[]; gene_stats=[]; py_betas={}; n_pass=0; n_total=0

        for chrom in chromosomes:
            if chrom not in cache_py.chromosomes: continue
            genes = lod_r_df[lod_r_df['chr']==chrom]['gene_id'].unique().tolist()
            if args.n_genes: genes = genes[:args.n_genes]
            log.info(f'\n[chr{chrom}] {len(genes)} genes')
            k_chr       = kinship_py[chrom]
            ev, evc     = np.linalg.eigh(k_chr)
            x_rot       = evc.T @ covar_base
            locus_ids   = cache_py._data[chrom]['locus_ids']

            for gene in genes:
                if gene not in pheno_strain.columns: continue
                y = (pheno_strain.drop_duplicates('subject.name')
                     .set_index('subject.name').reindex(cache_s)[gene]
                     .values.astype(float))
                if np.all(np.isnan(y)): continue
                y_rot = evc.T @ _rint(y[~np.isnan(y)]) if not np.all(np.isnan(y)) else None
                if y_rot is None: continue
                y_rint = _rint(y)
                y_rot  = evc.T @ y_rint
                h2, _  = lmm_module.estimate_h2_null(y_rot, x_rot, ev)

                rg = lod_r_df[(lod_r_df['gene_id']==gene) &
                               (lod_r_df['chr']==chrom)][['locus','lod','pos_mb']]
                rg = rg.rename(columns={'lod':'lod_r'})
                if rg.empty: continue

                lod_py=[]; betas_py=[]
                for locus in rg['locus']:
                    idx = np.where(locus_ids==locus)[0]
                    if not len(idx): lod_py.append(np.nan); betas_py.append(None); continue
                    h_l   = cache_py.get_locus(chrom, int(idx[0]))
                    h_rot = evc.T @ h_l[:, :-1]
                    res   = lmm_module.fit_lmm_at_locus(y_rot, h_rot, x_rot, ev, h2)
                    lod_py.append(res.lod); betas_py.append(res.betas)

                rg = rg.copy(); rg['lod_py'] = lod_py
                mg = rg.dropna(subset=['lod_r','lod_py'])
                if len(mg) < 3: continue

                pk_r  = float(mg['lod_r'].max())
                pk_py = float(mg['lod_py'].max())
                py_pk = mg.loc[mg['lod_py'].idxmax()]
                lead  = py_pk['locus']
                r_at_py = (float(mg.loc[mg['locus']==lead,'lod_r'].iloc[0])
                           if (mg['locus']==lead).any() else 0.0)
                pm    = r_at_py >= 0.99 * pk_r
                md    = float((mg['lod_py']-mg['lod_r']).abs().max())
                rel   = md / max(pk_r, 0.01)
                rc    = float(np.corrcoef(mg['lod_r'], mg['lod_py'])[0,1])
                strict  = pm and rel < 0.10
                lenient = pm and rel < 0.20
                tier    = 'strict' if strict else ('lenient' if lenient else 'fail')
                n_total += 1
                if strict or lenient: n_pass += 1
                log.info(f'  {"PASS" if strict or lenient else "FAIL"}({tier:<7}) '
                         f'{gene:25s} r={rc:.4f} rel={rel:.3f} pm={pm}')
                mg = mg.copy(); mg['gene_id']=gene; mg['chr']=chrom
                comparisons.append(mg)
                gene_stats.append({'gene_id':gene,'chr':chrom,'tier':tier,
                                   'r':rc,'rel_error':rel,'peak_match':pm,
                                   'peak_lod_r':pk_r,'peak_lod_py':pk_py,'h2_null':h2})
                ll = rg['locus'].tolist()
                if lead in ll and betas_py[ll.index(lead)] is not None:
                    py_betas[gene] = (chrom, lead, cache_py.founders[:-1],
                                      betas_py[ll.index(lead)])

        if comparisons:
            comp_df = pd.concat(comparisons, ignore_index=True)
            stat_df = pd.DataFrame(gene_stats)
            save_csv(comp_df, 'lod_comparison.csv')
            save_csv(stat_df, 'gene_summary.csv')
            n_s = (stat_df['tier']=='strict').sum()
            n_l = (stat_df['tier']=='lenient').sum()
            n_f = (stat_df['tier']=='fail').sum()
            lod_pass = (n_s + n_l) / max(len(stat_df), 1) >= 0.50
            qc_rows.append({'section':'lmm_scan','pass':lod_pass,
                            'detail':f'{n_s} strict + {n_l} lenient / {n_f} fail | '
                                     f'median_r={stat_df["r"].median():.4f}'})

            # QC plots
            chr_ord_local = sorted(comp_df['chr'].unique(), key=_chr_sort)
            cc_fn = lambda s: [CHR_PAL[chr_ord_local.index(c) % 2]
                               if c in chr_ord_local else '#aaa' for c in s]
            tier_colors = {'strict':'#4F6EAF','lenient':'#55AF5B','fail':'#D83026'}

            # LOD correlation
            fig, ax = plt.subplots(figsize=(5,5))
            ax.scatter(comp_df['lod_r'], comp_df['lod_py'],
                       c=cc_fn(comp_df['chr']), s=3, alpha=1.0, edgecolors='none')
            lim = max(comp_df[['lod_r','lod_py']].max().max()*1.05, 1.0)
            ax.plot([0,lim],[0,lim],'--',color='#D83026',lw=1)
            dif = comp_df['lod_r']-comp_df['lod_py']
            r_all = float(np.corrcoef(comp_df['lod_r'],comp_df['lod_py'])[0,1])
            ax.text(0.05,0.95,f'$r$={r_all:.4f}\nmean Δ={dif.mean():.3f}\nmax|Δ|={dif.abs().max():.3f}',
                    transform=ax.transAxes, fontsize=9, va='top')
            ax.add_artist(plt.legend(handles=[
                mpatches.Patch(color=CHR_PAL[0],label='odd chr'),
                mpatches.Patch(color=CHR_PAL[1],label='even chr'),
                plt.Line2D([0],[0],ls='--',color='#D83026',label='$y=x$')],
                fontsize=8, loc='lower right'))
            ax.set(xlim=(0,lim),ylim=(0,lim),
                   xlabel='miQTL LOD (R)',ylabel='mosaiQTL LOD (Python)',
                   title='LOD comparison')
            sns.despine(); plt.tight_layout()
            save_plot(fig,'lod_correlation.png')

            # LOD residuals
            fig, ax = plt.subplots(figsize=(14,3))
            diff = comp_df['lod_py']-comp_df['lod_r']
            ax.scatter(range(len(diff)), diff,
                       c=cc_fn(comp_df['chr']), s=2, alpha=1.0, edgecolors='none')
            ax.axhline(0, color='#D83026', lw=0.8, ls='--')
            ax.text(0.99,0.97,f'mean={diff.mean():.3f}  sd={diff.std():.3f}  '
                    f'max|Δ|={diff.abs().max():.3f}',
                    transform=ax.transAxes, fontsize=9, va='top', ha='right')
            offset=0
            for c in chr_ord_local:
                nc = (comp_df['chr']==c).sum()
                if offset: ax.axvline(offset, color='white', lw=0.4)
                ax.text(offset+nc/2, 0.02, f'chr{c}', ha='center', fontsize=6.5,
                        color='#555', transform=ax.get_xaxis_transform())
                offset += nc
            ax.set(xlim=(0,len(diff)), xlabel='locus index',
                   ylabel='ΔLOD (Python − R)', title='LOD residuals by chromosome')
            sns.despine(); plt.tight_layout()
            save_plot(fig,'lod_residuals.png')

            # gene summary
            fig, axes = plt.subplots(1, 2, figsize=(11,4))
            ax = axes[0]
            bins=[0,0.05,0.10,0.20,0.40,0.80]
            lbls=['0-5%','5-10%','10-20%','20-40%','40-80%']
            stat_df['_bin'] = pd.cut(stat_df['rel_error'].clip(upper=0.8),
                                     bins=bins, labels=lbls, right=True)
            cnts = (stat_df.groupby(['_bin','tier'], observed=True)
                    .size().unstack(fill_value=0)
                    .reindex(columns=['fail','lenient','strict'], fill_value=0))
            xp = np.arange(len(lbls)); w=0.25
            for i,(tier,color) in enumerate([('fail','#D83026'),('lenient','#55AF5B'),('strict','#4F6EAF')]):
                v = cnts[tier].values if tier in cnts.columns else np.zeros(len(lbls))
                ax.bar(xp+(i-1)*w, v, w, color=color, edgecolor='none',
                       label=f'{tier} ({stat_df["tier"].eq(tier).sum():,})')
            ax.set(xticks=xp, xticklabels=lbls,
                   xlabel='rel error (max|Δlod|/peak lod R)',
                   ylabel='genes',
                   title=f'LOD relative error by tier\nstrict={n_s:,}  lenient={n_l:,}  fail={n_f:,}')
            ax.legend(fontsize=9); sns.despine(ax=ax)
            ax = axes[1]
            pm = stat_df.groupby(['tier','peak_match'], observed=True).size().unstack(fill_value=0)
            pm = pm.reindex(['fail','lenient','strict']).fillna(0)
            pm.plot(kind='bar', ax=ax, color=['#C0BFBF','#4F6EAF'],
                    edgecolor='none', width=0.6)
            ax.set(xlabel='', ylabel='genes',
                   title='peak locus concordance by tier')
            ax.set_xticklabels(['fail','lenient','strict'], rotation=0)
            ax.legend(['peak mismatch','peak match'], fontsize=9)
            sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig,'gene_summary.png')

            # founder effects
            if ae_r is not None and py_betas:
                gw_ae = set(ae_r['gene_id'].unique())
                best  = next((g for g in
                              (stat_df[stat_df['gene_id'].isin(gw_ae)]
                               .sort_values('peak_lod_r', ascending=False)['gene_id'])
                              if g in py_betas), None)
                if best:
                    chrom_b, peak_loc, f_names, py_eff = py_betas[best]
                    gene_ae = ae_r[ae_r['gene_id']==best]
                    r_ae    = gene_ae[gene_ae['locus']==peak_loc]
                    if r_ae.empty and not gene_ae.empty:
                        r_ae = ae_r[(ae_r['gene_id']==best) &
                                    (ae_r['locus']==gene_ae['locus'].iloc[0])]
                    fig, ax = plt.subplots(figsize=(9,4))
                    x = np.arange(len(f_names)); ww=0.35
                    cf = [FOUNDER_COLORS.get(f,'#888') for f in f_names]
                    ax.bar(x-ww/2, py_eff, ww, color=cf, edgecolor='white', lw=0.5)
                    if not r_ae.empty:
                        r_eff = [float(r_ae.loc[r_ae['founder']==f,'effect'].iloc[0])
                                 if f in r_ae['founder'].values else 0.0
                                 for f in f_names]
                        ax.bar(x+ww/2, r_eff, ww, color=cf, hatch='///',
                               edgecolor='#333', lw=0.6, alpha=0.7)
                        _r2 = (float(np.corrcoef(py_eff,r_eff)[0,1])
                               if np.std(py_eff)>0 and np.std(r_eff)>0 else np.nan)
                        ax.text(0.02,0.97, f'r={_r2:.4f}' if np.isfinite(_r2) else 'r=n/a',
                                transform=ax.transAxes, fontsize=10, va='top')
                    ax.legend(handles=[
                        mpatches.Patch(fc='#888',ec='white',label='mosaiQTL (Python)'),
                        mpatches.Patch(fc='#888',ec='#333',hatch='///',alpha=0.7,label='miQTL (R)')],
                        fontsize=9)
                    ax.axhline(0, color='#444', lw=0.5)
                    ax.set(xticks=x, title=f'founder effects — {best}  chr{chrom_b}  {peak_loc}',
                           ylabel='founder effect size')
                    ax.set_xticklabels(f_names, rotation=45, ha='right', fontsize=10)
                    sns.despine(); plt.tight_layout()
                    save_plot(fig,'founder_effects.png')
        else:
            qc_rows.append({'section':'lmm_scan','pass':False,'detail':'no comparisons'})

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SECTIONS 4-10
# ══════════════════════════════════════════════════════════════════════════════

# helper: load R csv if present
def load_r(key):
    p = paths[key]
    if os.path.exists(p):
        df = pd.read_csv(p)
        if 'chr' in df.columns: df['chr'] = df['chr'].astype(str)
        return df
    return None

# ── SECTION 4 — genome cache collapse ────────────────────────────────────────
section('SECTION 4 — CACHE COLLAPSE')
if not hasattr(GenomeCache, 'collapse'):
    skip('cache_collapse', 'GenomeCache.collapse() not implemented')
else:
    r_coll = load_r('collapsed_r')
    log.info('  collapsing Python cache (l2.norm tol=0.02)...')
    cache_coll = cache_py.collapse(criterion='l2.norm', tol=0.02)
    rows_c = []
    for chrom in chr_order:
        n_full = len(cache_py._data[chrom]['locus_ids'])
        n_coll = len(cache_coll._data[chrom]['locus_ids'])
        pct    = 100*(n_full-n_coll)/n_full
        lids, pos, probs = cache_coll.get_chromosome(chrom)
        rsum_ok = np.allclose(probs.sum(axis=2), 1.0, atol=1e-4)
        n_r = n_r_c = np.nan
        if r_coll is not None and not r_coll.empty and 'status' not in r_coll.columns:
            row = r_coll[r_coll['chr']==chrom]
            if not row.empty:
                n_r   = int(row['n_loci_orig'].iloc[0])
                n_r_c = int(row['n_loci_coll'].iloc[0])
        rows_c.append({'chr':chrom,'n_full':n_full,'n_collapsed':n_coll,
                       'pct_reduced':pct,'row_sums_ok':rsum_ok,
                       'n_r_orig':n_r,'n_r_coll':n_r_c})
        log.info(f'  chr{chrom}: {n_full}→{n_coll} ({pct:.1f}%)  row_sums={rsum_ok}')
    coll_df = pd.DataFrame(rows_c)
    save_csv(coll_df, 'collapse_comparison.csv')
    # kinship stability after collapse
    k_coll = kin_module.calc_kinship(cache_coll, method='loco')
    k_cors = [float(np.corrcoef(kinship_py[c].ravel(), k_coll[c].ravel())[0,1])
              for c in chr_order if c in kinship_py and c in k_coll]
    mean_k_r  = float(np.mean(k_cors)) if k_cors else 0.0
    mean_red   = float(coll_df['pct_reduced'].mean())
    rows_ok    = bool(coll_df['row_sums_ok'].all())
    coll_pass  = mean_red >= 5.0 and rows_ok and mean_k_r > 0.99
    qc_rows.append({'section':'cache_collapse','pass':coll_pass,
                    'detail':f'py_reduction={mean_red:.1f}%  kinship_r={mean_k_r:.4f}  row_sums={rows_ok}'})
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    ax=axes[0]
    ax.bar(range(len(coll_df)), coll_df['pct_reduced'],
           color='#4F6EAF', edgecolor='none', label='mosaiQTL (Python)')
    if r_coll is not None and 'status' not in (r_coll.columns if r_coll is not None else []):
        pct_r = [(100*(row['n_r_orig']-row['n_r_coll'])/row['n_r_orig']
                  if row['n_r_orig']>0 else 0) for _,row in coll_df.iterrows()]
        ax.bar(np.arange(len(coll_df))+0.35, pct_r,
               0.35, color='#55AF5B', edgecolor='none', alpha=0.85, label='miQTL (R)')
    ax.set(xticks=range(len(coll_df)),
           xticklabels=[f'chr{c}' for c in coll_df['chr']],
           xlabel='chromosome', ylabel='loci reduced (%)',
           title=f'cache collapse — l2.norm tol=0.02\nmean={mean_red:.1f}%')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); sns.despine(ax=ax)
    ax=axes[1]
    ax.scatter(range(len(k_cors)), k_cors, s=20, color='#4F6EAF', edgecolors='none')
    ax.axhline(0.99, color='#D83026', lw=1, ls='--', label='r=0.99')
    ax.set(xlabel='chromosome index', ylabel='kinship r (full vs collapsed)',
           title=f'kinship preservation after collapse\nmean r={mean_k_r:.4f}',
           ylim=(0.9,1.01))
    ax.legend(fontsize=9); sns.despine(ax=ax)
    plt.tight_layout(); save_plot(fig,'collapse_comparison.png')

# ── SECTION 5 — permutation thresholds ───────────────────────────────────────
section('SECTION 5 — PERMUTATION THRESHOLDS')
if not has_fn('permutation','get_gev_thresholds'):
    skip('permutation_thresholds','mosaiqtl.permutation not available')
else:
    thresh_r = load_r('thresholds_r')
    if thresh_r is None:
        skip('permutation_thresholds','thresholds_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl import permutation as perm_mod
        from scipy.stats import genextreme as _gev
        thresh_gene = str(thresh_r['gene'].iloc[0])
        thresh_chr  = str(thresh_r['chr'].iloc[0])
        thresh_r_v  = float(thresh_r['lod_threshold'].iloc[0])
        n_p         = int(thresh_r['n_perms'].iloc[0])
        log.info(f'  R threshold: {thresh_r_v:.3f}  gene={thresh_gene}  chr={thresh_chr}')

        pheno_col = pheno_strain.set_index('subject.name').reindex(
            cache_py.sample_ids)[thresh_gene].values.astype(float)
        k_c  = kinship_py[thresh_chr]
        ev,evc = np.linalg.eigh(k_c)
        valid  = ~np.isnan(pheno_col)
        y_v    = _rint(pheno_col[valid])
        x_v    = covar_base[valid]
        evc_v  = evc[valid]
        h2, _  = lmm_module.estimate_h2_null(evc_v.T@y_v, evc_v.T@x_v, ev)

        rng_p = np.random.default_rng(42)
        max_lods_py = []
        log.info(f'  running {n_p} permutations...')
        for _ in range(n_p):
            y_perm = rng_p.permutation(y_v)
            cr = lmm_module.scan_chromosome(
                y_perm, cache_py, thresh_chr, x_v, ev, evc_v, h2)
            max_lods_py.append(float(cr['lod'].max()))
        max_lods_py = np.array(max_lods_py)
        thresh_py, gev_p = perm_mod.get_gev_thresholds(max_lods_py, 0.95)
        delta = abs(thresh_py - thresh_r_v)
        tp = delta < 0.5
        log.info(f'  Python threshold: {thresh_py:.3f}  Δ={delta:.3f}  {"PASS ✓" if tp else "FAIL ✗"}')
        qc_rows.append({'section':'permutation_thresholds','pass':tp,
                        'detail':f'py={thresh_py:.3f}  r={thresh_r_v:.3f}  Δ={delta:.3f}'})
        save_csv(pd.DataFrame([{'gene':thresh_gene,'chr':thresh_chr,'n_perms':n_p,
                                 'threshold_py':thresh_py,'threshold_r':thresh_r_v,
                                 'gev_c':gev_p['c'],'gev_loc':gev_p['loc'],
                                 'gev_scale':gev_p['scale']}]),
                 'threshold_comparison.csv')
        fig,ax = plt.subplots(figsize=(7,4))
        sns.histplot(max_lods_py, bins=30, color='#4F6EAF', ax=ax,
                     edgecolor='none', stat='density', label='permutation max-LOD')
        xf = np.linspace(max_lods_py.min(), max_lods_py.max()*1.1, 200)
        ax.plot(xf, _gev.pdf(xf, gev_p['c'], loc=gev_p['loc'], scale=gev_p['scale']),
                color='#C0BFBF', lw=2, label='GEV fit')
        ax.axvline(thresh_py, color='#4F6EAF', lw=1.5, ls='--',
                   label=f'Python 95% = {thresh_py:.2f}')
        ax.axvline(thresh_r_v, color='#D83026', lw=1.5, ls=':',
                   label=f'R 95% = {thresh_r_v:.2f}')
        ax.set(xlabel='genome-wide max LOD', ylabel='density',
               title=f'permutation threshold — {thresh_gene} chr{thresh_chr}')
        ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
        save_plot(fig,'threshold_distribution.png')

# ── SECTION 6 — multiple imputation ──────────────────────────────────────────
section('SECTION 6 — MULTIPLE IMPUTATION')
if not has_fn('scan','scan_h2lmm_mi'):
    skip('multi_imputation','scan_h2lmm_mi not available')
else:
    mi_r = load_r('mi_lod_r')
    if mi_r is None:
        skip('multi_imputation','mi_lod_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl.scan import scan_h2lmm_mi as _mi
        mi_gene = mi_r.groupby('gene_id').size().idxmax()
        mi_chr  = mi_r[mi_r['gene_id']==mi_gene]['chr'].iloc[0]
        mi_gene_r = mi_r[mi_r['gene_id']==mi_gene][['locus','lod']].rename(columns={'lod':'lod_r'})
        log.info(f'  MI scan: {mi_gene} chr{mi_chr}')
        pheno_mi = pheno_strain.set_index('subject.name').reindex(
            cache_py.sample_ids)[mi_gene].values.astype(float)
        kd_mi = kin_module.decomp_kinship({mi_chr: kinship_py[mi_chr]})
        res_mi = _mi(pheno_mi, cache_py, covar_base, kd_mi, n_imp=11, seed=1,
                     phenotype_id=mi_gene)
        py_mi = pd.DataFrame({'locus':res_mi.locus_ids,'lod_py':res_mi.lod,
                              'chr':res_mi.chromosomes})
        py_mi = py_mi[py_mi['chr']==mi_chr]
        mg_mi = py_mi.merge(mi_gene_r, on='locus', how='inner')
        if len(mg_mi) < 3:
            skip('multi_imputation', f'insufficient locus overlap ({len(mg_mi)})')
        else:
            r_mi = float(np.corrcoef(mg_mi['lod_r'], mg_mi['lod_py'])[0,1])
            mp = r_mi > 0.95
            qc_rows.append({'section':'multi_imputation','pass':mp,
                            'detail':f'r={r_mi:.4f}  n_loci={len(mg_mi)}'})
            save_csv(mg_mi,'mi_lod_comparison.csv')
            fig,ax = plt.subplots(figsize=(10,4))
            ax.plot(mg_mi['lod_r'],  color='#D83026',lw=1.2,alpha=0.85,label='miQTL (R) MI')
            ax.plot(mg_mi['lod_py'], color='#4F6EAF',lw=1.2,alpha=0.85,label='mosaiQTL MI')
            ax.set(xlabel='locus index',ylabel='LOD',
                   title=f'multiple imputation — {mi_gene} chr{mi_chr}\nr={r_mi:.4f}  n_imp=11')
            ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
            save_plot(fig,'mi_lod_profile.png')

# ── SECTION 7 — SNP scan ─────────────────────────────────────────────────────
section('SECTION 7 — SNP SCAN')
if not has_fn('snp_scan','imputed_snp_scan_h2lmm'):
    skip('snp_scan','mosaiqtl.snp_scan not available')
else:
    snp_r = load_r('snp_lod_r')
    if snp_r is None:
        skip('snp_scan','snp_lod_r.csv missing — run port_validation.R with allele_dir')
    else:
        skip('snp_scan','allele_dir format differs from Python snp_genos API — '
             'manual alignment required; validation planned with VCF-based SNP input')

# ── SECTION 8 — mediation ────────────────────────────────────────────────────
section('SECTION 8 — MEDIATION')
if not has_fn('mediation','mediation_scan_qr'):
    skip('mediation','mosaiqtl.mediation not available')
else:
    med_r = load_r('mediation_r')
    if med_r is None:
        skip('mediation','mediation_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl import mediation as med_mod
        med_gene = str(med_r['gene_id'].iloc[0])
        med_med  = str(med_r['mediator'].iloc[0])
        med_chr  = str(med_r['chr'].iloc[0])
        log.info(f'  mediation: outcome={med_gene}  mediator={med_med}  chr={med_chr}')
        pheno_o = pheno_strain.set_index('subject.name').reindex(
            cache_py.sample_ids)[med_gene].values.astype(float)
        pheno_m = pheno_strain.set_index('subject.name').reindex(
            cache_py.sample_ids)[med_med].values.astype(float)
        kd_med = kin_module.decomp_kinship({med_chr: kinship_py[med_chr]})
        py_med = med_mod.mediation_scan_qr(
            pheno_o, pheno_m, cache_py, covar_base, kd_med,
            outcome_id=med_gene, mediator_id=med_med)
        pm_c = py_med[py_med['chr']==med_chr]
        mg_med = pm_c.merge(med_r[['locus','lod']].rename(columns={'lod':'lod_r'}),
                            on='locus', how='inner')
        if len(mg_med) < 3:
            skip('mediation',f'insufficient locus overlap ({len(mg_med)})')
        else:
            r_med = float(np.corrcoef(mg_med['lod_r'], mg_med['lod'])[0,1])
            medp  = r_med > 0.90
            qc_rows.append({'section':'mediation','pass':medp,
                            'detail':f'r={r_med:.4f}  n_loci={len(mg_med)}'})
            save_csv(mg_med,'mediation_comparison.csv')
            fig, axes = plt.subplots(1,2,figsize=(11,4))
            ax=axes[0]
            lim = max(mg_med[['lod_r','lod']].max().max()*1.05,1.0)
            ax.scatter(mg_med['lod_r'], mg_med['lod'],
                       s=4, alpha=1.0, color='#4F6EAF', edgecolors='none')
            ax.plot([0,lim],[0,lim],'--',color='#D83026',lw=1,label='$y=x$')
            ax.set(xlim=(0,lim),ylim=(0,lim),
                   xlabel='miQTL mediation LOD (R)',
                   ylabel='mosaiQTL mediation LOD (Python)',
                   title=f'mediation — {med_gene} / {med_med}\nr={r_med:.4f}')
            ax.legend(fontsize=9); sns.despine(ax=ax)
            ax=axes[1]
            ax.plot(pm_c['pos_mb'], pm_c['mediation_lod'],
                    color='#D83026', lw=1.2, label='mediation LOD')
            ax.set(xlabel='position (Mb)',ylabel='LOD base − LOD mediated',
                   title=f'mediation effect — chr{med_chr}')
            ax.legend(fontsize=9); sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig,'mediation_comparison.png')

# ── SECTION 9 — positional CI ────────────────────────────────────────────────
section('SECTION 9 — POSITIONAL CI')
if not has_fn('permutation','run_positional_scans'):
    skip('positional_ci','run_positional_scans not available')
else:
    ci_r = load_r('ci_r')
    if ci_r is None:
        skip('positional_ci','ci_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl.permutation import run_positional_scans as _rps
        from mosaiqtl.scan import scan_h2lmm as _sh
        ci_gene = str(ci_r['gene_id'].iloc[0])
        ci_chr  = str(ci_r['chr'].iloc[0])
        ci_lo_r = float(ci_r['ci_lo_mb'].iloc[0])
        ci_hi_r = float(ci_r['ci_hi_mb'].iloc[0])
        n_boot  = int(ci_r['n_boot'].iloc[0]) if 'n_boot' in ci_r else 100
        log.info(f'  CI: {ci_gene} chr{ci_chr}  R=[{ci_lo_r:.2f},{ci_hi_r:.2f}]')
        pheno_ci = pheno_strain.set_index('subject.name').reindex(
            cache_py.sample_ids)[ci_gene].values.astype(float)
        kd_ci  = kin_module.decomp_kinship({ci_chr: kinship_py[ci_chr]})
        ci_res = _sh(pheno_ci, cache_py, covar_base, kd_ci, phenotype_id=ci_gene)
        rng_ci = np.random.default_rng(1)
        valid_ci = ~np.isnan(pheno_ci)
        y_ci = _rint(pheno_ci[valid_ci]) ; y_ci -= y_ci.mean()
        boot_oc = np.column_stack([rng_ci.choice(y_ci,size=len(y_ci),replace=True)
                                    for _ in range(n_boot)])
        boot_pos, ci_info = _rps(boot_oc, pheno_ci, cache_py, covar_base, kd_ci,
                                  peak_chrom=ci_chr)
        ci_lo_py = ci_info['ci_lo']; ci_hi_py = ci_info['ci_hi']
        overlap  = ci_lo_py <= ci_hi_r and ci_lo_r <= ci_hi_py
        qc_rows.append({'section':'positional_ci','pass':overlap,
                        'detail':f'py=[{ci_lo_py:.2f},{ci_hi_py:.2f}]  '
                                  f'r=[{ci_lo_r:.2f},{ci_hi_r:.2f}]'})
        save_csv(pd.DataFrame([{'gene_id':ci_gene,'chr':ci_chr,
                                 'ci_lo_py':ci_lo_py,'ci_hi_py':ci_hi_py,
                                 'ci_lo_r':ci_lo_r, 'ci_hi_r':ci_hi_r}]),
                 'ci_comparison.csv')
        fig,ax = plt.subplots(figsize=(7,4))
        vp = boot_pos[~np.isnan(boot_pos)]
        sns.histplot(vp, bins=30, color='#4F6EAF', ax=ax,
                     edgecolor='none', label='bootstrap peaks')
        ax.axvline(ci_lo_py, color='#4F6EAF', lw=1.2, ls='--')
        ax.axvline(ci_hi_py, color='#4F6EAF', lw=1.2, ls='--',
                   label=f'Python 95% CI [{ci_lo_py:.2f},{ci_hi_py:.2f}]')
        ax.axvline(ci_lo_r, color='#D83026', lw=1.2, ls=':')
        ax.axvline(ci_hi_r, color='#D83026', lw=1.2, ls=':',
                   label=f'R 95% CI [{ci_lo_r:.2f},{ci_hi_r:.2f}]')
        ax.set(xlabel='peak position (Mb)', ylabel='bootstraps',
               title=f'positional CI — {ci_gene} chr{ci_chr}')
        ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
        save_plot(fig,'positional_ci.png')

# ── SECTION 10 — fixed-effect / instability scan ─────────────────────────────
section('SECTION 10 — FIXED-EFFECT SCANS')
if not has_fn('scan','instability_lm_scan'):
    skip('fixef_scans','instability_lm_scan not available')
else:
    fx_r = load_r('fixef_r')
    if fx_r is None:
        skip('fixef_scans','fixef_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl.scan import instability_lm_scan as _inst
        fx_gene = str(fx_r['gene_id'].iloc[0])
        fx_chr  = str(fx_r['chr'].iloc[0])
        log.info(f'  instability scan: {fx_gene} chr{fx_chr}')
        pheno_f = pheno_strain.set_index('subject.name').reindex(
            cache_py.sample_ids)[fx_gene].values.astype(float)
        loci_r  = set(fx_r[fx_r['gene_id']==fx_gene]['locus'].tolist())
        inst_py = _inst(pheno_f, cache_py, covar_base,
                        n_samples=100, seed=1, phenotype_id=fx_gene,
                        just_these_loci=loci_r)
        py_inst = pd.DataFrame({'locus':inst_py.locus_ids,
                                 'lod_py':inst_py.lod,
                                 'chr':inst_py.chromosomes})
        py_inst = py_inst[py_inst['chr']==fx_chr]
        mg_fx = py_inst.merge(
            fx_r[fx_r['gene_id']==fx_gene][['locus','lod']].rename(
                columns={'lod':'lod_r'}), on='locus', how='inner')
        if len(mg_fx) < 3:
            skip('fixef_scans',f'insufficient locus overlap ({len(mg_fx)})')
        else:
            r_fx  = float(np.corrcoef(mg_fx['lod_r'], mg_fx['lod_py'])[0,1])
            fxp   = r_fx > 0.90
            qc_rows.append({'section':'fixef_scans','pass':fxp,
                            'detail':f'instability r={r_fx:.4f}  n_loci={len(mg_fx)}'})
            save_csv(mg_fx,'fixef_comparison.csv')
            fig,ax = plt.subplots(figsize=(10,4))
            ax.plot(mg_fx['lod_r'],  color='#D83026',lw=1.2,alpha=0.85,
                    label='miQTL instability.lm.scan (R)')
            ax.plot(mg_fx['lod_py'], color='#4F6EAF',lw=1.2,alpha=0.85,
                    label='mosaiQTL instability_lm_scan (Python)')
            ax.set(xlabel='locus index',ylabel='mean bootstrap LOD',
                   title=f'instability scan — {fx_gene} chr{fx_chr}\nr={r_fx:.4f}')
            ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
            save_plot(fig,'fixef_instability.png')

# ══════════════════════════════════════════════════════════════════════════════
# QC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section('QC SUMMARY')
qc_df = pd.DataFrame(qc_rows)
save_csv(qc_df,'qc_summary.csv')
hi('')
hi('='*60)
for row in qc_rows:
    st = 'PASS' if row['pass'] is True else ('SKIP' if row['pass'] is None else 'FAIL')
    hi(f'  {st:<4}  {row["section"]:<28}  {row["detail"]}')
hi(f'  outputs: {val_dir}/')
hi(f'  log:     {_log_file}')
hi('='*60)
implemented = [r for r in qc_rows if r['pass'] is not None]
passed      = [r for r in implemented if r['pass'] is True]
sys.exit(0 if (not implemented or len(passed)==len(implemented)) else 1)
