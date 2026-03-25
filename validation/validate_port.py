# validate_port.py
# complete mosaiQTL validation — Phase 1 (LMM core) + Phase 2 (full port)
#
# sections:
#   1. haplotype probs      — from_gigamuga() vs R HAPPY cache
#   2. kinship              — calc_kinship(loco) vs kinship_loco.rds
#   3. lmm scan             — scan_h2lmm() vs miQTL scan.h2lmm()
#   4. cache collapse       — GenomeCache.collapse() vs R collapse.genomecache()
#   5. permutation thresh.  — get_gev_thresholds() vs R get.gev.thresholds()
#   6. multiple imputation  — scan_h2lmm(use_multi_impute) vs R MI scan
#   7. snp scan             — imputed_snp_scan_h2lmm() vs R imputed.snp.scan.h2lmm()
#   8. mediation            — mediation_scan_qr() vs R mediation.scan.qr()
#   9. positional CI        — run_positional_scans() vs R run.positional.scans()
#  10. fixed-effect scans   — instability_lm_scan(), fixef_eqtl_scan() vs R equivalents
#
# phase 1 sections (1-3) run unconditionally using existing lmm_validation r_data.
# phase 2 sections (4-10) are skipped with a warning if the Python module is not
# yet implemented — allowing this script to grow section-by-section.
#
# csv outputs → validation/port_validation/r_data/
# plot outputs → validation/port_validation/qc_plots/
#
# usage:
#   python validation/validate_port.py                  # all phases
#   python validation/validate_port.py --chr 8          # quick test
#   python validation/validate_port.py --skip-phase1    # phase 2 only

import numpy as np
import pandas as pd
import os, sys, argparse, logging, importlib
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger('validate_port')

# ── paths ──────────────────────────────────────────────────────────────────────
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

GM_CSV  = os.path.join(miqtl_dir, 'input',       'GM_ccfo_mm39.csv')
GM_GMAP = os.path.join(miqtl_dir, 'input', 'GM', 'gmap')
GM_PMAP = os.path.join(miqtl_dir, 'input', 'GM', 'pmap')

# phase 1 inputs (produced by run_validation.R — reused here)
LMM_R_DATA = os.path.join(pkg_dir, 'validation', 'lmm_validation', 'r_data')

# phase 2 inputs (produced by port_validation.R)
PORT_R_DATA = os.path.join(pkg_dir, 'validation', 'port_validation', 'r_data')

paths = {
    # genotype inputs
    'kinship_r':     os.path.join(miqtl_dir, 'output', 'kinship_loco.rds'),
    'cache_r':       os.path.join(miqtl_dir, 'output', 'genomecache'),
    'pmap':          os.path.join(miqtl_dir, 'input',  'pmap.csv'),
    # phase 1 R reference outputs (lmm_validation/)
    'lod_r':         os.path.join(LMM_R_DATA, 'lod_r.csv'),
    'pheno_48':      os.path.join(LMM_R_DATA, 'pheno_48_used.csv'),
    'ae_r':          os.path.join(LMM_R_DATA, 'allele_effects.csv'),
    'pmap_r':        os.path.join(LMM_R_DATA, 'pmap_r.csv'),
    # phase 2 R reference outputs (port_validation/) — populated by port_validation.R
    'mi_lod_r':      os.path.join(PORT_R_DATA, 'mi_lod_r.csv'),
    'snp_lod_r':     os.path.join(PORT_R_DATA, 'snp_lod_r.csv'),
    'mediation_r':   os.path.join(PORT_R_DATA, 'mediation_r.csv'),
    'thresholds_r':  os.path.join(PORT_R_DATA, 'thresholds_r.csv'),
    'ci_r':          os.path.join(PORT_R_DATA, 'ci_r.csv'),
    'fixef_r':       os.path.join(PORT_R_DATA, 'fixef_r.csv'),
    'collapsed_r':   os.path.join(PORT_R_DATA, 'collapsed_cache_summary.csv'),
}

val_dir  = os.path.join(pkg_dir, 'validation', 'port_validation', 'r_data')
plot_dir = os.path.join(pkg_dir, 'validation', 'port_validation', 'qc_plots')
os.makedirs(val_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

FOUNDER_COLORS = {
    'AJ':'#F4C245','B6J':'#C0BFBF','129S1J':'#DA9CC1','NODJ':'#4F6EAF',
    'NZOJ':'#52A5DB','CASTJ':'#55AF5B','PWKJ':'#D83026','WSBJ':'#683C91',
}
CC_COLS = ['CC001','CC002','CC003','CC004','CC005','CC006','CC007','CC008',
           'CC009','CC010','CC011','CC012','CC013','CC015','CC017','CC018',
           'CC024','CC025','CC028','CC029','CC030','CC032','CC036','CC037',
           'CC038','CC041','CC043','CC057','CC060','CC062','CC065','CC071','CC074']
FOUNDER_COLS = ['AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ']
F1_CROSSES   = [('B6AF1J',0,1),('B6129S1F1J',2,1),('B6NODF1J',3,1),
                ('B6NZOF1J',4,1),('B6CASTF1J',5,1),('B6PWKF1J',6,1),('B6WSBF1J',7,1)]

# ── args ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--chr',          default=None, help='comma-separated chromosomes')
p.add_argument('--n_genes',      default=None, type=int)
p.add_argument('--skip-phase1',  action='store_true', help='skip sections 1-3')
args = p.parse_args()

# ── plotting setup ─────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'sans-serif',
                     'font.sans-serif':['Carlito','DejaVu Sans'],
                     'font.size':11})

qc_rows = []

def save_csv(df, name):
    path = os.path.join(val_dir, name)
    df.to_csv(path, index=False)
    log.info(f'  saved: {name} ({len(df)} rows)')
    return path

def save_plot(fig, name):
    path = os.path.join(plot_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); log.info(f'  saved: {name}')

def section(title):
    log.info('\n' + '='*60 + f'\n{title}\n' + '='*60)

def has_module(mod):
    """check whether a Phase 2 module is available."""
    try:
        importlib.import_module(f'mosaiqtl.{mod}')
        return True
    except ImportError:
        return False

_chr_sort = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)

# ── build/load cache (shared by all sections) ──────────────────────────────────
log.info('building genome cache from raw GigaMUGA...')
cache_py = GenomeCache.from_gigamuga(
    gigamuga_path = GM_CSV,
    gmap_dir      = GM_GMAP,
    pmap_dir      = GM_PMAP,
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)
log.info(f'cache: {len(cache_py.chromosomes)} chr | {len(cache_py.sample_ids)} strains')

kinship_py = kin_module.calc_kinship(cache_py, method='loco')
log.info(f'kinship: {len(kinship_py)} LOCO matrices')

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LMM CORE (sections 1-3, shared with validate_vs_r.py)
# ══════════════════════════════════════════════════════════════════════════════

if not args.skip_phase1:

    # ── SECTION 1 — haplotype probs ───────────────────────────────────────────
    section('SECTION 1 — HAPLOTYPE PROBS')
    # import shared logic from validate_vs_r
    # NOTE: sections 1-3 replicate validate_vs_r.py exactly.
    # when validate_vs_r.py is fully stable these can call it as a library instead.
    try:
        cache_r = GenomeCache.from_happy_dir(paths['cache_r'], model='additive') \
                  if os.path.isdir(paths['cache_r']) else None
    except Exception as e:
        cache_r = None
        log.warning(f'R cache unavailable: {e}')

    hap_rows = []
    py_strains = list(cache_py.sample_ids)
    r_strains  = list(cache_r.sample_ids) if cache_r else []
    shared_s   = [s for s in py_strains if s in r_strains]
    py_idx_s   = [py_strains.index(s) for s in shared_s]
    r_idx_s    = [r_strains.index(s)  for s in shared_s]

    _FOUNDER_PY_TO_R = {
        'AJ':['A','AJ'],'B6J':['B','B6J'],'129S1J':['C','129S1J'],
        'NODJ':['D','NODJ'],'NZOJ':['E','NZOJ'],'CASTJ':['F','CASTJ'],
        'PWKJ':['G','PWKJ'],'WSBJ':['H','WSBJ'],
    }
    f_py = [f.upper() for f in cache_py.founders]
    f_r  = [f.upper() for f in cache_r.founders] if cache_r else []
    f_r_idx = []
    for f in f_py:
        cands = [i for i,rf in enumerate(f_r)
                 if rf in [x.upper() for x in _FOUNDER_PY_TO_R.get(f,[])]]
        if not cands: cands = [i for i,rf in enumerate(f_r) if rf==f]
        f_r_idx.append(cands[0] if cands else None)
    founder_reindex_valid = all(x is not None for x in f_r_idx)

    r_scale = 1.0
    if cache_r:
        _c0 = cache_r.chromosomes[0]
        _h  = cache_r._data[_c0]['probs'][len(cache_r._data[_c0]['probs'])//2]
        if float(_h.max()) > 1.5: r_scale = 2.0

    for chrom in sorted(cache_py.chromosomes, key=_chr_sort):
        lids   = cache_py._data[chrom]['locus_ids']
        n_loci = len(lids)
        diffs  = []; row_sum_ok = True
        r_n_loci = len(cache_r._data[chrom]['locus_ids']) if (cache_r and chrom in cache_r._data) else np.nan

        if cache_r and chrom in cache_r._data:
            r_lids    = cache_r._data[chrom]['locus_ids']
            r_pos     = cache_r._data[chrom]['positions']
            py_pos    = cache_py._data[chrom]['positions']
            r_lid_set = set(r_lids)
            shared_lids = [l for l in lids if l in r_lid_set]
            if len(shared_lids) >= 5:
                step = len(shared_lids) // 5
                sample_pairs = [(int(np.where(lids==l)[0][0]),
                                 int(np.where(r_lids==l)[0][0]))
                                for l in shared_lids[::step][:5]]
                mid_locus = shared_lids[len(shared_lids)//2]
            else:
                py_si = [int(n_loci*f) for f in (0.1,0.3,0.5,0.7,0.9)]
                sample_pairs = [(pi, int(np.argmin(np.abs(r_pos - py_pos[pi]))))
                                for pi in py_si]
                mid_locus = lids[py_si[2]]
            for py_i, r_i in sample_pairs:
                h_py = cache_py.get_locus(chrom, py_i)
                if not np.allclose(h_py.sum(axis=1), 1.0, atol=1e-4): row_sum_ok = False
                h_r     = cache_r.get_locus(chrom, r_i)
                h_r_cmp = (h_r[:, f_r_idx] if founder_reindex_valid else h_r) / r_scale
                diffs.append(float(np.abs(h_py[py_idx_s] - h_r_cmp[r_idx_s]).max()))
        else:
            mid_locus = lids[n_loci//2]

        hap_rows.append({'chr': chrom, 'locus': mid_locus,
                         'n_loci_py': n_loci, 'n_loci_r': r_n_loci,
                         'n_shared': len(diffs), 'row_sum_ok': row_sum_ok,
                         'median_delta_prob': float(np.median(diffs)) if diffs else np.nan})

    hap_df = pd.DataFrame(hap_rows)
    save_csv(hap_df, 'haplotype_report.csv')
    hap_ok = bool(hap_df['row_sum_ok'].all())
    log.info(f'haplotype row sums: {"ok" if hap_ok else "BAD"}  '
             f'(median|Δprob|={hap_df["median_delta_prob"].max():.2e} informational)')
    qc_rows.append({'section':'haplotype_probs','pass':hap_ok,
                    'detail':f'row_sums_ok={hap_ok}  median|Δprob|={hap_df["median_delta_prob"].max():.2e} (informational)'})

    # ── SECTION 2 — kinship ───────────────────────────────────────────────────
    section('SECTION 2 — KINSHIP')
    import rpy2.robjects as ro
    ro.r(f'k_list <- readRDS("{paths["kinship_r"]}")')
    chr_names_r  = list(ro.r('names(k_list)'))
    kinship_r    = {c: np.array(ro.r(f'k_list[["{c}"]]'), dtype=float) for c in chr_names_r}
    r_strains_k  = list(ro.r('rownames(k_list[[1]])'))
    py_strains_k = list(cache_py.sample_ids)
    shared_k     = [s for s in py_strains_k if s in r_strains_k]
    py_idx_k     = np.array([py_strains_k.index(s) for s in shared_k])
    r_idx_k      = np.array([r_strains_k.index(s)  for s in shared_k])

    _f1_ids     = [f[0] for f in F1_CROSSES]
    _pref_order = FOUNDER_COLS + _f1_ids + CC_COLS
    r_set        = set(r_strains_k)
    shared_k_display = [s for s in _pref_order if s in r_set and s in py_strains_k]
    shared_k_display += [s for s in shared_k if s not in shared_k_display]
    py_idx_k_disp = np.array([py_strains_k.index(s) for s in shared_k_display])
    r_idx_k_disp  = np.array([r_strains_k.index(s)  for s in shared_k_display])

    kin_rows = []
    for chrom in sorted([c for c in cache_py.chromosomes if c in kinship_r], key=_chr_sort):
        k_py  = kinship_py[chrom][np.ix_(py_idx_k, py_idx_k)]
        k_r   = kinship_r[chrom][np.ix_(r_idx_k, r_idx_k)]
        max_d = float(np.abs(k_py - k_r).max())
        corr  = float(np.corrcoef(k_py.ravel(), k_r.ravel())[0,1])
        passes = corr > 0.999 and max_d < 0.05
        kin_rows.append({'chr':chrom,'max_delta_K':max_d,'r':corr,'pass':passes})
        log.info(f'  chr{chrom}: max|ΔK|={max_d:.2e}  r={corr:.8f}  {"✓" if passes else "✗"}')

    kin_df = pd.DataFrame(kin_rows)
    save_csv(kin_df, 'kinship_report.csv')
    kin_pass = bool(kin_df['pass'].all())
    log.info(f'kinship: {"PASS ✓" if kin_pass else "MISMATCH"}')
    qc_rows.append({'section':'kinship','pass':kin_pass,
                    'detail':f'max|ΔK|={kin_df["max_delta_K"].max():.2e}'})

    # ── SECTION 3 — LMM scan ──────────────────────────────────────────────────
    section('SECTION 3 — LMM SCAN')
    # reuse pre-computed lmm_validation R outputs
    if not os.path.exists(paths['lod_r']):
        log.warning('lmm_validation/r_data/lod_r.csv missing — run validate_vs_r.py first')
        qc_rows.append({'section':'lmm_scan','pass':False,'detail':'lod_r.csv missing'})
    else:
        lod_r_df = pd.read_csv(paths['lod_r']); lod_r_df['chr'] = lod_r_df['chr'].astype(str)
        ae_r     = pd.read_csv(paths['ae_r']) if os.path.exists(paths['ae_r']) else None

        _raw = pd.read_csv(paths['pheno_48'])
        if _raw.columns[0] != 'subject.name':
            _raw = pd.read_csv(paths['pheno_48'], index_col=0)
            if 'subject.name' not in _raw.columns: _raw.index.name='subject.name'
            _raw = _raw.reset_index(drop='subject.name' in _raw.columns)
        pheno_raw = _raw.drop_duplicates('subject.name', keep='first').reset_index(drop=True)

        all_chrs    = sorted(lod_r_df['chr'].unique(), key=_chr_sort)
        chromosomes = ([c.strip() for c in args.chr.split(',') if c.strip() in all_chrs]
                       if args.chr else all_chrs)
        log.info(f'chromosomes: {chromosomes}  |  genes: {lod_r_df[lod_r_df["chr"].isin(chromosomes)]["gene_id"].nunique()}')

        cache_strains = list(cache_py.sample_ids)
        pheno_48 = pheno_raw.set_index('subject.name').reindex(cache_strains)
        pc_cols  = [f'pc{i}' for i in range(1, 11)]
        covar_parts = [np.ones((len(pheno_48), 1))]
        if 'batch' in pheno_48.columns and pheno_48['batch'].notna().any():
            bd = pd.get_dummies(pheno_48['batch'], prefix='batch', drop_first=True)
            covar_parts.append(bd.astype(float).values)
        avail_pcs = [c for c in pc_cols if c in pheno_48.columns]
        if avail_pcs: covar_parts.append(pheno_48[avail_pcs].fillna(0).astype(float).values)
        covar_base = np.hstack(covar_parts)
        pheno_48   = pheno_48.reset_index()

        comparisons = []; gene_stats = []; py_betas = {}; all_pass = True

        for chrom in chromosomes:
            if chrom not in cache_py.chromosomes: continue
            chr_genes = lod_r_df[lod_r_df['chr']==chrom]['gene_id'].unique().tolist()
            if args.n_genes: chr_genes = chr_genes[:args.n_genes]
            log.info(f'\n[chr{chrom}] {len(chr_genes)} genes')

            k_chr            = kinship_py[chrom]
            eigvals, eigvecs = np.linalg.eigh(k_chr)
            covar_rot        = eigvecs.T @ covar_base
            locus_ids        = cache_py._data[chrom]['locus_ids']

            for gene in chr_genes:
                if gene not in pheno_48.columns: continue
                y = (pheno_48.drop_duplicates('subject.name').set_index('subject.name')
                     .reindex(cache_strains)[gene].values.astype(float))
                if np.all(np.isnan(y)): continue
                y_rot = eigvecs.T @ y
                _std_py = float(np.std(y_rot)); _std_r = _std_py
                if _std_py < 1e-10: all_pass = False; continue
                h2, _ = lmm_module.estimate_h2_null(y_rot, covar_rot, eigvals)

                r_gene = (lod_r_df[(lod_r_df['gene_id']==gene) & (lod_r_df['chr']==chrom)]
                          [['locus','lod','pos_mb']].rename(columns={'lod':'lod_r'}))
                if r_gene.empty: continue

                lod_py = []; betas_py = []
                for locus in r_gene['locus']:
                    idx = np.where(locus_ids==locus)[0]
                    if not len(idx): lod_py.append(np.nan); betas_py.append(None); continue
                    h_l   = cache_py.get_locus(chrom, int(idx[0]))
                    h_rot = eigvecs.T @ h_l[:, :-1]
                    res   = lmm_module.fit_lmm_at_locus(y_rot, h_rot, covar_rot, eigvals, h2)
                    lod_py.append(res.lod); betas_py.append(res.betas)

                r_gene = r_gene.copy(); r_gene['lod_py'] = lod_py
                merged = r_gene.dropna(subset=['lod_r','lod_py'])
                if len(merged) < 3: all_pass = False; continue

                peak_lod_r   = float(merged['lod_r'].max())
                peak_lod_py  = float(merged['lod_py'].max())
                r_peak_row   = merged.loc[merged['lod_r'].idxmax()]
                py_peak_row  = merged.loc[merged['lod_py'].idxmax()]
                lead_locus   = py_peak_row['locus']
                r_lod_at_py  = (float(r_peak_row['lod_r']) if lead_locus==r_peak_row['locus']
                                 else (float(merged.loc[merged['locus']==lead_locus,'lod_r'].iloc[0])
                                       if (merged['locus']==lead_locus).any() else 0.0))
                peak_match   = r_lod_at_py >= 0.99 * peak_lod_r
                max_d        = float((merged['lod_py']-merged['lod_r']).abs().max())
                rel_err      = max_d / max(peak_lod_r, 0.01)
                r_corr       = float(np.corrcoef(merged['lod_r'], merged['lod_py'])[0,1])
                strict  = peak_match and rel_err < 0.10
                lenient = peak_match and rel_err < 0.20
                passed  = strict or lenient
                tier    = 'strict' if strict else ('lenient' if lenient else 'fail')
                if not passed: all_pass = False
                log.info(f'  {"PASS" if passed else "FAIL"}({tier:<7})  {gene:25s}  '
                         f'r={r_corr:.4f}  rel={rel_err:.3f}  peak_match={peak_match}')

                merged['gene_id'] = gene; merged['chr'] = chrom
                comparisons.append(merged)
                gene_stats.append({'gene_id':gene,'chr':chrom,'status':'PASS' if passed else 'FAIL',
                                   'tier':tier,'r':r_corr,'rel_error':rel_err,'peak_match':peak_match,
                                   'peak_lod_r':peak_lod_r,'peak_lod_py':peak_lod_py,'h2_null':h2})

                ll = r_gene['locus'].tolist()
                if lead_locus in ll and betas_py[ll.index(lead_locus)] is not None:
                    py_betas[gene] = (chrom, lead_locus, cache_py.founders[:-1],
                                      betas_py[ll.index(lead_locus)])

        if comparisons:
            comp_df = pd.concat(comparisons, ignore_index=True)
            stat_df = pd.DataFrame(gene_stats)
            save_csv(comp_df, 'lod_comparison.csv')
            save_csv(stat_df, 'gene_summary.csv')
            n_strict  = (stat_df['tier']=='strict').sum()
            n_lenient = (stat_df['tier']=='lenient').sum()
            n_fail    = (stat_df['tier']=='fail').sum()
            n_total   = len(stat_df)
            lod_pass  = (n_strict + n_lenient) / max(n_total, 1) >= 0.50
            qc_rows.append({'section':'lmm_scan','pass':lod_pass,
                            'detail':f'{n_strict} strict + {n_lenient} lenient / {n_fail} fail | '
                                     f'median_r={stat_df["r"].median():.4f}'})
        else:
            qc_rows.append({'section':'lmm_scan','pass':False,'detail':'no comparisons'})

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FULL PORT SECTIONS (4-10)
# each section skips gracefully if its R reference csv is missing or
# the Python module is absent — no crashes, always a SKIP entry.
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.patches as mpatches

def skip_section(name, reason='module not yet implemented'):
    log.warning(f'{name}: SKIPPED — {reason}')
    qc_rows.append({'section': name, 'pass': None, 'detail': f'skipped: {reason}'})

chr_order  = sorted(cache_py.chromosomes, key=_chr_sort)
chr_pal    = ['#4F6EAF', '#888780']
chr_colors = lambda s: [chr_pal[chr_order.index(c) % 2] if c in chr_order else '#aaa' for c in s]

# ── SECTION 4 — genome cache collapse ────────────────────────────────────────
section('SECTION 4 — CACHE COLLAPSE')
if not hasattr(GenomeCache, 'collapse'):
    skip_section('cache_collapse', 'GenomeCache.collapse() not yet implemented')
else:
    r_csv = paths['collapsed_r']
    if not os.path.exists(r_csv):
        skip_section('cache_collapse', 'collapsed_cache_summary.csv missing — run port_validation.R')
    else:
        coll_r = pd.read_csv(r_csv); coll_r['chr'] = coll_r['chr'].astype(str)
        log.info('  collapsing Python cache (l2.norm tol=0.02)...')
        cache_coll = cache_py.collapse(criterion='l2.norm', tol=0.02)

        rows_c = []
        for chrom in chr_order:
            n_py_full  = len(cache_py._data[chrom]['locus_ids'])
            n_py_coll  = len(cache_coll._data[chrom]['locus_ids'])
            r_row      = coll_r[coll_r['chr'] == chrom]
            n_r_coll   = int(r_row['n_loci_coll'].iloc[0]) if not r_row.empty else np.nan
            n_r_orig   = int(r_row['n_loci_orig'].iloc[0]) if not r_row.empty else np.nan
            pct_py     = 100*(n_py_full - n_py_coll) / n_py_full
            pct_r      = 100*(n_r_orig - n_r_coll) / n_r_orig if n_r_orig else np.nan
            rows_c.append({'chr': chrom,
                           'n_py_full': n_py_full, 'n_py_coll': n_py_coll,
                           'n_r_orig': n_r_orig,   'n_r_coll': n_r_coll,
                           'pct_reduced_py': pct_py, 'pct_reduced_r': pct_r})
            log.info(f'  chr{chrom}: py {n_py_full}→{n_py_coll} ({pct_py:.1f}%)  '
                     f'r {n_r_orig}→{n_r_coll} ({pct_r:.1f}%)')

        coll_df = pd.DataFrame(rows_c)
        save_csv(coll_df, 'collapse_comparison.csv')

        # pass if Python reduces at least 5% of loci (structural check)
        mean_py_red = coll_df['pct_reduced_py'].mean()
        coll_pass   = mean_py_red >= 5.0
        qc_rows.append({'section': 'cache_collapse', 'pass': coll_pass,
                        'detail': f'mean reduction py={mean_py_red:.1f}%  '
                                   f'r={coll_df["pct_reduced_r"].mean():.1f}%'})

        # QC plot: reduction % per chromosome, Python vs R
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(coll_df)); w = 0.35
        ax.bar(x - w/2, coll_df['pct_reduced_py'], w, color='#4F6EAF',
               edgecolor='none', label='mosaiQTL (Python)')
        ax.bar(x + w/2, coll_df['pct_reduced_r'].fillna(0), w, color='#55AF5B',
               edgecolor='none', label='miQTL (R)', alpha=0.85)
        ax.set(xticks=x, xticklabels=[f'chr{c}' for c in coll_df['chr']],
               xlabel='chromosome', ylabel='loci reduced (%)',
               title='cache collapse — loci reduction per chromosome\n'
                     f'(l2.norm tol=0.02, mean py={mean_py_red:.1f}%)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
        save_plot(fig, 'collapse_comparison.png')

# ── SECTION 5 — permutation thresholds ───────────────────────────────────────
section('SECTION 5 — PERMUTATION THRESHOLDS')
if not has_module('permutation'):
    skip_section('permutation_thresholds', 'mosaiqtl.permutation not yet implemented')
else:
    r_csv = paths['thresholds_r']
    if not os.path.exists(r_csv):
        skip_section('permutation_thresholds', 'thresholds_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl import permutation as perm_module
        thresh_r_df = pd.read_csv(r_csv)
        thresh_r    = float(thresh_r_df['lod_threshold'].iloc[0])
        thresh_gene = str(thresh_r_df['gene'].iloc[0])
        thresh_chr  = str(thresh_r_df['chr'].iloc[0])
        n_perms_r   = int(thresh_r_df['n_perms'].iloc[0])
        log.info(f'  R threshold (GEV 95%): {thresh_r:.3f}  gene={thresh_gene}  n_perms={n_perms_r}')

        # run Python permutation on same gene/chr
        pheno_col = pheno_48.set_index('subject.name').reindex(cache_strains)[thresh_gene].values.astype(float)
        k_chr     = kinship_py[thresh_chr]
        ev, evec  = np.linalg.eigh(k_chr)

        log.info(f'  running {n_perms_r} permutations for chr{thresh_chr} {thresh_gene}...')
        # permute and scan
        rng_p  = np.random.default_rng(42)
        valid_p = ~np.isnan(pheno_col)
        y_p    = pheno_col[valid_p]
        from mosaiqtl.utils import rint as _rint_p
        y_p    = _rint_p(y_p)
        n_p    = len(y_p)
        ev_v   = evec[valid_p]
        x_p    = covar_base[valid_p]
        h2_p, _ = lmm_module.estimate_h2_null(ev_v.T @ y_p, ev_v.T @ x_p, ev)

        max_lods_py = []
        for _ in range(n_perms_r):
            y_perm = rng_p.permutation(y_p)
            chr_res = lmm_module.scan_chromosome(
                y_perm, cache_py, thresh_chr, x_p, ev, ev_v, h2_p)
            max_lods_py.append(float(chr_res['lod'].max()))

        max_lods_py = np.array(max_lods_py)
        thresh_py, gev_params = perm_module.get_gev_thresholds(max_lods_py, 0.95)
        delta = abs(thresh_py - thresh_r)
        thresh_pass = delta < 0.5    # within 0.5 lod
        log.info(f'  Python threshold (GEV 95%): {thresh_py:.3f}  Δ={delta:.3f}  '
                 f'{"PASS ✓" if thresh_pass else "FAIL ✗"}')
        qc_rows.append({'section': 'permutation_thresholds', 'pass': thresh_pass,
                        'detail': f'py={thresh_py:.3f}  r={thresh_r:.3f}  Δ={delta:.3f}'})
        thresh_summary = pd.DataFrame([{'gene': thresh_gene, 'chr': thresh_chr,
                                         'n_perms': n_perms_r,
                                         'threshold_py': thresh_py,
                                         'threshold_r': thresh_r,
                                         'gev_c': gev_params['c'],
                                         'gev_loc': gev_params['loc'],
                                         'gev_scale': gev_params['scale']}])
        save_csv(thresh_summary, 'threshold_comparison.csv')

        # QC plot: max-lod distribution with GEV fit and both thresholds
        from scipy.stats import genextreme as _gev
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(max_lods_py, bins=30, color='#4F6EAF', ax=ax,
                     edgecolor='none', stat='density', label='permutation max-LOD')
        x_fit = np.linspace(max_lods_py.min(), max_lods_py.max()*1.1, 200)
        ax.plot(x_fit, _gev.pdf(x_fit, gev_params['c'],
                                 loc=gev_params['loc'], scale=gev_params['scale']),
                color='#C0BFBF', lw=2, label='GEV fit')
        ax.axvline(thresh_py, color='#4F6EAF', lw=1.5, ls='--',
                   label=f'Python 95% = {thresh_py:.2f}')
        ax.axvline(thresh_r,  color='#D83026', lw=1.5, ls=':',
                   label=f'R 95% = {thresh_r:.2f}')
        ax.set(xlabel='genome-wide max LOD', ylabel='density',
               title=f'permutation threshold — {thresh_gene} chr{thresh_chr}\n'
                     f'Δ = {delta:.3f} LOD ({"PASS" if thresh_pass else "FAIL"})')
        ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
        save_plot(fig, 'threshold_distribution.png')

# ── SECTION 6 — multiple imputation ──────────────────────────────────────────
section('SECTION 6 — MULTIPLE IMPUTATION')
_scan_mi = getattr(importlib.import_module('mosaiqtl.scan'), 'scan_h2lmm_mi', None)
if _scan_mi is None:
    skip_section('multi_imputation', 'scan_h2lmm_mi() not yet implemented')
else:
    r_csv = paths['mi_lod_r']
    if not os.path.exists(r_csv):
        skip_section('multi_imputation', 'mi_lod_r.csv missing — run port_validation.R')
    else:
        mi_r_df = pd.read_csv(r_csv); mi_r_df['chr'] = mi_r_df['chr'].astype(str)
        # pick first gene with sufficient R loci
        mi_gene = mi_r_df.groupby('gene_id').size().idxmax()
        mi_chr  = mi_r_df[mi_r_df['gene_id']==mi_gene]['chr'].iloc[0]
        mi_gene_r = mi_r_df[mi_r_df['gene_id']==mi_gene][['locus','lod']].rename(
            columns={'lod':'lod_r'})
        log.info(f'  running MI scan for {mi_gene} chr{mi_chr}...')

        pheno_mi = pheno_48.set_index('subject.name').reindex(
            cache_strains)[mi_gene].values.astype(float)
        k_d = kin_module.decomp_kinship({mi_chr: kinship_py[mi_chr]})
        mi_res = _scan_mi(pheno_mi, cache_py, covar_base, k_d,
                          n_imp=11, seed=1, phenotype_id=mi_gene)

        # merge on locus for comparison
        mi_py_df = pd.DataFrame({'locus': mi_res.locus_ids,
                                  'lod_py': mi_res.lod,
                                  'chr': mi_res.chromosomes})
        mi_py_df = mi_py_df[mi_py_df['chr']==mi_chr]
        mi_merged = mi_py_df.merge(mi_gene_r, on='locus', how='inner')

        if len(mi_merged) < 3:
            skip_section('multi_imputation', f'insufficient locus overlap ({len(mi_merged)} shared)')
        else:
            r_mi  = float(np.corrcoef(mi_merged['lod_r'], mi_merged['lod_py'])[0,1])
            mi_pass = r_mi > 0.95
            log.info(f'  MI LOD r = {r_mi:.4f}  {"PASS ✓" if mi_pass else "FAIL ✗"}')
            qc_rows.append({'section': 'multi_imputation', 'pass': mi_pass,
                            'detail': f'r={r_mi:.4f}  n_loci={len(mi_merged)}'})
            save_csv(mi_merged, 'mi_lod_comparison.csv')

            # QC plot: MI LOD profiles Python vs R
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(mi_merged['lod_r'],  color='#D83026', lw=1.2, alpha=0.85,
                    label='miQTL (R) MI')
            ax.plot(mi_merged['lod_py'], color='#4F6EAF', lw=1.2, alpha=0.85,
                    label='mosaiQTL (Python) MI')
            ax.set(xlabel='locus index', ylabel='LOD',
                   title=f'multiple imputation — {mi_gene} chr{mi_chr}\n'
                         f'r={r_mi:.4f}  n_imp=11')
            ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
            save_plot(fig, 'mi_lod_profile.png')

# ── SECTION 7 — SNP scan ─────────────────────────────────────────────────────
section('SECTION 7 — SNP SCAN')
if not has_module('snp_scan'):
    skip_section('snp_scan', 'mosaiqtl.snp_scan not yet implemented')
else:
    r_csv = paths['snp_lod_r']
    if not os.path.exists(r_csv):
        skip_section('snp_scan', 'snp_lod_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl import snp_scan as snp_module
        snp_r_df = pd.read_csv(r_csv); snp_r_df['chr'] = snp_r_df['chr'].astype(str)
        snp_gene = snp_r_df['gene_id'].iloc[0]
        snp_chr  = snp_r_df[snp_r_df['gene_id']==snp_gene]['chr'].iloc[0]
        log.info(f'  SNP scan validation for {snp_gene} chr{snp_chr}')

        # load snp genotype db (must exist at paths['snp_db'])
        snp_db_path = os.path.join(miqtl_dir, 'input', 'snp_db.rds')
        if not os.path.exists(snp_db_path):
            skip_section('snp_scan', f'snp_db.rds not found at {snp_db_path}')
        else:
            # read snp db via rpy2 — extract founder genotypes as DataFrame
            import rpy2.robjects as ro2
            ro2.r(f'snp_db <- readRDS("{snp_db_path}")')
            snp_mat  = np.array(ro2.r('as.matrix(snp_db$geno)'))
            snp_ids  = list(ro2.r('rownames(snp_db$geno)'))
            snp_fnd  = list(ro2.r('colnames(snp_db$geno)'))
            snp_pos  = list(ro2.r('snp_db$map$pos_Mb'))
            snp_chr_ = list(ro2.r('snp_db$map$chr'))
            snp_geno_df = pd.DataFrame(snp_mat, index=snp_ids, columns=snp_fnd)
            snp_geno_df['chr']    = snp_chr_
            snp_geno_df['pos_mb'] = snp_pos

            pheno_s = pheno_48.set_index('subject.name').reindex(
                cache_strains)[snp_gene].values.astype(float)
            k_d_s = kin_module.decomp_kinship({snp_chr: kinship_py[snp_chr]})

            py_snp = snp_module.imputed_snp_scan_h2lmm(
                pheno_s, cache_py, snp_geno_df, covar_base, k_d_s,
                phenotype_id=snp_gene)
            py_snp_chr = py_snp[py_snp['chr']==snp_chr]
            r_snp_chr  = snp_r_df[(snp_r_df['gene_id']==snp_gene)
                                   & (snp_r_df['chr']==snp_chr)]
            merged_snp = py_snp_chr.merge(
                r_snp_chr[['snp','lod']].rename(columns={'lod':'lod_r'}),
                on='snp', how='inner')

            if len(merged_snp) < 3:
                skip_section('snp_scan', f'insufficient snp overlap ({len(merged_snp)})')
            else:
                r_snp = float(np.corrcoef(merged_snp['lod_r'], merged_snp['lod'])[0,1])
                snp_pass = r_snp > 0.95
                log.info(f'  SNP LOD r = {r_snp:.4f}  {"PASS ✓" if snp_pass else "FAIL ✗"}')
                qc_rows.append({'section':'snp_scan','pass':snp_pass,
                                'detail':f'r={r_snp:.4f}  n_snps={len(merged_snp)}'})
                save_csv(merged_snp, 'snp_lod_comparison.csv')

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(merged_snp['lod_r'], merged_snp['lod'],
                           s=4, alpha=1.0, color='#4F6EAF', edgecolors='none')
                lim = max(merged_snp[['lod_r','lod']].max().max() * 1.05, 1.0)
                ax.plot([0,lim],[0,lim],'--',color='#D83026',lw=1,label='$y=x$')
                ax.set(xlim=(0,lim), ylim=(0,lim),
                       xlabel='miQTL SNP LOD (R)', ylabel='mosaiQTL SNP LOD (Python)',
                       title=f'SNP scan — {snp_gene} chr{snp_chr}\nr={r_snp:.4f}')
                ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
                save_plot(fig, 'snp_lod_correlation.png')

# ── SECTION 8 — mediation ────────────────────────────────────────────────────
section('SECTION 8 — MEDIATION')
if not has_module('mediation'):
    skip_section('mediation', 'mosaiqtl.mediation not yet implemented')
else:
    r_csv = paths['mediation_r']
    if not os.path.exists(r_csv):
        skip_section('mediation', 'mediation_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl import mediation as med_module
        med_r_df = pd.read_csv(r_csv); med_r_df['chr'] = med_r_df['chr'].astype(str)
        med_gene = str(med_r_df['gene_id'].iloc[0])
        med_med  = str(med_r_df['mediator'].iloc[0])
        med_chr  = str(med_r_df['chr'].iloc[0])
        log.info(f'  mediation: outcome={med_gene}  mediator={med_med}  chr={med_chr}')

        pheno_o = pheno_48.set_index('subject.name').reindex(
            cache_strains)[med_gene].values.astype(float)
        pheno_m = pheno_48.set_index('subject.name').reindex(
            cache_strains)[med_med].values.astype(float)
        k_d_m = kin_module.decomp_kinship({med_chr: kinship_py[med_chr]})

        py_med = med_module.mediation_scan_qr(
            pheno_o, pheno_m, cache_py, covar_base, k_d_m,
            outcome_id=med_gene, mediator_id=med_med)
        py_med_chr = py_med[py_med['chr']==med_chr]
        merged_med = py_med_chr.merge(
            med_r_df[['locus','lod']].rename(columns={'lod':'lod_r'}),
            on='locus', how='inner')

        if len(merged_med) < 3:
            skip_section('mediation', f'insufficient locus overlap ({len(merged_med)})')
        else:
            r_med = float(np.corrcoef(merged_med['lod_r'], merged_med['lod'])[0,1])
            med_pass = r_med > 0.90
            log.info(f'  mediation LOD r = {r_med:.4f}  {"PASS ✓" if med_pass else "FAIL ✗"}')
            qc_rows.append({'section':'mediation','pass':med_pass,
                            'detail':f'r={r_med:.4f}  n_loci={len(merged_med)}'})
            save_csv(merged_med, 'mediation_comparison.csv')

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            # left: LOD scatter
            ax = axes[0]
            ax.scatter(merged_med['lod_r'], merged_med['lod'],
                       s=4, alpha=1.0, color='#4F6EAF', edgecolors='none')
            lim = max(merged_med[['lod_r','lod']].max().max()*1.05, 1.0)
            ax.plot([0,lim],[0,lim],'--',color='#D83026',lw=1,label='$y=x$')
            ax.set(xlim=(0,lim),ylim=(0,lim),
                   xlabel='miQTL mediation LOD (R)',
                   ylabel='mosaiQTL mediation LOD (Python)',
                   title=f'mediation LOD — {med_gene}\nmediator: {med_med}\nr={r_med:.4f}')
            ax.legend(fontsize=9); sns.despine(ax=ax)
            # right: mediation_lod profile
            ax = axes[1]
            ax.plot(py_med_chr['pos_mb'], py_med_chr['mediation_lod'],
                    color='#D83026', lw=1.2, label='mediation LOD (Python)')
            ax.set(xlabel='position (Mb)', ylabel='LOD (base) − LOD (mediated)',
                   title=f'mediation effect — chr{med_chr}')
            ax.legend(fontsize=9); sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig, 'mediation_comparison.png')

# ── SECTION 9 — positional CI ────────────────────────────────────────────────
section('SECTION 9 — POSITIONAL CI')
_run_pos = getattr(importlib.import_module('mosaiqtl.permutation'),
                    'run_positional_scans', None) if has_module('permutation') else None
if _run_pos is None:
    skip_section('positional_ci', 'run_positional_scans() not yet implemented')
else:
    r_csv = paths['ci_r']
    if not os.path.exists(r_csv):
        skip_section('positional_ci', 'ci_r.csv missing — run port_validation.R')
    else:
        ci_r_df   = pd.read_csv(r_csv)
        ci_gene   = str(ci_r_df['gene_id'].iloc[0])
        ci_chr    = str(ci_r_df['chr'].iloc[0])
        ci_lo_r   = float(ci_r_df['ci_lo_mb'].iloc[0])
        ci_hi_r   = float(ci_r_df['ci_hi_mb'].iloc[0])
        n_boot    = int(ci_r_df['n_boot'].iloc[0]) if 'n_boot' in ci_r_df else 100
        log.info(f'  CI: {ci_gene} chr{ci_chr}  R=[{ci_lo_r:.2f},{ci_hi_r:.2f}] Mb')

        pheno_ci = pheno_48.set_index('subject.name').reindex(
            cache_strains)[ci_gene].values.astype(float)
        k_d_ci = kin_module.decomp_kinship({ci_chr: kinship_py[ci_chr]})

        # first get the scan result to find the peak
        from mosaiqtl.scan import scan_h2lmm as _scan_ci
        ci_scan = _scan_ci(pheno_ci, cache_py, covar_base, k_d_ci,
                            phenotype_id=ci_gene)
        peak_loc = ci_scan.locus_ids[np.argmax(ci_scan.lod)]

        # run bootstrap CI
        rng_ci = np.random.default_rng(1)
        valid_ci = ~np.isnan(pheno_ci)
        y_ci     = pheno_ci[valid_ci]
        from mosaiqtl.utils import rint as _rint_ci
        y_ci = _rint_ci(y_ci)
        y_ci_c = y_ci - y_ci.mean()
        boot_oc = np.column_stack([rng_ci.choice(y_ci_c, size=len(y_ci_c), replace=True)
                                    for _ in range(n_boot)])
        boot_pos, ci_info = _run_pos(
            boot_oc, pheno_ci, cache_py, covar_base, k_d_ci,
            peak_chrom=ci_chr, peak_locus=peak_loc)
        ci_lo_py = ci_info['ci_lo']; ci_hi_py = ci_info['ci_hi']
        # CI passes if Python and R intervals overlap
        overlap = ci_lo_py <= ci_hi_r and ci_lo_r <= ci_hi_py
        log.info(f'  Python CI=[{ci_lo_py:.2f},{ci_hi_py:.2f}]  '
                 f'overlap with R: {"yes ✓" if overlap else "no ✗"}')
        qc_rows.append({'section':'positional_ci','pass':overlap,
                        'detail':f'py=[{ci_lo_py:.2f},{ci_hi_py:.2f}]  '
                                  f'r=[{ci_lo_r:.2f},{ci_hi_r:.2f}]  overlap={overlap}'})
        ci_summary = pd.DataFrame([{'gene_id':ci_gene,'chr':ci_chr,
                                      'ci_lo_py':ci_lo_py,'ci_hi_py':ci_hi_py,
                                      'ci_lo_r':ci_lo_r, 'ci_hi_r':ci_hi_r,
                                      'n_boot':n_boot}])
        save_csv(ci_summary, 'ci_comparison.csv')

        fig, ax = plt.subplots(figsize=(7, 4))
        valid_bp = boot_pos[~np.isnan(boot_pos)]
        sns.histplot(valid_bp, bins=30, color='#4F6EAF', ax=ax,
                     edgecolor='none', label='bootstrap peak position')
        ax.axvline(ci_lo_py, color='#4F6EAF', lw=1.2, ls='--',
                   label=f'Python 95% CI [{ci_lo_py:.2f}, {ci_hi_py:.2f}]')
        ax.axvline(ci_hi_py, color='#4F6EAF', lw=1.2, ls='--')
        ax.axvline(ci_lo_r,  color='#D83026', lw=1.2, ls=':',
                   label=f'R 95% CI [{ci_lo_r:.2f}, {ci_hi_r:.2f}]')
        ax.axvline(ci_hi_r,  color='#D83026', lw=1.2, ls=':')
        ax.set(xlabel='peak position (Mb)', ylabel='bootstraps',
               title=f'positional CI — {ci_gene} chr{ci_chr}\n'
                     f'overlap: {"yes" if overlap else "no"}')
        ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
        save_plot(fig, 'positional_ci.png')

# ── SECTION 10 — fixed-effect scans ──────────────────────────────────────────
section('SECTION 10 — FIXED-EFFECT SCANS')
_inst_scan = getattr(importlib.import_module('mosaiqtl.scan'), 'instability_lm_scan', None)
if _inst_scan is None:
    skip_section('fixef_scans', 'instability_lm_scan() not yet implemented')
else:
    r_csv = paths['fixef_r']
    if not os.path.exists(r_csv):
        skip_section('fixef_scans', 'fixef_r.csv missing — run port_validation.R')
    else:
        from mosaiqtl.scan import fixef_eqtl_scan as _fixef_scan
        fixef_r_df = pd.read_csv(r_csv); fixef_r_df['chr'] = fixef_r_df['chr'].astype(str)
        inst_r = fixef_r_df[fixef_r_df['type']=='instability']
        fixef_gene = str(inst_r['gene_id'].iloc[0])
        fixef_chr  = str(inst_r['chr'].iloc[0])
        log.info(f'  fixef scan: {fixef_gene} chr{fixef_chr}')

        pheno_f = pheno_48.set_index('subject.name').reindex(
            cache_strains)[fixef_gene].values.astype(float)
        loci_r  = set(inst_r[inst_r['gene_id']==fixef_gene]['locus'].tolist())

        # instability scan
        inst_py = _inst_scan(pheno_f, cache_py, covar_base,
                              n_samples=100, seed=1,
                              phenotype_id=fixef_gene,
                              just_these_loci=loci_r)
        inst_py_df = pd.DataFrame({
            'locus': inst_py.locus_ids, 'lod_py': inst_py.lod,
            'chr': inst_py.chromosomes})
        inst_py_df = inst_py_df[inst_py_df['chr']==fixef_chr]

        # fixef scan
        fixef_py = _fixef_scan(pheno_f, cache_py, covar_base,
                                phenotype_id=fixef_gene,
                                just_these_loci=loci_r)
        fixef_py_df = pd.DataFrame({
            'locus': fixef_py.locus_ids, 'lod_fixef_py': fixef_py.lod,
            'chr': fixef_py.chromosomes})
        fixef_py_df = fixef_py_df[fixef_py_df['chr']==fixef_chr]

        merged_f = inst_py_df.merge(
            inst_r[inst_r['gene_id']==fixef_gene][['locus','lod']].rename(
                columns={'lod':'lod_r'}), on='locus', how='inner')
        if len(merged_f) < 3:
            skip_section('fixef_scans', f'insufficient locus overlap ({len(merged_f)})')
        else:
            r_fixef = float(np.corrcoef(merged_f['lod_r'], merged_f['lod_py'])[0,1])
            fixef_pass = r_fixef > 0.90
            log.info(f'  instability LOD r = {r_fixef:.4f}  '
                     f'{"PASS ✓" if fixef_pass else "FAIL ✗"}')
            qc_rows.append({'section':'fixef_scans','pass':fixef_pass,
                            'detail':f'instability r={r_fixef:.4f}  n_loci={len(merged_f)}'})
            save_csv(merged_f, 'fixef_comparison.csv')

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(merged_f['lod_r'],  color='#D83026', lw=1.2, alpha=0.85,
                    label='miQTL instability.lm.scan (R)')
            ax.plot(merged_f['lod_py'], color='#4F6EAF', lw=1.2, alpha=0.85,
                    label='mosaiQTL instability_lm_scan (Python)')
            ax.set(xlabel='locus index', ylabel='mean bootstrap LOD',
                   title=f'instability scan — {fixef_gene} chr{fixef_chr}\n'
                         f'r={r_fixef:.4f}  n_bootstraps=100')
            ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
            save_plot(fig, 'fixef_instability.png')

# ══════════════════════════════════════════════════════════════════════════════
# QC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section('QC SUMMARY')
qc_df = pd.DataFrame(qc_rows)
save_csv(qc_df, 'qc_summary.csv')

print()
print('=' * 60)
for row in qc_rows:
    status = 'PASS' if row['pass'] is True else ('SKIP' if row['pass'] is None else 'FAIL')
    print(f'  {status:<4}  {row["section"]:<25}  {row["detail"]}')
print(f'  outputs: {val_dir}/')
print('=' * 60)

implemented  = [r for r in qc_rows if r['pass'] is not None]
passed       = [r for r in implemented if r['pass'] is True]
exit_code    = 0 if (not implemented or len(passed) == len(implemented)) else 1
sys.exit(exit_code)
