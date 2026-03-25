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
# each section is skipped with a warning if its Python module is not yet available.
# ══════════════════════════════════════════════════════════════════════════════

def skip_section(name, reason='module not yet implemented'):
    log.warning(f'{name}: SKIPPED — {reason}')
    qc_rows.append({'section': name, 'pass': None, 'detail': f'skipped: {reason}'})

# ── SECTION 4 — genome cache collapse ────────────────────────────────────────
section('SECTION 4 — CACHE COLLAPSE')
if not hasattr(GenomeCache, 'collapse'):
    skip_section('cache_collapse', 'GenomeCache.collapse() not yet implemented')
else:
    r_csv = paths['collapsed_r']
    if not os.path.exists(r_csv):
        skip_section('cache_collapse', 'collapsed_cache_summary.csv missing — run port_validation.R')
    else:
        # TODO: compare n_loci_py_collapsed vs n_loci_r_collapsed per chromosome
        skip_section('cache_collapse', 'comparison logic pending')

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
        # TODO: compare Python GEV threshold vs R get.gev.thresholds() at 5th percentile
        skip_section('permutation_thresholds', 'comparison logic pending')

# ── SECTION 6 — multiple imputation ──────────────────────────────────────────
section('SECTION 6 — MULTIPLE IMPUTATION')
if not has_module('scan') or not hasattr(
        importlib.import_module('mosaiqtl.scan'), 'scan_h2lmm_mi'):
    skip_section('multi_imputation', 'scan_h2lmm(use_multi_impute=True) not yet implemented')
else:
    r_csv = paths['mi_lod_r']
    if not os.path.exists(r_csv):
        skip_section('multi_imputation', 'mi_lod_r.csv missing — run port_validation.R')
    else:
        # TODO: compare MI LOD profiles vs R scan.h2lmm(use.multi.impute=TRUE, num.imp=11)
        skip_section('multi_imputation', 'comparison logic pending')

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
        # TODO: compare SNP LOD vs R imputed.snp.scan.h2lmm()
        skip_section('snp_scan', 'comparison logic pending')

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
        # TODO: compare mediation p-values vs R mediation.scan.qr()
        skip_section('mediation', 'comparison logic pending')

# ── SECTION 9 — positional CI ────────────────────────────────────────────────
section('SECTION 9 — POSITIONAL CI')
if not has_module('permutation') or not hasattr(
        importlib.import_module('mosaiqtl.permutation'), 'run_positional_scans'):
    skip_section('positional_ci', 'run_positional_scans() not yet implemented')
else:
    r_csv = paths['ci_r']
    if not os.path.exists(r_csv):
        skip_section('positional_ci', 'ci_r.csv missing — run port_validation.R')
    else:
        # TODO: compare CI boundaries vs R run.positional.scans()
        skip_section('positional_ci', 'comparison logic pending')

# ── SECTION 10 — fixed-effect scans ──────────────────────────────────────────
section('SECTION 10 — FIXED-EFFECT SCANS')
if not has_module('scan') or not hasattr(
        importlib.import_module('mosaiqtl.scan'), 'instability_lm_scan'):
    skip_section('fixef_scans', 'instability_lm_scan() / fixef_eqtl_scan() not yet implemented')
else:
    r_csv = paths['fixef_r']
    if not os.path.exists(r_csv):
        skip_section('fixef_scans', 'fixef_r.csv missing — run port_validation.R')
    else:
        # TODO: compare fixed-effect LOD vs R instability.lm.scan() / fixef.eQTL.scan()
        skip_section('fixef_scans', 'comparison logic pending')

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
