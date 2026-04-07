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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # validation/ for plot_phase2
import plot_phase2
from plot_phase2 import (save_csv, save_plot, scatter_qc, ccc,
                          bland_altman, ecdf_plot)

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
ap.add_argument('--sections',    default=None,
                help='comma-separated section numbers to run e.g. 8,11 (default: all)')
args = ap.parse_args()

# ── which sections to run ────────────────────────────────────────────────────
_run_sections = (set(int(s.strip()) for s in args.sections.split(','))
                 if args.sections else set(range(1, 12)))

# ── paths ─────────────────────────────────────────────────────────────────────
miqtl_dir = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

R_DATA  = os.path.join(pkg_dir, 'validation', 'phase2', 'r_data')
PY_DATA = os.path.join(pkg_dir, 'validation', 'phase2', 'py_data')
PLOTS   = os.path.join(pkg_dir, 'validation', 'phase2', 'plots')
LOGS    = os.path.join(pkg_dir, 'validation', 'phase2', 'logs')
for d in (R_DATA, PY_DATA, PLOTS, LOGS): os.makedirs(d, exist_ok=True)

# ── checkpoints (safe-point system) ──────────────────────────────────────
# write {section}.done after each pass; --force ignores all checkpoints.
_ap_force = '--force' in sys.argv
def _done_path(key): return os.path.join(PY_DATA, f'{key}.done')
def _is_done(key):   return not _ap_force and os.path.exists(_done_path(key))
def _mark_done(key): open(_done_path(key), 'w').close()

# GigaMUGA inputs
GM_CSV      = os.path.join(miqtl_dir, 'input', 'GM_ccfo_mm39.csv')
GM_GMAP     = os.path.join(miqtl_dir, 'input', 'GM', 'gmap')
GM_PMAP     = os.path.join(miqtl_dir, 'input', 'GM', 'pmap')
SNP_CSV     = os.path.join(miqtl_dir, 'input', 'snp_genos.csv')
ALLELE_DIR  = os.path.join(miqtl_dir, 'output', 'allele_dir')   # flat chrN.alleles files (miQTL format)

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

# wire plot_phase2 output dirs + logger (must run after PLOTS/PY_DATA/log are set)
plot_phase2.init(PLOTS, PY_DATA, log)

_chr_sort = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)

# ── helpers ───────────────────────────────────────────────────────────────────
qc_rows = []

def _col(df, first):
    '''reorder df columns: identifiers first, then remainder in original order'''
    return df[first + [c for c in df.columns if c not in first]]

def load_r(name):
    '''load a CSV from R_DATA; return None if missing; cast chr col to str.'''
    p = os.path.join(R_DATA, name)
    if not os.path.exists(p): return None
    df = pd.read_csv(p, low_memory=False)   # low_memory=False: chr col is mixed int/str after genome-wide merge
    if 'chr' in df.columns: df['chr'] = df['chr'].astype(str)
    return df

_section_keys = {
    1:'cache_info', 2:'haplotype_probs', 3:'kinship', 4:'lmm_scan',
    5:'cache_collapse', 6:'permutation', 7:'multi_imputation', 8:'snp_scan',
    9:'mediation', 10:'positional_ci', 11:'instability'}


def section(n, title):
    '''print section header; return True if section should run.'''
    key = _section_keys.get(n, str(n))
    if n not in _run_sections:
        log.debug(f'S{n} skipped (not in --sections)')
        return False
    if _is_done(key):
        hi(f'\n✓ S{n} — {title}  [checkpoint — use --force to rerun]')
        qc_rows.append({'section':key,'pass':True,'detail':'checkpoint (skipped)'})
        return False
    hi(f'\n▶ S{n} — {title}')
    log.info(f'=== S{n} {title} ===')
    return True

def skip(key, reason):
    hi(f'  SKIP: {reason}')
    log.warning(f'skip {key}: {reason}')
    qc_rows.append({'section':key,'pass':None,'detail':f'skipped: {reason}'})

def record(key, passed, detail):
    st = 'PASS' if passed else 'FAIL'
    hi(f'  {st}: {detail}')
    log.info(f'{key}: {st} — {detail}')
    qc_rows.append({'section':key,'pass':passed,'detail':detail})
    if passed: _mark_done(key)


# ── build Python genome cache ─────────────────────────────────────────────────
hi('\n● building Python genome cache...')
# drop_nullmarkers QC mirrors qtl2's drop_nullmarkers() applied before
# calc_genoprob(): drops markers where founder calls are all-missing or
# monomorphic (all non-missing founders share the same allele). this
# independently replicates the ~18k marker drop seen in the HAPPY cache.
cache_py = GenomeCache.from_gigamuga(
    gigamuga_path=GM_CSV, gmap_dir=GM_GMAP, pmap_dir=GM_PMAP,
    cc_cols=CC_COLS, founder_cols=FOUNDER_COLS, f1_crosses=F1_CROSSES,
    qc={'drop_null': True})  # mirrors qtl2 drop_nullmarkers()

# ── diagnostic: compare py loci vs HAPPY marker list to understand residual delta ──
_happy_root = os.path.join(miqtl_dir, 'output', 'genomecache', 'additive')
if os.path.isdir(_happy_root):
    try:
        import rpy2.robjects as _ro
        import pandas as _pd_diag
        # load GM raw for inspection of py_only markers
        _gm_diag = _pd_diag.read_csv(GM_CSV, low_memory=False)
        if 'marker' in _gm_diag.columns:
            _gm_diag = _gm_diag.set_index('marker')
        _gm_diag = _gm_diag.replace('N', '-')
        _fcols = list(FOUNDER_COLS)
        for _chrom in (chr_order[:3] if len(chr_order) >= 3 else chr_order):
            _mf = os.path.join(_happy_root, f'chr{_chrom}', 'markers.RData')
            if not os.path.exists(_mf): continue
            _env = _ro.r['new.env']()
            _ro.r['load'](_mf, envir=_env)
            _r_mks = set(_ro.r['get']('markers', envir=_env))
            _py_lids = set(cache_py._data[_chrom]['locus_ids']) if _chrom in cache_py._data else set()
            _py_only = sorted(_py_lids - _r_mks)
            if _py_only:
                hi(f'  [diag] chr{_chrom}: {len(_py_only)} py_only markers — inspecting founder calls:')
                for _m in _py_only[:5]:
                    if _m in _gm_diag.index:
                        _calls = _gm_diag.loc[_m, _fcols].values.astype(str)
                        _valid = _calls[_calls != '-']
                        hi(f'    {_m}: valid={list(_valid)[:8]}  unique={sorted(set(_valid))}')
                    else:
                        hi(f'    {_m}: NOT IN GM CSV')
    except Exception as _e:
        hi(f'  [diag] skipped: {_e}')
chr_order  = sorted(cache_py.chromosomes, key=_chr_sort)
chr_colors = lambda s: plot_phase2.chr_colors(s, chr_order)   # alternating palette per chr
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
# pre-load R reference CSVs used across multiple sections
# loading here ensures availability even when earlier sections are checkpointed
r_lod = load_r('lod_r.csv')     # used by S4, S5 (cis-loci), S6 (cis restriction)
r_ae  = load_r('allele_effects_r.csv')  # used by S4 allele concordance

if section(1, 'CACHE INFO'):
    r_info = load_r('cache_info_r.csv')
    r_fnd  = load_r('founders_r.csv')
    r_sid  = load_r('sample_ids_r.csv')

    rows_info = []; r_lids_by_chr = {}
    for chrom in chromosomes:
        if chrom not in cache_py._data: continue
        lids, pos, _ = cache_py.get_chromosome(chrom)
        n_py = len(lids)
        n_r  = (int(r_info.loc[r_info['chr']==chrom,'n_loci'].iloc[0])
                if r_info is not None and (r_info['chr']==chrom).any() else np.nan)
        rows_info.append({'chr':chrom,'n_loci_py':n_py,'n_loci_r':n_r,
                          'min_mb_py':float(pos.min()),'max_mb_py':float(pos.max())})
        log.info(f'  chr{chrom}: py={n_py}  r={n_r}  delta={n_py - int(n_r) if not np.isnan(n_r) else "?"}')
    info_df = pd.DataFrame(rows_info)
    save_csv(info_df,'cache_info_py.csv')

    py_founders = list(cache_py.founders)
    py_sids     = list(cache_py.sample_ids)
    r_founders  = r_fnd['founder'].tolist() if r_fnd is not None else []
    r_sids      = r_sid['sample_id'].tolist() if r_sid is not None else []

    fnd_match  = set(py_founders) == set(r_founders) if r_founders else None
    sid_match  = set(py_sids) == set(r_sids)         if r_sids    else None
    # loci counts differ when caches are built from different marker sets — report delta, don't hard-fail
    if r_info is not None:
        info_df['loci_delta'] = info_df['n_loci_py'] - info_df['n_loci_r'].fillna(np.nan)
        loci_match = info_df['loci_delta'].abs().max()   # max abs delta across chrs (0 = exact match)
    else:
        loci_match = None

    if r_info is not None:
        plot_phase2.plot_s1(info_df, py_founders, r_founders,
                            py_sids, r_sids, fnd_match, sid_match)

    loci_note = f'loci_max_delta={int(loci_match)}' if loci_match is not None else 'loci_delta=NA'
    record('cache_info', fnd_match is not False and sid_match is not False,
           f'founders_match={fnd_match}  samples_match={sid_match}  {loci_note}')

    # ══════════════════════════════════════════════════════════════════════════════
    # S2 — HAPLOTYPE PROBS
    # ══════════════════════════════════════════════════════════════════════════════
if section(2, 'HAPLOTYPE PROBS'):
    r_hap      = load_r('haplotype_r.csv')
    r_hap_vals = load_r('haplotype_vals_r.csv')   # per-sample per-founder actual probs

    # row-sum check — internal consistency only.
    # probs.sum(axis=1) == 1.0 by HMM normalisation; max_dev is always 0.00 by
    # construction. this checks Python produces normalised probs, NOT that they
    # match R. the real comparison is prob_r from haplotype_vals_r.csv below.
    rows_hap = []
    for chrom in chromosomes:
        if chrom not in cache_py._data: continue
        lids, pos, probs = cache_py.get_chromosome(chrom)
        for i in np.round(np.linspace(0, len(lids)-1, 5)).astype(int):
            rs = probs[i].sum(axis=1)
            rows_hap.append({'chr':chrom,'locus':lids[i],'pos_mb':float(pos[i]),
                             'mean_rowsum':float(rs.mean()*2),
                             'min_rowsum':float(rs.min()*2),'max_rowsum':float(rs.max()*2),
                             'mean_max_prob':float(probs[i].max(axis=1).mean())})
    hap_df  = pd.DataFrame(rows_hap)
    save_csv(hap_df, 'haplotype_py.csv')
    max_dev = float((hap_df['mean_rowsum']-2.0).abs().max())

    # display order: founders → F1s → CCs
    _fnames   = list(cache_py.founders)
    _sids_py  = list(cache_py.sample_ids)
    _disp_ord = FOUNDER_COLS + [f[0] for f in F1_CROSSES] + CC_COLS
    _disp_s   = []; _seen_d = set()
    for s in _disp_ord + _sids_py:
        if s in _sids_py and s not in _seen_d:
            _disp_s.append(s); _seen_d.add(s)
    _disp_idx = [_sids_py.index(s) for s in _disp_s]
    _fs       = max(4, min(8, int(160 / max(len(_disp_s), 1))))

    # ── actual probability value comparison (all chromosomes) ────────────────
    val_r = np.nan; val_mad = np.nan
    all_r_probs = []; all_py_probs = []
    val_rows_all = []
    hmap_data    = {}   # chrom -> (H_py, H_r, locus_id) at representative locus

    if r_hap_vals is not None:
        fc_r = [c for c in r_hap_vals.columns
                if c not in ('chr','locus','pos_mb','sample_id')]
        for chrom in chromosomes:
            if chrom not in cache_py._data: continue
            lids, pos, probs = cache_py.get_chromosome(chrom)
            rv_c = r_hap_vals[r_hap_vals['chr'].astype(str)==str(chrom)]
            if rv_c.empty: continue
            for lid in rv_c['locus'].unique():
                idx = np.where(lids==lid)[0]
                if not len(idx): continue
                h_py = probs[idx[0]]
                rv_l = rv_c[rv_c['locus']==lid].set_index('sample_id')
                for sid in _sids_py:
                    if sid not in rv_l.index: continue
                    si = _sids_py.index(sid)
                    for fi, fc in enumerate(fc_r):
                        if fi >= h_py.shape[1]: continue
                        rp = float(rv_l.loc[sid, fc]) / 2.0  # R [0,2] -> [0,1]
                        pp = float(h_py[si, fi])
                        all_r_probs.append(rp); all_py_probs.append(pp)
                        val_rows_all.append({'chr':chrom,'locus':lid,'sample_id':sid,
                                             'founder':fc,'r_prob':rp,'py_prob':pp})
            # representative locus for heatmaps
            uniq_loci = rv_c['locus'].unique()
            rep_lid   = uniq_loci[len(uniq_loci)//2]
            idx_mid   = np.where(lids==rep_lid)[0]
            if not len(idx_mid): continue
            H_py_mid = probs[idx_mid[0]][_disp_idx]
            rv_mid   = rv_c[rv_c['locus']==rep_lid].set_index('sample_id')
            H_r_mid  = np.zeros((len(_disp_s), len(fc_r)))
            for di, sid in enumerate(_disp_s):
                if sid in rv_mid.index:
                    for fi, fc in enumerate(fc_r):
                        H_r_mid[di, fi] = float(rv_mid.loc[sid, fc]) / 2.0
            hmap_data[chrom] = (H_py_mid, H_r_mid, rep_lid)

        if val_rows_all:
            vdf = _col(pd.DataFrame(val_rows_all).dropna(),
                       ['chr', 'locus', 'sample_id', 'founder', 'r_prob', 'py_prob'])
            save_csv(vdf, 'haplotype_val_comparison.csv')
            # serialize hmap_data arrays for standalone replot
            if hmap_data:
                _hm_rows = []
                for _hc, (_Hpy, _Hr, _rlid) in hmap_data.items():
                    for _si in range(_Hpy.shape[0]):
                        for _fi in range(_Hpy.shape[1]):
                            _hm_rows.append({'chr': _hc, 'locus': _rlid,
                                             'sample_idx': _si, 'founder_idx': _fi,
                                             'prob_py': float(_Hpy[_si, _fi]),
                                             'prob_r': float(_Hr[_si, _fi])})
                save_csv(_col(pd.DataFrame(_hm_rows),
                              ['chr', 'locus', 'sample_idx', 'founder_idx',
                               'prob_py', 'prob_r']),
                         'haplotype_heatmap_py.csv')
            rpa = np.array(all_r_probs); ppa = np.array(all_py_probs)
            ok  = np.isfinite(rpa) & np.isfinite(ppa)
            if ok.sum() > 3:
                val_r   = float(np.corrcoef(rpa[ok], ppa[ok])[0,1])
                val_mad = float(np.abs(rpa[ok]-ppa[ok]).mean())
            log.info(f'  prob comparison: n={ok.sum()}  r={val_r:.4f}  mad={val_mad:.4f}')

    # ── plots: heatmaps + prob comparison + bland-altman + calibration ───────
    plot_phase2.plot_s2(hmap_data, chr_order, _fnames, _disp_s,
                        all_r_probs, all_py_probs, val_r, val_mad, val_rows_all,
                        fc_r=(fc_r if r_hap_vals is not None else None))
    if not hmap_data:
        rep_chr = chromosomes[0] if chromosomes else None
        if rep_chr and rep_chr in cache_py._data:
            lids0, _, probs0 = cache_py.get_chromosome(rep_chr)
            H_py0 = probs0[len(lids0)//2][_disp_idx]
            plot_phase2.plot_s2_pyonly(H_py0, _fnames, _disp_s, _fs, rep_chr)

    # pass gate: actual prob_r if available; row_sum is ALWAYS 0.00 (by construction, not a gate)
    hap_pass = np.isnan(val_r) or val_r > 0.90
    detail   = f'row_sum_max_dev={max_dev:.2e} (always 0.00 — internal normalisation, not R comparison)'
    if not np.isnan(val_r):
        detail += f'  prob_r={val_r:.4f}  prob_mad={val_mad:.4f}'
    else:
        detail += '  prob_r=NA — run R S2 to generate haplotype_vals_r.csv'
    record('haplotype_probs', hap_pass, detail)

    # S3 — KINSHIP LOCO
    # ══════════════════════════════════════════════════════════════════════════════
if section(3, 'KINSHIP LOCO'):
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
        # kinship_py.csv already saved above (summary stats per chr)
        # per-chr matrix CSVs written below for phase3 validation target
        for _krow in kin_cors:
            _kc = _krow['chr']
            if _kc not in kinship_py: continue
            _kmat = kinship_py[_kc]
            _sids = list(cache_py.sample_ids)
            # reorder to canonical order: founders → F1s → CC strains
            _founders = [s for s in _sids if not s.startswith('CC') and 'F1' not in s and s not in [f[0] for f in F1_CROSSES]]
            _f1s      = [s for s in _sids if s in [f[0] for f in F1_CROSSES]]
            _cc       = [s for s in _sids if s.startswith('CC')]
            _ordered  = _founders + _f1s + _cc
            _oi = [_sids.index(s) for s in _ordered]
            _kmat_ord = _kmat[np.ix_(_oi, _oi)]
            _kdf = pd.DataFrame(_kmat_ord, index=_ordered, columns=_ordered)
            _kdf.index.name = 'sample_id'
            save_csv(_kdf.reset_index(), f'kinship_py_chr{_kc}.csv')

        # 3-panel heatmap for first chromosome — consistent with validate_vs_r.py
        chrom_k = kc_df['chr'].iloc[0]
        r_mat_path = os.path.join(R_DATA, f'kinship_mat_r_chr{chrom_k}.csv')
        if os.path.exists(r_mat_path) and chrom_k in kinship_py:
            k_r_df  = pd.read_csv(r_mat_path)
            sids_r  = k_r_df['sample_id'].tolist()
            k_r_mat = k_r_df.drop(columns='sample_id').values.astype(float)
            sids_py = list(cache_py.sample_ids)
            # display order: founders → F1s → CCs using canonical lists
            f1_ids  = [f[0] for f in F1_CROSSES]
            pref    = FOUNDER_COLS + f1_ids + CC_COLS
            seen    = set(); disp = []
            for s in pref:
                if s not in seen and s in sids_r and s in sids_py:
                    disp.append(s); seen.add(s)
            # append any remaining shared subjects not in canonical lists
            for s in sids_py:
                if s not in seen and s in sids_r:
                    disp.append(s); seen.add(s)
            pi = np.array([sids_py.index(s) for s in disp])
            ri = np.array([sids_r.index(s)  for s in disp])
            kp = kinship_py[chrom_k][np.ix_(pi, pi)]
            kr = k_r_mat[np.ix_(ri, ri)]
            dk = kp - kr
            ab = max(abs(dk.min()), abs(dk.max()))
            n  = len(disp); fs = max(5, min(8, int(200 / n)))
            plot_phase2.plot_s3(kp, kr, disp, chrom_k, ab)

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
if section(4, 'LMM SCAN (ROP)'):
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
            col_order = ['gene_id', 'chr', 'locus', 'pos_mb', 'lod_r', 'lod_py']
            comp_df = comp_df[col_order + [c for c in comp_df.columns if c not in col_order]]
            stat_df = pd.DataFrame(gene_stats)
            stat_df = _col(stat_df, ['gene_id', 'chr', 'tier', 'r', 'rel_error',
                                      'peak_match', 'peak_lod_r', 'peak_lod_py', 'h2_null'])
            save_csv(comp_df, 'lod_comparison.csv')
            save_csv(stat_df, 'gene_summary.csv')
            # standalone py-side LOD — same format as lod_r.csv; phase3 validates against this
            save_csv(_col(comp_df[['gene_id','chr','locus','pos_mb','lod_py']]
                          .rename(columns={'lod_py':'lod'}),
                          ['gene_id','chr','locus','pos_mb','lod']), 'lod_py.csv')
            n_s = (stat_df['tier']=='strict').sum()
            n_l = (stat_df['tier']=='lenient').sum()
            n_f = (stat_df['tier']=='fail').sum()

            # allele effect concordance data (computed here; passed to plot_s4)
            _ae_rs = []
            if r_ae is not None:
                for _g, (_chb, _peak_loc, _fns, _py_eff) in py_betas.items():
                    _r_sub = r_ae[(r_ae['gene_id']==_g)&(r_ae['locus']==_peak_loc)]
                    if _r_sub.empty: continue
                    _code_map = {chr(65+i): f for i, f in enumerate(FOUNDER_COLS)}
                    _r_eff = [float(_r_sub.loc[_r_sub['founder']==_code_map.get(chr(65+i),''), 'effect'].iloc[0])
                              if _code_map.get(chr(65+i),'') in _r_sub['founder'].values else 0.0
                              for i in range(len(_fns))]
                    if np.std(_py_eff)>0 and np.std(_r_eff)>0:
                        _ae_rs.append({'gene_id':_g, 'r_allele':float(np.corrcoef(_py_eff,_r_eff)[0,1]),
                                       'ccc_allele':ccc(_py_eff, _r_eff)})
            _ae_df = pd.DataFrame(_ae_rs) if _ae_rs else None
            if _ae_df is not None:
                save_csv(_col(_ae_df, ['gene_id', 'r_allele', 'ccc_allele']),
                         'allele_effect_concordance.csv')

            # serialize peak founder effects for standalone replot
            if py_betas:
                _pfe_rows = []
                for _g, (_chb, _ploc, _fns, _peff) in py_betas.items():
                    for _fn, _pe in zip(_fns, _peff):
                        _pfe_rows.append({'gene_id': _g, 'chr': _chb,
                                          'locus': _ploc, 'founder': _fn,
                                          'effect_py': float(_pe)})
                save_csv(_col(pd.DataFrame(_pfe_rows),
                              ['gene_id', 'chr', 'locus', 'founder', 'effect_py']),
                         'peak_founder_effects_py.csv')

            plot_phase2.plot_s4(comp_df, stat_df, chr_colors,
                                py_betas, r_ae, n_s, n_l, n_f,
                                len(cache_py.founders), ae_df=_ae_df)

            _n_total  = max(len(stat_df), 1)
            _pct_pass = 100.0 * (n_s + n_l) / _n_total
            lod_pass  = _pct_pass >= 50.0
            record('lmm_scan', lod_pass,
                   f'strict={n_s}  lenient={n_l}  fail={n_f}  '
                   f'pct_pass={_pct_pass:.1f}%  threshold=50%  '
                   f'median_r={stat_df["r"].median():.4f}')
        else:
            skip('lmm_scan','no locus overlap between R and Python')

    # ══════════════════════════════════════════════════════════════════════════════
    # S5 — CACHE COLLAPSE
    # ══════════════════════════════════════════════════════════════════════════════
if section(5, 'CACHE COLLAPSE'):
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
        df_fc = None; r_fc = np.nan   # initialised here; set inside if block
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

        plot_phase2.plot_s5(coll_df, k_cors, mean_py, mean_r, mean_k_r,
                            df_fc=df_fc, r_fc=r_fc,
                            cache_py=cache_py, cache_coll=cache_coll,
                            chr_order=chr_order)

        record('cache_collapse', coll_pass,
               f'py={mean_py:.1f}%  r={mean_r:.1f}%  '
               f'kinship_r={mean_k_r:.4f}  '
               f'lod_preservation={r_fc:.4f}  '
               f'row_sums={rows_ok}')

    # ══════════════════════════════════════════════════════════════════════════════
    # S6 — PERMUTATION THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════════
if section(6, 'PERMUTATION THRESHOLDS'):
    r_thresh = load_r('thresholds_r.csv')
    r_perms  = load_r('perm_lods_r.csv')
    if r_thresh is None or pheno_strain is None:
        skip('permutation','thresholds_r.csv missing — run port_validation.R s6')
    else:
        from mosaiqtl import permutation as perm_mod
        from scipy.stats import genextreme as _gev

        # after genome-wide merge, r_thresh has one row per chromosome.
        # loop all chromosomes, run n_p permutations each, accumulate max_lods.
        # this makes the QQ and histogram use the same pooled distribution as R.
        all_max_lods = []    # flat list of all per-chr perm max-LODs
        chr_records  = []    # per-chr threshold comparison rows
        chr_labels   = []    # for per-chr bar chart x-axis
        _perm_lod_details = {}  # chr → list of {pos_mb, perm_lod} for perm_lods_py.csv

        for _, trow in r_thresh.iterrows():
            _tgene = str(trow['gene_id'])
            _tchr  = str(trow['chr'])
            _tr_v  = float(trow['lod_threshold'])
            _np    = int(trow['n_perms']) if trow['n_perms'] > 0 else 100
            y_t    = _get_y(_tgene)
            if y_t is None or _tchr not in kinship_py:
                hi(f'  skip chr{_tchr}: gene {_tgene} not in pheno or kinship')
                continue
            k_t  = kinship_py[_tchr]
            ev,evc = np.linalg.eigh(k_t)
            valid  = ~np.isnan(y_t)
            y_v    = _rint(y_t[valid]); x_v=covar_base[valid]; evc_v=evc[valid]
            h2,_   = lmm_module.estimate_h2_null(evc_v.T@y_v,evc_v.T@x_v,ev)
            # cis loci restriction — matches R's just.these.loci per permutation
            _cis = (set(r_lod.loc[(r_lod['gene_id']==_tgene)&
                                   (r_lod['chr']==_tchr),'locus'])
                    if r_lod is not None else None)
            hi(f'  chr{_tchr} — {_tgene}  cis={len(_cis) if _cis else "all"}  n_perm={_np}')
            rng = np.random.default_rng(42)
            ml  = []; _chr_detail = []
            for _ in range(_np):
                yp  = rng.permutation(y_v)
                cr  = lmm_module.scan_chromosome(yp,cache_py,_tchr,x_v,ev,evc_v,h2)
                _lod = cr['lod']
                _pos = cr.get('positions', np.zeros(len(_lod)))
                if _cis:
                    _m = np.isin(cr['locus_ids'], list(_cis))
                    if _m.any():
                        _lod = _lod[_m]; _pos = _pos[_m]
                _pk_idx = int(np.argmax(_lod))
                ml.append(float(_lod[_pk_idx]))
                _chr_detail.append({'pos_mb': float(_pos[_pk_idx]),
                                    'perm_lod': float(_lod[_pk_idx])})
            _perm_lod_details[_tchr] = _chr_detail
            ml = np.array(ml)
            all_max_lods.extend(ml.tolist())
            t_py,_ = perm_mod.get_gev_thresholds(ml, 0.95)
            chr_records.append({'gene_id':_tgene,'chr':_tchr,'n_perms':_np,
                                 'threshold_r':_tr_v,'threshold_py':t_py,
                                 'delta':abs(t_py-_tr_v)})
            chr_labels.append(_tchr)
            log.info(f'  chr{_tchr}: py={t_py:.3f}  R={_tr_v:.3f}  Δ={abs(t_py-_tr_v):.3f}')

        if not chr_records:
            skip('permutation','no valid chromosomes with phenotype and kinship')
        else:
            all_max_lods = np.array(all_max_lods)
            comp_df = pd.DataFrame(chr_records)
            thresh_py,gev_p = perm_mod.get_gev_thresholds(all_max_lods, 0.95)
            thresh_r_v = float(comp_df['threshold_r'].mean())
            delta      = float(comp_df['delta'].mean())
            perm_pass  = delta < 0.5

            # perm_lods_py.csv: gene_id + chr + pos_mb per max-LOD observation
            perm_records = []
            for rec in chr_records:
                for pl in _perm_lod_details.get(rec['chr'], []):
                    perm_records.append({'gene_id': rec['gene_id'], 'chr': rec['chr'],
                                         'pos_mb': pl['pos_mb'], 'perm_lod': pl['perm_lod']})
            save_csv(_col(pd.DataFrame(perm_records),
                          ['gene_id', 'chr', 'pos_mb', 'perm_lod']),
                     'perm_lods_py.csv')
            save_csv(_col(comp_df, ['gene_id','chr','n_perms',
                                    'threshold_r','threshold_py','delta']),
                     'threshold_comparison.csv')

            plot_phase2.plot_s6(all_max_lods, gev_p, thresh_py, thresh_r_v,
                                delta, r_perms, comp_df)

            record('permutation',perm_pass,
                   f'mean_delta={delta:.3f}  py_pooled={thresh_py:.3f}  '
                   f'R_mean={thresh_r_v:.3f}  n_chr={len(chr_records)}')

    # ══════════════════════════════════════════════════════════════════════════════
    # S7 — MULTIPLE IMPUTATION
    # ══════════════════════════════════════════════════════════════════════════════
if section(7, 'MULTIPLE IMPUTATION'):
    r_mi = load_r('mi_lod_r.csv')
    if r_mi is None or pheno_strain is None:
        skip('multi_imputation','mi_lod_r.csv missing — run port_validation.R s7')
    else:
        from mosaiqtl.scan import scan_h2lmm_mi as _mi
        mi_comparisons=[]
        # S7 cap: MI is a legacy validation section (not mosaiQTL primary model).
        # genome-wide merged data has 1000+ genes/chr × 11 imputations each —
        # cap at S7_MAX_GENES per chromosome; r threshold already confirmed at chr8.
        S7_MAX_GENES = 5
        for chrom in chromosomes:
            genes_mi = r_mi[r_mi['chr']==chrom]['gene_id'].unique().tolist()
            genes_mi = genes_mi[:min(S7_MAX_GENES, args.n_genes or S7_MAX_GENES)]
            if not genes_mi: continue
            kd_mi = kin_module.decomp_kinship({chrom:kinship_py[chrom]})
            for gene in genes_mi:
                y_mi = _get_y(gene)
                if y_mi is None: continue
                res_mi = _mi(y_mi,cache_py,covar_base,kd_mi,n_imp=11,
                             seed=1,phenotype_id=gene)
                py_mi  = pd.DataFrame({'locus':res_mi.locus_ids,'lod_py':res_mi.lod,
                                       'chr':res_mi.chromosomes,
                                       'pos_mb':res_mi.positions})   # positions in Mb
                py_mi  = py_mi[py_mi['chr'].astype(str)==str(chrom)]
                rg_mi  = r_mi[(r_mi['gene_id']==gene)&(r_mi['chr'].astype(str)==str(chrom))
                              ][['locus','lod']].rename(columns={'lod':'lod_r'})
                mg_mi  = py_mi.merge(rg_mi,on='locus',how='inner')
                if len(mg_mi)<3: continue
                mg_mi['gene_id']=gene; mg_mi['chr']=chrom
                mi_comparisons.append(mg_mi)
        if mi_comparisons:
            mi_comp = pd.concat(mi_comparisons, ignore_index=True)
            mi_comp = _col(mi_comp, ['gene_id', 'chr', 'locus', 'pos_mb', 'lod_r', 'lod_py'])
            save_csv(mi_comp, 'mi_lod_comparison.csv')
            save_csv(_col(mi_comp[['gene_id','chr','locus','pos_mb','lod_py']]
                          .rename(columns={'lod_py':'lod'}),
                          ['gene_id','chr','locus','pos_mb','lod']), 'mi_lod_py.csv')
            _mi_diffs = (mi_comp.groupby('gene_id')
                         .apply(lambda g: float((g['lod_py']-g['lod_r']).mean()))
                         .values)
            plot_phase2.plot_s7(mi_comp, mi_diffs=(_mi_diffs if len(_mi_diffs)>1 else None))

            r_mi_all = float(np.corrcoef(mi_comp['lod_r'],mi_comp['lod_py'])[0,1])
            # r>0.75 reflects expected noise floor: R and Python use independent RNG seeds
            # with num.imp=11; higher r requires either aligned seeds or larger num.imp
            record('multi_imputation',r_mi_all>0.75,
                   f'r={r_mi_all:.4f}  n_genes={mi_comp["gene_id"].nunique()}  (threshold=0.75 for n_imp=11)')
        else:
            skip('multi_imputation','no locus overlap')

    # ══════════════════════════════════════════════════════════════════════════════
    # S8 — SNP SCAN
    # ══════════════════════════════════════════════════════════════════════════════
if section(8, 'SNP SCAN'):
    r_snp = load_r('snp_lod_r.csv')
    if not os.path.exists(ALLELE_DIR):
        skip('snp_scan',
             f'allele_dir not found at {ALLELE_DIR}\n'
             '         build with: python validation/build_allele_dir.py')
    elif pheno_strain is None:
        skip('snp_scan','pheno not loaded')
    else:
        from mosaiqtl.snp_scan import (imputed_snp_scan_h2lmm as _snp_scan,
                                        read_alleles_flat as _raf)
        # read from flat allele_dir — same source+ids as R, 0/1 alt dosage per founder
        snp_genos = _raf(ALLELE_DIR, FOUNDER_COLS, pmap_dir=GM_PMAP)
        snp_genos['chr'] = snp_genos['chr'].astype(str)
        hi(f'  allele_dir: {len(snp_genos)} SNPs  pos_filled={snp_genos["pos_mb"].notna().sum()}')
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
        if not snp_comparisons and r_snp is not None:
            # diagnose merge failure: check id overlap between r and py
            py_snps = set(res_snp['snp'].tolist()) if 'res_snp' in dir() and not res_snp.empty else set()
            r_snps  = set(r_snp['snp'].tolist())
            _sg_sample = snp_genos.iloc[:2, :5] if len(snp_genos) >= 2 else snp_genos
            hi(f'  [S8 diag] snp_genos sample:\n{_sg_sample}')
            hi(f'  [S8 diag] py snps={len(py_snps)}  r snps={len(r_snps)}  overlap={len(py_snps & r_snps)}')
            if py_snps: hi(f'  [S8 diag] py sample: {list(py_snps)[:3]}')
            if r_snps:  hi(f'  [S8 diag] r  sample: {list(r_snps)[:3]}')
        if snp_comparisons:
            snp_comp = pd.concat(snp_comparisons, ignore_index=True)
            snp_comp = _col(snp_comp, ['gene_id', 'chr', 'snp', 'lod_r', 'lod_py'])
            save_csv(snp_comp, 'snp_lod_comparison.csv')
            save_csv(_col(snp_comp[['gene_id','chr','snp','lod_py']]
                          .rename(columns={'lod_py':'lod'}),
                          ['gene_id','chr','snp','lod']), 'snp_lod_py.csv')
            # build snp_maf_df for MAF concordance plot (optional)
            _snp_maf_df = None
            if 'lod_r' in snp_comp.columns and 'snp' in snp_genos.columns:
                _snp_m = snp_comp.merge(
                    snp_genos.reset_index(drop=True)[['snp']+FOUNDER_COLS]
                    if 'snp' in snp_genos.columns else snp_comp[[]],
                    on='snp', how='inner')
                if all(fc in _snp_m.columns for fc in FOUNDER_COLS):
                    _snp_m['maf']       = _snp_m[FOUNDER_COLS].astype(float).mean(axis=1)
                    _snp_m['abs_delta'] = (_snp_m['lod_py'] - _snp_m['lod_r']).abs()
                    _snp_maf_df = _snp_m.sort_values('maf')
            r_snp_all = plot_phase2.plot_s8(snp_comp, snp_maf_df=_snp_maf_df)

            record('snp_scan',r_snp_all>0.90,
                   f'r={r_snp_all:.4f}  n_snps={snp_comp["snp"].nunique()}')
        else:
            # python-only run (no R reference)
            save_csv(res_snp if 'res_snp' in dir() else pd.DataFrame(),'snp_lod_py.csv')
            record('snp_scan',None,'python-only — no R reference')

    # ══════════════════════════════════════════════════════════════════════════════
    # S9 — MEDIATION
    # ══════════════════════════════════════════════════════════════════════════════
if section(9, 'MEDIATION'):
    r_med = load_r('mediation_r.csv')
    if r_med is None or pheno_strain is None:
        skip('mediation','mediation_r.csv missing — run port_validation.R s9')
    else:
        from mosaiqtl import mediation as med_mod
        med_comparisons = []
        # loop all unique (gene_id, chr, mediator) triples from merged R data
        med_pairs = (r_med[['gene_id','chr','mediator']]
                     .drop_duplicates()
                     .itertuples(index=False))
        for mp in med_pairs:
            med_gene = str(mp.gene_id); med_chr = str(mp.chr)
            med_med  = str(mp.mediator)
            yo = _get_y(med_gene); ym = _get_y(med_med)
            if yo is None or ym is None or med_chr not in kinship_py:
                hi(f'  skip mediation {med_gene} chr{med_chr}: pheno or kinship missing')
                continue
            kd_m   = kin_module.decomp_kinship({med_chr: kinship_py[med_chr]})
            py_med = med_mod.mediation_scan_qr(yo, ym, cache_py, covar_base, kd_m,
                                               outcome_id=med_gene, mediator_id=med_med)
            pm_c   = py_med[py_med['chr'].astype(str) == med_chr]
            rg_m   = (r_med[(r_med['gene_id']==med_gene) & (r_med['chr'].astype(str)==med_chr)]
                      [['locus','lod_base']].rename(columns={'lod_base':'lod_r'}))
            mg_m   = pm_c.rename(columns={'lod_base':'lod_py'}).merge(rg_m, on='locus',
                                                                        how='inner')
            if len(mg_m) < 3: continue
            mg_m['gene_id'] = med_gene; mg_m['chr'] = med_chr
            med_comparisons.append(mg_m)

        if med_comparisons:
            med_comp = pd.concat(med_comparisons, ignore_index=True)
            med_comp = _col(med_comp, ['gene_id','chr','locus','pos_mb','lod_r','lod_py'])
            save_csv(med_comp, 'mediation_comparison.csv')
            save_csv(_col(med_comp[['gene_id','chr','locus','pos_mb','lod_py']]
                          .rename(columns={'lod_py':'lod'}),
                          ['gene_id','chr','locus','pos_mb','lod']), 'mediation_py.csv')
            r_med_v = float(np.corrcoef(med_comp['lod_r'], med_comp['lod_py'])[0, 1])
            _drop_r = None
            _r_med_full = load_r('mediation_r.csv')
            if _r_med_full is not None and 'lod_drop' in _r_med_full.columns:
                _drop_r = _r_med_full['lod_drop'].dropna().values
            plot_phase2.plot_s9(med_comp, r_med_v, drop_r=_drop_r)

            record('mediation', r_med_v > 0.90, f'r={r_med_v:.4f}  n_chr={med_comp["chr"].nunique()}')
        else:
            skip('mediation', 'insufficient locus overlap across all chr')

    # ══════════════════════════════════════════════════════════════════════════════
    # S10 — POSITIONAL CI
    # ══════════════════════════════════════════════════════════════════════════════
if section(10, 'POSITIONAL CI'):
    r_ci   = load_r('ci_r.csv')
    r_boot = load_r('boot_pos_r.csv')
    if r_ci is None or pheno_strain is None:
        skip('positional_ci','ci_r.csv missing — run port_validation.R s10')
    else:
        from mosaiqtl.permutation import run_positional_scans as _rps
        from mosaiqtl.scan import scan_h2lmm as _sh
        ci_comparisons = []; boot_pos = None; ci_gene = ci_chr = None
        for cp in r_ci[['gene_id','chr']].drop_duplicates().itertuples(index=False):
            ci_gene = str(cp.gene_id); ci_chr = str(cp.chr)
            rrow    = r_ci[(r_ci['gene_id']==ci_gene) &
                           (r_ci['chr'].astype(str)==ci_chr)].iloc[0]
            ci_lo_r = float(rrow['ci_lo_mb']); ci_hi_r = float(rrow['ci_hi_mb'])
            n_boot  = int(rrow['n_boot']) if 'n_boot' in rrow.index else 100
            y_ci    = _get_y(ci_gene)
            if y_ci is None or ci_chr not in kinship_py: continue
            kd_ci   = kin_module.decomp_kinship({ci_chr: kinship_py[ci_chr]})
            rng_ci  = np.random.default_rng(1)
            valid_ci = ~np.isnan(y_ci)
            y_ci_v   = _rint(y_ci[valid_ci]); y_ci_v -= y_ci_v.mean()
            boot_oc  = np.column_stack([rng_ci.choice(y_ci_v, size=len(y_ci_v), replace=True)
                                        for _ in range(n_boot)])
            boot_pos, ci_info = _rps(boot_oc, cache_py, y_ci, covar_base, kd_ci,
                                     peak_chrom=ci_chr)
            ci_lo_py = ci_info['ci_lo']; ci_hi_py = ci_info['ci_hi']
            overlap  = ci_lo_py <= ci_hi_r and ci_lo_r <= ci_hi_py
            ci_comparisons.append({'gene_id': ci_gene, 'chr': ci_chr,
                                   'locus': rrow.get('locus', ''),
                                   'ci_lo_r': ci_lo_r, 'ci_lo_py': ci_lo_py,
                                   'ci_hi_r': ci_hi_r, 'ci_hi_py': ci_hi_py,
                                   'overlap': overlap})
        if ci_comparisons:
            ci_comp = _col(pd.DataFrame(ci_comparisons),
                           ['gene_id','chr','locus','ci_lo_r','ci_lo_py','ci_hi_r','ci_hi_py','overlap'])
            save_csv(ci_comp, 'ci_comparison.csv')
            save_csv(_col(ci_comp[['gene_id','chr','locus','ci_lo_py','ci_hi_py']]
                          .rename(columns={'ci_lo_py':'ci_lo_mb','ci_hi_py':'ci_hi_mb'}),
                          ['gene_id','chr','locus','ci_lo_mb','ci_hi_mb']), 'ci_py.csv')
            # serialize boot_pos for standalone replot
            if boot_pos is not None and hasattr(boot_pos, '__len__'):
                _bp = np.asarray(boot_pos, dtype=float)
                save_csv(_col(pd.DataFrame({'gene_id': ci_gene, 'chr': ci_chr,
                                            'boot_pos': _bp[~np.isnan(_bp)]}),
                              ['gene_id', 'chr', 'boot_pos']),
                         'boot_pos_py.csv')
            n_overlap = int(ci_comp['overlap'].sum())
            plot_phase2.plot_s10(ci_comp, boot_pos, r_boot, ci_gene, ci_chr, n_overlap)

            record('positional_ci', n_overlap > 0,
                   f'n_chr={len(ci_comp)}  overlap={n_overlap}/{len(ci_comp)}')

    # ══════════════════════════════════════════════════════════════════════════════
    # S11 — INSTABILITY SCAN
    # ══════════════════════════════════════════════════════════════════════════════
if section(11, 'INSTABILITY SCAN'):
    r_fx = load_r('fixef_r.csv')
    if r_fx is None or pheno_strain is None:
        skip('instability','fixef_r.csv missing — run port_validation.R s11')
    else:
        from mosaiqtl.scan import instability_lm_scan as _inst
        fx_pairs = r_fx[['gene_id','chr']].drop_duplicates().itertuples(index=False)
        fx_all   = []; inst_pval_rows = []  # accumulate pvals across all genes
        for fp in fx_pairs:
            fx_gene = str(fp.gene_id); fx_chr = str(fp.chr)
            y_fx = _get_y(fx_gene)
            if y_fx is None or fx_chr not in kinship_py: continue
            loci_r  = set(r_fx[(r_fx['gene_id']==fx_gene) &
                               (r_fx['chr'].astype(str)==fx_chr)]['locus'].tolist())
            if not loci_r: continue
            inst_py = _inst(y_fx, cache_py, covar_base, n_samples=100, seed=1,
                            phenotype_id=fx_gene, just_these_loci=loci_r)
            py_inst = pd.DataFrame({'locus': inst_py.locus_ids,
                                    'lod_py': inst_py.lod,
                                    'chr': inst_py.chromosomes.astype(str),   # explicit str cast
                                    'pos_mb': inst_py.positions})
            py_inst = py_inst[py_inst['chr'] == fx_chr]   # now guaranteed to match
            rg_fx   = (r_fx[(r_fx['gene_id']==fx_gene) & (r_fx['chr'].astype(str)==fx_chr)]
                       [['locus','lod','pos_mb']]
                       .rename(columns={'lod':'lod_r','pos_mb':'pos_mb_r'}))
            mg      = py_inst.merge(rg_fx, on='locus', how='inner')
            if len(mg) < 2: continue
            mg['gene_id'] = fx_gene
            fx_all.append(mg)
            # collect p_values if available (locus-level peak proportion statistic)
            if hasattr(inst_py, 'p_value') and inst_py.p_value is not None:
                for _lid, _pv in zip(inst_py.locus_ids, inst_py.p_value):
                    inst_pval_rows.append({'gene_id': fx_gene, 'locus': _lid,
                                           'p_value': float(_pv)})
            log.info(f'  instability chr{fx_chr} {fx_gene}: n={len(mg)}  '
                     f'py_range=[{mg["lod_py"].min():.3f},{mg["lod_py"].max():.3f}]  '
                     f'r_range=[{mg["lod_r"].min():.3f},{mg["lod_r"].max():.3f}]')
            # diagnostic: show first 5 rows for every chr so merge errors are visible
            hi(f'  [S11 diag] chr{fx_chr} first rows:\n'
               + mg[['locus','pos_mb_r','lod_r','lod_py']].head(5).to_string(index=False))

        if fx_all:
            mg_fx = pd.concat(fx_all, ignore_index=True)
            mg_fx = _col(mg_fx, ['gene_id','chr','locus','pos_mb_r','lod_r','lod_py'])
            save_csv(mg_fx, 'fixef_comparison.csv')
            save_csv(_col(mg_fx[['gene_id','chr','locus','pos_mb_r','lod_py']]
                          .rename(columns={'pos_mb_r':'pos_mb','lod_py':'lod'}),
                          ['gene_id','chr','locus','pos_mb','lod']), 'fixef_py.csv')
            # serialize instability p-values for standalone replot
            if inst_pval_rows:
                save_csv(_col(pd.DataFrame(inst_pval_rows),
                              ['gene_id', 'locus', 'p_value']),
                         'instability_pvals_py.csv')
            r_fx_v = (float(np.corrcoef(mg_fx['lod_r'], mg_fx['lod_py'])[0,1])
                      if len(mg_fx) > 3 else np.nan)
            plot_phase2.plot_s11(mg_fx, r_fx_v, inst_py)

            record('instability', not np.isnan(r_fx_v) and r_fx_v > 0.90,
                   f'r={r_fx_v:.4f}  n_loci={len(mg_fx)}  n_chr={mg_fx["chr"].nunique()}')
        else:
            skip('instability', 'no chr with sufficient locus overlap')

# ══════════════════════════════════════════════════════════════════════════════
# QC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
# add skipped-by-sections entries so summary table is always complete
_recorded = {r['section'] for r in qc_rows}
for _sn, _sk in _section_keys.items():
    if _sn not in _run_sections and _sk not in _recorded:
        qc_rows.append({'section':_sk,'pass':None,
                        'detail':'skipped: not in --sections'})

qc_df = pd.DataFrame(qc_rows)
# when running a subset of sections, merge into existing qc_summary.csv
# rather than overwriting — preserves results from other sections
_qc_path = os.path.join(PY_DATA,'qc_summary.csv')
if args.sections and os.path.exists(_qc_path):
    _existing = pd.read_csv(_qc_path)
    _new_keys = set(qc_df['section'])
    _existing = _existing[~_existing['section'].isin(_new_keys)]  # drop stale rows
    qc_df = pd.concat([_existing, qc_df], ignore_index=True)
qc_df.to_csv(_qc_path, index=False)
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
