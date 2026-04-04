# validate_phase3.py — phase 3 novel extension validation
#
# four novel mosaiQTL modules validated against:
#   (a) internal consistency / biological priors (independent QC)
#   (b) phase 2 LMM py_data outputs for cross-model discrepancy analysis
#
# sections:
#   p1. scan_nb          — NB GLM genome-wide scan (primary model)
#   p2. scan_vqtl        — dispersion QTL on known candidates
#   p3. scan_epistasis   — GxG interaction: Piezo1 × Ccdc171
#   p4. scan_conditional — forward-stepwise multi-signal decomposition
#
# each section produces:
#   independent QC plots  — model diagnostics, input data quality
#   comparison plots      — NB vs LMM discrepancy analysis
#   CSV outputs           — full data for downstream inspection
#
# usage:
#   python validation/validate_phase3.py
#   python validation/validate_phase3.py --cell_type type2x --n_genes 500
#   python validation/validate_phase3.py --sections nb,vqtl
#
# slurm:
#   sbatch --mem=64G --cpus-per-task=4 --time=12:00:00 \
#          --wrap 'conda run -n bioinfo python validation/validate_phase3.py'

import numpy as np
import pandas as pd
import os, sys, argparse, logging, datetime
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module
from mosaiqtl.utils import rint as _rint

# ── paths ─────────────────────────────────────────────────────────────────────
HPC_BASE  = '/share/crsp/lab/seyedam/gfilimba'
MIQTL_DIR = f'{HPC_BASE}/projects/igvf_8cube/short-read/data/gastroc'
ADATA_PATH= f'{MIQTL_DIR}/gastroc_all_processed2.h5ad'
PKG_DIR   = f'{HPC_BASE}/packages/mosaiqtl'

_LOCAL = os.path.expanduser(
    '~/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl')
if not os.path.exists(HPC_BASE):
    MIQTL_DIR  = f'{_LOCAL}/miqtl'
    ADATA_PATH = f'{_LOCAL}/../../../data/gastroc/gastroc_all_processed2.h5ad'
    PKG_DIR    = f'{_LOCAL}/mosaiqtl'

GM_CSV      = f'{MIQTL_DIR}/input/GM_ccfo_mm39.csv'
GM_GMAP     = f'{MIQTL_DIR}/input/GM/gmap'
GM_PMAP     = f'{MIQTL_DIR}/input/GM/pmap'
HAPPY_CACHE = f'{MIQTL_DIR}/output/genomecache'

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

# known LMM hits — biological anchors for sanity checks
KNOWN_HITS = {
    'Piezo1':  {'chr':'8', 'approx_mb':122.0, 'lod_lmm':21.8},
    'Ccdc171': {'chr':'4', 'approx_mb': 84.0, 'lod_lmm': 8.0},
    'Pmis2':   {'chr':'7', 'approx_mb': 85.0, 'lod_lmm': 6.0},
}

# ── args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--cell_type',     default='type2x')
ap.add_argument('--cell_type_col', default=None)
ap.add_argument('--strain_col',    default=None)
ap.add_argument('--n_genes',       default=500,  type=int)
ap.add_argument('--min_cells',     default=10,   type=int)
ap.add_argument('--sections',      default='nb,vqtl,epistasis,conditional')
ap.add_argument('--n_cores',       default=1,    type=int)
args = ap.parse_args()
_run = set(s.strip() for s in args.sections.split(','))

# ── outputs ───────────────────────────────────────────────────────────────────
PY_DATA = os.path.join(PKG_DIR, 'validation', 'port_validation', 'py_data')
PLOTS   = os.path.join(PKG_DIR, 'validation', 'port_validation', 'qc_plots')
LOGS    = os.path.join(PKG_DIR, 'validation', 'port_validation', 'logs')
for d in (PY_DATA, PLOTS, LOGS): os.makedirs(d, exist_ok=True)

_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('phase3')
log.setLevel(logging.DEBUG)
_fh = logging.FileHandler(os.path.join(LOGS, f'phase3_{_ts}.log'))
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
log.addHandler(_fh)
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(_sh)

def hi(msg): log.info(msg)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'sans-serif',
                     'font.sans-serif':['Carlito','DejaVu Sans'],
                     'font.size':11})

_chr_sort = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)
CHR_PAL   = ['#4F6EAF','#888780']

def save_csv(df, name):
    p = os.path.join(PY_DATA, name)
    df.to_csv(p, index=False)
    hi(f'  saved {name} ({len(df)} rows)')

def save_plot(fig, name):
    p = os.path.join(PLOTS, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    hi(f'  saved {name}')

def _col(df, first):
    return df[first + [c for c in df.columns if c not in first]]

def scatter_r(ax, x, y, xlabel, ylabel, title, color='#4F6EAF'):
    '''scatter with y=x reference line and Pearson r. axes start at 0.'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok   = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    lim  = max(float(np.nanmax(np.abs(np.concatenate([x, y])))), 0.1) * 1.05
    ax.scatter(x, y, s=4, alpha=0.7, color=color, edgecolors='none')
    ax.plot([0, lim], [0, lim], '--', color='#D83026', lw=1)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 else np.nan
    ax.text(0.05, 0.95, f'$r$={r:.4f}', transform=ax.transAxes, fontsize=9, va='top')
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=xlabel, ylabel=ylabel, title=title)
    sns.despine(ax=ax)
    return r

qc_rows = []
def record(key, passed, detail):
    st = 'PASS' if passed else ('NA' if passed is None else 'FAIL')
    hi(f'  {st}: {detail}')
    qc_rows.append({'section': key, 'pass': passed, 'detail': detail})

# ── load LMM reference (phase 2 py_data) ─────────────────────────────────────
hi('\n● loading LMM reference data')
def _load_py(name):
    p = os.path.join(PY_DATA, name)
    if not os.path.exists(p):
        hi(f'  WARN: {name} not found — run validate_port.py first')
        return None
    df = pd.read_csv(p, low_memory=False)
    if 'chr' in df.columns: df['chr'] = df['chr'].astype(str)
    return df

lmm_gene_summary   = _load_py('gene_summary.csv')
lmm_lod_comp       = _load_py('lod_comparison.csv')
lmm_allele_effects = _load_py('allele_effects_r.csv')

# gene_id → {lod, chr} from LMM miQTL R reference
lmm_peak = {}
if lmm_gene_summary is not None:
    for _, row in lmm_gene_summary.iterrows():
        lmm_peak[row['gene_id']] = {'lod': float(row['peak_lod_r']),
                                     'chr': str(row['chr'])}
hi(f'  LMM genes: {len(lmm_peak)}')

# ── load h5ad ────────────────────────────────────────────────────────────────
hi(f'\n● loading {ADATA_PATH}')
if not os.path.exists(ADATA_PATH):
    hi('  ERROR: h5ad not found'); sys.exit(1)

import scanpy as sc
sc.settings.verbosity = 0
adata = sc.read_h5ad(ADATA_PATH)
hi(f'  {adata.shape[0]} cells × {adata.shape[1]} genes')

# ── detect obs columns ────────────────────────────────────────────────────────
def _detect(candidates, cols, label):
    for c in candidates:
        if c in cols: return c
    hi(f'  WARN: {label} not detected — tried {candidates}'); return None

_obs    = adata.obs.columns.tolist()
ct_col  = args.cell_type_col or _detect(
    ['cell_type','celltype','leiden','cluster','subtype'], _obs, 'cell_type')
str_col = args.strain_col or _detect(
    ['strain','subject.name','subject_name','sample','mouse_id'], _obs, 'strain')
bat_col = _detect(['batch','Batch','sequencing_batch'], _obs, 'batch')

if ct_col is None or str_col is None:
    hi('  ERROR: cannot identify cell_type or strain column'); sys.exit(1)
hi(f'  cell_type_col={ct_col}  strain_col={str_col}  batch_col={bat_col}')

# ── subset + filter ───────────────────────────────────────────────────────────
mask_ct  = adata.obs[ct_col].astype(str) == args.cell_type
adata_ct = adata[mask_ct].copy()
hi(f'  {args.cell_type}: {adata_ct.n_obs} cells')
if adata_ct.n_obs < 100:
    hi('  ERROR: <100 cells'); sys.exit(1)

strain_counts = adata_ct.obs[str_col].value_counts()
keep_strains  = strain_counts[strain_counts >= args.min_cells].index.tolist()
adata_ct = adata_ct[adata_ct.obs[str_col].isin(keep_strains)].copy()
hi(f'  strains ≥{args.min_cells} cells: {len(keep_strains)}/{len(strain_counts)}')

# ── raw counts ────────────────────────────────────────────────────────────────
if adata_ct.raw is not None:
    X_raw = adata_ct.raw.X
    gene_names_all = list(adata_ct.raw.var_names)
else:
    X_raw = adata_ct.X
    gene_names_all = list(adata_ct.var_names)
if hasattr(X_raw, 'toarray'): X_raw = X_raw.toarray()
X_raw = np.array(X_raw, dtype=np.float64)   # (n_cells, n_genes_all)
strain_ids_cells = np.array(adata_ct.obs[str_col].astype(str))
lib_sizes_cells  = X_raw.sum(axis=1)
hi(f'  counts: {X_raw.shape}  median lib: {np.median(lib_sizes_cells):.0f}')

# ── gene selection ────────────────────────────────────────────────────────────
cand_genes = [g for g in KNOWN_HITS if g in gene_names_all]
if 'highly_variable' in adata_ct.var.columns:
    hvg_genes = adata_ct.var_names[adata_ct.var['highly_variable']].tolist()
else:
    mean_e = X_raw.mean(axis=0); std_e = X_raw.std(axis=0)
    cv     = np.where(mean_e > 0.1, std_e/mean_e, 0.0)
    hvg_genes = [gene_names_all[i] for i in np.argsort(cv)[::-1] if mean_e[i] > 0.1]
gene_set = list(cand_genes)
for g in hvg_genes:
    if g not in gene_set: gene_set.append(g)
    if len(gene_set) >= args.n_genes: break
gene_set = [g for g in gene_set if g in gene_names_all]
gene_idx = np.array([gene_names_all.index(g) for g in gene_set])
counts_ct = X_raw[:, gene_idx].T.copy()   # (n_genes, n_cells)
hi(f'  gene set: {len(gene_set)}  candidates found: {cand_genes}')

# ── covariates ────────────────────────────────────────────────────────────────
cov_parts = [np.ones((adata_ct.n_obs, 1))]
if bat_col and adata_ct.obs[bat_col].notna().any():
    bd = pd.get_dummies(adata_ct.obs[bat_col].astype(str),
                        prefix='batch', drop_first=True).astype(float).values
    cov_parts.append(bd)
if 'X_pca' in adata_ct.obsm:
    cov_parts.append(adata_ct.obsm['X_pca'][:, :10].astype(float))
covar_cells = np.hstack(cov_parts).astype(float)   # (n_cells, p)

# strain-level covariate matrix
strains_sorted, inv = np.unique(strain_ids_cells, return_inverse=True)
n_strains = len(strains_sorted)
cov_sum   = np.zeros((n_strains, covar_cells.shape[1]))
cnt_s     = np.zeros(n_strains)
np.add.at(cov_sum, inv, covar_cells)
np.add.at(cnt_s,   inv, 1)
covar_strains = cov_sum / cnt_s[:, None]   # (n_strains, p)
hi(f'  covariates: {covar_cells.shape[1]} cols')

# ── genome cache + kinship ────────────────────────────────────────────────────
hi('\n● building genome cache + kinship')
import glob, rpy2.robjects as _ro
_mf = glob.glob(os.path.join(HAPPY_CACHE, 'additive', 'chr*', 'markers.RData'))
if _mf:
    _happy_markers = []
    for mf in sorted(_mf):
        _e = _ro.r['new.env']()
        _ro.r['load'](mf, envir=_e)
        _happy_markers.extend(list(_ro.r['get']('markers', envir=_e)))
else:
    _happy_markers = None

cache_py       = GenomeCache.from_gigamuga(
    gigamuga_path=GM_CSV, gmap_dir=GM_GMAP, pmap_dir=GM_PMAP,
    cc_cols=CC_COLS, founder_cols=FOUNDER_COLS, f1_crosses=F1_CROSSES,
    just_these_loci=_happy_markers)
chr_order      = sorted(cache_py.chromosomes, key=_chr_sort)
kinship_py     = kin_module.calc_kinship(cache_py, method='loco')
kinship_decomp = kin_module.decomp_kinship(kinship_py)
cache_sids     = list(cache_py.sample_ids)
cache_sid_set  = set(cache_sids)
hi(f'  cache: {len(chr_order)} chr | {len(cache_sids)} strains')

# ── shared helpers ────────────────────────────────────────────────────────────
def _pb_rint(gene_name):
    '''pseudobulk log1p sum → RINT aligned to cache_sids.'''
    if gene_name not in gene_names_all: return None
    gi  = gene_names_all.index(gene_name)
    pb  = {}
    for sid, cnt in zip(strain_ids_cells, X_raw[:, gi]):
        pb[sid] = pb.get(sid, 0.0) + cnt
    y = np.array([pb.get(s, np.nan) for s in cache_sids])
    return _rint(np.log1p(y))

def _cache_cov():
    '''strain-level covariates aligned to cache_sids.'''
    sid_idx = {s: i for i, s in enumerate(strains_sorted)}
    return np.array([covar_strains[sid_idx[s]] if s in sid_idx
                     else np.zeros(covar_strains.shape[1])
                     for s in cache_sids])

# ── cis loci ─────────────────────────────────────────────────────────────────
pmap_path = os.path.join(MIQTL_DIR, 'input', 'pmap.csv')
if not os.path.exists(pmap_path):
    pmap_path = os.path.join(MIQTL_DIR, 'output', 'pmap.csv')
cis_loci_dict = None
if os.path.exists(pmap_path):
    pmap = pd.read_csv(pmap_path)
    pmap.columns = [c.lower().replace('.', '_') for c in pmap.columns]
    pmap['chr'] = pmap['chr'].astype(str)
    _pc = _detect(['start','gene_start','chromStart'], adata.var.columns.tolist(), 'pos')
    _cc = _detect(['chromosome','chrom','chr'],        adata.var.columns.tolist(), 'chr')
    if _pc and _cc:
        ginfo = adata.var[[_cc, _pc]].copy()
        ginfo.columns = ['gene_chr', 'gene_pos_bp']
        cis_loci_dict = {}
        for g in gene_set:
            if g not in ginfo.index: continue
            gc = str(ginfo.loc[g, 'gene_chr'])
            gp = float(ginfo.loc[g, 'gene_pos_bp']) / 1e6
            cis = pmap[(pmap['chr']==gc) &
                       (pmap['pos_mb'] >= gp-5) &
                       (pmap['pos_mb'] <= gp+5)]['marker'].tolist()
            if cis: cis_loci_dict[g] = set(cis)
        hi(f'  cis loci: {len(cis_loci_dict)} genes')

# ══════════════════════════════════════════════════════════════════════════════
# P1 — SCAN_NB
# ══════════════════════════════════════════════════════════════════════════════
if 'nb' in _run:
    hi(f'\n▶ P1 — SCAN_NB  ({len(gene_set)} genes, {args.cell_type})')
    from mosaiqtl.nb import scan_nb, scan_nb_gene, pseudobulk_sum, fit_null_sc

    # ── independent QC: cell count + library size per strain ─────────────────
    s_ncells = pd.Series(strain_ids_cells).value_counts().sort_index()
    s_medlib = {s: float(np.median(lib_sizes_cells[strain_ids_cells==s]))
                for s in s_ncells.index}
    cell_qc  = pd.DataFrame({'strain': s_ncells.index,
                              'n_cells': s_ncells.values,
                              'median_lib': [s_medlib[s] for s in s_ncells.index]})
    save_csv(cell_qc, 'p1_cell_qc.csv')

    fig_cq, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(cell_qc)), cell_qc['n_cells'].values,
                color='#4F6EAF', edgecolor='none')
    axes[0].axhline(args.min_cells, color='#D83026', lw=1, ls='--',
                    label=f'min={args.min_cells}')
    axes[0].set(xlabel='strain index', ylabel='n cells',
                title=f'cells per strain ({args.cell_type})', ylim=(0, None))
    axes[0].legend(fontsize=8); sns.despine(ax=axes[0])
    axes[1].bar(range(len(cell_qc)), cell_qc['median_lib'].values,
                color='#55AF5B', edgecolor='none')
    axes[1].set(xlabel='strain index', ylabel='median UMI',
                title='median library size per strain', ylim=(0, None))
    sns.despine(ax=axes[1])
    plt.tight_layout(); save_plot(fig_cq, 'p1_cell_qc.png')

    # ── run scan_nb ───────────────────────────────────────────────────────────
    try:
        results_nb = scan_nb(
            counts_cells       = counts_ct,
            genome_cache       = cache_py,
            strain_ids_cells   = strain_ids_cells,
            gene_ids           = gene_set,
            covariates_cells   = covar_cells,
            covariates_strains = covar_strains,
            norm_method        = 'TMM',
            shrink_disp        = True,
            cis_loci_dict      = cis_loci_dict,
            n_cores            = args.n_cores,
            verbose            = True,
        )
        hi(f'  scan_nb: {len(results_nb)} genes')
    except Exception as e:
        hi(f'  ERROR scan_nb: {e}'); log.exception('scan_nb'); results_nb = {}

    if results_nb:
        J = cache_py.n_founders; df_chi2 = J - 1

        # ── peak summary ──────────────────────────────────────────────────────
        peak_rows = []
        for gid, sr in results_nb.items():
            if not len(sr.lod): continue
            pk = sr.peak_info()
            peak_rows.append({'gene_id': gid, 'chr': pk['chromosome'],
                              'locus': pk['locus_id'], 'pos_mb': pk['position'],
                              'lod_nb': pk['lod']})
        peak_df = (pd.DataFrame(peak_rows)
                   .sort_values('lod_nb', ascending=False)
                   .reset_index(drop=True))
        peak_df['p_value'] = sp_stats.chi2.sf(
            2*np.log(10)*peak_df['lod_nb'].values, df=df_chi2)
        _, fdr, _, _ = multipletests(peak_df['p_value'].values, method='fdr_bh')
        peak_df['fdr_bh']  = fdr
        peak_df['egene']   = fdr < 0.05
        peak_df['rank_nb'] = peak_df['lod_nb'].rank(ascending=False).astype(int)
        n_egenes = int(peak_df['egene'].sum())

        # ── NB vs LMM merge ───────────────────────────────────────────────────
        nb_lmm = None
        if lmm_gene_summary is not None:
            lmm_sub = (lmm_gene_summary[['gene_id','chr','peak_lod_r','peak_lod_py']]
                       .rename(columns={'chr':'chr_lmm','peak_lod_r':'lod_lmm_r',
                                        'peak_lod_py':'lod_lmm_py'}))
            nb_lmm = peak_df.merge(lmm_sub, on='gene_id', how='inner').copy()
            nb_lmm['rank_lmm']       = nb_lmm['lod_lmm_r'].rank(ascending=False).astype(int)
            nb_lmm['peak_chr_match'] = nb_lmm['chr'] == nb_lmm['chr_lmm']
            nb_lmm['lod_ratio']      = nb_lmm['lod_nb'] / nb_lmm['lod_lmm_r'].replace(0, np.nan)
            nb_lmm = _col(nb_lmm, ['gene_id','chr','locus','pos_mb',
                                    'lod_nb','lod_lmm_r','lod_lmm_py',
                                    'rank_nb','rank_lmm','peak_chr_match','lod_ratio',
                                    'fdr_bh','egene'])
            save_csv(nb_lmm, 'p1_nb_vs_lmm_comparison.csv')
            pct_chr = nb_lmm['peak_chr_match'].mean()
            rs = sp_stats.spearmanr(nb_lmm['rank_nb'], nb_lmm['rank_lmm']).statistic
            hi(f'  NB vs LMM: n={len(nb_lmm)}  chr_match={pct_chr:.1%}  rank_rho={rs:.4f}')

        save_csv(_col(peak_df, ['gene_id','chr','locus','pos_mb',
                                'lod_nb','p_value','fdr_bh','egene','rank_nb']),
                 'p1_nb_peak_summary.csv')

        # genome-wide LOD (capped for file size)
        lod_rows = []
        for gid, sr in results_nb.items():
            if not len(sr.lod): continue
            lod_rows.append(pd.DataFrame({'gene_id':gid,'chr':sr.chromosomes,
                                           'pos_mb':sr.positions,'lod':sr.lod}))
        lod_df = pd.concat(lod_rows, ignore_index=True)
        save_csv(lod_df.nlargest(50000, 'lod'), 'p1_nb_lod_top50k.csv')

        # ── independent QC 1: chi2 QQ plot (null calibration check) ──────────
        all_lod = lod_df['lod'].values
        chi2_obs = np.sort(2*np.log(10)*all_lod)
        n_s = len(chi2_obs)
        chi2_exp = sp_stats.chi2.ppf(
            np.linspace(1/(n_s+1), n_s/(n_s+1), n_s), df=df_chi2)
        step = max(1, n_s//5000)
        fig_qq, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(chi2_exp[::step], chi2_obs[::step],
                   s=2, alpha=0.5, color='#4F6EAF', edgecolors='none',
                   label='observed')
        ax.plot([0, chi2_exp.max()], [0, chi2_exp.max()],
                '--', color='#D83026', lw=1, label=r'$\chi^2$ null')
        ax.set(xlabel=r'expected $\chi^2_{J-1}$', ylabel=r'observed $\chi^2$',
               title=f'NB score statistic QQ  ($J-1$={df_chi2} df)',
               xlim=(0, None), ylim=(0, None))
        ax.legend(fontsize=8); sns.despine(ax=ax)
        plt.tight_layout(); save_plot(fig_qq, 'p1_nb_chi2_qq.png')

        # ── independent QC 2: LOD histogram + known hit markers ──────────────
        fig_lh, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(peak_df['lod_nb'], bins=50, color='#4F6EAF',
                     edgecolor='none', ax=ax)
        for gname, info in KNOWN_HITS.items():
            row = peak_df[peak_df['gene_id']==gname]
            if row.empty: continue
            lv = float(row['lod_nb'].iloc[0])
            ax.axvline(lv, color=FOUNDER_COLORS.get('PWKJ','#D83026'),
                       lw=1.5, ls='--', label=f'{gname} LOD={lv:.1f}')
        ax.set(xlabel='peak LOD (NB)', ylabel='genes',
               title=f'NB peak LOD distribution  {args.cell_type}',
               xlim=(0, None), ylim=(0, None))
        ax.legend(fontsize=8); sns.despine(ax=ax)
        plt.tight_layout(); save_plot(fig_lh, 'p1_nb_lod_distribution.png')

        # ── independent QC 3: Piezo1 chr8 profile ────────────────────────────
        if 'Piezo1' in results_nb:
            sr_p  = results_nb['Piezo1']
            mask8 = sr_p.chromosomes.astype(str) == '8'
            fig_p8, ax = plt.subplots(figsize=(9, 4))
            ax.fill_between(sr_p.positions[mask8], 0, sr_p.lod[mask8],
                            color='#4F6EAF', alpha=0.8, step='mid')
            pk_pos = float(sr_p.positions[mask8][np.argmax(sr_p.lod[mask8])])
            pk_lod = float(sr_p.lod[mask8].max())
            ax.axvline(pk_pos, color='#D83026', lw=1.2, ls='--',
                       label=f'NB peak {pk_pos:.1f} Mb  LOD={pk_lod:.2f}')
            ax.axvline(KNOWN_HITS['Piezo1']['approx_mb'], color='#55AF5B',
                       lw=1.0, ls=':', label=f'LMM ref ~{KNOWN_HITS["Piezo1"]["approx_mb"]:.0f} Mb')
            ax.set(xlabel='position chr8 (Mb)', ylabel='LOD (NB)',
                   title='Piezo1 NB scan — chr8', xlim=(0, None), ylim=(0, None))
            ax.legend(fontsize=8); sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig_p8, 'p1_nb_piezo1_chr8.png')

        # ── comparison 1: NB vs LMM LOD scatter + rank concordance ──────────
        r_lod_nb_lmm = np.nan
        if nb_lmm is not None and len(nb_lmm) > 3:
            fig_sc, axes = plt.subplots(1, 2, figsize=(11, 4))
            r_lod_nb_lmm = scatter_r(axes[0],
                                     nb_lmm['lod_lmm_r'].values,
                                     nb_lmm['lod_nb'].values,
                                     'peak LOD (LMM miQTL R)',
                                     'peak LOD (NB mosaiQTL)',
                                     'NB vs LMM peak LOD')
            rs = sp_stats.spearmanr(nb_lmm['rank_lmm'],
                                    nb_lmm['rank_nb']).statistic
            scatter_r(axes[1],
                      nb_lmm['rank_lmm'].values.astype(float),
                      nb_lmm['rank_nb'].values.astype(float),
                      'rank (LMM)', 'rank (NB)',
                      r'rank concordance  $\rho$=' + f'{rs:.4f}')
            plt.tight_layout(); save_plot(fig_sc, 'p1_nb_vs_lmm_lod_scatter.png')

        # ── comparison 2: dual Manhattan chr8 (NB top / LMM bottom) ─────────
        if 'Piezo1' in results_nb and lmm_lod_comp is not None:
            sr_p   = results_nb['Piezo1']
            mask8  = sr_p.chromosomes.astype(str) == '8'
            lmm_p8 = lmm_lod_comp[(lmm_lod_comp['gene_id']=='Piezo1') &
                                    (lmm_lod_comp['chr']=='8')]
            fig_dm, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axes[0].fill_between(sr_p.positions[mask8], 0, sr_p.lod[mask8],
                                  color='#4F6EAF', alpha=0.8, step='mid', label='NB')
            axes[0].set(ylabel='LOD (NB)', ylim=(0, None))
            axes[0].legend(fontsize=8); sns.despine(ax=axes[0])
            if not lmm_p8.empty:
                axes[1].fill_between(lmm_p8['pos_mb'], 0, lmm_p8['lod_py'],
                                      color='#D83026', alpha=0.8, step='mid', label='LMM')
            axes[1].set(xlabel='position chr8 (Mb)', ylabel='LOD (LMM)', ylim=(0, None))
            axes[1].legend(fontsize=8); sns.despine(ax=axes[1])
            plt.suptitle('Piezo1 — NB vs LMM scan (chr8)', y=1.01, fontsize=11)
            plt.tight_layout(); save_plot(fig_dm, 'p1_piezo1_nb_vs_lmm_chr8.png')

        # ── comparison 3: top 20 eGenes NB + LMM side-by-side bar ───────────
        top20 = peak_df.head(20).copy()
        if nb_lmm is not None:
            top20 = top20.merge(nb_lmm[['gene_id','lod_lmm_r']], on='gene_id', how='left')
        fig_t, ax = plt.subplots(figsize=(8, 5))
        ypos = np.arange(len(top20))[::-1]
        ax.barh(ypos - 0.18, top20['lod_nb'].values, 0.35,
                color='#4F6EAF', edgecolor='none', label='NB')
        if 'lod_lmm_r' in top20.columns:
            ax.barh(ypos + 0.18, top20['lod_lmm_r'].fillna(0).values, 0.35,
                    color='#D83026', alpha=0.7, edgecolor='none', label='LMM (R)')
        ax.set(yticks=ypos, yticklabels=top20['gene_id'].tolist(),
               xlabel='peak LOD', title='top 20 eGenes — NB vs LMM', xlim=(0, None))
        ax.legend(fontsize=9); sns.despine(ax=ax)
        plt.tight_layout(); save_plot(fig_t, 'p1_top20_nb_vs_lmm.png')

        piezo_lod = (float(peak_df.loc[peak_df['gene_id']=='Piezo1','lod_nb'].iloc[0])
                     if 'Piezo1' in peak_df['gene_id'].values else 0.0)
        lod_skew  = float(sp_stats.skew(peak_df['lod_nb'].values))
        record('scan_nb', lod_skew > 0.5 and piezo_lod > 3.0,
               f'n_egenes={n_egenes}  piezo1_lod={piezo_lod:.2f}  lod_skew={lod_skew:.2f}'
               + (f'  nb_vs_lmm_r={r_lod_nb_lmm:.4f}' if np.isfinite(r_lod_nb_lmm) else ''))
    else:
        record('scan_nb', False, 'scan_nb returned no results')

# ══════════════════════════════════════════════════════════════════════════════
# P2 — SCAN_VQTL
# ══════════════════════════════════════════════════════════════════════════════
if 'vqtl' in _run:
    hi(f'\n▶ P2 — SCAN_VQTL  ({list(KNOWN_HITS.keys())})')
    from mosaiqtl.vqtl import scan_vqtl, estimate_dispersion_per_subject

    vqtl_rows   = []
    phi_qc_rows = []
    cov_cache   = _cache_cov()
    shared_mask = np.array([s in cache_sid_set for s in strains_sorted])
    cov_shared  = covar_strains[shared_mask]

    for gname, info in KNOWN_HITS.items():
        if gname not in gene_names_all:
            hi(f'  skip {gname}: not in gene list'); continue
        gi       = gene_names_all.index(gname)
        counts_g = X_raw[:, gi]
        kd_g     = ({info['chr']: kinship_decomp[info['chr']]}
                    if info['chr'] in kinship_decomp else None)

        try:
            res_v = scan_vqtl(
                counts         = counts_g,
                subject_ids    = strain_ids_cells,
                genome_cache   = cache_py,
                library_sizes  = lib_sizes_cells,
                covariates     = cov_shared,
                kinship_decomp = kd_g,
                phi_method     = 'moments',
                min_cells      = args.min_cells,
                use_rint       = True,
                gene_id        = gname,
                also_scan_mean = True,
            )
        except Exception as e:
            hi(f'  ERROR scan_vqtl({gname}): {e}'); log.exception(f'vqtl {gname}'); continue

        if res_v is None:
            hi(f'  {gname}: scan_vqtl returned None'); continue

        # correct field names: scan_result, mean_scan_result, phi_estimates
        sr_v  = res_v.scan_result
        sr_m  = res_v.mean_scan_result
        phi_v = res_v.phi_estimates
        pk_v  = sr_v.peak_info()
        hi(f'  {gname}: vQTL chr{pk_v["chromosome"]} {pk_v["position"]:.1f} Mb LOD={pk_v["lod"]:.3f}')

        # per-strain phi for QC CSV
        subj_v, phi_hat_v = estimate_dispersion_per_subject(
            counts_g, strain_ids_cells, lib_sizes_cells,
            method='moments', min_cells=args.min_cells)
        pb_mean_g = {}
        for sid, cnt in zip(strain_ids_cells, counts_g):
            pb_mean_g.setdefault(sid, []).append(float(cnt))
        pb_mean_g = {s: float(np.mean(v)) for s, v in pb_mean_g.items()}
        for si, sid in enumerate(subj_v):
            phi_qc_rows.append({'gene_id': gname, 'strain': sid,
                                 'phi_hat': float(phi_hat_v[si]),
                                 'log_phi': float(np.log(max(float(phi_hat_v[si]), 1e-9))),
                                 'mean_expr': pb_mean_g.get(sid, np.nan)})

        lmm_info = lmm_peak.get(gname, {})
        vqtl_rows.append({'gene_id':       gname,
                          'chr_vqtl':      pk_v['chromosome'],
                          'locus_vqtl':    pk_v['locus_id'],
                          'pos_mb_vqtl':   pk_v['position'],
                          'lod_vqtl':      pk_v['lod'],
                          'lod_mean_eqtl': float(sr_m.lod.max()) if sr_m else np.nan,
                          'chr_lmm':       lmm_info.get('chr', ''),
                          'lod_lmm':       lmm_info.get('lod', np.nan),
                          'eqtl_approx_mb':info['approx_mb'],
                          'colocalises':   (pk_v['chromosome'] == info['chr'] and
                                            abs(pk_v['position'] - info['approx_mb']) < 10)})

        # ── independent QC: vQTL profile on candidate chr ────────────────────
        fig_vq, axes = plt.subplots(1, 2, figsize=(12, 4))
        mask_c = sr_v.chromosomes.astype(str) == info['chr']
        axes[0].fill_between(sr_v.positions[mask_c], 0, sr_v.lod[mask_c],
                              color='#4F6EAF', alpha=0.8, step='mid', label='vQTL')
        axes[0].axvline(info['approx_mb'], color='#D83026', lw=1.0, ls='--',
                        label=f'eQTL ref ~{info["approx_mb"]:.0f} Mb')
        axes[0].set(xlabel=f'position chr{info["chr"]} (Mb)',
                    ylabel=r'LOD  (dispersion $\log\hat{\phi}$)',
                    title=f'{gname} — dispersion vQTL', xlim=(0, None), ylim=(0, None))
        axes[0].legend(fontsize=8); sns.despine(ax=axes[0])

        # ── comparison: mean eQTL (NB pseudobulk) vs LMM on same chr ────────
        if sr_m is not None:
            mask_m = sr_m.chromosomes.astype(str) == info['chr']
            axes[1].fill_between(sr_m.positions[mask_m], 0, sr_m.lod[mask_m],
                                  color='#55AF5B', alpha=0.8, step='mid',
                                  label='mean eQTL (NB pseudobulk)')
        if lmm_lod_comp is not None:
            lmm_g = lmm_lod_comp[(lmm_lod_comp['gene_id']==gname) &
                                   (lmm_lod_comp['chr']==info['chr'])]
            if not lmm_g.empty:
                axes[1].plot(lmm_g['pos_mb'], lmm_g['lod_py'],
                             color='#D83026', lw=1.2, ls='--', alpha=0.8,
                             label='LMM (Python)')
        axes[1].set(xlabel=f'position chr{info["chr"]} (Mb)', ylabel='LOD (mean eQTL)',
                    title=f'{gname} — mean eQTL vs LMM', xlim=(0, None), ylim=(0, None))
        axes[1].legend(fontsize=8); sns.despine(ax=axes[1])
        plt.tight_layout(); save_plot(fig_vq, f'p2_vqtl_{gname.lower()}.png')

    # ── independent QC: phi distribution + phi vs mean expr ─────────────────
    if phi_qc_rows:
        phi_df = pd.DataFrame(phi_qc_rows)
        save_csv(phi_df, 'p2_vqtl_strain_phi.csv')

        fig_ph, axes = plt.subplots(1, 2, figsize=(11, 4))
        for gn in phi_df['gene_id'].unique():
            sub = phi_df[phi_df['gene_id']==gn].dropna(subset=['log_phi'])
            sns.kdeplot(sub['log_phi'], ax=axes[0], label=gn, lw=1.5)
        axes[0].set(xlabel=r'$\log\hat{\phi}$ (per strain)',
                    ylabel='density', title='dispersion distribution per gene')
        axes[0].legend(fontsize=8); sns.despine(ax=axes[0])
        for gn in phi_df['gene_id'].unique():
            sub = phi_df[phi_df['gene_id']==gn].dropna(subset=['log_phi','mean_expr'])
            if len(sub) < 3: continue
            axes[1].scatter(np.log1p(sub['mean_expr']), sub['log_phi'],
                            s=8, alpha=0.6, label=gn, edgecolors='none')
        axes[1].set(xlabel=r'$\log(1+$ mean expr$)$', ylabel=r'$\log\hat{\phi}$',
                    title='mean–variance relationship per gene', xlim=(0, None))
        axes[1].legend(fontsize=8); sns.despine(ax=axes[1])
        plt.tight_layout(); save_plot(fig_ph, 'p2_vqtl_phi_qc.png')

    if vqtl_rows:
        vqtl_df = pd.DataFrame(vqtl_rows)
        save_csv(vqtl_df, 'p2_vqtl_summary.csv')
        record('scan_vqtl', True,
               f'n_genes={len(vqtl_rows)}  '
               f'colocalise={int(vqtl_df["colocalises"].sum())}/{len(vqtl_df)}')
    else:
        record('scan_vqtl', False, 'no vQTL results')

# ══════════════════════════════════════════════════════════════════════════════
# P3 — SCAN_EPISTASIS
# ══════════════════════════════════════════════════════════════════════════════
if 'epistasis' in _run:
    hi('\n▶ P3 — SCAN_EPISTASIS  (Piezo1 chr8 × Ccdc171 chr4)')
    from mosaiqtl.epistasis import scan_epistasis, test_haplotype_combinations

    def _peak_locus(gname):
        if 'results_nb' in dir() and gname in results_nb:
            return str(results_nb[gname].peak_info()['locus_id'])
        info = KNOWN_HITS.get(gname)
        if info is None: return None
        lids, pos, _ = cache_py.get_chromosome(info['chr'])
        return str(lids[int(np.argmin(np.abs(pos - info['approx_mb'])))])

    locus_a = _peak_locus('Piezo1')
    locus_b = _peak_locus('Ccdc171')
    hi(f'  locus_a={locus_a}  locus_b={locus_b}')

    if None in (locus_a, locus_b) or 'Piezo1' not in gene_names_all:
        record('scan_epistasis', False, 'missing loci or Piezo1 not in gene set')
    else:
        y_epi   = _pb_rint('Piezo1')
        cov_epi = _cache_cov()

        try:
            epi_res = scan_epistasis(
                phenotype      = y_epi,
                genome_cache   = cache_py,
                locus_a        = locus_a,
                locus_b        = locus_b,
                covariates     = cov_epi,
                kinship_decomp = kinship_decomp,
                use_rint       = False,
                phenotype_id   = 'Piezo1',
            )
        except Exception as e:
            hi(f'  ERROR: {e}'); log.exception('epistasis'); epi_res = None

        if epi_res is not None:
            hi(f'  LOD_additive={epi_res["lod_additive"]:.3f}  '
               f'LOD_epistasis={epi_res["lod_epistasis"]:.3f}')
            save_csv(pd.DataFrame([{
                'gene_id':'Piezo1', 'locus_a':locus_a, 'locus_b':locus_b,
                'lod_additive':    epi_res['lod_additive'],
                'lod_interaction': epi_res['lod_interaction'],
                'lod_epistasis':   epi_res['lod_epistasis'],
            }]), 'p3_epistasis_result.csv')

            # ── independent QC: haplotype combination heatmap ─────────────────
            try:
                hap_res = test_haplotype_combinations(
                    y_epi, cache_py, locus_a, locus_b,
                    covariates=cov_epi, kinship_decomp=kinship_decomp,
                    use_rint=False, phenotype_id='Piezo1')
                if hap_res is not None and 'means' in hap_res:
                    mat = np.array(hap_res['means'])
                    fig_hm, ax = plt.subplots(figsize=(6.5, 5))
                    sns.heatmap(mat, ax=ax, cmap='YlOrRd',
                                xticklabels=FOUNDER_COLS,
                                yticklabels=FOUNDER_COLS,
                                cbar_kws={'label': 'mean Piezo1 (RINT)'})
                    ax.set_xlabel(f'Ccdc171 haplotype ({locus_b})', fontsize=9)
                    ax.set_ylabel(f'Piezo1 haplotype ({locus_a})', fontsize=9)
                    plt.tight_layout()
                    save_plot(fig_hm, 'p3_epistasis_haplotype_heatmap.png')
            except Exception as e:
                hi(f'  haplotype heatmap skipped: {e}')

            # ── comparison: NB founder effects vs LMM at Piezo1 peak ─────────
            nb_fe_rows = []
            try:
                from mosaiqtl.nb import fit_null_sc, scan_nb_gene, pseudobulk_sum
                gi_p   = gene_names_all.index('Piezo1')
                pb_p, _, pb_strains = pseudobulk_sum(
                    X_raw[:, gi_p:gi_p+1].T, strain_ids_cells)
                s2idx  = {s: i for i, s in enumerate(pb_strains)}
                pb_al  = np.array([pb_p[0, s2idx[s]] if s in s2idx else 0.0
                                   for s in cache_sids])
                null_f = fit_null_sc(X_raw[:, gi_p:gi_p+1].T, strain_ids_cells,
                                     covar_cells, norm_method='TMM', shrink=True)
                phi_g  = float(null_f['dispersion_use'][0]) if null_f['gene_mask'].any() else 1.0
                sig_g  = float(null_f['sigma_sample'][0])   if null_f['gene_mask'].any() else 0.0
                sid_to_row = {s: i for i, s in enumerate(strains_sorted)}
                cov_al = np.array([covar_strains[sid_to_row[s]] if s in sid_to_row
                                   else np.zeros(covar_strains.shape[1])
                                   for s in cache_sids])
                sr_nb_p = scan_nb_gene(pb_al, cache_py, np.array(cache_sids),
                                        phi=phi_g, sigma_sample=sig_g,
                                        covariates=cov_al, use_lrt=True,
                                        just_these_loci={locus_a},
                                        phenotype_id='Piezo1')
                if sr_nb_p is not None and sr_nb_p.betas is not None and len(sr_nb_p.betas):
                    nb_betas = sr_nb_p.betas[0]
                    for fi, fc in enumerate(FOUNDER_COLS[:-1]):
                        nb_fe_rows.append({'founder': fc, 'beta_nb': float(nb_betas[fi])})
            except Exception as e:
                hi(f'  NB founder effects skipped: {e}')

            if nb_fe_rows:
                nb_fe = pd.DataFrame(nb_fe_rows)
                save_csv(nb_fe, 'p3_nb_founder_effects.csv')

                fig_fe, ax = plt.subplots(figsize=(9, 4))
                x  = np.arange(len(FOUNDER_COLS)-1); w = 0.35
                cf = [FOUNDER_COLORS.get(f, '#888') for f in FOUNDER_COLS[:-1]]
                ax.bar(x - w/2, nb_fe['beta_nb'].values, w,
                       color=cf, edgecolor='white', lw=0.5, label='NB (mosaiQTL)')
                # LMM effects from allele_effects_r.csv
                if lmm_allele_effects is not None:
                    lmm_fe = lmm_allele_effects[
                        (lmm_allele_effects['gene_id']=='Piezo1') &
                        (lmm_allele_effects['locus']==locus_a)]
                    if not lmm_fe.empty:
                        code_map = {chr(65+i): f for i, f in enumerate(FOUNDER_COLS)}
                        r_eff = [float(lmm_fe.loc[lmm_fe['founder']==code_map.get(chr(65+i),''),'effect'].iloc[0])
                                 if code_map.get(chr(65+i),'') in lmm_fe['founder'].values else 0.0
                                 for i in range(len(FOUNDER_COLS)-1)]
                        ax.bar(x + w/2, r_eff, w, color=cf, hatch='///',
                               edgecolor='#333', lw=0.5, alpha=0.7, label='LMM (miQTL R)')
                ax.axhline(0, color='#444', lw=0.5)
                ax.set(xticks=x, xticklabels=FOUNDER_COLS[:-1],
                       xlabel='founder', ylabel='effect size',
                       title=f'Piezo1 founder effects — NB vs LMM  ({locus_a})',
                       xlim=(-0.5, len(FOUNDER_COLS)-1.5))
                ax.set_xticklabels(FOUNDER_COLS[:-1], rotation=45, ha='right', fontsize=10)
                ax.legend(fontsize=9); sns.despine(ax=ax)
                plt.tight_layout(); save_plot(fig_fe, 'p3_founder_effects_nb_vs_lmm.png')

            record('scan_epistasis', epi_res['lod_additive'] > 0,
                   f'lod_additive={epi_res["lod_additive"]:.3f}  '
                   f'lod_epistasis={epi_res["lod_epistasis"]:.3f}')
        else:
            record('scan_epistasis', False, 'scan_epistasis raised error')

# ══════════════════════════════════════════════════════════════════════════════
# P4 — SCAN_CONDITIONAL_WITH_Y
# ══════════════════════════════════════════════════════════════════════════════
if 'conditional' in _run:
    hi('\n▶ P4 — SCAN_CONDITIONAL_WITH_Y  (Piezo1)')
    from mosaiqtl.epistasis import scan_conditional_with_y
    from mosaiqtl.scan import scan_h2lmm

    if 'Piezo1' not in gene_names_all:
        record('scan_conditional', False, 'Piezo1 not in gene set')
    else:
        y_cond   = _pb_rint('Piezo1')
        cov_cond = _cache_cov()

        hi('  base LMM scan (seed for conditional)')
        try:
            sr_base = scan_h2lmm(y_cond, cache_py, cov_cond,
                                  kinship_decomp, use_rint=False,
                                  phenotype_id='Piezo1')
        except Exception as e:
            hi(f'  ERROR base scan: {e}'); log.exception('base'); sr_base = None
            record('scan_conditional', False, f'base scan failed: {e}')

        if sr_base is not None:
            pk0 = sr_base.peak_info()
            hi(f'  signal 1: chr{pk0["chromosome"]} {pk0["position"]:.1f} Mb '
               f'LOD={pk0["lod"]:.3f}')
            try:
                signals = scan_conditional_with_y(
                    phenotype          = y_cond,
                    scan_result        = sr_base,
                    genome_cache       = cache_py,
                    covariates         = cov_cond,
                    kinship_decomp     = kinship_decomp,
                    n_signals          = 3,
                    lod_drop_threshold = 1.5,
                    use_rint           = False,
                )
            except Exception as e:
                hi(f'  ERROR conditional: {e}'); log.exception('conditional')
                signals = [sr_base]

            sig_rows = []
            for si, sr in enumerate(signals):
                pk = sr.peak_info()
                hi(f'  signal {si+1}: chr{pk["chromosome"]} {pk["position"]:.1f} Mb '
                   f'LOD={pk["lod"]:.3f}')
                sig_rows.append({'signal': si+1, 'chr': pk['chromosome'],
                                  'locus': pk['locus_id'], 'pos_mb': pk['position'],
                                  'lod': pk['lod']})
                # genome-wide CSV per signal
                save_csv(pd.DataFrame({'chr': sr.chromosomes,
                                        'pos_mb': sr.positions,
                                        'lod': sr.lod,
                                        'locus': sr.locus_ids}),
                         f'p4_signal{si+1}_genome_wide.csv')
            save_csv(pd.DataFrame(sig_rows), 'p4_conditional_signals.csv')

            chr_plot   = pk0['chromosome']
            colors_sig = ['#4F6EAF', '#D83026', '#55AF5B']

            # ── independent QC 1: signal profiles on peak chr ────────────────
            fig_cp, ax = plt.subplots(figsize=(9, 4))
            for si, sr in enumerate(signals):
                mask = sr.chromosomes.astype(str) == chr_plot
                ax.plot(sr.positions[mask], sr.lod[mask],
                        color=colors_sig[si % len(colors_sig)], lw=1.5, alpha=0.85,
                        label=f'signal {si+1}  LOD={sr.lod.max():.2f}')
            ax.set(xlabel=f'position chr{chr_plot} (Mb)', ylabel='LOD',
                   title=f'Piezo1 conditional — chr{chr_plot}',
                   xlim=(0, None), ylim=(0, None))
            ax.legend(fontsize=9); sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig_cp, 'p4_conditional_chr_profile.png')

            # ── independent QC 2: LOD decay bar ──────────────────────────────
            fig_ld, ax = plt.subplots(figsize=(5, 4))
            sig_lods = [float(sr.lod.max()) for sr in signals]
            ax.bar([f'signal {i+1}' for i in range(len(signals))],
                   sig_lods,
                   color=colors_sig[:len(signals)], edgecolor='none')
            ax.set(ylabel='peak LOD', title='LOD decay across signals',
                   ylim=(0, None))
            sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig_ld, 'p4_lod_decay.png')

            # ── comparison: signal 1 vs LMM on peak chr ──────────────────────
            if lmm_lod_comp is not None:
                lmm_pc = lmm_lod_comp[(lmm_lod_comp['gene_id']=='Piezo1') &
                                       (lmm_lod_comp['chr']==chr_plot)]
                fig_cm, ax = plt.subplots(figsize=(9, 4))
                sr_s1   = signals[0]
                mask_s1 = sr_s1.chromosomes.astype(str) == chr_plot
                ax.plot(sr_s1.positions[mask_s1], sr_s1.lod[mask_s1],
                        color='#4F6EAF', lw=1.5, label='conditional signal 1')
                if not lmm_pc.empty:
                    ax.plot(lmm_pc['pos_mb'], lmm_pc['lod_py'],
                            color='#D83026', lw=1.2, ls='--', alpha=0.8,
                            label='LMM (Python)')
                ax.set(xlabel=f'position chr{chr_plot} (Mb)', ylabel='LOD',
                       title=f'Piezo1 signal 1 vs LMM  chr{chr_plot}',
                       xlim=(0, None), ylim=(0, None))
                ax.legend(fontsize=9); sns.despine(ax=ax)
                plt.tight_layout(); save_plot(fig_cm, 'p4_signal1_vs_lmm.png')

            lod_drop = (float(pk0['lod']) - float(signals[-1].lod.max())
                        if len(signals) > 1 else 0.0)
            record('scan_conditional', float(pk0['lod']) > 0,
                   f'n_signals={len(signals)}  signal1_lod={pk0["lod"]:.3f}  '
                   f'lod_drop={lod_drop:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(os.path.join(PY_DATA, 'phase3_qc_summary.csv'), index=False)
hi('\n' + '='*65)
hi(f'{"section":<28} {"status":<6} detail')
hi('─'*65)
for row in qc_rows:
    st = 'PASS' if row['pass'] is True else ('NA' if row['pass'] is None else 'FAIL')
    hi(f'  {row["section"]:<26} {st:<6} {row["detail"]}')
hi(f'  cell_type: {args.cell_type}  n_genes: {len(gene_set)}')
hi(f'  outputs:   {PY_DATA}/')
hi(f'  plots:     {PLOTS}/')
hi('='*65)
impl   = [r for r in qc_rows if r['pass'] is not None]
passed = [r for r in impl if r['pass'] is True]
sys.exit(0 if not impl or len(passed)==len(impl) else 1)
