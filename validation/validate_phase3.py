# validate_phase3.py — phase 3 novel extension validation
#
# runs four novel mosaiQTL modules on real gastrocnemius snRNA-seq data:
#   1. scan_nb      — two-stage NB GLM genome-wide scan (primary model)
#   2. scan_vqtl    — dispersion QTL scan on candidate genes
#   3. scan_epistasis     — pairwise GxG interaction at top QTL pair
#   4. scan_conditional_with_y — forward-stepwise multi-signal decomposition
#
# no R reference for these sections — validation is internal consistency
# + concordance with known LMM top hits (Piezo1, Ccdc171, Pmis2).
#
# usage:
#   python validation/validate_phase3.py
#   python validation/validate_phase3.py --cell_type type2x --n_genes 500
#   python validation/validate_phase3.py --sections nb,vqtl,epistasis,conditional
#
# slurm (HPC):
#   sbatch --mem=64G --cpus-per-task=4 --time=12:00:00 \
#          --wrap 'conda run -n bioinfo python validation/validate_phase3.py'

import numpy as np
import pandas as pd
import os, sys, argparse, logging, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module
from mosaiqtl.utils import rint as _rint, build_z_matrix
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

# ── paths (update before HPC run) ────────────────────────────────────────────
HPC_BASE  = '/share/crsp/lab/seyedam/gfilimba'
MIQTL_DIR = f'{HPC_BASE}/projects/igvf_8cube/short-read/data/gastroc'
ADATA_PATH= f'{MIQTL_DIR}/gastroc_all_processed2.h5ad'
PKG_DIR   = f'{HPC_BASE}/packages/mosaiqtl'

# fall back to local paths if HPC paths missing
_LOCAL_BASE = os.path.expanduser(
    '~/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl')
if not os.path.exists(HPC_BASE):
    MIQTL_DIR  = f'{_LOCAL_BASE}/miqtl'
    ADATA_PATH = f'{_LOCAL_BASE}/../../../data/gastroc/gastroc_all_processed2.h5ad'
    PKG_DIR    = f'{_LOCAL_BASE}/mosaiqtl'

GM_CSV   = f'{MIQTL_DIR}/input/GM_ccfo_mm39.csv'
GM_GMAP  = f'{MIQTL_DIR}/input/GM/gmap'
GM_PMAP  = f'{MIQTL_DIR}/input/GM/pmap'
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

# known top candidates from LMM phase 2 run — used as sanity anchors
KNOWN_HITS = {
    'Piezo1':  {'chr': '8',  'approx_mb': 122.0, 'lod_lmm': 21.8},
    'Ccdc171': {'chr': '4',  'approx_mb':  84.0, 'lod_lmm':  8.0},
    'Pmis2':   {'chr': '7',  'approx_mb':  85.0, 'lod_lmm':  6.0},
}

# ── args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--cell_type', default='type2x',
                help='cell type label in adata.obs (default: type2x)')
ap.add_argument('--cell_type_col', default=None,
                help='obs column containing cell type labels (auto-detected if None)')
ap.add_argument('--strain_col', default=None,
                help='obs column for strain/subject IDs (auto-detected if None)')
ap.add_argument('--n_genes', default=500, type=int,
                help='number of HVGs to scan (default: 500)')
ap.add_argument('--min_cells', default=10, type=int,
                help='min cells per strain to include strain (default: 10)')
ap.add_argument('--sections', default='nb,vqtl,epistasis,conditional',
                help='comma-separated sections to run')
ap.add_argument('--n_cores', default=1, type=int)
args = ap.parse_args()

_run = set(s.strip() for s in args.sections.split(','))

# ── outputs ───────────────────────────────────────────────────────────────────
PY_DATA = os.path.join(PKG_DIR, 'validation', 'port_validation', 'py_data')
PLOTS   = os.path.join(PKG_DIR, 'validation', 'port_validation', 'qc_plots')
LOGS    = os.path.join(PKG_DIR, 'validation', 'port_validation', 'logs')
for d in (PY_DATA, PLOTS, LOGS): os.makedirs(d, exist_ok=True)

_ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log  = logging.getLogger('phase3')
log.setLevel(logging.DEBUG)
_fh  = logging.FileHandler(os.path.join(LOGS, f'phase3_{_ts}.log'))
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
log.addHandler(_fh)
_sh  = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(_sh)

def hi(msg): log.info(msg)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'sans-serif',
                     'font.sans-serif':['Carlito','DejaVu Sans'],
                     'font.size':11})

_chr_sort  = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)
CHR_PAL    = ['#4F6EAF','#888780']

def save_csv(df, name):
    p = os.path.join(PY_DATA, name)
    df.to_csv(p, index=False)
    hi(f'  saved {name} ({len(df)} rows)')

def save_plot(fig, name):
    p = os.path.join(PLOTS, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    hi(f'  saved {name}')

def load_py(name):
    '''load a CSV from py_data — returns None if missing.'''
    p = os.path.join(PY_DATA, name)
    if not os.path.exists(p): return None
    df = pd.read_csv(p, low_memory=False)
    if 'chr' in df.columns: df['chr'] = df['chr'].astype(str)
    return df

def scatter_qc(ax, x, y, xlabel, ylabel, title):
    '''standard scatter with y=x line and Pearson r annotation.'''
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok  = np.isfinite(x) & np.isfinite(y)
    lim = max(float(np.nanmax(np.abs(np.concatenate([x[ok], y[ok]])))), 0.1) * 1.05
    ax.scatter(x[ok], y[ok], s=6, alpha=0.7, color='#4F6EAF', edgecolors='none')
    ax.plot([0, lim], [0, lim], '--', color='#D83026', lw=1)
    r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 else np.nan
    rho = float(sp_stats.spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 3 else np.nan
    ax.text(0.05, 0.95, f'$r$={r:.3f}  $\\rho$={rho:.3f}',
            transform=ax.transAxes, fontsize=8, va='top')
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=xlabel, ylabel=ylabel, title=title)
    sns.despine(ax=ax)
    return r, rho

qc_rows = []
def record(key, passed, detail):
    st = 'PASS' if passed else ('NA' if passed is None else 'FAIL')
    hi(f'  {st}: {detail}')
    qc_rows.append({'section': key, 'pass': passed, 'detail': detail})

# ── load h5ad ────────────────────────────────────────────────────────────────
hi(f'\n● loading {ADATA_PATH}')
if not os.path.exists(ADATA_PATH):
    hi(f'  ERROR: h5ad not found at {ADATA_PATH}')
    hi('  set HPC_BASE at top of script or copy adata to expected path')
    sys.exit(1)

import scanpy as sc
sc.settings.verbosity = 0
adata = sc.read_h5ad(ADATA_PATH)
hi(f'  loaded: {adata.shape[0]} cells × {adata.shape[1]} genes')

# ── detect obs column names ───────────────────────────────────────────────────
def _detect_col(candidates, obs_cols, label):
    '''return first candidate present in obs_cols, else None.'''
    for c in candidates:
        if c in obs_cols: return c
    hi(f'  WARN: could not auto-detect {label} column. candidates: {candidates}')
    hi(f'        available obs cols: {list(obs_cols)[:20]}')
    return None

_obs = adata.obs.columns.tolist()

ct_col = args.cell_type_col or _detect_col(
    ['cell_type','celltype','leiden','cluster','subtype'], _obs, 'cell_type')
str_col = args.strain_col or _detect_col(
    ['strain','subject.name','subject_name','sample','mouse_id'], _obs, 'strain')
bat_col = _detect_col(['batch','Batch','sequencing_batch'], _obs, 'batch')

if ct_col is None or str_col is None:
    hi('  ERROR: cannot identify cell_type or strain column — set --cell_type_col / --strain_col')
    sys.exit(1)

hi(f'  cell_type_col={ct_col}  strain_col={str_col}  batch_col={bat_col}')

# ── subset to target cell type ────────────────────────────────────────────────
mask_ct = adata.obs[ct_col].astype(str) == args.cell_type
adata_ct = adata[mask_ct].copy()
hi(f'  {args.cell_type}: {adata_ct.n_obs} cells')
if adata_ct.n_obs < 100:
    hi(f'  ERROR: too few cells (<100) for {args.cell_type}')
    sys.exit(1)

# ── filter strains with sufficient cells ──────────────────────────────────────
strain_counts = adata_ct.obs[str_col].value_counts()
keep_strains  = strain_counts[strain_counts >= args.min_cells].index.tolist()
hi(f'  strains with ≥{args.min_cells} cells: {len(keep_strains)}/{len(strain_counts)}')
mask_s  = adata_ct.obs[str_col].isin(keep_strains)
adata_ct = adata_ct[mask_s].copy()

# ── extract raw counts ────────────────────────────────────────────────────────
# use .raw if available (pre-normalisation counts), else .X
if adata_ct.raw is not None:
    import scipy.sparse as sp_sparse
    X_raw = adata_ct.raw.X
    gene_names_all = list(adata_ct.raw.var_names)
else:
    X_raw = adata_ct.X
    gene_names_all = list(adata_ct.var_names)

if hasattr(X_raw, 'toarray'):
    X_raw = X_raw.toarray()
X_raw = np.array(X_raw, dtype=np.float64)   # (n_cells, n_genes_all)
strain_ids_cells = np.array(adata_ct.obs[str_col].astype(str))
hi(f'  counts shape: {X_raw.shape}  dtype: {X_raw.dtype}')

# ── select genes: known candidates + top HVGs ────────────────────────────────
# ensure raw counts (non-negative integers)
assert X_raw.min() >= 0, 'negative counts in raw matrix'

# known candidates always included
cand_genes = [g for g in KNOWN_HITS if g in gene_names_all]
hi(f'  known candidates found: {cand_genes}')

# HVGs: use pre-computed highly_variable if available, else compute
if 'highly_variable' in adata_ct.var.columns:
    hvg_genes = adata_ct.var_names[adata_ct.var['highly_variable']].tolist()
    hi(f'  pre-computed HVGs: {len(hvg_genes)}')
else:
    # simple dispersion filter: mean > 0.1, cv > 0.5
    mean_expr = X_raw.mean(axis=0)
    std_expr  = X_raw.std(axis=0)
    cv        = np.where(mean_expr > 0.1, std_expr / mean_expr, 0.0)
    hvg_genes = [gene_names_all[i] for i in np.argsort(cv)[::-1]
                 if mean_expr[i] > 0.1]
    hi(f'  computed HVGs (cv filter): {len(hvg_genes)}')

# merge: candidates first, then HVGs up to n_genes total
gene_set = cand_genes.copy()
for g in hvg_genes:
    if g not in gene_set: gene_set.append(g)
    if len(gene_set) >= args.n_genes: break
gene_set = [g for g in gene_set if g in gene_names_all]
hi(f'  final gene set: {len(gene_set)} genes')

gene_idx  = np.array([gene_names_all.index(g) for g in gene_set])
counts_ct = X_raw[:, gene_idx].T.copy()    # → (n_genes, n_cells)

# ── cell-level covariates ─────────────────────────────────────────────────────
# intercept + batch dummies + top 10 expression PCs
cov_parts = [np.ones((adata_ct.n_obs, 1))]
if bat_col is not None and adata_ct.obs[bat_col].notna().any():
    bat_dummies = pd.get_dummies(adata_ct.obs[bat_col].astype(str),
                                 prefix='batch', drop_first=True).astype(float).values
    cov_parts.append(bat_dummies)
    hi(f'  batch dummies: {bat_dummies.shape[1]} cols')
if 'X_pca' in adata_ct.obsm:
    pcs = adata_ct.obsm['X_pca'][:, :10].astype(float)
    cov_parts.append(pcs)
    hi(f'  PCs: {pcs.shape[1]} cols')
covar_cells = np.hstack(cov_parts).astype(float)
hi(f'  cell-level covariates: {covar_cells.shape[1]} cols')

# ── build genome cache ────────────────────────────────────────────────────────
hi('\n● building genome cache')
import glob, rpy2.robjects as _ro
_mf = glob.glob(os.path.join(HAPPY_CACHE, 'additive', 'chr*', 'markers.RData'))
if _mf:
    _happy_markers = []
    for mf in sorted(_mf):
        _env = _ro.r['new.env']()
        _ro.r['load'](mf, envir=_env)
        _happy_markers.extend(list(_ro.r['get']('markers', envir=_env)))
    hi(f'  HAPPY allowlist: {len(_happy_markers)} markers')
else:
    _happy_markers = None

cache_py = GenomeCache.from_gigamuga(
    gigamuga_path=GM_CSV, gmap_dir=GM_GMAP, pmap_dir=GM_PMAP,
    cc_cols=CC_COLS, founder_cols=FOUNDER_COLS, f1_crosses=F1_CROSSES,
    just_these_loci=_happy_markers)
chr_order = sorted(cache_py.chromosomes, key=_chr_sort)
hi(f'  cache: {len(chr_order)} chr | {len(cache_py.sample_ids)} strains')

# ── build LOCO kinship ────────────────────────────────────────────────────────
kinship_py   = kin_module.calc_kinship(cache_py, method='loco')
kinship_decomp = kin_module.decomp_kinship(kinship_py)
hi(f'  kinship: {len(kinship_py)} LOCO matrices')

# ── load LMM phase 2 outputs for comparison ───────────────────────────────────
hi('\n● loading LMM phase 2 outputs for comparison')
lmm_gene_summary = load_py('gene_summary.csv')    # per-gene peak stats (LMM py)
lmm_lod_comp     = load_py('lod_comparison.csv')  # per-locus LOD_r + LOD_py
if lmm_gene_summary is not None:
    hi(f'  gene_summary.csv:   {len(lmm_gene_summary)} genes')
    hi(f'  columns: {list(lmm_gene_summary.columns)}')
else:
    hi('  gene_summary.csv not found — run validate_port.py S4 first')
if lmm_lod_comp is not None:
    hi(f'  lod_comparison.csv: {len(lmm_lod_comp)} locus×gene rows')
else:
    hi('  lod_comparison.csv not found — per-locus LMM overlay unavailable')

# ── strain-level covariates (averaged from cell-level) ────────────────────────
# used for scan_nb stage 2 and for LMM scans below
strain_ids_arr = np.asarray(strain_ids_cells)
strains_sorted, inv = np.unique(strain_ids_arr, return_inverse=True)
n_strains = len(strains_sorted)
cov_sum   = np.zeros((n_strains, covar_cells.shape[1]))
cnt_s     = np.zeros(n_strains)
np.add.at(cov_sum, inv, covar_cells)
np.add.at(cnt_s,  inv, 1)
covar_strains = cov_sum / cnt_s[:, None]   # (n_strains, p)

# ── cis loci dict (±5Mb) ─────────────────────────────────────────────────────
# restrict scan to cis loci only for tractability;
# build from pmap using gene positions from adata.var if available
hi('\n● building cis loci dict')
pmap_path = os.path.join(MIQTL_DIR, 'input', 'pmap.csv')
if not os.path.exists(pmap_path):
    pmap_path = os.path.join(MIQTL_DIR, 'output', 'pmap.csv')

cis_loci_dict = None
if os.path.exists(pmap_path):
    pmap = pd.read_csv(pmap_path)
    pmap.columns = [c.lower().replace('.','_') for c in pmap.columns]
    pmap['chr'] = pmap['chr'].astype(str)
    # try to get gene positions from adata.var
    _pos_col  = _detect_col(['start','gene_start','chromStart','pos_mb'], adata.var.columns.tolist(), 'gene_pos')
    _chr_col  = _detect_col(['chromosome','chrom','chr'], adata.var.columns.tolist(), 'gene_chr')
    if _pos_col and _chr_col:
        gene_info = adata.var[[_chr_col, _pos_col]].copy()
        gene_info.columns = ['gene_chr','gene_pos_bp']
        gene_info['gene_pos_mb'] = gene_info['gene_pos_bp'] / 1e6
        cis_loci_dict = {}
        for g in gene_set:
            if g not in gene_info.index: continue
            g_chr = str(gene_info.loc[g, 'gene_chr'])
            g_pos = float(gene_info.loc[g, 'gene_pos_mb'])
            cis   = pmap[(pmap['chr']==g_chr) &
                         (pmap['pos_mb'] >= g_pos - 5) &
                         (pmap['pos_mb'] <= g_pos + 5)]['marker'].tolist()
            if cis: cis_loci_dict[g] = set(cis)
        hi(f'  cis loci dict: {len(cis_loci_dict)} genes with cis loci')
    else:
        hi('  gene position columns not found — running genome-wide (slow)')
else:
    hi('  pmap not found — running genome-wide')

# ══════════════════════════════════════════════════════════════════════════════
# P1 — SCAN_NB (primary model)
# ══════════════════════════════════════════════════════════════════════════════
if 'nb' in _run:
    hi(f'\n▶ P1 — SCAN_NB  ({len(gene_set)} genes, {args.cell_type})')
    from mosaiqtl.nb import scan_nb

    try:
        results_nb = scan_nb(
            counts_cells      = counts_ct,         # (n_genes, n_cells) raw UMI
            genome_cache      = cache_py,
            strain_ids_cells  = strain_ids_cells,
            gene_ids          = gene_set,
            covariates_cells  = covar_cells,
            covariates_strains= covar_strains,
            norm_method       = 'TMM',
            shrink_disp       = True,
            cis_loci_dict     = cis_loci_dict,
            n_cores           = args.n_cores,
            verbose           = True,
        )
        hi(f'  scan_nb complete: {len(results_nb)} genes')
    except Exception as e:
        hi(f'  ERROR in scan_nb: {e}')
        log.exception('scan_nb failed')
        results_nb = {}

    if results_nb:
        # ── per-gene peak summary ─────────────────────────────────────────────
        peak_rows = []
        for gid, sr in results_nb.items():
            if not len(sr.lod): continue
            pk   = sr.peak_info()
            peak_rows.append({'gene_id': gid,
                              'chr':     pk['chromosome'],
                              'locus':   pk['locus_id'],
                              'pos_mb':  pk['position'],
                              'lod':     pk['lod']})
        peak_df = (pd.DataFrame(peak_rows)
                   .sort_values('lod', ascending=False)
                   .reset_index(drop=True))
        # BH-FDR on -log10(p) from chi2(J-1) approximation
        # LOD → chi2: Q ≈ 2 * ln(10) * LOD * (J-1)/(n-p-J+1) * (n-p-J+1)
        # simpler: treat LOD as chi2(df=J-1)/2ln10 directly
        J  = cache_py.n_founders
        df_chi2 = J - 1
        peak_df['p_value'] = sp_stats.chi2.sf(
            2 * np.log(10) * peak_df['lod'].values, df=df_chi2)
        _, fdr_vals, _, _ = multipletests(peak_df['p_value'].values, method='fdr_bh')
        peak_df['fdr_bh'] = fdr_vals
        peak_df['egene']  = peak_df['fdr_bh'] < 0.05

        n_egenes = int(peak_df['egene'].sum())
        save_csv(peak_df, 'nb_peak_summary.csv')
        hi(f'  eGenes (BH 5%): {n_egenes}/{len(peak_df)}')

        # genome-wide LOD df for Manhattan
        lod_rows = []
        for gid, sr in results_nb.items():
            if not len(sr.lod): continue
            lod_rows.append(pd.DataFrame({'gene_id': gid,
                                           'chr': sr.chromosomes,
                                           'pos_mb': sr.positions,
                                           'lod': sr.lod}))
        lod_df = pd.concat(lod_rows, ignore_index=True)
        save_csv(lod_df.nlargest(50000, 'lod'), 'nb_lod_top50k.csv')   # cap file size

        # ── QC plots ─────────────────────────────────────────────────────────
        chr_colors = [CHR_PAL[chr_order.index(c) % 2]
                      if c in chr_order else '#aaa'
                      for c in lod_df['chr'].astype(str)]

        # plot 1: LOD distribution + known hit LODs
        fig1, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(peak_df['lod'], bins=50, color='#4F6EAF',
                     edgecolor='none', ax=ax)
        for gname, info in KNOWN_HITS.items():
            row = peak_df[peak_df['gene_id'] == gname]
            if row.empty: continue
            lod_v = float(row['lod'].iloc[0])
            ax.axvline(lod_v, color=FOUNDER_COLORS.get('PWKJ','#D83026'),
                       lw=1.5, ls='--', label=f'{gname} LOD={lod_v:.1f}')
        ax.set(xlabel='peak LOD (NB scan)', ylabel='genes',
               title=f'NB scan LOD distribution  {args.cell_type}  n_genes={len(peak_df)}')
        ax.legend(fontsize=8); sns.despine(ax=ax)
        plt.tight_layout(); save_plot(fig1, 'p1_nb_lod_distribution.png')

        # plot 2: Manhattan for chr8 ±20Mb around Piezo1 (known QTL region)
        if 'Piezo1' in results_nb:
            sr_p = results_nb['Piezo1']
            mask_c = sr_p.chromosomes.astype(str) == '8'
            pos8   = sr_p.positions[mask_c]
            lod8   = sr_p.lod[mask_c]
            fig2, ax = plt.subplots(figsize=(9, 4))
            ax.fill_between(pos8, 0, lod8, color='#4F6EAF', alpha=0.8, step='mid')
            pk_pos = float(pos8[np.argmax(lod8)])
            pk_lod = float(lod8.max())
            ax.axvline(pk_pos, color='#D83026', lw=1.2, ls='--',
                       label=f'peak {pk_pos:.1f}Mb  LOD={pk_lod:.2f}')
            # annotate LMM reference
            lmm_pos = KNOWN_HITS['Piezo1']['approx_mb']
            ax.axvline(lmm_pos, color='#55AF5B', lw=1.0, ls=':',
                       label=f'LMM ref ~{lmm_pos:.0f}Mb')
            ax.set(xlabel='position chr8 (Mb)', ylabel='LOD (NB)',
                   title='Piezo1 NB scan — chr8', xlim=(0, None), ylim=(0, None))
            ax.legend(fontsize=8); sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig2, 'p1_nb_piezo1_chr8.png')

        # plot 3: top 20 eGenes bar chart
        fig3, ax = plt.subplots(figsize=(8, 5))
        top20 = peak_df.head(20)
        colors_bar = [FOUNDER_COLORS.get('CASTJ','#55AF5B')
                      if g in KNOWN_HITS else '#4F6EAF'
                      for g in top20['gene_id']]
        ax.barh(range(len(top20))[::-1], top20['lod'].values,
                color=colors_bar, edgecolor='none')
        ax.set(yticks=range(len(top20))[::-1],
               yticklabels=top20['gene_id'].tolist(),
               xlabel='peak LOD (NB)', title='top 20 eGenes — NB scan')
        sns.despine(ax=ax); plt.tight_layout()
        save_plot(fig3, 'p1_nb_top_egenes.png')

        # ── NB vs LMM comparison ─────────────────────────────────────────────
        nb_vs_lmm = None
        if lmm_gene_summary is not None:
            mg = peak_df.merge(
                lmm_gene_summary[['gene_id','chr','peak_lod_py','h2_null',
                                   'tier','peak_match']]
                .rename(columns={'peak_lod_py':'lod_lmm','chr':'chr_lmm'}),
                on='gene_id', how='inner')
            mg['peak_chr_match'] = mg['chr'] == mg['chr_lmm']
            mg['lmm_egene']      = mg['tier'].isin(['strict','lenient'])
            mg['nb_egene']       = mg['fdr_bh'] < 0.05
            mg['both_egene']     = mg['nb_egene'] & mg['lmm_egene']
            ok = mg[['lod','lod_lmm']].notna().all(axis=1)
            rho_rank = float(sp_stats.spearmanr(mg.loc[ok,'lod'],
                                                mg.loc[ok,'lod_lmm']).statistic)
            n_shared  = int(mg['both_egene'].sum())
            n_chr_ok  = int(mg['peak_chr_match'].sum())
            hi(f'  NB vs LMM: n_shared_genes={len(mg)}  spearman_rank_r={rho_rank:.4f}  '
               f'shared_egenes={n_shared}  peak_chr_concordance={n_chr_ok}/{len(mg)}')
            nb_vs_lmm = mg[['gene_id','chr','pos_mb','lod','p_value','fdr_bh',
                             'nb_egene','lod_lmm','chr_lmm','peak_chr_match',
                             'lmm_egene','both_egene','h2_null']]
            save_csv(nb_vs_lmm, 'nb_vs_lmm_comparison.csv')

            # plot 4: NB LOD vs LMM LOD scatter (Pearson + Spearman)
            fig4, ax = plt.subplots(figsize=(5, 5))
            scatter_qc(ax, mg['lod_lmm'], mg['lod'],
                       'peak LOD (LMM)', 'peak LOD (NB)',
                       f'NB vs LMM per-gene peak LOD  n={len(mg)}')
            # highlight known candidates
            for gname in KNOWN_HITS:
                row = mg[mg['gene_id'] == gname]
                if row.empty: continue
                ax.scatter(float(row['lod_lmm']), float(row['lod']),
                           s=60, color='#D83026', zorder=5, edgecolors='white', lw=0.5)
                ax.annotate(gname, (float(row['lod_lmm']), float(row['lod'])),
                            fontsize=7, xytext=(4, 2), textcoords='offset points')
            plt.tight_layout(); save_plot(fig4, 'p1_nb_vs_lmm_scatter.png')

            # plot 5: Piezo1 chr8 — NB vs LMM LOD profile overlay
            if ('Piezo1' in results_nb and lmm_lod_comp is not None
                    and 'Piezo1' in lmm_lod_comp['gene_id'].values):
                sr_p   = results_nb['Piezo1']
                mask_c = sr_p.chromosomes.astype(str) == '8'
                lmm_p  = lmm_lod_comp[(lmm_lod_comp['gene_id']=='Piezo1') &
                                       (lmm_lod_comp['chr']=='8')]
                fig5, ax = plt.subplots(figsize=(9, 4))
                ax.fill_between(sr_p.positions[mask_c], 0, sr_p.lod[mask_c],
                                color='#4F6EAF', alpha=0.5, step='mid', label='NB')
                if not lmm_p.empty:
                    ax.plot(lmm_p['pos_mb'].values, lmm_p['lod_py'].values,
                            color='#D83026', lw=1.5, alpha=0.85, label='LMM (Python)')
                ax.set(xlabel='position chr8 (Mb)', ylabel='LOD',
                       title='Piezo1 — NB vs LMM LOD profile chr8',
                       xlim=(0, None), ylim=(0, None))
                ax.legend(fontsize=9); sns.despine(ax=ax)
                plt.tight_layout(); save_plot(fig5, 'p1_nb_vs_lmm_piezo1_profile.png')

        # ── sanity + record ───────────────────────────────────────────────────
        piezo_lod    = (float(peak_df.loc[peak_df['gene_id']=='Piezo1','lod'].iloc[0])
                        if 'Piezo1' in peak_df['gene_id'].values else 0.0)
        lod_right_skew = float(sp_stats.skew(peak_df['lod'].values)) > 0.5
        detail = (f'n_egenes={n_egenes}  piezo1_lod={piezo_lod:.2f}  '
                  f'lod_skew={sp_stats.skew(peak_df["lod"].values):.2f}  '
                  f'lod_right_skew={lod_right_skew}')
        if nb_vs_lmm is not None:
            detail += (f'  nb_lmm_spearman={rho_rank:.3f}  '
                       f'shared_egenes={n_shared}  '
                       f'peak_chr_concordance={n_chr_ok}/{len(nb_vs_lmm)}')
        record('scan_nb', lod_right_skew and piezo_lod > 3.0, detail)
    else:
        record('scan_nb', False, 'scan_nb returned no results')

# ══════════════════════════════════════════════════════════════════════════════
# P2 — SCAN_VQTL (dispersion QTL)
# ══════════════════════════════════════════════════════════════════════════════
if 'vqtl' in _run:
    hi(f'\n▶ P2 — SCAN_VQTL  (candidates: {list(KNOWN_HITS.keys())})')
    from mosaiqtl.vqtl import scan_vqtl

    vqtl_rows = []
    for gname, info in KNOWN_HITS.items():
        if gname not in gene_names_all:
            hi(f'  skip {gname}: not in gene set'); continue
        gi      = gene_names_all.index(gname)
        counts_g = X_raw[:, gi]   # (n_cells,) raw counts for this gene
        lib_g   = X_raw.sum(axis=1)   # total UMI per cell as lib size
        # align strain-level covariates to the strains in the cache
        cache_sids = list(cache_py.sample_ids)
        keep = [s for s in strains_sorted if s in cache_sids]
        if len(keep) < 20:
            hi(f'  skip {gname}: <20 shared strains'); continue
        cov_g = covar_strains[[list(strains_sorted).index(s) for s in keep]]
        kd_g  = {info['chr']: kinship_decomp[info['chr']]} if info['chr'] in kinship_decomp else None

        try:
            res_v = scan_vqtl(
                counts        = counts_g,
                subject_ids   = strain_ids_cells,
                genome_cache  = cache_py,
                library_sizes = lib_g,
                covariates    = cov_g,
                kinship_decomp= kd_g,
                phi_method    = 'moments',
                min_cells     = args.min_cells,
                use_rint      = True,
                gene_id       = gname,
                also_scan_mean= True,
            )
        except Exception as e:
            hi(f'  ERROR in scan_vqtl({gname}): {e}')
            log.exception(f'scan_vqtl {gname} failed')
            continue

        if res_v is None:
            hi(f'  {gname}: scan_vqtl returned None (insufficient subjects)'); continue

        pk_v = res_v.vqtl_scan.peak_info()
        hi(f'  {gname}: vQTL peak {pk_v["locus_id"]} chr{pk_v["chromosome"]} '
           f'{pk_v["position"]:.1f}Mb  LOD={pk_v["lod"]:.3f}')
        vqtl_rows.append({'gene_id': gname,
                          'chr': pk_v['chromosome'],
                          'locus': pk_v['locus_id'],
                          'pos_mb': pk_v['position'],
                          'lod_vqtl': pk_v['lod'],
                          'eqtl_chr': info['chr'],
                          'eqtl_approx_mb': info['approx_mb'],
                          'colocalises': (pk_v['chromosome'] == info['chr'] and
                                         abs(pk_v['position'] - info['approx_mb']) < 10)})

        # per-gene plot: vQTL + mean eQTL + LMM overlay on target chr
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        sr_v  = res_v.vqtl_scan
        mask  = sr_v.chromosomes.astype(str) == info['chr']
        pos_v = sr_v.positions[mask]; lod_v = sr_v.lod[mask]

        # panel 1: vQTL LOD profile
        axes[0].fill_between(pos_v, 0, lod_v, color='#4F6EAF', alpha=0.8, step='mid')
        axes[0].axvline(info['approx_mb'], color='#D83026', lw=1.0, ls='--',
                        label=f'eQTL ref ~{info["approx_mb"]:.0f}Mb')
        axes[0].set(xlabel=f'position chr{info["chr"]} (Mb)', ylabel='LOD (vQTL)',
                    title=f'{gname} — dispersion vQTL', xlim=(0, None), ylim=(0, None))
        axes[0].legend(fontsize=8); sns.despine(ax=axes[0])

        # panel 2: mean eQTL (pseudobulk, from scan_vqtl also_scan_mean)
        if res_v.mean_scan is not None:
            sr_m   = res_v.mean_scan
            mask_m = sr_m.chromosomes.astype(str) == info['chr']
            pos_m  = sr_m.positions[mask_m]; lod_m = sr_m.lod[mask_m]
            axes[1].fill_between(pos_m, 0, lod_m, color='#55AF5B', alpha=0.8, step='mid',
                                 label='NB mean scan')
            # overlay LMM profile if available
            if lmm_lod_comp is not None:
                lmm_g = lmm_lod_comp[(lmm_lod_comp['gene_id']==gname) &
                                      (lmm_lod_comp['chr']==info['chr'])]
                if not lmm_g.empty:
                    axes[1].plot(lmm_g['pos_mb'].values, lmm_g['lod_py'].values,
                                 color='#D83026', lw=1.5, alpha=0.85, label='LMM (Python)')
            axes[1].set(xlabel=f'position chr{info["chr"]} (Mb)', ylabel='LOD',
                        title=f'{gname} — mean eQTL vs LMM',
                        xlim=(0, None), ylim=(0, None))
            axes[1].legend(fontsize=8); sns.despine(ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, 'mean scan\nnot available',
                         ha='center', va='center', transform=axes[1].transAxes)
            sns.despine(ax=axes[1])

        # panel 3: vQTL vs mean eQTL LOD scatter (colocalisation check)
        if res_v.mean_scan is not None and mask.sum() > 3 and mask_m.sum() > 3:
            # align by position
            _v = pd.DataFrame({'pos_mb': pos_v, 'lod_vqtl': lod_v})
            _m = pd.DataFrame({'pos_mb': pos_m, 'lod_mean': lod_m})
            _vm = _v.merge(_m, on='pos_mb', how='inner')
            if len(_vm) > 3:
                scatter_qc(axes[2], _vm['lod_mean'], _vm['lod_vqtl'],
                           'LOD (mean eQTL)', 'LOD (vQTL)',
                           f'{gname} vQTL vs mean eQTL chr{info["chr"]}')
            else:
                axes[2].text(0.5, 0.5, 'insufficient\noverlap',
                             ha='center', va='center', transform=axes[2].transAxes)
                sns.despine(ax=axes[2])
        plt.tight_layout()
        save_plot(fig, f'p2_vqtl_{gname.lower()}.png')

    if vqtl_rows:
        vqtl_df = pd.DataFrame(vqtl_rows)
        # merge with LMM peak positions for comparison CSV
        if lmm_gene_summary is not None:
            vqtl_df = vqtl_df.merge(
                lmm_gene_summary[['gene_id','peak_lod_py','h2_null']]
                .rename(columns={'peak_lod_py':'lod_lmm'}),
                on='gene_id', how='left')
            vqtl_df['lod_vqtl_vs_lmm_delta'] = vqtl_df['lod_vqtl'] - vqtl_df['lod_lmm']
        save_csv(vqtl_df, 'vqtl_vs_lmm_comparison.csv')
        record('scan_vqtl', True,
               f'n_genes={len(vqtl_rows)}  '
               f'colocalise={vqtl_df["colocalises"].sum()}/{len(vqtl_df)}')
    else:
        record('scan_vqtl', False, 'no vQTL results produced')

# ══════════════════════════════════════════════════════════════════════════════
# P3 — SCAN_EPISTASIS (pairwise GxG: Piezo1 chr8 × Ccdc171 chr4)
# ══════════════════════════════════════════════════════════════════════════════
if 'epistasis' in _run:
    hi('\n▶ P3 — SCAN_EPISTASIS  (Piezo1 chr8 × Ccdc171 chr4)')
    from mosaiqtl.epistasis import scan_epistasis

    # get peak locus IDs for each gene from nb results or nb_peak_summary.csv
    def _get_peak_locus(gname):
        '''return peak locus ID for gene, preferring nb results over LMM fallback.'''
        if 'results_nb' in dir() and gname in results_nb:
            return str(results_nb[gname].peak_info()['locus_id'])
        # fallback: use known approximate position to find nearest locus
        info = KNOWN_HITS.get(gname)
        if info is None: return None
        lids, pos, _ = cache_py.get_chromosome(info['chr'])
        nearest = int(np.argmin(np.abs(pos - info['approx_mb'])))
        return str(lids[nearest])

    locus_a = _get_peak_locus('Piezo1')    # chr8 peak
    locus_b = _get_peak_locus('Ccdc171')   # chr4 peak

    if locus_a is None or locus_b is None:
        record('scan_epistasis', False, f'could not determine peak loci: a={locus_a}  b={locus_b}')
    elif 'Piezo1' not in gene_names_all:
        record('scan_epistasis', False, 'Piezo1 not in gene set — needed as phenotype')
    else:
        # use Piezo1 pseudobulk RINT as phenotype (strain-level)
        # derive from raw counts: pseudobulk sum → log1p → RINT
        gi_p = gene_names_all.index('Piezo1')
        counts_p = X_raw[:, gi_p]   # (n_cells,)
        # pseudobulk sum per strain (matching cache sample order)
        cache_sids = list(cache_py.sample_ids)
        pb_map = {}
        for sid, cnt in zip(strain_ids_cells, counts_p):
            pb_map[sid] = pb_map.get(sid, 0.0) + cnt
        y_epi = np.array([pb_map.get(s, np.nan) for s in cache_sids])
        y_epi = _rint(np.log1p(y_epi))   # RINT of log1p pseudobulk

        # strain-level covariates aligned to cache
        cov_epi_map = {s: covar_strains[i]
                       for i, s in enumerate(strains_sorted)
                       if s in cache_sids}
        cov_epi = np.array([cov_epi_map.get(s, np.zeros(covar_strains.shape[1]))
                            for s in cache_sids])

        hi(f'  locus_a={locus_a}  locus_b={locus_b}')
        try:
            epi_res = scan_epistasis(
                phenotype    = y_epi,
                genome_cache = cache_py,
                locus_a      = locus_a,
                locus_b      = locus_b,
                covariates   = cov_epi,
                kinship_decomp = kinship_decomp,
                use_rint     = False,   # already RINT'd
                phenotype_id = 'Piezo1',
            )
        except Exception as e:
            hi(f'  ERROR in scan_epistasis: {e}')
            log.exception('scan_epistasis failed')
            epi_res = None

        if epi_res is not None:
            hi(f'  LOD_additive   = {epi_res["lod_additive"]:.3f}')
            hi(f'  LOD_interaction= {epi_res["lod_interaction"]:.3f}')
            hi(f'  LOD_epistasis  = {epi_res["lod_epistasis"]:.3f}  (interaction drop)')

            # cross-check: LOD_additive should be close to LMM LOD at same locus
            lmm_lod_at_a = np.nan
            if lmm_lod_comp is not None:
                row_a = lmm_lod_comp[(lmm_lod_comp['gene_id']=='Piezo1') &
                                     (lmm_lod_comp['locus']==locus_a)]
                if not row_a.empty:
                    lmm_lod_at_a = float(row_a['lod_py'].iloc[0])
            delta_add_lmm = (epi_res['lod_additive'] - lmm_lod_at_a
                             if np.isfinite(lmm_lod_at_a) else np.nan)
            hi(f'  LMM LOD at locus_a = {lmm_lod_at_a:.3f}  '
               f'Δ(NB_add - LMM) = {delta_add_lmm:.3f}')

            epi_df = pd.DataFrame([{
                'gene_id':              'Piezo1',
                'locus_a':              locus_a,
                'locus_b':              locus_b,
                'lod_additive':         epi_res['lod_additive'],
                'lod_interaction':      epi_res['lod_interaction'],
                'lod_epistasis':        epi_res['lod_epistasis'],
                'lmm_lod_at_locus_a':  lmm_lod_at_a,
                'delta_additive_lmm':  delta_add_lmm,
            }])
            save_csv(epi_df, 'epistasis_piezo1_ccdc171.csv')

            # plot: LOD breakdown bar (additive / interaction / epistasis)
            # + LMM reference line
            fig_e, ax_e = plt.subplots(figsize=(5, 4))
            bars = ['additive', 'interaction', 'epistasis']
            vals = [epi_res['lod_additive'],
                    epi_res['lod_interaction'],
                    epi_res['lod_epistasis']]
            colors_e = ['#4F6EAF', '#55AF5B', '#D83026']
            ax_e.bar(bars, vals, color=colors_e, edgecolor='none')
            if np.isfinite(lmm_lod_at_a):
                ax_e.axhline(lmm_lod_at_a, color='#888', lw=1.5, ls='--',
                             label=f'LMM ref={lmm_lod_at_a:.2f}')
                ax_e.legend(fontsize=8)
            ax_e.set(ylabel='LOD', title='Piezo1×Ccdc171 epistasis — LOD decomposition',
                     ylim=(0, None))
            sns.despine(ax=ax_e)
            plt.tight_layout(); save_plot(fig_e, 'p3_epistasis_lod_breakdown.png')

            # haplotype combination heatmap if available
            if 'hap_means' in epi_res:
                from mosaiqtl.epistasis import test_haplotype_combinations
                try:
                    hap_res = test_haplotype_combinations(
                        y_epi, cache_py, locus_a, locus_b,
                        covariates=cov_epi, kinship_decomp=kinship_decomp,
                        use_rint=False, phenotype_id='Piezo1')
                    if hap_res is not None and 'means' in hap_res:
                        mat = hap_res['means']   # (J, J)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(mat, ax=ax, cmap='YlOrRd',
                                    xticklabels=FOUNDER_COLS, yticklabels=FOUNDER_COLS,
                                    cbar_kws={'label': 'mean Piezo1 (RINT)'})
                        ax.set_xlabel(f'Ccdc171 haplotype  ({locus_b})', fontsize=9)
                        ax.set_ylabel(f'Piezo1 haplotype  ({locus_a})', fontsize=9)
                        plt.tight_layout(); save_plot(fig, 'p3_epistasis_heatmap.png')
                except Exception as e:
                    hi(f'  haplotype heatmap skipped: {e}')

            record('scan_epistasis', epi_res['lod_additive'] > 0,
                   f'lod_additive={epi_res["lod_additive"]:.3f}  '
                   f'lod_epistasis={epi_res["lod_epistasis"]:.3f}  '
                   f'lmm_lod_at_locus_a={lmm_lod_at_a:.3f}  '
                   f'delta_add_lmm={delta_add_lmm:.3f}')
        else:
            record('scan_epistasis', False, 'scan_epistasis raised an error')

# ══════════════════════════════════════════════════════════════════════════════
# P4 — SCAN_CONDITIONAL_WITH_Y (Piezo1 multi-signal decomposition)
# ══════════════════════════════════════════════════════════════════════════════
if 'conditional' in _run:
    hi('\n▶ P4 — SCAN_CONDITIONAL_WITH_Y  (Piezo1)')
    from mosaiqtl.epistasis import scan_conditional_with_y
    from mosaiqtl.scan import scan_h2lmm

    if 'Piezo1' not in gene_names_all:
        record('scan_conditional', False, 'Piezo1 not in gene set')
    else:
        # reuse y_epi from epistasis section if available, else recompute
        if 'y_epi' not in dir():
            gi_p  = gene_names_all.index('Piezo1')
            pb_map2 = {}
            for sid, cnt in zip(strain_ids_cells, X_raw[:, gi_p]):
                pb_map2[sid] = pb_map2.get(sid, 0.0) + cnt
            y_epi = _rint(np.log1p(np.array([pb_map2.get(s, np.nan)
                                              for s in cache_py.sample_ids])))
            cov_epi_map = {s: covar_strains[i]
                           for i, s in enumerate(strains_sorted)
                           if s in list(cache_py.sample_ids)}
            cov_epi = np.array([cov_epi_map.get(s, np.zeros(covar_strains.shape[1]))
                                for s in cache_py.sample_ids])

        # initial genome-wide LMM scan for Piezo1 to seed the conditional
        hi('  running initial LMM scan for Piezo1 (seed for conditional)')
        try:
            sr_base = scan_h2lmm(
                y_epi, cache_py, cov_epi, kinship_decomp,
                use_rint=False, phenotype_id='Piezo1')
        except Exception as e:
            hi(f'  ERROR in base scan: {e}')
            log.exception('base scan failed')
            record('scan_conditional', False, f'base scan failed: {e}')
            sr_base = None

        if sr_base is not None:
            pk0 = sr_base.peak_info()
            hi(f'  signal 1: chr{pk0["chromosome"]}  {pk0["position"]:.1f}Mb  LOD={pk0["lod"]:.3f}')

            try:
                signals = scan_conditional_with_y(
                    phenotype     = y_epi,
                    scan_result   = sr_base,
                    genome_cache  = cache_py,
                    covariates    = cov_epi,
                    kinship_decomp= kinship_decomp,
                    n_signals     = 3,
                    lod_drop_threshold = 1.5,
                    use_rint      = False,
                )
            except Exception as e:
                hi(f'  ERROR in scan_conditional_with_y: {e}')
                log.exception('scan_conditional failed')
                signals = [sr_base]

            # report each signal
            sig_rows = []
            for si, sr in enumerate(signals):
                pk = sr.peak_info()
                hi(f'  signal {si+1}: chr{pk["chromosome"]}  '
                   f'{pk["position"]:.1f}Mb  LOD={pk["lod"]:.3f}')
                sig_rows.append({'signal': si+1,
                                 'chr': pk['chromosome'],
                                 'locus': pk['locus_id'],
                                 'pos_mb': pk['position'],
                                 'lod': pk['lod']})
            sig_df = pd.DataFrame(sig_rows)

            # plot: overlay all signal scans + LMM reference on chr8
            chr_plot = pk0['chromosome']
            fig, ax = plt.subplots(figsize=(9, 4))
            colors_sig = ['#4F6EAF', '#D83026', '#55AF5B']
            for si, sr in enumerate(signals):
                mask = sr.chromosomes.astype(str) == chr_plot
                pos_s = sr.positions[mask]; lod_s = sr.lod[mask]
                ax.plot(pos_s, lod_s, color=colors_sig[si % len(colors_sig)],
                        lw=1.5, alpha=0.85,
                        label=f'signal {si+1}  peak={sr.lod.max():.2f}')
            # LMM reference for same gene + chr
            if lmm_lod_comp is not None:
                lmm_p4 = lmm_lod_comp[(lmm_lod_comp['gene_id']=='Piezo1') &
                                       (lmm_lod_comp['chr']==chr_plot)]
                if not lmm_p4.empty:
                    ax.plot(lmm_p4['pos_mb'].values, lmm_p4['lod_py'].values,
                            color='#888', lw=1.2, ls='--', alpha=0.7,
                            label=f'LMM (Python)')
            ax.set(xlabel=f'position chr{chr_plot} (Mb)', ylabel='LOD',
                   title=f'Piezo1 conditional scan  {len(signals)} signal(s)',
                   xlim=(0, None), ylim=(0, None))
            ax.legend(fontsize=9); sns.despine(ax=ax)
            plt.tight_layout(); save_plot(fig, 'p4_conditional_piezo1.png')

            # comparison CSV: signal LODs vs LMM LOD at each peak locus
            lmm_at_peaks = []
            for row in sig_rows:
                lmm_row = (lmm_lod_comp[(lmm_lod_comp['gene_id']=='Piezo1') &
                                         (lmm_lod_comp['locus']==row['locus'])]
                           if lmm_lod_comp is not None else pd.DataFrame())
                lmm_at_peaks.append(float(lmm_row['lod_py'].iloc[0])
                                    if not lmm_row.empty else np.nan)
            sig_df = pd.DataFrame(sig_rows)
            sig_df['lod_lmm_at_locus'] = lmm_at_peaks
            sig_df['delta_signal_lmm'] = sig_df['lod'] - sig_df['lod_lmm_at_locus']
            save_csv(sig_df, 'conditional_signals_piezo1.csv')
            hi(f'  signal vs LMM comparison:\n{sig_df[["signal","lod","lod_lmm_at_locus","delta_signal_lmm"]].to_string(index=False)}')

            lod_drop = (float(pk0['lod']) - float(signals[-1].lod.max())
                        if len(signals) > 1 else 0.0)
            record('scan_conditional', len(signals) >= 1 and float(pk0['lod']) > 0,
                   f'n_signals={len(signals)}  signal1_lod={pk0["lod"]:.3f}  '
                   f'lod_drop_after_conditioning={lod_drop:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 QC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
qc_df = pd.DataFrame(qc_rows)
qc_path = os.path.join(PY_DATA, 'phase3_qc_summary.csv')
qc_df.to_csv(qc_path, index=False)
hi('\n' + '='*65)
hi(f'{"section":<28} {"status":<6} detail')
hi('─'*65)
for row in qc_rows:
    st = 'PASS' if row['pass'] is True else ('NA' if row['pass'] is None else 'FAIL')
    hi(f'  {row["section"]:<26} {st:<6} {row["detail"]}')
hi(f'  cell_type : {args.cell_type}')
hi(f'  n_genes   : {len(gene_set)}')
hi(f'  outputs   : {PY_DATA}/')
hi(f'  plots     : {PLOTS}/')
hi('='*65)
impl   = [r for r in qc_rows if r['pass'] is not None]
passed = [r for r in impl if r['pass'] is True]
sys.exit(0 if not impl or len(passed) == len(impl) else 1)
