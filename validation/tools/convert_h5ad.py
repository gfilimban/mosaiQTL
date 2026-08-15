"""Convert pseudobulk h5ad to HDF5 + metadata CSV for mosaiQTL / miQTL scan.

Whole-tissue mode (default):
    reads bulk h5ad (one obs per sample)
    writes to --out/:
        counts_lognorm.h5   metadata.csv   exp_gene_info_{tissue}.csv

Cell-type mode (--celltype / --all_celltypes):
    reads cell-type h5ad (one obs per sample × subtype)
    subsets to the requested subtype
    writes counts_lognorm.h5 AND exp_gene_info_{ct}.csv to --out/celltypes/{ct}/
    metadata.csv is shared from --out/ (same samples for all cell types)

    --all_celltypes converts every subtype in the h5ad in one pass.
    --exclude_celltypes skips named subtypes (default: low-quality).

Gene coordinates (gene_id, chr, start, end, strand) are read directly from
adata.var — no external gene_info CSV required.  tss_bp is derived from those
columns (start for + strand, end for - strand) and written to exp_gene_info_*.csv
for use downstream as cis_dist_mb = (locus_pos - tss_bp) / 1e6.

Usage:
    python validation/tools/convert_h5ad.py --tissue gastroc
    python validation/tools/convert_h5ad.py --tissue gastroc --celltype myonuclei
    python validation/tools/convert_h5ad.py --tissue gastroc --list_celltypes
    python validation/tools/convert_h5ad.py --tissue ctxhipp --out validation/ctxhipp_data \\
        --h5ad /path/to/ctxhipp_all_ctpb.h5ad --all_celltypes
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import re

import anndata
import h5py
import numpy as np
import pandas as pd

from mosaiqtl.genotype.founders import STRAIN_ORDER

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

_NON_CC_ORDER: list[str] = list(STRAIN_ORDER)

_DEFAULT_BULK_H5AD = (pathlib.Path.home() / 'Documents/journey/learning/graduate'
                      '/lab/projects/igvf/analysis/qtl/input/adata'
                      '/gastroc_all_pb_bulk.h5ad')
_DEFAULT_CT_H5AD   = (pathlib.Path.home() / 'Documents/journey/learning/graduate'
                      '/lab/projects/igvf/analysis/qtl/input/adata'
                      '/gastroc_all_pb.h5ad')


def _add_tss_bp(df: pd.DataFrame) -> pd.DataFrame:
    """Add tss_bp: genomic position of the transcription start site.

    tss_bp = start  for + strand genes
    tss_bp = end    for - strand genes (TSS is at the 3' end of the annotation interval)
    Used downstream for cis_dist_mb = (locus_pos - tss_bp) / 1e6.
    Rows with strand == NaN get tss_bp = NaN.
    """
    if 'start' not in df.columns or 'strand' not in df.columns:
        return df
    tss = df['start'].copy().astype('float64')
    neg_mask = df['strand'] == '-'
    tss.loc[neg_mask] = df.loc[neg_mask, 'end'].astype('float64')
    tss.loc[df['strand'].isna()] = float('nan')
    df['tss_bp'] = tss.astype('Int64')
    return df


def _strip_gene_id_version(df: pd.DataFrame) -> pd.DataFrame:
    """Strip version suffix from gene_id (ENSMUSG00000102693.2 → ENSMUSG00000102693)."""
    if 'gene_id' in df.columns:
        df['gene_id'] = df['gene_id'].str.replace(r'\.\d+$', '', regex=True)
    return df


def _strain_sort_key(genotype: str) -> tuple[int, str]:
    if genotype in _NON_CC_ORDER:
        return (0, str(_NON_CC_ORDER.index(genotype)).zfill(3))
    return (1, genotype)


def _ct_safe(cell_type: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', cell_type.lower()).strip('_')


def _detect_celltype_col(obs: pd.DataFrame) -> str:
    """Return the obs column holding cell type labels at pseudobulk resolution.

    Prefers 'subtype' because pseudobulks are built at subtype resolution;
    'celltype' is a coarser grouping label and may map multiple subtypes to
    the same mouse, breaking the assumption of one obs per (sample, label).
    Falls back to 'celltype' for h5ads that lack a subtype column (e.g. gastroc bulk).
    Raises ValueError if neither is present.
    """
    for col in ('subtype', 'celltype'):
        if col in obs.columns:
            return col
    raise ValueError(
        f'no cell type column found in obs; expected "subtype" or "celltype". '
        f'Available columns: {list(obs.columns)}')


def _metadata_cols(obs: pd.DataFrame) -> list[str]:
    """Return the ordered list of metadata columns to write to metadata.csv.

    Always includes: sample_id, genotype, sex, dataset.
    Appended when present: body_weight_g, tissue_weight_mg  (SWE-10 covariates).
    """
    base = ['sample_id', 'genotype', 'sex', 'dataset']
    optional = ['body_weight_g', 'tissue_weight_mg']
    return base + [c for c in optional if c in obs.columns]



def _cpm_filter(X: np.ndarray, lib_sizes: np.ndarray, var: pd.DataFrame,
                min_cpm: float = 1.0,
                min_frac: float = 0.2) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (bool mask, report DataFrame) for CPM > min_cpm in >= min_frac of samples.

    lib_sizes: (n_samples,) total UMI per sample from adata.obs['psbulk_counts'].
    Zero-library-size obs are excluded from the CPM comparison (treated as NaN).
    """
    n_samples = X.shape[0]
    lib_safe = lib_sizes.copy()
    lib_safe[lib_safe == 0] = np.nan          # NaN CPM for zero-library obs; NaN > 1 = False
    with np.errstate(invalid='ignore'):        # suppress divide-by-zero / invalid warnings
        cpm = X / lib_safe[:, None] * 1e6
    n_pass = (cpm > min_cpm).sum(axis=0)
    threshold = int(np.ceil(min_frac * n_samples))
    keep = n_pass >= threshold

    X_dense = X if not hasattr(X, 'toarray') else X.toarray()
    report = pd.DataFrame({'gene_name': var.index})
    if 'gene_id' in var.columns:
        report['gene_id'] = var['gene_id'].values
    for col in ('chr', 'start', 'end', 'total_counts', 'mean_counts',
                'n_cells_by_counts', 'pct_dropout_by_counts'):
        if col in var.columns:
            report[col] = var[col].values
    report['n_samples_cpm_gt1'] = n_pass
    report['frac_samples_cpm_gt1'] = (n_pass / n_samples).round(4)
    report['mean_cpm'] = cpm.mean(axis=0).round(4)
    report['median_cpm'] = np.median(cpm, axis=0).round(4)
    report['n_samples_zero'] = (X_dense == 0).sum(axis=0)
    report['frac_samples_zero'] = ((X_dense == 0).sum(axis=0) / n_samples).round(4)
    return keep, report


def _write_counts_h5(adata: anndata.AnnData, sorted_idx: 'pd.Index',
                     out_path: pathlib.Path) -> None:
    sub = adata[sorted_idx]
    # exclude obs with zero library size (sample had no cells of this CT)
    valid_mask = sub.obs['psbulk_counts'].values > 0
    n_zero = (~valid_mask).sum()
    if n_zero:
        log.debug('  excluding %d obs with psbulk_counts=0 from HDF5', n_zero)
        sub = sub[valid_mask]
    X = sub.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    lib_sizes = sub.obs['psbulk_counts'].values.astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        lognorm = np.log2(X / lib_sizes[:, None] * 1e6 + 1.0)
    genes   = np.array(adata.var_names, dtype='S')
    samples = sub.obs['sample_id'].values.astype('S')
    log.info('writing %s', out_path)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('counts', data=lognorm.T, dtype='float32',
                         compression='gzip', compression_opts=4,
                         chunks=(min(1000, adata.n_vars), min(50, len(sorted_idx))))
        f.create_dataset('genes',   data=genes)
        f.create_dataset('samples', data=samples)


def init_celltypes(h5ad_path: pathlib.Path, out_dir: pathlib.Path, tissue: str) -> None:
    """Initialise shared outputs from a cell-type pseudobulk h5ad (one obs per sample × CT).

    Writes to out_dir/:
        metadata.csv             — one row per unique sample_id (shared across all CTs)
        zero_psbulk_samples.csv  — (sample, CT) pairs with psbulk_counts=0
        exp_gene_info_{tissue}.csv — gene coordinates for ALL var_names in the h5ad;
                                  this is the gene universe passed to convert_celltype().
                                  No CPM filter is applied here — a cross-CT CPM threshold
                                  would incorrectly drop CT-specific genes (a gene expressed
                                  in only one CT can reach at most 1/n_ct ≈ 5% of obs,
                                  always failing a 20% threshold even if all samples express
                                  it in that CT).  Per-CT CPM filtering is done exclusively
                                  inside convert_celltype() on valid samples of that CT.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d obs × %d genes  (%d unique samples)',
             adata.n_obs, adata.n_vars, adata.obs['sample_id'].nunique())

    # ── zero-psbulk report: (sample, CT) pairs with no cells of that type ────
    ct_col = _detect_celltype_col(adata.obs)
    zero_mask = adata.obs['psbulk_counts'] == 0
    if zero_mask.any():
        zero_cols = ['sample_id', ct_col, 'genotype', 'dataset', 'psbulk_counts']
        if 'psbulk_cells' in adata.obs.columns:
            zero_cols.append('psbulk_cells')
        zero_df = adata.obs.loc[zero_mask, zero_cols].reset_index(drop=True)
        zero_path = out_dir / 'zero_psbulk_samples.csv'
        zero_df.to_csv(zero_path, index=False)
        log.warning('  %d (sample, %s) pairs with psbulk_counts=0 → %s',
                    int(zero_mask.sum()), ct_col, zero_path)
    else:
        log.info('  no zero-psbulk obs found')

    # ── metadata: one row per sample ─────────────────────────────────────────
    meta_cols = _metadata_cols(adata.obs)
    obs_all = adata.obs[meta_cols].copy()
    obs_all['_sort'] = obs_all['genotype'].astype(str).map(_strain_sort_key)
    obs_unique = (obs_all.drop_duplicates('sample_id')
                         .sort_values(['_sort', 'sample_id'])
                         .drop(columns='_sort')
                         .reset_index(drop=True))
    meta_path = out_dir / 'metadata.csv'
    log.info('writing %s  (%d samples, cols: %s)', meta_path, len(obs_unique),
             ', '.join(meta_cols))
    obs_unique.to_csv(meta_path, index=False)

    # ── exp_gene_info_{tissue}.csv: coordinates for ALL genes in var_names ───
    # No CPM filter — per-CT filtering is done in convert_celltype().
    # Coordinates come directly from adata.var (gene_id, chr, start, end, strand).
    coord_cols = [c for c in ('gene_id', 'chr', 'start', 'end', 'strand')
                  if c in adata.var.columns]
    exp_gene_info = adata.var[coord_cols].copy().reset_index(names='gene_name')
    n_missing_coords = exp_gene_info[coord_cols].isna().any(axis=1).sum()
    if n_missing_coords:
        log.warning('  %d genes missing one or more coordinate columns in adata.var',
                    n_missing_coords)
    exp_gene_info = _add_tss_bp(exp_gene_info)
    exp_gene_info = _strip_gene_id_version(exp_gene_info)
    exp_path = out_dir / f'exp_gene_info_{tissue}.csv'
    log.info('writing %s  (%d genes — full universe, no CPM filter)', exp_path, len(exp_gene_info))
    exp_gene_info.to_csv(exp_path, index=False)

    log.info('init done — %d samples  %d gene universe', len(obs_unique), adata.n_vars)
    log.info('  per-CT gene sets determined by per-CT CPM filter in convert_celltype()')
    for ds, grp in obs_unique.groupby('dataset', sort=True, observed=True):
        log.info('  dataset=%-12s  %d samples  genotypes: %s',
                 ds, len(grp), ', '.join(sorted(grp['genotype'].astype(str).unique())))


def convert(h5ad_path: pathlib.Path, out_dir: pathlib.Path, tissue: str) -> None:
    """Whole-tissue conversion: writes counts_lognorm.h5, metadata.csv, exp_gene_info_{tissue}.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d samples × %d genes', adata.n_obs, adata.n_vars)

    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    X = adata.X if not hasattr(adata.X, 'toarray') else adata.X.toarray()
    lib_sizes = adata.obs['psbulk_counts'].values.astype(np.float32)
    keep, report = _cpm_filter(X.astype(np.float32), lib_sizes, adata.var)
    log.info('  CPM filter: %d / %d genes pass (CPM>1 in ≥20%% of samples)',
             int(keep.sum()), adata.n_vars)
    adata = adata[:, keep].copy()

    _write_counts_h5(adata, obs.index, out_dir / 'counts_lognorm.h5')

    meta_path = out_dir / 'metadata.csv'
    log.info('writing %s', meta_path)
    obs.reset_index(drop=True).to_csv(meta_path, index=False)

    # ── exp_gene_info_{tissue}.csv ────────────────────────────────────────────
    # coordinates from adata.var; CPM stats from filter report
    coord_cols = [c for c in ('gene_id', 'chr', 'start', 'end', 'strand')
                  if c in adata.var.columns]
    cpm_stats_cols = ['gene_name', 'total_counts', 'mean_counts', 'n_cells_by_counts',
                      'pct_dropout_by_counts', 'n_samples_cpm_gt1', 'frac_samples_cpm_gt1',
                      'mean_cpm', 'median_cpm', 'n_samples_zero', 'frac_samples_zero']
    exp_gene_info = (
        adata.var[coord_cols].copy().reset_index(names='gene_name')
        .merge(report[[c for c in cpm_stats_cols if c in report.columns]],
               on='gene_name', how='left')
    )
    exp_gene_info = _add_tss_bp(exp_gene_info)
    exp_gene_info = _strip_gene_id_version(exp_gene_info)
    exp_path = out_dir / f'exp_gene_info_{tissue}.csv'
    log.info('writing %s  (%d genes)', exp_path, len(exp_gene_info))
    exp_gene_info.to_csv(exp_path, index=False)

    log.info('done — %d samples, %d genes', adata.n_obs, adata.n_vars)
    for ds, grp in obs.groupby('dataset', sort=True, observed=True):
        log.info('  dataset=%-12s  %d samples  genotypes: %s',
                 ds, len(grp), ', '.join(sorted(grp['genotype'].astype(str).unique())))


def convert_celltype(h5ad_path: pathlib.Path, celltype: str,
                     out_root: pathlib.Path, tissue: str,
                     min_cells: int = 10,
                     min_counts: int = 10,
                     min_gene_counts: int = 5,
                     min_valid_samples: int = 50) -> None:
    """Cell-type conversion: subsets pb h5ad; writes counts_lognorm.h5 + exp_gene_info_{ct}.csv.

    Sample validity gate (applied before any gene filter):
        psbulk_counts >= min_counts  AND  psbulk_cells >= min_cells
    Both thresholds guard against unreliable pseudobulks from low-coverage samples.
    Excluded (sample, CT) pairs are pre-reported in zero_psbulk_samples.csv by init_celltypes().

    Two-step gene filter on valid samples:
      1. Intersect with whole-tissue gene universe (exp_gene_info_{tissue}.csv).
      2a. Per-CT CPM filter: CPM > 1 in ≥20% of valid samples.
      2b. Minimum total counts: sum across valid samples >= min_gene_counts.
    Both 2a and 2b must pass. All CPM stats denominators use n_valid, not total n.
    metadata.csv is shared from out_root.parent (all samples, all CTs).
    """
    ct = _ct_safe(celltype)

    # ── load whole-tissue gene universe ───────────────────────────────────────
    wt_exp_path = out_root.parent / f'exp_gene_info_{tissue}.csv'
    if not wt_exp_path.exists():
        raise FileNotFoundError(
            f'exp_gene_info_{tissue}.csv not found at {wt_exp_path} — '
            f'run init_celltypes() or convert() first')
    passing_genes = set(pd.read_csv(wt_exp_path, usecols=['gene_name'])['gene_name'])
    log.info('  whole-tissue gene universe: %d genes', len(passing_genes))

    log.info('reading %s', h5ad_path)
    adata_full = anndata.read_h5ad(h5ad_path)
    ct_col = _detect_celltype_col(adata_full.obs)

    # ── subset to this cell type ──────────────────────────────────────────────
    adata = adata_full[adata_full.obs[ct_col] == celltype].copy()
    if adata.n_obs == 0:
        available = sorted(adata_full.obs[ct_col].unique())
        raise ValueError(f'{ct_col}={celltype!r} not found. available: {available}')

    # ── step 1: intersect with whole-tissue gene universe ────────────────────
    ct_genes = [g for g in adata.var_names if g in passing_genes]
    adata = adata[:, ct_genes].copy()

    # ── step 2: identify valid samples ───────────────────────────────────────
    lib_sizes   = adata.obs['psbulk_counts'].values.astype(np.float32)
    cell_counts = (adata.obs['psbulk_cells'].values.astype(np.float32)
                   if 'psbulk_cells' in adata.obs.columns
                   else np.full(adata.n_obs, np.inf, dtype=np.float32))
    valid_mask = (lib_sizes >= min_counts) & (cell_counts >= min_cells)
    n_excluded = int((~valid_mask).sum())
    n_valid    = int(valid_mask.sum())
    if n_excluded:
        n_zero   = int((lib_sizes == 0).sum())
        n_lowlib = int(((lib_sizes > 0) & (lib_sizes < min_counts)).sum())
        n_lowcel = int(((cell_counts < min_cells) & (lib_sizes >= min_counts)).sum())
        log.info('  excluded %d / %d obs  (zero: %d  low-counts<%d: %d  low-cells<%d: %d)',
                 n_excluded, adata.n_obs, n_zero,
                 min_counts, n_lowlib, min_cells, n_lowcel)

    if n_valid < min_valid_samples:
        log.warning('  SKIPPING %s=%r — only %d valid samples (< min_valid_samples=%d)',
                    ct_col, celltype, n_valid, min_valid_samples)
        return

    X_all = adata.X if not hasattr(adata.X, 'toarray') else adata.X.toarray()
    X_all = X_all.astype(np.float32)
    X_valid = X_all[valid_mask]
    lib_valid = lib_sizes[valid_mask]

    # ── step 3: per-CT gene filter on valid samples ───────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        cpm = X_valid / lib_valid[:, None] * 1e6
    ct_threshold = int(np.ceil(0.2 * n_valid))
    cpm_keep        = (cpm > 1).sum(axis=0) >= ct_threshold   # 2a: CPM breadth
    gene_count_keep = X_valid.sum(axis=0) >= min_gene_counts  # 2b: minimum total counts
    ct_keep = cpm_keep & gene_count_keep

    n_wt = adata.n_vars
    adata   = adata[:, ct_keep].copy()
    X_valid = X_valid[:, ct_keep]
    cpm     = cpm[:, ct_keep]
    log.info('  %s=%r  %d valid samples × %d genes  '
             '(%d dropped: %d failed CPM, %d failed min-counts, %d failed both)',
             ct_col, celltype, n_valid, adata.n_vars, n_wt - adata.n_vars,
             int((~cpm_keep & gene_count_keep).sum()),
             int((cpm_keep & ~gene_count_keep).sum()),
             int((~cpm_keep & ~gene_count_keep).sum()))

    # ── sort valid obs by genotype then sample_id ─────────────────────────────
    meta_cols = _metadata_cols(adata.obs)
    obs_valid = adata.obs.loc[valid_mask, meta_cols].copy()
    obs_valid['_sort'] = obs_valid['genotype'].astype(str).map(_strain_sort_key)
    obs_valid = obs_valid.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    ct_dir = out_root / ct
    ct_dir.mkdir(parents=True, exist_ok=True)

    # ── CT-specific CPM stats (all denominators = n_valid) ────────────────────
    ct_stats = pd.DataFrame({
        'gene_name':            adata.var_names,
        'n_samples_valid':      n_valid,
        'n_samples_zero_lib':   n_excluded,
        'n_samples_cpm_gt1':    (cpm > 1).sum(axis=0),
        'frac_samples_cpm_gt1': ((cpm > 1).sum(axis=0) / n_valid).round(4),
        'mean_cpm':             cpm.mean(axis=0).round(4),
        'median_cpm':           np.median(cpm, axis=0).round(4),
        'n_samples_zero':       (X_valid == 0).sum(axis=0),
        'frac_samples_zero':    ((X_valid == 0).sum(axis=0) / n_valid).round(4),
    })

    # ── exp_gene_info_{ct}.csv: coordinates from adata.var + CT CPM stats ────
    coord_cols = [c for c in ('gene_id', 'chr', 'start', 'end', 'strand')
                  if c in adata.var.columns]
    exp_gene_info_ct = (
        adata.var[coord_cols].copy().reset_index(names='gene_name')
        .merge(ct_stats, on='gene_name', how='left')
    )
    exp_gene_info_ct = _add_tss_bp(exp_gene_info_ct)
    exp_gene_info_ct = _strip_gene_id_version(exp_gene_info_ct)
    exp_path = ct_dir / f'exp_gene_info_{ct}.csv'
    exp_gene_info_ct.to_csv(exp_path, index=False)
    log.info('writing %s  (%d genes)', exp_path, len(exp_gene_info_ct))

    # ── counts_lognorm.h5: valid samples × CT-passing genes ──────────────────
    _write_counts_h5(adata, obs_valid.index, ct_dir / 'counts_lognorm.h5')

    log.info('done — %d valid samples  %d genes  (metadata.csv shared from %s)',
             n_valid, adata.n_vars, out_root.parent)
    for ds, grp in obs_valid.groupby('dataset', sort=True, observed=True):
        log.info('  dataset=%-12s  %d samples', ds, len(grp))


def main() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    gastroc_data = repo_root / 'validation' / 'gastroc_data'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--h5ad', type=pathlib.Path, default=None,
                        help='path to input h5ad (defaults depend on mode)')
    parser.add_argument('--out', type=pathlib.Path, default=gastroc_data,
                        help='output root (default: validation/gastroc_data)')
    parser.add_argument('--tissue', type=str, default='gastroc',
                        help='tissue label used in output filenames (default: gastroc)')
    parser.add_argument('--celltype', type=str, default=None,
                        help='single cell type to extract (e.g. myonuclei)')
    parser.add_argument('--all_celltypes', action='store_true',
                        help='convert every cell type found in the h5ad in one pass')
    parser.add_argument('--exclude_celltypes', type=str,
                        default='low-quality,low quality',
                        help='comma-separated cell types to skip '
                             '(default: "low-quality,low quality")')
    parser.add_argument('--min_valid_samples', type=int, default=50,
                        help='skip cell types with fewer than this many valid samples '
                             'after applying min_cells/min_counts gates (default: 50)')
    parser.add_argument('--min_cells', type=int, default=10,
                        help='minimum cells per (sample, CT) to include sample '
                             '(default: 10; requires psbulk_cells column)')
    parser.add_argument('--min_counts', type=int, default=10,
                        help='minimum psbulk_counts per (sample, CT) to include sample '
                             '(default: 10; replaces the >0 gate)')
    parser.add_argument('--min_gene_counts', type=int, default=5,
                        help='minimum total counts summed across valid samples for a gene '
                             'to be included in per-CT output (default: 5)')
    parser.add_argument('--list_celltypes', action='store_true',
                        help='list available cell types in the cell-type h5ad and exit')
    args = parser.parse_args()

    exclude = {c.strip() for c in args.exclude_celltypes.split(',') if c.strip()}

    if args.list_celltypes:
        h5ad = args.h5ad or _DEFAULT_CT_H5AD
        adata = anndata.read_h5ad(h5ad)
        ct_col = _detect_celltype_col(adata.obs)
        log.info('cell type column: %r', ct_col)
        for ct in sorted(adata.obs[ct_col].unique()):
            n = (adata.obs[ct_col] == ct).sum()
            flag = '  [excluded]' if ct in exclude else ''
            print(f'  {ct!r:50s}  {n} obs{flag}')
        return

    if args.all_celltypes:
        h5ad = args.h5ad or _DEFAULT_CT_H5AD
        adata_full = anndata.read_h5ad(h5ad)
        ct_col = _detect_celltype_col(adata_full.obs)
        log.info('cell type column: %r', ct_col)
        all_cts = sorted(adata_full.obs[ct_col].unique())
        ct_root = args.out / 'celltypes'
        # init shared outputs from the CT pb h5ad (deduplicated metadata + gene set)
        # do NOT call convert() here — that expects one obs per sample (bulk h5ad)
        init_celltypes(h5ad, args.out, args.tissue)
        for ct in all_cts:
            if ct in exclude:
                log.info('skipping excluded cell type %r', ct)
                continue
            convert_celltype(h5ad, ct, ct_root, args.tissue,
                             min_cells=args.min_cells,
                             min_counts=args.min_counts,
                             min_gene_counts=args.min_gene_counts,
                             min_valid_samples=args.min_valid_samples)
    elif args.celltype is not None:
        h5ad = args.h5ad or _DEFAULT_CT_H5AD
        ct_root = args.out / 'celltypes'
        convert_celltype(h5ad, args.celltype, ct_root, args.tissue,
                         min_cells=args.min_cells,
                         min_counts=args.min_counts,
                         min_gene_counts=args.min_gene_counts,
                         min_valid_samples=args.min_valid_samples)
    else:
        h5ad = args.h5ad or _DEFAULT_BULK_H5AD
        convert(h5ad, args.out, args.tissue)


if __name__ == '__main__':
    main()
