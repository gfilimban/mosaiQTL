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
    exp_gene_info_{tissue}_universe.csv (unfiltered gene universe, written once by
    init_celltypes()) is shared from --out/ too — distinct filename from whole-tissue
    mode's CPM-filtered exp_gene_info_{tissue}.csv so the two never collide even when
    both target the same --out.

    --all_celltypes converts every subtype in the h5ad in one pass.
    --exclude_celltypes skips named subtypes (default: low-quality).

gene_id (versioned Ensembl ID, e.g. 'ENSMUSG00000051951.6') is the identity key
for every gene, everywhere in the pipeline: the HDF5 'genes' dataset in
counts_lognorm.h5, the join key in every exp_gene_info_*.csv, and the key
convert_celltype() uses to intersect against the whole-tissue gene universe.
adata.var_names is NOT used for gene identity anywhere downstream — it is
cohort-dependent (some cohorts index var by gene symbol, others by the same
versioned Ensembl ID as gene_id) and was the root cause of a stale, ambiguous
'gene_name' column mirroring whatever var_names happened to be per cohort; that
column has been removed. adata.var must carry a 'gene_id' column — this is
required, not optional, and conversion raises if it is absent.

Gene coordinates (chr, start, end, strand) are read directly from adata.var
when present. If adata.var lacks them entirely (e.g. a pseudobulk h5ad built
from an annotated h5ad whose var carried no coordinates), they are filled from
the external vM36 reference at sibling qtl/input/gene_info.vm36.csv, matched on
gene_id — an exact (versioned) match first, falling back to a version-stripped
match, then to gene_name (the reference file's own column, unrelated to the
identity key above). gene_id itself is never modified: the versioned form is
preserved exactly as sourced in every output. tss_bp is derived from those
columns (start for + strand, end for - strand) and written to
exp_gene_info_*.csv for use downstream as cis_dist_mb = (locus_pos - tss_bp) / 1e6.
gene_symbol is the real display name (adata.var['gene_name'] when the cohort
ships it separately from its index, else var_names itself) — see
_derive_gene_symbol. It is a display-only column, never a join key.

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


def _derive_gene_symbol(var: pd.DataFrame) -> pd.Series:
    """Return the real display gene symbol per gene, indexed like `var`.

    adata.var_names is NOT reliably a gene symbol across cohorts — confirmed
    for one ctxhipp cohort where var_names is the versioned Ensembl gene ID
    (e.g. 'ENSMUSG00000102693.2') for 100% of genes, with the real symbol
    sitting unused in a separate adata.var['gene_name'] column. Other cohorts
    index var directly by symbol (e.g. 'Xkr4') and carry no such column.

    Prefers adata.var['gene_name'] when present (the cohort already ships the
    real symbol separately from its index); falls back to var_names itself
    otherwise. Purely a display-symbol source for the 'gene_symbol' column —
    gene identity/joins always use gene_id, never this.
    """
    if 'gene_name' in var.columns:
        return var['gene_name']
    return pd.Series(var.index.astype(str), index=var.index)


# sibling qtl/input/ dir holds the vM36 gene coordinate reference; repo_root.parent
# from mosaiqtl root (never a hardcoded home path). vM36 only — no vM32 fallback.
_EXT_GENE_INFO_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent.parent
                       / 'input' / 'gene_info.vm36.csv')


def _load_ext_gene_coords() -> pd.DataFrame:
    """Load the external vM36 gene coordinate reference (gene_id, chr, start, end, strand).

    Fallback merge source for `_fill_missing_coords`, used only when the input h5ad's
    adata.var lacks these columns entirely (e.g. a pseudobulk h5ad built upstream from
    an annotated h5ad whose var carried no coordinate columns). gene_id is kept exactly
    as sourced (versioned, e.g. 'ENSMUSG00000102693.2') — never stripped here; matching
    against it tries an exact match first, see _fill_missing_coords.
    """
    if not _EXT_GENE_INFO_PATH.exists():
        raise FileNotFoundError(
            f'external gene coordinate reference not found at {_EXT_GENE_INFO_PATH}; '
            'required because adata.var is missing chr/start/end/strand')
    return pd.read_csv(_EXT_GENE_INFO_PATH, dtype={'chr': str})


def _fill_missing_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Fill chr/start/end/strand from the external vM36 reference when absent from adata.var.

    No-op when adata.var already carried all four columns (the normal case described
    in the module docstring). df['gene_id'] is never modified — matching checks both
    versioned forms: an exact gene_id match first, then a version-stripped match (on
    both sides) for anything left unmatched, covering cross-release version drift.
    Falls back to gene_name for any gene_id matched by neither. Genes matched by none
    of these are left with NaN coordinates and counted in the logged warning.
    """
    coord_cols = ['chr', 'start', 'end', 'strand']
    missing = [c for c in coord_cols if c not in df.columns]
    if not missing:
        return df
    if 'gene_id' not in df.columns and 'gene_name' not in df.columns:
        raise ValueError('cannot fill gene coordinates: adata.var has neither gene_id nor gene_name')

    ext = _load_ext_gene_coords()
    log.warning('  adata.var missing %s — filling from %s', ', '.join(missing), _EXT_GENE_INFO_PATH)
    df = df.copy()
    for c in missing:
        df[c] = pd.Series(pd.NA, index=df.index, dtype='object')

    if 'gene_id' in df.columns:
        # exact (versioned) match first
        ext_by_id = ext.drop_duplicates('gene_id').set_index('gene_id')[coord_cols]
        for c in missing:
            df[c] = df['gene_id'].map(ext_by_id[c])

        # version-stripped fallback (both sides) for anything still unmatched
        unmatched_id = df[missing[0]].isna()
        if unmatched_id.any():
            gene_id_base = df.loc[unmatched_id, 'gene_id'].str.replace(r'\.\d+$', '', regex=True)
            ext_base = ext.copy()
            ext_base['gene_id'] = ext_base['gene_id'].str.replace(r'\.\d+$', '', regex=True)
            ext_by_id_base = ext_base.drop_duplicates('gene_id').set_index('gene_id')[coord_cols]
            for c in missing:
                df.loc[unmatched_id, c] = gene_id_base.map(ext_by_id_base[c])

    unmatched = df[missing[0]].isna()
    if unmatched.any() and 'gene_name' in df.columns:
        ext_by_name = ext.drop_duplicates('gene_name').set_index('gene_name')[coord_cols]
        for c in missing:
            df.loc[unmatched, c] = df.loc[unmatched, 'gene_name'].map(ext_by_name[c])

    for c in ('start', 'end'):
        if c in missing:
            df[c] = df[c].astype('Int64')

    n_unmatched = df[missing[0]].isna().sum()
    if n_unmatched:
        log.warning('  %d / %d genes not found in %s — coordinates left as NaN',
                    n_unmatched, len(df), _EXT_GENE_INFO_PATH.name)
    return df


def _strain_sort_key(genotype: str) -> tuple[int, str]:
    if genotype in _NON_CC_ORDER:
        return (0, str(_NON_CC_ORDER.index(genotype)).zfill(3))
    return (1, genotype)


def _ct_safe(cell_type: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', cell_type.lower()).strip('_')


_CELLTYPE_COL_CHOICES = ('subtype', 'celltype', 'general_celltype')


def _detect_celltype_col(obs: pd.DataFrame, override: str | None = None) -> str:
    """Return the obs column holding cell type labels at pseudobulk resolution.

    override: explicit column name (one of _CELLTYPE_COL_CHOICES) — bypasses
        auto-detection entirely. Must be present in obs; raises ValueError otherwise.
        Only 'subtype' carries the one-row-per-(sample,label) guarantee pseudobulk
        resolution requires — 'celltype' and 'general_celltype' are coarser groupings
        that can map several subtype rows for the same mouse onto one label, which
        breaks that guarantee. A warning is logged whenever override selects either.

    Auto-detect (override=None) prefers 'subtype' because pseudobulks are built at
    subtype resolution; falls back to 'celltype' for h5ads that lack a subtype column
    (e.g. gastroc bulk). Raises ValueError if neither is present.
    """
    if override is not None:
        if override not in obs.columns:
            raise ValueError(
                f'celltype_col={override!r} not found in obs. '
                f'Available columns: {list(obs.columns)}')
        if override != 'subtype':
            log.warning(
                '  celltype_col=%r is coarser than subtype — it may map multiple '
                'subtype rows for the same mouse onto one label, breaking the '
                'one-obs-per-(sample,label) assumption pseudobulk conversion relies on',
                override)
        return override
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
                min_count: float = 5.0,
                min_frac: float = 0.2) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (bool mask, report DataFrame) for the combined expression-breadth gate.

    A gene passes in a given sample when BOTH conditions hold: CPM > min_cpm AND
    raw count >= min_count. A gene is kept overall when this combined condition
    holds in >= min_frac of samples. The single-condition CPM test is gameable by
    shallow-library samples, where CPM crosses min_cpm at a trivial raw count
    (e.g. count=1 clears CPM>1 once lib_size < 1e6); the count floor closes that
    gap without over-penalizing deep-library samples, where a count floor alone
    would be too lenient. The same combined gate and default min_count are used
    identically at the whole-tissue level (this function) and the per-cell-type
    level (convert_celltype, called on the valid-sample subset) so gene inclusion
    logic does not silently diverge between the two granularities.

    lib_sizes: (n_samples,) total UMI per sample from adata.obs['psbulk_counts'].
    Zero-library-size obs are excluded from the CPM comparison (treated as NaN).
    """
    n_samples = X.shape[0]
    lib_safe = lib_sizes.copy()
    lib_safe[lib_safe == 0] = np.nan          # NaN CPM for zero-library obs; NaN > 1 = False
    with np.errstate(invalid='ignore'):        # suppress divide-by-zero / invalid warnings
        cpm = X / lib_safe[:, None] * 1e6
    X_dense = X if not hasattr(X, 'toarray') else X.toarray()
    cpm_pass = cpm > min_cpm
    count_pass = X_dense >= min_count
    combined_pass = cpm_pass & count_pass
    n_pass = combined_pass.sum(axis=0)
    threshold = int(np.ceil(min_frac * n_samples))
    keep = n_pass >= threshold

    if 'gene_id' not in var.columns:
        raise ValueError(
            "adata.var has no 'gene_id' column — gene_id is the required identity "
            'key; var_names is cohort-dependent and is never used for gene identity')
    # deliberately excludes chr/start/end — callers already have coordinates
    # from adata.var directly and merge only the stats columns (see _CPM_STATS_COLS)
    # to avoid pandas _x/_y suffix collisions on the shared gene_id key.
    report = pd.DataFrame({'gene_id': var['gene_id'].values})
    for col in ('total_counts', 'mean_counts', 'n_cells_by_counts', 'pct_dropout_by_counts'):
        if col in var.columns:
            report[col] = var[col].values
    report['n_samples_pass'] = n_pass
    report['frac_samples_pass'] = (n_pass / n_samples).round(4)
    report['n_samples_cpm_gt1'] = cpm_pass.sum(axis=0)
    report['frac_samples_cpm_gt1'] = (cpm_pass.sum(axis=0) / n_samples).round(4)
    report['n_samples_count_ge_min'] = count_pass.sum(axis=0)
    report['frac_samples_count_ge_min'] = (count_pass.sum(axis=0) / n_samples).round(4)
    report['mean_cpm'] = cpm.mean(axis=0).round(4)
    report['median_cpm'] = np.median(cpm, axis=0).round(4)
    report['n_samples_zero'] = (X_dense == 0).sum(axis=0)
    report['frac_samples_zero'] = ((X_dense == 0).sum(axis=0) / n_samples).round(4)
    return keep, report


# gene_id (the join key) plus stats-only columns from _cpm_filter's report —
# deliberately excludes chr/start/end/strand, which the caller already has from
# adata.var; merging the full report (coordinates included) against an
# adata.var-derived frame on 'gene_id' produces pandas _x/_y suffix collisions
# on every shared coordinate column.
_CPM_STATS_COLS = ['gene_id', 'total_counts', 'mean_counts', 'n_cells_by_counts',
                   'pct_dropout_by_counts', 'n_samples_pass', 'frac_samples_pass',
                   'n_samples_cpm_gt1', 'frac_samples_cpm_gt1',
                   'n_samples_count_ge_min', 'frac_samples_count_ge_min',
                   'mean_cpm', 'median_cpm', 'n_samples_zero', 'frac_samples_zero']

# canonical leading columns for every exp_gene_info_*.csv; stats columns follow.
_EXP_GENE_INFO_BASE_COLS = ['gene_id', 'gene_symbol', 'chr', 'start', 'end', 'strand', 'tss_bp']


def _order_exp_gene_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with canonical leading columns; remaining columns appended as-is.

    Drops 'gene_name' unconditionally — it leaks in as the adata.var index name
    for cohorts whose var_names were originally the gene_name column.  gene_symbol
    is the canonical display column; gene_name is never written to exp_gene_info_*.csv.
    """
    df = df.drop(columns=['gene_name'], errors='ignore')
    base = [c for c in _EXP_GENE_INFO_BASE_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in set(base)]
    return df[base + rest]


def _write_counts_h5(adata: anndata.AnnData, sorted_idx: 'pd.Index',
                     out_path: pathlib.Path) -> None:
    if 'gene_id' not in adata.var.columns:
        raise ValueError(
            "adata.var has no 'gene_id' column — gene_id is the required identity "
            'key for the HDF5 genes dataset; var_names is cohort-dependent and is '
            'never used for gene identity')
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
    genes   = np.array(adata.var['gene_id'], dtype='S')
    samples = sub.obs['sample_id'].values.astype('S')
    log.info('writing %s', out_path)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('counts', data=lognorm.T, dtype='float32',
                         compression='gzip', compression_opts=4,
                         chunks=(min(1000, adata.n_vars), min(50, len(sorted_idx))))
        f.create_dataset('genes',   data=genes)
        f.create_dataset('samples', data=samples)


def init_celltypes(h5ad_path: pathlib.Path, out_dir: pathlib.Path, tissue: str,
                   celltype_col: str | None = None) -> None:
    """Initialise shared outputs from a cell-type pseudobulk h5ad (one obs per sample × CT).

    celltype_col: explicit obs column override — see _detect_celltype_col().

    Writes to out_dir/:
        metadata.csv                       — one row per unique sample_id (shared across all CTs)
        zero_psbulk_samples.csv            — (sample, CT) pairs with psbulk_counts=0
        exp_gene_info_{tissue}_universe.csv — gene coordinates for ALL var_names in the h5ad;
                                  this is the gene universe passed to convert_celltype().
                                  No CPM filter is applied here — a cross-CT CPM threshold
                                  would incorrectly drop CT-specific genes (a gene expressed
                                  in only one CT can reach at most 1/n_ct ≈ 5% of obs,
                                  always failing a 20% threshold even if all samples express
                                  it in that CT).  Per-CT CPM filtering is done exclusively
                                  inside convert_celltype() on valid samples of that CT.
                                  Named distinctly (_universe suffix) from convert()'s
                                  exp_gene_info_{tissue}.csv — that file is the CPM-filtered
                                  whole-tissue gene list read by validate_tissue.py's
                                  whole-tissue scan; the two have incompatible content
                                  (unfiltered universe vs. filtered list) and must never
                                  share a filename, even when both land in the same --out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d obs × %d genes  (%d unique samples)',
             adata.n_obs, adata.n_vars, adata.obs['sample_id'].nunique())

    # ── zero-psbulk report: (sample, CT) pairs with no cells of that type ────
    ct_col = _detect_celltype_col(adata.obs, override=celltype_col)
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

    # ── exp_gene_info_{tissue}_universe.csv: coordinates for ALL genes ───────
    # No CPM filter — per-CT filtering is done in convert_celltype(). Named with a
    # _universe suffix so this never collides with convert()'s CPM-filtered
    # exp_gene_info_{tissue}.csv (whole-tissue mode writes to the same out_dir).
    # Coordinates come directly from adata.var; gene_id is the identity key
    # (required) — convert_celltype() re-reads this file's gene_id column to
    # build the whole-tissue gene universe set.
    if 'gene_id' not in adata.var.columns:
        raise ValueError(
            "adata.var has no 'gene_id' column — gene_id is the required identity "
            'key; var_names is cohort-dependent and is never used for gene identity')
    coord_cols = ['gene_id'] + [c for c in ('chr', 'start', 'end', 'strand')
                                 if c in adata.var.columns]
    exp_gene_info = adata.var[coord_cols].copy()
    exp_gene_info['gene_symbol'] = _derive_gene_symbol(adata.var)
    exp_gene_info = exp_gene_info.reset_index(drop=True)
    n_missing_coords = exp_gene_info[coord_cols].isna().any(axis=1).sum()
    if n_missing_coords:
        log.warning('  %d genes missing one or more coordinate columns in adata.var',
                    n_missing_coords)
    exp_gene_info = _fill_missing_coords(exp_gene_info)
    exp_gene_info = _add_tss_bp(exp_gene_info)
    exp_path = out_dir / f'exp_gene_info_{tissue}_universe.csv'
    log.info('writing %s  (%d genes — full universe, no CPM filter)', exp_path, len(exp_gene_info))
    _order_exp_gene_cols(exp_gene_info).to_csv(exp_path, index=False)

    log.info('init done — %d samples  %d gene universe', len(obs_unique), adata.n_vars)
    log.info('  per-CT gene sets determined by per-CT CPM filter in convert_celltype()')
    for ds, grp in obs_unique.groupby('dataset', sort=True, observed=True):
        log.info('  dataset=%-12s  %d samples  genotypes: %s',
                 ds, len(grp), ', '.join(sorted(grp['genotype'].astype(str).unique())))


def convert(h5ad_path: pathlib.Path, out_dir: pathlib.Path, tissue: str,
           min_count: int = 5) -> None:
    """Whole-tissue conversion: writes counts_lognorm.h5, metadata.csv, exp_gene_info_{tissue}.csv.

    Gene filter: combined expression-breadth gate via _cpm_filter — CPM > 1 AND
    raw count >= min_count, in >= 20% of samples. Identical gate (same min_count
    default) is applied at the per-cell-type level in convert_celltype(), on the
    valid-sample subset, so gene inclusion logic never diverges between the two
    conversion granularities.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info('reading %s', h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    log.info('  shape: %d samples × %d genes', adata.n_obs, adata.n_vars)

    obs = adata.obs[['sample_id', 'genotype', 'sex', 'dataset']].copy()
    obs['_sort'] = obs['genotype'].astype(str).map(_strain_sort_key)
    obs = obs.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    X = adata.X if not hasattr(adata.X, 'toarray') else adata.X.toarray()
    lib_sizes = adata.obs['psbulk_counts'].values.astype(np.float32)
    keep, report = _cpm_filter(X.astype(np.float32), lib_sizes, adata.var, min_count=min_count)
    log.info('  expression filter: %d / %d genes pass (CPM>1 AND count>=%d in ≥20%% of samples)',
             int(keep.sum()), adata.n_vars, min_count)
    adata = adata[:, keep].copy()

    _write_counts_h5(adata, obs.index, out_dir / 'counts_lognorm.h5')

    meta_path = out_dir / 'metadata.csv'
    log.info('writing %s', meta_path)
    obs.reset_index(drop=True).to_csv(meta_path, index=False)

    # ── exp_gene_info_{tissue}.csv ────────────────────────────────────────────
    # coordinates from adata.var; expression filter stats from filter report;
    # gene_id is the identity key (required — same column already carried the
    # HDF5 'genes' dataset written above, so counts_lognorm.h5 and this CSV
    # always join cleanly on gene_id)
    coord_cols = ['gene_id'] + [c for c in ('chr', 'start', 'end', 'strand')
                                 if c in adata.var.columns]
    exp_gene_info = adata.var[coord_cols].copy()
    exp_gene_info['gene_symbol'] = _derive_gene_symbol(adata.var)
    exp_gene_info = (
        exp_gene_info.reset_index(drop=True)
        .merge(report[[c for c in _CPM_STATS_COLS if c in report.columns]],
               on='gene_id', how='left')
    )
    exp_gene_info = _fill_missing_coords(exp_gene_info)
    exp_gene_info = _add_tss_bp(exp_gene_info)
    exp_path = out_dir / f'exp_gene_info_{tissue}.csv'
    log.info('writing %s  (%d genes)', exp_path, len(exp_gene_info))
    _order_exp_gene_cols(exp_gene_info).to_csv(exp_path, index=False)

    log.info('done — %d samples, %d genes', adata.n_obs, adata.n_vars)
    for ds, grp in obs.groupby('dataset', sort=True, observed=True):
        log.info('  dataset=%-12s  %d samples  genotypes: %s',
                 ds, len(grp), ', '.join(sorted(grp['genotype'].astype(str).unique())))


def convert_celltype(h5ad_path: pathlib.Path, celltype: str,
                     out_root: pathlib.Path, tissue: str,
                     min_cells: int = 10,
                     min_counts: int = 10,
                     min_count: int = 5,
                     min_valid_samples: int = 50,
                     celltype_col: str | None = None) -> None:
    """Cell-type conversion: subsets pb h5ad; writes counts_lognorm.h5 + exp_gene_info_{ct}.csv.

    celltype_col: explicit obs column override — see _detect_celltype_col(). Must match
        the same override passed to init_celltypes() for a given --out (the gene universe
        and metadata.csv it wrote are shared across all cell types run against that --out).

    Sample validity gate (applied before any gene filter):
        psbulk_counts >= min_counts  AND  psbulk_cells >= min_cells
    Both thresholds guard against unreliable pseudobulks from low-coverage samples.
    Excluded (sample, CT) pairs are pre-reported in zero_psbulk_samples.csv by init_celltypes().

    Two-step gene filter:
      1. Intersect with whole-tissue gene universe (exp_gene_info_{tissue}_universe.csv,
         written by init_celltypes() — the unfiltered gene_id set, distinct from
         convert()'s CPM-filtered exp_gene_info_{tissue}.csv).
      2. Combined expression-breadth gate via _cpm_filter, on valid samples only:
         CPM > 1 AND raw count >= min_count, in >= 20% of valid samples.
    Identical gate and default min_count as convert()'s whole-tissue filter — gene
    inclusion logic does not diverge between the two conversion granularities.
    All CPM stats denominators use n_valid, not total n.
    metadata.csv is shared from out_root.parent (all samples, all CTs).
    """
    ct = _ct_safe(celltype)

    # ── load whole-tissue gene universe ───────────────────────────────────────
    wt_exp_path = out_root.parent / f'exp_gene_info_{tissue}_universe.csv'
    if not wt_exp_path.exists():
        raise FileNotFoundError(
            f'exp_gene_info_{tissue}_universe.csv not found at {wt_exp_path} — '
            f'run init_celltypes() first')
    passing_genes = set(pd.read_csv(wt_exp_path, usecols=['gene_id'])['gene_id'])
    log.info('  whole-tissue gene universe: %d genes', len(passing_genes))

    log.info('reading %s', h5ad_path)
    adata_full = anndata.read_h5ad(h5ad_path)
    ct_col = _detect_celltype_col(adata_full.obs, override=celltype_col)

    # ── subset to this cell type ──────────────────────────────────────────────
    adata = adata_full[adata_full.obs[ct_col] == celltype].copy()
    if adata.n_obs == 0:
        available = sorted(adata_full.obs[ct_col].unique())
        raise ValueError(f'{ct_col}={celltype!r} not found. available: {available}')

    # ── step 1: intersect with whole-tissue gene universe ────────────────────
    if 'gene_id' not in adata.var.columns:
        raise ValueError(
            "adata.var has no 'gene_id' column — gene_id is the required identity "
            'key; var_names is cohort-dependent and is never used for gene identity')
    ct_mask = adata.var['gene_id'].isin(passing_genes)
    adata = adata[:, ct_mask].copy()

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

    # ── step 3: combined expression-breadth gate on valid samples ────────────
    # same _cpm_filter as convert()'s whole-tissue gate, denominators = n_valid
    ct_keep, ct_stats = _cpm_filter(X_valid, lib_valid, adata.var, min_count=min_count)

    n_wt = adata.n_vars
    adata    = adata[:, ct_keep].copy()
    X_valid  = X_valid[:, ct_keep]
    ct_stats = ct_stats.loc[ct_keep].reset_index(drop=True)
    log.info('  %s=%r  %d valid samples × %d genes  '
             '(%d / %d genes pass expression filter: CPM>1 AND count>=%d in ≥20%% of valid samples)',
             ct_col, celltype, n_valid, adata.n_vars, adata.n_vars, n_wt, min_count)

    # ── sort valid obs by genotype then sample_id ─────────────────────────────
    meta_cols = _metadata_cols(adata.obs)
    obs_valid = adata.obs.loc[valid_mask, meta_cols].copy()
    obs_valid['_sort'] = obs_valid['genotype'].astype(str).map(_strain_sort_key)
    obs_valid = obs_valid.sort_values(['_sort', 'sample_id']).drop(columns='_sort')

    ct_dir = out_root / ct
    ct_dir.mkdir(parents=True, exist_ok=True)

    # ── CT-specific bookkeeping added onto the shared _cpm_filter report ─────
    ct_stats['n_samples_valid'] = n_valid
    ct_stats['n_samples_zero_lib'] = n_excluded

    # ── exp_gene_info_{ct}.csv: coordinates from adata.var + CT CPM stats ────
    # gene_id is the identity key (required, see step 1 above)
    coord_cols = ['gene_id'] + [c for c in ('chr', 'start', 'end', 'strand')
                                 if c in adata.var.columns]
    # stats-only merge (see _CPM_STATS_COLS) plus the two CT-specific columns added
    # onto ct_stats above — coordinates come from adata.var, never from ct_stats,
    # to avoid pandas _x/_y suffix collisions on the shared chr/start/end/strand.
    ct_stats_cols = _CPM_STATS_COLS + ['n_samples_valid', 'n_samples_zero_lib']
    exp_gene_info_ct = adata.var[coord_cols].copy()
    exp_gene_info_ct['gene_symbol'] = _derive_gene_symbol(adata.var)
    exp_gene_info_ct = (
        exp_gene_info_ct.reset_index(drop=True)
        .merge(ct_stats[[c for c in ct_stats_cols if c in ct_stats.columns]],
               on='gene_id', how='left')
    )
    exp_gene_info_ct = _fill_missing_coords(exp_gene_info_ct)
    exp_gene_info_ct = _add_tss_bp(exp_gene_info_ct)
    exp_path = ct_dir / f'exp_gene_info_{ct}.csv'
    _order_exp_gene_cols(exp_gene_info_ct).to_csv(exp_path, index=False)
    log.info('writing %s  (%d genes)', exp_path, len(exp_gene_info_ct))

    # ── counts_lognorm.h5: valid samples × CT-passing genes ──────────────────
    _write_counts_h5(adata, obs_valid.index, ct_dir / 'counts_lognorm.h5')

    log.info('done — %d valid samples  %d genes  (metadata.csv shared from %s)',
             n_valid, adata.n_vars, out_root.parent)
    for ds, grp in obs_valid.groupby('dataset', sort=True, observed=True):
        log.info('  dataset=%-12s  %d samples', ds, len(grp))


def main() -> None:
    repo_root = pathlib.Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--h5ad', type=pathlib.Path, required=True,
                        help='path to input h5ad — required; no per-tissue default exists '
                             '(bulk h5ad for whole-tissue mode, cell-type-pseudobulk h5ad '
                             'for --celltype/--all_celltypes/--list_celltypes)')
    parser.add_argument('--out', type=pathlib.Path, default=None,
                        help='output root (default: validation/{tissue}_data, or '
                             'validation/{tissue}_data/{dataset} if --dataset is set); '
                             'an explicit --out overrides --dataset')
    parser.add_argument('--tissue', type=str, default='gastroc',
                        help='tissue label used in output filenames (default: gastroc)')
    parser.add_argument('--dataset', type=str, default=None, choices=['next', 'nova'],
                        help='dataset generation subdirectory under validation/{tissue}_data/ '
                             '(default: none — writes directly under {tissue}_data); ignored '
                             'if --out is also given')
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
    parser.add_argument('--min_count', type=int, default=5,
                        help='minimum raw count in a sample, alongside CPM>1, for that '
                             'sample to count toward a gene\'s expression-breadth gate '
                             '(default: 5; same gate and default at whole-tissue and '
                             'per-cell-type granularity — see _cpm_filter)')
    parser.add_argument('--list_celltypes', action='store_true',
                        help='list available cell types in the cell-type h5ad and exit')
    parser.add_argument('--celltype_col', type=str, default=None,
                        choices=list(_CELLTYPE_COL_CHOICES),
                        help='obs column to subset cell types on (default: auto-detect, '
                             'preferring "subtype" over "celltype"). "celltype" and '
                             '"general_celltype" are coarser groupings that can map '
                             'several subtype rows for the same mouse onto one label — '
                             'only "subtype" carries the one-obs-per-(sample,label) '
                             'guarantee pseudobulk conversion relies on')
    args = parser.parse_args()

    if args.out is not None:
        out = args.out
    else:
        out = repo_root / 'validation' / f'{args.tissue}_data'
        if args.dataset is not None:
            out = out / args.dataset
    exclude = {c.strip().lower() for c in args.exclude_celltypes.split(',') if c.strip()}

    if args.list_celltypes:
        adata = anndata.read_h5ad(args.h5ad)
        ct_col = _detect_celltype_col(adata.obs, override=args.celltype_col)
        log.info('cell type column: %r', ct_col)
        for ct in sorted(adata.obs[ct_col].unique()):
            n = (adata.obs[ct_col] == ct).sum()
            flag = '  [excluded]' if ct.lower() in exclude else ''
            print(f'  {ct!r:50s}  {n} obs{flag}')
        return

    if args.all_celltypes:
        adata_full = anndata.read_h5ad(args.h5ad)
        ct_col = _detect_celltype_col(adata_full.obs, override=args.celltype_col)
        log.info('cell type column: %r', ct_col)
        all_cts = sorted(adata_full.obs[ct_col].unique())
        ct_root = out / 'celltypes'
        # init shared outputs from the CT pb h5ad (deduplicated metadata + gene set)
        # do NOT call convert() here — that expects one obs per sample (bulk h5ad)
        init_celltypes(args.h5ad, out, args.tissue, celltype_col=args.celltype_col)
        for ct in all_cts:
            if ct.lower() in exclude:
                log.info('skipping excluded cell type %r', ct)
                continue
            convert_celltype(args.h5ad, ct, ct_root, args.tissue,
                             min_cells=args.min_cells,
                             min_counts=args.min_counts,
                             min_count=args.min_count,
                             min_valid_samples=args.min_valid_samples,
                             celltype_col=args.celltype_col)
    elif args.celltype is not None:
        ct_root = out / 'celltypes'
        convert_celltype(args.h5ad, args.celltype, ct_root, args.tissue,
                         min_cells=args.min_cells,
                         min_counts=args.min_counts,
                         min_count=args.min_count,
                         min_valid_samples=args.min_valid_samples,
                         celltype_col=args.celltype_col)
    else:
        convert(args.h5ad, out, args.tissue, min_count=args.min_count)


if __name__ == '__main__':
    main()
