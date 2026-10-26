"""mosaiQTL command-line interface.

Each command below is a self-contained step in the QTL mapping workflow: it
reads its inputs from files and writes its results to files, so any step can
be rerun on its own without repeating earlier ones. Run `mosaiqtl --help` for
the full option listing.

Typical order of steps
-----------------------
1. fit-null  — per-gene baseline model (no genetic effect) → phi_shrunk.csv
2. scan       — genome-wide association scan → peak_lod.csv
3. permute    — significance threshold from shuffled data → perm_max_lods.csv
4. estimate   — founder strain effects at each gene's peak → allele_effects.csv
5. report     — summary table + Manhattan plot → eqtl_summary.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure root logger; DEBUG when verbose, INFO otherwise."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _detect_ct_col(obs_columns: list[str]) -> Optional[str]:
    """Return first obs column whose name contains both 'cell' and 'type', or None."""
    for c in obs_columns:
        cl = c.lower()
        if "cell" in cl and "type" in cl:
            return c
    return None


def _gene_window_lookup(gene_info) -> dict:
    """Build {gene_id: (chr, tss_mb)} from a gene-info table.

    TSS is the strand-aware gene boundary (start for '+', end for '-') so a
    local-window scan is centered on the actual transcription start site, not
    just the lower genomic coordinate. Requires gene_id/chr/start columns;
    end/strand are optional (missing strand assumes '+').
    """
    has_strand = "strand" in gene_info.columns
    lookup: dict = {}
    for _, row in gene_info.iterrows():
        if has_strand and row["strand"] == "-" and "end" in gene_info.columns:
            tss_mb = float(row["end"]) / 1e6
        else:
            tss_mb = float(row["start"]) / 1e6
        lookup[str(row["gene_id"])] = (str(row["chr"]), tss_mb)
    return lookup


def _build_cis_loci_dict(
    genomecache, gene_window: dict, gene_ids: list, window_mb: float
) -> dict:
    """Build {gene_id: {locus_id, ...}} restricting each gene to ±window_mb of its TSS.

    Genes missing from `gene_window` (no gene-info row) are omitted — callers
    fall back to a genome-wide scan for those via cis_loci_dict.get(gid, None).
    """
    import numpy as np

    cis_loci_dict: dict = {}
    pos_cache: dict = {}
    for gid in gene_ids:
        win = gene_window.get(str(gid))
        if win is None:
            continue
        gchr, tss_mb = win
        if gchr not in pos_cache:
            lids_c, pos_c, _ = genomecache.get_chromosome(gchr)
            pos_cache[gchr] = (lids_c, pos_c)
        lids_c, pos_c = pos_cache[gchr]
        mask = np.abs(pos_c - tss_mb) <= window_mb
        cis_loci_dict[str(gid)] = set(lids_c[mask].tolist())
    return cis_loci_dict


@click.group()
@click.version_option(package_name="mosaiqtl")
def main() -> None:
    """mosaiQTL — haplotype-based QTL mapping for multiparent populations."""


# ── fit-null ──────────────────────────────────────────────────────────────


@main.command("fit-null")
@click.option(
    "--cache",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GenomeCache HDF5 file.",
)
@click.option(
    "--counts",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X (cells × genes).",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (must contain sample_id column).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output CSV for null-fit parameters (phi_shrunk, sigma2 per gene).",
)
@click.option(
    "--cell-type", default=None, help="Cell type label to subset from AnnData obs."
)
@click.option(
    "--covariates",
    default="sex,dataset",
    show_default=True,
    help="Comma-separated metadata columns to include as fixed effects. "
    "Categorical columns are one-hot encoded (first level dropped). "
    "Pass '' to use an intercept-only model.",
)
@click.option("--verbose", is_flag=True, default=False)
def fit_null(
    cache: Path,
    counts: Path,
    metadata: Path,
    output: Path,
    cell_type: Optional[str],
    covariates: str,
    verbose: bool,
) -> None:
    """Stage 1: fit NB-GLMM null (dispersion + random effect) per gene.

    Reads raw counts from adata.raw.X. Writes a CSV with columns:
    gene_id, phi_raw, phi_shrunk, sigma2.
    This output is required by the `scan` subcommand.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    import anndata
    from .qtl.nb import fit_null_sc_native

    logger.info("Loading counts: %s", counts)
    adata = anndata.read_h5ad(counts, backed="r")
    if cell_type is not None:
        ct_col = _detect_ct_col(list(adata.obs.columns))
        if ct_col is None:
            raise click.ClickException(
                f"No cell-type column found in {counts}.obs; "
                'expected a column whose name contains both "cell" and "type".'
            )
        mask = adata.obs[ct_col] == cell_type
        adata = adata[mask]
        logger.info("Subset to cell type %r: %d cells", cell_type, adata.n_obs)

    if adata.raw is None:
        raise click.ClickException(
            "adata.raw is None — counts file must have raw counts in .raw"
        )

    # raw.X: cells × genes → transpose to genes × cells for fit_null_sc_native
    import scipy.sparse as _sp

    raw_x = adata.raw.X
    if _sp.issparse(raw_x):
        raw_x = raw_x.toarray()
    counts_mat = np.asarray(raw_x, dtype=np.float64).T  # (n_genes, n_cells)
    gene_ids = list(adata.raw.var.index)

    # sample_ids: one per cell — use obs index (sample-level barcode)
    sample_ids = np.asarray(adata.obs.index)

    meta = pd.read_csv(metadata)
    # build cell-level covariates from metadata (optional)
    covariates_cells: Optional[np.ndarray] = None
    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    if "sample_id" in meta.columns and not meta.empty:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta, covariates=_cov_cols)
        # align to cell order via sample_id
        cell_sample = pd.Series(sample_ids, name="sample_id")
        cov_aligned = cell_sample.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates_cells = np.vstack(cov_aligned.values)

    logger.info(
        "fit_null_sc_native: %d genes × %d cells",
        counts_mat.shape[0],
        counts_mat.shape[1],
    )
    result = fit_null_sc_native(
        counts_mat,
        sample_ids,
        covariates_cells=covariates_cells,
        verbose=verbose,
    )

    gene_mask = result["gene_mask"]
    passing_gene_ids = [g for g, ok in zip(gene_ids, gene_mask) if ok]
    out_df = pd.DataFrame(
        {
            "gene_id": passing_gene_ids,
            "phi_raw": result["dispersion"],
            "phi_shrunk": result["dispersion_shrunk"],
            "sigma2": result["sigma_sample"],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    logger.info("Null-fit written to %s (%d genes)", output, len(out_df))


# ── scan ──────────────────────────────────────────────────────────────────


@main.command("scan")
@click.option(
    "--cache",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GenomeCache HDF5 file.",
)
@click.option(
    "--counts",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (must contain sample_id column).",
)
@click.option(
    "--null-fit",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="CSV produced by `fit-null` (gene_id, phi_shrunk, sigma2 columns).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output CSV for per-gene peak LOD results (PEAK_LOD_COLS schema).",
)
@click.option(
    "--model",
    default="nb",
    show_default=True,
    type=click.Choice(["nb", "lmm"]),
    help="Scan model: nb (NB-GLMM, primary) or lmm (Gaussian LMM port).",
)
@click.option(
    "--cell-type", default=None, help="Cell type label to subset from AnnData obs."
)
@click.option(
    "--gene-info",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="CSV with gene_id/chr/start[/end/strand] columns; required for "
    "--scan-window local.",
)
@click.option(
    "--scan-window",
    default="genome",
    show_default=True,
    type=click.Choice(["genome", "local"]),
    help="'genome' scans every locus; 'local' restricts each gene to "
    "±window-mb of its own TSS (needs --gene-info). Works for both "
    "--model nb and --model lmm.",
)
@click.option(
    "--window-mb",
    default=1.0,
    show_default=True,
    help="Half-window in Mb around each gene's TSS for --scan-window local.",
)
@click.option("--n-cores", default=1, show_default=True, help="Parallel scan threads.")
@click.option(
    "--eqtl-scan/--no-eqtl-scan",
    default=False,
    show_default=True,
    help="Also write eqtl_scan.csv (EQTL_SCAN_COLS — all loci × all genes) "
    "alongside peak_lod.csv.  Omit for large genome-wide runs where the "
    "per-locus file would be very large; include for cis-window scans.",
)
@click.option(
    "--covariates",
    default="sex,dataset",
    show_default=True,
    help="Comma-separated metadata columns to include as fixed effects. "
    "Categorical columns are one-hot encoded (first level dropped). "
    "Pass '' to use an intercept-only model.",
)
@click.option("--verbose", is_flag=True, default=False)
def scan(
    cache: Path,
    counts: Path,
    metadata: Path,
    null_fit: Path,
    output: Path,
    model: str,
    cell_type: Optional[str],
    gene_info: Optional[Path],
    scan_window: str,
    window_mb: float,
    n_cores: int,
    eqtl_scan: bool,
    covariates: str,
    verbose: bool,
) -> None:
    """Stage 2: genome-wide QTL scan → per-gene peak LOD.

    Reads null-fit CSV from `fit-null`. Writes peak_lod.csv with columns
    matching pipeline.PEAK_LOD_COLS. For NB model, reads null dispersion
    from phi_shrunk column. For LMM model, phi_shrunk column is ignored.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    import anndata
    from .genotype.genomecache import GenomeCache
    from .pipeline import discover

    if scan_window == "local" and gene_info is None:
        raise click.ClickException("--scan-window local requires --gene-info.")

    logger.info("Loading genome cache: %s", cache)
    gc = GenomeCache.from_hdf5(cache)

    logger.info("Loading counts: %s", counts)
    adata = anndata.read_h5ad(counts, backed="r")
    if cell_type is not None:
        ct_col = _detect_ct_col(list(adata.obs.columns))
        if ct_col is None:
            raise click.ClickException(
                f"No cell-type column found in {counts}.obs; "
                'expected a column whose name contains both "cell" and "type".'
            )
        adata = adata[adata.obs[ct_col] == cell_type]
        logger.info("Subset to cell type %r: %d cells", cell_type, adata.n_obs)

    if adata.raw is None:
        raise click.ClickException(
            "adata.raw is None — counts file must have raw counts in .raw"
        )

    import scipy.sparse as _sp

    raw_x = adata.raw.X
    if _sp.issparse(raw_x):
        raw_x = raw_x.toarray()
    # discover expects (n_samples, n_genes) phenotype matrix for nb model
    counts_mat = np.asarray(raw_x, dtype=np.float64)  # (n_cells, n_genes) — cell-level
    gene_ids = list(adata.raw.var.index)
    sample_ids = list(adata.obs.index)

    meta = pd.read_csv(metadata)
    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta, covariates=_cov_cols)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates_arr = np.vstack(cov_aligned.values)

    null_params = pd.read_csv(null_fit)
    # build per-gene lib_sizes from cell-level counts (sum over genes → per cell total)
    lib_sizes = np.log(np.maximum(counts_mat.sum(axis=1), 1.0))

    # filter gene_ids to those present in null_fit
    null_gene_set = set(null_params["gene_id"].astype(str))
    gene_mask = np.array([g in null_gene_set for g in gene_ids])
    counts_filt = counts_mat[:, gene_mask]
    gene_ids_filt = [g for g, ok in zip(gene_ids, gene_mask) if ok]
    logger.info(
        "Scan: %d genes × %d cells (model=%s, scan_window=%s)",
        len(gene_ids_filt),
        counts_filt.shape[0],
        model,
        scan_window,
    )

    cis_loci_dict: Optional[dict] = None
    gene_chr_map: Optional[dict] = None
    gene_pos_mb_map: Optional[dict] = None
    if scan_window == "local":
        gene_info_df = pd.read_csv(gene_info)
        gene_window = _gene_window_lookup(gene_info_df)
        n_missing = sum(1 for g in gene_ids_filt if str(g) not in gene_window)
        if n_missing:
            logger.warning(
                "Scan: %d/%d genes missing from --gene-info — scanned genome-wide",
                n_missing,
                len(gene_ids_filt),
            )
        if model == "nb":
            # nb's score-test scan takes an explicit per-gene locus subset
            # (qtl/nb.py::scan_nb) rather than a window it resolves itself.
            cis_loci_dict = _build_cis_loci_dict(
                gc, gene_window, gene_ids_filt, window_mb
            )
        else:  # lmm — scan_multi_gene resolves the window itself per gene
            gene_chr_map = {gid: chrom for gid, (chrom, _) in gene_window.items()}
            gene_pos_mb_map = {gid: tss for gid, (_, tss) in gene_window.items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    disc = discover(
        phenotype_matrix=counts_filt,
        gene_ids=gene_ids_filt,
        genomecache=gc,
        covariates=covariates_arr,
        sample_ids=sample_ids,
        model=model,
        lib_sizes=lib_sizes,
        cis_loci_dict=cis_loci_dict,
        n_cores=n_cores,
        scan_window=scan_window if model == "lmm" else "genome",
        gene_chr_map=gene_chr_map,
        gene_pos_mb_map=gene_pos_mb_map,
        window_mb=window_mb,
    )

    # flatten ScanResult per gene to peak-lod rows
    rows = []
    for gid, sr in disc.scan.items():
        if len(sr.lod) == 0:
            continue
        pi = int(np.argmax(sr.lod))
        rows.append(
            {
                "gene_symbol": gid,
                "gene_id": gid,
                "chr": str(sr.chr),
                "peak_locus": sr.loci[pi],
                "peak_mb": float(sr.pos_mb[pi]),
                "peak_lod": float(sr.lod[pi]),
            }
        )
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output, index=False)
    logger.info("Scan written to %s (%d genes)", output, len(result_df))

    if eqtl_scan:
        from .io.results_io import write_eqtl_scan as _write_eqtl_scan

        eqtl_scan_path = output.parent / "eqtl_scan.csv"
        _write_eqtl_scan(disc.scan, gc, eqtl_scan_path)
        logger.info("eqtl_scan written to %s", eqtl_scan_path)


# ── permute ───────────────────────────────────────────────────────────────


@main.command("permute")
@click.option(
    "--cache",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GenomeCache HDF5 file.",
)
@click.option(
    "--counts",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (must contain sample_id column).",
)
@click.option(
    "--null-fit",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="CSV produced by `fit-null` (gene_id column required).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory; writes perm_max_lods.csv + per-chr CSVs.",
)
@click.option(
    "--n-perms",
    default=1000,
    show_default=True,
    help="Number of permutations for GEV threshold estimation.",
)
@click.option(
    "--gene",
    default=None,
    help="Representative gene_id for h² estimation (defaults to first passing gene).",
)
@click.option(
    "--perm-method",
    default="parametric",
    show_default=True,
    type=click.Choice(["parametric", "plain"]),
    help="Permutation mode: 'parametric' (draws null from fitted h² distribution) or 'plain' (unconditional shuffle).",
)
@click.option(
    "--gene-info",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Gene-info CSV (gene_id, chr, start[, end, strand]); required for "
    "--scan-window local.",
)
@click.option(
    "--scan-window",
    default="genome",
    show_default=True,
    type=click.Choice(["genome", "local"]),
    help="'genome' draws the null from the whole genome/chromosome (matches a "
    "genome-wide or chromosome-wide real scan); 'local' restricts the "
    "representative gene's null scan to ±window-mb of its own TSS, matching a "
    "real scan that never leaves that window.",
)
@click.option(
    "--window-mb",
    default=1.0,
    show_default=True,
    help="Half-window in Mb around the representative gene's TSS, used only "
    "when --scan-window local.",
)
@click.option(
    "--covariates",
    default="sex,dataset",
    show_default=True,
    help="Comma-separated metadata columns to include as fixed effects. "
    "Categorical columns are one-hot encoded (first level dropped). "
    "Pass '' to use an intercept-only model.",
)
@click.option("--verbose", is_flag=True, default=False)
def permute(
    cache: Path,
    counts: Path,
    metadata: Path,
    null_fit: Path,
    output: Path,
    n_perms: int,
    gene: Optional[str],
    perm_method: str,
    gene_info: Optional[Path],
    scan_window: str,
    window_mb: float,
    covariates: str,
    verbose: bool,
) -> None:
    """Permutation thresholds → GEV fit on max-LOD distribution.

    A representative gene's phenotype is used for h² estimation.
    Writes perm_max_lods.csv and perm_max_lods_chr{c}.csv into OUTPUT directory.
    """
    _setup_logging(verbose)
    if scan_window == "local" and gene_info is None:
        raise click.ClickException("--scan-window local requires --gene-info.")
    import numpy as np
    import pandas as pd
    import anndata
    from .genotype.genomecache import GenomeCache
    from .genotype import kinship as kin_module
    from .qtl.permutation import generate_null_outcomes, run_threshold_scans
    from .qtl.thresholds import get_gev_thresholds

    gc = GenomeCache.from_hdf5(cache)
    adata = anndata.read_h5ad(counts, backed="r")
    null_params = pd.read_csv(null_fit)

    import scipy.sparse as _sp

    raw_x = adata.raw.X if adata.raw is not None else adata.X
    if _sp.issparse(raw_x):
        raw_x = raw_x.toarray()
    counts_mat = np.asarray(raw_x, dtype=np.float64)  # (n_cells, n_genes)
    gene_ids = list(adata.raw.var.index if adata.raw is not None else adata.var.index)

    # choose representative gene
    null_gene_ids = list(null_params["gene_id"].astype(str))
    rep_gene = (
        gene if gene is not None else (null_gene_ids[0] if null_gene_ids else None)
    )
    if rep_gene is None:
        raise click.ClickException("No genes found in null-fit CSV.")
    if rep_gene not in gene_ids:
        raise click.ClickException(
            f"Representative gene {rep_gene!r} not found in counts file."
        )
    gi = gene_ids.index(rep_gene)
    phenotype = counts_mat[:, gi]

    meta = pd.read_csv(metadata)
    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta, covariates=_cov_cols)
        sample_ids = list(adata.obs.index)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates_arr = np.vstack(cov_aligned.values)

    kd_raw = kin_module.calc_kinship(gc, method="loco")
    kinship_decomp = kin_module.decomp_kinship(kd_raw)

    null_outcomes = generate_null_outcomes(
        phenotype, n_perms=n_perms, method="permutation"
    )

    perm_chrom: Optional[str] = None
    perm_gene_pos_mb: Optional[float] = None
    if scan_window == "local":
        gene_info_df = pd.read_csv(gene_info)
        gene_window = _gene_window_lookup(gene_info_df)
        win = gene_window.get(str(rep_gene))
        if win is None:
            logger.warning(
                "permute: %s not found in --gene-info — falling back to "
                "genome-wide null",
                rep_gene,
            )
        else:
            perm_chrom, perm_gene_pos_mb = win
            logger.info(
                "permute: local-window null: chr%s TSS=%.3fMb ±%.1fMb",
                perm_chrom,
                perm_gene_pos_mb,
                window_mb,
            )

    max_lods, chr_null_lods = run_threshold_scans(
        null_outcomes,
        gc,
        phenotype=phenotype,
        covariates=covariates_arr,
        kinship_decomp=kinship_decomp,
        kinship_loco=kd_raw,
        chrom=perm_chrom,
        gene_pos_mb=perm_gene_pos_mb,
        window_mb=window_mb if perm_chrom is not None else None,
        permutation_method=perm_method,
    )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"max_lod": max_lods}).to_csv(
        output / "perm_max_lods.csv", index=False
    )
    for chrom, lods in chr_null_lods.items():
        pd.DataFrame({"max_lod": lods}).to_csv(
            output / f"perm_max_lods_chr{chrom}.csv",
            index=False,
        )
    thr, _ = get_gev_thresholds(max_lods)
    if thr is not None:
        logger.info("GEV 95th-pct threshold: %.3f LOD", thr)
    logger.info("Permutation results written to %s", output)


# ── estimate ──────────────────────────────────────────────────────────────


@main.command("estimate")
@click.option(
    "--cache",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GenomeCache HDF5 file.",
)
@click.option(
    "--scan-result",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Peak LOD CSV produced by `scan` (gene_symbol, chr, peak_locus columns).",
)
@click.option(
    "--counts",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (must contain sample_id column).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output CSV for founder haplotype effects (ALLELE_EFFECTS_BASE_COLS + beta_*).",
)
@click.option(
    "--method",
    default="fixef",
    show_default=True,
    type=click.Choice(["fixef", "ranef"]),
    help="Effect estimation method: fixef (pseudoinverse) or ranef (ridge BLUP).",
)
@click.option(
    "--cell-type", default=None, help="Cell type label to subset from AnnData obs."
)
@click.option(
    "--covariates",
    default="sex,dataset",
    show_default=True,
    help="Comma-separated metadata columns to include as fixed effects. "
    "Categorical columns are one-hot encoded (first level dropped). "
    "Pass '' to use an intercept-only model.",
)
@click.option("--verbose", is_flag=True, default=False)
def estimate(
    cache: Path,
    scan_result: Path,
    counts: Path,
    metadata: Path,
    output: Path,
    method: str,
    cell_type: Optional[str],
    covariates: str,
    verbose: bool,
) -> None:
    """Estimate haplotype effects at each gene's peak locus.

    Writes allele_effects.csv with columns defined in pipeline.ALLELE_EFFECTS_BASE_COLS
    plus beta_{founder} and se_{founder} columns per founder.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    import anndata
    from .genotype.genomecache import GenomeCache
    from .genotype import kinship as kin_module
    from .qtl.effects import get_allele_effects

    gc = GenomeCache.from_hdf5(cache)
    peaks = pd.read_csv(scan_result)

    adata = anndata.read_h5ad(counts, backed="r")
    if cell_type is not None:
        ct_col = _detect_ct_col(list(adata.obs.columns))
        if ct_col is None:
            raise click.ClickException(
                f"No cell-type column found in {counts}.obs; "
                'expected a column whose name contains both "cell" and "type".'
            )
        adata = adata[adata.obs[ct_col] == cell_type]

    import scipy.sparse as _sp

    raw_x = adata.raw.X if adata.raw is not None else adata.X
    if _sp.issparse(raw_x):
        raw_x = raw_x.toarray()
    counts_mat = np.asarray(raw_x, dtype=np.float64)  # (n_cells, n_genes)
    gene_ids = list(adata.raw.var.index if adata.raw is not None else adata.var.index)
    sample_ids = list(adata.obs.index)

    meta = pd.read_csv(metadata)
    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta, covariates=_cov_cols)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates_arr = np.vstack(cov_aligned.values)

    kd_raw = kin_module.calc_kinship(gc, method="loco")
    kinship_decomp = kin_module.decomp_kinship(kd_raw)

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}
    for _, row in peaks.iterrows():
        gid = str(row.get("gene_symbol", row.get("gene_id", "")))
        chrom = str(row["chr"])
        locus_id = str(row["peak_locus"])
        if gid not in gene_id_to_idx:
            logger.warning("Gene %r not found in counts file — skipping", gid)
            continue
        gi = gene_id_to_idx[gid]
        phenotype = counts_mat[:, gi]
        try:
            founder_effects = get_allele_effects(
                phenotype=phenotype,
                genomecache=gc,
                locus_id=locus_id,
                chrom=chrom,
                kinship_decomp=kinship_decomp,
                covariates=covariates_arr,
                sample_ids=sample_ids,
                method=method,
            )
            effect_row: dict = {
                "gene_symbol": gid,
                "gene_id": gid,
                "chr": chrom,
                "peak_locus": locus_id,
                "peak_mb": float(row.get("peak_mb", float("nan"))),
                "peak_lod": float(row.get("peak_lod", float("nan"))),
            }
            for fe in founder_effects:
                effect_row[f"beta_{fe.founder}"] = fe.estimate
                effect_row[f"se_{fe.founder}"] = fe.se
            rows.append(effect_row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("effects failed for %s (locus %s): %s", gid, locus_id, exc)
    pd.DataFrame(rows).to_csv(output, index=False)
    logger.info("Effects written to %s (%d genes)", output, len(rows))


# ── report ────────────────────────────────────────────────────────────────


@main.command("report")
@click.option(
    "--scan-result",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Peak LOD CSV produced by `scan`.",
)
@click.option(
    "--perm-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory produced by `permute` (contains perm_max_lods.csv).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for summary tables and QC plots.",
)
@click.option(
    "--fdr",
    default=0.05,
    show_default=True,
    help="LOD threshold for eGene calling (applied after GEV 95th-pct); also "
    "used as the Storey q-value cutoff when --pval-perm is given.",
)
@click.option(
    "--allele-effects",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="allele_effects.csv from `estimate` — merges beta_{founder} effect "
    "columns into eqtl_peak.csv. Omit to write eqtl_peak.csv with effect "
    "columns blank.",
)
@click.option(
    "--pval-perm",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="CSV with gene_id,pval_perm columns (per-gene calibrated permutation "
    "p-value from qtl.permutation.calibrate_gene_pvalue) — enables the "
    "Storey q-value significance call written into eqtl_peak.csv. Omit to "
    "leave pval_perm/fdr/significance blank there.",
)
@click.option(
    "--suffix",
    default=None,
    help="Optional tag appended to output filenames: eqtl_peak_<suffix>.csv, "
    "eqtl_stats_<suffix>.csv, eqtl_summary_<suffix>.csv. Convention: "
    "<tissue>_<celltype>_<scan-tag>, e.g. ctxhipp_glutamatergic_neuron_Lw1.",
)
@click.option("--verbose", is_flag=True, default=False)
def report(
    scan_result: Path,
    perm_dir: Path,
    output_dir: Path,
    fdr: float,
    allele_effects: Optional[Path],
    pval_perm: Optional[Path],
    suffix: Optional[str],
    verbose: bool,
) -> None:
    """Summary tables + QC plots from a completed scan.

    Reads peak_lod.csv + perm directory from prior subcommands.
    Writes eqtl_summary.csv (eGene table with Storey FDR columns),
    genome_summary.png, eqtl_peak.csv (one row per eGene, haplotype-level stats),
    and eqtl_stats.csv (run-level summary) into OUTPUT_DIR. Allele-contrast
    fields (variant, REF/ALT founders, AF, effect size) remain NaN until
    Sanger fine-mapping supplies real REF/ALT calls.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    from .genotype.cc import FOUNDER_ORDER
    from .io.results_io import write_eqtl_peak, write_eqtl_stats
    from .pipeline import calibrate_egenes_storey, call_egenes, finalize_egene_summary
    from .qtl.thresholds import get_gev_thresholds, load_per_chr_thresholds
    from .utils import git_hash

    peaks = pd.read_csv(scan_result)
    perm_path = perm_dir / "perm_max_lods.csv"
    if not perm_path.exists():
        raise click.ClickException(f"perm_max_lods.csv not found in {perm_dir}")
    max_lods = pd.read_csv(perm_path)["max_lod"].values
    gw_thr, _ = get_gev_thresholds(max_lods)
    if gw_thr is None:
        gw_thr = float(np.percentile(max_lods, 95))
        logger.warning("GEV fit failed; using empirical p95 = %.3f", gw_thr)
    else:
        logger.info("GEV 95th-pct threshold: %.3f LOD", gw_thr)

    output_dir.mkdir(parents=True, exist_ok=True)
    # derive tag from scan-result stem when not supplied:
    # peak_lod_glutamatergic_neuron_Lw1.csv  →  glutamatergic_neuron_Lw1
    _tag = suffix or scan_result.stem.removeprefix("peak_lod_")
    egenes_df = call_egenes(peaks, threshold=gw_thr)
    is_egene = (
        peaks["peak_lod"] >= gw_thr
        if "peak_lod" in peaks.columns
        else (peaks.index.isin(egenes_df.index))
    )
    summary = finalize_egene_summary(egenes_df)
    summary.to_csv(output_dir / f"eqtl_summary_{_tag}.csv", index=False)
    logger.info("eQTL summary written: %d eGenes", len(egenes_df))

    # genome summary Manhattan
    try:
        import matplotlib

        matplotlib.use("Agg")
        from .viz.genome import genome_summary

        thr_by_chr = load_per_chr_thresholds(perm_dir)
        fig = genome_summary(peaks, is_egene, thr_by_chr=thr_by_chr)
        fig.savefig(output_dir / "genome_summary.png", bbox_inches="tight", dpi=300)
        import matplotlib.pyplot as plt

        plt.close(fig)
        logger.info("genome_summary.png written")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not produce genome_summary.png: %s", exc)

    # eqtl_peak.csv — IGVF EQTL_PEAK_COLS (minus biallelic-contrast fields)
    effects_egenes = egenes_df.copy()
    founder_names = list(FOUNDER_ORDER)
    if allele_effects is not None:
        ae = pd.read_csv(allele_effects)
        beta_cols = [c for c in ae.columns if c.startswith("beta_")]
        if beta_cols:
            founder_names = [c[len("beta_") :] for c in beta_cols]  # noqa: E203
            join_col = "gene_id" if "gene_id" in ae.columns else "gene_symbol"
            effects_egenes = effects_egenes.merge(
                ae[[join_col, *beta_cols]], on=join_col, how="left"
            )
        else:
            logger.warning(
                "%s has no beta_* columns; effect_* left blank", allele_effects
            )
    else:
        logger.info("--allele-effects not given; effect_* columns left blank")

    if pval_perm is not None:
        _pv_df = pd.read_csv(pval_perm)
        if "gene_id" not in _pv_df.columns or "pval_perm" not in _pv_df.columns:
            raise click.ClickException(
                f"--pval-perm expects a CSV with columns 'gene_id' and 'pval_perm' "
                f"(produced by S6.5 per-gene calibration in validate_tissue.py). "
                f"Got columns: {list(_pv_df.columns)}. "
                f"Did you accidentally pass the GEV threshold file?"
            )
        pv_series = _pv_df.set_index("gene_id")["pval_perm"]
        effects_egenes = calibrate_egenes_storey(
            effects_egenes, pv_series, fdr_threshold=fdr
        )
    else:
        logger.info(
            "--pval-perm not given; pval_perm/fdr/significance left blank in eqtl_peak.csv"
        )

    write_eqtl_peak(effects_egenes, founder_names, output_dir / f"eqtl_peak_{_tag}.csv")
    logger.info("eqtl_peak_%s.csv written: %d rows", _tag, len(effects_egenes))

    # eqtl_stats.csv — IGVF EQTL_STATS_COLS, per-chromosome + genome-wide rows
    chrom_stats = []
    for pf in sorted(perm_dir.glob("perm_max_lods_chr*.csv")):
        chrom = pf.stem.replace("perm_max_lods_chr", "")
        try:
            chr_nulls = pd.read_csv(pf)["max_lod"].values
            chr_thr, params = get_gev_thresholds(chr_nulls)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fit GEV for chr%s: %s", chrom, exc)
            continue
        chr_peaks = (
            peaks[peaks["chr"].astype(str) == chrom]
            if "chr" in peaks.columns
            else peaks.iloc[0:0]
        )
        chrom_stats.append(
            {
                "chr": chrom,
                "lod_threshold": chr_thr,
                "gev_mu": params.get("loc") if params else None,
                "gev_sigma": params.get("scale") if params else None,
                "gev_xi": params.get("c") if params else None,
                "n_genes_scanned": int(len(chr_peaks)),
                "n_egenes": (
                    int((chr_peaks["peak_lod"] >= chr_thr).sum())
                    if chr_thr is not None
                    else None
                ),
                "n_perms": int(len(chr_nulls)),
            }
        )
    genome_stats = {
        "lod_threshold": gw_thr,
        "n_genes_scanned": int(len(peaks)),
        "n_egenes": int(len(egenes_df)),
        "n_perms": int(len(max_lods)),
        "fdr_threshold": fdr,
        "git_hash": git_hash(),
    }
    write_eqtl_stats(chrom_stats, genome_stats, output_dir / f"eqtl_stats_{_tag}.csv")
    logger.info("eqtl_stats_%s.csv written: %d rows", _tag, len(chrom_stats) + 1)


# ── finemap ───────────────────────────────────────────────────────────────


@main.command("finemap")
@click.option(
    "--egenes",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="eqtl_peak.csv from `report` — must have ci_lo_bp and ci_hi_bp columns.",
)
@click.option(
    "--counts",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (must contain sample_id column).",
)
@click.option(
    "--cache",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GenomeCache HDF5 file.",
)
@click.option(
    "--alleles",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Founder allele HDF5 built by build_founder_allele_gm.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for eqtl_finemapping*.csv output files.",
)
@click.option(
    "--cell-type", default=None, help="Cell type label to subset from AnnData obs."
)
@click.option(
    "--counts-layer",
    default=None,
    help="AnnData layer name for counts (e.g. 'decontXcounts').  "
    "Falls back to adata.raw.X when omitted.",
)
@click.option(
    "--coverage",
    default=0.95,
    show_default=True,
    help="Credible-set coverage for Wakefield ABF (default 0.95).",
)
@click.option(
    "--ref-founder",
    default="B6J",
    show_default=True,
    help="REF founder for biallelic allele collapse.",
)
@click.option(
    "--prior-var",
    default=0.04,
    show_default=True,
    help="Wakefield prior variance.",
)
@click.option(
    "--covariates",
    default="sex,dataset",
    show_default=True,
    help="Comma-separated metadata columns to include as fixed effects. "
    "Categorical columns are one-hot encoded (first level dropped). "
    "Pass '' to use an intercept-only model.",
)
@click.option(
    "--suffix",
    default=None,
    help="Tag appended to output filenames, e.g. gastroc_Lw1.  "
    "Inferred from --egenes stem when omitted.",
)
@click.option("--verbose", is_flag=True, default=False)
def finemap(
    egenes: Path,
    counts: Path,
    metadata: Path,
    cache: Path,
    alleles: Path,
    output_dir: Path,
    cell_type: Optional[str],
    counts_layer: Optional[str],
    coverage: float,
    ref_founder: str,
    prior_var: float,
    covariates: str,
    suffix: Optional[str],
    verbose: bool,
) -> None:
    """Sanger fine-mapping of eGenes using Wakefield ABF within LOD-drop CIs.

    Reads eqtl_peak.csv (ci_lo_bp and ci_hi_bp columns required; produced by
    the `report` subcommand after LOD support CI computation). For each eGene, loads
    Sanger alleles once per chromosome, imputes biallelic dosages via
    make_alt_mask, runs the LMM SNP scan within the CI window, and computes
    Wakefield ABF posterior inclusion probabilities.

    Writes eqtl_finemapping_{suffix}.csv (EQTL_FINEMAPPING_COLS, one row per
    credible-set SNP) and eqtl_finemapping_peak_{suffix}.csv
    (EQTL_FINEMAPPING_PEAK_COLS, one row per eGene lead SNP).
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    import anndata
    import scipy.sparse as _sp
    from .genotype.genomecache import GenomeCache
    from .genotype import kinship as kin_module
    from .pipeline import finemap_egenes as _finemap_egenes

    egene_df = pd.read_csv(egenes)
    _req = {"gene_id", "chr", "ci_lo_bp", "ci_hi_bp"}
    _missing = _req - set(egene_df.columns)
    if _missing:
        raise click.ClickException(
            f"--egenes file missing required columns: {sorted(_missing)}.  "
            "Run `report` first, then the characterise step to compute ci_lo_bp/ci_hi_bp."
        )

    logger.info("Loading genome cache: %s", cache)
    gc = GenomeCache.from_hdf5(cache)

    logger.info("Loading counts: %s", counts)
    adata = anndata.read_h5ad(counts, backed="r")
    if cell_type is not None:
        ct_col = _detect_ct_col(list(adata.obs.columns))
        if ct_col is None:
            raise click.ClickException(
                f"No cell-type column found in {counts}.obs; "
                'expected a column whose name contains both "cell" and "type".'
            )
        adata = adata[adata.obs[ct_col] == cell_type]
        logger.info("Subset to cell type %r: %d cells", cell_type, adata.n_obs)

    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise click.ClickException(
                f"--counts-layer {counts_layer!r} not found in {counts}.layers.  "
                f"Available layers: {list(adata.layers.keys())}"
            )
        raw_x = adata.layers[counts_layer]
        gene_ids = list(adata.var.index)
        logger.info("Using counts from layer %r", counts_layer)
    else:
        if adata.raw is None:
            raise click.ClickException(
                "adata.raw is None — pass --counts-layer or store raw counts in .raw"
            )
        raw_x = adata.raw.X
        gene_ids = list(adata.raw.var.index)
    if _sp.issparse(raw_x):
        raw_x = raw_x.toarray()
    counts_mat = np.asarray(raw_x, dtype=np.float64)  # (n_cells, n_genes)
    sample_ids = list(adata.obs.index)

    meta = pd.read_csv(metadata)
    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta, covariates=_cov_cols)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates_arr = np.vstack(cov_aligned.values)

    logger.info("Computing LOCO kinship...")
    kd_raw = kin_module.calc_kinship(gc, method="loco")
    kinship_decomp = kin_module.decomp_kinship(kd_raw)

    _tag = suffix or egenes.stem.removeprefix("eqtl_peak_")
    _finemap_egenes(
        egene_df=egene_df,
        phenotype_matrix=counts_mat.T,  # (n_genes, n_cells)
        gene_ids=gene_ids,
        genomecache=gc,
        alleles_h5=alleles,
        out_dir=output_dir,
        covariates=covariates_arr,
        kinship_decomp=kinship_decomp,
        tag=_tag,
        coverage=coverage,
        ref_founder=ref_founder,
        prior_var=prior_var,
    )


if __name__ == "__main__":
    main()
