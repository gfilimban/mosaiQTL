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
        format="%(asctime)s,%(msecs)03d %(levelname)-5s [%(module)s] %(message)s",
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
@click.option("--verbose", is_flag=True, default=False)
def fit_null(
    cache: Path,
    counts: Path,
    metadata: Path,
    output: Path,
    cell_type: Optional[str],
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
    if "sample_id" in meta.columns and not meta.empty:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta)
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
@click.option("--n-cores", default=1, show_default=True, help="Parallel scan threads.")
@click.option("--verbose", is_flag=True, default=False)
def scan(
    cache: Path,
    counts: Path,
    metadata: Path,
    null_fit: Path,
    output: Path,
    model: str,
    cell_type: Optional[str],
    n_cores: int,
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
    covariates: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates = np.vstack(cov_aligned.values)

    null_params = pd.read_csv(null_fit)
    # build per-gene lib_sizes from cell-level counts (sum over genes → per cell total)
    lib_sizes = np.log(np.maximum(counts_mat.sum(axis=1), 1.0))

    # filter gene_ids to those present in null_fit
    null_gene_set = set(null_params["gene_id"].astype(str))
    gene_mask = np.array([g in null_gene_set for g in gene_ids])
    counts_filt = counts_mat[:, gene_mask]
    gene_ids_filt = [g for g, ok in zip(gene_ids, gene_mask) if ok]
    logger.info(
        "Scan: %d genes × %d cells (model=%s)",
        len(gene_ids_filt),
        counts_filt.shape[0],
        model,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    disc = discover(
        phenotype_matrix=counts_filt,
        gene_ids=gene_ids_filt,
        genomecache=gc,
        covariates=covariates,
        sample_ids=sample_ids,
        model=model,
        lib_sizes=lib_sizes,
        n_cores=n_cores,
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
    help="Permutation mode: 'parametric' (miQTL default) or 'plain'.",
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
    verbose: bool,
) -> None:
    """Permutation thresholds → GEV fit on max-LOD distribution.

    A representative gene's phenotype is used for h² estimation.
    Writes perm_max_lods.csv and perm_max_lods_chr{c}.csv into OUTPUT directory.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    import anndata
    from .genotype.genomecache import GenomeCache
    from .genotype import kinship as kin_module
    from .qtl.permutation import (
        generate_null_outcomes,
        run_threshold_scans,
        get_gev_thresholds,
    )

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
    covariates: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta)
        sample_ids = list(adata.obs.index)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates = np.vstack(cov_aligned.values)

    kd_raw = kin_module.calc_kinship(gc, method="loco")
    kinship_decomp = kin_module.decomp_kinship(kd_raw)

    null_outcomes = generate_null_outcomes(
        phenotype, n_perms=n_perms, method="permutation"
    )
    max_lods, chr_null_lods = run_threshold_scans(
        null_outcomes,
        gc,
        phenotype=phenotype,
        covariates=covariates,
        kinship_decomp=kinship_decomp,
        kinship_loco=kd_raw,
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
@click.option("--verbose", is_flag=True, default=False)
def estimate(
    cache: Path,
    scan_result: Path,
    counts: Path,
    metadata: Path,
    output: Path,
    method: str,
    cell_type: Optional[str],
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
    covariates: Optional[np.ndarray] = None
    if "sample_id" in meta.columns:
        from .io.covariates import build_covariate_matrix

        cov_df = build_covariate_matrix(meta)
        sid_series = pd.Series(sample_ids, name="sample_id")
        cov_aligned = sid_series.map(
            cov_df.set_index(cov_df.index).apply(lambda r: r.tolist(), axis=1)
        )
        if not cov_aligned.isna().any():
            covariates = np.vstack(cov_aligned.values)

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
                covariates=covariates,
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
@click.option("--verbose", is_flag=True, default=False)
def report(
    scan_result: Path,
    perm_dir: Path,
    output_dir: Path,
    fdr: float,
    allele_effects: Optional[Path],
    pval_perm: Optional[Path],
    verbose: bool,
) -> None:
    """Summary tables + QC plots from a completed scan.

    Reads peak_lod.csv + perm directory from prior subcommands.
    Writes eqtl_summary.csv, genome_summary.png, eqtl_peak.csv, and
    eqtl_stats.csv into OUTPUT_DIR. eqtl_peak.csv/eqtl_stats.csv populate the
    IGVF EQTL_PEAK_COLS/EQTL_STATS_COLS schemas from data already produced by
    this pipeline; the biallelic-contrast fields (variant/ref/ref_founders/
    alt_founders/af/maf/mac/effect_size/...) stay NaN until the Sanger
    fine-mapping subsystem supplies real REF/ALT calls.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    from .genotype.founders import FOUNDER_ORDER
    from .io.results_io import write_eqtl_peak, write_eqtl_stats
    from .pipeline import calibrate_egenes_storey, call_egenes, finalize_egene_summary
    from .qtl.permutation import get_gev_thresholds, load_per_chr_thresholds
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
    egenes_df = call_egenes(peaks, threshold=gw_thr)
    is_egene = (
        peaks["peak_lod"] >= gw_thr
        if "peak_lod" in peaks.columns
        else (peaks.index.isin(egenes_df.index))
    )
    summary = finalize_egene_summary(egenes_df)
    summary.to_csv(output_dir / "eqtl_summary.csv", index=False)
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
            founder_names = [c[len("beta_"):] for c in beta_cols]
            join_col = "gene_id" if "gene_id" in ae.columns else "gene_symbol"
            effects_egenes = effects_egenes.merge(
                ae[[join_col, *beta_cols]], on=join_col, how="left"
            )
        else:
            logger.warning("%s has no beta_* columns; effect_* left blank", allele_effects)
    else:
        logger.info("--allele-effects not given; effect_* columns left blank")

    if pval_perm is not None:
        pv_series = pd.read_csv(pval_perm).set_index("gene_id")["pval_perm"]
        effects_egenes = calibrate_egenes_storey(effects_egenes, pv_series, fdr_threshold=fdr)
    else:
        logger.info(
            "--pval-perm not given; pval_perm/fdr/significance left blank in eqtl_peak.csv"
        )

    write_eqtl_peak(effects_egenes, founder_names, output_dir / "eqtl_peak.csv")
    logger.info("eqtl_peak.csv written: %d rows", len(effects_egenes))

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
    write_eqtl_stats(chrom_stats, genome_stats, output_dir / "eqtl_stats.csv")
    logger.info("eqtl_stats.csv written: %d rows", len(chrom_stats) + 1)


if __name__ == "__main__":
    main()
