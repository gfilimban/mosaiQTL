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


def _covariates_for_samples(
    metadata_path: "Path",
    sample_ids: "list[str]",
    cov_cols: "list[str]",
) -> "Optional[np.ndarray]":
    """Load metadata, restrict to scan samples, build covariate matrix.

    Returns (n_samples, n_covariates) float64 array aligned to sample_ids,
    or None when sample_id column is absent or any sample is missing.
    """
    from .io.covariates import build_covariate_matrix

    meta = pd.read_csv(metadata_path)
    if "sample_id" not in meta.columns:
        return None
    meta_scan = meta[meta["sample_id"].isin(sample_ids)].drop_duplicates("sample_id")
    cov_df = build_covariate_matrix(meta_scan, covariates=cov_cols)
    cov_aligned = cov_df.reindex(sample_ids)
    if cov_aligned.isna().any().any():
        return None
    return cov_aligned.values


def _load_phenotypes_h5(
    h5_path: "Path",
    gene_ids: "list[str] | None" = None,
) -> "tuple[np.ndarray, list[str], list[str]]":
    """Load phenotype matrix from counts_lognorm.h5 (custom HDF5).

    Returns (mat, gene_ids_out, sample_ids) where mat is (n_genes, n_samples).
    When gene_ids is provided only those genes are returned (in order given).
    """
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        _sid_key = "sample_ids" if "sample_ids" in f else "samples"
        _gid_key = "gene_ids" if "gene_ids" in f else "genes"
        sids: list[str] = [
            s.decode() if isinstance(s, bytes) else s for s in f[_sid_key][:]
        ]
        all_gids: list[str] = [
            g.decode() if isinstance(g, bytes) else g for g in f[_gid_key][:]
        ]
        if gene_ids is None:
            mat = f["counts"][:].astype(np.float64)
            return mat, all_gids, sids
        gid_idx = {g: i for i, g in enumerate(all_gids)}
        idxs = [gid_idx[g] for g in gene_ids if g in gid_idx]
        found = [g for g in gene_ids if g in gid_idx]
        mat = f["counts"][np.array(idxs), :].astype(np.float64) if idxs else np.empty(
            (0, len(sids))
        )
    return mat, found, sids


def _load_kinship_h5(
    h5_path: "Path",
) -> "dict[str, dict[str, np.ndarray]]":
    """Load all-chromosome LOCO kinship eigendecomp from HDF5.

    Returns {chrom: {'values': ndarray, 'vectors': ndarray}}.
    """
    import h5py
    import numpy as np

    decomp: dict = {}
    with h5py.File(h5_path, "r") as f:
        for key in f:
            if key.startswith("chr"):
                c = key[3:]
                decomp[c] = {
                    "values": f[key]["values"][:].astype(np.float64),
                    "vectors": f[key]["vectors"][:].astype(np.float64),
                }
    return decomp


def _pick_median_h2_gene(
    h5_path: "Path",
    gc: "object",
    kinship_decomp: "dict",
    n_sample: int = 200,
    seed: int = 1,
) -> "tuple[str, float]":
    """Sample genes, estimate h² for each, return (gene_id, median_h²).

    Mirrors validate_tissue.py S5 median-h² representative gene selection.
    """
    import h5py
    import numpy as np
    from natsort import natsorted
    from .qtl.lmm import estimate_h2_null
    from .qtl.scan import _resolve_z
    from .utils import rint

    with h5py.File(h5_path, "r") as f:
        _sid_key = "sample_ids" if "sample_ids" in f else "samples"
        _gid_key = "gene_ids" if "gene_ids" in f else "genes"
        sids: list[str] = [
            s.decode() if isinstance(s, bytes) else s for s in f[_sid_key][:]
        ]
        all_gids: list[str] = [
            g.decode() if isinstance(g, bytes) else g for g in f[_gid_key][:]
        ]
        rng = np.random.default_rng(seed)
        n = min(n_sample, len(all_gids))
        idxs = np.sort(rng.choice(len(all_gids), n, replace=False))
        mat = f["counts"][idxs, :].astype(np.float64)
        sampled_gids = [all_gids[i] for i in idxs]

    Z, _ = _resolve_z(sids, gc)
    n_per_strain = Z.sum(axis=0)
    ref_chr = natsorted(kinship_decomp.keys())[0]
    ev = kinship_decomp[ref_chr]["values"]
    evc = kinship_decomp[ref_chr]["vectors"]
    n_strains = len(gc.sample_ids)
    x_rot = evc.T @ np.ones((n_strains, 1))

    h2_pairs: list[tuple[str, float]] = []
    for i, gid in enumerate(sampled_gids):
        y_s = np.where(n_per_strain > 0, (Z.T @ mat[i]) / n_per_strain, 0.0)
        y_s = rint(y_s)
        if np.isnan(y_s).any():
            continue
        try:
            h2, _ = estimate_h2_null(evc.T @ y_s, x_rot, ev)
            h2_pairs.append((gid, float(h2)))
        except Exception:  # noqa: BLE001
            pass

    if not h2_pairs:
        logger.warning("h² estimation failed for all sampled genes; using first gene")
        return all_gids[0], float("nan")

    h2_arr = np.array([h for _, h in h2_pairs])
    h2_med = float(np.median(h2_arr))
    best = int(np.argmin(np.abs(h2_arr - h2_med)))
    rep_gene, rep_h2 = h2_pairs[best]
    logger.info(
        "median h²=%.3f  representative gene: %s (h²=%.3f)", h2_med, rep_gene, rep_h2
    )
    return rep_gene, h2_med


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

    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_cells = _covariates_for_samples(metadata, list(sample_ids), _cov_cols)

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

    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr = _covariates_for_samples(metadata, sample_ids, _cov_cols)

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
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X (NB mode; omit for LMM mode).",
)
@click.option(
    "--phenotypes",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Phenotype HDF5 file (counts_lognorm.h5 format; LMM mode).",
)
@click.option(
    "--kinship",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Pre-built LOCO kinship HDF5 (skips recompute; recommended for LMM mode).",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (must contain sample_id column).",
)
@click.option(
    "--null-fit",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="CSV produced by `fit-null` (gene_id column required; NB mode only).",
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
    counts: Optional[Path],
    phenotypes: Optional[Path],
    kinship: Optional[Path],
    metadata: Path,
    null_fit: Optional[Path],
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

    LMM mode (default): provide --phenotypes (counts_lognorm.h5) and optionally
    --kinship (pre-built LOCO HDF5). Representative gene is chosen by median-h²,
    matching validate_tissue.py S5.

    NB mode: provide --counts (AnnData h5ad) and --null-fit.

    Writes perm_max_lods.csv and perm_max_lods_chr{c}.csv into OUTPUT directory.
    """
    _setup_logging(verbose)
    lmm_mode = phenotypes is not None
    if not lmm_mode and (counts is None or null_fit is None):
        raise click.ClickException(
            "LMM mode requires --phenotypes; NB mode requires both --counts and --null-fit."
        )
    if scan_window == "local" and gene_info is None:
        raise click.ClickException("--scan-window local requires --gene-info.")

    import numpy as np
    import pandas as pd
    from .genotype.genomecache import GenomeCache
    from .genotype import kinship as kin_module
    from .qtl.permutation import generate_null_outcomes, run_threshold_scans
    from .qtl.thresholds import get_gev_thresholds

    gc = GenomeCache.from_hdf5(cache)

    # ── kinship ────────────────────────────────────────────────────────────────
    if kinship is not None:
        kinship_decomp = _load_kinship_h5(kinship)
        kd_raw = None
    else:
        kd_raw = kin_module.calc_kinship(gc, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # ── representative gene + phenotype ───────────────────────────────────────
    if lmm_mode:
        if gene is not None:
            mat, found_gids, sample_ids_list = _load_phenotypes_h5(phenotypes, [gene])
            if not found_gids:
                raise click.ClickException(
                    f"--gene {gene!r} not found in --phenotypes file."
                )
            phenotype_vec = mat[0]
            rep_gene = gene
        else:
            rep_gene, _ = _pick_median_h2_gene(phenotypes, gc, kinship_decomp)
            mat, found_gids, sample_ids_list = _load_phenotypes_h5(
                phenotypes, [rep_gene]
            )
            phenotype_vec = mat[0]
        sample_ids: list = sample_ids_list
    else:
        import anndata
        import scipy.sparse as _sp

        adata = anndata.read_h5ad(counts, backed="r")
        null_params = pd.read_csv(null_fit)
        raw_x = adata.raw.X if adata.raw is not None else adata.X
        if _sp.issparse(raw_x):
            raw_x = raw_x.toarray()
        counts_mat = np.asarray(raw_x, dtype=np.float64)
        gene_ids = list(adata.raw.var.index if adata.raw is not None else adata.var.index)
        sample_ids = list(adata.obs.index)
        null_gene_ids = list(null_params["gene_id"].astype(str))
        rep_gene = gene if gene is not None else (null_gene_ids[0] if null_gene_ids else None)
        if rep_gene is None:
            raise click.ClickException("No genes found in null-fit CSV.")
        if rep_gene not in gene_ids:
            raise click.ClickException(
                f"Representative gene {rep_gene!r} not found in counts file."
            )
        phenotype_vec = counts_mat[:, gene_ids.index(rep_gene)]

    # ── local-window setup (mirrors validate_tissue.py S5) ────────────────────
    perm_chrom: Optional[str] = None
    perm_gene_pos_mb: Optional[float] = None
    if scan_window == "local":
        gene_info_df = pd.read_csv(gene_info)
        perm_row = gene_info_df[gene_info_df["gene_id"].astype(str) == str(rep_gene)]
        if perm_row.empty:
            logger.warning(
                "permute: %s not found in --gene-info — falling back to genome-wide null",
                rep_gene,
            )
        else:
            _prow = perm_row.iloc[0]
            perm_chrom = str(_prow["chr"])
            if "strand" in perm_row.columns and _prow["strand"] == "-":
                perm_gene_pos_mb = float(_prow["end"]) / 1e6
            else:
                perm_gene_pos_mb = float(_prow["start"]) / 1e6
            logger.info(
                "permute: local-window null: chr%s TSS=%.3fMb ±%.1fMb",
                perm_chrom,
                perm_gene_pos_mb,
                window_mb,
            )

    # ── permutation scan (covariates=None — matches validate_tissue.py S5) ────
    null_outcomes = generate_null_outcomes(
        phenotype_vec, n_perms=n_perms, method="permutation"
    )
    max_lods, chr_null_lods = run_threshold_scans(
        null_outcomes,
        gc,
        phenotype=phenotype_vec,
        covariates=None,
        kinship_decomp=kinship_decomp,
        kinship_loco=kd_raw,
        chrom=perm_chrom,
        gene_pos_mb=perm_gene_pos_mb,
        window_mb=window_mb if perm_chrom is not None else None,
        permutation_method=perm_method,
        sample_ids=list(sample_ids),
    )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"max_lod": max_lods}).to_csv(output / "perm_max_lods.csv", index=False)
    for chrom, lods in chr_null_lods.items():
        pd.DataFrame({"max_lod": lods}).to_csv(
            output / f"perm_max_lods_chr{chrom}.csv", index=False
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
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X (NB/legacy mode).",
)
@click.option(
    "--phenotypes",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="counts_lognorm.h5 (LMM mode, preferred over --counts).",
)
@click.option(
    "--kinship",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Pre-built LOCO kinship HDF5 from KINSHIP_LOCO step (avoids recomputation).",
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
    counts: Optional[Path],
    phenotypes: Optional[Path],
    kinship: Optional[Path],
    metadata: Path,
    output: Path,
    method: str,
    cell_type: Optional[str],
    covariates: str,
    verbose: bool,
) -> None:
    """Estimate haplotype effects at each gene's peak locus.

    LMM mode (preferred): provide --phenotypes (counts_lognorm.h5) and optionally
    --kinship (pre-built LOCO HDF5). NB mode: provide --counts (AnnData h5ad).

    Writes allele_effects.csv with columns defined in pipeline.ALLELE_EFFECTS_BASE_COLS
    plus beta_{founder} and se_{founder} columns per founder.
    """
    _setup_logging(verbose)
    lmm_mode = phenotypes is not None
    if not lmm_mode and counts is None:
        raise click.ClickException("Provide --phenotypes (LMM mode) or --counts (NB mode).")

    import numpy as np
    import pandas as pd
    from .genotype.genomecache import GenomeCache
    from .genotype import kinship as kin_module
    from .qtl.effects import get_allele_effects

    gc = GenomeCache.from_hdf5(cache)
    peaks = pd.read_csv(scan_result)

    # ── kinship ────────────────────────────────────────────────────────────────
    if kinship is not None:
        kinship_decomp = _load_kinship_h5(kinship)
    else:
        kd_raw = kin_module.calc_kinship(gc, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # ── phenotype matrix ───────────────────────────────────────────────────────
    if lmm_mode:
        counts_mat, gene_ids, sample_ids = _load_phenotypes_h5(phenotypes)
        # mat is (n_genes, n_samples) — transpose to (n_samples, n_genes) for index lookup
        counts_mat = counts_mat.T
    else:
        import anndata
        import scipy.sparse as _sp

        adata = anndata.read_h5ad(counts, backed="r")
        if cell_type is not None:
            ct_col = _detect_ct_col(list(adata.obs.columns))
            if ct_col is None:
                raise click.ClickException(
                    f"No cell-type column found in {counts}.obs; "
                    'expected a column whose name contains both "cell" and "type".'
                )
            adata = adata[adata.obs[ct_col] == cell_type]
        raw_x = adata.raw.X if adata.raw is not None else adata.X
        if _sp.issparse(raw_x):
            raw_x = raw_x.toarray()
        counts_mat = np.asarray(raw_x, dtype=np.float64)  # (n_samples, n_genes)
        gene_ids = list(adata.raw.var.index if adata.raw is not None else adata.var.index)
        sample_ids = list(adata.obs.index)

    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr = _covariates_for_samples(metadata, sample_ids, _cov_cols)

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}
    for _, row in peaks.iterrows():
        gid = str(row["gene_id"])
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
                "gene_id": gid,
                "gene_symbol": str(row.get("gene_symbol", "")),
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


# ── characterise ──────────────────────────────────────────────────────────


@main.command("characterise")
@click.option(
    "--egenes",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="eqtl_peak.csv from `report` (ci_lo_bp/ci_hi_bp will be filled in).",
)
@click.option(
    "--phenotypes",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="counts_lognorm.h5 — LMM log-normalised expression.",
)
@click.option(
    "--kinship",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="kinship_loco.h5 from `build-kinship`.",
)
@click.option(
    "--cache",
    "gc_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="GenomeCache HDF5.",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Sample metadata CSV (sample_id column mandatory).",
)
@click.option(
    "--covariates",
    default="sex,dataset",
    show_default=True,
    help="Comma-separated covariate columns from metadata.",
)
@click.option(
    "--output-dir",
    default=".",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for output files.",
)
@click.option("-v", "--verbose", is_flag=True)
def characterise(
    egenes: Path,
    phenotypes: Path,
    kinship: Path,
    gc_path: Path,
    metadata: Path,
    covariates: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """LOD-drop CI for eGenes; fills ci_lo_bp/ci_hi_bp in eqtl_peak.csv.

    Reads eqtl_peak.csv (one row per eGene from `report`), computes 1.5-LOD
    support intervals by rescanning each eGene's chromosome with LMM, converts
    ci_lo/ci_hi from Mb to integer bp, and writes eqtl_peak_ci.csv alongside
    the original file.  eqtl_peak_ci.csv is the input expected by `finemap`.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd
    from .genotype.genomecache import GenomeCache
    from .qtl.ci import ci_scan_multi_gene

    egene_df = pd.read_csv(egenes)
    if egene_df.empty:
        logger.info("characterise: no eGenes — writing empty eqtl_peak_ci.csv")
        output_dir.mkdir(parents=True, exist_ok=True)
        _out_stem = egenes.stem.replace("eqtl_peak", "eqtl_peak_ci")
        egene_df.to_csv(output_dir / f"{_out_stem}.csv", index=False)
        return

    gene_ids = list(egene_df["gene_id"].astype(str))
    mat, found_ids, sample_ids = _load_phenotypes_h5(phenotypes, gene_ids)
    if not found_ids:
        raise click.ClickException("No eGene IDs found in phenotypes file.")

    logger.info("characterise: %d eGenes, %d found in h5", len(gene_ids), len(found_ids))

    kinship_decomp = _load_kinship_h5(kinship)
    gc = GenomeCache.from_hdf5(str(gc_path))

    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr = _covariates_for_samples(metadata, sample_ids, _cov_cols)

    # build per-gene lookup dicts
    row_by_id = egene_df.set_index("gene_id")
    expr_cache: dict = {}
    gene_chr_map: dict = {}
    peak_mb_map: dict = {}
    peak_lod_map: dict = {}
    for i, gid in enumerate(found_ids):
        expr_cache[gid] = mat[i].astype(float)
        row = row_by_id.loc[gid] if gid in row_by_id.index else None
        if row is not None:
            gene_chr_map[gid] = str(row.get("chr", ""))
            peak_mb_map[gid] = float(row.get("peak_mb", np.nan))
            peak_lod_map[gid] = float(row.get("lod", np.nan))

    output_dir.mkdir(parents=True, exist_ok=True)
    _out_stem = egenes.stem.replace("eqtl_peak", "eqtl_peak_ci")
    _ci_out = output_dir / "ci_raw.csv"

    ci_df = ci_scan_multi_gene(
        expr_cache,
        gene_chr_map,
        peak_mb_map,
        gc,
        peak_lod_map=peak_lod_map,
        covariates=covariates_arr,
        kinship_decomp=kinship_decomp,
        sample_ids=sample_ids,
        output_path=_ci_out,
    )
    del expr_cache

    if ci_df.empty:
        logger.warning("characterise: ci_scan_multi_gene returned no rows")
        egene_df.to_csv(output_dir / f"{_out_stem}.csv", index=False)
        return

    # convert Mb → integer bp and merge back into eqtl_peak
    ci_bp = ci_df[["gene_id", "ci_lo", "ci_hi"]].copy()
    ci_bp["ci_lo_bp"] = (ci_bp["ci_lo"] * 1e6).round().astype("Int64")
    ci_bp["ci_hi_bp"] = (ci_bp["ci_hi"] * 1e6).round().astype("Int64")
    ci_bp = ci_bp.drop(columns=["ci_lo", "ci_hi"])

    out = egene_df.copy()
    if "ci_lo_bp" in out.columns:
        out = out.drop(columns=["ci_lo_bp", "ci_hi_bp"], errors="ignore")
    out = out.merge(ci_bp, on="gene_id", how="left")

    out_path = output_dir / f"{_out_stem}.csv"
    out.to_csv(out_path, index=False)
    logger.info(
        "characterise: %d eGenes with CI written to %s", len(out), out_path
    )


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
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="AnnData H5AD file with raw counts in .raw.X (NB/legacy mode).",
)
@click.option(
    "--phenotypes",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="counts_lognorm.h5 (LMM mode, preferred over --counts).",
)
@click.option(
    "--kinship",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Pre-built LOCO kinship HDF5 (avoids recomputation).",
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
    counts: Optional[Path],
    phenotypes: Optional[Path],
    kinship: Optional[Path],
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

    LMM mode (preferred): provide --phenotypes (counts_lognorm.h5) and optionally
    --kinship. NB/legacy mode: provide --counts (AnnData h5ad).
    """
    _setup_logging(verbose)
    lmm_mode = phenotypes is not None
    if not lmm_mode and counts is None:
        raise click.ClickException("Provide --phenotypes (LMM mode) or --counts (NB mode).")

    import numpy as np
    import pandas as pd
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

    gc = GenomeCache.from_hdf5(cache)

    if kinship is not None:
        kinship_decomp = _load_kinship_h5(kinship)
    else:
        kd_raw = kin_module.calc_kinship(gc, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    if lmm_mode:
        counts_mat, gene_ids, sample_ids = _load_phenotypes_h5(phenotypes)
        # _load_phenotypes_h5 returns (n_genes, n_samples); finemap_egenes expects same
        pheno_matrix = counts_mat
    else:
        import anndata
        import scipy.sparse as _sp

        adata = anndata.read_h5ad(counts, backed="r")
        if cell_type is not None:
            ct_col = _detect_ct_col(list(adata.obs.columns))
            if ct_col is None:
                raise click.ClickException(
                    f"No cell-type column found in {counts}.obs; "
                    'expected a column whose name contains both "cell" and "type".'
                )
            adata = adata[adata.obs[ct_col] == cell_type]
        if counts_layer is not None:
            if counts_layer not in adata.layers:
                raise click.ClickException(
                    f"--counts-layer {counts_layer!r} not found in {counts}.layers."
                )
            raw_x = adata.layers[counts_layer]
            gene_ids = list(adata.var.index)
        else:
            if adata.raw is None:
                raise click.ClickException(
                    "adata.raw is None — pass --counts-layer or store raw counts in .raw"
                )
            raw_x = adata.raw.X
            gene_ids = list(adata.raw.var.index)
        import scipy.sparse as _sp2
        if _sp2.issparse(raw_x):
            raw_x = raw_x.toarray()
        pheno_matrix = np.asarray(raw_x, dtype=np.float64).T  # (n_genes, n_cells)
        sample_ids = list(adata.obs.index)

    _cov_cols = [c for c in covariates.split(",") if c.strip()]
    covariates_arr = _covariates_for_samples(metadata, sample_ids, _cov_cols)

    _tag = suffix or egenes.stem.removeprefix("eqtl_peak_")
    _finemap_egenes(
        egene_df=egene_df,
        phenotype_matrix=pheno_matrix,
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
