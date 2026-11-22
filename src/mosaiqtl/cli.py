"""mosaiQTL command-line interface.

Each command is one self-contained step of the haplotype eQTL workflow: it
reads its inputs from files and writes its results to files, so any step can
be rerun on its own and steps that do not depend on each other can run in
parallel (the per-chromosome scan in particular). Run `mosaiqtl --help` for
the full option listing.

Order of steps
--------------
Shared setup (once per genotype cache):
  build-kinship   LOCO kinship eigendecomposition → kinship_loco.h5
  inspect-cache   genotype-cache / kinship QC tables

Per cell type:
  scan-chr        replicate-aware LMM scan of one chromosome (run per chromosome)
  merge-scan      per-chromosome outputs → lod / allele_effects / peak_lod tables
  permute         permutation null → perm_max_lods*.csv + threshold.csv
  calibrate       per-gene calibrated permutation p-values → pval_perm.csv
  call-egenes     LOD gate + Storey q-values → eqtl_summary.csv
  characterise    1.5-LOD support intervals → ci + eqtl_peak_ci tables
  impute          imputed SNP association scan inside each interval (optional)
  finemap         ABF credible sets from the imputed scan (optional)
  report          submission tables eqtl_peak + eqtl_stats (+ fine-mapping)

Follow-up scans on eGenes: scan-mi, mediate, assess-stability.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import click

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from .genotype.genomecache import GenomeCache


logger = logging.getLogger(__name__)

_SCAN_TYPES = click.Choice(["L", "C", "G"])
_SCAN_TYPE_HELP = (
    "L = loci within ±window-mb of each gene's TSS | C = all loci on the gene's "
    "chromosome | G = every locus in the genome."
)


# ─────────────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool) -> None:
    """Configure root logger; DEBUG when verbose, INFO otherwise."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _covariates_for_samples(
    metadata_path: Path,
    sample_ids: "list[str]",
    cov_cols: "list[str]",
) -> "Optional[np.ndarray]":
    """Load metadata, restrict to scan samples, build covariate matrix.

    Returns (n_samples, n_covariates) float64 array aligned to sample_ids,
    or None when sample_id column is absent or any sample is missing.
    """
    import pandas as pd

    from .io.covariates import build_cov_array

    meta = pd.read_csv(metadata_path)
    if "sample_id" not in meta.columns:
        return None
    try:
        return build_cov_array(meta, sample_ids, cov_cols)
    except ValueError:
        return None


def _load_phenotypes_h5(
    h5_path: Path,
    gene_ids: "list[str] | None" = None,
) -> "tuple[np.ndarray, list[str], list[str]]":
    """Load phenotype matrix from counts_lognorm.h5.

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
        found = [g for g in gene_ids if g in gid_idx]
        if not found:
            return np.empty((0, len(sids))), found, sids
        idxs = np.array([gid_idx[g] for g in found])
        # h5py fancy indexing needs increasing indices; restore requested order after
        order = np.argsort(idxs)
        block = f["counts"][idxs[order], :].astype(np.float64)
        mat = np.empty_like(block)
        mat[order] = block
    return mat, found, sids


def _load_kinship_h5(h5_path: Path) -> "dict[str, dict[str, np.ndarray]]":
    """Load all-chromosome LOCO kinship eigendecomp → {chrom: {'values', 'vectors'}}."""
    import h5py
    import numpy as np

    decomp: dict = {}
    with h5py.File(h5_path, "r") as f:
        for key in f:
            if key.startswith("chr"):
                decomp[key[3:]] = {
                    "values": f[key]["values"][:].astype(np.float64),
                    "vectors": f[key]["vectors"][:].astype(np.float64),
                }
    return decomp


def _pick_median_h2_gene(
    h5_path: Path,
    gc: "GenomeCache",
    kinship_decomp: dict,
    n_sample: int = 200,
    seed: int = 1,
    geno_map: "dict[str, str] | None" = None,
) -> "tuple[str, float]":
    """Sample genes, estimate strain-level h² for each, return (gene_id, median_h²)."""
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

    geno_sids = [geno_map.get(s, s) for s in sids] if geno_map else sids
    z_mat, _ = _resolve_z(geno_sids, gc)
    n_per_strain = z_mat.sum(axis=0)
    ref_chr = natsorted(kinship_decomp.keys())[0]
    ev = kinship_decomp[ref_chr]["values"]
    evc = kinship_decomp[ref_chr]["vectors"]
    x_rot = evc.T @ np.ones((len(gc.sample_ids), 1))

    h2_pairs: list[tuple[str, float]] = []
    for i, gid in enumerate(sampled_gids):
        y_s = rint(np.where(n_per_strain > 0, (z_mat.T @ mat[i]) / n_per_strain, 0.0))
        if np.isnan(y_s).any():
            continue
        try:
            h2, _ = estimate_h2_null(evc.T @ y_s, x_rot, ev)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("h² estimation failed for %s: %s", gid, exc)
            continue
        h2_pairs.append((gid, float(h2)))

    if not h2_pairs:
        logger.warning("h² estimation failed for all sampled genes; using first gene")
        return all_gids[0], float("nan")

    h2_arr = np.array([h for _, h in h2_pairs])
    h2_med = float(np.median(h2_arr))
    rep_gene, rep_h2 = h2_pairs[int(np.argmin(np.abs(h2_arr - h2_med)))]
    logger.info("median h²=%.3f  representative gene: %s (h²=%.3f)", h2_med, rep_gene, rep_h2)
    return rep_gene, h2_med


def _median_h2_gene_from_effects(allele_effects_csv: Path) -> "tuple[str, str, float]":
    """Return (gene_id, gene_symbol, median_h²) for the gene whose h² is closest to the median.

    Uses the per-gene null-model h² already estimated during the scan, so the
    representative gene is drawn from every scanned gene rather than a sample.
    """
    import numpy as np
    import pandas as pd

    head = pd.read_csv(allele_effects_csv, nrows=0).columns
    cols = ["gene_id", "h2"] + (["gene_symbol"] if "gene_symbol" in head else [])
    eff = pd.read_csv(allele_effects_csv, usecols=cols, dtype={"gene_id": str})
    eff = eff.dropna(subset=["h2"]).reset_index(drop=True)
    if eff.empty:
        raise click.ClickException(
            f"{allele_effects_csv} has no finite h2 values; cannot pick a representative gene."
        )
    h2_med = float(eff["h2"].median())
    best = eff.iloc[int(np.argmin(np.abs(eff["h2"].to_numpy() - h2_med)))]
    rep_id = str(best["gene_id"])
    rep_symbol = str(best["gene_symbol"]) if "gene_symbol" in eff.columns else rep_id
    logger.info(
        "median h²=%.3f  representative gene: %s (h²=%.3f)",
        h2_med,
        rep_symbol,
        float(best["h2"]),
    )
    return rep_id, rep_symbol, h2_med


def _meta_geno_map(metadata_path: Path) -> "dict[str, str]":
    """Return {sample_id: genotype} from a sample metadata CSV (empty if no genotype column)."""
    import pandas as pd

    meta = pd.read_csv(metadata_path)
    if "sample_id" not in meta.columns or "genotype" not in meta.columns:
        return {}
    return dict(zip(meta["sample_id"].astype(str), meta["genotype"].astype(str)))


def _split_cols(spec: str) -> "list[str]":
    """Split a comma-separated column spec into a list, dropping empty entries."""
    return [c.strip() for c in spec.split(",") if c.strip()]


def _genome_wide_threshold(perm_dir: Path) -> Optional[float]:
    """Return the genome-wide LOD threshold for a `permute` output directory.

    Reads the genome_wide row of threshold.csv; when that file is absent, fits a
    95th-percentile GEV to perm_max_lods.csv. A threshold of exactly 0.0 means
    the null was all-zeros and is reported as None (invalid).
    """
    import pandas as pd

    from .qtl.thresholds import get_gev_thresholds

    thr_csv = perm_dir / "threshold.csv"
    thr: Optional[float] = None
    if thr_csv.exists():
        thr_df = pd.read_csv(thr_csv)
        gw = thr_df[thr_df["chr"].astype(str) == "genome_wide"]
        if not gw.empty and pd.notna(gw.iloc[0]["threshold"]):
            thr = float(gw.iloc[0]["threshold"])
    else:
        perm_csv = perm_dir / "perm_max_lods.csv"
        if not perm_csv.exists():
            raise click.ClickException(
                f"neither threshold.csv nor perm_max_lods.csv found in {perm_dir}"
            )
        thr, _ = get_gev_thresholds(pd.read_csv(perm_csv)["max_lod"].values, percentile=0.95)
    if thr is not None and thr == 0.0:
        return None
    return thr


def _load_cache(cache_path: Path) -> "GenomeCache":
    """Load a GenomeCache HDF5 with its strain axis in canonical order.

    Kinship eigenvectors are indexed by strain, so the cache that builds a
    kinship file and every cache that later consumes it must share one strain
    order; canonical order (founders, F1s, CC strains) is that shared order.
    """
    from .genotype.genomecache import GenomeCache

    return GenomeCache.from_hdf5(str(cache_path)).reorder_to_canonical()


def _required_covariates(
    metadata_path: Path,
    sample_ids: "list[str]",
    cov_cols: "list[str]",
) -> "Optional[np.ndarray]":
    """Covariate matrix for sample_ids; None only when no covariates were requested.

    Raises instead of silently dropping to an intercept-only model when a
    requested covariate cannot be built for every scanned sample.
    """
    if not cov_cols:
        return None
    cov = _covariates_for_samples(metadata_path, sample_ids, cov_cols)
    if cov is None:
        raise click.ClickException(
            f"could not build covariates {cov_cols} for all {len(sample_ids)} samples "
            f"from {metadata_path} (missing sample_id column or samples)."
        )
    return cov


def _tss_mb(row: "pd.Series") -> float:
    """Strand-aware TSS in Mb: end for '-' strand genes, start otherwise."""
    if "strand" in row.index and row["strand"] == "-" and "end" in row.index:
        return float(row["end"]) / 1e6
    return float(row["start"]) / 1e6


def _load_gene_tss(gene_tss_path: Path) -> "pd.DataFrame":
    """Transcript TSS table → DataFrame[gene_symbol, tss_mb], strand-aware, per transcript."""
    import numpy as np
    import pandas as pd

    tss = pd.read_csv(gene_tss_path, usecols=["gene_name", "start", "end", "strand"])
    tss["tss_mb"] = np.where(tss["strand"] == "-", tss["end"] / 1e6, tss["start"] / 1e6)
    return tss.rename(columns={"gene_name": "gene_symbol"})[["gene_symbol", "tss_mb"]]


def _write_scan_provenance(
    meta_df: "pd.DataFrame",
    output_dir: Path,
    celltype: Optional[str],
    scan_type: str,
    chrs_scanned: "list[str]",
    n_genes: Optional[int],
) -> None:
    """Write sample_breakdown.csv + scan_info.json describing the scanned sample set.

    Strains are listed founders → F1s/CC in canonical order, then any remaining
    strain names in natural order; dataset_breakdown is added only when the
    metadata spans more than one dataset.
    """
    import json
    from importlib.metadata import version as pkg_version

    import pandas as pd
    from natsort import natsorted

    from .genotype.cc import STRAIN_ORDER
    from .utils import git_hash

    req = {"sample_id", "genotype"}
    if not req.issubset(meta_df.columns):
        logger.warning(
            "metadata missing columns %s; scan_info.json/sample_breakdown.csv skipped",
            sorted(req - set(meta_df.columns)),
        )
        return

    counts = meta_df.groupby("genotype").size()
    rank = {s: i for i, s in enumerate(STRAIN_ORDER)}
    known = [s for s in STRAIN_ORDER if s in counts.index]
    other = natsorted([s for s in counts.index if s not in rank])
    strains = [(s, int(counts[s])) for s in known + other]
    multi_dataset = "dataset" in meta_df.columns and meta_df["dataset"].nunique() > 1

    rows: list[dict] = []
    for strain, n in strains:
        row: dict = {"strain": strain, "n_samples": n}
        if multi_dataset:
            ds = meta_df[meta_df["genotype"] == strain].groupby("dataset").size().to_dict()
            row["dataset_breakdown"] = ";".join(f"{d}:{k}" for d, k in sorted(ds.items()))
        rows.append(row)
    cols = ["strain", "n_samples"] + (["dataset_breakdown"] if multi_dataset else [])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(output_dir / "sample_breakdown.csv", index=False)

    info = {
        "mosaiqtl_version": pkg_version("mosaiqtl"),
        "git_hash": git_hash(),
        "run_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "celltype": celltype or "whole_tissue",
        "scan_type": scan_type,
        "chr": ",".join(chrs_scanned),
        "n_genes": n_genes,
        "samples": {
            "n_total": int(len(meta_df)),
            "n_strains": len(strains),
            "min_reps": min((n for _, n in strains), default=0),
            "max_reps": max((n for _, n in strains), default=0),
            "per_strain": {str(s): n for s, n in strains},
        },
    }
    (output_dir / "scan_info.json").write_text(json.dumps(info, indent=2))
    logger.info("scan_info.json + sample_breakdown.csv written (%d strains)", len(strains))


def _egene_context(
    summary_csv: Path,
) -> "tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, float]]":
    """Load eqtl_summary.csv → (eGenes, gene_meta, {gene_id: peak_mb}).

    gene_meta carries the display/diagnostic columns joined into every
    per-eGene follow-up scan's output, with the main scan's peak renamed to
    peak_lod_lmm / peak_mb_lmm so it is distinguishable from the follow-up
    scan's own peak.
    """
    import pandas as pd

    egenes = pd.read_csv(summary_csv, dtype={"chr": str, "gene_id": str})
    if egenes.empty:
        return egenes, None, {}
    gm_cols = [
        c
        for c in ["gene_id", "gene_symbol", "near_tss_mb", "cis_dist_mb", "peak_lod", "peak_mb"]
        if c in egenes.columns
    ]
    gene_meta = (
        egenes[gm_cols]
        .drop_duplicates("gene_id")
        .rename(columns={"peak_lod": "peak_lod_lmm", "peak_mb": "peak_mb_lmm"})
        .copy()
    )
    peak_mb_map = {
        str(r["gene_id"]): float(r["peak_mb_lmm"])
        for _, r in gene_meta.iterrows()
        if "peak_mb_lmm" in gene_meta.columns and pd.notna(r["peak_mb_lmm"])
    }
    return egenes, gene_meta, peak_mb_map


def _load_expression_for(
    phenotypes: Path,
    metadata: Path,
    gene_ids: "list[str]",
    cov_cols: "list[str]",
) -> "tuple[dict[str, np.ndarray], list[str], Optional[np.ndarray]]":
    """Load expression for gene_ids → ({gene_id: y}, genotype-level IDs, covariates).

    Covariates stay at sample level and genotype IDs are the strain each
    sample belongs to, which is what every replicate-aware scan expects.
    """
    mat, found, sids = _load_phenotypes_h5(phenotypes, gene_ids)
    expr = {gid: mat[i].astype(float) for i, gid in enumerate(found)}
    geno_map = _meta_geno_map(metadata)
    geno_ids = [geno_map.get(s, s) for s in sids]
    cov = _required_covariates(metadata, sids, cov_cols)
    return expr, geno_ids, cov


def _ci_bp(ci_csv: Path) -> "pd.DataFrame":
    """ci table (Mb bounds) → DataFrame[gene_id, ci_lo_bp, ci_hi_bp] in integer bp."""
    import pandas as pd

    ci = pd.read_csv(ci_csv, dtype={"gene_id": str})
    if ci.empty or not {"ci_lo", "ci_hi"}.issubset(ci.columns):
        return pd.DataFrame(columns=["gene_id", "ci_lo_bp", "ci_hi_bp"])
    ci = ci[["gene_id", "ci_lo", "ci_hi"]].copy()
    ci["ci_lo_bp"] = (ci["ci_lo"] * 1e6).round().astype("Int64")
    ci["ci_hi_bp"] = (ci["ci_hi"] * 1e6).round().astype("Int64")
    return ci[["gene_id", "ci_lo_bp", "ci_hi_bp"]]


def _gev_row(chrom: str, max_lods: "np.ndarray") -> dict:
    """eqtl_stats row for one permutation null: threshold, n_perms, GEV loc/scale/shape."""
    from .qtl.thresholds import get_gev_thresholds

    thr, params = get_gev_thresholds(max_lods)
    row: dict = {"chr": chrom, "lod_threshold": thr, "n_perms": len(max_lods)}
    if params:
        row["gev_mu"] = params.get("loc")
        row["gev_sigma"] = params.get("scale")
        row["gev_xi"] = params.get("c")
    return row


def _perm_chr_files(perm_dir: Path) -> "list[tuple[str, Path]]":
    """[(chrom, perm_max_lods_chr{c}.csv)] in chromosome order."""
    from .utils import chr_sort_key

    pairs = [(p.stem.replace("perm_max_lods_chr", ""), p)
             for p in perm_dir.glob("perm_max_lods_chr*.csv")]
    return sorted(pairs, key=lambda cp: chr_sort_key(cp[0]))


# ─────────────────────────────────────────────────────────────────────────────
# command group
# ─────────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="mosaiqtl")
def main() -> None:
    """mosaiQTL — haplotype-based QTL mapping for multiparent populations."""


# ── build-kinship ─────────────────────────────────────────────────────────


@main.command("build-kinship")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--output", required=True, type=click.Path(path_type=Path),
              help="Output HDF5 holding chr{c}/values and chr{c}/vectors per chromosome.")
@click.option("--verbose", is_flag=True, default=False)
def build_kinship(cache: Path, output: Path, verbose: bool) -> None:
    """Leave-one-chromosome-out kinship eigendecomposition, built once per run.

    Every downstream command that takes --kinship reads this file, so the
    kinship matrices are computed a single time and shared across cell types
    and chromosomes.
    """
    _setup_logging(verbose)
    import h5py
    import numpy as np

    from .genotype.kinship import calc_kinship, decomp_kinship

    gc = _load_cache(cache)
    decomp = decomp_kinship(calc_kinship(gc, method="loco"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        for chrom, d in decomp.items():
            grp = f.create_group(f"chr{chrom}")
            grp.create_dataset("values", data=np.asarray(d["values"], dtype=np.float64))
            grp.create_dataset("vectors", data=np.asarray(d["vectors"], dtype=np.float64))
    logger.info("build-kinship: %d chromosomes → %s", len(decomp), output)


# ── inspect-cache ─────────────────────────────────────────────────────────


@main.command("inspect-cache")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for cache_info.csv, haplotype_probs.csv, kinship_summary.csv.")
@click.option("--verbose", is_flag=True, default=False)
def inspect_cache(cache: Path, kinship: Path, output_dir: Path, verbose: bool) -> None:
    """Genotype-cache and kinship QC tables (independent of cell type and scan type).

    cache_info.csv lists loci per chromosome; haplotype_probs.csv exports the
    founder probabilities of every genotype at five evenly spaced loci on
    chr8 (or the first chromosome when chr8 is absent); kinship_summary.csv
    summarises each LOCO kinship matrix (diagonal / off-diagonal means and
    eigenvalue range).
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd

    gc = _load_cache(cache)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_loci = {c: len(gc.get_chromosome(c)[0]) for c in gc.chromosomes}
    pd.DataFrame([{"chr": c, "n_loci": n} for c, n in n_loci.items()]).to_csv(
        output_dir / "cache_info.csv", index=False
    )
    logger.info(
        "cache: %d chromosomes  total loci=%d  founders=%d  genotypes=%d",
        len(gc.chromosomes),
        sum(n_loci.values()),
        len(gc.founders),
        len(gc.sample_ids),
    )

    founder_cols = list(gc.founders)
    probe_chr = "8" if "8" in gc.chromosomes else gc.chromosomes[0]
    lids, _, probs = gc.get_chromosome(probe_chr)
    step = max(1, len(lids) // 5)
    hap_rows: list[dict] = []
    for li in list(range(0, len(lids), step))[:5]:
        for si, gtype in enumerate(gc.sample_ids):
            row: dict = {"chr": probe_chr, "locus": lids[li], "genotype": gtype}
            row.update(dict(zip(founder_cols, probs[li, si, :].tolist())))
            hap_rows.append(row)
    pd.DataFrame(hap_rows).to_csv(output_dir / "haplotype_probs.csv", index=False)
    del probs

    kin_rows: list[dict] = []
    for chrom, kd in _load_kinship_h5(kinship).items():
        k_mat = kd["vectors"] @ np.diag(kd["values"]) @ kd["vectors"].T
        n = k_mat.shape[0]
        kin_rows.append(
            {
                "chr": chrom,
                "n_genotypes": n,
                "mean_diag": float(np.diag(k_mat).mean()),
                "mean_offdiag": float(k_mat[np.tril_indices(n, -1)].mean()),
                "min_eig": float(kd["values"].min()),
                "max_eig": float(kd["values"].max()),
            }
        )
    pd.DataFrame(kin_rows).to_csv(output_dir / "kinship_summary.csv", index=False)
    logger.info("inspect-cache: tables written to %s", output_dir)


# ── scan-chr ──────────────────────────────────────────────────────────────


@main.command("scan-chr")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--phenotypes", required=True, type=click.Path(exists=True, path_type=Path),
              help="counts_lognorm.h5 — log-normalised expression (genes × samples).")
@click.option("--gene-info", required=True, type=click.Path(exists=True, path_type=Path),
              help="Gene-info CSV (gene_id, chr, start[, end, strand, gene_symbol]).")
@click.option("--metadata", required=True, type=click.Path(exists=True, path_type=Path),
              help="Sample metadata CSV with sample_id and genotype columns.")
@click.option("--chr", "chrom", required=True, help="Chromosome to scan (e.g. 1, X).")
@click.option(
    "--scan-type",
    default="L",
    show_default=True,
    type=_SCAN_TYPES,
    help="L = genes on this chromosome, loci within ±window-mb of the TSS | "
    "C = genes on this chromosome, all loci on it | "
    "G = every gene, all loci on this chromosome (one slice of a genome-wide scan).",
)
@click.option("--window-mb", default=1.0, show_default=True, type=float,
              help="Half-window in Mb around each gene's TSS for --scan-type L.")
@click.option("--n-genes", default=None, type=int,
              help="Scan only the first N selected genes (quick test runs).")
@click.option("--covariates", default="sex,dataset", show_default=True,
              help="Comma-separated metadata columns used as fixed-effect covariates.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for lod_chr{c}.csv and effects_chr{c}.csv.")
@click.option("--verbose", is_flag=True, default=False)
def scan_chr(
    cache: Path,
    kinship: Path,
    phenotypes: Path,
    gene_info: Path,
    metadata: Path,
    chrom: str,
    scan_type: str,
    window_mb: float,
    n_genes: Optional[int],
    covariates: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Replicate-aware LMM haplotype scan of one chromosome with its LOCO kinship.

    Writes the per-locus LOD profile of every scanned gene (lod_chr{c}.csv,
    LOD_COLS) and one peak row per gene with its null-model h², residual
    diagnostics and founder effects at the peak (effects_chr{c}.csv,
    ALLELE_EFFECTS_BASE_COLS + beta_{founder}). `merge-scan` combines the
    per-chromosome files into the genome-wide outputs.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd

    from .genotype.cc import display_founders
    from .pipeline import ALLELE_EFFECTS_BASE_COLS, LOD_COLS
    from .qtl.scan import scan_multi_gene_lmm
    from .utils import build_z_matrix

    chrom = str(chrom)
    output_dir.mkdir(parents=True, exist_ok=True)
    lod_out = output_dir / f"lod_chr{chrom}.csv"
    eff_out = output_dir / f"effects_chr{chrom}.csv"

    gc = _load_cache(cache)
    disp = display_founders(list(gc.founders))
    eff_cols = ALLELE_EFFECTS_BASE_COLS + [f"beta_{f}" for f in disp]

    def _write_empty(reason: str) -> None:
        logger.info("scan-chr chr%s: %s — writing header-only outputs", chrom, reason)
        pd.DataFrame(columns=LOD_COLS).to_csv(lod_out, index=False)
        pd.DataFrame(columns=eff_cols).to_csv(eff_out, index=False)

    gi = pd.read_csv(gene_info, dtype={"chr": str, "gene_id": str})
    ch_genes = gi if scan_type == "G" else gi[gi["chr"] == chrom]
    gene_ids = ch_genes["gene_id"].tolist()
    if n_genes is not None:
        gene_ids = gene_ids[:n_genes]
    if not gene_ids:
        _write_empty("no genes selected")
        return

    mat, found, sids = _load_phenotypes_h5(phenotypes, gene_ids)
    if not found:
        _write_empty("no selected genes present in --phenotypes")
        return

    geno_map = _meta_geno_map(metadata)
    geno_ids = [geno_map.get(s, s) for s in sids]
    cov = _required_covariates(metadata, sids, _split_cols(covariates))
    _, valid = build_z_matrix(geno_ids, list(gc.sample_ids))

    expr: dict = {}
    n_nan = 0
    for i, gid in enumerate(found):
        y = mat[i].astype(float)
        if np.isnan(y[valid]).any():
            n_nan += 1
            continue
        expr[gid] = y
    del mat
    if n_nan:
        logger.warning(
            "scan-chr chr%s: %d genes with missing expression values skipped", chrom, n_nan
        )
    if not expr:
        _write_empty("no genes with complete expression")
        return

    kd_all = _load_kinship_h5(kinship)
    if chrom not in kd_all:
        raise click.ClickException(f"chr{chrom} not present in {kinship}")
    kd = {chrom: kd_all[chrom]}
    del kd_all

    g_rows = ch_genes.drop_duplicates("gene_id").set_index("gene_id").loc[list(expr)]
    chr_map = {gid: str(c) for gid, c in g_rows["chr"].items()}
    tss_map: dict[str, float] = (
        {gid: _tss_mb(r) for gid, r in g_rows.iterrows()} if "start" in g_rows.columns else {}
    )
    symbols = (
        g_rows["gene_symbol"].astype(str)
        if "gene_symbol" in g_rows.columns
        else pd.Series(g_rows.index, index=g_rows.index)
    )
    gene_meta = pd.DataFrame({"gene_id": list(expr), "gene_symbol": symbols.to_list()})

    scan_window = {"C": "cis", "G": "genome", "L": "local"}[scan_type]
    if scan_window == "local" and not tss_map:
        logger.warning("scan-chr: --gene-info has no start column; scanning chr-wide (cis)")
        scan_window = "cis"

    raw_lod = output_dir / f".lod_chr{chrom}.raw.csv"
    logger.info(
        "scan-chr chr%s: %d genes  %d samples  (%s mode)", chrom, len(expr), len(sids), scan_type
    )
    peaks = scan_multi_gene_lmm(
        expr,
        chr_map,
        gc,
        covariates=cov,
        kinship_decomp=kd,
        sample_ids=geno_ids,
        use_rint=True,
        scan_window=scan_window,
        gene_pos_mb_map=tss_map if scan_window == "local" else None,
        window_mb=window_mb,
        lod_path=raw_lod,
        gene_meta=gene_meta,
        founder_names=disp,
    )
    del expr

    sym_map = dict(zip(gene_meta["gene_id"], gene_meta["gene_symbol"]))
    first = True
    if raw_lod.exists():
        for chunk in pd.read_csv(
            raw_lod, dtype={"chr": str, "gene_id": str}, chunksize=1_000_000
        ):
            chunk["gene_symbol"] = chunk["gene_id"].map(sym_map)
            chunk.reindex(columns=LOD_COLS).to_csv(
                lod_out, mode="w" if first else "a", header=first, index=False
            )
            first = False
        raw_lod.unlink()
    if first:
        pd.DataFrame(columns=LOD_COLS).to_csv(lod_out, index=False)

    if not peaks.empty:
        peaks["start_mb"] = peaks["gene_id"].map({g: round(t, 3) for g, t in tss_map.items()})
    peaks.reindex(columns=eff_cols).to_csv(eff_out, index=False)
    logger.info("scan-chr chr%s: %d gene peaks → %s", chrom, len(peaks), eff_out)


# ── merge-scan ────────────────────────────────────────────────────────────


@main.command("merge-scan")
@click.option("--scan-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory holding lod_chr*.csv / effects_chr*.csv from `scan-chr`.")
@click.option("--gene-tss", default=None, type=click.Path(exists=True, path_type=Path),
              help="Transcript TSS table (gene_name, start, end, strand); adds near_tss_mb.")
@click.option("--metadata", required=True, type=click.Path(exists=True, path_type=Path),
              help="Sample metadata CSV (sample_id, genotype[, dataset]) for scan provenance.")
@click.option("--celltype", default=None, help="Cell-type label recorded in scan_info.json.")
@click.option("--scan-type", default="L", show_default=True, type=_SCAN_TYPES,
              help=_SCAN_TYPE_HELP)
@click.option("--chrs", default="all", show_default=True,
              help="Chromosomes requested for the scan, recorded in scan_info.json.")
@click.option("--n-genes", default=None, type=int,
              help="Gene limit used for the scan, recorded in scan_info.json.")
@click.option("--suffix", required=True,
              help="Run suffix for output names, e.g. glutamatergic_neuron_Lw1.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for lod_/allele_effects_/peak_lod_{suffix}.csv + provenance.")
@click.option("--verbose", is_flag=True, default=False)
def merge_scan(
    scan_dir: Path,
    gene_tss: Optional[Path],
    metadata: Path,
    celltype: Optional[str],
    scan_type: str,
    chrs: str,
    n_genes: Optional[int],
    suffix: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Combine per-chromosome scan outputs into genome-wide result tables.

    lod_{suffix}.csv concatenates every chromosome's LOD profile (streamed,
    never held in memory). allele_effects_{suffix}.csv keeps each gene's
    highest-LOD peak row; peak_lod_{suffix}.csv adds near_tss_mb (transcript
    TSS nearest the peak) and cis_dist_mb (peak − near_tss_mb, falling back
    to the gene's own TSS when no transcript TSS is known).
    """
    _setup_logging(verbose)
    import pandas as pd

    from .pipeline import LOD_COLS, PEAK_LOD_COLS
    from .utils import chr_sort_key, compute_near_tss_mb

    def _chr_of(p: Path, prefix: str) -> str:
        return p.stem[len(prefix):]

    lod_files = sorted(scan_dir.glob("lod_chr*.csv"),
                       key=lambda p: chr_sort_key(_chr_of(p, "lod_chr")))
    eff_files = sorted(scan_dir.glob("effects_chr*.csv"),
                       key=lambda p: chr_sort_key(_chr_of(p, "effects_chr")))
    if not eff_files:
        raise click.ClickException(f"no effects_chr*.csv files in {scan_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    lod_out = output_dir / f"lod_{suffix}.csv"
    first = True
    for f in lod_files:
        for chunk in pd.read_csv(f, dtype={"chr": str, "gene_id": str}, chunksize=1_000_000):
            chunk.reindex(columns=LOD_COLS).to_csv(
                lod_out, mode="w" if first else "a", header=first, index=False
            )
            first = False
    if first:
        pd.DataFrame(columns=LOD_COLS).to_csv(lod_out, index=False)

    eff = pd.concat(
        [pd.read_csv(f, dtype={"chr": str, "gene_id": str}) for f in eff_files],
        ignore_index=True,
    )
    eff = (
        eff.sort_values("peak_lod", ascending=False, kind="stable")
        .drop_duplicates("gene_id")
        .sort_index()
        .reset_index(drop=True)
    )
    eff.to_csv(output_dir / f"allele_effects_{suffix}.csv", index=False)

    peak = eff[[c for c in eff.columns if not c.startswith("beta_")]].copy()
    if gene_tss is not None and not peak.empty:
        peak["near_tss_mb"] = compute_near_tss_mb(peak, _load_gene_tss(gene_tss))
        peak["cis_dist_mb"] = (peak["peak_mb"] - peak["near_tss_mb"]).round(6)
        miss = peak["near_tss_mb"].isna()
        peak.loc[miss, "cis_dist_mb"] = (
            peak.loc[miss, "peak_mb"] - peak.loc[miss, "start_mb"]
        ).round(6)
    else:
        peak["cis_dist_mb"] = (peak["peak_mb"] - peak["start_mb"]).round(6)
    peak.reindex(columns=PEAK_LOD_COLS).to_csv(output_dir / f"peak_lod_{suffix}.csv", index=False)

    chrs_scanned = (
        [_chr_of(f, "effects_chr") for f in eff_files] if chrs == "all" else _split_cols(chrs)
    )
    _write_scan_provenance(
        pd.read_csv(metadata), output_dir, celltype, scan_type, chrs_scanned, n_genes
    )
    logger.info(
        "merge-scan: %d chromosomes  %d genes → %s", len(eff_files), len(eff), output_dir
    )


# ── permute ───────────────────────────────────────────────────────────────


@main.command("permute")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--phenotypes", required=True, type=click.Path(exists=True, path_type=Path),
              help="counts_lognorm.h5 — log-normalised expression.")
@click.option("--metadata", required=True, type=click.Path(exists=True, path_type=Path),
              help="Sample metadata CSV with sample_id and genotype columns.")
@click.option("--allele-effects", default=None, type=click.Path(exists=True, path_type=Path),
              help="allele_effects_{suffix}.csv from `merge-scan`. When given and --gene "
              "is omitted, the representative gene is the scanned gene whose null-model "
              "h² is closest to the median h² of the whole scan.")
@click.option("--gene", default=None,
              help="Representative gene_id (overrides the median-h² choice).")
@click.option("--gene-info", default=None, type=click.Path(exists=True, path_type=Path),
              help="Gene-info CSV (gene_id, chr, start[, end, strand]); required for "
              "--scan-type L.")
@click.option("--scan-type", default="L", show_default=True, type=_SCAN_TYPES,
              help="L restricts the representative gene's null scan to ±window-mb of its "
              "own TSS, matching a real scan that never leaves that window; C and G draw "
              "the null from every locus.")
@click.option("--window-mb", default=1.0, show_default=True, type=float,
              help="Half-window in Mb around the representative gene's TSS (L only).")
@click.option("--n-perms", default=1000, show_default=True, type=int,
              help="Number of permutations for the GEV fit.")
@click.option("--perm-method", default="parametric", show_default=True,
              type=click.Choice(["parametric", "plain"]),
              help="'parametric' scans each permutation at the real phenotype's h²; "
              "'plain' is an unconditional shuffle.")
@click.option("--seed", default=1, show_default=True, type=int,
              help="Seed for the phenotype permutations.")
@click.option("--output", required=True, type=click.Path(path_type=Path),
              help="Output directory; writes perm_max_lods*.csv + threshold.csv.")
@click.option("--verbose", is_flag=True, default=False)
def permute(
    cache: Path,
    kinship: Path,
    phenotypes: Path,
    metadata: Path,
    allele_effects: Optional[Path],
    gene: Optional[str],
    gene_info: Optional[Path],
    scan_type: str,
    window_mb: float,
    n_perms: int,
    perm_method: str,
    seed: int,
    output: Path,
    verbose: bool,
) -> None:
    """Permutation null of the genome-wide max LOD → GEV threshold.

    One representative gene (median h²) is permuted; perm_max_lods.csv holds
    the genome-wide max-LOD null, perm_max_lods_chr{c}.csv each chromosome's
    null, and threshold.csv the GEV 95th-percentile thresholds (genome_wide
    row drives eGene calling).
    """
    _setup_logging(verbose)
    if scan_type == "L" and gene_info is None:
        raise click.ClickException("--scan-type L requires --gene-info.")

    import numpy as np
    import pandas as pd

    from .qtl.permutation import generate_null_outcomes, run_threshold_scans
    from .qtl.thresholds import get_gev_thresholds

    gc = _load_cache(cache)
    kinship_decomp = _load_kinship_h5(kinship)
    geno_map = _meta_geno_map(metadata)

    rep_label: Optional[str] = None
    if gene is not None:
        rep_gene, h2_median = gene, float("nan")
    elif allele_effects is not None:
        rep_gene, rep_label, h2_median = _median_h2_gene_from_effects(allele_effects)
    else:
        rep_gene, h2_median = _pick_median_h2_gene(
            phenotypes, gc, kinship_decomp, geno_map=geno_map
        )
    mat, found, sids = _load_phenotypes_h5(phenotypes, [rep_gene])
    if not found:
        raise click.ClickException(f"representative gene {rep_gene!r} not in {phenotypes}.")
    y = mat[0].astype(float)

    perm_chrom: Optional[str] = None
    perm_gene_pos_mb: Optional[float] = None
    if scan_type == "L" and gene_info is not None:
        gi = pd.read_csv(gene_info, dtype={"chr": str, "gene_id": str})
        row = gi[gi["gene_id"] == str(rep_gene)]
        if row.empty:
            logger.warning(
                "permute: %s not in --gene-info — falling back to genome-wide null", rep_gene
            )
        else:
            perm_chrom = str(row.iloc[0]["chr"])
            perm_gene_pos_mb = _tss_mb(row.iloc[0])
            logger.info(
                "permute: local-window null chr%s TSS=%.3fMb ±%.1fMb",
                perm_chrom,
                perm_gene_pos_mb,
                window_mb,
            )

    null_outcomes = generate_null_outcomes(y, n_perms=n_perms, seed=seed)
    max_lods, chr_null_lods = run_threshold_scans(
        null_outcomes,
        gc,
        phenotype=y,
        covariates=None,
        kinship_decomp=kinship_decomp,
        chrom=perm_chrom,
        permutation_method=perm_method,
        sample_ids=[geno_map.get(s, s) for s in sids],
        gene_pos_mb=perm_gene_pos_mb,
        window_mb=window_mb if perm_chrom is not None else None,
    )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"max_lod": max_lods}).to_csv(output / "perm_max_lods.csv", index=False)
    for chrom, lods in chr_null_lods.items():
        pd.DataFrame({"max_lod": lods}).to_csv(
            output / f"perm_max_lods_chr{chrom}.csv", index=False
        )
    thr, _ = get_gev_thresholds(max_lods, percentile=0.95)
    if thr is not None and thr == 0.0:
        logger.error(
            "permutation null is all-zeros (kinship singular on every chromosome); "
            "threshold=0.0 is invalid"
        )
        thr = None
    n_valid = int(np.isfinite(max_lods).sum())
    logger.info("permute: genome-wide GEV threshold=%s (n_perms=%d)",
                f"{thr:.3f}" if thr is not None else "n/a", n_valid)

    perm_gene_label = rep_label if rep_label is not None else rep_gene
    thr_rows: list[dict] = [
        {"chr": "genome_wide", "threshold": thr, "perm_gene": perm_gene_label,
         "h2_median": h2_median, "n_perms": n_valid}
    ]
    for chrom, lods in chr_null_lods.items():
        chr_thr, _ = get_gev_thresholds(lods, percentile=0.95)
        thr_rows.append(
            {"chr": str(chrom), "threshold": chr_thr, "perm_gene": perm_gene_label,
             "h2_median": None, "n_perms": int(np.isfinite(lods).sum())}
        )
    pd.DataFrame(thr_rows).to_csv(output / "threshold.csv", index=False)
    logger.info("permute: results written to %s", output)


# ── calibrate ─────────────────────────────────────────────────────────────


@main.command("calibrate")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--peak-lod", required=True, type=click.Path(exists=True, path_type=Path),
              help="peak_lod_{suffix}.csv from `merge-scan`.")
@click.option("--perm-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory written by `permute` (threshold.csv).")
@click.option("--phenotypes", required=True, type=click.Path(exists=True, path_type=Path),
              help="counts_lognorm.h5 — log-normalised expression.")
@click.option("--metadata", required=True, type=click.Path(exists=True, path_type=Path),
              help="Sample metadata CSV with sample_id and genotype columns.")
@click.option("--covariates", default="sex,dataset", show_default=True,
              help="Comma-separated metadata columns used as fixed-effect covariates.")
@click.option("--n-perms", default=200, show_default=True, type=int,
              help="Permutations per gene.")
@click.option("--scan-type", default="L", show_default=True, type=_SCAN_TYPES,
              help=_SCAN_TYPE_HELP)
@click.option("--window-mb", default=1.0, show_default=True, type=float,
              help="Half-window in Mb for --scan-type L.")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path),
              help="Output pval_perm.csv path.")
@click.option("--verbose", is_flag=True, default=False)
def calibrate(
    cache: Path,
    kinship: Path,
    peak_lod: Path,
    perm_dir: Path,
    phenotypes: Path,
    metadata: Path,
    covariates: str,
    n_perms: int,
    scan_type: str,
    window_mb: float,
    output_path: Path,
    verbose: bool,
) -> None:
    """Per-gene Beta-calibrated permutation p-values for Storey q-value FDR.

    Genes whose peak LOD strictly exceeds the genome-wide permutation threshold
    each get a parametric null at their own h², computed on strain means. In
    local mode the null is restricted to ±window_mb around the gene's peak,
    matching the window the real scan tested.
    """
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd

    from .genotype.kinship import calc_kinship
    from .qtl.fdr import calibrate_gene_pvalue
    from .utils import build_z_matrix

    output_path.parent.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(columns=["gene_id", "pval_perm"])

    peaks = pd.read_csv(peak_lod, dtype={"chr": str, "gene_id": str})
    gw_thr = _genome_wide_threshold(perm_dir)
    if gw_thr is None:
        logger.warning("calibrate: no valid genome-wide threshold — writing empty pval_perm")
        empty.to_csv(output_path, index=False)
        return
    egenes = peaks[peaks["peak_lod"] > gw_thr]
    if egenes.empty:
        logger.warning("calibrate: no genes passed the LOD gate — writing empty pval_perm")
        empty.to_csv(output_path, index=False)
        return
    logger.info("calibrate: threshold=%.3f  %d genes to calibrate", gw_thr, len(egenes))

    gc = _load_cache(cache)
    kinship_decomp = _load_kinship_h5(kinship)
    kinship_loco = calc_kinship(gc, method="loco")
    geno_map = _meta_geno_map(metadata)
    cov_cols = _split_cols(covariates)

    rows: list[dict] = []
    for _, eg in egenes.iterrows():
        gid = str(eg["gene_id"])
        mat, found, sids = _load_phenotypes_h5(phenotypes, [gid])
        if not found:
            logger.warning("calibrate: %s not in phenotypes — skipped", gid)
            continue
        geno_ids = [geno_map.get(s, s) for s in sids]
        z_mat, valid = build_z_matrix(geno_ids, list(gc.sample_ids))
        z_sub = z_mat[valid]
        r = z_sub.sum(axis=0)
        y_g = (z_sub.T @ mat[0][valid]) / r
        cov = _required_covariates(metadata, sids, cov_cols)
        cov_g = (z_sub.T @ cov[valid]) / r[:, np.newaxis] if cov is not None else None
        pos_mb = float(eg["peak_mb"]) if pd.notna(eg.get("peak_mb")) else None
        p_adj, _, _, _ = calibrate_gene_pvalue(
            y_gene=y_g,
            genomecache=gc,
            chrom=str(eg["chr"]),
            peak_lod=float(eg["peak_lod"]),
            covariates=cov_g,
            kinship_decomp=kinship_decomp,
            kinship_loco=kinship_loco,
            n_perms=n_perms,
            gene_pos_mb=pos_mb,
            window_mb=window_mb if scan_type == "L" else None,
            use_rint=True,
        )
        rows.append({"gene_id": gid, "pval_perm": float(p_adj)})

    pd.DataFrame(rows, columns=["gene_id", "pval_perm"]).to_csv(output_path, index=False)
    logger.info("calibrate: %d genes → %s", len(rows), output_path)


# ── call-egenes ───────────────────────────────────────────────────────────


@main.command("call-egenes")
@click.option("--peak-lod", required=True, type=click.Path(exists=True, path_type=Path),
              help="peak_lod_{suffix}.csv from `merge-scan`.")
@click.option("--perm-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory written by `permute` (threshold.csv + perm_max_lods.csv).")
@click.option("--allele-effects", default=None, type=click.Path(exists=True, path_type=Path),
              help="allele_effects_{suffix}.csv; fills positional columns absent from "
              "--peak-lod.")
@click.option("--pval-perm", default=None, type=click.Path(exists=True, path_type=Path),
              help="pval_perm.csv from `calibrate`; adds Storey q-values and significance.")
@click.option("--fdr", default=0.05, show_default=True, type=float,
              help="FDR threshold for significance='T'.")
@click.option("--call-rule", default="qvalue", show_default=True,
              type=click.Choice(["qvalue", "bh", "lfdr"]),
              help="Value thresholded by --fdr: Storey q-value, Benjamini-Hochberg "
                   "or local false discovery rate, over all tested genes.")
@click.option("--output", required=True, type=click.Path(path_type=Path),
              help="Output eqtl_summary CSV path.")
@click.option("--verbose", is_flag=True, default=False)
def call_egenes_cmd(
    peak_lod: Path,
    perm_dir: Path,
    allele_effects: Optional[Path],
    pval_perm: Optional[Path],
    fdr: float,
    call_rule: str,
    output: Path,
    verbose: bool,
) -> None:
    """Two-stage eGene calling → eqtl_summary (EGENE_SUMMARY_COLS).

    Stage 1 keeps genes whose peak LOD strictly exceeds the genome-wide
    permutation threshold; Stage 2 converts each kept gene's own calibrated
    permutation p-value into a Storey q-value across all kept genes.
    """
    _setup_logging(verbose)
    import pandas as pd

    from .pipeline import (
        EGENE_SUMMARY_COLS,
        calibrate_egenes_storey,
        call_egenes,
        finalize_egene_summary,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    peaks = pd.read_csv(peak_lod, dtype={"chr": str, "gene_id": str})
    thr = _genome_wide_threshold(perm_dir)
    if thr is None or peaks.empty:
        logger.warning("call-egenes: no valid threshold or no peaks — writing empty summary")
        pd.DataFrame(columns=EGENE_SUMMARY_COLS).to_csv(output, index=False)
        return

    eff: Optional[pd.DataFrame] = None
    if allele_effects is not None:
        eff = pd.read_csv(
            allele_effects,
            usecols=["gene_id", "gene_symbol", "chr", "peak_locus", "peak_mb", "peak_lod", "h2"],
            dtype={"chr": str, "gene_id": str},
        )
    frames = [
        call_egenes(
            ch_df[ch_df["peak_lod"] > thr],
            thr,
            effects_df=eff[eff["chr"] == ch] if eff is not None else None,
        )
        for ch, ch_df in peaks.groupby(peaks["chr"].astype(str))
    ]
    egenes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info(
        "call-egenes: genome-wide threshold=%.3f → %d / %d genes", thr, len(egenes), len(peaks)
    )
    if egenes.empty:
        pd.DataFrame(columns=EGENE_SUMMARY_COLS).to_csv(output, index=False)
        return

    if pval_perm is not None:
        pv = pd.read_csv(pval_perm, dtype={"gene_id": str}).dropna(subset=["pval_perm"])
        if not pv.empty:
            egenes = calibrate_egenes_storey(
                egenes, dict(zip(pv["gene_id"], pv["pval_perm"])),
                fdr_threshold=fdr, call_rule=call_rule,
            )
            logger.info(
                "call-egenes: significant(%s<=%.2f)=%d / %d",
                call_rule,
                fdr,
                int((egenes["significance"] == "T").sum()),
                len(egenes),
            )

    perm_csv = perm_dir / "perm_max_lods.csv"
    if perm_csv.exists():
        egenes["perm_n"] = len(pd.read_csv(perm_csv))
        egenes["perm_chr"] = str(peaks.loc[peaks["peak_lod"].idxmax(), "chr"])

    finalize_egene_summary(egenes).to_csv(output, index=False)
    logger.info("call-egenes: %d eGenes → %s", len(egenes), output)


# ── characterise ──────────────────────────────────────────────────────────


@main.command("characterise")
@click.option("--egenes", required=True, type=click.Path(exists=True, path_type=Path),
              help="eqtl_summary CSV from `call-egenes`.")
@click.option("--phenotypes", required=True, type=click.Path(exists=True, path_type=Path),
              help="counts_lognorm.h5 — log-normalised expression (genes × samples).")
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--metadata", required=True, type=click.Path(exists=True, path_type=Path),
              help="Sample metadata CSV with sample_id and genotype columns.")
@click.option("--covariates", default="sex,dataset", show_default=True,
              help="Comma-separated metadata columns used as fixed-effect covariates.")
@click.option("--suffix", required=True, help="Run suffix for output names.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for ci_{suffix}.csv and eqtl_peak_ci_{suffix}.csv.")
@click.option("--verbose", is_flag=True, default=False)
def characterise(
    egenes: Path,
    phenotypes: Path,
    kinship: Path,
    cache: Path,
    metadata: Path,
    covariates: str,
    suffix: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """1.5-LOD support interval of every eGene → ci + eqtl_peak_ci tables.

    Each eGene's chromosome is rescanned with its LOCO kinship and the
    support interval is taken around its main-scan peak. ci_{suffix}.csv
    holds the interval in Mb (CI_COLS); eqtl_peak_ci_{suffix}.csv is one row
    per eGene with the peak position and the interval in bp, the input to
    `impute` and `report`.
    """
    _setup_logging(verbose)
    import pandas as pd

    from .pipeline import CI_COLS
    from .qtl.ci import ci_scan_multi_gene

    output_dir.mkdir(parents=True, exist_ok=True)
    ci_out = output_dir / f"ci_{suffix}.csv"
    peak_out = output_dir / f"eqtl_peak_ci_{suffix}.csv"
    pre_cols = ["gene_id", "gene_symbol", "chr", "peak_mb", "peak_lod", "peak_locus"]
    peak_cols = pre_cols + ["pos", "lod", "variant", "ci_lo_bp", "ci_hi_bp"]

    eg, gene_meta, peak_mb_map = _egene_context(egenes)
    if eg.empty:
        logger.info("characterise: no eGenes — writing header-only outputs")
        pd.DataFrame(columns=CI_COLS).to_csv(ci_out, index=False)
        pd.DataFrame(columns=peak_cols).to_csv(peak_out, index=False)
        return

    expr, geno_ids, cov = _load_expression_for(
        phenotypes, metadata, eg["gene_id"].tolist(), _split_cols(covariates)
    )
    chr_map = {str(r["gene_id"]): str(r["chr"]) for _, r in eg.iterrows()}
    logger.info("characterise: %d eGenes (%d with expression)", len(eg), len(expr))

    ci_df = ci_scan_multi_gene(
        expr,
        {g: chr_map[g] for g in expr},
        {g: peak_mb_map[g] for g in expr if g in peak_mb_map},
        _load_cache(cache),
        covariates=cov,
        kinship_decomp=_load_kinship_h5(kinship),
        sample_ids=geno_ids,
        output_path=ci_out,
        gene_meta=gene_meta,
    )
    del expr
    if ci_df.empty:
        logger.warning("characterise: no support intervals computed")
        pd.DataFrame(columns=CI_COLS).to_csv(ci_out, index=False)
    else:
        logger.info(
            "characterise: %d intervals  median width=%.2f Mb",
            len(ci_df),
            float(ci_df["ci_width_mb"].median()),
        )

    pre = eg.reindex(columns=pre_cols).copy()
    pre["pos"] = (pre["peak_mb"] * 1e6).round().astype("Int64")
    pre["lod"] = pre["peak_lod"]
    pre["variant"] = pre["peak_locus"]
    pre = pre.merge(_ci_bp(ci_out), on="gene_id", how="left")
    pre[peak_cols].to_csv(peak_out, index=False)
    logger.info("characterise: %s written (%d eGenes)", peak_out.name, len(pre))


# ── impute ────────────────────────────────────────────────────────────────


@main.command("impute")
@click.option("--egenes", required=True, type=click.Path(exists=True, path_type=Path),
              help="eqtl_peak_ci_{suffix}.csv from `characterise`.")
@click.option("--phenotypes", required=True, type=click.Path(exists=True, path_type=Path),
              help="counts_lognorm.h5 — log-normalised expression.")
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 file.")
@click.option("--metadata", required=True, type=click.Path(exists=True, path_type=Path),
              help="Sample metadata CSV with sample_id and genotype columns.")
@click.option("--alleles", required=True, type=click.Path(exists=True, path_type=Path),
              help="Founder allele HDF5 (build_founder_allele_mgp / build_founder_allele_gm).")
@click.option("--covariates", default="sex,dataset", show_default=True,
              help="Comma-separated metadata columns used as fixed-effect covariates.")
@click.option("--ref-founder", default="B6J", show_default=True,
              help="Founder carrying the reference allele.")
@click.option("--rint/--no-rint", "use_rint", required=True,
              help="Rank-inverse normal transform of each phenotype; must match the haplotype scan.")
@click.option("--suffix", required=True, help="Run suffix for output names.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for imputed_snps_{suffix}.csv.")
@click.option("--verbose", is_flag=True, default=False)
def impute(
    egenes: Path,
    phenotypes: Path,
    kinship: Path,
    cache: Path,
    metadata: Path,
    alleles: Path,
    covariates: str,
    ref_founder: str,
    use_rint: bool,
    suffix: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Imputed SNP association scan inside each eGene's support interval.

    Founder alleles at every SNP in the interval are combined with the
    haplotype probabilities into per-strain allele dosages, and each SNP is
    tested with the eGene's LOCO kinship. eGenes without an interval are
    skipped. Output: imputed_snps_{suffix}.csv (IMPUTED_SNP_COLS), the input
    to `finemap`.
    """
    _setup_logging(verbose)
    import pandas as pd

    from .pipeline import IMPUTED_SNP_COLS, impute_egenes

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"imputed_snps_{suffix}.csv"

    eg = pd.read_csv(egenes, dtype={"chr": str, "gene_id": str})
    required = {"gene_id", "chr", "pos", "lod", "variant", "ci_lo_bp", "ci_hi_bp"}
    missing = required - set(eg.columns)
    if missing:
        raise click.ClickException(
            f"--egenes missing columns {sorted(missing)}; pass eqtl_peak_ci from `characterise`."
        )
    if eg.empty:
        logger.info("impute: no eGenes — writing header-only output")
        pd.DataFrame(columns=IMPUTED_SNP_COLS).to_csv(out, index=False)
        return

    mat, found, sids = _load_phenotypes_h5(phenotypes, eg["gene_id"].tolist())
    geno_map = _meta_geno_map(metadata)
    imputed = impute_egenes(
        egene_df=eg,
        phenotype_matrix=mat,
        gene_ids=found,
        genomecache=_load_cache(cache),
        alleles_h5=alleles,
        covariates=_required_covariates(metadata, sids, _split_cols(covariates)),
        kinship_decomp=_load_kinship_h5(kinship),
        sample_ids=[geno_map.get(s, s) for s in sids],
        ref_founder=ref_founder,
        use_rint=use_rint,
    )
    imputed.to_csv(out, index=False)
    logger.info(
        "impute: %d SNPs across %d eGenes → %s",
        len(imputed),
        imputed["gene_id"].nunique() if not imputed.empty else 0,
        out,
    )


# ── finemap ───────────────────────────────────────────────────────────────


@main.command("finemap")
@click.option("--imputed", required=True, type=click.Path(exists=True, path_type=Path),
              help="imputed_snps_{suffix}.csv from `impute`.")
@click.option("--prior-var", default=0.04, show_default=True, type=float,
              help="Prior variance of the standardised SNP effect (ABF).")
@click.option("--coverage", default=0.95, show_default=True, type=float,
              help="Credible-set coverage.")
@click.option("--suffix", required=True, help="Run suffix for output names.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for eqtl_finemapping[_peak]_{suffix}.csv.")
@click.option("--verbose", is_flag=True, default=False)
def finemap(
    imputed: Path,
    prior_var: float,
    coverage: float,
    suffix: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Approximate-Bayes-factor credible sets from the imputed SNP scan.

    Writes eqtl_finemapping_{suffix}.csv (every credible-set SNP with its
    posterior inclusion probability) and eqtl_finemapping_peak_{suffix}.csv
    (the lead SNP of each eGene).
    """
    _setup_logging(verbose)
    import pandas as pd

    from .pipeline import finemap_imputed

    imp = pd.read_csv(imputed, dtype={"chr": str, "gene_id": str})
    fm, fp = finemap_imputed(
        imp, output_dir, prior_var=prior_var, coverage=coverage, tag=suffix
    )
    logger.info("finemap: %d credible-set SNPs across %d eGenes", len(fm), len(fp))


# ── report ────────────────────────────────────────────────────────────────


@main.command("report")
@click.option("--egenes", required=True, type=click.Path(exists=True, path_type=Path),
              help="eqtl_summary CSV from `call-egenes`.")
@click.option("--allele-effects", required=True, type=click.Path(exists=True, path_type=Path),
              help="allele_effects_{suffix}.csv from `merge-scan` (founder betas at peaks).")
@click.option("--ci", required=True, type=click.Path(exists=True, path_type=Path),
              help="eqtl_peak_ci_{suffix}.csv from `characterise` (ci_lo_bp / ci_hi_bp).")
@click.option("--finemapping", default=None, type=click.Path(exists=True, path_type=Path),
              help="eqtl_finemapping_{suffix}.csv from `finemap`; schema-checked and "
              "published with the submission tables.")
@click.option("--pval-perm", default=None, type=click.Path(exists=True, path_type=Path),
              help="pval_perm.csv from `calibrate`; recomputes Storey q-values when the "
              "eGene table does not already carry them.")
@click.option("--perm-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory written by `permute` (perm_max_lods.csv + per-chr files).")
@click.option("--cache", required=True, type=click.Path(exists=True, path_type=Path),
              help="GenomeCache HDF5 (founder names for the effect columns).")
@click.option("--peak-lod", default=None, type=click.Path(exists=True, path_type=Path),
              help="peak_lod_{suffix}.csv; when given, also draws genome_summary_{suffix}.png.")
@click.option("--scan-type", default="L", show_default=True, type=_SCAN_TYPES,
              help=_SCAN_TYPE_HELP)
@click.option("--window-mb", default=1.0, show_default=True, type=float,
              help="Local scan half-window, recorded in eqtl_stats when --scan-type L.")
@click.option("--fdr", default=0.05, show_default=True, type=float,
              help="FDR threshold recorded in eqtl_stats and applied to the --call-rule value.")
@click.option("--call-rule", default="qvalue", show_default=True,
              type=click.Choice(["qvalue", "bh", "lfdr"]),
              help="Value thresholded by --fdr: Storey q-value, Benjamini-Hochberg or local false discovery rate.")
@click.option("--suffix", required=True, help="Run suffix for output names.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory for eqtl_peak_{suffix}.csv and eqtl_stats_{suffix}.csv.")
@click.option("--verbose", is_flag=True, default=False)
def report(
    egenes: Path,
    allele_effects: Path,
    ci: Path,
    finemapping: Optional[Path],
    pval_perm: Optional[Path],
    perm_dir: Path,
    cache: Path,
    peak_lod: Optional[Path],
    scan_type: str,
    window_mb: float,
    fdr: float,
    call_rule: str,
    suffix: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Submission tables: per-eGene eqtl_peak and run-level eqtl_stats.

    eqtl_peak_{suffix}.csv joins each eGene with its founder effects and
    support interval. eqtl_stats_{suffix}.csv has one row per chromosome
    (permutation threshold, GEV parameters, eGene count) and one genome row
    (genome-wide GEV fit, FDR threshold, scan window, code version). With
    --finemapping, the credible-set table is checked against
    EQTL_FINEMAPPING_COLS and written as eqtl_finemapping_{suffix}.csv here.
    """
    _setup_logging(verbose)
    import pandas as pd

    from .genotype.cc import display_founders
    from .io.atomic import atomic_write_csv
    from .io.results_io import write_eqtl_peak, write_eqtl_stats
    from .pipeline import EQTL_FINEMAPPING_COLS, calibrate_egenes_storey
    from .utils import git_hash

    output_dir.mkdir(parents=True, exist_ok=True)
    eg = pd.read_csv(egenes, dtype={"chr": str, "gene_id": str})
    if eg.empty:
        logger.info("report: no eGenes — nothing to write")
        return

    sub = eg.copy()
    if pval_perm is not None and sub.get("pval_perm", pd.Series(dtype=float)).isna().all():
        pv = pd.read_csv(pval_perm, dtype={"gene_id": str}).dropna(subset=["pval_perm"])
        if not pv.empty:
            sub = calibrate_egenes_storey(
                sub, dict(zip(pv["gene_id"], pv["pval_perm"])),
                fdr_threshold=fdr, call_rule=call_rule,
            )
    eff = pd.read_csv(allele_effects, dtype={"gene_id": str})
    beta_cols = [c for c in eff.columns if c.startswith("beta_")]
    if beta_cols:
        sub = sub.merge(eff[["gene_id"] + beta_cols], on="gene_id", how="left")
    ci_df = pd.read_csv(ci, dtype={"gene_id": str})
    if not {"ci_lo_bp", "ci_hi_bp"}.issubset(ci_df.columns):
        raise click.ClickException(f"--ci {ci} has no ci_lo_bp/ci_hi_bp columns.")
    sub = sub.merge(ci_df[["gene_id", "ci_lo_bp", "ci_hi_bp"]], on="gene_id", how="left")

    gc = _load_cache(cache)
    peak_path = output_dir / f"eqtl_peak_{suffix}.csv"
    write_eqtl_peak(sub, display_founders(list(gc.founders)), peak_path)
    logger.info("report: %s written (%d eGenes)", peak_path.name, len(sub))

    chr_stats: list[dict] = []
    for ch, pf in _perm_chr_files(perm_dir):
        row = _gev_row(ch, pd.read_csv(pf)["max_lod"].values)
        row["n_egenes"] = int((sub["chr"].astype(str) == ch).sum())
        chr_stats.append(row)
    gw_stats: dict = {
        "chr": "genome",
        "n_egenes": len(sub),
        "fdr_threshold": fdr,
        "scan_window_mb": window_mb if scan_type == "L" else None,
        "git_hash": git_hash(),
    }
    perm_csv = perm_dir / "perm_max_lods.csv"
    if perm_csv.exists():
        gw = _gev_row("genome", pd.read_csv(perm_csv)["max_lod"].values)
        gw_stats.update({k: v for k, v in gw.items() if k != "chr"})
    stats_path = output_dir / f"eqtl_stats_{suffix}.csv"
    write_eqtl_stats(chr_stats, gw_stats, stats_path)
    logger.info("report: %s written (%d chr rows + genome row)", stats_path.name, len(chr_stats))

    if finemapping is not None:
        fm = pd.read_csv(finemapping, dtype={"chr": str, "gene_id": str})
        missing = [c for c in EQTL_FINEMAPPING_COLS if c not in fm.columns]
        if missing:
            raise click.ClickException(
                f"--finemapping {finemapping} is missing columns {missing}."
            )
        fm_path = output_dir / f"eqtl_finemapping_{suffix}.csv"
        if fm_path.resolve() != finemapping.resolve():
            atomic_write_csv(fm[EQTL_FINEMAPPING_COLS], fm_path)
        n_fm = fm["gene_id"].nunique() if not fm.empty else 0
        logger.info("report: %s — %d credible-set SNPs, %d / %d eGenes fine-mapped",
                    fm_path.name, len(fm), n_fm, len(sub))

    if peak_lod is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .qtl.thresholds import load_per_chr_thresholds
        from .viz.genome import genome_summary

        peaks = pd.read_csv(peak_lod, dtype={"chr": str, "gene_id": str})
        is_egene = peaks["gene_id"].isin(set(sub["gene_id"]))
        fig = genome_summary(peaks, is_egene, thr_by_chr=load_per_chr_thresholds(perm_dir))
        fig.savefig(output_dir / f"genome_summary_{suffix}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("report: genome_summary_%s.png written", suffix)


# ── follow-up scans on eGenes (scan-mi / mediate / assess-stability) ──────


def _followup_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Shared options of the per-eGene follow-up scans."""
    opts = [
        click.option("--egenes", required=True,
                     type=click.Path(exists=True, path_type=Path),
                     help="eqtl_summary CSV from `call-egenes`."),
        click.option("--phenotypes", required=True,
                     type=click.Path(exists=True, path_type=Path),
                     help="counts_lognorm.h5 — log-normalised expression."),
        click.option("--cache", required=True,
                     type=click.Path(exists=True, path_type=Path),
                     help="GenomeCache HDF5 file."),
        click.option("--metadata", required=True,
                     type=click.Path(exists=True, path_type=Path),
                     help="Sample metadata CSV with sample_id and genotype columns."),
        click.option("--covariates", default="sex,dataset", show_default=True,
                     help="Comma-separated covariate columns from metadata."),
        click.option("--output", required=True, type=click.Path(path_type=Path),
                     help="Output CSV path."),
        click.option("--verbose", is_flag=True, default=False),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


@main.command("scan-mi")
@_followup_options
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--n-imp", default=11, show_default=True, type=int,
              help="Number of hard founder-haplotype imputations.")
@click.option("--seed", default=1, show_default=True, type=int)
def scan_mi(
    egenes: Path,
    phenotypes: Path,
    cache: Path,
    metadata: Path,
    covariates: str,
    output: Path,
    verbose: bool,
    kinship: Path,
    n_imp: int,
    seed: int,
) -> None:
    """Multiple-imputation haplotype scan around each eGene's peak.

    LOD is averaged over hard founder imputations drawn from the haplotype
    probabilities, integrating haplotype-assignment uncertainty.
    """
    _setup_logging(verbose)
    import pandas as pd

    from .qtl.mi import scan_multi_gene_mi

    output.parent.mkdir(parents=True, exist_ok=True)
    eg, gene_meta, peak_mb_map = _egene_context(egenes)
    if eg.empty:
        logger.info("scan-mi: no eGenes — writing empty output")
        pd.DataFrame().to_csv(output, index=False)
        return
    expr, geno_ids, cov = _load_expression_for(
        phenotypes, metadata, eg["gene_id"].tolist(), _split_cols(covariates)
    )
    chr_map = {str(r["gene_id"]): str(r["chr"]) for _, r in eg.iterrows() if r["gene_id"] in expr}
    df = scan_multi_gene_mi(
        expr,
        chr_map,
        _load_cache(cache),
        covariates=cov,
        kinship_decomp=_load_kinship_h5(kinship),
        sample_ids=geno_ids,
        n_imp=n_imp,
        seed=seed,
        peak_mb_map=peak_mb_map,
        output_path=output,
        gene_meta=gene_meta,
    )
    if df.empty:
        pd.DataFrame().to_csv(output, index=False)
    logger.info("scan-mi: %d eGenes scanned → %s", len(chr_map), output)


@main.command("mediate")
@_followup_options
@click.option("--kinship", required=True, type=click.Path(exists=True, path_type=Path),
              help="LOCO kinship HDF5 from `build-kinship`.")
@click.option("--gene-info", required=True, type=click.Path(exists=True, path_type=Path),
              help="Gene-info CSV (gene_id, chr, start) used to pair nearby eGenes.")
@click.option("--window-mb", default=10.0, show_default=True, type=float,
              help="Mediator TSS must lie within this distance of the outcome's peak.")
def mediate(
    egenes: Path,
    phenotypes: Path,
    cache: Path,
    metadata: Path,
    covariates: str,
    output: Path,
    verbose: bool,
    kinship: Path,
    gene_info: Path,
    window_mb: float,
) -> None:
    """Test whether nearby eGenes mediate each other's haplotype QTL signal.

    Mediator candidates are restricted to eGenes: a gene with no haplotype
    effect of its own cannot carry the outcome's genetic signal.
    """
    _setup_logging(verbose)
    import pandas as pd

    from .qtl.mediation import mediation_scan_multi_outcome, select_cis_mediation_pairs

    output.parent.mkdir(parents=True, exist_ok=True)
    eg, gene_meta, _ = _egene_context(egenes)
    if len(eg) < 2 or not {"chr", "peak_mb"}.issubset(eg.columns):
        logger.info("mediate: need >= 2 eGenes with chr/peak_mb — writing empty output")
        pd.DataFrame().to_csv(output, index=False)
        return
    gi = pd.read_csv(gene_info, dtype={"chr": str, "gene_id": str})
    gi_eg = gi[gi["gene_id"].isin(set(eg["gene_id"].dropna()))]
    pairs = select_cis_mediation_pairs(
        eg, gi_eg, window_mb=window_mb, gene_col="gene_id", info_gene_col="gene_id"
    )
    logger.info("mediate: %d (outcome, mediator) pairs", len(pairs))
    if not pairs:
        pd.DataFrame().to_csv(output, index=False)
        return
    by_outcome: dict[str, list[tuple[str, str]]] = {}
    for outcome, mediator, ch in pairs:
        by_outcome.setdefault(outcome, []).append((mediator, ch))
    ids = sorted(set(by_outcome) | {m for v in by_outcome.values() for m, _ in v})
    expr, geno_ids, cov = _load_expression_for(phenotypes, metadata, ids, _split_cols(covariates))
    df = mediation_scan_multi_outcome(
        by_outcome,
        expr,
        _load_cache(cache),
        covariates=cov,
        kinship_decomp=_load_kinship_h5(kinship),
        sample_ids=geno_ids,
        output_path=output,
        gene_meta=gene_meta,
    )
    if df.empty:
        pd.DataFrame().to_csv(output, index=False)
    logger.info("mediate: %d outcomes tested → %s", len(by_outcome), output)


@main.command("assess-stability")
@_followup_options
@click.option("--n-subsamples", default=100, show_default=True, type=int,
              help="Number of sample subsets rescanned per eGene.")
@click.option("--seed", default=1, show_default=True, type=int)
def assess_stability(
    egenes: Path,
    phenotypes: Path,
    cache: Path,
    metadata: Path,
    covariates: str,
    output: Path,
    verbose: bool,
    n_subsamples: int,
    seed: int,
) -> None:
    """Peak-position stability of each eGene under repeated sample subsampling."""
    _setup_logging(verbose)
    import pandas as pd

    from .qtl.instability import instability_scan_multi_gene

    output.parent.mkdir(parents=True, exist_ok=True)
    eg, gene_meta, peak_mb_map = _egene_context(egenes)
    if eg.empty:
        logger.info("assess-stability: no eGenes — writing empty output")
        pd.DataFrame().to_csv(output, index=False)
        return
    expr, geno_ids, cov = _load_expression_for(
        phenotypes, metadata, eg["gene_id"].tolist(), _split_cols(covariates)
    )
    chr_map = {str(r["gene_id"]): str(r["chr"]) for _, r in eg.iterrows() if r["gene_id"] in expr}
    df = instability_scan_multi_gene(
        expr,
        chr_map,
        _load_cache(cache),
        covariates=cov,
        sample_ids=geno_ids,
        n_subsamples=n_subsamples,
        seed=seed,
        peak_mb_map=peak_mb_map,
        output_path=output,
        gene_meta=gene_meta,
    )
    if df.empty:
        pd.DataFrame().to_csv(output, index=False)
    logger.info("assess-stability: %d eGenes scanned → %s", len(chr_map), output)


if __name__ == "__main__":
    main()
