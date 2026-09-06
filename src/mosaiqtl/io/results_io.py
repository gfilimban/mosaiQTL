"""results_io — helpers for turning scan results into tables for output files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import stats as _sp_stats

from mosaiqtl.pipeline import (
    _LOCUS_COLS,
    _SIG_COLS,
    EQTL_PEAK_COLS,
    EQTL_SCAN_COLS,
    EQTL_STATS_COLS,
    LOD_COLS,
)

if TYPE_CHECKING:
    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.models.records import ScanResult


def pos_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_val: str,
    right_val: str,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
    round_dp: int = 3,
) -> pd.DataFrame:
    """join two scan result tables on matching (chromosome, position).

    keeps only the highest-value row per (chromosome, position) in each table
    before joining, so multiple genes sharing a locus don't create duplicate
    matched rows.

    parameters
    ----------
    left, right:
        DataFrames that each contain at least ``chr``, ``pos_mb``, and
        the column named by ``left_val`` / ``right_val``.
    left_val, right_val:
        column names to extract from each DataFrame.
    left_label, right_label:
        output column names. when omitted: uses ``left_val`` / ``right_val``
        directly when they differ; adds ``_left`` / ``_right`` suffixes when
        both names are the same.
    round_dp:
        decimal places to round ``pos_mb`` to before matching (default 3).

    returns
    -------
    DataFrame with columns ``_chr``, ``_pos``, and the two value columns.
    """
    left = left.copy()
    right = right.copy()
    left["_chr"] = left["chr"].astype(str)
    right["_chr"] = right["chr"].astype(str)
    left["_pos"] = left["pos_mb"].round(round_dp)
    right["_pos"] = right["pos_mb"].round(round_dp)

    same_name = (left_val == right_val) and left_label is None and right_label is None
    left_out = left_label or (left_val + ("_left" if same_name else ""))
    right_out = right_label or (right_val + ("_right" if same_name else ""))

    left_d = (
        left[["_chr", "_pos", left_val]]
        .rename(columns={left_val: left_out})
        .sort_values(left_out, ascending=False)
        .drop_duplicates(["_chr", "_pos"])
    )
    right_d = (
        right[["_chr", "_pos", right_val]]
        .rename(columns={right_val: right_out})
        .sort_values(right_out, ascending=False)
        .drop_duplicates(["_chr", "_pos"])
    )
    return pd.merge(left_d, right_d, on=["_chr", "_pos"]).dropna()


def scan_result_to_df(
    scan: "ScanResult",
    gene_id: str,
    gene_symbol: str = "",
) -> pd.DataFrame:
    """Convert a ScanResult to a per-locus DataFrame.

    Parameters
    ----------
    scan:
        Completed scan result from ``scan_h2lmm`` or equivalent.
    gene_id:
        ENSMUSG ID (version-stripped) — stable join key written to the
        ``gene_id`` column.
    gene_symbol:
        Display name (e.g. 'Ankrd39') written to the ``gene_symbol`` column.
        Empty string when not available.

    Returns
    -------
    DataFrame with columns given by ``pipeline.LOD_COLS`` (the single source
    of truth for this schema — never redefined inline here).
    """
    values = {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "chr": np.asarray(scan.chromosomes),
        "locus": np.asarray(scan.locus_ids),
        "pos_mb": np.asarray(scan.positions),
        "lod": np.asarray(scan.lod),
    }
    return pd.DataFrame({col: values[col] for col in LOD_COLS})


def _atomic_write_csv(df: pd.DataFrame, path: "Path | str") -> None:
    """Write ``df`` to ``path`` atomically: temp file in the same dir, then rename.

    Parameters
    ----------
    df:
        DataFrame to write (no index column).
    path:
        Destination CSV path. Parent directory must already exist.
    """
    dest = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False)
        os.replace(tmp_name, dest)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def build_eqtl_peak_df(
    effects_egenes_df: pd.DataFrame,
    founder_names: list[str],
) -> pd.DataFrame:
    """Reindex a calibrated eGene table to the ``EQTL_PEAK_COLS`` schema.

    Populates every field derivable from data the scan pipeline already
    produces, and leaves the haplotype-block-to-biallelic-contrast fields
    (``variant``, ``ref``, ``effect_allele``, ``other_allele``,
    ``ref_founders``, ``alt_founders``, ``af``, ``maf``, ``mac``,
    ``ma_sample``, ``effect_size``, ``se_effect_size``) as NaN — collapsing
    an 8-founder haplotype dosage locus into a 2-allele contrast is not yet
    implemented anywhere in the codebase (it depends on the unbuilt Sanger
    fine-mapping subsystem, ``genotype/interpolation.py`` +
    ``qtl/imputation.py``, which supplies real REF/ALT calls). Filling those
    columns here would mean inventing a founder-split rule with no grounding
    in real variant data.

    Parameters
    ----------
    effects_egenes_df:
        One row per eGene, produced by merging ``pipeline.calibrate_egenes_storey``
        output (adds ``pval_perm``/``fdr``/``fdr_method``/``significance``) with
        ``allele_effects.csv`` (``pipeline.ALLELE_EFFECTS_BASE_COLS`` +
        ``beta_{founder}`` columns) on ``gene_id`` — the same merge pattern
        already used for plotting in validate_tissue.py. Must contain at
        least: gene_id, gene_symbol, chr, peak_mb, peak_lod, h2, n_valid,
        rss0, pval_perm, fdr, fdr_method, significance, and one
        ``beta_{founder}`` column per entry in ``founder_names``.
        ``cis_dist_mb`` is carried through when present (e.g. joined from
        ``peak_lod.csv``); NaN otherwise.
    founder_names:
        Display founder names (e.g. from ``_display_founders``) identifying
        the ``beta_{founder}`` columns to rename to ``effect_{founder}``.

    Returns
    -------
    DataFrame with columns given by ``pipeline.EQTL_PEAK_COLS``, in order.
    """
    out = pd.DataFrame(index=effects_egenes_df.index)
    src = effects_egenes_df

    out["chr"] = src.get("chr")
    out["pos"] = (
        (src["peak_mb"] * 1e6).round().astype("Int64") if "peak_mb" in src else pd.NA
    )
    out["hard_call_dosage"] = "F"
    out["gene_id"] = src.get("gene_id")
    out["gene_symbol"] = src.get("gene_symbol")
    if "rss0" in src and "n_valid" in src:
        out["sd_phenotype"] = np.sqrt(src["rss0"] / (src["n_valid"] - 1))
    out["n_sample"] = src.get("n_valid")
    out["lod"] = src.get("peak_lod")
    out["h2_null_chr"] = src.get("h2")
    if "peak_lod" in src:
        chi2_haplotype = 2.0 * np.log(10) * src["peak_lod"]
        out["pval_nominal"] = _sp_stats.chi2.sf(
            chi2_haplotype, df=len(founder_names) - 1
        )
        with np.errstate(divide="ignore"):
            out["log10p_nominal"] = -np.log10(out["pval_nominal"])
    out["cis_dist_mb"] = src.get("cis_dist_mb", pd.NA)

    # not yet computable — see docstring
    for col in (
        "variant",
        "ref",
        "effect_allele",
        "other_allele",
        "ref_founders",
        "alt_founders",
        "af",
        "maf",
        "mac",
        "ma_sample",
        "effect_size",
        "se_effect_size",
    ):
        out[col] = pd.NA

    for founder in founder_names:
        beta_col = f"beta_{founder}"
        out[f"effect_{founder}"] = src[beta_col] if beta_col in src else pd.NA

    for col in _SIG_COLS:
        out[col] = src.get(col, pd.NA)

    out["interaction"] = "F"
    out["interaction_context"] = ""
    out["conditional"] = ""
    out["conditional_method"] = ""

    for col in _LOCUS_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    return out.reindex(columns=EQTL_PEAK_COLS).reset_index(drop=True)


def write_eqtl_peak(
    effects_egenes_df: pd.DataFrame,
    founder_names: list[str],
    out_path: "Path | str",
) -> pd.DataFrame:
    """Build the ``_eqtl_peak.csv`` table and write it atomically.

    See ``build_eqtl_peak_df`` for the field-population rules and the
    still-NaN haplotype-contrast columns.

    Returns
    -------
    The DataFrame that was written (``pipeline.EQTL_PEAK_COLS`` schema).
    """
    df = build_eqtl_peak_df(effects_egenes_df, founder_names)
    _atomic_write_csv(df, out_path)
    return df


def build_eqtl_stats_df(
    chrom_stats: "list[dict]",
    genome_stats: "dict | None" = None,
) -> pd.DataFrame:
    """Assemble the ``_eqtl_stats.csv`` table (per-chr + genome-wide rows).

    Parameters
    ----------
    chrom_stats:
        One dict per chromosome, keyed by any subset of
        ``pipeline.EQTL_STATS_COLS`` (e.g. ``tissue``, ``celltype``, ``chr``,
        ``lod_threshold``, ``gev_mu``, ``gev_sigma``, ``gev_xi``,
        ``n_genes_scanned``, ``n_egenes``, ``n_loci``, ``n_perms``). A GEV
        params dict from ``qtl.permutation.get_gev_thresholds`` uses scipy's
        ``genextreme`` naming (``loc``/``scale``/``c``) rather than
        ``gev_mu``/``gev_sigma``/``gev_xi`` — remap before calling this
        (``loc``→``gev_mu``, ``scale``→``gev_sigma``, ``c``→``gev_xi``,
        ``threshold``→``lod_threshold``).
    genome_stats:
        Optional single dict for the genome-wide summary row (``chr`` set to
        ``'genome'`` if not already present) — the only row where
        ``fdr_threshold``/``scan_window_mb``/``scan_seed``/``gencode_version``/
        ``ensembl_release``/``git_hash`` are populated.

    Returns
    -------
    DataFrame with columns given by ``pipeline.EQTL_STATS_COLS``, in order.
    """
    rows = list(chrom_stats)
    if genome_stats is not None:
        genome_row = dict(genome_stats)
        genome_row.setdefault("chr", "genome")
        rows.append(genome_row)
    out = pd.DataFrame(rows)
    for col in EQTL_STATS_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out.reindex(columns=EQTL_STATS_COLS).reset_index(drop=True)


def write_eqtl_stats(
    chrom_stats: "list[dict]",
    genome_stats: "dict | None",
    out_path: "Path | str",
) -> pd.DataFrame:
    """Build the ``_eqtl_stats.csv`` table and write it atomically.

    See ``build_eqtl_stats_df`` for the input dict shapes.

    Returns
    -------
    The DataFrame that was written (``pipeline.EQTL_STATS_COLS`` schema).
    """
    df = build_eqtl_stats_df(chrom_stats, genome_stats)
    _atomic_write_csv(df, out_path)
    return df


# _SANGER_COLS — subset of _LOCUS_COLS that the Sanger fine-mapping subsystem
# populates; left as NaN in the haplotype-only scan output.
_SANGER_COLS: list[str] = [
    "variant",
    "ref",
    "effect_allele",
    "other_allele",
    "ref_founders",
    "alt_founders",
    "af",
    "maf",
    "mac",
    "ma_sample",
    "effect_size",
    "se_effect_size",
]

# _NULL_MODEL_COLS — columns derivable from the per-gene null model fit that are
# not stored on ScanResult (they come from the separate null-fit CSV).  Also NaN
# in haplotype-scan eqtl_scan output; populated when the eqtl_peak writer merges
# in the allele-effects DataFrame.
_NULL_MODEL_COLS: list[str] = [
    "h2_null_chr",
    "n_sample",
    "sd_phenotype",
    "cis_dist_mb",
]


def build_eqtl_scan_df(
    scan: "dict[str, ScanResult]",
    genomecache: "GenomeCache",
    gene_symbol_map: "dict[str, str] | None" = None,
    n_founders: int = 7,
) -> pd.DataFrame:
    """Reindex all per-locus scan rows to the ``EQTL_SCAN_COLS`` schema.

    Produces one row per (gene × scanned locus), which is the full intermediate
    scan table before peak-filtering.  The ``peak_locus`` column flags the
    argmax-LOD row per gene with ``'T'``; all others are ``'F'``.

    Fields populated from ``ScanResult``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``chr``, ``pos`` (integer bp), ``hard_call_dosage`` (constant ``'F'``),
    ``gene_id``, ``gene_symbol``, ``lod``, ``pval_nominal`` (chi² with
    ``n_founders`` df), ``log10p_nominal``, and ``peak_locus``.

    Fields left as NaN
    ~~~~~~~~~~~~~~~~~~
    * Sanger fine-mapping fields (``variant``, ``ref``, ``effect_allele``,
      ``other_allele``, ``ref_founders``, ``alt_founders``, ``af``, ``maf``,
      ``mac``, ``ma_sample``, ``effect_size``, ``se_effect_size``) — require the
      unbuilt Sanger fine-mapping subsystem.
    * Null-model fields (``h2_null_chr``, ``n_sample``, ``sd_phenotype``,
      ``cis_dist_mb``) — not stored on ``ScanResult``; available in the
      per-gene null-fit CSV and added by the ``eqtl_peak`` writer after merging
      allele effects.

    Parameters
    ----------
    scan:
        ``{gene_id: ScanResult}`` mapping produced by ``pipeline.discover()``.
        Genes with empty scan results (zero loci) are silently skipped.
    genomecache:
        ``GenomeCache`` used for the scan — consulted to resolve locus IDs to
        chromosome labels.  Only locus-ID and position arrays are accessed (no
        haplotype probability load).
    gene_symbol_map:
        Optional ``{gene_id: gene_symbol}`` for the display-name column.  When
        ``None``, ``gene_symbol`` is left as an empty string.
    n_founders:
        Chi-squared degrees of freedom for ``pval_nominal`` (default 7 = eight
        founders minus one intercept constraint).

    Returns
    -------
    DataFrame with columns given by ``pipeline.EQTL_SCAN_COLS``, in order.
    """
    # Build locus-id → chromosome lookup from the cache.  Accessing lids and
    # positions per chromosome is lightweight — no haplotype-probability loads.
    locus_chr: dict[str, str] = {}
    for chrom in genomecache.chromosomes:
        lids, _pos, _probs = genomecache.get_chromosome(chrom)
        for lid in lids:
            locus_chr[lid] = chrom

    sym_map = gene_symbol_map or {}
    chi2_scale = 2.0 * np.log(10)  # convert LOD to chi2 statistic

    rows: list[dict] = []
    for gene_id, sr in scan.items():
        if len(sr.lod) == 0:
            continue
        gene_symbol = sym_map.get(gene_id, "")
        peak_idx = int(np.argmax(sr.lod))

        for i, (lid, pos_mb_val, lod_val) in enumerate(zip(sr.loci, sr.pos_mb, sr.lod)):
            chrom = locus_chr.get(lid, str(sr.chr))
            pos_bp = int(round(float(pos_mb_val) * 1e6))

            chi2_val = chi2_scale * float(lod_val)
            if chi2_val >= 0.0:
                pval = float(_sp_stats.chi2.sf(chi2_val, df=n_founders))
                with np.errstate(divide="ignore"):
                    log10p = float(-np.log10(pval)) if pval > 0.0 else np.inf
            else:
                pval = np.nan
                log10p = np.nan

            rows.append(
                {
                    "chr": chrom,
                    "pos": pos_bp,
                    "hard_call_dosage": "F",
                    "gene_id": gene_id,
                    "gene_symbol": gene_symbol,
                    "lod": float(lod_val),
                    "pval_nominal": pval,
                    "log10p_nominal": log10p,
                    "peak_locus": "T" if i == peak_idx else "F",
                }
            )

    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=EQTL_SCAN_COLS)

    # NaN-fill all columns the haplotype scan cannot populate.
    for col in _SANGER_COLS + _NULL_MODEL_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    # Remaining EQTL_SCAN_COLS not yet present also get NaN.
    for col in EQTL_SCAN_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    return out.reindex(columns=EQTL_SCAN_COLS).reset_index(drop=True)


def write_eqtl_scan(
    scan: "dict[str, ScanResult]",
    genomecache: "GenomeCache",
    out_path: "Path | str",
    gene_symbol_map: "dict[str, str] | None" = None,
    n_founders: int = 7,
) -> pd.DataFrame:
    """Build the ``_eqtl_scan.csv`` table and write it atomically.

    See ``build_eqtl_scan_df`` for field-population rules and NaN columns.

    Returns
    -------
    The DataFrame that was written (``pipeline.EQTL_SCAN_COLS`` schema).
    """
    df = build_eqtl_scan_df(scan, genomecache, gene_symbol_map, n_founders)
    _atomic_write_csv(df, out_path)
    return df
