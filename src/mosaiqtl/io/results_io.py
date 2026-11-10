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
    _FOUNDER_EFFECT_COLS,
    _SCAN_COLS,
    _SIG_COLS,
    CS_RESOLVED_PIP_THRESHOLD,
    EQTL_FINEMAPPING_COLS,
    EQTL_FINEMAPPING_PEAK_COLS,
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


def _atomic_write_csv(
    df: pd.DataFrame,
    path: "Path | str",
    float_format: "str | None" = None,
) -> None:
    """Write ``df`` to ``path`` atomically: temp file in the same dir, then rename.

    Parameters
    ----------
    df:
        DataFrame to write (no index column).
    path:
        Destination CSV path. Parent directory must already exist.
    float_format:
        Optional format string passed to ``DataFrame.to_csv``.  Pass
        ``'%.6g'`` for tables that contain very small floats (e.g.
        heritability estimates) so they are written in scientific notation
        rather than rounded to zero.
    """
    dest = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            df.to_csv(fh, index=False, float_format=float_format)
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
    produces.  Sanger allele annotation columns (``_ALLELE_COLS``) are not
    part of this schema — they live in ``EQTL_FINEMAPPING_PEAK_COLS`` where
    the Sanger SNP is the row key.

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

    for founder in founder_names:
        beta_col = f"beta_{founder}"
        out[f"effect_{founder}"] = src[beta_col] if beta_col in src else pd.NA

    for col in _SIG_COLS:
        out[col] = src.get(col, pd.NA)

    for col in _SCAN_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    return out.reindex(columns=EQTL_PEAK_COLS).reset_index(drop=True)


def write_eqtl_peak(
    effects_egenes_df: pd.DataFrame,
    founder_names: list[str],
    out_path: "Path | str",
) -> pd.DataFrame:
    """Build the ``_eqtl_peak.csv`` table and write it atomically.

    See ``build_eqtl_peak_df`` for the field-population rules.

    Returns
    -------
    The DataFrame that was written (``pipeline.EQTL_PEAK_COLS`` schema).
    """
    df = build_eqtl_peak_df(effects_egenes_df, founder_names)
    _atomic_write_csv(df, out_path, float_format='%.6g')
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


def build_gev_row(chrom: str, max_lods: "np.ndarray") -> dict:
    """build a GEV-threshold dict for one chromosome (or ``'genome'``).

    Parameters
    ----------
    chrom :
        chromosome label, e.g. ``'19'`` or ``'genome'``.
    max_lods :
        permutation max-LOD array for this chromosome.

    Returns
    -------
    dict with keys ``chr``, ``lod_threshold``, ``n_perms``
    and (when GEV fit converged) ``gev_mu``, ``gev_sigma``, ``gev_xi``.
    """
    from mosaiqtl.qtl.thresholds import get_gev_thresholds
    thr, params = get_gev_thresholds(max_lods)
    row: dict = {'chr': chrom, 'lod_threshold': thr, 'n_perms': len(max_lods)}
    if params:
        row['gev_mu']    = params.get('loc')
        row['gev_sigma'] = params.get('scale')
        row['gev_xi']    = params.get('c')
    return row


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
    _atomic_write_csv(df, out_path, float_format='%.6g')
    return df


# _SANGER_COLS — subset of _LOCUS_COLS that the Sanger fine-mapping subsystem
# populates; left as NaN in the haplotype-only scan output.
_SANGER_COLS: list[str] = [
    "variant",
    "rsid",
    "ref",
    "alt",
    "effect_allele",
    "other_allele",
    "dosage",
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
    ``chr``, ``pos`` (integer bp), ``gene_id``, ``gene_symbol``, ``lod``,
    ``pval_nominal`` (chi² with ``n_founders`` df), ``log10p_nominal``,
    and ``peak_locus``.

    Fields left as NaN
    ~~~~~~~~~~~~~~~~~~
    * Sanger fine-mapping fields (``variant``, ``ref``, ``alt``,
      ``ref_founders``, ``alt_founders``, ``af``, ``maf``,
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
    # build locus-id → chromosome lookup — get_locus_metadata() reads lids/pos
    # already in RAM; probs (h5py Dataset) are never touched
    locus_chr: dict[str, str] = {}
    for chrom in genomecache.chromosomes:
        lids, _pos = genomecache.get_locus_metadata(chrom)
        for lid in lids:
            locus_chr[lid] = chrom

    sym_map = gene_symbol_map or {}
    chi2_scale = 2.0 * np.log(10)

    # per-gene vectorized ops: one chi2.sf call per gene over all loci,
    # one DataFrame construction per gene, then a single pd.concat
    gene_frames: list[pd.DataFrame] = []
    for gene_id, sr in scan.items():
        if len(sr.lod) == 0:
            continue
        n = len(sr.lod)
        lod_arr = np.asarray(sr.lod, dtype=np.float64)
        pos_bp = (np.asarray(sr.pos_mb, dtype=np.float64) * 1e6).round().astype(np.int64)

        chi2 = chi2_scale * np.maximum(lod_arr, 0.0)
        pval = _sp_stats.chi2.sf(chi2, df=n_founders)
        with np.errstate(divide="ignore"):
            log10p = -np.log10(np.maximum(pval, 1e-300))

        peak_flag = np.full(n, "F", dtype=object)
        peak_flag[int(np.argmax(lod_arr))] = "T"

        gene_frames.append(pd.DataFrame({
            "chr": [locus_chr.get(lid, str(sr.chr)) for lid in sr.loci],
            "pos": pos_bp,
            "gene_id": gene_id,
            "gene_symbol": sym_map.get(gene_id, ""),
            "lod": lod_arr,
            "pval_nominal": pval,
            "log10p_nominal": log10p,
            "peak_locus": peak_flag,
        }))

    out = pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame(columns=EQTL_SCAN_COLS)
    # reindex fills all missing EQTL_SCAN_COLS (Sanger + null-model) with NaN
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


def build_eqtl_finemapping_df(
    fm_df: pd.DataFrame,
    gene_id: str,
    chr_: str,
    hap_peak_pos: int,
    hap_peak_variant: str,
    hap_peak_lod: float,
    ci_lo_bp: "int | None" = None,
    ci_hi_bp: "int | None" = None,
    gene_symbol: str = "",
    alleles_df: "pd.DataFrame | None" = None,
    founder_effects: "dict[str, float] | None" = None,
    sig_values: "dict[str, object] | None" = None,
    null_model_values: "dict[str, object] | None" = None,
) -> pd.DataFrame:
    """Build the ``EQTL_FINEMAPPING_COLS`` DataFrame for one gene's credible set.

    Consumes the output of :func:`qtl.imputation.fine_map_region` filtered to
    credible-set members (``in_cs == True``).  Sanger allele annotations (``ref``,
    ``alt``, ``ref_founders``, etc.) are merged from ``alleles_df`` when supplied;
    they remain NaN otherwise.  Gene-level haplotype betas and significance are
    broadcast to every CS row so the file is self-contained.

    Parameters
    ----------
    fm_df:
        Subset of :func:`~qtl.imputation.fine_map_region` output with
        ``in_cs == True``, sorted by PIP descending (the first row is the lead
        SNP).  Must contain ``snp``, ``lod``, ``pip``; optionally ``pos_mb``.
    gene_id:
        ENSMUSG ID (version-stripped) — stable join key.
    chr_:
        Chromosome string (e.g. ``'8'``).
    hap_peak_pos:
        Integer bp position of the GigaMUGA haplotype peak — back-link to
        ``_eqtl_peak.csv``.
    hap_peak_variant:
        GigaMUGA locus ID of the haplotype peak.
    hap_peak_lod:
        LOD score at the haplotype peak.
    ci_lo_bp, ci_hi_bp:
        LOD-drop CI bounds (bp).  ``None`` writes NaN.
    gene_symbol:
        Display gene name.
    alleles_df:
        Optional per-SNP Sanger allele annotations with a ``snp`` column
        matching ``fm_df["snp"]`` and any subset of ``_SANGER_COLS``.
    founder_effects:
        Per-founder haplotype betas at the GigaMUGA peak, keyed by
        ``effect_{founder}`` column names.  Broadcast to every CS row.
        ``None`` leaves ``_FOUNDER_EFFECT_COLS`` as NaN.
    sig_values:
        Gene-level significance block keyed by ``_SIG_COLS`` names.
        Broadcast to every CS row.  ``None`` leaves ``_SIG_COLS`` as NaN.
    null_model_values:
        Null-model summary keyed by ``_NULL_MODEL_COLS`` names
        (``h2_null_chr``, ``n_sample``, ``sd_phenotype``, ``cis_dist_mb``).
        Broadcast to every CS row.  ``None`` leaves those columns as NaN.

    Returns
    -------
    DataFrame with columns given by ``pipeline.EQTL_FINEMAPPING_COLS``, in
    order.
    """
    # restrict to CS members; fm_df is already sorted by PIP desc from fine_map_region
    cs = fm_df[fm_df["in_cs"]].reset_index(drop=True)
    n = len(cs)

    lod_arr = np.asarray(cs["lod"], dtype=np.float64)
    chi2_scale = 2.0 * np.log(10)
    pval = _sp_stats.chi2.sf(chi2_scale * np.maximum(lod_arr, 0.0), df=1)
    with np.errstate(divide="ignore"):
        log10p = -np.log10(np.maximum(pval, 1e-300))

    pos_col: "np.ndarray | object"
    if "pos_mb" in cs.columns:
        pos_col = (np.asarray(cs["pos_mb"], dtype=np.float64) * 1e6).round().astype(np.int64)
    else:
        pos_col = pd.NA

    lead_flag = np.full(n, "F", dtype=object)
    lead_flag[0] = "T"   # first row is highest-PIP SNP (fine_map_region sorts desc)

    out = pd.DataFrame({
        "variant": cs["snp"].to_numpy(),
        "chr": f'chr{chr_}' if not str(chr_).startswith('chr') else str(chr_),
        "pos": pos_col,
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "lod": lod_arr,
        "pval_nominal": pval,
        "log10p_nominal": log10p,
        "pip": np.asarray(cs["pip"], dtype=np.float64),
        "pip_method": "ABF",
        "CS": "T",
        "lead_snp": lead_flag,
        "ci_lo_bp": ci_lo_bp if ci_lo_bp is not None else pd.NA,
        "ci_hi_bp": ci_hi_bp if ci_hi_bp is not None else pd.NA,
        "hap_peak_pos": hap_peak_pos,
        "hap_peak_variant": hap_peak_variant,
        "hap_peak_lod": hap_peak_lod,
    })

    if alleles_df is not None and "snp" in alleles_df.columns:
        avail = [c for c in _SANGER_COLS if c in alleles_df.columns]
        if avail:
            out = out.merge(
                alleles_df[["snp"] + avail],
                left_on="variant", right_on="snp",
                how="left",
            ).drop(columns="snp")

    # broadcast gene-level betas and significance to every CS row
    eff = founder_effects or {}
    for col in _FOUNDER_EFFECT_COLS:
        out[col] = eff.get(col, pd.NA)
    sig = sig_values or {}
    for col in _SIG_COLS:
        out[col] = sig.get(col, pd.NA)
    nm = null_model_values or {}
    for col in _NULL_MODEL_COLS:
        out[col] = nm.get(col, pd.NA)
    # mac/ma_sample from imputed dosages: maf is dataset-level (48 strains)
    if "maf" in out.columns:
        _maf = pd.to_numeric(out["maf"], errors="coerce")
        out["mac"] = (_maf * 48).round().astype("Int64")
        _ns = pd.to_numeric(out.get("n_sample", pd.Series(dtype=float)), errors="coerce")
        if _ns.notna().any():
            out["ma_sample"] = (_maf * _ns).round().astype("Int64")

    _PVAL_COLS = {"pval_perm", "fdr", "pval_nominal"}
    for _pc in _PVAL_COLS:
        if _pc in out.columns:
            out[_pc] = out[_pc].apply(
                lambda x: f"{x:.3e}" if pd.notna(x) and isinstance(x, float) else x
            )
    # round remaining float columns to 3 dp
    for _fc in out.columns:
        if _fc not in _PVAL_COLS and pd.api.types.is_float_dtype(out[_fc]):
            out[_fc] = out[_fc].round(3)

    # Preserve _FOUNDER_EFFECT_COLS alongside EQTL_FINEMAPPING_COLS so that
    # build_eqtl_finemapping_peak_df can carry them into the peak file.
    # write_eqtl_finemapping trims to EQTL_FINEMAPPING_COLS at write time.
    return out.reindex(columns=EQTL_FINEMAPPING_COLS + _FOUNDER_EFFECT_COLS).reset_index(drop=True)


def write_eqtl_finemapping(
    fm_df: pd.DataFrame,
    gene_id: str,
    chr_: str,
    hap_peak_pos: int,
    hap_peak_variant: str,
    hap_peak_lod: float,
    out_path: "Path | str",
    ci_lo_bp: "int | None" = None,
    ci_hi_bp: "int | None" = None,
    gene_symbol: str = "",
    alleles_df: "pd.DataFrame | None" = None,
    founder_effects: "dict[str, float] | None" = None,
    sig_values: "dict[str, object] | None" = None,
) -> pd.DataFrame:
    """Build the ``_eqtl_finemapping.csv`` table and write it atomically.

    See ``build_eqtl_finemapping_df`` for field-population rules.

    Returns
    -------
    The DataFrame that was written (``pipeline.EQTL_FINEMAPPING_COLS`` schema).
    """
    df = build_eqtl_finemapping_df(
        fm_df, gene_id, chr_, hap_peak_pos, hap_peak_variant, hap_peak_lod,
        ci_lo_bp=ci_lo_bp, ci_hi_bp=ci_hi_bp, gene_symbol=gene_symbol,
        alleles_df=alleles_df, founder_effects=founder_effects, sig_values=sig_values,
    )
    # trim to EQTL_FINEMAPPING_COLS (per-SNP schema); _FOUNDER_EFFECT_COLS are
    # haplotype-peak properties carried in the intermediate df for the peak builder.
    _atomic_write_csv(df.reindex(columns=EQTL_FINEMAPPING_COLS), out_path)
    return df


def build_eqtl_finemapping_peak_df(
    fm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Filter the finemapping CS table to the lead SNP row.

    Takes the output of :func:`build_eqtl_finemapping_df` (one CS per gene),
    keeps the single ``lead_snp == 'T'`` row, and drops ``lead_snp``.
    All effect, significance, and run columns are already present from
    :func:`build_eqtl_finemapping_df`.

    Parameters
    ----------
    fm_df:
        Output of :func:`build_eqtl_finemapping_df` for one gene.

    Returns
    -------
    DataFrame with columns given by ``pipeline.EQTL_FINEMAPPING_PEAK_COLS``,
    in order (one row).
    """
    lead = fm_df[fm_df["lead_snp"] == "T"].drop(columns="lead_snp").copy().reset_index(drop=True)
    lead["cs_resolved"] = np.where(
        lead["pip"].ge(CS_RESOLVED_PIP_THRESHOLD), "T", "F"
    )
    return lead.reindex(columns=EQTL_FINEMAPPING_PEAK_COLS).reset_index(drop=True)


def write_eqtl_finemapping_peak(
    fm_df: pd.DataFrame,
    out_path: "Path | str",
) -> pd.DataFrame:
    """Build the ``_eqtl_finemapping_peak.csv`` table and write it atomically.

    See ``build_eqtl_finemapping_peak_df`` for field-population rules.

    Returns
    -------
    The DataFrame that was written
    (``pipeline.EQTL_FINEMAPPING_PEAK_COLS`` schema).
    """
    df = build_eqtl_finemapping_peak_df(fm_df)
    _atomic_write_csv(df, out_path)
    return df
