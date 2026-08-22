"""celltypes — per-cell-type QTL cross-comparison plots.

Data-loading helpers, path utilities, and high-level plotting functions for
comparing QTL results across cell types and against whole-tissue scans.
All functions read pre-computed outputs; none run scans.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Carlito"
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

import seaborn as sns

from ..genotype.founders import FOUNDER_ORDER
from ._theme import INK, MUTED
from .compare import (
    lod_scatter as _cmp_scatter,
    lod_density_comparison as _cmp_lod_density,
    allele_effects_comparison as _cmp_allele_effects,
    lod_profile_overlay as _cmp_lod_overlay,
    cis_distance_comparison as _cmp_cis_distance,
)

logger = logging.getLogger("mosaiqtl")

_BETA_COLS: list[str] = [f"beta_{f}" for f in FOUNDER_ORDER]
_FOUNDERS_DISPLAY: list[str] = FOUNDER_ORDER

__all__ = [
    "safe_ct",
    "resolve_ct_scan_dir",
    "discover_cell_types",
    "build_ct_palette",
    "pair_order",
    "melt_betas",
    "load_peak_lod",
    "load_allele_effects",
    "load_gene_lod",
    "load_threshold",
    "ct_vs_whole_lod_scatter",
    "ct_vs_whole_effects_scatter",
    "ct_lod_profile_overlay",
    "ct_pairwise_lod_scatter",
    "ct_pairwise_effects_scatter",
    "ct_lod_distributions",
    "ct_cis_distance_comparison",
]


# ── Path utilities ────────────────────────────────────────────────────────────


def safe_ct(ct: str) -> str:
    """Return a filesystem-safe cell-type token (spaces → underscores).

    Parameters
    ----------
    ct :
        Raw cell-type label as used in metadata and plot titles.

    Returns
    -------
    str
        Same label with spaces replaced by underscores.
    """
    return ct.replace(" ", "_")


def resolve_ct_scan_dir(data_dir: Path, ct: str) -> Path:
    """Return the py_data path for a cell type under the standard layout.

    Layout: ``data_dir/celltypes/<safe_ct>/py_data/``

    Parameters
    ----------
    data_dir :
        Root data directory (e.g. ``gastroc_data/``).
    ct :
        Cell-type label (spaces allowed; sanitised internally).

    Returns
    -------
    Path
        Resolved path (not guaranteed to exist).
    """
    return data_dir / "celltypes" / safe_ct(ct) / "py_data"


def discover_cell_types(celltypes_dir: Path) -> list[str]:
    """Auto-discover cell type names under the standard directory layout.

    Scans for ``celltypes_dir/<name>/py_data/peak_lod.csv``.

    Parameters
    ----------
    celltypes_dir :
        Directory containing one subdirectory per cell type.

    Returns
    -------
    list[str]
        Sorted list of discovered cell-type names (filesystem tokens).
    """
    cts: list[str] = []
    if not celltypes_dir.exists():
        return cts
    for sub in sorted(celltypes_dir.iterdir()):
        if sub.is_dir() and sub.name != "compare":
            if (sub / "py_data" / "peak_lod.csv").exists():
                cts.append(sub.name)
    return cts


# ── Color palette ─────────────────────────────────────────────────────────────


def build_ct_palette(cell_types: list[str]) -> dict[str, str]:
    """Assign a distinct categorical color to each cell type.

    Cell types are sorted alphabetically then assigned colors cycling through
    seaborn's ``tab10`` palette.  Cell types are not founders — founder colors
    must not be used here.

    Parameters
    ----------
    cell_types :
        List of cell-type labels.

    Returns
    -------
    dict[str, str]
        Mapping from cell-type label to hex color string.
    """
    sorted_cts = sorted(cell_types)
    colors = [
        c for c in sns.color_palette("tab10", n_colors=max(len(sorted_cts), 1)).as_hex()
    ]
    return {ct: colors[i % len(colors)] for i, ct in enumerate(sorted_cts)}


# ── Pair ordering ─────────────────────────────────────────────────────────────


def pair_order(ct_a: str, ct_b: str) -> tuple[str, str]:
    """Return ``(x_axis_ct, y_axis_ct)`` for a pairwise comparison.

    ``myonuclei`` is always the x-axis.  Otherwise the alphabetically earlier
    cell type is placed on the x-axis.

    Parameters
    ----------
    ct_a, ct_b :
        Cell-type labels to order.

    Returns
    -------
    tuple[str, str]
        ``(x_ct, y_ct)`` with consistent axis assignment.
    """
    if ct_a == "myonuclei":
        return ct_a, ct_b
    if ct_b == "myonuclei":
        return ct_b, ct_a
    return (ct_a, ct_b) if ct_a <= ct_b else (ct_b, ct_a)


# ── Beta melting ──────────────────────────────────────────────────────────────


def melt_betas(eff_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Melt an allele-effects DataFrame to long form.

    Parameters
    ----------
    eff_df :
        Wide-form effects table with ``gene`` and ``beta_*`` columns.
    value_name :
        Name for the melted value column.

    Returns
    -------
    DataFrame
        Columns: ``gene_symbol`` (when present), ``beta_col``, ``value_name``.
    """
    id_cols = ["gene_symbol"] if "gene_symbol" in eff_df.columns else []
    beta_present = [c for c in _BETA_COLS if c in eff_df.columns]
    if not beta_present:
        return pd.DataFrame(columns=id_cols + ["beta_col", value_name])
    return eff_df[id_cols + beta_present].melt(
        id_vars=id_cols,
        value_vars=beta_present,
        var_name="beta_col",
        value_name=value_name,
    )


# ── Data loading helpers ──────────────────────────────────────────────────────


def load_peak_lod(
    path: Path,
    suffix: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load a peak-LOD CSV; rename ``peak_lod`` → ``peak_lod_<suffix>``.

    Parameters
    ----------
    path :
        Path to the CSV file (typically ``peak_lod.csv``).
    suffix :
        When provided, renames the ``peak_lod`` column to ``peak_lod_<suffix>``
        to allow safe column-wise merges between sources.

    Returns
    -------
    DataFrame or None
        ``None`` when the file is absent or unreadable.
    """
    if not path.exists():
        logger.warning("missing: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        if suffix and "peak_lod" in df.columns:
            df = df.rename(columns={"peak_lod": f"peak_lod_{suffix}"})
        return df
    except Exception as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def load_allele_effects(path: Path) -> Optional[pd.DataFrame]:
    """Load an allele-effects CSV.

    Parameters
    ----------
    path :
        Path to the CSV (typically ``allele_effects.csv``).

    Returns
    -------
    DataFrame or None
        ``None`` on error.
    """
    if not path.exists():
        logger.warning("missing: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def load_gene_lod(path: Path, gene: str) -> Optional[pd.DataFrame]:
    """Load a LOD profile CSV and filter to a single gene.

    Streams the file in 1 M-row chunks to handle multi-GB ``lod.csv``
    files without materialising the full matrix.

    Parameters
    ----------
    path :
        Path to the LOD profile CSV.
    gene :
        Gene name to extract.

    Returns
    -------
    DataFrame or None
        All rows for ``gene``, or ``None`` if the file is absent or the gene
        has no rows.
    """
    if not path.exists():
        logger.debug("not found: %s", path)
        return None
    try:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=1_000_000):
            if "gene_symbol" in chunk.columns:
                hit = chunk[chunk["gene_symbol"] == gene]
                if not hit.empty:
                    chunks.append(hit.copy())
        if not chunks:
            return None
        return pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def load_threshold(py_data: Path) -> Optional[float]:
    """Read the genome-wide FWER threshold from ``py_data/threshold.csv``.

    Parameters
    ----------
    py_data :
        Directory containing ``threshold.csv``.

    Returns
    -------
    float or None
        Threshold LOD value, or ``None`` if the file is absent or unparseable.
    """
    path = py_data / "threshold.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "chr" in df.columns:
            row = df[df["chr"] == "genome_wide"]
            if not row.empty:
                return float(row["threshold"].iloc[0])
        if "threshold" in df.columns:
            return float(df["threshold"].iloc[0])
    except Exception as exc:
        logger.warning("could not read threshold from %s: %s", path, exc)
    return None


# ── Section functions ─────────────────────────────────────────────────────────


def ct_vs_whole_lod_scatter(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    whole_py_data: Path,
    ct_out_dirs: "dict[str, Path]",
    compare_dir: Path,
    palette: "dict[str, str]",
    threshold: Optional[float],
    rep_gene: Optional[str] = None,
) -> None:
    """Per-cell-type LOD scatter vs whole tissue; export quadrant gene lists.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    whole_py_data :
        Whole-tissue ``py_data/`` directory.
    ct_out_dirs :
        Mapping from cell-type label to its output plot directory.
    compare_dir :
        Shared comparison output directory.
    palette :
        Color palette from :func:`build_ct_palette`.
    threshold :
        FWER threshold LOD value for quadrant lines, or ``None``.
    rep_gene :
        Optional gene name to highlight on every scatter.
    """
    whole_df = load_peak_lod(whole_py_data / "peak_lod.csv", suffix="whole")
    if whole_df is None:
        logger.warning("ct_vs_whole_lod_scatter: whole-tissue peak_lod.csv not found")
        return

    for ct in cell_types:
        try:
            ct_df = load_peak_lod(ct_dirs[ct] / "peak_lod.csv", suffix=ct)
            if ct_df is None:
                continue
            merged = whole_df[["gene_symbol", "peak_lod_whole"]].merge(
                ct_df[["gene_symbol", f"peak_lod_{ct}"]], on="gene_symbol", how="inner"
            )
            if merged.empty:
                logger.warning("ct_vs_whole_lod_scatter: no shared genes for %s", ct)
                continue
            logger.debug(
                "ct_vs_whole_lod_scatter %s vs whole: %d shared genes", ct, len(merged)
            )
            _cmp_scatter(
                merged,
                x_col="peak_lod_whole",
                y_col=f"peak_lod_{ct}",
                x_label="Whole tissue peak LOD",
                y_label=f"{ct} peak LOD",
                threshold_x=threshold,
                threshold_y=threshold,
                priority_genes={rep_gene: INK} if rep_gene else None,
                save_path=ct_out_dirs[ct] / f"{safe_ct(ct)}_vs_whole_lod.png",
            )
            plt.close("all")
            if threshold is not None:
                _cols = ["gene_symbol", "peak_lod_whole", f"peak_lod_{ct}"]
                _w = merged["peak_lod_whole"]
                _c = merged[f"peak_lod_{ct}"]
                _quadrants = {
                    "shared_egenes_vs_whole.csv": merged[
                        (_w >= threshold) & (_c >= threshold)
                    ].sort_values(f"peak_lod_{ct}", ascending=False),
                    "ct_specific_egenes.csv": merged[
                        (_w < threshold) & (_c >= threshold)
                    ].sort_values(f"peak_lod_{ct}", ascending=False),
                    "whole_specific_egenes.csv": merged[
                        (_w >= threshold) & (_c < threshold)
                    ].sort_values("peak_lod_whole", ascending=False),
                }
                for fname, df_q in _quadrants.items():
                    out_csv = ct_dirs[ct] / fname
                    df_q[_cols].to_csv(out_csv, index=False)
                    logger.info(
                        "ct_vs_whole_lod_scatter %s: %d genes → %s",
                        ct,
                        len(df_q),
                        out_csv,
                    )
        except Exception as exc:
            logger.warning("ct_vs_whole_lod_scatter %s: %s", ct, exc, exc_info=True)


def ct_vs_whole_effects_scatter(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    whole_py_data: Path,
    ct_out_dirs: "dict[str, Path]",
) -> None:
    """Per-cell-type allele-effects scatter vs whole tissue.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    whole_py_data :
        Whole-tissue ``py_data/`` directory.
    ct_out_dirs :
        Mapping from cell-type label to its output plot directory.
    """
    whole_eff = load_allele_effects(whole_py_data / "allele_effects.csv")
    if whole_eff is None:
        logger.warning(
            "ct_vs_whole_effects_scatter: whole-tissue allele_effects.csv not found"
        )
        return

    whole_long = melt_betas(whole_eff, "beta_whole")

    for ct in cell_types:
        try:
            ct_eff = load_allele_effects(ct_dirs[ct] / "allele_effects.csv")
            if ct_eff is None:
                continue
            ct_long = melt_betas(ct_eff, f"beta_{ct}")
            if ct_long.empty:
                logger.warning(
                    "ct_vs_whole_effects_scatter: no beta columns for %s", ct
                )
                continue

            merge_cols = (
                ["gene_symbol", "beta_col"]
                if "gene_symbol" in whole_long.columns
                else ["beta_col"]
            )
            merged = whole_long[merge_cols + ["beta_whole"]].merge(
                ct_long[merge_cols + [f"beta_{ct}"]],
                on=merge_cols,
                how="inner",
            )
            if merged.empty:
                logger.warning(
                    "ct_vs_whole_effects_scatter: no overlapping data for %s", ct
                )
                continue

            founders_present = [
                f for f in _FOUNDERS_DISPLAY if f"beta_{f}" in merged["beta_col"].values
            ]
            logger.debug(
                "ct_vs_whole_effects_scatter %s vs whole: %d pairs", ct, len(merged)
            )
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            _cmp_allele_effects(
                betas_a=merged[f"beta_{ct}"].values,
                betas_b=merged["beta_whole"].values,
                founders=founders_present,
                label_a=ct,
                label_b="Whole",
                ax=ax,
            )
            fig.tight_layout()
            save_path = ct_out_dirs[ct] / f"{safe_ct(ct)}_vs_whole_effects.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
            plt.close(fig)
            logger.debug("ct_vs_whole_effects_scatter: saved %s", save_path)
        except Exception as exc:
            logger.warning("ct_vs_whole_effects_scatter %s: %s", ct, exc, exc_info=True)


def ct_lod_profile_overlay(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    whole_py_data: Path,
    compare_dir: Path,
    palette: "dict[str, str]",
    threshold: Optional[float],
    rep_gene: str,
) -> None:
    """LOD profile overlay for a representative gene across cell types and whole tissue.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    whole_py_data :
        Whole-tissue ``py_data/`` directory.
    compare_dir :
        Shared comparison output directory.
    palette :
        Color palette from :func:`build_ct_palette`.
    threshold :
        FWER threshold LOD value, or ``None``.
    rep_gene :
        Gene name for which to overlay LOD profiles.
    """
    whole_peak = load_peak_lod(whole_py_data / "peak_lod.csv")
    peak_chr: Optional[str] = None
    if whole_peak is not None and "gene_symbol" in whole_peak.columns:
        row = whole_peak[whole_peak["gene_symbol"] == rep_gene]
        if not row.empty:
            peak_chr = str(row["chr"].iloc[0])
    if peak_chr is None:
        logger.warning(
            "ct_lod_profile_overlay: %s not in whole-tissue peak_lod", rep_gene
        )

    def _get_trace(
        lod_path: Path,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Load (positions, lod) for rep_gene from lod_path; None if missing."""
        df = load_gene_lod(lod_path, rep_gene)
        if df is None or df.empty:
            return None
        if peak_chr is not None and "chr" in df.columns:
            df = df[df["chr"].astype(str) == peak_chr]
        if df.empty:
            return None
        df = df.sort_values("pos_mb")
        return df["pos_mb"].values, df["lod"].values

    whole_trace = _get_trace(whole_py_data / "lod.csv")

    ct_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ct in cell_types:
        try:
            tr = _get_trace(ct_dirs[ct] / "lod.csv")
            if tr is not None:
                ct_traces[ct] = tr
            else:
                logger.warning(
                    "ct_lod_profile_overlay: no LOD profile for gene=%s ct=%s",
                    rep_gene,
                    ct,
                )
        except Exception as exc:
            logger.warning("ct_lod_profile_overlay trace load %s: %s", ct, exc)

    if whole_trace is None and not ct_traces:
        logger.warning("ct_lod_profile_overlay: no data available for %s", rep_gene)
        return

    ct_list = list(ct_traces.keys())
    pairs: list[tuple[str, Optional[tuple], str, Optional[tuple]]] = []
    if ct_list:
        pairs.append(("Whole", whole_trace, ct_list[0], ct_traces[ct_list[0]]))
        for ct in ct_list[1:]:
            pairs.append(("Whole", whole_trace, ct, ct_traces[ct]))
    elif whole_trace is not None:
        pairs.append(("Whole", whole_trace, "", None))

    for idx, (label_a, trace_a, label_b, trace_b) in enumerate(pairs):
        try:
            fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
            pos_a: np.ndarray = trace_a[0] if trace_a is not None else np.array([])
            lod_a: np.ndarray = trace_a[1] if trace_a is not None else np.array([])
            pos_b: Optional[np.ndarray] = trace_b[0] if trace_b is not None else None
            lod_b: Optional[np.ndarray] = trace_b[1] if trace_b is not None else None
            color_a = INK if label_a == "Whole" else palette.get(label_a, MUTED)
            color_b = palette.get(label_b, MUTED) if label_b else MUTED
            _cmp_lod_overlay(
                pos_a=pos_a,
                lod_a=lod_a,
                pos_b=pos_b,
                lod_b=lod_b,
                label_a=label_a,
                label_b=label_b if label_b else "B",
                color_a=color_a,
                color_b=color_b,
                chrom=peak_chr or "",
                gene=rep_gene,
                fwer_thr=threshold,
                ax=ax,
            )
            fig.tight_layout()
            if idx == 0:
                save_name = f"lod_overlay_{rep_gene}.png"
            else:
                safe_label = label_b.replace(" ", "_")
                save_name = f"lod_overlay_{rep_gene}_{safe_label}.png"
            save_path = compare_dir / save_name
            fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
            plt.close(fig)
            logger.debug("ct_lod_profile_overlay: saved %s", save_path)
        except Exception as exc:
            logger.warning(
                "ct_lod_profile_overlay pair %s/%s: %s",
                label_a,
                label_b,
                exc,
                exc_info=True,
            )

    logger.info("ct_lod_profile_overlay: %d figures for %s", len(pairs), rep_gene)


def ct_pairwise_lod_scatter(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    ct_out_dirs: "dict[str, Path]",
    compare_dir: Path,
    threshold: Optional[float],
    rep_gene: Optional[str] = None,
) -> None:
    """Pairwise cell-type LOD scatter + lower-triangle summary grid.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    ct_out_dirs :
        Mapping from cell-type label to its output plot directory.
    compare_dir :
        Shared comparison output directory.
    threshold :
        FWER threshold LOD value, or ``None``.
    rep_gene :
        Optional gene name to highlight on every scatter.
    """
    ct_frames: dict[str, Optional[pd.DataFrame]] = {
        ct: load_peak_lod(ct_dirs[ct] / "peak_lod.csv", suffix=ct) for ct in cell_types
    }

    pairs: list[tuple[str, str]] = []
    for i, ct_a in enumerate(cell_types):
        for ct_b in cell_types[i + 1 :]:  # noqa: E203
            pairs.append(pair_order(ct_a, ct_b))

    for x_ct, y_ct in pairs:
        try:
            df_x = ct_frames.get(x_ct)
            df_y = ct_frames.get(y_ct)
            if df_x is None or df_y is None:
                logger.warning(
                    "ct_pairwise_lod_scatter: missing data for %s / %s", x_ct, y_ct
                )
                continue
            merged = df_x[["gene_symbol", f"peak_lod_{x_ct}"]].merge(
                df_y[["gene_symbol", f"peak_lod_{y_ct}"]], on="gene_symbol", how="inner"
            )
            if merged.empty:
                logger.warning(
                    "ct_pairwise_lod_scatter: no shared genes %s / %s", x_ct, y_ct
                )
                continue
            logger.debug(
                "ct_pairwise_lod_scatter %s vs %s: %d genes", x_ct, y_ct, len(merged)
            )
            _cmp_scatter(
                merged,
                x_col=f"peak_lod_{x_ct}",
                y_col=f"peak_lod_{y_ct}",
                x_label=f"{x_ct} peak LOD",
                y_label=f"{y_ct} peak LOD",
                threshold_x=threshold,
                threshold_y=threshold,
                priority_genes={rep_gene: INK} if rep_gene else None,
                save_path=compare_dir / f"{safe_ct(x_ct)}_vs_{safe_ct(y_ct)}_lod.png",
            )
            plt.close("all")
            if threshold is not None:
                _cols = ["gene_symbol", f"peak_lod_{x_ct}", f"peak_lod_{y_ct}"]
                _x = merged[f"peak_lod_{x_ct}"]
                _y = merged[f"peak_lod_{y_ct}"]
                _quadrants = {
                    (x_ct, f"shared_egenes_vs_{safe_ct(y_ct)}.csv"): merged[
                        (_x >= threshold) & (_y >= threshold)
                    ].sort_values(f"peak_lod_{x_ct}", ascending=False),
                    (x_ct, f"specific_egenes_vs_{safe_ct(y_ct)}.csv"): merged[
                        (_x >= threshold) & (_y < threshold)
                    ].sort_values(f"peak_lod_{x_ct}", ascending=False),
                    (y_ct, f"specific_egenes_vs_{safe_ct(x_ct)}.csv"): merged[
                        (_x < threshold) & (_y >= threshold)
                    ].sort_values(f"peak_lod_{y_ct}", ascending=False),
                }
                for (owner_ct, fname), df_q in _quadrants.items():
                    out_csv = ct_dirs[owner_ct] / fname
                    df_q[_cols].to_csv(out_csv, index=False)
                    logger.info(
                        "ct_pairwise_lod_scatter %s/%s: %d genes → %s",
                        x_ct,
                        y_ct,
                        len(df_q),
                        out_csv,
                    )
        except Exception as exc:
            logger.warning(
                "ct_pairwise_lod_scatter %s/%s: %s", x_ct, y_ct, exc, exc_info=True
            )

    n = len(cell_types)
    if n < 2:
        return
    try:
        fig = plt.figure(figsize=(4 * n, 4 * n), dpi=300)
        gs = GridSpec(n, n, figure=fig, hspace=0.45, wspace=0.45)
        for i, ct_y in enumerate(cell_types):
            for j, ct_x in enumerate(cell_types):
                if j >= i:
                    continue
                try:
                    x_ct, y_ct = pair_order(ct_x, ct_y)
                    df_x = ct_frames.get(x_ct)
                    df_y = ct_frames.get(y_ct)
                    if df_x is None or df_y is None:
                        continue
                    merged = df_x[["gene_symbol", f"peak_lod_{x_ct}"]].merge(
                        df_y[["gene_symbol", f"peak_lod_{y_ct}"]],
                        on="gene_symbol",
                        how="inner",
                    )
                    if merged.empty:
                        continue
                    ax = fig.add_subplot(gs[i, j])
                    _cmp_lod_density(
                        lods_a=merged[f"peak_lod_{x_ct}"].values,
                        lods_b=merged[f"peak_lod_{y_ct}"].values,
                        label_a=x_ct,
                        label_b=y_ct,
                        threshold=threshold,
                        ax=ax,
                    )
                    ax.set_title(f"{x_ct} vs {y_ct}", fontsize=9, fontweight="bold")
                except Exception as exc:
                    logger.warning(
                        "ct_pairwise_lod_scatter grid [%d,%d]: %s", i, j, exc
                    )
        fig.suptitle(
            "Pairwise cell-type LOD distributions",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        grid_path = compare_dir / "pairwise_lod_grid.png"
        fig.savefig(grid_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        logger.info("ct_pairwise_lod_scatter: saved grid %s", grid_path)
    except Exception as exc:
        logger.warning("ct_pairwise_lod_scatter grid failed: %s", exc, exc_info=True)


def ct_pairwise_effects_scatter(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    ct_out_dirs: "dict[str, Path]",
) -> None:
    """Pairwise cell-type allele-effects scatter.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    ct_out_dirs :
        Mapping from cell-type label to its output plot directory.
    """
    ct_effects: dict[str, Optional[pd.DataFrame]] = {
        ct: load_allele_effects(ct_dirs[ct] / "allele_effects.csv") for ct in cell_types
    }

    pairs: list[tuple[str, str]] = []
    for i, ct_a in enumerate(cell_types):
        for ct_b in cell_types[i + 1 :]:  # noqa: E203
            pairs.append(pair_order(ct_a, ct_b))

    for x_ct, y_ct in pairs:
        try:
            eff_x = ct_effects.get(x_ct)
            eff_y = ct_effects.get(y_ct)
            if eff_x is None or eff_y is None:
                logger.warning(
                    "ct_pairwise_effects_scatter: missing for %s / %s", x_ct, y_ct
                )
                continue

            long_x = melt_betas(eff_x, f"beta_{x_ct}")
            long_y = melt_betas(eff_y, f"beta_{y_ct}")
            if long_x.empty or long_y.empty:
                logger.warning(
                    "ct_pairwise_effects_scatter: no betas for %s / %s", x_ct, y_ct
                )
                continue

            merge_cols = (
                ["gene_symbol", "beta_col"]
                if "gene_symbol" in long_x.columns and "gene_symbol" in long_y.columns
                else ["beta_col"]
            )
            merged = long_x[merge_cols + [f"beta_{x_ct}"]].merge(
                long_y[merge_cols + [f"beta_{y_ct}"]],
                on=merge_cols,
                how="inner",
            )
            if merged.empty:
                logger.warning(
                    "ct_pairwise_effects_scatter: no overlap %s / %s", x_ct, y_ct
                )
                continue

            founders_present = [
                f
                for f in _FOUNDERS_DISPLAY
                if f"beta_{f}" in merged.get("beta_col", pd.Series(dtype=str)).values
            ]
            logger.debug(
                "ct_pairwise_effects_scatter %s vs %s: %d pairs",
                x_ct,
                y_ct,
                len(merged),
            )
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            _cmp_allele_effects(
                betas_a=merged[f"beta_{x_ct}"].values,
                betas_b=merged[f"beta_{y_ct}"].values,
                founders=founders_present,
                label_a=x_ct,
                label_b=y_ct,
                ax=ax,
            )
            fig.tight_layout()
            save_path = (
                ct_out_dirs[x_ct] / f"{safe_ct(x_ct)}_vs_{safe_ct(y_ct)}_effects.png"
            )
            fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
            plt.close(fig)
            logger.debug("ct_pairwise_effects_scatter: saved %s", save_path)
        except Exception as exc:
            logger.warning(
                "ct_pairwise_effects_scatter %s/%s: %s", x_ct, y_ct, exc, exc_info=True
            )


def ct_lod_distributions(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    whole_py_data: Path,
    compare_dir: Path,
    palette: "dict[str, str]",
    threshold: Optional[float],
) -> None:
    """Per-cell-type LOD density vs whole tissue; multi-panel summary grid.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    whole_py_data :
        Whole-tissue ``py_data/`` directory.
    compare_dir :
        Shared comparison output directory.
    palette :
        Color palette from :func:`build_ct_palette`.
    threshold :
        FWER threshold LOD value, or ``None``.
    """
    whole_df = load_peak_lod(whole_py_data / "peak_lod.csv", suffix="whole")
    whole_lods: Optional[np.ndarray] = (
        whole_df["peak_lod_whole"].values
        if whole_df is not None and "peak_lod_whole" in whole_df.columns
        else None
    )

    for ct in cell_types:
        try:
            ct_df = load_peak_lod(ct_dirs[ct] / "peak_lod.csv", suffix=ct)
            if ct_df is None:
                continue
            ct_col = f"peak_lod_{ct}"
            if ct_col not in ct_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
            _cmp_lod_density(
                lods_a=whole_lods,
                lods_b=ct_df[ct_col].values,
                label_a="Whole",
                label_b=ct,
                color_a=INK,
                color_b=palette.get(ct, MUTED),
                threshold=threshold,
                ax=ax,
            )
            ax.set_title(
                f"{ct} vs Whole — peak LOD distribution", fontsize=11, fontweight="bold"
            )
            fig.tight_layout()
            save_path = compare_dir / f"{safe_ct(ct)}_vs_whole_lod_density.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
            plt.close(fig)
            logger.debug("ct_lod_distributions: saved %s", save_path)
        except Exception as exc:
            logger.warning("ct_lod_distributions %s: %s", ct, exc, exc_info=True)

    n_ct = len(cell_types)
    if n_ct < 1:
        return
    try:
        ncols = min(3, n_ct)
        nrows = (n_ct + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), dpi=300)
        axes_flat: list[plt.Axes] = (
            [axes] if n_ct == 1 else list(np.array(axes).ravel())  # type: ignore[arg-type]
        )
        for ax in axes_flat:
            ax.set_visible(False)
        for i, ct in enumerate(cell_types):
            try:
                ct_df = load_peak_lod(ct_dirs[ct] / "peak_lod.csv", suffix=ct)
                if ct_df is None:
                    continue
                ct_col = f"peak_lod_{ct}"
                if ct_col not in ct_df.columns:
                    continue
                ax = axes_flat[i]
                ax.set_visible(True)
                _cmp_lod_density(
                    lods_a=whole_lods,
                    lods_b=ct_df[ct_col].values,
                    label_a="Whole",
                    label_b=ct,
                    color_a=INK,
                    color_b=palette.get(ct, MUTED),
                    threshold=threshold,
                    ax=ax,
                )
                ax.set_title(ct, fontsize=10, fontweight="bold")
            except Exception as exc:
                logger.warning("ct_lod_distributions panel %s: %s", ct, exc)
        fig.suptitle(
            "Peak LOD distributions — each cell type vs whole tissue",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()
        summary_path = compare_dir / "lod_distributions.png"
        fig.savefig(summary_path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        logger.info("ct_lod_distributions: saved summary %s", summary_path)
    except Exception as exc:
        logger.warning("ct_lod_distributions summary failed: %s", exc, exc_info=True)


def ct_cis_distance_comparison(
    cell_types: list[str],
    ct_dirs: "dict[str, Path]",
    whole_py_data: Path,
    compare_dir: Path,
    palette: "dict[str, str]",
) -> None:
    """Signed cis-distance violin + proximal-fraction curves across sources.

    Parameters
    ----------
    cell_types :
        Cell-type labels to process.
    ct_dirs :
        Mapping from cell-type label to its ``py_data/`` directory.
    whole_py_data :
        Whole-tissue ``py_data/`` directory.
    compare_dir :
        Shared comparison output directory.
    palette :
        Color palette from :func:`build_ct_palette`.
    """
    sources: dict[str, tuple] = {}

    def _tss_col(df: pd.DataFrame) -> str:
        """Return the TSS column name present in df (near_tss_mb preferred)."""
        return "near_tss_mb" if "near_tss_mb" in df.columns else "start_mb"

    w_path = whole_py_data / "peak_lod.csv"
    if w_path.exists():
        _wdf = pd.read_csv(w_path)
        _wtss = _tss_col(_wdf)
        if {_wtss, "peak_mb", "peak_lod"}.issubset(_wdf.columns):
            _wdf = _wdf.dropna(subset=[_wtss, "peak_mb", "peak_lod"])
            sources["whole"] = (
                _wdf["peak_lod"].values,
                _wdf[_wtss].values,
                _wdf["peak_mb"].values,
            )

    for ct in cell_types:
        ct_path = ct_dirs[ct] / "peak_lod.csv"
        if not ct_path.exists():
            continue
        _cdf = pd.read_csv(ct_path)
        _ctss = _tss_col(_cdf)
        if not {_ctss, "peak_mb", "peak_lod"}.issubset(_cdf.columns):
            continue
        _cdf = _cdf.dropna(subset=[_ctss, "peak_mb", "peak_lod"])
        sources[ct] = (
            _cdf["peak_lod"].values,
            _cdf[_ctss].values,
            _cdf["peak_mb"].values,
        )

    if len(sources) < 2:
        logger.warning("ct_cis_distance_comparison: fewer than 2 sources — skipping")
        return

    pal = {**{"whole": MUTED}, **palette}
    fig = _cmp_cis_distance(sources, palette=pal, zoom_mb=1.0)
    out = compare_dir / "cis_distance_comparison.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    logger.info("ct_cis_distance_comparison: %s  (%d sources)", out, len(sources))
