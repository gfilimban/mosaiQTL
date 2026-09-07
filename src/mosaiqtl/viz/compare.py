"""plots that compare two sets of results side by side.

every function here takes two named series — ``label_a``/``label_b`` — and
draws them together (scatter, overlaid histograms, or overlaid profiles) so
the two can be checked for agreement. the usual pairing is a Python run
(series A, blue) against a reference run in another language (series B,
red), but any two comparable result sets work.

all colors come from ``mosaiqtl.genotype.founders`` — colors are never
hardcoded here.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from scipy.stats import gaussian_kde

from ..genotype.founders import (
    COLOR_CC_FALLBACK,
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_THRESHOLD,
    FOUNDER_COLORS,
)
from ._theme import WASH

# Default comparison colors — series A (blue) vs series B (red).
_COLOR_A: str = FOUNDER_COLORS["NODJ"]  # series A: blue (#4F6EAF)
_COLOR_B: str = FOUNDER_COLORS["PWKJ"]  # series B: red  (#D83026)

_CMAP = "crest"  # color scale for LOD-score scatter points
_CMAP_DEVIATION = "YlOrRd"  # color scale for how far two allele effects disagree


def _set_style() -> None:
    """apply consistent plot style."""
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Carlito", "Calibri", "DejaVu Sans"],
            "font.size": 11,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# joint scatter (LOD × LOD with KDE marginals)
# ─────────────────────────────────────────────────────────────────────────────


def lod_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    *,
    cbar_label: str = "peak LOD score",
    priority_genes: Optional[dict[str, str]] = None,
    sparse_mask: Optional[pd.Series] = None,
    threshold: Optional[float] = None,
    threshold_x: Optional[float] = None,
    threshold_y: Optional[float] = None,
    quadrant_color: str = COLOR_THRESHOLD,
    save_path: Optional[object] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """scatter of one gene's LOD score in series A vs series B, with density curves on each axis.

    a dashed diagonal marks perfect agreement (y = x), and a box reports the
    correlation. when a threshold is supplied (via *threshold* for a shared
    cutoff, or *threshold_x*/*threshold_y* for two different cutoffs — e.g.
    comparing a cell-type scan against a whole-tissue scan, which can have
    different significance thresholds), dotted lines split the plot into
    four quadrants and each quadrant is labeled with how many genes fall in
    it (e.g. genes significant in both series vs. only one).

    Args:
        df: merged data containing *x_col* and *y_col*.
        x_col: column name for the x axis.
        y_col: column name for the y axis.
        x_label: x-axis label text (should read "... LOD score").
        y_label: y-axis label text (should read "... LOD score").
        cbar_label: colorbar label (default ``'peak LOD score'``).
        priority_genes: ``{gene_symbol: color_hex}`` dict; matching genes are
            drawn bigger, labeled, and on top of the other points. requires a
            ``'gene_symbol'`` column in *df*.
        sparse_mask: boolean Series; ``True`` rows are drawn grey (collapsed
            loci), ``False`` rows are colored by peak LOD score.
        threshold: genome-wide FWER probability threshold applied to both
            axes when *threshold_x*/*threshold_y* are not given individually.
        threshold_x: FWER probability threshold for the x-axis series;
            overrides *threshold* on that axis only. required together with
            *threshold_y* (or *threshold*) to draw quadrant lines.
        threshold_y: FWER probability threshold for the y-axis series;
            overrides *threshold* on that axis only.
        quadrant_color: hex color for the threshold lines and gene-count
            annotation boxes (default ``'#E8870A'`` orange).
        save_path: file path to save the figure; skipped when ``None``.

    Returns:
        tuple of ``(Figure, joint Axes)``.
    """
    import logging

    logger = logging.getLogger("mosaiqtl")

    _set_style()

    r_pearson, _ = sp_stats.pearsonr(df[x_col], df[y_col])
    r_spearman, _ = sp_stats.spearmanr(df[x_col], df[y_col])
    r2 = r_pearson**2

    df = df.copy()
    df["_lod_mean"] = (df[x_col] + df[y_col]) / 2

    g = sns.JointGrid(data=df, x=x_col, y=y_col, height=6)

    if sparse_mask is not None:
        sns.scatterplot(
            data=df[sparse_mask],
            x=x_col,
            y=y_col,
            color=COLOR_CC_FALLBACK,
            s=4,
            alpha=0.3,
            linewidths=0,
            rasterized=True,
            ax=g.ax_joint,
            legend=False,
        )
        sns.scatterplot(
            data=df[~sparse_mask],
            x=x_col,
            y=y_col,
            hue="_lod_mean",
            palette=_CMAP,
            s=5,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            ax=g.ax_joint,
            legend=False,
        )
    else:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="_lod_mean",
            palette=_CMAP,
            s=5,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            ax=g.ax_joint,
            legend=False,
        )

    sns.kdeplot(
        data=df, x=x_col, ax=g.ax_marg_x, fill=True, color=COLOR_CC_FALLBACK, alpha=0.4
    )
    sns.kdeplot(
        data=df, y=y_col, ax=g.ax_marg_y, fill=True, color=COLOR_CC_FALLBACK, alpha=0.4
    )

    _lim = max(df[x_col].max(), df[y_col].max()) * 1.05
    g.ax_joint.set_xlim(0, _lim)
    g.ax_joint.set_ylim(0, _lim)
    g.ax_joint.plot(
        [0, _lim],
        [0, _lim],
        color=COLOR_NEUTRAL_DARK,
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
    )

    if priority_genes and "gene_symbol" in df.columns:
        for gene, color in priority_genes.items():
            sub = df[df["gene_symbol"] == gene]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                g.ax_joint.annotate(
                    gene,
                    (row[x_col], row[y_col]),
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
            sns.scatterplot(
                data=sub,
                x=x_col,
                y=y_col,
                s=60,
                color=color,
                zorder=5,
                ax=g.ax_joint,
                legend=False,
            )

    # ── threshold lines + per-quadrant gene counts ────────────────────────────
    # threshold_x/threshold_y (per-axis) take priority over the shared
    # threshold; if neither is set, no threshold lines are drawn at all.
    _thr_x: Optional[float] = threshold_x if threshold_x is not None else threshold
    _thr_y: Optional[float] = threshold_y if threshold_y is not None else threshold

    if _thr_x is not None and _thr_y is not None:
        ax_j = g.ax_joint
        ax_j.axvline(_thr_x, color=quadrant_color, lw=1.4, ls=":", zorder=4)
        ax_j.axhline(_thr_y, color=quadrant_color, lw=1.4, ls=":", zorder=4)
        # repeat the threshold line on the marginal density panels too
        g.ax_marg_x.axvline(_thr_x, color=quadrant_color, lw=1.2, ls=":", alpha=0.8)
        g.ax_marg_y.axhline(_thr_y, color=quadrant_color, lw=1.2, ls=":", alpha=0.8)

        x_vals = df[x_col].values
        y_vals = df[y_col].values
        _box_kw = dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=quadrant_color,
            linewidth=1.5,
            alpha=0.92,
        )
        # top-right: significant in both series
        n_tr = int(np.sum((x_vals >= _thr_x) & (y_vals >= _thr_y)))
        ax_j.text(
            0.97,
            0.97,
            f"{n_tr:,} eGenes",
            transform=ax_j.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=quadrant_color,
            bbox=_box_kw,
            zorder=5,
        )
        # top-left: significant only in the y-axis series
        n_tl = int(np.sum((x_vals < _thr_x) & (y_vals >= _thr_y)))
        ax_j.text(
            0.03,
            0.97,
            f"{n_tl:,} eGenes",
            transform=ax_j.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=quadrant_color,
            bbox=_box_kw,
            zorder=5,
        )
        # bottom-right: significant only in the x-axis series
        n_br = int(np.sum((x_vals >= _thr_x) & (y_vals < _thr_y)))
        ax_j.text(
            0.97,
            0.03,
            f"{n_br:,} eGenes",
            transform=ax_j.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=quadrant_color,
            bbox=_box_kw,
            zorder=5,
        )
        # bottom-left: not significant in either series
        n_bl = int(np.sum((x_vals < _thr_x) & (y_vals < _thr_y)))
        ax_j.text(
            0.03,
            0.03,
            f"{n_bl:,} eGenes",
            transform=ax_j.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=quadrant_color,
            bbox=_box_kw,
            zorder=5,
        )

    # stats box sits top-left; nudged down when quadrant counts are also shown
    _stats_va = "top"
    _stats_y = 0.95 if (_thr_x is None or _thr_y is None) else 0.80
    stats_text = f"r²   = {r2:.4f}\n" f"r_sp = {r_spearman:.4f}\n" f"n    = {len(df):,}"
    g.ax_joint.text(
        0.05,
        _stats_y,
        stats_text,
        transform=g.ax_joint.transAxes,
        fontsize=8,
        verticalalignment=_stats_va,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.85,
            edgecolor=COLOR_NEUTRAL_DARK,
        ),
    )

    g.ax_joint.set_xlabel(x_label, fontsize=13, fontweight="bold")
    g.ax_joint.set_ylabel(y_label, fontsize=13, fontweight="bold")
    g.ax_joint.tick_params(labelsize=11)

    norm = plt.Normalize(df["_lod_mean"].min(), df["_lod_mean"].max())
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette(_CMAP, as_cmap=True), norm=norm)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([0.312, -0.02, 0.3, 0.02])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    sns.despine(ax=g.ax_joint)
    g.ax_joint.grid(
        True, alpha=0.15, linewidth=0.4, color=COLOR_NEUTRAL_LIGHT, zorder=0
    )
    g.fig.subplots_adjust(bottom=0.1)

    if save_path:
        g.fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        logger.info("saved: %s", save_path)
    return g.fig, g.ax_joint


#: backward-compatible alias.
joint_lod_scatter = lod_scatter


# ─────────────────────────────────────────────────────────────────────────────
# LOD density overlay
# ─────────────────────────────────────────────────────────────────────────────


def lod_density_comparison(
    lods_a: Optional[np.ndarray],
    lods_b: Optional[np.ndarray],
    *,
    label_a: str = "A",
    label_b: str = "B",
    color_a: str = _COLOR_A,
    color_b: str = _COLOR_B,
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """overlay the LOD-score distributions of two series as histograms with smoothed curves.

    Args:
        lods_a: series-A peak-LOD values; skipped when ``None`` or empty.
        lods_b: series-B peak-LOD values; skipped when ``None`` or empty.
        label_a: legend label for series A (default ``'A'``).
        label_b: legend label for series B (default ``'B'``).
        color_a: hex color for series A (default blue).
        color_b: hex color for series B (default red).
        threshold: FWER probability threshold drawn as a dotted vertical line when provided.
        ax: existing Axes to draw on; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    _sources = [
        (lods_a, label_a, color_a),
        (lods_b, label_b, color_b),
    ]
    for vals, label, color in _sources:
        if vals is None or len(vals) < 2:
            continue
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        sns.histplot(
            vals,
            bins=40,
            stat="density",
            alpha=0.35,
            color=color,
            label=f"{label} (n={len(vals):,})",
            ax=ax,
        )
        xr = np.linspace(0, vals.max() * 1.1, 300)
        try:
            kde = gaussian_kde(vals, bw_method=0.3)
            sns.lineplot(x=xr, y=kde(xr), color=color, lw=1.5, ax=ax)
        except Exception:
            pass
        med = float(np.median(vals))
        ax.axvline(
            med,
            color=color,
            lw=1,
            ls="--",
            alpha=0.6,
            label=f"{label} median={med:.2f}",
        )

    if threshold is not None:
        ax.axvline(
            threshold,
            color=COLOR_NEUTRAL_DARK,
            lw=1.4,
            ls=":",
            label=f"FWER thr={threshold:.2f}",
        )

    ax.set_xlabel("Peak LOD Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# allele effects scatter
# ─────────────────────────────────────────────────────────────────────────────


def allele_effects_comparison(
    betas_a: np.ndarray,
    betas_b: np.ndarray,
    founders: list[str],
    *,
    label_a: str = "A (per-haplotype)",
    label_b: str = "B (per-allele)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """scatter of per-founder betas from two series, colored by ``|A − 2B|``.

    note: the ``|A − 2B|`` deviation metric assumes the canonical Python/R
    2× scale convention where Python stores per-haplotype effects (0-1 scale)
    and R stores per-allele effects (0-2 scale). pass pre-scaled betas and
    ignore the colorbar when comparing two same-scale series.

    Args:
        betas_a: series-A allele effects, shape ``(n,)``.
        betas_b: series-B allele effects, shape ``(n,)``.
        founders: founder display-name list retained for API consistency
            (not used in the scatter body; callers use it for titles).
        label_a: legend / axis label for series A.
        label_b: legend / axis label for series B.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), dpi=300)

    b_a = np.asarray(betas_a, dtype=float)
    b_b = np.asarray(betas_b, dtype=float)
    mask = np.isfinite(b_a) & np.isfinite(b_b)
    a_m, b_m = b_a[mask], b_b[mask]

    dev = np.abs(a_m - 2.0 * b_m)
    sc = ax.scatter(
        b_m,
        a_m,
        c=dev,
        cmap=_CMAP_DEVIATION,
        alpha=0.4,
        s=5,
        linewidths=0,
        rasterized=True,
    )
    cbar = ax.get_figure().colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"|{label_a.split()[0]} − 2×{label_b.split()[0]}|", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    lim = float(np.abs(np.concatenate([b_m, a_m])).max()) * 1.05
    sns.lineplot(
        x=[-lim, lim],
        y=[-2 * lim, 2 * lim],
        color=COLOR_NEUTRAL_DARK,
        linewidth=0.8,
        linestyle="--",
        label="y = 2x (expected)",
        errorbar=None,
        ax=ax,
    )
    sns.lineplot(
        x=[-lim, lim],
        y=[-lim, lim],
        color=COLOR_CC_FALLBACK,
        linewidth=0.5,
        linestyle="-",
        alpha=0.5,
        label="y = x",
        errorbar=None,
        ax=ax,
    )

    r2_val = (
        float(np.corrcoef(a_m, b_m)[0, 1]) ** 2 if mask.sum() >= 2 else float("nan")
    )

    ax.text(
        0.98,
        0.97,
        f"r² = {r2_val:.4f}\nnote: A=2×B scale",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontfamily="monospace",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=COLOR_NEUTRAL_DARK,
            alpha=0.88,
        ),
    )

    ax.set_xlabel(f"{label_b} allele effect (RINT)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{label_a} allele effect (RINT)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# stacked LOD profile panels
# ─────────────────────────────────────────────────────────────────────────────


def lod_profiles_comparison(
    df_a: pd.DataFrame,
    df_b: Optional[pd.DataFrame],
    genes: list[str],
    chrom: str,
    *,
    label_a: str = "A",
    label_b: str = "B",
    color_a: str = _COLOR_A,
    color_b: str = _COLOR_B,
    thr: Optional[float] = None,
) -> plt.Figure:
    """stacked panels (one per gene) showing LOD traces for two series.

    Args:
        df_a: series-A LOD data with columns ``[gene, pos_mb, lod]``.
        df_b: series-B LOD data with columns ``[gene, pos_mb, lod]``; skipped
            when ``None`` or when a gene is absent.
        genes: gene names to include (one panel per gene).
        chrom: chromosome label for the shared x-axis title.
        label_a: legend label for series A (default ``'A'``).
        label_b: legend label for series B (default ``'B'``).
        color_a: hex color for series A.
        color_b: hex color for series B.
        thr: FWER probability threshold drawn as a dotted horizontal line when provided.

    Returns:
        the Figure containing the stacked panels.
    """
    _set_style()
    n = len(genes)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True, dpi=300)
    if n == 1:
        axes = [axes]

    for ax, gene in zip(axes, genes):
        g_a = df_a[df_a["gene_symbol"] == gene].sort_values("pos_mb")
        ax.fill_between(g_a["pos_mb"], g_a["lod"], alpha=0.15, color=color_a, zorder=1)
        sns.lineplot(
            x=g_a["pos_mb"],
            y=g_a["lod"],
            color=color_a,
            lw=1.2,
            label=label_a,
            ax=ax,
            zorder=3,
        )

        if df_b is not None and gene in df_b["gene_symbol"].values:
            g_b = df_b[df_b["gene_symbol"] == gene].sort_values("pos_mb")
            sns.lineplot(
                x=g_b["pos_mb"],
                y=g_b["lod"],
                color=color_b,
                lw=1,
                linestyle="--",
                label=label_b,
                alpha=0.85,
                ax=ax,
                zorder=2,
            )

        if thr is not None:
            ax.axhline(
                thr, color=COLOR_NEUTRAL_DARK, ls=":", lw=0.8, alpha=0.6, zorder=0
            )

        ax.text(
            0.02,
            0.95,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_a,
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("LOD Score", fontsize=11)
        if ax is axes[0]:
            ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
        sns.despine(ax=ax)

    axes[-1].set_xlabel(f"Chr {chrom} position (Mb)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOD profile overlay (single panel, arbitrary-length position arrays)
# ─────────────────────────────────────────────────────────────────────────────


def lod_profile_overlay(
    pos_a: np.ndarray,
    lod_a: np.ndarray,
    pos_b: Optional[np.ndarray] = None,
    lod_b: Optional[np.ndarray] = None,
    *,
    label_a: str = "A",
    label_b: str = "B",
    color_a: str = _COLOR_A,
    color_b: str = _COLOR_B,
    chrom: str = "",
    gene: str = "",
    fwer_thr: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """single-panel LOD profile with an optional second overlay trace.

    Args:
        pos_a: genomic positions (Mb) for series A, shape ``(n_loci,)``.
        lod_a: LOD scores for series A, shape ``(n_loci,)``.
        pos_b: positions for series B; omitted when ``None``.
        lod_b: LOD scores for series B; omitted when ``None``.
        label_a: legend label for series A.
        label_b: legend label for series B.
        color_a: hex color for series A.
        color_b: hex color for series B.
        chrom: chromosome label appended to the x-axis title.
        gene: gene name shown as a bold top-left annotation.
        fwer_thr: FWER probability threshold drawn as a horizontal line when provided.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

    ax.fill_between(pos_a, lod_a, alpha=0.12, color=color_a, zorder=1)
    sns.lineplot(
        x=pos_a, y=lod_a, color=color_a, lw=1.4, label=label_a, ax=ax, zorder=3
    )

    if pos_b is not None and lod_b is not None:
        sns.lineplot(
            x=pos_b,
            y=lod_b,
            color=color_b,
            lw=1.0,
            linestyle="--",
            label=label_b,
            alpha=0.85,
            ax=ax,
            zorder=2,
        )

    if fwer_thr is not None:
        ax.axhline(
            fwer_thr,
            color=COLOR_NEUTRAL_DARK,
            ls=":",
            lw=0.8,
            alpha=0.6,
            label=f"FWER {fwer_thr:.2f}",
        )

    if gene:
        ax.text(
            0.02,
            0.97,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_a,
        )

    ax.set_ylim(bottom=0)
    xlabel = f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)"
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# kinship heatmap comparison
# ─────────────────────────────────────────────────────────────────────────────


def kinship_comparison(
    k_a: np.ndarray,
    k_b: np.ndarray,
    *,
    label_a: str = "A",
    label_b: str = "B",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """side-by-side heatmaps and element-wise scatter for two kinship matrices.

    Args:
        k_a: kinship matrix for series A, shape ``(m, m)``.
        k_b: kinship matrix for series B, shape ``(m, m)``.
        label_a: label for series A (default ``'A'``).
        label_b: label for series B (default ``'B'``).
        ax: existing Axes for the scatter panel; creates a new figure if ``None``.
            when provided the figure is expected to already contain heat-map
            panels alongside it — otherwise only the scatter is drawn.

    Returns:
        the scatter Axes.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), dpi=300)

    flat_a = k_a.ravel()
    flat_b = k_b.ravel()
    mask = np.isfinite(flat_a) & np.isfinite(flat_b)

    sns.scatterplot(
        x=flat_b[mask],
        y=flat_a[mask],
        s=2,
        alpha=0.3,
        color=_COLOR_A,
        linewidths=0,
        rasterized=True,
        ax=ax,
    )

    lim = max(flat_a[mask].max(), flat_b[mask].max()) * 1.05
    sns.lineplot(
        x=[0, lim],
        y=[0, lim],
        color=COLOR_NEUTRAL_DARK,
        linewidth=0.8,
        linestyle="--",
        errorbar=None,
        ax=ax,
    )

    r2_val = float(np.corrcoef(flat_a[mask], flat_b[mask])[0, 1]) ** 2

    ax.text(
        0.98,
        0.97,
        f"r² = {r2_val:.6f}\nn = {mask.sum():,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontfamily="monospace",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=COLOR_NEUTRAL_DARK,
            alpha=0.88,
        ),
    )

    ax.set_xlabel(f"{label_b} kinship", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{label_a} kinship", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# founder-effect bar comparison
# ─────────────────────────────────────────────────────────────────────────────


def founder_effects_comparison(
    betas_a: np.ndarray,
    betas_b: np.ndarray,
    founders: list[str],
    *,
    label_a: str = "A",
    label_b: str = "B",
    color_a: str = _COLOR_A,
    color_b: str = _COLOR_B,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """grouped bar chart of per-founder effects from two series at a single locus.

    Args:
        betas_a: series-A per-founder effects, shape ``(n_founders,)``.
        betas_b: series-B per-founder effects, shape ``(n_founders,)``.
        founders: founder display names, length ``n_founders``.
        label_a: legend label for series A.
        label_b: legend label for series B.
        color_a: bar fill color for series A.
        color_b: bar fill color for series B.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4), dpi=300)

    x = np.arange(len(founders))
    width = 0.35
    founder_colors = [FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in founders]

    bars_a = ax.bar(
        x - width / 2,
        betas_a,
        width,
        color=founder_colors,
        edgecolor="white",
        lw=0.5,
        label=label_a,
    )
    bars_b = ax.bar(
        x + width / 2,
        betas_b,
        width,
        color=founder_colors,
        edgecolor=color_b,
        lw=1.0,
        alpha=0.55,
        label=label_b,
    )
    _ = bars_a, bars_b  # suppress unused-var lint

    ax.axhline(0, color=COLOR_NEUTRAL_DARK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(founders, rotation=45, ha="right", fontsize=10)
    ax.tick_params(labelsize=11)
    ax.set_ylabel("Allele effect (RINT)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# permutation max-LOD comparison
# ─────────────────────────────────────────────────────────────────────────────


def perm_lod_comparison(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    label_a: str = "A",
    label_b: str = "B",
    color_a: str = _COLOR_A,
    color_b: str = _COLOR_B,
    threshold_a: Optional[float] = None,
    threshold_b: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """overlaid histograms of permutation max-LOD distributions for two series.

    Args:
        series_a: series-A max-LOD null values (one per permutation).
        series_b: series-B max-LOD null values.
        label_a: legend label for series A.
        label_b: legend label for series B.
        color_a: hex color for series A.
        color_b: hex color for series B.
        threshold_a: GEV 95th-percentile threshold for series A; drawn as a
            vertical dashed line when provided.
        threshold_b: GEV 95th-percentile threshold for series B.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), dpi=300)

    for vals, label, color, thr in [
        (series_a, label_a, color_a, threshold_a),
        (series_b, label_b, color_b, threshold_b),
    ]:
        if vals is None or len(vals) < 2:
            continue
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        sns.histplot(
            v,
            bins=30,
            stat="density",
            alpha=0.35,
            color=color,
            label=f"{label} (n={len(v):,})",
            ax=ax,
        )
        xr = np.linspace(v.min(), v.max(), 200)
        try:
            kde = gaussian_kde(v, bw_method=0.3)
            sns.lineplot(x=xr, y=kde(xr), color=color, lw=1.5, ax=ax)
        except Exception:
            pass
        if thr is not None:
            ax.axvline(
                thr, color=color, ls="--", lw=1.2, label=f"{label} p95={thr:.2f}"
            )

    ax.set_xlabel("Max LOD Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# MI vs ROP scan overlay
# ─────────────────────────────────────────────────────────────────────────────


def mi_rop_comparison(
    mi_pos: np.ndarray,
    mi_lod: np.ndarray,
    rop_pos: np.ndarray,
    rop_lod: np.ndarray,
    *,
    chrom: str = "",
    gene: str = "",
    fwer_thr: Optional[float] = None,
    color_mi: str = _COLOR_A,
    color_rop: str = _COLOR_B,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """overlay MI-averaged LOD and ROP LOD profiles for one gene.

    Args:
        mi_pos: genomic positions (Mb) for the MI scan, shape ``(n_loci,)``.
        mi_lod: MI-averaged LOD scores, shape ``(n_loci,)``.
        rop_pos: genomic positions (Mb) for the ROP scan, shape ``(n_loci,)``.
        rop_lod: ROP LOD scores, shape ``(n_loci,)``.
        chrom: chromosome label for the x-axis.
        gene: gene name shown as a bold top-left annotation.
        fwer_thr: FWER probability threshold drawn as a horizontal line.
        color_mi: hex color for the MI trace (default blue).
        color_rop: hex color for the ROP trace (default red).
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

    sns.lineplot(
        x=rop_pos,
        y=rop_lod,
        color=color_rop,
        lw=1.0,
        linestyle="--",
        label="ROP",
        alpha=0.8,
        ax=ax,
        zorder=2,
    )
    ax.fill_between(mi_pos, mi_lod, alpha=0.12, color=color_mi, zorder=1)
    sns.lineplot(
        x=mi_pos, y=mi_lod, color=color_mi, lw=1.4, label="MI", ax=ax, zorder=3
    )

    if fwer_thr is not None:
        ax.axhline(
            fwer_thr,
            color=COLOR_NEUTRAL_DARK,
            ls=":",
            lw=0.8,
            label=f"FWER {fwer_thr:.2f}",
        )

    if gene:
        ax.text(
            0.02,
            0.97,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_mi,
        )

    ax.set_ylim(bottom=0)
    xlabel = f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)"
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Mediation LOD overlay
# ─────────────────────────────────────────────────────────────────────────────


def mediation_lod_comparison(
    base_pos: np.ndarray,
    base_lod: np.ndarray,
    mediated_pos: np.ndarray,
    mediated_lod: np.ndarray,
    *,
    chrom: str = "",
    gene: str = "",
    mediator: str = "",
    color_base: str = _COLOR_A,
    color_mediated: str = _COLOR_B,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """overlay base LOD and mediated LOD profiles for one (outcome, mediator) pair.

    Args:
        base_pos: positions (Mb) for the base scan.
        base_lod: base LOD scores (outcome ~ H, no mediator covariate).
        mediated_pos: positions (Mb) for the mediated scan.
        mediated_lod: mediated LOD scores (outcome ~ H | mediator).
        chrom: chromosome label for the x-axis.
        gene: outcome gene name shown as a bold top-left annotation.
        mediator: mediator gene name appended to the annotation.
        color_base: hex color for the base LOD trace.
        color_mediated: hex color for the mediated LOD trace.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

    ax.fill_between(base_pos, base_lod, alpha=0.12, color=color_base, zorder=1)
    sns.lineplot(
        x=base_pos,
        y=base_lod,
        color=color_base,
        lw=1.4,
        label="base LOD",
        ax=ax,
        zorder=3,
    )
    sns.lineplot(
        x=mediated_pos,
        y=mediated_lod,
        color=color_mediated,
        lw=1.0,
        linestyle="--",
        label="mediated LOD",
        alpha=0.85,
        ax=ax,
        zorder=2,
    )

    label_parts = [g for g in (gene, mediator) if g]
    if label_parts:
        ax.text(
            0.02,
            0.97,
            " → ".join(label_parts),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_base,
        )

    ax.set_ylim(bottom=0)
    xlabel = f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)"
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Instability LOD overlay
# ─────────────────────────────────────────────────────────────────────────────


def instability_lod_comparison(
    inst_pos: np.ndarray,
    inst_lod: np.ndarray,
    rop_pos: np.ndarray,
    rop_lod: np.ndarray,
    *,
    chrom: str = "",
    gene: str = "",
    fwer_thr: Optional[float] = None,
    color_inst: str = _COLOR_A,
    color_rop: str = _COLOR_B,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """overlay instability-scan LOD and ROP LOD profiles for one gene.

    Args:
        inst_pos: positions (Mb) for the instability scan.
        inst_lod: instability LOD scores (mean over subsamples).
        rop_pos: positions (Mb) for the ROP scan.
        rop_lod: ROP LOD scores.
        chrom: chromosome label for the x-axis.
        gene: gene name shown as a bold top-left annotation.
        fwer_thr: FWER probability threshold drawn as a horizontal line.
        color_inst: hex color for the instability trace.
        color_rop: hex color for the ROP trace.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

    sns.lineplot(
        x=rop_pos,
        y=rop_lod,
        color=color_rop,
        lw=1.0,
        linestyle="--",
        label="ROP",
        alpha=0.8,
        ax=ax,
        zorder=2,
    )
    ax.fill_between(inst_pos, inst_lod, alpha=0.12, color=color_inst, zorder=1)
    sns.lineplot(
        x=inst_pos,
        y=inst_lod,
        color=color_inst,
        lw=1.4,
        label="instability",
        ax=ax,
        zorder=3,
    )

    if fwer_thr is not None:
        ax.axhline(
            fwer_thr,
            color=COLOR_NEUTRAL_DARK,
            ls=":",
            lw=0.8,
            label=f"FWER {fwer_thr:.2f}",
        )

    if gene:
        ax.text(
            0.02,
            0.97,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_inst,
        )

    ax.set_ylim(bottom=0)
    xlabel = f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)"
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# NB Stage 1 — dispersion distribution
# ─────────────────────────────────────────────────────────────────────────────


def phi_distribution(
    phi_vals: np.ndarray,
    phi_raw: Optional[np.ndarray] = None,
    *,
    gene: str = "",
    color_shrunk: str = _COLOR_A,
    color_raw: str = _COLOR_B,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """histogram of EB-shrunk (and optionally raw) per-gene dispersion estimates.

    Args:
        phi_vals: EB-shrunk overdispersion values ``φ`` per gene.
        phi_raw: raw (pre-shrinkage) overdispersion values per gene; omitted
            when ``None``.
        gene: gene label shown as a bold top-left annotation.
        color_shrunk: hex color for the shrunk-φ histogram.
        color_raw: hex color for the raw-φ histogram.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), dpi=300)

    phi_clean = phi_vals[np.isfinite(phi_vals)]
    p99 = float(np.nanpercentile(phi_clean, 99)) if len(phi_clean) else 1.0
    bins = np.linspace(0, max(p99, 1e-3), 40)

    sns.histplot(
        phi_clean, bins=bins, color=color_shrunk, alpha=0.7, label="shrunk φ", ax=ax
    )
    if phi_raw is not None:
        raw_clean = phi_raw[np.isfinite(phi_raw)]
        sns.histplot(
            raw_clean, bins=bins, color=color_raw, alpha=0.4, label="raw φ", ax=ax
        )
        ax.legend(fontsize=9, frameon=False, loc="upper right")

    if gene:
        ax.text(
            0.02,
            0.97,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_shrunk,
        )

    ax.set_xlim(left=0)
    ax.set_xlabel("Overdispersion φ", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gene count", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# NB vs LMM LOD scatter
# ─────────────────────────────────────────────────────────────────────────────


def nb_vs_lmm_comparison(
    lod_nb: np.ndarray,
    lod_lmm: np.ndarray,
    *,
    gene: str = "",
    color_nb: str = _COLOR_A,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """scatter of NB score LOD vs Gaussian LMM LOD at matched loci.

    Args:
        lod_nb: NB score LOD values, shape ``(n_loci,)``.
        lod_lmm: Gaussian LMM LOD values at the same loci, shape ``(n_loci,)``.
        gene: gene label shown as a bold top-left annotation.
        color_nb: hex color for the scatter points.
        ax: existing Axes; creates a new figure (dpi=300) if ``None``.

    Returns:
        the Axes containing the plot.
    """
    _set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), dpi=300)

    nb_c = np.asarray(lod_nb, dtype=float)
    lm_c = np.asarray(lod_lmm, dtype=float)
    mask = np.isfinite(nb_c) & np.isfinite(lm_c)

    sns.scatterplot(
        x=lm_c[mask],
        y=nb_c[mask],
        color=color_nb,
        s=5,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
        ax=ax,
    )

    lim = max(lm_c[mask].max(), nb_c[mask].max()) * 1.05
    sns.lineplot(
        x=[0, lim],
        y=[0, lim],
        color=COLOR_CC_FALLBACK,
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="y = x",
        errorbar=None,
        ax=ax,
    )

    if mask.sum() >= 2:
        r_val = float(np.corrcoef(lm_c[mask], nb_c[mask])[0, 1])
        ax.text(
            0.98,
            0.97,
            f"r = {r_val:.4f}\nn = {mask.sum():,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontfamily="monospace",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=COLOR_NEUTRAL_DARK,
                alpha=0.88,
            ),
        )

    if gene:
        ax.text(
            0.02,
            0.97,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color_nb,
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("LMM LOD", fontsize=13, fontweight="bold")
    ax.set_ylabel("NB score LOD", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# cis-distance comparison across sources (cell types / whole tissue)
# ─────────────────────────────────────────────────────────────────────────────


def cis_distance_comparison(
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    palette: Optional[dict[str, str]] = None,
    zoom_mb: float = 1.0,
    windows_mb: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
) -> plt.Figure:
    """two-panel cis-distance comparison across cell types or scan runs.

    Args:
        sources: mapping of ``source_label`` →
            ``(peak_lod, tss_mb, peak_mb)`` arrays, each shape ``(n_genes,)``.
            sources are plotted in iteration order; put whole-tissue first.
        palette: optional mapping of source label → hex color. falls back to
            ``COLOR_CC_FALLBACK`` for any label not in the dict.
        zoom_mb: half-window (Mb) for the ±zoom shaded band on the violin
            panel; default 1.0.
        windows_mb: cumulative window sizes (Mb) for the proximal-fraction
            curve panel.

    Returns:
        two-panel ``plt.Figure`` — violin distribution | proximal-fraction
        curves.
    """
    _set_style()
    source_names = list(sources.keys())
    _pal = palette or {}

    fig, (ax_v, ax_b) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=300, facecolor="white"
    )

    # ── panel 1: violin of signed cis distances ───────────────────────────────
    for i, name in enumerate(source_names):
        _, tss, peak = sources[name]
        dist = np.asarray(peak, dtype=float) - np.asarray(tss, dtype=float)
        dist = dist[np.isfinite(dist)]
        if len(dist) < 2:
            continue
        color = _pal.get(name, COLOR_CC_FALLBACK)
        vp = ax_v.violinplot(
            [dist], positions=[i], widths=0.65, showmedians=True, showextrema=False
        )
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor("white")
            pc.set_linewidth(0.5)
        vp["cmedians"].set_color(COLOR_NEUTRAL_DARK)
        vp["cmedians"].set_linewidth(2.0)

    ax_v.axhline(0, color=COLOR_NEUTRAL_DARK, ls="--", lw=1.1, alpha=0.6, zorder=0)
    ax_v.axhspan(-zoom_mb, zoom_mb, alpha=0.07, color=WASH, zorder=0)
    ax_v.set_xticks(range(len(source_names)))
    ax_v.set_xticklabels(source_names, rotation=30, ha="right", fontsize=9)
    ax_v.set_ylabel("Cis distance  (peak − TSS, ±Mb)", fontsize=11)
    ax_v.set_title("Distribution by source", fontsize=12, fontweight="bold")
    sns.despine(ax=ax_v)

    # ── panel 2: % genes within cumulative Mb windows ─────────────────────────
    wlist = list(windows_mb)
    for name in source_names:
        _, tss, peak = sources[name]
        dist_abs = np.abs(np.asarray(peak, dtype=float) - np.asarray(tss, dtype=float))
        dist_abs = dist_abs[np.isfinite(dist_abs)]
        if len(dist_abs) < 2:
            continue
        color = _pal.get(name, COLOR_CC_FALLBACK)
        pcts = [100.0 * float((dist_abs <= w).mean()) for w in wlist]
        ax_b.plot(
            wlist, pcts, "o-", color=color, label=name, linewidth=1.8, markersize=5
        )

    ax_b.set_xlabel("Window size (Mb)", fontsize=11)
    ax_b.set_ylabel("% genes with |cis dist| ≤ window", fontsize=11)
    ax_b.set_title("Proximal peak fraction", fontsize=12, fontweight="bold")
    ax_b.set_xlim(0, max(wlist) + 0.5)
    ax_b.set_ylim(0, 105)
    ax_b.legend(fontsize=8.5, frameon=False, loc="upper left")
    sns.despine(ax=ax_b)

    fig.tight_layout()
    return fig
