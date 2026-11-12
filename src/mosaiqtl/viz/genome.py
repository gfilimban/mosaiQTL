"""LOD profiles, Manhattan plots, region zooms, and SNP association plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._theme import (
    BRAND_PRIMARY,
    BRAND_SECONDARY,
    BRAND_ACCENT,
    INK,
    MUTED,
    CHR_BANDS,
    COLOR_UTR,
    CMAP_MAGNITUDE,
    CMAP_SIGNED as _CMAP_EFFECT,
    FONT_STACK,
    KW_THRESHOLD,
    KW_REFERENCE,
    KW_DOT_MUTED,
    KW_BOX,
    apply_theme,
    info_box,
)
from .effects import effect_bars as _effect_bars

_CHR_ORDER: list[str] = [str(i) for i in range(1, 20)] + ["X"]
_CHR_GAP_MB: float = 10.0

# ── color aliases for overlay series ─────────────────────────────────────────
# MI overlay = secondary color; instability = accent color; SNP = muted grey.
_MI_COLOR: str = BRAND_SECONDARY
_INST_COLOR: str = BRAND_ACCENT
_SNP_COLOR: str = MUTED


def lod_profile(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    threshold: Optional[float] = None,
    peak_pos: Optional[float] = None,
    color: str = BRAND_PRIMARY,
    linewidth: float = 1.2,
    fill: bool = False,
    fill_alpha: float = 0.15,
    h2: Optional[float] = None,
    gene: str = "",
    tss_mb: Optional[float] = None,
    locus_id: Optional[str] = None,
    marker_rug: bool = False,
    annotate_peak: bool = False,
    label: str = "",
) -> plt.Axes:
    """Single-chromosome LOD profile.

    Parameters
    ----------
    positions : np.ndarray
        Marker positions in Mb, shape (n_loci,).
    lod : np.ndarray
        LOD scores, same shape as positions.
    ax : plt.Axes, optional
        Axes to plot on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis.
    threshold : float, optional
        Permutation threshold — drawn as horizontal dashed line with label.
    peak_pos : float, optional
        Peak position in Mb — drawn as vertical dotted line; label added when
        ``annotate_peak=True``.
    color : str
        Line color; must be imported from founders.py.
    linewidth : float
        Line width.
    fill : bool
        Shade area under the LOD curve.
    fill_alpha : float
        Alpha for the fill band.
    h2 : float, optional
        LOCO null heritability — shown as annotation in upper-right corner.
    gene : str
        Gene name — shown as bold text in upper-left corner.
    tss_mb : float, optional
        Gene TSS position — drawn as a dashed grey vertical line.
    locus_id : str, optional
        Peak locus identifier — shown alongside the LOD annotation.
    marker_rug : bool
        When True draws a rug of tick marks at each tested locus position.
    annotate_peak : bool
        When True, adds a LOD text label at the peak position (requires
        ``peak_pos``). Defaults to False so callers that only want a vertical
        line do not affect ``ax.texts``.
    label : str
        Legend label for the LOD line. When non-empty, ``ax.legend()`` is
        called so that successive overlay calls accumulate a legend.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
    order = np.argsort(positions)
    positions = positions[order]
    lod = lod[order]
    sns.lineplot(
        x=positions,
        y=lod,
        color=color,
        linewidth=linewidth,
        errorbar=None,
        ax=ax,
        zorder=2,
        label=label if label else "_nolegend_",
    )
    if fill:
        ax.fill_between(positions, 0, lod, alpha=fill_alpha, color=color, zorder=1)
    if marker_rug:
        sns.rugplot(x=positions, ax=ax, color=MUTED, alpha=0.4, height=0.03)
    if positions[0] != positions[-1]:
        ax.set_xlim(float(positions[0]), float(positions[-1]))
    ax.set_ylim(0, max(float(lod.max()) * 1.18, 1.0))
    ax.set_xlabel(
        f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)",
        fontsize=13,
    )
    ax.set_ylabel("LOD Score", fontsize=13)
    ax.tick_params(labelsize=11)
    if threshold is not None:
        ax.axhline(threshold, **KW_THRESHOLD)
    if tss_mb is not None:
        ax.axvline(tss_mb, color=MUTED, lw=0.9, ls="--", zorder=1, label="TSS")
    if peak_pos is not None:
        ax.axvline(
            peak_pos,
            color=BRAND_SECONDARY,
            linewidth=1.3,
            linestyle="-",
            zorder=3,
            label="peak",
        )
        if annotate_peak and locus_id:
            # place label beside (not on) the line: anchor at the line in
            # data-x / axes-y coords, then shift 9 pts right in display space.
            # ha/va="center" keeps the rotated text block centered on that
            # offset point so it doesn't overlap the line.
            ax.annotate(
                locus_id,
                xy=(peak_pos, 0.5),
                xycoords=ax.get_xaxis_transform(),
                xytext=(9, 0),
                textcoords="offset points",
                fontsize=9,
                color=BRAND_SECONDARY,
                va="center",
                ha="center",
                rotation=-90,
                annotation_clip=False,
            )
    # gene name in the upper-left corner
    if gene:
        ax.text(
            0.02,
            0.96,
            gene,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            fontweight="bold",
            color=color,
        )
    # heritability (h²) in the bottom-right, away from the legend
    if h2 is not None:
        ax.text(
            0.98,
            0.04,
            f"h²={h2:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="right",
            color=INK,
        )
    if label:
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    apply_theme(ax)
    return ax


def manhattan(
    scans: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    figsize: tuple[float, float] = (14, 3.5),
    threshold: Optional[float] = None,
    highlight_chrom: Optional[str] = None,
    highlight_color: str = BRAND_ACCENT,
    egene_scans: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    low_expr_mask: Optional[dict[str, np.ndarray]] = None,
    top_labels: Optional[dict[str, list[str]]] = None,
    top_n_labels: int = 5,
    title: Optional[str] = None,
) -> plt.Figure:
    """Genome-wide Manhattan plot of LOD scores.

    Parameters
    ----------
    scans : dict[str, tuple[np.ndarray, np.ndarray]]
        ``{chrom: (positions_Mb, lod)}`` one entry per chromosome.
        Chromosomes absent from _CHR_ORDER are appended at the end.
    figsize : tuple[float, float]
        Figure dimensions in inches.
    threshold : float, optional
        Permutation threshold — drawn as horizontal dashed line.
    highlight_chrom : str, optional
        Single chromosome drawn in highlight_color.
    highlight_color : str
        Color for highlighted chromosome; from founders.py.
    egene_scans : dict, optional
        ``{chrom: (positions_Mb, lod)}`` for eGene loci only.
        Overlaid as larger colored dots on top of the grey background.
    low_expr_mask : dict, optional
        ``{chrom: bool_array}`` parallel to ``scans``; True entries are drawn
        as light-grey dots instead of the chromosome color.
    top_labels : dict, optional
        ``{chrom: [gene_symbol, ...]}`` parallel to scans positions;
        annotates the top-LOD genes by name.
    top_n_labels : int
        Number of top-LOD genes to label when top_labels is not provided.

    Returns
    -------
    plt.Figure
    """
    ordered = [c for c in _CHR_ORDER if c in scans]
    extra = [c for c in scans if c not in ordered]
    chroms = ordered + extra

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    offset = 0.0
    xticks: list[float] = []
    xlabels: list[str] = []
    # collect (x, lod, chrom) for top-gene labeling
    all_pts: list[tuple[float, float, str]] = []
    _hi_xspan: "tuple[float, float] | None" = None

    for ci, chrom in enumerate(chroms):
        pos, lod = scans[chrom]
        order = np.argsort(pos)
        pos = pos[order]
        lod = lod[order]
        x = pos + offset
        if chrom == highlight_chrom:
            _hi_xspan = (float(x[0]), float(x[-1]))
        chr_color = CHR_BANDS[ci % 2]
        # low-expr genes: light grey, plotted before chromosome-coloured dots
        if low_expr_mask and chrom in low_expr_mask:
            _lo = low_expr_mask[chrom][order]
            if _lo.any():
                sns.scatterplot(
                    x=x[_lo],
                    y=lod[_lo],
                    **KW_DOT_MUTED,
                    ax=ax,
                )
            if (~_lo).any():
                sns.scatterplot(
                    x=x[~_lo],
                    y=lod[~_lo],
                    color=chr_color,
                    s=5,
                    alpha=0.4,
                    linewidths=0,
                    rasterized=True,
                    ax=ax,
                )
        else:
            sns.scatterplot(
                x=x,
                y=lod,
                color=chr_color,
                s=5,
                alpha=0.4,
                linewidths=0,
                rasterized=True,
                ax=ax,
            )
        # eGene overlay: accent orange, larger — marks FDR-significant genes
        if egene_scans and chrom in egene_scans:
            epos, elod = egene_scans[chrom]
            eorder = np.argsort(epos)
            ex = epos[eorder] + offset
            sns.scatterplot(
                x=ex,
                y=elod[eorder],
                color=BRAND_ACCENT,
                s=18,
                linewidths=0,
                rasterized=True,
                ax=ax,
            )
        mid = float(offset + (float(pos[-1]) - float(pos[0])) / 2)
        xticks.append(mid)
        xlabels.append(chrom)
        for xi, li in zip(x, lod):
            all_pts.append((float(xi), float(li), chrom))
        offset += float(pos[-1]) + _CHR_GAP_MB

    if _hi_xspan is not None:
        ax.axvspan(
            _hi_xspan[0], _hi_xspan[1], alpha=0.06, color=highlight_color, zorder=0
        )

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel("LOD Score", fontsize=13)
    ax.set_xlabel("Chromosome", fontsize=13)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.tick_params(axis="x", length=2, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    if threshold is not None:
        ax.axhline(
            threshold,
            color=BRAND_ACCENT,
            linewidth=1.0,
            linestyle=":",
            zorder=1,
            label=f"p95 = {threshold:.2f}",
        )
        ax.legend(fontsize=9, frameon=False, loc="upper right")
    # label top-N LOD points across the whole genome
    if top_n_labels > 0 and all_pts:
        top_pts = sorted(all_pts, key=lambda t: t[1], reverse=True)[:top_n_labels]
        for tx, ty, _ in top_pts:
            ax.annotate(
                f"{ty:.1f}",
                (tx, ty),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=10,
                ha="center",
                color=INK,
            )
    if title:
        ax.set_title(title, fontsize=8)
    apply_theme(ax)
    return fig


def lod_interval(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    lod_drop: float = 1.5,
    chrom: str = "",
    color: str = BRAND_PRIMARY,
) -> plt.Axes:
    """LOD support interval: shades region where LOD >= peak − lod_drop.

    Parameters
    ----------
    positions : np.ndarray
        Marker positions in Mb.
    lod : np.ndarray
        LOD scores.
    lod_drop : float
        LOD units below peak defining the interval boundary (1.5 ≈ 95% CI).
    chrom : str
        Chromosome label for x-axis.
    color : str
        Line and shading color; from founders.py.

    Returns
    -------
    plt.Axes
    """
    order = np.argsort(positions)
    positions = positions[order]
    lod = lod[order]
    ax = lod_profile(positions, lod, ax=ax, chrom=chrom, color=color)
    cutoff = float(lod.max()) - lod_drop
    ax.fill_between(
        positions, 0, lod, where=(lod >= cutoff), alpha=0.20, color=color, linewidth=0
    )
    ax.axhline(cutoff, color=MUTED, linewidth=0.6, linestyle=":", zorder=1)
    return ax


def bayes_interval(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    prob: float = 0.95,
    chrom: str = "",
    color: str = BRAND_PRIMARY,
) -> plt.Axes:
    """Bayes credible interval: shades highest-density region containing `prob` mass.

    Converts LOD scores to likelihoods, normalizes to a density, then shades
    the smallest contiguous region containing the requested probability mass.

    Parameters
    ----------
    positions : np.ndarray
        Marker positions in Mb.
    lod : np.ndarray
        LOD scores.
    prob : float
        Credible interval mass (0.95 = 95% BCI).
    chrom : str
        Chromosome label.
    color : str
        Line and shading color; from founders.py.

    Returns
    -------
    plt.Axes
    """
    ax = lod_profile(positions, lod, ax=ax, chrom=chrom, color=color)
    likelihood = 10.0 ** (lod - float(lod.max()))
    density = likelihood / likelihood.sum()
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(density[order])
    in_hdr = np.zeros(len(lod), dtype=bool)
    in_hdr[order[cumulative <= prob]] = True
    ax.fill_between(
        positions, 0, lod, where=in_hdr, alpha=0.20, color=color, linewidth=0
    )
    return ax


def region_zoom(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    start_mb: Optional[float] = None,
    end_mb: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    threshold: Optional[float] = None,
    snp_positions: Optional[np.ndarray] = None,
    snp_neg_log10p: Optional[np.ndarray] = None,
    snp_color: str = MUTED,
    color: str = BRAND_PRIMARY,
) -> plt.Axes:
    """Zoomed-in LOD profile for a genomic region, with optional SNP overlay.

    Slices the scan to [start_mb, end_mb] and optionally overlays SNP
    association scores on a secondary y-axis.

    Parameters
    ----------
    positions : np.ndarray
        Full-chromosome marker positions in Mb.
    lod : np.ndarray
        Full-chromosome LOD scores.
    start_mb : float, optional
        Left boundary of region in Mb; defaults to chromosome start.
    end_mb : float, optional
        Right boundary of region in Mb; defaults to chromosome end.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis.
    threshold : float, optional
        Permutation threshold line.
    snp_positions : np.ndarray, optional
        SNP positions in Mb for the secondary axis overlay.
    snp_neg_log10p : np.ndarray, optional
        −log10(p) values per SNP; requires snp_positions.
    snp_color : str
        Color for SNP points; from founders.py palette.
    color : str
        LOD line color; from founders.py.

    Returns
    -------
    plt.Axes
        Primary axes (LOD); secondary axes accessible via ax.get_figure().
    """
    lo = start_mb if start_mb is not None else float(positions[0])
    hi = end_mb if end_mb is not None else float(positions[-1])
    mask = (positions >= lo) & (positions <= hi)
    ax = lod_profile(
        positions[mask], lod[mask], ax=ax, chrom=chrom, color=color, threshold=threshold
    )
    ax.set_xlim(lo, hi)

    if snp_positions is not None and snp_neg_log10p is not None:
        snp_mask = (snp_positions >= lo) & (snp_positions <= hi)
        ax2 = ax.twinx()
        ax2.scatter(
            snp_positions[snp_mask],
            snp_neg_log10p[snp_mask],
            color=snp_color,
            s=4,
            alpha=0.6,
            zorder=1,
        )
        ax2.set_ylabel("−log₁₀ P (SNP)", fontsize=8)
        ax2.set_ylim(0, None)
        sns.despine(ax=ax2, right=False)

    return ax


def snp_manhattan(
    snp_positions: np.ndarray,
    snp_neg_log10p: np.ndarray,
    snp_chroms: np.ndarray,
    *,
    figsize: tuple[float, float] = (14, 3),
    threshold: Optional[float] = None,
    highlight_chrom: Optional[str] = None,
    highlight_color: str = BRAND_ACCENT,
    point_size: float = 2.0,
) -> plt.Figure:
    """Genome-wide SNP association Manhattan plot (−log10 p-value).

    Parameters
    ----------
    snp_positions : np.ndarray
        SNP positions in Mb, shape (n_snps,).
    snp_neg_log10p : np.ndarray
        −log10(p-value) per SNP, shape (n_snps,).
    snp_chroms : np.ndarray
        Chromosome label per SNP, shape (n_snps,), dtype str.
    figsize : tuple[float, float]
        Figure dimensions in inches.
    threshold : float, optional
        Significance threshold — drawn as horizontal dashed line.
    highlight_chrom : str, optional
        Chromosome drawn in highlight_color.
    highlight_color : str
        Color for highlighted chromosome; from founders.py.
    point_size : float
        Scatter point size.

    Returns
    -------
    plt.Figure
    """
    ordered = [c for c in _CHR_ORDER if c in set(snp_chroms)]
    extra = sorted(set(snp_chroms) - set(ordered))
    chroms = ordered + extra

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    offset = 0.0
    xticks: list[float] = []
    xlabels: list[str] = []
    _hi_xspan_p: "tuple[float, float] | None" = None

    for ci, chrom in enumerate(chroms):
        mask = snp_chroms == chrom
        pos = snp_positions[mask]
        y = snp_neg_log10p[mask]
        x = pos + offset
        if chrom == highlight_chrom:
            _hi_xspan_p = (float(x[0]), float(x[-1]))
        sns.scatterplot(
            x=x,
            y=y,
            color=CHR_BANDS[ci % 2],
            s=point_size,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            ax=ax,
        )
        xticks.append(float(offset + (pos.max() - pos.min()) / 2))
        xlabels.append(chrom)
        offset += float(pos.max()) + _CHR_GAP_MB

    if _hi_xspan_p is not None:
        ax.axvspan(
            _hi_xspan_p[0], _hi_xspan_p[1], alpha=0.06, color=highlight_color, zorder=0
        )

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel("−log₁₀ P")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis="x", length=2)
    if threshold is not None:
        ax.axhline(threshold, **KW_THRESHOLD)
    apply_theme(ax)
    return fig


def pvalue_qq(
    p_values: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """QQ plot of scan p-values against uniform null with genomic inflation factor λ_GC.

    Parameters
    ----------
    p_values : np.ndarray
        Per-locus p-values from a genome scan, shape (n_loci,).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=300)

    pv = np.sort(p_values[p_values > 0])
    n = len(pv)
    observed = -np.log10(pv)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    sns.scatterplot(
        x=expected,
        y=observed,
        s=5,
        alpha=0.4,
        color=BRAND_PRIMARY,
        linewidths=0,
        rasterized=True,
        zorder=2,
        ax=ax,
    )
    _diag = np.array([0.0, float(expected.max())])
    sns.lineplot(
        x=_diag,
        y=_diag,
        color=INK,
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        errorbar=None,
        ax=ax,
        zorder=1,
    )

    lambda_gc = float(np.median(pv)) / 0.5
    ax.text(
        0.05,
        0.95,
        f"λ_GC = {1/lambda_gc:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Expected −log₁₀(p)")
    ax.set_ylabel("Observed −log₁₀(p)")
    apply_theme(ax, from_zero=False)
    return ax


def lod_histogram(
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
) -> plt.Axes:
    """LOD score distribution histogram across all loci in a scan.

    Useful for model calibration check — heavily right-skewed with a long
    tail is expected for a well-calibrated scan.

    Parameters
    ----------
    lod : np.ndarray
        LOD scores from a genome scan, shape (n_loci,).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    threshold : float, optional
        Significance threshold — drawn as vertical dashed line.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=300)

    sns.histplot(lod, bins=80, color=BRAND_PRIMARY, alpha=0.7, ax=ax, linewidth=0)
    if threshold is not None:
        ax.axvline(
            threshold,
            color=BRAND_ACCENT,
            linewidth=1.4,
            linestyle=":",
            alpha=0.85,
            label=f"threshold = {threshold:.1f}",
        )
        ax.legend(fontsize=9, frameon=False, loc="upper right")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("LOD Score")
    ax.set_ylabel("Number of loci")
    apply_theme(ax)
    return ax


def gene_manhattan(
    gene_positions: np.ndarray,
    gene_chroms: np.ndarray,
    peak_lod: np.ndarray,
    *,
    figsize: tuple[float, float] = (14, 3),
    threshold: Optional[float] = None,
    highlight_chrom: Optional[str] = None,
    highlight_color: str = BRAND_ACCENT,
    point_size: float = 6.0,
) -> plt.Figure:
    """Gene-level Manhattan: one point per gene at its genome-wide peak LOD.

    Parameters
    ----------
    gene_positions : np.ndarray
        Gene TSS positions in Mb, shape (n_genes,).
    gene_chroms : np.ndarray
        Chromosome per gene, shape (n_genes,), dtype str.
    peak_lod : np.ndarray
        Genome-wide peak LOD per gene, shape (n_genes,).
    figsize : tuple[float, float]
        Figure dimensions in inches.
    threshold : float, optional
        Significance threshold line.
    highlight_chrom : str, optional
        Chromosome drawn in highlight_color.
    highlight_color : str
        Color for highlighted chromosome; from founders.py.
    point_size : float
        Scatter point size.

    Returns
    -------
    plt.Figure
    """
    ordered = [c for c in _CHR_ORDER if c in set(gene_chroms)]
    extra = sorted(set(gene_chroms) - set(ordered))
    chroms = ordered + extra

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    offset = 0.0
    xticks: list[float] = []
    xlabels: list[str] = []

    _hi_xspan_g: "tuple[float, float] | None" = None

    for ci, chrom in enumerate(chroms):
        mask = gene_chroms == chrom
        pos = gene_positions[mask]
        y = peak_lod[mask]
        x = pos + offset
        if chrom == highlight_chrom:
            _hi_xspan_g = (float(x[0]), float(x[-1]))
        sns.scatterplot(
            x=x,
            y=y,
            color=CHR_BANDS[ci % 2],
            s=point_size,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            ax=ax,
        )
        xticks.append(float(offset + (pos.max() - pos.min()) / 2))
        xlabels.append(chrom)
        offset += float(pos.max()) + _CHR_GAP_MB

    if _hi_xspan_g is not None:
        ax.axvspan(
            _hi_xspan_g[0],
            _hi_xspan_g[1],
            alpha=0.06,
            color=highlight_color,
            zorder=0,
        )

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel("Peak LOD Score")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis="x", length=2)
    if threshold is not None:
        ax.axhline(threshold, **KW_THRESHOLD)
    apply_theme(ax)
    return fig


def cis_distance_histogram(
    tss_positions: np.ndarray,
    peak_positions: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    cis_window_mb: float = 5.0,
    proximal_mb: float = 1.0,
    bins: int = 60,
) -> plt.Axes:
    """Annotated histogram of eQTL peak distance from gene TSS.

    Bars are colored by direction: upstream (peak < TSS) in blue, downstream
    in green.  A KDE overlay, stats box, and directional count annotations
    are included.

    Args:
        tss_positions: Gene TSS positions in Mb, shape ``(n_egenes,)``.
        peak_positions: Peak locus positions in Mb, same shape.
        ax: Axes to draw on; created at ``dpi=300`` if ``None``.
        cis_window_mb: Cis-window boundary — drawn as vertical dotted lines.
        proximal_mb: Half-window for "proximal" fraction reported in stats
            box; default 1.0 Mb.
        bins: Number of histogram bins.

    Returns:
        Populated ``plt.Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.8), dpi=300)

    _dist = np.asarray(peak_positions, dtype=float) - np.asarray(
        tss_positions, dtype=float
    )
    _n = len(_dist)
    _up = _dist[_dist < 0]
    _dn = _dist[_dist >= 0]
    _n_up = len(_up)
    _n_dn = len(_dn)
    _med = float(np.median(np.abs(_dist))) if _n > 0 else 0.0
    _prox = 100 * (np.abs(_dist) <= proximal_mb).mean() if _n > 0 else 0.0

    # shared bin edges so upstream/downstream bars align perfectly
    _lo = float(np.floor(_dist.min())) if _n > 0 else -cis_window_mb
    _hi = float(np.ceil(_dist.max())) if _n > 0 else cis_window_mb
    _edges = np.linspace(_lo, _hi, bins + 1)
    _w = (_edges[1] - _edges[0]) * 0.88

    _cnt_up, _e_up = np.histogram(_up, bins=_edges)
    _cnt_dn, _e_dn = np.histogram(_dn, bins=_edges)
    _ctr = (_edges[:-1] + _edges[1:]) / 2

    ax.bar(
        _ctr,
        _cnt_up,
        width=_w,
        color=BRAND_PRIMARY,
        alpha=0.75,
        linewidth=0,
        label=f"upstream  ({_n_up:,})",
        zorder=2,
    )
    ax.bar(
        _ctr,
        _cnt_dn,
        width=_w,
        color=BRAND_SECONDARY,
        alpha=0.75,
        linewidth=0,
        label=f"downstream  ({_n_dn:,})",
        zorder=2,
    )

    # KDE overlay (neutral dark line, no fill)
    if _n > 4:
        _kde_ax = ax.twinx()
        sns.kdeplot(_dist, ax=_kde_ax, color=INK, linewidth=1.6, alpha=0.6)
        _kde_ax.set_ylabel("")
        _kde_ax.set_yticks([])
        sns.despine(ax=_kde_ax, left=True, right=True, top=True, bottom=True)

    # ±proximal_mb shaded band
    ax.axvspan(-proximal_mb, proximal_mb, alpha=0.07, color=BRAND_PRIMARY, zorder=0)

    # reference and boundary lines
    ax.axvline(0, color=INK, lw=1.2, zorder=3)
    for _s in (-1, 1):
        ax.axvline(_s * cis_window_mb, color=MUTED, lw=0.9, ls=":", zorder=1)

    # directional count annotations just above x-axis
    _ymax = max((_cnt_up + _cnt_dn).max(), 1) if _n > 0 else 1
    ax.text(
        -cis_window_mb * 0.05,
        _ymax * 0.96,
        f"{_n_up:,}\n({100*_n_up/_n:.0f}%)",
        fontsize=8.5,
        ha="right",
        va="top",
        color=BRAND_PRIMARY,
        fontweight="bold",
    )
    ax.text(
        cis_window_mb * 0.05,
        _ymax * 0.96,
        f"{_n_dn:,}\n({100*_n_dn/_n:.0f}%)",
        fontsize=8.5,
        ha="left",
        va="top",
        color=BRAND_SECONDARY,
        fontweight="bold",
    )

    info_box(
        ax,
        f"n = {_n:,}\nmedian |d| = {_med:.2f} Mb\n"
        f"±{proximal_mb:.0f} Mb: {_prox:.0f}%",
        corner="tr",
        fontsize=8,
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Cis distance  (peak − TSS, Mb)", fontsize=11)
    ax.set_ylabel("Number of eGenes", fontsize=11)
    _leg = ax.legend(fontsize=8.5, frameon=True, loc="lower right", ncol=1)
    _leg.get_frame().set(**KW_BOX)
    apply_theme(ax)
    return ax


def cis_distance_panel(
    peak_lod: np.ndarray,
    tss_positions: np.ndarray,
    peak_positions: np.ndarray,
    *,
    is_egene: Optional[np.ndarray] = None,
    is_low_expr: Optional[np.ndarray] = None,
    zoom_mb: float = 1.0,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """Three-panel cis-distance figure: full scatter | ±zoom_mb zoom | KDE.

    Args:
        peak_lod: Peak LOD per gene, shape ``(n,)``.
        tss_positions: TSS position in Mb per gene, shape ``(n,)``.
        peak_positions: Peak locus position in Mb per gene, shape ``(n,)``.
        is_egene: Boolean mask marking eGenes, shape ``(n,)``.  All False if
            ``None``.
        is_low_expr: Boolean mask for low-expression genes (frac_cpm_gt1 < 0.5),
            shape ``(n,)``.  These are drawn as light-grey dots behind the colored
            scatter and excluded from the eGene KDE.  All False if ``None``.
        zoom_mb: Half-window (Mb) for the central zoom panel; default 1.0.
        threshold: Peak-LOD significance threshold — drawn as a horizontal
            orange line on the scatter panels.  ``None`` omits the line.

    Returns:
        Three-panel ``plt.Figure`` — full scatter | ±zoom scatter | KDE.
    """
    _lod = np.asarray(peak_lod, dtype=float)
    _tss = np.asarray(tss_positions, dtype=float)
    _peak = np.asarray(peak_positions, dtype=float)
    _dist = _peak - _tss
    _is_eg: np.ndarray = (
        np.asarray(is_egene, dtype=bool)
        if is_egene is not None
        else np.zeros(len(_lod), dtype=bool)
    )
    _is_lo: np.ndarray = (
        np.asarray(is_low_expr, dtype=bool)
        if is_low_expr is not None
        else np.zeros(len(_lod), dtype=bool)
    )
    _hi = ~_is_lo  # expressed genes
    _n = len(_lod)
    _n_eg = int(_is_eg.sum())
    _vmax = float(np.percentile(_lod, 95)) if _n > 0 else 1.0
    _pct1 = 100 * (np.abs(_dist) <= zoom_mb).mean() if _n > 0 else 0.0
    _pct5 = 100 * (np.abs(_dist) <= 5).mean() if _n > 0 else 0.0
    _med = float(np.median(np.abs(_dist))) if _n > 0 else 0.0

    fig, (ax_kd, ax_full, ax_zoom) = plt.subplots(
        1, 3, figsize=(19, 5.5), dpi=300, facecolor="white", constrained_layout=True
    )

    _scatter_hi_kw: dict = dict(
        cmap=CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=_vmax,
        s=5,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )
    _scatter_lo_kw: dict = dict(
        **KW_DOT_MUTED,
    )
    _ref_kw: dict = KW_REFERENCE.copy()
    _thr_kw: dict = KW_THRESHOLD.copy()

    # ── panel 1: full-range scatter ──────────────────────────────────────────
    if _is_lo.any():
        ax_full.scatter(_dist[_is_lo], _lod[_is_lo], **_scatter_lo_kw)
    ax_full.scatter(_dist[_hi], _lod[_hi], c=_lod[_hi], **_scatter_hi_kw)
    ax_full.axvline(0, **_ref_kw)
    if threshold is not None:
        ax_full.axhline(threshold, **_thr_kw)
    ax_full.set_xlabel("")
    ax_full.set_ylabel("Peak LOD", fontsize=11)
    ax_zoom.sharey(ax_full)
    ax_full.set_title("Full range", fontsize=12, fontweight="bold")
    ax_full.text(
        0.03,
        0.97,
        f"n = {_n:,}\neGenes = {_n_eg:,}\n"
        f"±{zoom_mb:.0f} Mb: {_pct1:.0f}%\n"
        f"±5 Mb: {_pct5:.0f}%\n"
        f"median |d|: {_med:.1f} Mb",
        transform=ax_full.transAxes,
        fontsize=8.5,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=INK,
            alpha=0.9,
        ),
    )
    sns.despine(ax=ax_full)
    apply_theme(ax_full)

    # ── panel 2: ±zoom_mb zoom ────────────────────────────────────────────────
    _mask_z = np.abs(_dist) <= zoom_mb
    _mask_z_hi = _mask_z & _hi
    _mask_z_lo = _mask_z & _is_lo
    _dz, _lz = _dist[_mask_z_hi], _lod[_mask_z_hi]
    _n_z_eg = int((_mask_z & _is_eg).sum())
    _vmax_z = float(np.percentile(_lz, 95)) if len(_lz) > 0 else _vmax
    if _mask_z_lo.any():
        ax_zoom.scatter(_dist[_mask_z_lo], _lod[_mask_z_lo], **_scatter_lo_kw)
    ax_zoom.scatter(
        _dz,
        _lz,
        c=_lz,
        cmap=CMAP_MAGNITUDE,
        vmin=0.0,
        vmax=_vmax_z,
        s=8,
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    ax_zoom.axvline(0, **_ref_kw)
    if threshold is not None:
        ax_zoom.axhline(threshold, **_thr_kw)
    ax_zoom.set_xlim(-zoom_mb, zoom_mb)
    ax_zoom.set_xlabel("")
    ax_zoom.set_ylabel("")
    plt.setp(ax_zoom.get_yticklabels(), visible=False)
    ax_zoom.set_title(f"±{zoom_mb:.0f} Mb zoom", fontsize=12, fontweight="bold")
    _pct_half = 100 * (np.abs(_dist) <= zoom_mb / 2).mean() if _n > 0 else 0.0
    _med_z = float(np.median(np.abs(_dz))) if len(_dz) > 0 else 0.0
    ax_zoom.text(
        0.03,
        0.97,
        f"n = {_mask_z.sum():,}  ({_pct1:.0f}% of all)\n"
        f"eGenes = {_n_z_eg:,}\n"
        f"±{zoom_mb/2:.2g} Mb: {_pct_half:.0f}%\n"
        f"median |d|: {_med_z:.3f} Mb",
        transform=ax_zoom.transAxes,
        fontsize=8.5,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=INK,
            alpha=0.9,
        ),
    )
    sns.despine(ax=ax_zoom)
    apply_theme(ax_zoom)

    # ── panel 3: KDE — all scanned genes (background) vs eGenes (overlay) ────
    _eg_d = _dist[_is_eg & _hi]
    ax_kd.axvline(0, **_ref_kw)
    if _n > 2:
        sns.kdeplot(
            _dist,
            ax=ax_kd,
            color=MUTED,
            fill=True,
            alpha=0.25,
            linewidth=1.8,
            label=f"all scanned  ({_n:,})",
        )
    if len(_eg_d) > 2:
        sns.kdeplot(
            _eg_d,
            ax=ax_kd,
            color=BRAND_ACCENT,
            fill=True,
            alpha=0.35,
            linewidth=2.2,
            label=f"eGenes  ({_n_eg:,})",
        )
    ax_kd.set_xlabel("")
    ax_kd.set_ylabel("Density", fontsize=11)
    ax_kd.set_title("Cis distance distribution", fontsize=12, fontweight="bold")
    ax_kd.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax_kd)
    apply_theme(ax_kd)

    fig.supxlabel("Cis distance  (peak − TSS, ±Mb)", fontsize=11)
    return fig


def nb_vs_lmm_lod(
    lod_nb: np.ndarray,
    lod_lmm: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.4,
    point_size: float = 5.0,
) -> plt.Axes:
    """NB-GLMM vs LMM LOD scatter across loci.

    Parameters
    ----------
    lod_nb : np.ndarray
        NB-GLMM LOD scores per locus, shape (n,).
    lod_lmm : np.ndarray
        LMM LOD scores per locus, same shape.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    alpha : float
        Point opacity.
    point_size : float
        Scatter point size.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=300)

    sns.scatterplot(
        x=lod_lmm,
        y=lod_nb,
        s=point_size,
        alpha=alpha,
        color=BRAND_PRIMARY,
        linewidths=0,
        zorder=2,
        rasterized=True,
        ax=ax,
    )
    lim = max(float(lod_nb.max()), float(lod_lmm.max())) * 1.05
    _diag = np.array([0.0, lim])
    sns.lineplot(
        x=_diag,
        y=_diag,
        color=INK,
        linewidth=0.8,
        linestyle="--",
        errorbar=None,
        ax=ax,
        zorder=1,
    )
    r = float(np.corrcoef(lod_lmm, lod_nb)[0, 1])
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("LMM LOD Score")
    ax.set_ylabel("NB-GLMM LOD Score")
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def library_size_qc(
    library_sizes: np.ndarray,
    sample_ids: Optional[list[str]] = None,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Library size (total UMI) per strain — NB offset validity QC.

    Parameters
    ----------
    library_sizes : np.ndarray
        Total UMI counts per strain, shape (n_strains,).
    sample_ids : list[str], optional
        Strain labels for x-axis; omitted if n > 60.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), dpi=300)

    n = len(library_sizes)
    order = np.argsort(library_sizes)
    x = np.arange(n)

    sns.barplot(
        x=x,
        y=library_sizes[order],
        color=BRAND_PRIMARY,
        alpha=0.8,
        width=0.8,
        ax=ax,
        errorbar=None,
    )
    ax.axhline(
        float(np.median(library_sizes)),
        color=BRAND_SECONDARY,
        linewidth=0.9,
        linestyle="--",
        label="median",
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    if sample_ids is not None and n <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels([sample_ids[i] for i in order], rotation=90, fontsize=6)
    else:
        ax.set_xlabel("Strains (sorted by library size)")

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Total UMI")
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def loci_per_chr(
    chr_labels: list[str],
    n_loci: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of loci count per chromosome.

    Parameters
    ----------
    chr_labels : list[str]
        Chromosome names in display order.
    n_loci : np.ndarray
        Loci count per chromosome, same length as chr_labels.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3), dpi=300)
    x = np.arange(len(chr_labels))
    n_arr = np.asarray(n_loci, dtype=float)
    # gradient color: lighter for fewer loci, darker for more
    norm_counts = (n_arr - n_arr.min()) / max(n_arr.max() - n_arr.min(), 1.0)
    base = np.array(plt.matplotlib.colors.to_rgb(BRAND_PRIMARY))
    colors = [(1 - 0.45 * nc) * base + 0.45 * nc * np.zeros(3) for nc in norm_counts]
    ax.bar(x, n_arr, color=colors, width=0.7)
    # value labels on top of each bar
    y_max = float(n_arr.max())
    for xi, ni in zip(x, n_arr):
        ax.text(
            xi, ni + y_max * 0.01, f"{int(ni):,}", ha="center", va="bottom", fontsize=6
        )
    # mean reference line
    mean_loci = float(n_arr.mean())
    ax.axhline(mean_loci, color=INK, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(
        len(x) - 0.5,
        mean_loci + y_max * 0.01,
        f"mean={mean_loci:.0f}",
        fontsize=6,
        ha="right",
        color=INK,
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(chr_labels, fontsize=8)
    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel("Number of loci", fontsize=10)
    ax.set_ylim(bottom=0, top=y_max * 1.12)
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def perm_distribution(
    max_lods: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
    chrom: str = "",
    bins: int = 30,
    obs_lods: Optional[np.ndarray] = None,
    ref_gene: str = "",
    ref_chr: str = "",
) -> plt.Axes:
    """Permutation max-LOD histogram with GEV density overlay.

    Parameters
    ----------
    max_lods : np.ndarray
        Per-permutation genome-wide max LOD, shape (n_perms,).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    threshold : float, optional
        GEV 95th-percentile threshold; drawn as vertical line.
    chrom : str
        Reference chromosome label.
    bins : int
        Histogram bin count.
    obs_lods : np.ndarray, optional
        Observed per-gene peak LOD values; overlaid as a rug when provided.
    ref_gene : str
        Reference gene name shown in the info box.
    ref_chr : str
        Reference chromosome label shown in the info box (separate from
        *chrom* which labels the x-axis).

    Returns
    -------
    plt.Axes
    """
    from scipy.stats import genextreme

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=300)

    vals = max_lods[np.isfinite(max_lods)]
    if len(vals) == 0:
        ax.text(
            0.5,
            0.5,
            "no permutation data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=MUTED,
        )
        return ax

    from matplotlib.patches import Patch as _Patch

    _hist_clr = MUTED
    sns.histplot(
        vals,
        bins=bins,
        stat="density",
        color=_hist_clr,
        alpha=0.55,
        ax=ax,
        linewidth=0,
    )

    c, loc, scale = genextreme.fit(vals, f0=None)
    xr = np.linspace(vals.min() - 0.5, vals.max() + 1.5, 300)
    pdf_vals = genextreme.pdf(xr, c, loc, scale)
    sns.lineplot(
        x=xr,
        y=pdf_vals,
        color=BRAND_PRIMARY,
        linewidth=2.0,
        errorbar=None,
        ax=ax,
        label="GEV fit",
    )

    if threshold is not None:
        # value in legend label — not repeated in info box
        ax.axvline(
            threshold,
            color=BRAND_ACCENT,
            linewidth=1.4,
            linestyle=":",
            alpha=0.85,
            label=f"p95 = {threshold:.2f}",
        )
        tail_x = xr[xr >= threshold]
        ax.fill_between(
            tail_x,
            genextreme.pdf(tail_x, c, loc, scale),
            alpha=0.20,
            color=BRAND_ACCENT,
        )

    if obs_lods is not None:
        obs_f = obs_lods[np.isfinite(obs_lods)]
        sns.rugplot(x=obs_f, ax=ax, color=MUTED, alpha=0.5, height=0.04, zorder=0)

    # info box — count only; threshold value is in the legend label
    _parts: list[str] = [f"n_perms = {len(vals):,}"]
    if ref_gene:
        _parts.append(f"ref_gene = {ref_gene}")
    if ref_chr:
        _parts.append(f"ref_chr  = {ref_chr}")
    elif chrom:
        _parts.append(f"ref_chr  = {chrom}")
    ax.text(
        0.03,
        0.04,
        "\n".join(_parts),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontfamily="monospace",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=INK,
            alpha=0.88,
        ),
    )

    ax.set_xlabel("Max LOD Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    # add the histogram's legend swatch manually — the plotting library doesn't
    # always pick it up on its own
    _hist_proxy = _Patch(facecolor=_hist_clr, alpha=0.55, label="permutations")
    _h, _l = ax.get_legend_handles_labels()
    ax.legend(
        handles=[_hist_proxy] + _h,
        labels=["permutations"] + _l,
        fontsize=8,
        frameon=False,
        loc="upper right",
    )
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def egene_lod_distribution(
    peak_lods: np.ndarray,
    is_egene: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
    chrom: str = "",
    bins: int = 40,
) -> plt.Axes:
    """Stacked peak-LOD histogram coloured by eGene status.

    Parameters
    ----------
    peak_lods : np.ndarray
        Per-gene peak LOD, shape (n_genes,).
    is_egene : np.ndarray
        Boolean mask same length as peak_lods.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    threshold : float, optional
        Significance threshold — drawn as vertical line.
    chrom : str
        Chromosome label for title.
    bins : int
        Histogram bin count.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=300)

    n_total = int(len(peak_lods))
    n_egene = int(is_egene.sum())
    # shared bins so stacking is aligned
    shared_bins = np.histogram_bin_edges(peak_lods[np.isfinite(peak_lods)], bins=bins)
    below = peak_lods[~is_egene]
    above = peak_lods[is_egene]
    sns.histplot(
        below,
        bins=shared_bins,
        color=MUTED,
        alpha=0.70,
        ax=ax,
        linewidth=0,
        label="below threshold",
    )
    sns.histplot(
        above,
        bins=shared_bins,
        color=BRAND_PRIMARY,
        alpha=0.90,
        ax=ax,
        linewidth=1.1,
        edgecolor=INK,
        zorder=3,
        label="eGenes",
    )

    if threshold is not None:
        # value in legend label — not repeated in info box
        ax.axvline(
            threshold,
            color=BRAND_ACCENT,
            linewidth=1.4,
            linestyle=":",
            alpha=0.85,
            label=f"GEV 95%  ({threshold:.2f})",
        )

    ax.set_xlabel("Peak LOD Score", fontsize=11)
    ax.set_ylabel("Gene count", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_ylim(bottom=0)
    pct = 100.0 * n_egene / n_total if n_total > 0 else 0.0
    info_lines = [
        f"{pct:.1f}%  eGenes",
        f"eGenes  = {n_egene:,}",
        f"total   = {n_total:,}",
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=INK,
            alpha=0.9,
        ),
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("eGene LOD Score Distribution", fontsize=11)
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def mediation_profile(
    positions: np.ndarray,
    lod_base: np.ndarray,
    lod_mediated: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    gene: str = "",
) -> plt.Axes:
    """LOD profile with mediation overlay — base, mediated, and drop shading.

    Parameters
    ----------
    positions : np.ndarray
        Marker positions in Mb, shape (n_loci,).
    lod_base : np.ndarray
        Base (unmediated) LOD scores.
    lod_mediated : np.ndarray
        LOD after conditioning on mediator.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis.
    gene : str
        Gene name for title.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5), dpi=300)

    order = np.argsort(positions)
    pos = positions[order]
    base = lod_base[order]
    med = lod_mediated[order]

    sns.lineplot(
        x=pos,
        y=base,
        color=BRAND_PRIMARY,
        linewidth=1.5,
        errorbar=None,
        ax=ax,
        label="LOD Score (base)",
    )
    sns.lineplot(
        x=pos,
        y=med,
        color=MUTED,
        linewidth=1.5,
        linestyle="--",
        errorbar=None,
        ax=ax,
        label="LOD Score (mediated)",
    )
    drop = base - med
    ax.fill_between(
        pos,
        med,
        base,
        where=(base > med),
        alpha=0.22,
        color=BRAND_SECONDARY,
        label="LOD Score drop",
    )

    # annotate peak drop position
    if drop.max() > 0:
        peak_idx = int(np.argmax(drop))
        peak_pos_drop = float(pos[peak_idx])
        peak_base_lod = float(base[peak_idx])
        peak_drop_lod = float(drop[peak_idx])
        prop_med = peak_drop_lod / max(peak_base_lod, 1e-9)
        ax.annotate(
            f"Δ={peak_drop_lod:.1f}  ({prop_med * 100:.0f}% mediated)",
            (peak_pos_drop, (float(base[peak_idx]) + float(med[peak_idx])) / 2),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=BRAND_SECONDARY,
            va="center",
        )
        ax.axvline(
            peak_pos_drop,
            color=BRAND_SECONDARY,
            linewidth=0.8,
            linestyle=":",
            alpha=0.6,
        )

    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)", fontsize=11
    )
    ax.set_ylabel("LOD Score", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    title = gene if gene else "Mediation"
    if chrom:
        title += f"  chr{chrom}"
    ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def ci_strip(
    genes: list[str],
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    peak_mb: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
) -> plt.Axes:
    """Horizontal CI strips per gene (LOD support interval style).

    Parameters
    ----------
    genes : list[str]
        Gene names, one per row.
    ci_lo : np.ndarray
        CI lower bound in Mb, shape (n_genes,).
    ci_hi : np.ndarray
        CI upper bound in Mb, shape (n_genes,).
    peak_mb : np.ndarray
        Peak position in Mb, shape (n_genes,).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis.

    Returns
    -------
    plt.Axes
    """
    n = len(genes)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, n * 0.38)), dpi=300)

    widths = ci_hi - ci_lo
    med_width = float(np.nanmedian(widths))
    # color bars by CI width: narrow (cool blue) → wide (warm red)
    cmap = plt.matplotlib.colormaps.get_cmap("RdYlBu_r")
    w_max = float(np.nanmax(widths))
    w_min = float(np.nanmin(widths))
    w_range = max(w_max - w_min, 1e-9)

    for i, (lo, hi, pk) in enumerate(zip(ci_lo, ci_hi, peak_mb)):
        norm_w = (float(hi - lo) - w_min) / w_range
        bar_color = cmap(norm_w)
        ax.plot([lo, hi], [i, i], color=bar_color, linewidth=3, solid_capstyle="round")
        sns.scatterplot(x=[pk], y=[i], color=INK, s=18, zorder=3, ax=ax)

    # vertical line marking the median confidence-interval width
    ax.axvline(
        float(np.nanmean(peak_mb)) + med_width / 2,
        color=INK,
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
    )
    ax.text(
        float(np.nanmean(peak_mb)) + med_width / 2 + 0.1,
        n - 0.5,
        f"med width={med_width:.1f} Mb",
        fontsize=7,
        color=INK,
        va="top",
    )

    # annotate widest-CI genes by name
    widest_idx = np.argsort(widths)[::-1][: min(3, n)]
    for wi in widest_idx:
        ax.text(
            float(ci_hi[wi]) + 0.1,
            wi,
            genes[wi],
            fontsize=6,
            va="center",
            color=BRAND_SECONDARY,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlabel(
        f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)", fontsize=11
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(f"Bootstrap CI  {n} eGenes", fontsize=11)
    # colorbar for CI width
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=w_min, vmax=w_max),
    )
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("CI width (Mb)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def snp_lod_profile(
    positions: np.ndarray,
    snp_lod: np.ndarray,
    beta_dosage: np.ndarray,
    *,
    haplo_positions: Optional[np.ndarray] = None,
    haplo_lod: Optional[np.ndarray] = None,
    peak_pos: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    gene: str = "",
) -> plt.Axes:
    """Single-chromosome SNP fine-mapping LOD profile with dosage effect colorbar.

    Args:
        positions: SNP positions in Mb, shape (n_snps,).
        snp_lod: SNP LOD scores, shape (n_snps,).
        beta_dosage: Estimated dosage effect per SNP, shape (n_snps,).
        haplo_positions: Haplotype scan positions in Mb; when provided, draws a
            grey filled area behind the SNP scatter.
        haplo_lod: Haplotype LOD scores matching haplo_positions.
        peak_pos: Peak SNP position — annotated with a star marker.
        ax: Existing Axes; creates a new figure (dpi=300) if None.
        chrom: Chromosome label for x-axis.
        gene: Gene name shown as bold corner text.

    Returns:
        The Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5), dpi=300)

    if haplo_positions is not None and haplo_lod is not None:
        ord_h = np.argsort(haplo_positions)
        ax.fill_between(
            haplo_positions[ord_h],
            haplo_lod[ord_h],
            alpha=0.12,
            color=MUTED,
            zorder=1,
            label="haplotype LOD",
        )

    ord_s = np.argsort(positions)
    pos_s = positions[ord_s]
    lod_s = snp_lod[ord_s]
    bet_s = beta_dosage[ord_s]

    sc = ax.scatter(
        pos_s, lod_s, c=bet_s, cmap=_CMAP_EFFECT, s=8, linewidths=0, alpha=0.7, zorder=3
    )
    cbar = ax.get_figure().colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("β dosage", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if peak_pos is not None:
        pk_idx = int(np.argmin(np.abs(pos_s - peak_pos)))
        pk_lod = float(lod_s[pk_idx])
        ax.scatter(
            [peak_pos],
            [pk_lod],
            marker="*",
            s=120,
            color=BRAND_SECONDARY,
            zorder=5,
        )

    if gene:
        ax.text(
            0.02,
            0.96,
            gene,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            fontweight="bold",
            color=BRAND_PRIMARY,
        )

    ax.set_xlabel(
        f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_xlim(float(pos_s[0]), float(pos_s[-1]))
    ax.set_ylim(0, float(lod_s.max()) * 1.18)
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def instability_lod_overlay(
    positions: np.ndarray,
    rop_lod: np.ndarray,
    inst_scores: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    gene: str = "",
    peak_pos: Optional[float] = None,
    threshold: Optional[float] = None,
) -> plt.Axes:
    """Overlay instability scan score against ROP LOD on a shared position axis.

    Args:
        positions: Locus positions in Mb, shape (n_loci,).
        rop_lod: ROP LOD scores, shape (n_loci,).
        inst_scores: Instability scores (0–1), shape (n_loci,); plotted on
            a secondary y-axis (right side).
        ax: Primary Axes; creates a new figure (dpi=300) if None.
        chrom: Chromosome label for x-axis.
        gene: Gene name shown as bold corner text.
        peak_pos: Peak LOD position — vertical dotted line.
        threshold: ROP permutation threshold — horizontal dashed line.

    Returns:
        The primary (LOD) Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5), dpi=300)

    ord_i = np.argsort(positions)
    pos = positions[ord_i]
    rop = rop_lod[ord_i]
    inst = inst_scores[ord_i]

    ax.fill_between(pos, rop, alpha=0.15, color=BRAND_PRIMARY, zorder=1)
    sns.lineplot(
        x=pos,
        y=rop,
        color=BRAND_PRIMARY,
        linewidth=1.2,
        errorbar=None,
        ax=ax,
        zorder=2,
        label="ROP LOD",
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            color=BRAND_ACCENT,
            lw=1.0,
            ls=":",
            alpha=0.85,
            zorder=0,
        )

    ax2 = ax.twinx()
    ax2.fill_between(pos, inst, alpha=0.10, color=_INST_COLOR, zorder=1)
    sns.lineplot(
        x=pos,
        y=inst,
        color=_INST_COLOR,
        linewidth=1.0,
        linestyle="--",
        errorbar=None,
        ax=ax2,
        zorder=2,
        label="instability",
    )
    ax2.set_ylabel("Instability Score", fontsize=11, color=_INST_COLOR)
    ax2.tick_params(labelsize=9, colors=_INST_COLOR)
    ax2.set_ylim(0, max(float(inst.max()) * 1.3, 0.2))

    if peak_pos is not None:
        ax.axvline(peak_pos, color=BRAND_SECONDARY, lw=0.8, ls=":", zorder=3)

    if gene:
        ax.text(
            0.02,
            0.96,
            gene,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            fontweight="bold",
            color=BRAND_PRIMARY,
        )

    # concordance info box
    from scipy.stats import spearmanr

    rho, _ = spearmanr(rop, inst)
    ax.text(
        0.98,
        0.97,
        f"inst–LOD ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontfamily="monospace",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=INK,
            alpha=0.9,
        ),
    )

    ax.set_xlabel(
        f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("ROP LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(0, float(rop.max()) * 1.18)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend goes lower-left since the gene name and info box already fill the top corners
    ax.legend(
        lines1 + lines2, labels1 + labels2, fontsize=9, frameon=False, loc="lower left"
    )
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def mi_scan_overlay(
    positions: np.ndarray,
    scan_lod: np.ndarray,
    mi_lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    gene: str = "",
    peak_pos: Optional[float] = None,
    threshold: Optional[float] = None,
) -> plt.Axes:
    """Overlay MI LOD profile against the base scan LOD on a shared position axis.

    Args:
        positions: Locus positions in Mb, shape (n_loci,).
        scan_lod: Base scan LOD scores (additive haplotype scan), shape (n_loci,).
        mi_lod: Multiple-imputation LOD scores, shape (n_loci,).
        ax: Axes; creates a new figure (dpi=300) if None.
        chrom: Chromosome label for x-axis.
        gene: Gene name shown as bold corner text.
        peak_pos: Shared peak position annotated as dotted vertical line.
        threshold: Permutation threshold annotated as dashed horizontal line.

    Returns:
        The Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5), dpi=300)

    ord_i = np.argsort(positions)
    pos = positions[ord_i]
    scan = scan_lod[ord_i]
    mi = mi_lod[ord_i]

    ax.fill_between(pos, scan, alpha=0.12, color=BRAND_PRIMARY, zorder=1)
    sns.lineplot(
        x=pos,
        y=scan,
        color=BRAND_PRIMARY,
        linewidth=1.2,
        errorbar=None,
        ax=ax,
        zorder=2,
        label="Scan",
    )
    ax.fill_between(pos, mi, alpha=0.08, color=_MI_COLOR, zorder=1)
    sns.lineplot(
        x=pos,
        y=mi,
        color=_MI_COLOR,
        linewidth=1.0,
        linestyle="--",
        errorbar=None,
        ax=ax,
        zorder=2,
        label="MI",
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            color=BRAND_ACCENT,
            lw=1.0,
            ls=":",
            alpha=0.85,
            zorder=0,
        )

    if peak_pos is not None:
        ax.axvline(peak_pos, color=BRAND_SECONDARY, lw=0.8, ls=":", zorder=3)

    if gene:
        ax.text(
            0.02,
            0.96,
            gene,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            fontweight="bold",
            color=BRAND_PRIMARY,
        )

    delta = float(np.max(mi) - np.max(scan))
    ax.text(
        0.98,
        0.97,
        f"ΔLODpeak = {delta:+.2f}\nScan peak = {float(np.max(scan)):.2f}\n"
        f"MI   peak = {float(np.max(mi)):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontfamily="monospace",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=INK,
            alpha=0.9,
        ),
    )

    ax.set_xlabel(
        f"Chr {chrom} position (Mb)" if chrom else "Position (Mb)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(0, max(float(np.concatenate([scan, mi]).max()) * 1.2, 1.0))
    # legend goes lower-left since the gene name and info box already fill the top corners
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def perm_threshold_comparison(
    py_thresholds: dict[str, float],
    r_thresholds: Optional[dict[str, float]] = None,
    *,
    genome_wide_thr: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of per-chromosome FWER thresholds, Python vs R.

    Args:
        py_thresholds: ``{chrom: threshold_lod}`` from Python GEV fit.
        r_thresholds: ``{chrom: threshold_lod}`` from R GEV fit; drawn as a
            second grouped bar when provided.
        genome_wide_thr: Genome-wide FWER threshold drawn as a dashed horizontal
            line when provided.
        ax: Axes; creates a new figure (dpi=300) if None.

    Returns:
        The Axes containing the plot.
    """
    from .compare import (
        _COLOR_A as _CPY,
        _COLOR_B as _CR,
    )  # local import avoids circular

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5), dpi=300)

    chrs = [c for c in _CHR_ORDER if c in py_thresholds]
    extra = sorted(set(py_thresholds) - set(chrs))
    chrs = chrs + extra

    x = np.arange(len(chrs))
    w = 0.35 if r_thresholds else 0.6
    offset = -w / 2 if r_thresholds else 0.0

    py_vals = np.array([py_thresholds.get(c, float("nan")) for c in chrs])
    ax.bar(x + offset, py_vals, width=w, color=_CPY, alpha=0.8, label="Python")

    if r_thresholds:
        r_vals = np.array([r_thresholds.get(c, float("nan")) for c in chrs])
        ax.bar(x + w / 2, r_vals, width=w, color=_CR, alpha=0.8, label="R")

    if genome_wide_thr is not None:
        ax.axhline(
            genome_wide_thr,
            color=INK,
            lw=1.4,
            ls="--",
            label=f"genome-wide={genome_wide_thr:.2f}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"chr{c}" for c in chrs], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Chromosome", fontsize=13, fontweight="bold")
    ax.set_ylabel("FWER Threshold (LOD)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def ci_genomic_interval(
    genes: list[str],
    chroms: list[str],
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    peak_lod: np.ndarray,
    peak_mb: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chr_offsets: Optional[dict[str, float]] = None,
) -> plt.Axes:
    """Manhattan-style CI display: LOD-height vertical segments per gene.

    Args:
        genes: Gene names, shape (n_genes,).
        chroms: Chromosome per gene, shape (n_genes,).
        ci_lo: CI lower bound in Mb, shape (n_genes,).
        ci_hi: CI upper bound in Mb, shape (n_genes,).
        peak_lod: Peak LOD per gene — segment height, shape (n_genes,).
        peak_mb: Peak position in Mb — segment x-position, shape (n_genes,).
        ax: Axes; creates a new figure (dpi=300) if None.
        chr_offsets: ``{chrom: offset_mb}`` for genome-wide layout; computed
            automatically from peak positions when None.

    Returns:
        The Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4.5), dpi=300)

    chrs_list = list(dict.fromkeys(c for c in _CHR_ORDER if c in set(chroms))) + sorted(
        set(chroms) - set(_CHR_ORDER)
    )

    if chr_offsets is None:
        off = 0.0
        chr_offsets = {}
        for c in chrs_list:
            mask = [ch == c for ch in chroms]
            pks = peak_mb[mask] if any(mask) else np.array([0.0])
            chr_offsets[c] = off
            off += float(np.nanmax(pks)) + 15.0

    widths = ci_hi - ci_lo
    cmap = plt.matplotlib.colormaps.get_cmap("RdYlBu_r")
    w_max = float(np.nanmax(widths))
    w_min = float(np.nanmin(widths))
    w_range = max(w_max - w_min, 1e-9)

    xticks: list[float] = []
    xlabels: list[str] = []

    for ci_, (chrom, lo, hi, lod_v, pk) in enumerate(
        zip(chroms, ci_lo, ci_hi, peak_lod, peak_mb)
    ):
        off = chr_offsets.get(str(chrom), 0.0)
        x = pk + off
        norm_w = (float(hi - lo) - w_min) / w_range
        bar_color = cmap(norm_w)
        half_w = float(hi - lo) / 2.0
        ax.vlines(x, 0, lod_v, color=bar_color, lw=2, alpha=0.8)
        ax.plot(
            [x - half_w, x + half_w],
            [0, 0],
            color=bar_color,
            lw=0.4,
            alpha=0.3,
        )

    for c in chrs_list:
        off = chr_offsets.get(c, 0.0)
        mask = np.array([ch == c for ch in chroms])
        if mask.any():
            mid = float(peak_mb[mask].mean()) + off
            xticks.append(mid)
            xlabels.append(f"chr{c}")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak LOD", fontsize=13, fontweight="bold")
    ax.set_xlabel("Genomic Position (Mb, cumulative)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cb = ax.get_figure().colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("CI width (Mb)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def scan_throughput(
    chr_labels: list[str],
    py_genes_per_min: np.ndarray,
    *,
    r_genes_per_min: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    py_total_label: str = "",
    r_total_label: str = "",
) -> plt.Axes:
    """Bar chart of per-chromosome scan throughput in genes/min.

    Args:
        chr_labels: Chromosome labels, shape (n_chr,).
        py_genes_per_min: Python throughput per chromosome, shape (n_chr,).
        r_genes_per_min: R throughput per chromosome; second grouped bar
            when provided.
        ax: Axes; creates a new figure (dpi=300) if None.
        py_total_label: Summary text for the Python total (e.g. '259 genes/min').
        r_total_label: Summary text for the R total.

    Returns:
        The Axes containing the plot.
    """
    from .compare import _COLOR_A as _CPY, _COLOR_B as _CR

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5), dpi=300)

    x = np.arange(len(chr_labels))
    w = 0.35 if r_genes_per_min is not None else 0.6
    offset = -w / 2 if r_genes_per_min is not None else 0.0

    ax.bar(
        x + offset,
        py_genes_per_min,
        width=w,
        color=_CPY,
        alpha=0.85,
        label=f"Python  {py_total_label}".strip(),
    )

    if r_genes_per_min is not None:
        ax.bar(
            x + w / 2,
            r_genes_per_min,
            width=w,
            color=_CR,
            alpha=0.85,
            label=f"R (miQTL)  {r_total_label}".strip(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"chr{c}" for c in chr_labels], rotation=45, ha="right", fontsize=9
    )
    ax.set_xlabel("Chromosome", fontsize=13, fontweight="bold")
    ax.set_ylabel("Throughput (genes/min)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def mediation_summary_bar(
    chrs: list[str],
    mean_drop: np.ndarray,
    std_drop: Optional[np.ndarray] = None,
    n_pairs: Optional[np.ndarray] = None,
    *,
    n_outcomes: int = 0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Per-chromosome bar chart of mean mediation LOD-score drop.

    Parameters
    ----------
    chrs : list[str]
        Chromosome labels in display order (already sorted by caller).
    mean_drop : np.ndarray
        Mean LOD-score drop per chromosome.
    std_drop : np.ndarray, optional
        Standard deviation of LOD-score drop; drawn as error bars.
    n_pairs : np.ndarray, optional
        Number of (outcome, mediator) pairs per chromosome; annotated above bars.
    n_outcomes : int
        Total number of outcome genes; shown in info box.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, len(chrs) * 0.6), 4.5), dpi=300)

    x = np.arange(len(chrs))
    yerr = std_drop if std_drop is not None else None
    ax.bar(
        x,
        mean_drop,
        color=BRAND_PRIMARY,
        alpha=0.85,
        edgecolor="none",
        yerr=yerr,
        error_kw={"linewidth": 0.8, "ecolor": INK, "capsize": 3},
    )

    bar_max = float(np.nanmax(mean_drop)) if len(mean_drop) else 1.0
    if n_pairs is not None:
        for xi, np_val in enumerate(n_pairs):
            ax.text(
                xi,
                float(mean_drop[xi]) + bar_max * 0.02,
                f"n={int(np_val)}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=INK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(chrs, rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("Chromosome", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean LOD Score Drop (±1 SD)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0, top=bar_max * 1.25)

    _info = f"outcomes  = {n_outcomes:,}" if n_outcomes else ""
    if _info:
        ax.text(
            0.98,
            0.97,
            _info,
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=INK,
                alpha=0.9,
            ),
        )
    sns.despine(ax=ax)
    apply_theme(ax)
    return ax


def timing_memory_summary(
    labels: list[str],
    elapsed_s: np.ndarray,
    peak_heap_mb: Optional[np.ndarray] = None,
    *,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Horizontal bar chart of per-section wall time and optional peak heap.

    Parameters
    ----------
    labels : list[str]
        Section labels in display order (already sorted by caller).
    elapsed_s : np.ndarray
        Wall-clock elapsed seconds per section.
    peak_heap_mb : np.ndarray, optional
        Peak tracemalloc heap allocation in MB per section; second panel when provided.
    ax : plt.Axes, optional
        Ignored — always creates its own figure to support optional dual panels.
        Accepted for API consistency; pass None.

    Returns
    -------
    plt.Figure
        Figure with one or two horizontal bar panels.
    """
    has_mem = peak_heap_mb is not None and np.any(peak_heap_mb > 0)
    fig, axes = plt.subplots(
        1,
        2 if has_mem else 1,
        figsize=(9 if has_mem else 5, max(3, len(labels) * 0.45)),
        dpi=300,
    )
    ax_t = axes[0] if has_mem else axes

    elapsed_s = np.asarray(elapsed_s, dtype=float)
    sns.barplot(
        x=elapsed_s,
        y=labels,
        orient="h",
        color=BRAND_PRIMARY,
        alpha=0.85,
        edgecolor="none",
        ax=ax_t,
    )
    # per-section timings routinely span two-plus orders of magnitude (e.g. the
    # genome-wide LMM scan vs. a permutation pass) — a shared linear x-axis
    # reduces every smaller bar to an invisible hairline, so log-scale whenever
    # the spread warrants it and label each bar with its exact value.
    _finite = elapsed_s[elapsed_s > 0]
    if len(_finite) > 1 and _finite.max() / _finite.min() >= 20:
        ax_t.set_xscale("log")
    for _i, _v in enumerate(elapsed_s):
        ax_t.text(
            _v, _i, f"  {_v:,.1f}s", va="center", ha="left", fontsize=8.5, color=INK
        )
    ax_t.set_xlabel("Elapsed (s)", fontsize=13, fontweight="bold")
    ax_t.tick_params(labelsize=11)
    sns.despine(ax=ax_t)

    if has_mem:
        heap = peak_heap_mb.astype(float)
        sns.barplot(
            x=heap,
            y=labels,
            orient="h",
            color=MUTED,
            alpha=0.85,
            edgecolor="none",
            ax=axes[1],
        )
        for _i, _v in enumerate(heap):
            axes[1].text(
                _v,
                _i,
                f"  {_v:,.0f} MB",
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK,
            )
        axes[1].set_xlabel("Peak heap (MB)", fontsize=13, fontweight="bold")
        axes[1].tick_params(labelsize=11)
        axes[1].margins(x=0.12)
        sns.despine(ax=axes[1])

    ax_t.margins(x=0.12)
    apply_theme(ax_t)
    return fig


def chromosome_lod_profile(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    threshold: Optional[float] = None,
    color: str = BRAND_PRIMARY,
    fill: bool = True,
    annotate_n_loci: bool = True,
) -> plt.Axes:
    """Whole-chromosome LOD profile with auto-detected peak and loci count.

    Convenience wrapper around :func:`lod_profile` that highlights the peak
    position and optionally annotates the number of loci tested.

    Parameters
    ----------
    positions : np.ndarray
        Marker positions in Mb, shape (n_loci,).
    lod : np.ndarray
        LOD scores, same shape as positions.
    ax : plt.Axes, optional
        Axes to plot on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis title.
    threshold : float, optional
        Permutation threshold drawn as a dashed horizontal line.
    color : str
        Line and fill color; must be imported from founders.py.
    fill : bool
        Shade area under the LOD curve.
    annotate_n_loci : bool
        Add a text annotation with the loci count in the upper-left corner.

    Returns
    -------
    plt.Axes
    """
    peak_idx = int(np.argmax(lod))
    peak_pos = float(positions[peak_idx])
    ax = lod_profile(
        positions,
        lod,
        ax=ax,
        chrom=chrom,
        threshold=threshold,
        peak_pos=peak_pos,
        color=color,
        fill=fill,
    )
    if annotate_n_loci:
        ax.text(
            0.02,
            0.97,
            f"n={len(positions):,} loci",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color=INK,
        )
    return ax


def peak_locus_zoom(
    positions: np.ndarray,
    lod: np.ndarray,
    peak_mb: float,
    *,
    window_mb: float = 2.0,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
    threshold: Optional[float] = None,
    ci_lo_mb: Optional[float] = None,
    ci_hi_mb: Optional[float] = None,
    color: str = BRAND_PRIMARY,
    snp_positions: Optional[np.ndarray] = None,
    snp_neg_log10p: Optional[np.ndarray] = None,
) -> plt.Axes:
    """High-resolution LOD profile zoomed around a peak locus.

    Slices to [peak_mb - window_mb, peak_mb + window_mb] and optionally
    shades the confidence interval and overlays SNP association scores.

    Parameters
    ----------
    positions : np.ndarray
        Full-chromosome marker positions in Mb.
    lod : np.ndarray
        Full-chromosome LOD scores.
    peak_mb : float
        Peak locus position in Mb; centre of the zoom window.
    window_mb : float
        Half-width of the zoom window in Mb.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis.
    threshold : float, optional
        Permutation threshold line.
    ci_lo_mb : float, optional
        Lower bound of LOD-support confidence interval in Mb.
    ci_hi_mb : float, optional
        Upper bound of LOD-support confidence interval in Mb.
    color : str
        LOD line color; from founders.py.
    snp_positions : np.ndarray, optional
        SNP positions in Mb for secondary-axis overlay.
    snp_neg_log10p : np.ndarray, optional
        −log10(p) values per SNP; requires snp_positions.

    Returns
    -------
    plt.Axes
        Primary axes (LOD); secondary axes accessible via ax.get_figure().
    """
    start_mb = peak_mb - window_mb
    end_mb = peak_mb + window_mb
    ax = region_zoom(
        positions,
        lod,
        start_mb=start_mb,
        end_mb=end_mb,
        ax=ax,
        chrom=chrom,
        threshold=threshold,
        color=color,
        snp_positions=snp_positions,
        snp_neg_log10p=snp_neg_log10p,
    )
    if ci_lo_mb is not None and ci_hi_mb is not None:
        ax.axvspan(
            ci_lo_mb,
            ci_hi_mb,
            alpha=0.12,
            color=color,
            zorder=0,
        )
    mask = (positions >= start_mb) & (positions <= end_mb)
    if mask.any():
        peak_lod_val = float(lod[mask].max())
        ax.text(
            peak_mb,
            peak_lod_val * 1.05,
            f"LOD={peak_lod_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=BRAND_SECONDARY,
        )
    return ax


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _chr_sort_key_local(c: str) -> tuple[int, int]:
    """Numeric chromosome sort matching _CHR_ORDER (1–19, then X)."""
    try:
        return (0, _CHR_ORDER.index(c))
    except ValueError:
        try:
            return (1, int(c))
        except ValueError:
            return (2, 0)


def _ordered_chrs(chr_series: pd.Series) -> list[str]:
    """Return unique chromosomes in _CHR_ORDER then extras alphabetically."""
    present = set(chr_series.astype(str))
    ordered = [c for c in _CHR_ORDER if c in present]
    extra = sorted(present - set(_CHR_ORDER))
    return ordered + extra


# ---------------------------------------------------------------------------
# per-chromosome figure iterators
# ---------------------------------------------------------------------------


def lod_profiles_per_chr(
    lod_df: pd.DataFrame,
    pk_df: Optional[pd.DataFrame] = None,
    thr_by_chr: Optional[dict[str, Optional[float]]] = None,
    *,
    color: str = BRAND_PRIMARY,
    ylabel: str = "LOD Score",
    priority_genes: Optional[dict[str, str]] = None,
) -> dict[str, plt.Figure]:
    """Per-chromosome LOD profile figures for the top gene on each chromosome.

    Reusable for the main LOD scan (S4), MI scan (S7 with color=_MI_COLOR),
    and instability scan (S11 with color=_INST_COLOR / ylabel='Instability LOD').

    Parameters
    ----------
    lod_df : pd.DataFrame
        Required columns: gene (str), chr (str), pos_mb (float), lod (float).
    pk_df : pd.DataFrame, optional
        Peak table (gene, chr, peak_lod) for top-gene selection.
        Defaults to highest mean LOD gene per chromosome if omitted.
    thr_by_chr : dict, optional
        Per-chromosome permutation threshold (may contain None values).
    color : str
        LOD line colour; must be imported from genotype.cc.
    ylabel : str
        Y-axis label.
    priority_genes : dict, optional
        {chr: gene_symbol} annotated in the title when not the top gene.

    Returns
    -------
    dict[str, plt.Figure]
        Keyed by chromosome string.
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    chr_vals = lod_df["chr"].astype(str)
    chrs = _ordered_chrs(chr_vals)

    out: dict[str, plt.Figure] = {}
    for c in chrs:
        lod_c = lod_df[chr_vals == c]
        if lod_c.empty:
            continue

        if pk_df is not None:
            pk_c = pk_df[pk_df["chr"].astype(str) == c]
            top_gene = (
                str(pk_c.loc[pk_c["peak_lod"].idxmax(), "gene_symbol"])
                if not pk_c.empty
                else ""
            )
        else:
            gene_max = lod_c.groupby("gene_symbol")["lod"].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ""

        if not top_gene:
            continue
        lod_top = lod_c[lod_c["gene_symbol"] == top_gene].sort_values("pos_mb")
        if lod_top.empty:
            continue

        thr_val: Optional[float] = thr.get(c)
        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
        lod_profile(
            lod_top["pos_mb"].values,
            lod_top["lod"].values,
            ax=ax,
            chrom=c,
            threshold=thr_val,
            color=color,
            fill=True,
        )
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        title = f"{top_gene}  chr{c}"
        p1 = prio.get(c)
        if p1 and p1 != top_gene:
            p1_data = lod_c[lod_c["gene_symbol"] == p1].sort_values("pos_mb")
            if not p1_data.empty:
                p1_max = float(p1_data["lod"].max())
                sns.lineplot(
                    x=p1_data["pos_mb"].values,
                    y=p1_data["lod"].values,
                    color=BRAND_SECONDARY,
                    linewidth=1.2,
                    alpha=0.80,
                    errorbar=None,
                    ax=ax,
                    label=f"{p1}  LOD={p1_max:.1f}",
                    zorder=3,
                )
                ax.legend(fontsize=9, loc="upper right", frameon=False)
                title += f"  /  {p1}"
        ax.set_title(title, fontsize=13)
        sns.despine(ax=ax)
        fig.tight_layout()
        out[c] = fig

    return out


def perm_distributions_per_chr(
    perm_by_chr: dict[str, np.ndarray],
    thr_by_chr: dict[str, Optional[float]],
) -> dict[str, plt.Figure]:
    """Per-chromosome permutation null distribution figures.

    Parameters
    ----------
    perm_by_chr : dict[str, np.ndarray]
        Max-LOD null distribution per chromosome.
    thr_by_chr : dict[str, float | None]
        GEV threshold per chromosome.

    Returns
    -------
    dict[str, plt.Figure]
    """
    out: dict[str, plt.Figure] = {}
    for c in sorted(perm_by_chr, key=_chr_sort_key_local):
        max_lods = perm_by_chr[c]
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        perm_distribution(max_lods, ax=ax, threshold=thr_by_chr.get(c), chrom=c)
        sns.despine(ax=ax)
        fig.tight_layout()
        out[c] = fig
    return out


def snp_profiles_per_chr(
    snp_df: pd.DataFrame,
    pk_df: Optional[pd.DataFrame] = None,
    *,
    priority_genes: Optional[dict[str, str]] = None,
) -> dict[str, plt.Figure]:
    """Per-chromosome SNP fine-mapping association scatter figures.

    Parameters
    ----------
    snp_df : pd.DataFrame
        Columns: gene (str), chr (str), pos_mb (float), lod (float).
    pk_df : pd.DataFrame, optional
        Peak table for top-gene selection when snp_df covers multiple genes.
    priority_genes : dict, optional
        {chr: gene_symbol} forces the plotted gene per chromosome.

    Returns
    -------
    dict[str, plt.Figure]
    """
    prio = priority_genes or {}
    chr_vals = snp_df["chr"].astype(str)
    chrs = _ordered_chrs(chr_vals)

    out: dict[str, plt.Figure] = {}
    for c in chrs:
        snp_c = snp_df[chr_vals == c]
        if snp_c.empty:
            continue

        if prio.get(c) and prio[c] in snp_c["gene_symbol"].values:
            top_gene = prio[c]
        elif pk_df is not None:
            pk_c = pk_df[pk_df["chr"].astype(str) == c]
            top_gene = (
                str(pk_c.loc[pk_c["peak_lod"].idxmax(), "gene_symbol"])
                if not pk_c.empty
                else ""
            )
        else:
            gene_max = snp_c.groupby("gene_symbol")["lod"].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ""

        if not top_gene:
            continue
        snp_g = snp_c[snp_c["gene_symbol"] == top_gene].sort_values("pos_mb")
        if snp_g.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
        sns.scatterplot(
            x=snp_g["pos_mb"].values,
            y=snp_g["lod"].values,
            color=_SNP_COLOR,
            s=5,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            zorder=2,
            ax=ax,
        )
        if len(snp_g) > 0:
            top_row = snp_g.loc[snp_g["lod"].idxmax()]
            snp_label = str(top_row.get("snp", ""))
            if snp_label:
                ax.annotate(
                    snp_label,
                    xy=(top_row["pos_mb"], top_row["lod"]),
                    xytext=(8, 6),
                    textcoords="offset points",
                    fontsize=10,
                    color=BRAND_SECONDARY,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=BRAND_SECONDARY,
                        lw=0.8,
                    ),
                )
        ax.set_xlim(
            float(snp_g["pos_mb"].min()),
            float(snp_g["pos_mb"].max()),
        )
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"Chr {c} position (Mb)", fontsize=13)
        ax.set_ylabel("LOD Score (SNP)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_title(f"SNP fine-mapping  —  {top_gene}  chr{c}", fontsize=13)
        sns.despine(ax=ax)
        fig.tight_layout()
        out[c] = fig

    return out


def lod_panel_grid(
    lod_df: pd.DataFrame,
    pk_df: Optional[pd.DataFrame] = None,
    thr_by_chr: Optional[dict[str, Optional[float]]] = None,
    *,
    priority_genes: Optional[dict[str, str]] = None,
    n_cols: int = 4,
) -> plt.Figure:
    """Grid of LOD profiles, one panel per chromosome (LOD-only diagnostic).

    Parameters
    ----------
    lod_df : pd.DataFrame
        Columns: gene_symbol, chr (str), pos_mb, lod.
    pk_df : pd.DataFrame, optional
        Peak table for top-gene selection.
    thr_by_chr : dict, optional
        Per-chromosome permutation threshold.
    priority_genes : dict, optional
        {chr: gene_symbol} annotated in panel titles.
    n_cols : int
        Panels per row (default 4 → 4×5 for 20 chromosomes).

    Returns
    -------
    plt.Figure
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    chr_vals = lod_df["chr"].astype(str)
    chrs = _ordered_chrs(chr_vals)
    n_chrs = len(chrs)
    n_rows = (n_chrs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3, n_rows * 2.2),
        dpi=300,
        squeeze=False,
    )

    for i, c in enumerate(chrs):
        ax = axes[i // n_cols, i % n_cols]
        lod_c = lod_df[chr_vals == c]

        if pk_df is not None:
            pk_c = pk_df[pk_df["chr"].astype(str) == c]
            top_gene = (
                str(pk_c.loc[pk_c["peak_lod"].idxmax(), "gene_symbol"])
                if not pk_c.empty
                else ""
            )
        else:
            gene_max = lod_c.groupby("gene_symbol")["lod"].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ""

        if top_gene:
            lod_top = lod_c[lod_c["gene_symbol"] == top_gene].sort_values("pos_mb")
            if not lod_top.empty:
                lod_profile(
                    lod_top["pos_mb"].values,
                    lod_top["lod"].values,
                    ax=ax,
                    chrom="",
                    threshold=thr.get(c),
                    fill=True,
                    linewidth=0.8,
                )
                label = prio.get(c) or top_gene
                ax.set_title(f"chr{c}  {label[:14]}", fontsize=7)
                ax.set_xlabel(f"chr{c} (Mb)", fontsize=6)
                ax.tick_params(labelsize=5)
                sns.despine(ax=ax)
                continue
        ax.set_visible(False)

    for j in range(n_chrs, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle("LOD Score Profiles — Top Gene per Chromosome", fontsize=9, y=1.01)
    fig.tight_layout(h_pad=1.5)
    return fig


def perm_panel_grid(
    perm_by_chr: dict[str, np.ndarray],
    thr_by_chr: dict[str, Optional[float]],
    *,
    n_cols: int = 4,
) -> plt.Figure:
    """Grid of permutation null distributions, one panel per chromosome.

    Parameters
    ----------
    perm_by_chr : dict[str, np.ndarray]
        Max-LOD null distribution per chromosome.
    thr_by_chr : dict[str, float | None]
        GEV threshold per chromosome.
    n_cols : int
        Panels per row.

    Returns
    -------
    plt.Figure
    """
    chrs = sorted(perm_by_chr, key=_chr_sort_key_local)
    n_chrs = len(chrs)
    n_rows = (n_chrs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.5, n_rows * 2.5),
        dpi=300,
        squeeze=False,
    )

    for i, c in enumerate(chrs):
        ax = axes[i // n_cols, i % n_cols]
        perm_distribution(perm_by_chr[c], ax=ax, threshold=thr_by_chr.get(c), chrom=c)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("Max LOD Score", fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        try:
            legend = ax.get_legend()
            if legend:
                legend.prop.set_size(5)
        except AttributeError:
            pass
        sns.despine(ax=ax)

    for j in range(n_chrs, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle("Permutation null distributions by chromosome", fontsize=9, y=1.01)
    fig.tight_layout()
    return fig


def genome_summary(
    pk_df: pd.DataFrame,
    is_egene: np.ndarray,
    thr_by_chr: Optional[dict[str, Optional[float]]] = None,
    *,
    scan_stats: Optional[dict] = None,
) -> plt.Figure:
    """Genome-wide summary: Manhattan scatter (top) + eGene count per chr (bottom).

    Parameters
    ----------
    pk_df : pd.DataFrame
        Columns: gene, chr (str), peak_mb, peak_lod.
    is_egene : np.ndarray
        Boolean mask aligned to *pk_df* rows; True = eGene.
    thr_by_chr : dict, optional
        Per-chromosome threshold drawn as dashed horizontal lines.
    scan_stats : dict, optional
        Optional annotations: n_genes, n_egenes, pi0, perm_n, top_gene, peak_chr.

    Returns
    -------
    plt.Figure
    """
    stats = scan_stats or {}
    chr_arr = pk_df["chr"].astype(str).values
    pos_arr = pk_df["peak_mb"].values.astype(float)
    lod_arr = pk_df["peak_lod"].values.astype(float)
    gene_arr = (
        pk_df["gene_symbol"].values
        if "gene_symbol" in pk_df.columns
        else np.full(len(pk_df), "")
    )
    chrs = _ordered_chrs(pd.Series(chr_arr))

    fig, (ax_mh, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(14, 5.5),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    egene_color = BRAND_SECONDARY
    # colour range: 95th-percentile LOD drives vmax so bright = significant
    _fin_gs = lod_arr[np.isfinite(lod_arr)]
    _vmax_gs = float(np.percentile(_fin_gs, 95)) if len(_fin_gs) > 0 else 1.0
    offset = 0.0
    xticks: list[float] = []
    egene_counts: list[int] = []
    # track global top gene across chromosomes for annotation
    top_x: Optional[float] = None
    top_y: float = 0.0
    top_gene_symbol: str = ""

    for c in chrs:
        mask = chr_arr == c
        pos_c = pos_arr[mask]
        lod_c = lod_arr[mask]
        eg_c = is_egene[mask]
        genes_c = gene_arr[mask]

        if len(pos_c) == 0:
            egene_counts.append(0)
            xticks.append(offset)
            continue

        x = pos_c + offset
        # non-eGenes: LOD-magnitude colormap (consistent with manhattan())
        if (~eg_c).any():
            ax_mh.scatter(
                x[~eg_c],
                lod_c[~eg_c],
                c=lod_c[~eg_c],
                cmap=CMAP_MAGNITUDE,
                vmin=0.0,
                vmax=_vmax_gs,
                s=3,
                linewidths=0,
                zorder=2,
                alpha=0.45,
                rasterized=True,
            )
        # eGenes: PWKJ red overlay — marks FDR-significant genes
        if eg_c.any():
            sns.scatterplot(
                x=x[eg_c],
                y=lod_c[eg_c],
                color=egene_color,
                s=8,
                linewidths=0,
                zorder=3,
                rasterized=True,
                ax=ax_mh,
            )

        # track genome-wide peak for labeling
        if len(lod_c) > 0 and float(lod_c.max()) > top_y:
            top_y = float(lod_c.max())
            peak_i = int(np.argmax(lod_c))
            top_x = float(x[peak_i])
            top_gene_symbol = str(genes_c[peak_i]) if len(genes_c) > peak_i else ""

        if thr_by_chr and (thr_val := thr_by_chr.get(c)) is not None:
            ax_mh.hlines(
                thr_val,
                offset,
                offset + float(pos_c.max()),
                colors=BRAND_ACCENT,
                linewidth=1.4,
                linestyle=":",
                alpha=0.85,
                zorder=1,
            )

        xticks.append(float(offset + (float(pos_c.max()) - float(pos_c.min())) / 2.0))
        egene_counts.append(int(eg_c.sum()))
        offset += float(pos_c.max()) + _CHR_GAP_MB

    ax_mh.set_xlim(0, offset)
    ax_mh.set_ylim(bottom=0)
    ax_mh.set_xticks(xticks)
    ax_mh.set_xticklabels(chrs, fontsize=8)
    ax_mh.tick_params(axis="x", length=2)
    ax_mh.set_ylabel("Peak LOD Score", fontsize=11)
    n_genes = len(pk_df)
    n_eg = int(is_egene.sum())
    _mh_title = (
        f"Genome-wide peak LOD  ({n_genes:,} genes,"
        f"  {n_eg:,} eGenes  [{100*n_eg/max(n_genes,1):.1f}%])"
    )
    ax_mh.set_title(_mh_title, fontsize=11)
    # label the single genome-wide peak gene
    if top_x is not None and top_gene_symbol:
        ax_mh.annotate(
            top_gene_symbol,
            (top_x, top_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=egene_color,
            fontweight="bold",
        )
    # scan_stats summary box — top-right corner of Manhattan
    stat_lines: list[str] = []
    for key, label in [
        ("n_genes", "Genes"),
        ("n_egenes", "eGenes"),
        ("perm_n", "Perms"),
        ("top_gene", "Top gene"),
        ("peak_chr", "Peak chr"),
    ]:
        val = stats.get(key)
        if val is not None:
            stat_lines.append(f"{label}: {val}")
    if stat_lines:
        ax_mh.text(
            0.99,
            0.98,
            "\n".join(stat_lines),
            transform=ax_mh.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color=INK,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=MUTED,
                alpha=0.8,
            ),
        )
    sns.despine(ax=ax_mh)

    ax_bar.bar(range(len(chrs)), egene_counts, color=egene_color, width=0.7)
    # count labels on top of bars
    bar_max = max(egene_counts) if egene_counts else 1
    for xi, cnt in enumerate(egene_counts):
        if cnt > 0:
            ax_bar.text(
                xi, cnt + bar_max * 0.02, str(cnt), ha="center", va="bottom", fontsize=6
            )
    ax_bar.set_xticks(range(len(chrs)))
    ax_bar.set_xticklabels(chrs, fontsize=8)
    ax_bar.set_ylabel("eGenes", fontsize=10)
    ax_bar.set_xlim(-0.5, len(chrs) - 0.5)
    ax_bar.set_ylim(bottom=0, top=bar_max * 1.18)
    sns.despine(ax=ax_bar)

    fig.tight_layout(h_pad=2.0, pad=1.5)
    return fig


def genome_diagnostic(
    pk_df: pd.DataFrame,
    thr_by_chr: dict[str, Optional[float]],
    *,
    cis_dist_mb: Optional[np.ndarray] = None,
    scan_stats: Optional[dict] = None,
) -> plt.Figure:
    """Diagnostic overview: threshold per chr + optional cis-distance hist + stats.

    Parameters
    ----------
    pk_df : pd.DataFrame
        Columns: gene, chr (str), peak_mb, peak_lod.
    thr_by_chr : dict[str, float | None]
        Per-chromosome GEV thresholds.
    cis_dist_mb : np.ndarray, optional
        Cis distances in Mb for eGenes; adds a histogram panel when provided.
    scan_stats : dict, optional
        Keys: n_genes, n_egenes, pi0, perm_n, top_gene, peak_chr.

    Returns
    -------
    plt.Figure
    """
    has_cis = cis_dist_mb is not None
    n_panels = 3 if has_cis else 2
    width_ratios = [2, 1.5, 1] if has_cis else [2, 1]
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(n_panels * 4.0, 3.5),
        dpi=300,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = axes if n_panels > 1 else [axes]

    # genome_wide bar first (different color), then per-chr bars in chr order
    gw_thr = thr_by_chr.get("genome_wide")
    chrs_ordered = [c for c in _CHR_ORDER if c in thr_by_chr]
    chrs_ordered += sorted(set(thr_by_chr) - set(_CHR_ORDER) - {"genome_wide"})

    labels = (["genome\nwide"] if gw_thr is not None else []) + chrs_ordered
    values = ([gw_thr if gw_thr is not None else 0.0] if gw_thr is not None else []) + [
        thr_by_chr[c] if thr_by_chr[c] is not None else 0.0 for c in chrs_ordered
    ]
    colors = ([BRAND_ACCENT] if gw_thr is not None else []) + [MUTED] * len(
        chrs_ordered
    )

    ax0 = axes[0]
    ax0.bar(range(len(labels)), values, color=colors, width=0.7)
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax0.set_ylabel("GEV Threshold (LOD)", fontsize=12)
    ax0.set_title("Permutation threshold per chromosome", fontsize=12)
    ax0.set_xlim(-0.5, len(labels) - 0.5)
    ax0.set_ylim(bottom=0)
    ax0.tick_params(axis="y", labelsize=10)
    sns.despine(ax=ax0)

    if has_cis:
        ax1 = axes[1]
        finite = cis_dist_mb[np.isfinite(cis_dist_mb)]
        ax1.hist(finite, bins=40, color=BRAND_PRIMARY)
        ax1.set_xlabel("Cis distance (Mb)", fontsize=12)
        ax1.set_ylabel("eGene count", fontsize=12)
        ax1.set_title("eGene cis-distance", fontsize=12)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(labelsize=10)
        sns.despine(ax=ax1)
        ax_stats = axes[2]
    else:
        ax_stats = axes[1]

    stats = scan_stats or {}
    n_genes = stats.get("n_genes", len(pk_df))
    n_egenes = stats.get("n_egenes", 0)
    n_perms = stats.get("n_perms", 0)
    top_gene = stats.get("top_gene", "")
    peak_chr = stats.get("peak_chr", "")

    _perm_str = f"{n_perms:,}  (FWER via GEV)" if n_perms > 0 else "no permutations run"
    _egene_method = "FWER (GEV perm)" if n_perms > 0 else "LOD threshold"

    text_lines = [
        f"Genes tested:  {n_genes:,}",
        f"eGenes:        {n_egenes:,}  ({_egene_method})",
        f"Perms:         {_perm_str}",
        "",
        f"Top gene:      {top_gene}",
        f"Peak chr:      chr{peak_chr}",
    ]
    ax_stats.text(
        0.05,
        0.95,
        "\n".join(text_lines),
        transform=ax_stats.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        fontfamily=FONT_STACK,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=INK,
            alpha=0.9,
        ),
    )
    ax_stats.set_axis_off()

    fig.tight_layout()
    return fig


def nb_scan_recovery_summary(
    sim_df: pd.DataFrame,
    *,
    threshold_pctile: float = 95.0,
) -> plt.Figure:
    """Two-panel scan summary: planted locus recovery per simulation + peak LOD distribution.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Columns: ``sim_index`` (int), ``true_pctile`` (float), ``peak_lod`` (float).
    threshold_pctile : float
        LOD percentile threshold defining a successful recovery (default 95.0).

    Returns
    -------
    plt.Figure
    """
    sns.set_style("ticks")
    plot_df = sim_df.copy()
    plot_df["recovered"] = plot_df["true_pctile"] >= threshold_pctile
    plot_df["sim_index"] = plot_df["sim_index"].apply(lambda i: f"sim_{i}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    ax = axes[0]
    sns.barplot(
        data=plot_df,
        x="sim_index",
        y="true_pctile",
        hue="recovered",
        palette={True: BRAND_ACCENT, False: MUTED},
        ax=ax,
        legend=False,
        alpha=0.85,
    )
    ax.axhline(threshold_pctile, color=BRAND_ACCENT, ls=":", lw=1.4, alpha=0.85)
    ax.set_xlabel("Simulation index", fontsize=12)
    ax.set_ylabel("Planted locus LOD percentile", fontsize=12)
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)

    ax2 = axes[1]
    sns.histplot(
        data=plot_df,
        x="peak_lod",
        bins=max(5, len(sim_df) // 2),
        color=BRAND_PRIMARY,
        ax=ax2,
    )
    ax2.set_xlabel("Peak LOD (NB scan)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    sns.despine(ax=ax2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sample identity cube
# ---------------------------------------------------------------------------


def sample_identity_cube(
    metadata: pd.DataFrame,
    *,
    tissue_col: str = "tissue",
    replicate_col: str = "replicate",
    strain_col: str = "strain",
    hap_bins: "dict[str, str] | None" = None,
    tissue_order: "list[str] | None" = None,
    replicate_order: "list[str] | None" = None,
    figsize: "tuple[float, float]" = (11, 11),
    elev: float = 20.0,
    azim: float = -55.0,
    voxel_alpha: float = 0.88,
    edge_color: str = "white",
) -> plt.Figure:
    """8×8×8 voxel cube: sample identity across tissues, replicates, and founder haplotypes.

    Each axis maps to one experimental dimension:

    - **X** — up to 8 tissues
    - **Y** — up to 8 replicates
    - **Z** — 8 founder haplotype bins (FOUNDER_ORDER: AJ → WSBJ)

    A voxel is filled whenever at least one sample occupies that
    (tissue, replicate, founder-bin) cell and is colored by the founder
    palette.  CC strains and F1s are compressed to the 8-bin haplotype
    axis via ``hap_bins`` — a ``{strain: dominant_founder_name}`` mapping
    derived from ``argmax`` of genome-wide ancestry proportions.

    Parameters
    ----------
    metadata : pd.DataFrame
        One row per sample; must contain ``tissue_col``, ``replicate_col``,
        and ``strain_col``.
    tissue_col : str
        Column name for tissue labels.
    replicate_col : str
        Column name for replicate identifiers (any hashable type; up to 8).
    strain_col : str
        Column name for genotype/strain identifiers.
    hap_bins : dict[str, str], optional
        ``{strain: founder_name}`` — assigns every strain to one of the 8
        FOUNDER_ORDER bins.  Strains absent from the mapping are skipped.
        Pass ``None`` to use all strains without binning (requires strain
        names to already be founder names).
    tissue_order : list[str], optional
        Explicit tissue ordering (first 8 used).  Defaults to natsorted
        unique values from ``tissue_col``.
    replicate_order : list[str], optional
        Explicit replicate ordering (first 8 used).  Defaults to natsorted
        unique values from ``replicate_col``.
    figsize : tuple[float, float]
        Figure size in inches.
    elev : float
        3-D view elevation in degrees.
    azim : float
        3-D view azimuth in degrees.
    voxel_alpha : float
        Voxel face opacity (0–1).
    edge_color : str
        Voxel edge color (default white).

    Returns
    -------
    plt.Figure

    Examples
    --------
    Build ``hap_bins`` from genome-wide argmax of a GenomeCache:

    >>> import numpy as np
    >>> from mosaiqtl.genotype.cc import FOUNDER_ORDER
    >>> # probs: dict[strain, np.ndarray shape (n_founders,)]
    >>> hap_bins = {s: FOUNDER_ORDER[int(np.argmax(p))] for s, p in probs.items()}
    >>> fig = sample_identity_cube(meta_df, hap_bins=hap_bins)
    >>> fig.savefig('sample_identity_cube.png', bbox_inches='tight', dpi=300)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

    import natsort

    from ..genotype.cc import FOUNDER_COLORS, FOUNDER_ORDER

    # ── resolve axis orders (up to 8 each) ───────────────────────────────────
    def _nat8(series: pd.Series) -> list[str]:
        return natsort.natsorted(series.dropna().astype(str).unique())[:8]

    tissues = list(tissue_order[:8]) if tissue_order else _nat8(metadata[tissue_col])
    replicates = (
        list(replicate_order[:8]) if replicate_order else _nat8(metadata[replicate_col])
    )
    founders = list(FOUNDER_ORDER)  # 8 bins, canonical order

    n_t, n_r, n_f = len(tissues), len(replicates), len(founders)

    tissue_idx = {t: i for i, t in enumerate(tissues)}
    rep_idx = {r: i for i, r in enumerate(replicates)}
    founder_idx = {f: i for i, f in enumerate(founders)}

    # ── occupancy ────────────────────────────────────────────────────────────
    filled = np.zeros((n_t, n_r, n_f), dtype=bool)
    _bins = hap_bins or {}

    for _, row in metadata.iterrows():
        t = str(row[tissue_col])
        r = str(row[replicate_col])
        s = str(row[strain_col])

        if t not in tissue_idx or r not in rep_idx:
            continue

        # resolve founder bin: explicit map → identity (already a founder name)
        fname = _bins.get(s) or (s if s in founder_idx else None)
        fi = founder_idx.get(fname)  # type: ignore[arg-type]
        if fi is None:
            continue

        filled[tissue_idx[t], rep_idx[r], fi] = True

    # ── face color array (n_t, n_r, n_f, 4) ─────────────────────────────────
    def _hex_rgba(hex_str: str, alpha: float) -> "tuple[float, float, float, float]":
        h = hex_str.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b, alpha)

    fc: np.ndarray = np.zeros((n_t, n_r, n_f, 4), dtype=float)
    for fi, fname in enumerate(founders):
        fc[:, :, fi, :] = _hex_rgba(FOUNDER_COLORS[fname], voxel_alpha)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(filled, facecolors=fc, edgecolors=edge_color)

    # axis tick placement at voxel midpoints
    ax.set_xticks(np.arange(n_t) + 0.5)
    ax.set_yticks(np.arange(n_r) + 0.5)
    ax.set_zticks(np.arange(n_f) + 0.5)

    ax.set_xticklabels(tissues, fontsize=7, rotation=30, ha="right")
    ax.set_yticklabels(replicates, fontsize=7)
    ax.set_zticklabels(founders, fontsize=8, fontweight="bold")

    ax.set_xlabel("Tissue", labelpad=10, fontsize=9)
    ax.set_ylabel("Replicate", labelpad=10, fontsize=9)
    ax.set_zlabel("Founder haplotype", labelpad=10, fontsize=9)

    ax.view_init(elev=elev, azim=azim)

    return fig
