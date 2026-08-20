"""LOD profiles, Manhattan plots, region zooms, and SNP association plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._theme import (
    BRAND_PRIMARY, BRAND_SECONDARY, BRAND_ACCENT, INK, MUTED,
    CHR_BANDS, CMAP_MAGNITUDE, CMAP_SIGNED as _CMAP_EFFECT,
    KW_THRESHOLD, KW_REFERENCE, KW_DOT_MUTED,
    apply_theme, info_box,
)
from .effects import effect_bars as _effect_bars

_CHR_ORDER: list[str] = [str(i) for i in range(1, 20)] + ["X"]
_CHR_GAP_MB: float = 10.0

# ── semantic scan-series color aliases ───────────────────────────────────────
# MI overlay = secondary series; instability = accent; SNP = muted neutral.
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
        fontweight="bold",
    )
    ax.set_ylabel("LOD Score", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    if threshold is not None:
        ax.axhline(threshold, **KW_THRESHOLD)
    if tss_mb is not None:
        ax.axvline(
            tss_mb, color=MUTED, lw=0.9, ls="--", zorder=1, label="TSS"
        )
    if peak_pos is not None:
        ax.axvline(
            peak_pos,
            color=BRAND_ACCENT,
            linewidth=0.8,
            linestyle=":",
            zorder=1,
        )
        if annotate_peak:
            peak_idx = int(np.argmin(np.abs(positions - peak_pos)))
            peak_lod = float(lod[peak_idx])
            _ann = f"LOD={peak_lod:.1f}"
            if locus_id:
                _ann = f"{locus_id}\n{_ann}"
            # flip ha so annotation does not run past the plot edge
            _x_mid = float(positions[0] + positions[-1]) / 2.0
            _peak_ha = "right" if float(peak_pos) > _x_mid else "left"
            ax.text(
                peak_pos,
                peak_lod,
                _ann,
                fontsize=10,
                color=BRAND_ACCENT,
                va="bottom",
                ha=_peak_ha,
            )
    # gene name label — upper-left
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
    # h² — bottom-right so it never clashes with legend at upper-right
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
    _hi_xspan: 'tuple[float, float] | None' = None

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
                    x=x[_lo], y=lod[_lo],
                    **KW_DOT_MUTED,
                    ax=ax,
                )
            if (~_lo).any():
                sns.scatterplot(
                    x=x[~_lo], y=lod[~_lo],
                    color=chr_color,
                    s=5, alpha=0.4, linewidths=0, rasterized=True,
                    ax=ax,
                )
        else:
            sns.scatterplot(
                x=x, y=lod,
                color=chr_color,
                s=5, alpha=0.4, linewidths=0, rasterized=True,
                ax=ax,
            )
        # eGene overlay: accent orange, larger — marks FDR-significant genes
        if egene_scans and chrom in egene_scans:
            epos, elod = egene_scans[chrom]
            eorder = np.argsort(epos)
            ex = epos[eorder] + offset
            sns.scatterplot(
                x=ex, y=elod[eorder],
                color=BRAND_ACCENT,
                s=18, linewidths=0, rasterized=True,
                ax=ax,
            )
        mid = float(offset + (float(pos[-1]) - float(pos[0])) / 2)
        xticks.append(mid)
        xlabels.append(chrom)
        for xi, li in zip(x, lod):
            all_pts.append((float(xi), float(li), chrom))
        offset += float(pos[-1]) + _CHR_GAP_MB

    if _hi_xspan is not None:
        ax.axvspan(_hi_xspan[0], _hi_xspan[1], alpha=0.06, color=highlight_color, zorder=0)

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
    _hi_xspan_p: 'tuple[float, float] | None' = None

    for ci, chrom in enumerate(chroms):
        mask = snp_chroms == chrom
        pos = snp_positions[mask]
        y = snp_neg_log10p[mask]
        x = pos + offset
        if chrom == highlight_chrom:
            _hi_xspan_p = (float(x[0]), float(x[-1]))
        sns.scatterplot(
            x=x, y=y,
            color=CHR_BANDS[ci % 2],
            s=point_size, alpha=0.4, linewidths=0, rasterized=True,
            ax=ax,
        )
        xticks.append(float(offset + (pos.max() - pos.min()) / 2))
        xlabels.append(chrom)
        offset += float(pos.max()) + _CHR_GAP_MB

    if _hi_xspan_p is not None:
        ax.axvspan(_hi_xspan_p[0], _hi_xspan_p[1], alpha=0.06, color=highlight_color, zorder=0)

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

    sns.histplot(
        lod, bins=80, color=BRAND_PRIMARY, alpha=0.7, ax=ax, linewidth=0
    )
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

    _hi_xspan_g: 'tuple[float, float] | None' = None

    for ci, chrom in enumerate(chroms):
        mask = gene_chroms == chrom
        pos = gene_positions[mask]
        y = peak_lod[mask]
        x = pos + offset
        if chrom == highlight_chrom:
            _hi_xspan_g = (float(x[0]), float(x[-1]))
        sns.scatterplot(
            x=x, y=y,
            color=CHR_BANDS[ci % 2],
            s=point_size, alpha=0.4, linewidths=0, rasterized=True,
            ax=ax,
        )
        xticks.append(float(offset + (pos.max() - pos.min()) / 2))
        xlabels.append(chrom)
        offset += float(pos.max()) + _CHR_GAP_MB

    if _hi_xspan_g is not None:
        ax.axvspan(
            _hi_xspan_g[0], _hi_xspan_g[1], alpha=0.06, color=highlight_color, zorder=0,
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
        sns.kdeplot(
            _dist, ax=_kde_ax, color=INK, linewidth=1.6, alpha=0.6
        )
        _kde_ax.set_ylabel("")
        _kde_ax.set_yticks([])
        sns.despine(ax=_kde_ax, left=True, right=True, top=True, bottom=True)

    # ±proximal_mb shaded band
    ax.axvspan(
        -proximal_mb, proximal_mb, alpha=0.07, color=BRAND_PRIMARY, zorder=0
    )

    # reference and boundary lines
    ax.axvline(0, color=INK, lw=1.2, zorder=3)
    for _s in (-1, 1):
        ax.axvline(
            _s * cis_window_mb, color=MUTED, lw=0.9, ls=":", zorder=1
        )

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
        corner='tr',
        fontsize=8,
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Cis distance  (peak − TSS, Mb)", fontsize=11)
    ax.set_ylabel("Number of eGenes", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right", ncol=1)
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

    # ── panel 1: full range scatter ──────────────────────────────────────────
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
    ax.axhline(
        mean_loci, color=INK, linewidth=0.8, linestyle="--", alpha=0.7
    )
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
        sns.rugplot(
            x=obs_f, ax=ax, color=MUTED, alpha=0.5, height=0.04, zorder=0
        )

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
    # explicit Patch proxy guarantees histplot color swatch shows in legend
    # regardless of seaborn version (BarContainer labels not always auto-collected)
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
        alpha=0.85,
        ax=ax,
        linewidth=0,
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
    """Horizontal CI strips per gene (qtl2 lod_interval style).

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

    # median CI width reference line — vertical line at the median peak + offset
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
    # lower-left: gene text occupies upper-left; info box occupies upper-right
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
    # lower-left: gene text occupies upper-left; info box occupies upper-right
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
            [x - half_w + off, x + half_w + off],
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

    ax_t.barh(
        labels, elapsed_s, color=BRAND_PRIMARY, alpha=0.85, edgecolor="none"
    )
    ax_t.invert_yaxis()
    ax_t.set_xlabel("Elapsed (s)", fontsize=13, fontweight="bold")
    ax_t.tick_params(labelsize=11)
    sns.despine(ax=ax_t)

    if has_mem:
        axes[1].barh(
            labels,
            peak_heap_mb.astype(float),
            color=MUTED,
            alpha=0.85,
            edgecolor="none",
        )
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Peak heap (MB)", fontsize=13, fontweight="bold")
        axes[1].tick_params(labelsize=11)
        sns.despine(ax=axes[1])

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
# Private helpers
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
# Per-chromosome figure iterators
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
        LOD line colour; must be imported from genotype.founders.
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


def locus_panels_per_chr(
    lod_df: pd.DataFrame,
    eff_df: Optional[pd.DataFrame] = None,
    founders: Optional[list[str]] = None,
    thr_by_chr: Optional[dict[str, Optional[float]]] = None,
    *,
    priority_genes: Optional[dict[str, str]] = None,
    tss_mb_map: Optional[dict[str, float]] = None,
    near_tss_mb_map: Optional[dict[str, float]] = None,
    genes_df: Optional[pd.DataFrame] = None,
    zoom_window_mb: float = 5.0,
    gtf_path: Optional[str] = None,
) -> dict[str, plt.Figure]:
    """Per-chromosome locus panels (founder effects + LOD profile + zoom).

    The effect bar uses peak-locus betas from *eff_df*.
    When *tss_mb_map* is provided, a third panel shows a zoomed cis-window view
    with TSS and peak annotated relative to each other.

    Parameters
    ----------
    lod_df : pd.DataFrame
        Columns: gene_symbol, chr (str), pos_mb, lod.
    eff_df : pd.DataFrame, optional
        Columns: gene_symbol, chr (str), peak_mb, peak_lod, beta_AJ … beta_WSBJ.
        When provided, the top panel shows founder effect bars.
    founders : list[str], optional
        Display-name founder list matching beta column order (e.g. AJ, B6J …).
    thr_by_chr : dict, optional
        Per-chromosome GEV threshold.
    priority_genes : dict, optional
        {chr: gene_symbol}.
    tss_mb_map : dict, optional
        {gene_symbol: tss_mb}. When provided, adds a zoomed LOD panel per figure.
    near_tss_mb_map : dict, optional
        {gene_symbol: near_tss_mb}. TSS of the nearest gene to each QTL peak;
        shown as a dashed green marker in the zoom panel and gene track.
    genes_df : pd.DataFrame, optional
        Gene annotation table (``start_mb``, ``end_mb``, ``gene_symbol`` or
        ``gene_id``, optional ``strand``). When provided with *tss_mb_map*,
        adds an IGV gene body track below the zoom panel.
    zoom_window_mb : float
        Margin around the [TSS, peak] interval in the zoom panel. Default 1.0 Mb.
    gtf_path : str, optional
        Path to a ``.gtf`` or ``.gtf.gz`` file.  When provided, real
        exon/CDS/UTR intervals are loaded per panel and passed to the gene
        track — one GTF read per chromosome panel.

    Returns
    -------
    dict[str, plt.Figure]
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

        if eff_df is not None:
            eff_c = eff_df[eff_df["chr"].astype(str) == c]
            top_gene = (
                str(eff_c.loc[eff_c["peak_lod"].idxmax(), "gene_symbol"])
                if not eff_c.empty
                else ""
            )
        else:
            gene_max = lod_c.groupby("gene_symbol")["lod"].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ""

        if prio.get(c):
            candidates = (
                eff_df[eff_df["chr"].astype(str) == c]["gene_symbol"].values
                if eff_df is not None
                else lod_c["gene_symbol"].unique()
            )
            if prio[c] in candidates:
                top_gene = prio[c]

        if not top_gene:
            continue
        lod_top = lod_c[lod_c["gene_symbol"] == top_gene].sort_values("pos_mb")
        if lod_top.empty:
            continue

        peak_betas: Optional[np.ndarray] = None
        peak_pos: Optional[float] = None
        peak_locus_id: Optional[str] = None
        if eff_df is not None and founders is not None:
            eff_row_df = eff_df[
                (eff_df["chr"].astype(str) == c) & (eff_df["gene_symbol"] == top_gene)
            ]
            if not eff_row_df.empty:
                beta_cols = [b for b in eff_row_df.columns if b.startswith("beta_")]
                row0 = eff_row_df.iloc[0]
                peak_betas = np.array([float(row0[b]) for b in beta_cols])
                peak_pos = float(row0.get("peak_mb", lod_top["pos_mb"].iloc[0]))
                if "peak_locus" in eff_row_df.columns:
                    peak_locus_id = str(row0["peak_locus"])

        thr_val: Optional[float] = thr.get(c)
        tss = (tss_mb_map or {}).get(top_gene)
        near_tss = (near_tss_mb_map or {}).get(top_gene)
        fig = locus_panel(
            lod_top["pos_mb"].values,
            lod_top["lod"].values,
            founders or [],
            peak_betas=peak_betas,
            peak_pos=peak_pos,
            tss_mb=tss,
            near_tss_mb=near_tss,
            genes_df=genes_df,
            zoom_window_mb=zoom_window_mb,
            threshold=thr_val,
            chrom=c,
            highlight_gene=top_gene,
            gtf_path=gtf_path,
            peak_locus=peak_locus_id,
        )
        out[c] = fig

    return out


# ---------------------------------------------------------------------------
# IGV-style gene track helper
# ---------------------------------------------------------------------------


def _gene_track(
    ax: plt.Axes,
    genes_df: pd.DataFrame,
    xlim: tuple[float, float],
    *,
    highlight_gene: Optional[str] = None,
    peak_pos: Optional[float] = None,
    near_tss_mb: Optional[float] = None,
    exon_df: Optional[pd.DataFrame] = None,
) -> None:
    """Draw an IGV-style gene body track on *ax*.

    When *exon_df* is supplied (from :func:`~mosaiqtl.io.annotation_io.load_exons_from_gtf`),
    each gene is rendered with its real transcript/exon/CDS/UTR structure:

    * Thin intron backbone spanning the full canonical transcript extent.
    * UTR blocks (``feature == 'UTR'`` or exon portions not covered by CDS)
      at medium height.
    * CDS blocks at full height.
    * Directional chevrons (›/‹) in every intron gap.
    * TSS arrowhead at the 5′ end.

    When *exon_df* is ``None``, only a thin backbone with chevrons is drawn
    (honest representation when annotation data is unavailable).

    *peak_pos* and *near_tss_mb* are drawn as vertical lines so their
    position relative to surrounding gene bodies is immediately readable.

    Parameters
    ----------
    ax : plt.Axes
        Target axes — fully styled by this function.
    genes_df : pd.DataFrame
        Gene-level table; must have ``start_mb``, ``end_mb``, and either
        ``gene_symbol`` or ``gene_id``.  Optional ``strand`` column.
        Used for row-packing only (not for exon rendering).
    xlim : tuple[float, float]
        Visible x-range in Mb — should match the zoom panel directly above.
    highlight_gene : str, optional
        The gene being scanned; drawn in NODJ blue, others in grey.
    peak_pos : float, optional
        QTL peak locus (Mb); solid red vertical line.
    near_tss_mb : float, optional
        TSS of the nearest gene to the peak (Mb); dashed green vertical line.
    exon_df : pd.DataFrame, optional
        Output of ``load_exons_from_gtf``; columns ``gene_symbol``, ``feature``
        (``'transcript'`` / ``'exon'`` / ``'CDS'`` / ``'UTR'``), ``start_mb``,
        ``end_mb``, ``strand``.
    """
    lo, hi = xlim
    span = hi - lo
    gname_col = "gene_symbol" if "gene_symbol" in genes_df.columns else "gene_id"
    has_strand = "strand" in genes_df.columns

    # ── filter genes_df to visible window ────────────────────────────────────
    vis = (
        genes_df[(genes_df["end_mb"] >= lo) & (genes_df["start_mb"] <= hi)]
        .copy()
        .sort_values("start_mb")
        .reset_index(drop=True)
    )

    # ── build exon index ──────────────────────────────────────────────────────
    _exon_by_gene: dict[str, pd.DataFrame] = {}
    if exon_df is not None and not exon_df.empty:
        for _gn_key, _grp in exon_df.groupby("gene_symbol"):
            _exon_by_gene[str(_gn_key)] = _grp.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-GENE MODE — IGV-style: x-axis = gene body, large visible blocks
    # ══════════════════════════════════════════════════════════════════════════
    if len(vis) == 1 and not vis.empty:
        _g = vis.iloc[0]
        _gn = str(_g[gname_col])
        _clr = BRAND_PRIMARY if _gn == highlight_gene else MUTED
        _a = 0.95
        _str = str(_g.get("strand", "+")) if has_strand else "+"
        _y = 0.0

        _ann = _exon_by_gene.get(_gn)

        # authoritative gene extent from transcript record > exon span > gene row
        if _ann is not None:
            _tx = _ann[_ann["feature"] == "transcript"]
            if not _tx.empty:
                _glo = float(_tx["start_mb"].min())
                _ghi = float(_tx["end_mb"].max())
                _str = str(_tx.iloc[0]["strand"])
            else:
                _glo = float(_ann["start_mb"].min())
                _ghi = float(_ann["end_mb"].max())
        else:
            _glo = float(_g["start_mb"])
            _ghi = float(_g["end_mb"])

        _tss = _glo if _str == "+" else _ghi
        _gspan = max(_ghi - _glo, 1e-6)
        _margin = _gspan * 0.05

        # IGV block heights (tall enough to be clearly readable)
        _HUTR = 0.38  # UTR blocks (GTF feature 'UTR')
        _HCDS = 0.62  # coding exon blocks (GTF feature 'CDS'; legend label: 'exon')

        # Feature colors: CDS=NODJ blue, UTR=muted slate, intron/arrow=neutral
        _CLR_CDS = BRAND_PRIMARY  # #4A90C4 blue — coding exons
        _CLR_UTR = MUTED  # neutral grey — untranslated regions
        _CLR_BB = INK  # thin intron backbone
        _CLR_ARROW = MUTED  # chevron direction markers

        # 1. intron backbone — thin line across full transcript
        ax.plot(
            [_glo, _ghi],
            [_y, _y],
            color=_CLR_BB,
            lw=1.2,
            solid_capstyle="butt",
            zorder=2,
            alpha=_a,
            clip_on=True,
        )

        if _ann is not None:
            _exons = _ann[_ann["feature"] == "exon"].sort_values("start_mb")
            _cds = _ann[_ann["feature"] == "CDS"]
            _utrs = _ann[_ann["feature"] == "UTR"]

            # 2a. exon blocks at UTR height (full exon extent = UTR + CDS)
            #     use UTR color as the base — CDS will overdraw at full height
            for _, _ex in _exons.iterrows():
                _w = float(_ex["end_mb"]) - float(_ex["start_mb"])
                if _w > 0:
                    ax.broken_barh(
                        [(float(_ex["start_mb"]), _w)],
                        (_y - _HUTR / 2, _HUTR),
                        facecolors=_CLR_UTR,
                        edgecolors="none",
                        alpha=_a,
                        linewidth=0,
                        zorder=3,
                        clip_on=True,
                    )

            # 2b. CDS blocks — raised to full CDS height, distinct color
            for _, _cd in _cds.iterrows():
                _w = float(_cd["end_mb"]) - float(_cd["start_mb"])
                if _w > 0:
                    ax.broken_barh(
                        [(float(_cd["start_mb"]), _w)],
                        (_y - _HCDS / 2, _HCDS),
                        facecolors=_CLR_CDS,
                        edgecolors="none",
                        alpha=_a,
                        linewidth=0,
                        zorder=4,
                        clip_on=True,
                    )

            # 2c. explicit UTR records — redraw at UTR height over any CDS overlap
            for _, _ut in _utrs.iterrows():
                _w = float(_ut["end_mb"]) - float(_ut["start_mb"])
                if _w > 0:
                    ax.broken_barh(
                        [(float(_ut["start_mb"]), _w)],
                        (_y - _HUTR / 2, _HUTR),
                        facecolors=_CLR_UTR,
                        edgecolors="none",
                        alpha=_a,
                        linewidth=0,
                        zorder=5,
                        clip_on=True,
                    )

            # 3. directional arrow markers in intron gaps (ax.plot with marker,
            #    not data — same category as CI segment lines in viz CLAUDE.md)
            _ivs = sorted(
                [
                    (float(r["start_mb"]), float(r["end_mb"]))
                    for _, r in _exons.iterrows()
                ]
            )
            _mkr = ">" if _str == "+" else "<"
            _sp = max(_gspan * 0.07, 0.004)  # spacing between arrow markers
            _prev = _glo
            for _xs, _xe in _ivs:
                _gs, _ge = _prev, _xs  # intron gap
                if _ge - _gs > _sp * 2:
                    _pts = np.arange(_gs + _sp, _ge - _sp / 2, _sp)
                    if _pts.size:
                        ax.plot(
                            _pts,
                            np.full(_pts.size, _y),
                            ls="none",
                            marker=_mkr,
                            markersize=7,
                            color=_CLR_ARROW,
                            alpha=0.65,
                            zorder=3,
                            markeredgewidth=0,
                            clip_on=True,
                        )
                _prev = max(_prev, _xe)
            # gap after last exon → TES
            if _ghi - _prev > _sp * 2:
                _pts = np.arange(_prev + _sp, _ghi - _sp / 2, _sp)
                if _pts.size:
                    ax.plot(
                        _pts,
                        np.full(_pts.size, _y),
                        ls="none",
                        marker=_mkr,
                        markersize=7,
                        color=_CLR_ARROW,
                        alpha=0.65,
                        zorder=3,
                        markeredgewidth=0,
                        clip_on=True,
                    )

            # 3d. feature legend — small proxy patches in lower-right corner
            import matplotlib.patches as _mpatch

            _leg_handles = [
                _mpatch.Patch(facecolor=_CLR_CDS, label="exon", alpha=_a),
                _mpatch.Patch(facecolor=_CLR_UTR, label="UTR", alpha=_a),
                _mpatch.Patch(facecolor=_CLR_BB, label="intron", alpha=0.7),
            ]
            ax.legend(
                handles=_leg_handles,
                fontsize=6.5,
                frameon=False,
                loc="lower right",
                ncol=3,
                handlelength=1.0,
                handleheight=0.8,
                borderpad=0.3,
                columnspacing=0.8,
            )

        else:
            # no GTF annotation — backbone + directional markers across gene
            _mkr = ">" if _str == "+" else "<"
            _sp = max(_gspan * 0.07, 0.004)
            _pts = np.arange(_glo + _sp, _ghi - _sp / 2, _sp)
            if _pts.size:
                ax.plot(
                    _pts,
                    np.full(_pts.size, _y),
                    ls="none",
                    marker=_mkr,
                    markersize=7,
                    color=_CLR_ARROW,
                    alpha=0.55,
                    zorder=3,
                    markeredgewidth=0,
                    clip_on=True,
                )

        # 4. prominent TSS arrowhead — size proportional to gene length
        _adx = _gspan * 0.08
        if _str == "+":
            ax.annotate(
                "",
                xy=(_tss + _adx, _y),
                xytext=(_tss, _y),
                arrowprops=dict(arrowstyle="->", color=_clr, lw=2.0, mutation_scale=22),
                zorder=6,
                annotation_clip=False,
            )
        else:
            ax.annotate(
                "",
                xy=(_tss - _adx, _y),
                xytext=(_tss, _y),
                arrowprops=dict(arrowstyle="->", color=_clr, lw=2.0, mutation_scale=22),
                zorder=6,
                annotation_clip=False,
            )

        # 5. gene label above gene body
        ax.text(
            (_glo + _ghi) / 2,
            _y + _HCDS / 2 + 0.05,
            _gn,
            fontsize=9,
            ha="center",
            va="bottom",
            color=_clr,
            fontweight="bold",
            zorder=7,
            clip_on=False,
        )

        # 6. peak and near-TSS vertical lines (clip_on keeps them within gene xlim)
        _xleft = _glo - _margin
        _xright = _ghi + _margin
        if peak_pos is not None:
            ax.axvline(
                peak_pos,
                color=BRAND_SECONDARY,
                lw=1.5,
                ls="-",
                zorder=8,
                alpha=0.9,
            )
            if _xleft <= peak_pos <= _xright:
                ax.text(
                    peak_pos,
                    -_HCDS / 2 - 0.10,
                    "peak",
                    fontsize=7,
                    ha="center",
                    va="top",
                    color=BRAND_SECONDARY,
                    fontweight="bold",
                    clip_on=True,
                )
        if near_tss_mb is not None:
            ax.axvline(
                near_tss_mb,
                color=BRAND_ACCENT,
                lw=1.2,
                ls="--",
                zorder=8,
                alpha=0.9,
            )

        # 7. x-axis = gene body only (NOT the zoom window)
        ax.set_xlim(_xleft, _xright)
        ax.set_ylim(-0.95, 0.95)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        for _sp_name in ("left", "top", "right"):
            ax.spines[_sp_name].set_visible(False)
        return  # ── end single-gene mode ──────────────────────────────────

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-GENE MODE — row-packed, compact heights
    # ══════════════════════════════════════════════════════════════════════════
    _H_BACKBONE = 0.06
    _H_UTR = 0.14
    _H_CDS = 0.26
    _LABEL_MIN = span * 0.025
    _CHEV_SPACE = max(span * 0.025, 0.008)

    row_ends: list[float] = []
    row_idx: list[int] = []
    _GAP = 0.08
    for _, g in vis.iterrows():
        s = max(float(g["start_mb"]), lo)
        placed = False
        for ri, re in enumerate(row_ends):
            if re + _GAP < s:
                row_ends[ri] = min(float(g["end_mb"]), hi)
                row_idx.append(ri)
                placed = True
                break
        if not placed:
            row_ends.append(min(float(g["end_mb"]), hi))
            row_idx.append(len(row_ends) - 1)

    n_rows = max(len(row_ends), 1)

    # ── draw each gene ────────────────────────────────────────────────────────
    for (_, g), ri in zip(vis.iterrows(), row_idx):
        g_s = float(g["start_mb"])
        g_e = float(g["end_mb"])
        gn = str(g[gname_col])
        y = float(ri)
        is_hl = gn == highlight_gene
        clr = BRAND_PRIMARY if is_hl else MUTED
        alpha = 0.95 if is_hl else 0.55

        strand = "+"
        if has_strand:
            strand = str(g.get("strand", "+"))
        tss_pos = g_s if strand == "+" else g_e

        ann = _exon_by_gene.get(gn)

        if ann is not None:
            tx = ann[ann["feature"] == "transcript"]
            if not tx.empty:
                back_s = float(tx["start_mb"].min())
                back_e = float(tx["end_mb"].max())
                strand = str(tx.iloc[0]["strand"])
                tss_pos = back_s if strand == "+" else back_e
            else:
                back_s = float(ann["start_mb"].min())
                back_e = float(ann["end_mb"].max())
        else:
            back_s, back_e = g_s, g_e

        vis_back_s = max(back_s, lo)
        vis_back_e = min(back_e, hi)
        if vis_back_e <= vis_back_s:
            continue

        # backbone
        ax.broken_barh(
            [(vis_back_s, vis_back_e - vis_back_s)],
            (y - _H_BACKBONE / 2, _H_BACKBONE),
            facecolors=clr,
            edgecolors="none",
            alpha=alpha,
            linewidth=0.0,
            zorder=2,
        )

        if ann is not None:
            exons = ann[ann["feature"] == "exon"].sort_values("start_mb")
            cds = ann[ann["feature"] == "CDS"]
            utrs = ann[ann["feature"] == "UTR"]

            for _, ex in exons.iterrows():
                ex_s = max(float(ex["start_mb"]), lo)
                ex_e = min(float(ex["end_mb"]), hi)
                if ex_e > ex_s:
                    ax.broken_barh(
                        [(ex_s, ex_e - ex_s)],
                        (y - _H_UTR / 2, _H_UTR),
                        facecolors=clr,
                        edgecolors="none",
                        alpha=alpha,
                        linewidth=0,
                        zorder=3,
                    )
            for _, cd in cds.iterrows():
                cd_s = max(float(cd["start_mb"]), lo)
                cd_e = min(float(cd["end_mb"]), hi)
                if cd_e > cd_s:
                    ax.broken_barh(
                        [(cd_s, cd_e - cd_s)],
                        (y - _H_CDS / 2, _H_CDS),
                        facecolors=clr,
                        edgecolors="none",
                        alpha=alpha,
                        linewidth=0,
                        zorder=4,
                    )
            for _, ut in utrs.iterrows():
                ut_s = max(float(ut["start_mb"]), lo)
                ut_e = min(float(ut["end_mb"]), hi)
                if ut_e > ut_s:
                    ax.broken_barh(
                        [(ut_s, ut_e - ut_s)],
                        (y - _H_UTR / 2, _H_UTR),
                        facecolors=clr,
                        edgecolors="none",
                        alpha=alpha,
                        linewidth=0,
                        zorder=5,
                    )

            exon_intervals = sorted(
                [
                    (float(r["start_mb"]), float(r["end_mb"]))
                    for _, r in exons.iterrows()
                ]
            )
            mkr = ">" if strand == "+" else "<"
            prev_e = vis_back_s
            for ex_s_r, ex_e_r in exon_intervals:
                gap_s = max(prev_e, lo)
                gap_e = min(ex_s_r, hi)
                if gap_e - gap_s > _CHEV_SPACE * 2:
                    pts = np.arange(
                        gap_s + _CHEV_SPACE, gap_e - _CHEV_SPACE / 2, _CHEV_SPACE
                    )
                    if pts.size:
                        ax.plot(
                            pts,
                            np.full(pts.size, y),
                            ls="none",
                            marker=mkr,
                            markersize=4,
                            color=clr,
                            alpha=max(alpha - 0.2, 0.3),
                            zorder=3,
                            markeredgewidth=0,
                            clip_on=True,
                        )
                prev_e = max(prev_e, ex_e_r)
            gap_s = max(prev_e, lo)
            gap_e = min(vis_back_e, hi)
            if gap_e - gap_s > _CHEV_SPACE * 2:
                pts = np.arange(
                    gap_s + _CHEV_SPACE, gap_e - _CHEV_SPACE / 2, _CHEV_SPACE
                )
                if pts.size:
                    ax.plot(
                        pts,
                        np.full(pts.size, y),
                        ls="none",
                        marker=mkr,
                        markersize=4,
                        color=clr,
                        alpha=max(alpha - 0.2, 0.3),
                        zorder=3,
                        markeredgewidth=0,
                        clip_on=True,
                    )
        else:
            mkr = ">" if strand == "+" else "<"
            pts = np.arange(
                vis_back_s + _CHEV_SPACE, vis_back_e - _CHEV_SPACE / 2, _CHEV_SPACE
            )
            if pts.size:
                ax.plot(
                    pts,
                    np.full(pts.size, y),
                    ls="none",
                    marker=mkr,
                    markersize=4,
                    color=clr,
                    alpha=max(alpha - 0.2, 0.3),
                    zorder=3,
                    markeredgewidth=0,
                    clip_on=True,
                )

        # TSS arrowhead (compact for multi-gene)
        _arrow_dx = min(0.04, span * 0.018)
        if lo <= tss_pos <= hi:
            dx = _arrow_dx if strand == "+" else -_arrow_dx
            ax.annotate(
                "",
                xy=(tss_pos + dx, y),
                xytext=(tss_pos, y),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color=clr, lw=1.0, mutation_scale=8),
                zorder=6,
            )

        # label
        ctr = (vis_back_s + vis_back_e) / 2
        if (vis_back_e - vis_back_s) >= _LABEL_MIN:
            ax.text(
                ctr,
                y + _H_CDS / 2 + 0.04,
                gn,
                fontsize=5.5,
                ha="center",
                va="bottom",
                color=clr,
                clip_on=True,
                zorder=7,
                fontweight="bold" if is_hl else "normal",
            )

    # ── vertical annotations ──────────────────────────────────────────────────
    y_top = n_rows - 0.5
    if peak_pos is not None and lo <= peak_pos <= hi:
        ax.axvline(
            peak_pos, color=BRAND_SECONDARY, lw=1.3, ls="-", zorder=8, alpha=0.9
        )
        ax.text(
            peak_pos + span * 0.008,
            y_top,
            "peak",
            fontsize=6,
            color=BRAND_SECONDARY,
            ha="left",
            va="top",
            fontweight="bold",
            clip_on=True,
        )
    if near_tss_mb is not None and lo <= near_tss_mb <= hi:
        _nc = BRAND_ACCENT
        ax.axvline(near_tss_mb, color=_nc, lw=1.2, ls="--", zorder=8, alpha=0.9)
        ax.scatter(
            [near_tss_mb], [y_top], marker="v", s=20, color=_nc, zorder=9, clip_on=True
        )
        ax.text(
            near_tss_mb + span * 0.008,
            y_top - 0.15,
            "near TSS",
            fontsize=6,
            color=_nc,
            ha="left",
            va="top",
            clip_on=True,
        )

    # ── axis styling (multi-gene) ─────────────────────────────────────────────
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.60, n_rows - 0.25)
    ax.set_yticks([])
    ax.set_ylabel("genes", fontsize=7, rotation=0, labelpad=30, va="center")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="x", labelsize=7)
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# Single-figure grid summaries
# ---------------------------------------------------------------------------


def locus_panel(
    positions: np.ndarray,
    lod: np.ndarray,
    founders: list[str],
    *,
    peak_betas: Optional[np.ndarray] = None,
    peak_pos: Optional[float] = None,
    tss_mb: Optional[float] = None,
    near_tss_mb: Optional[float] = None,
    genes_df: Optional[pd.DataFrame] = None,
    zoom_window_mb: float = 5.0,
    threshold: Optional[float] = None,
    chrom: str = "",
    highlight_gene: Optional[str] = None,
    gtf_path: Optional[str] = None,
    peak_locus: Optional[str] = None,
) -> plt.Figure:
    """Locus view: founder effect bars + full LOD profile + zoom + gene track.

    Panel stack (top → bottom) depends on which optional arguments are supplied:
    - *peak_betas* → founder effect bar panel at the top.
    - always → full-chromosome LOD profile.
    - *tss_mb* → zoomed cis-window LOD panel with TSS and peak annotated.
    - *genes_df* (requires *tss_mb*) → IGV-style gene body track sharing the
      zoom panel's x-axis; shows *peak_pos* (red) and *near_tss_mb* (green
      dashed) as vertical markers so the QTL locus is immediately readable
      relative to surrounding gene bodies.

    Parameters
    ----------
    positions : np.ndarray
        Locus positions in Mb for the chromosome.
    lod : np.ndarray
        LOD scores matching *positions*.
    founders : list[str]
        Display-name founders in FOUNDER_ORDER for the effect bar axis labels.
    peak_betas : np.ndarray, optional
        Per-founder betas at the peak locus (shape (J,)); adds the effect bar panel.
    peak_pos : float, optional
        Peak locus position in Mb; annotated on both LOD panels and gene track.
    tss_mb : float, optional
        Gene TSS position in Mb; when provided, adds a zoomed cis-window panel.
    near_tss_mb : float, optional
        TSS of the nearest gene to the peak; shown as a dashed green line in the
        zoom panel and gene track (distinct from *tss_mb* of the scanned gene).
    genes_df : pd.DataFrame, optional
        Gene annotation table; requires columns ``start_mb``, ``end_mb``, and
        either ``gene_symbol`` or ``gene_id``. Optional ``strand`` column.  When
        provided together with *tss_mb*, adds an IGV-style gene body track below
        the zoom panel.
    zoom_window_mb : float
        Half-width of margin around the [TSS, peak] interval. Default 5.0 Mb.
    threshold : float, optional
        Permutation threshold; drawn as a dashed line on the LOD panels.
    chrom : str
        Chromosome label for the x-axis.
    highlight_gene : str, optional
        Gene name shown in the figure title and highlighted in the gene track.
    gtf_path : str, optional
        Path to a ``.gtf`` or ``.gtf.gz`` file (e.g. GENCODE vM36).  When
        provided together with *genes_df* and *tss_mb*, exon/CDS/UTR intervals
        are loaded for the zoom window and passed to the gene track so real
        exon/intron structure is drawn instead of plain gene-body bars.
    peak_locus : str, optional
        Variant/locus ID at the peak position (e.g. ``'UNC20195694'``).
        Annotated as rotated text beside the peak vertical line in the zoom panel.

    Returns
    -------
    plt.Figure
    """
    has_betas = peak_betas is not None
    has_zoom = tss_mb is not None
    has_track = genes_df is not None and has_zoom

    # ── pre-compute zoom bounds (needed before panel layout) ──────────────────
    zoom_lo: Optional[float] = None
    zoom_hi: Optional[float] = None
    if has_zoom:
        anchor = peak_pos if peak_pos is not None else tss_mb
        zoom_lo = min(tss_mb, anchor) - zoom_window_mb  # type: ignore[arg-type]
        zoom_hi = max(tss_mb, anchor) + zoom_window_mb  # type: ignore[arg-type]

    # ── load exon/CDS/UTR annotation (needed before panel layout for height) ──
    _exon_df: Optional[pd.DataFrame] = None
    if has_track and zoom_lo is not None and zoom_hi is not None and gtf_path and chrom:
        try:
            from ..io.annotation_io import load_exons_from_gtf

            _exon_df = load_exons_from_gtf(gtf_path, chrom, zoom_lo, zoom_hi)
        except Exception as _exc:  # noqa: BLE001
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "locus_panel: GTF load failed (%s); falling back to gene-body bars",
                _exc,
            )

    # ── build panel layout: gene track | zoom | full chr | founder effects ────
    ratios: list[float] = (
        ([0.8] if has_track else [])
        + ([1.0] if has_zoom else [])
        + [1.0]
        + ([2.0] if has_betas else [])
    )
    fig_h = 2.5 * sum(ratios)

    fig, axes = plt.subplots(
        len(ratios),
        1,
        figsize=(10, fig_h),
        dpi=300,
        gridspec_kw={"height_ratios": ratios},
        squeeze=False,
    )
    axes_flat = axes.ravel()
    ax_idx = 0

    title = highlight_gene or ""
    if chrom:
        title = f"{title}  chr{chrom}" if title else f"chr{chrom}"

    # ── panel 1: IGV gene body track (top) — highlight gene only ────────────
    if has_track and zoom_lo is not None and zoom_hi is not None:
        ax_track = axes_flat[ax_idx]
        ax_idx += 1
        # restrict to the scanned gene only — it serves as the x-axis reference
        _gn_col_lp = (  # type: ignore[union-attr]
            "gene_symbol" if "gene_symbol" in genes_df.columns else "gene_id"
        )
        if highlight_gene is not None:
            _track_genes = genes_df[genes_df[_gn_col_lp] == highlight_gene]  # type: ignore[index]
            if _track_genes.empty:
                _track_genes = genes_df  # type: ignore[assignment]
        else:
            _track_genes = genes_df  # type: ignore[assignment]
        _gene_track(
            ax_track,
            _track_genes,
            (zoom_lo, zoom_hi),
            highlight_gene=highlight_gene,
            peak_pos=peak_pos,
            near_tss_mb=near_tss_mb,
            exon_df=_exon_df,
        )
        if title:
            ax_track.set_title(title, fontsize=9)
        ax_track.set_xlabel("")
        # capture gene-body xlim so the zoom panel can shade it as a reference span
        _gene_body_xlim: tuple[float, float] = ax_track.get_xlim()
    else:
        _gene_body_xlim = (zoom_lo, zoom_hi) if zoom_lo is not None else (0.0, 0.0)

    # ── panel 2: zoomed cis-window LOD — ±window_mb around peak ─────────────
    # x-axis = zoom_lo..zoom_hi so the LOD bump is visible.
    # Gene body extent is shaded in blue so it's spatially anchored to the
    # gene track above.  Peak and TSS lines carry through from the full-chr
    # view below, creating a drill-down visual hierarchy:
    #   full chr → zoom window (shaded in chr panel) → gene body (shaded in zoom panel)
    if has_zoom and zoom_lo is not None and zoom_hi is not None:
        ax_zoom = axes_flat[ax_idx]
        ax_idx += 1

        mask = (positions >= zoom_lo) & (positions <= zoom_hi)
        pos_z = positions[mask]
        lod_z = lod[mask]

        if pos_z.size > 0:
            lod_profile(
                pos_z,
                lod_z,
                ax=ax_zoom,
                chrom=chrom,
                threshold=threshold,
                peak_pos=None,
                fill=True,
            )
        ax_zoom.set_xlim(zoom_lo, zoom_hi)

        # shade gene body so viewer can locate the gene within the LOD curve
        if has_track:
            _gb_lo, _gb_hi = _gene_body_xlim
            _span_lo = max(_gb_lo, zoom_lo)
            _span_hi = min(_gb_hi, zoom_hi)
            if _span_hi > _span_lo:
                ax_zoom.axvspan(
                    _span_lo,
                    _span_hi,
                    alpha=0.10,
                    color=BRAND_PRIMARY,
                    zorder=0,
                    label="gene body",
                )

        # consistent reference lines — same colors as full-chr panel
        if tss_mb is not None and zoom_lo <= tss_mb <= zoom_hi:
            ax_zoom.axvline(
                tss_mb,
                color=INK,
                linewidth=0.9,
                linestyle="--",
                zorder=2,
                label="TSS",
            )
        if peak_pos is not None and zoom_lo <= peak_pos <= zoom_hi:
            ax_zoom.axvline(
                peak_pos,
                color=BRAND_SECONDARY,
                linewidth=1.2,
                linestyle="-",
                zorder=2,
                label="peak",
            )
            if peak_locus:
                _yl = ax_zoom.get_ylim()
                # flip ha so rotated text does not run toward legend (upper-right)
                _pl_ha = (
                    "right" if float(peak_pos) > (zoom_lo + zoom_hi) / 2.0 else "left"
                )
                ax_zoom.text(
                    peak_pos,
                    _yl[1] * 0.97,
                    peak_locus,
                    fontsize=6,
                    color=BRAND_SECONDARY,
                    ha=_pl_ha,
                    va="top",
                    rotation=90,
                    clip_on=True,
                    zorder=5,
                )
        if near_tss_mb is not None and zoom_lo <= near_tss_mb <= zoom_hi:
            ax_zoom.axvline(
                near_tss_mb,
                color=BRAND_ACCENT,
                linewidth=0.9,
                linestyle="--",
                zorder=2,
                label="near TSS",
                alpha=0.8,
            )
        if not has_track and title:
            ax_zoom.set_title(title, fontsize=9)
        ax_zoom.set_xlabel(f"chr{chrom} position (Mb)")
        ax_zoom.legend(fontsize=7, frameon=False, loc="upper right")
        sns.despine(ax=ax_zoom)

    # ── panel 3: full-chromosome LOD profile ─────────────────────────────────
    ax_full = axes_flat[ax_idx]
    ax_idx += 1
    lod_profile(
        positions,
        lod,
        ax=ax_full,
        chrom=chrom,
        threshold=threshold,
        peak_pos=peak_pos,
        fill=True,
        tss_mb=tss_mb,
    )
    ax_full.set_title("")

    # shade the zoom window so the viewer can see which region was drilled into
    if zoom_lo is not None and zoom_hi is not None:
        ax_full.axvspan(
            zoom_lo, zoom_hi, alpha=0.10, color=INK, zorder=0
        )

    if tss_mb is not None:
        _ylim2 = ax_full.get_ylim()
        _arrow_y = _ylim2[1] * 0.06
        ax_full.annotate(
            "",
            xy=(tss_mb + 0.3, _arrow_y),
            xytext=(tss_mb, _arrow_y),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0),
        )
        _gene_label = highlight_gene or ""
        if _gene_label:
            ax_full.text(
                tss_mb,
                _arrow_y + _ylim2[1] * 0.04,
                _gene_label,
                fontsize=7,
                ha="left",
                color=MUTED,
                va="bottom",
            )
    if not has_betas and not has_zoom and not has_track and title:
        ax_full.text(
            0.02,
            0.96,
            title,
            transform=ax_full.transAxes,
            fontsize=9,
            va="top",
            fontweight="bold",
            color=BRAND_PRIMARY,
        )
    sns.despine(ax=ax_full)

    # ── panel 4: founder effect bars (bottom) ────────────────────────────────
    if has_betas:
        ax_eff = axes_flat[ax_idx]
        _effect_bars(peak_betas, founders, ax=ax_eff)
        ax_eff.set_xlabel("")
        sns.despine(ax=ax_eff)

    fig.tight_layout()
    return fig


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


def locus_panel_grid(
    lod_df: pd.DataFrame,
    eff_df: Optional[pd.DataFrame] = None,
    founders: Optional[list[str]] = None,
    thr_by_chr: Optional[dict[str, Optional[float]]] = None,
    *,
    priority_genes: Optional[dict[str, str]] = None,
    n_cols: int = 4,
) -> plt.Figure:
    """Grid of qtl2-style locus panels (founder effects + LOD), one per chromosome.

    Each cell contains an effect bar (top sub-panel) and a LOD profile (bottom
    sub-panel) using ``GridSpecFromSubplotSpec`` nested inside the outer grid.
    Falls back to LOD-only when *eff_df* or *founders* is not provided, or when
    a gene lacks effect data.

    Parameters
    ----------
    lod_df : pd.DataFrame
        Columns: gene_symbol, chr (str), pos_mb, lod.
    eff_df : pd.DataFrame, optional
        Columns: gene_symbol, chr (str), peak_mb, peak_lod, beta_A … beta_H.
    founders : list[str], optional
        Display-name founders matching beta column order.
    thr_by_chr : dict, optional
        Per-chromosome GEV threshold.
    priority_genes : dict, optional
        {chr: gene_symbol}.
    n_cols : int
        Panels per row.

    Returns
    -------
    plt.Figure
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    show_effects = eff_df is not None and founders is not None
    chr_vals = lod_df["chr"].astype(str)
    chrs = _ordered_chrs(chr_vals)
    n_chrs = len(chrs)
    n_rows = (n_chrs + n_cols - 1) // n_cols
    cell_h = 3.8 if show_effects else 2.2

    fig = plt.figure(figsize=(n_cols * 3, n_rows * cell_h), dpi=300)
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.4)

    for i, c in enumerate(chrs):
        row, col = i // n_cols, i % n_cols
        lod_c = lod_df[chr_vals == c]
        if lod_c.empty:
            continue

        if eff_df is not None:
            eff_c = eff_df[eff_df["chr"].astype(str) == c]
            top_gene = (
                str(eff_c.loc[eff_c["peak_lod"].idxmax(), "gene_symbol"])
                if not eff_c.empty
                else ""
            )
        else:
            gene_max = lod_c.groupby("gene_symbol")["lod"].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ""

        if prio.get(c):
            candidates = (
                eff_df[eff_df["chr"].astype(str) == c]["gene_symbol"].values
                if eff_df is not None
                else lod_c["gene_symbol"].unique()
            )
            if prio[c] in candidates:
                top_gene = prio[c]

        if not top_gene:
            continue
        lod_top = lod_c[lod_c["gene_symbol"] == top_gene].sort_values("pos_mb")
        if lod_top.empty:
            continue
        label = top_gene[:14]
        thr_val: Optional[float] = thr.get(c)

        if show_effects:
            eff_row_df = eff_df[
                (eff_df["chr"].astype(str) == c) & (eff_df["gene_symbol"] == top_gene)
            ]
            if not eff_row_df.empty:
                beta_cols = [b for b in eff_row_df.columns if b.startswith("beta_")]
                betas = np.array([float(eff_row_df.iloc[0][b]) for b in beta_cols])
                inner = gridspec.GridSpecFromSubplotSpec(
                    2,
                    1,
                    subplot_spec=outer[row, col],
                    height_ratios=[2, 1],
                    hspace=0.08,
                )
                ax_top = fig.add_subplot(inner[0])
                ax_bot = fig.add_subplot(inner[1])

                _effect_bars(betas, founders, ax=ax_top)  # type: ignore[arg-type]
                ax_top.set_xlabel("")
                ax_top.tick_params(labelsize=5)
                ax_top.set_title(f"chr{c}  {label}", fontsize=7)
                sns.despine(ax=ax_top)

                lod_profile(
                    lod_top["pos_mb"].values,
                    lod_top["lod"].values,
                    ax=ax_bot,
                    chrom="",
                    threshold=thr_val,
                    fill=True,
                    linewidth=0.8,
                )
                ax_bot.set_xlabel(f"chr{c} (Mb)", fontsize=6)
                ax_bot.tick_params(labelsize=5)
                ax_bot.set_title("")
                sns.despine(ax=ax_bot)
                continue

        # LOD-only fallback
        ax = fig.add_subplot(outer[row, col])
        lod_profile(
            lod_top["pos_mb"].values,
            lod_top["lod"].values,
            ax=ax,
            chrom="",
            threshold=thr_val,
            fill=True,
            linewidth=0.8,
        )
        ax.set_xlabel(f"chr{c} (Mb)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(f"chr{c}  {label}", fontsize=7)
        sns.despine(ax=ax)

    for j in range(n_chrs, n_rows * n_cols):
        ax_empty = fig.add_subplot(outer[j // n_cols, j % n_cols])
        ax_empty.set_visible(False)

    title = "Locus panels — top gene per chromosome"
    if show_effects:
        title += "  (founder effects + LOD)"
    fig.suptitle(title, fontsize=9, y=1.01)
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
    values = (
        ([gw_thr if gw_thr is not None else 0.0] if gw_thr is not None else [])
        + [thr_by_chr[c] if thr_by_chr[c] is not None else 0.0 for c in chrs_ordered]
    )
    colors = (
        ([BRAND_ACCENT] if gw_thr is not None else [])
        + [MUTED] * len(chrs_ordered)
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
        fontfamily=["Carlito", "Calibri", "DejaVu Sans"],
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
