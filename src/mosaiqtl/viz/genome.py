"""LOD profiles, Manhattan plots, region zooms, and SNP association plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..genotype.founders import (
    COLOR_CC_FALLBACK, COLOR_NEUTRAL_DARK, COLOR_NEUTRAL_LIGHT, FOUNDER_COLORS,
)
from .effects import effect_bars as _effect_bars

_CHR_ORDER: list[str] = [str(i) for i in range(1, 20)] + ['X']
_CHR_GAP_MB: float = 10.0
_CHR_COLORS: tuple[str, str] = (FOUNDER_COLORS['NODJ'], COLOR_CC_FALLBACK)
_MI_COLOR: str = FOUNDER_COLORS['WSBJ']
_INST_COLOR: str = FOUNDER_COLORS['CASTJ']
_SNP_COLOR: str = FOUNDER_COLORS['129S1J']


def _set_font(ax: plt.Axes) -> None:
    """Apply Carlito/Calibri to all text elements of ax."""
    for item in ([ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily(['Carlito', 'Calibri', 'DejaVu Sans'])


def lod_profile(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    threshold: Optional[float] = None,
    peak_pos: Optional[float] = None,
    color: str = FOUNDER_COLORS['NODJ'],
    linewidth: float = 1.2,
    fill: bool = False,
    fill_alpha: float = 0.15,
    h2: Optional[float] = None,
    gene: str = '',
    tss_mb: Optional[float] = None,
    locus_id: Optional[str] = None,
    marker_rug: bool = False,
    annotate_peak: bool = False,
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

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
    order = np.argsort(positions)
    positions = positions[order]
    lod = lod[order]
    ax.plot(positions, lod, color=color, linewidth=linewidth, zorder=2)
    if fill:
        ax.fill_between(positions, 0, lod, alpha=fill_alpha, color=color, zorder=1)
    if marker_rug:
        ax.plot(positions, np.zeros_like(positions),
                '|', color=COLOR_CC_FALLBACK, alpha=0.4, markersize=3, zorder=0)
    ax.set_xlim(float(positions[0]), float(positions[-1]))
    ax.set_ylim(0, max(float(lod.max()) * 1.18, 1.0))
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('LOD Score', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
                   linestyle='--', zorder=1)
        ax.text(float(positions[-1]), threshold + 0.05 * ax.get_ylim()[1],
                f'thr={threshold:.2f}', fontsize=10, ha='right', va='bottom',
                color=COLOR_NEUTRAL_DARK)
    if tss_mb is not None:
        ax.axvline(tss_mb, color=COLOR_CC_FALLBACK, lw=0.9, ls='--',
                   zorder=1, label='TSS')
    if peak_pos is not None:
        ax.axvline(peak_pos, color=FOUNDER_COLORS['PWKJ'],
                   linewidth=0.8, linestyle=':', zorder=1)
        if annotate_peak:
            peak_idx = int(np.argmin(np.abs(positions - peak_pos)))
            peak_lod = float(lod[peak_idx])
            _ann = f'LOD={peak_lod:.1f}'
            if locus_id:
                _ann = f'{locus_id}\n{_ann}'
            ax.text(peak_pos, peak_lod, _ann,
                    fontsize=10, color=FOUNDER_COLORS['PWKJ'],
                    va='bottom', ha='left')
    # gene name label
    if gene:
        ax.text(0.02, 0.96, gene, transform=ax.transAxes,
                fontsize=10, va='top', fontweight='bold', color=color)
    # h² annotation in upper-right corner
    if h2 is not None:
        ax.text(0.98, 0.96, f'h²={h2:.2f}', transform=ax.transAxes,
                fontsize=10, va='top', ha='right', color=COLOR_NEUTRAL_DARK)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def manhattan(
    scans: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    figsize: tuple[float, float] = (14, 3.5),
    threshold: Optional[float] = None,
    highlight_chrom: Optional[str] = None,
    highlight_color: str = FOUNDER_COLORS['PWKJ'],
    egene_scans: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    top_labels: Optional[dict[str, list[str]]] = None,
    top_n_labels: int = 5,
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
    top_labels : dict, optional
        ``{chrom: [gene_name, ...]}`` parallel to scans positions;
        annotates the top-LOD genes by name.
    top_n_labels : int
        Number of top-LOD genes to label when top_labels is not provided.

    Returns
    -------
    plt.Figure
    """
    ordered = [c for c in _CHR_ORDER if c in scans]
    extra   = [c for c in scans if c not in ordered]
    chroms  = ordered + extra

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    offset  = 0.0
    xticks: list[float] = []
    xlabels: list[str]  = []
    # collect (x, lod, chrom) for top-gene labeling
    all_pts: list[tuple[float, float, str]] = []

    for ci, chrom in enumerate(chroms):
        pos, lod = scans[chrom]
        order = np.argsort(pos)
        pos   = pos[order]
        lod   = lod[order]
        x     = pos + offset
        x_start = offset
        x_end   = float(x[-1])
        base_color = (highlight_color if chrom == highlight_chrom
                      else _CHR_COLORS[ci % 2])
        # alternating band — drawn behind scatter
        if ci % 2 == 0:
            ax.axvspan(x_start, x_end, alpha=0.10,
                       color=COLOR_NEUTRAL_LIGHT, zorder=0, linewidth=0)
        # scatter points
        ax.scatter(x, lod, color=base_color, s=4, linewidths=0, zorder=2,
                   alpha=0.75, rasterized=True)
        # eGene overlay: larger, opaque, highlight color
        if egene_scans and chrom in egene_scans:
            epos, elod = egene_scans[chrom]
            eorder = np.argsort(epos)
            ex = epos[eorder] + offset
            ax.scatter(ex, elod[eorder], color=FOUNDER_COLORS['PWKJ'],
                       s=18, linewidths=0, zorder=4, rasterized=True)
        mid = float(offset + (float(pos[-1]) - float(pos[0])) / 2)
        xticks.append(mid)
        xlabels.append(chrom)
        for xi, li in zip(x, lod):
            all_pts.append((float(xi), float(li), chrom))
        offset += float(pos[-1]) + _CHR_GAP_MB

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel('LOD Score', fontsize=13)
    ax.set_xlabel('Chromosome', fontsize=13)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.tick_params(axis='x', length=2, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, linewidth=1.0,
                   linestyle='--', zorder=1)
        ax.text(offset * 0.005, threshold,
                f'p95={threshold:.2f}', fontsize=10, color=COLOR_NEUTRAL_DARK,
                va='bottom')
    # label top-N LOD points across the whole genome
    if top_n_labels > 0 and all_pts:
        top_pts = sorted(all_pts, key=lambda t: t[1], reverse=True)[:top_n_labels]
        for tx, ty, _ in top_pts:
            ax.annotate(f'{ty:.1f}', (tx, ty),
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=10, ha='center', color=COLOR_NEUTRAL_DARK)
    sns.despine(ax=ax)
    _set_font(ax)
    return fig


def lod_interval(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    lod_drop: float = 1.5,
    chrom: str = '',
    color: str = FOUNDER_COLORS['NODJ'],
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
    ax     = lod_profile(positions, lod, ax=ax, chrom=chrom, color=color)
    cutoff = float(lod.max()) - lod_drop
    ax.fill_between(positions, 0, lod, where=(lod >= cutoff),
                    alpha=0.20, color=color, linewidth=0)
    ax.axhline(cutoff, color=COLOR_CC_FALLBACK, linewidth=0.6,
               linestyle=':', zorder=1)
    return ax


def bayes_interval(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    prob: float = 0.95,
    chrom: str = '',
    color: str = FOUNDER_COLORS['NODJ'],
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
    ax         = lod_profile(positions, lod, ax=ax, chrom=chrom, color=color)
    likelihood = 10.0 ** (lod - float(lod.max()))
    density    = likelihood / likelihood.sum()
    order      = np.argsort(density)[::-1]
    cumulative = np.cumsum(density[order])
    in_hdr     = np.zeros(len(lod), dtype=bool)
    in_hdr[order[cumulative <= prob]] = True
    ax.fill_between(positions, 0, lod, where=in_hdr,
                    alpha=0.20, color=color, linewidth=0)
    return ax


def region_zoom(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    start_mb: Optional[float] = None,
    end_mb: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    threshold: Optional[float] = None,
    snp_positions: Optional[np.ndarray] = None,
    snp_neg_log10p: Optional[np.ndarray] = None,
    snp_color: str = COLOR_CC_FALLBACK,
    color: str = FOUNDER_COLORS['NODJ'],
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
    ax   = lod_profile(positions[mask], lod[mask], ax=ax, chrom=chrom,
                       color=color, threshold=threshold)
    ax.set_xlim(lo, hi)

    if snp_positions is not None and snp_neg_log10p is not None:
        snp_mask = (snp_positions >= lo) & (snp_positions <= hi)
        ax2 = ax.twinx()
        ax2.scatter(snp_positions[snp_mask], snp_neg_log10p[snp_mask],
                    color=snp_color, s=4, alpha=0.6, zorder=1)
        ax2.set_ylabel('−log₁₀ P (SNP)', fontsize=8)
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
    highlight_color: str = FOUNDER_COLORS['PWKJ'],
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
    extra   = sorted(set(snp_chroms) - set(ordered))
    chroms  = ordered + extra

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    offset  = 0.0
    xticks: list[float] = []
    xlabels: list[str]  = []

    for ci, chrom in enumerate(chroms):
        mask  = snp_chroms == chrom
        pos   = snp_positions[mask]
        y     = snp_neg_log10p[mask]
        x     = pos + offset
        color = (highlight_color if chrom == highlight_chrom
                 else _CHR_COLORS[ci % 2])
        ax.scatter(x, y, color=color, s=point_size, linewidths=0, zorder=2)
        xticks.append(float(offset + (pos.max() - pos.min()) / 2))
        xlabels.append(chrom)
        offset += float(pos.max()) + _CHR_GAP_MB

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel('−log₁₀ P')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis='x', length=2)
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
                   linestyle='--', zorder=1)
    sns.despine(ax=ax)
    _set_font(ax)
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
    n  = len(pv)
    observed = -np.log10(pv)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    ax.scatter(expected, observed, s=6, alpha=0.5,
               color=FOUNDER_COLORS['NODJ'], linewidths=0,
               rasterized=True, zorder=2)
    ax.plot([0, float(expected.max())], [0, float(expected.max())],
            color=FOUNDER_COLORS['PWKJ'], linewidth=0.9,
            linestyle='--', zorder=1)

    lambda_gc = float(np.median(pv)) / 0.5
    ax.text(0.05, 0.95, f'λ_GC = {1/lambda_gc:.3f}',
            transform=ax.transAxes, fontsize=9, va='top')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Expected −log₁₀(p)')
    ax.set_ylabel('Observed −log₁₀(p)')
    sns.despine(ax=ax)
    _set_font(ax)
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

    sns.histplot(lod, bins=80, color=FOUNDER_COLORS['NODJ'],
                 alpha=0.7, ax=ax, linewidth=0)
    if threshold is not None:
        ax.axvline(threshold, color=FOUNDER_COLORS['PWKJ'],
                   linewidth=1.0, linestyle='--',
                   label=f'threshold = {threshold:.1f}')
        ax.legend(fontsize=8, frameon=False)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('LOD Score')
    ax.set_ylabel('Number of loci')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def gene_manhattan(
    gene_positions: np.ndarray,
    gene_chroms: np.ndarray,
    peak_lod: np.ndarray,
    *,
    figsize: tuple[float, float] = (14, 3),
    threshold: Optional[float] = None,
    highlight_chrom: Optional[str] = None,
    highlight_color: str = FOUNDER_COLORS['PWKJ'],
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
    extra   = sorted(set(gene_chroms) - set(ordered))
    chroms  = ordered + extra

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    offset  = 0.0
    xticks: list[float] = []
    xlabels: list[str]  = []

    for ci, chrom in enumerate(chroms):
        mask  = gene_chroms == chrom
        pos   = gene_positions[mask]
        y     = peak_lod[mask]
        color = (highlight_color if chrom == highlight_chrom
                 else _CHR_COLORS[ci % 2])
        ax.scatter(pos + offset, y, color=color, s=point_size,
                   linewidths=0, zorder=2)
        xticks.append(float(offset + (pos.max() - pos.min()) / 2))
        xlabels.append(chrom)
        offset += float(pos.max()) + _CHR_GAP_MB

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel('Peak LOD Score')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis='x', length=2)
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
                   linestyle='--', zorder=1)
    sns.despine(ax=ax)
    _set_font(ax)
    return fig


def cis_distance_histogram(
    tss_positions: np.ndarray,
    peak_positions: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    cis_window_mb: float = 5.0,
    bins: int = 60,
) -> plt.Axes:
    """Distribution of eQTL peak distance from gene TSS.

    Parameters
    ----------
    tss_positions : np.ndarray
        Gene TSS positions in Mb, shape (n_egenes,).
    peak_positions : np.ndarray
        eQTL peak positions in Mb, same shape (same-chromosome eGenes only).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    cis_window_mb : float
        Cis-window boundary — drawn as vertical dotted lines.
    bins : int
        Number of histogram bins.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=300)

    distance = peak_positions - tss_positions
    sns.histplot(distance, bins=bins, color=FOUNDER_COLORS['NODJ'],
                 alpha=0.7, ax=ax, linewidth=0)
    for sign in (-1, 1):
        ax.axvline(sign * cis_window_mb, color=COLOR_CC_FALLBACK,
                   linewidth=0.8, linestyle=':', zorder=1)
    ax.axvline(0, color=COLOR_NEUTRAL_DARK, linewidth=0.6, zorder=1)

    ax.set_ylim(bottom=0)
    ax.set_xlabel('Peak − TSS (Mb)')
    ax.set_ylabel('Number of eGenes')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


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

    ax.scatter(lod_lmm, lod_nb, s=point_size, alpha=alpha,
               color=FOUNDER_COLORS['NODJ'], linewidths=0,
               zorder=2, rasterized=True)
    lim = max(float(lod_nb.max()), float(lod_lmm.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=COLOR_NEUTRAL_DARK,
            linewidth=0.8, linestyle='--', zorder=1)
    r = float(np.corrcoef(lod_lmm, lod_nb)[0, 1])
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            fontsize=9, va='top')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('LMM LOD Score')
    ax.set_ylabel('NB-GLMM LOD Score')
    sns.despine(ax=ax)
    _set_font(ax)
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

    n     = len(library_sizes)
    order = np.argsort(library_sizes)
    x     = np.arange(n)

    ax.bar(x, library_sizes[order], color=FOUNDER_COLORS['NODJ'],
           alpha=0.8, width=0.8)
    ax.axhline(float(np.median(library_sizes)), color=FOUNDER_COLORS['PWKJ'],
               linewidth=0.9, linestyle='--', label='median')
    ax.legend(fontsize=8, frameon=False)

    if sample_ids is not None and n <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels([sample_ids[i] for i in order],
                           rotation=90, fontsize=6)
    else:
        ax.set_xlabel('Strains (sorted by library size)')

    ax.set_ylim(bottom=0)
    ax.set_ylabel('Total UMI')
    sns.despine(ax=ax)
    _set_font(ax)
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
    base = np.array(plt.matplotlib.colors.to_rgb(FOUNDER_COLORS['NODJ']))
    colors = [(1 - 0.45 * nc) * base + 0.45 * nc * np.zeros(3)
              for nc in norm_counts]
    ax.bar(x, n_arr, color=colors, width=0.7)
    # value labels on top of each bar
    y_max = float(n_arr.max())
    for xi, ni in zip(x, n_arr):
        ax.text(xi, ni + y_max * 0.01, f'{int(ni):,}',
                ha='center', va='bottom', fontsize=6)
    # mean reference line
    mean_loci = float(n_arr.mean())
    ax.axhline(mean_loci, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
               linestyle='--', alpha=0.7)
    ax.text(len(x) - 0.5, mean_loci + y_max * 0.01,
            f'mean={mean_loci:.0f}', fontsize=6, ha='right',
            color=COLOR_NEUTRAL_DARK, va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(chr_labels, fontsize=8)
    ax.set_xlabel('Chromosome', fontsize=10)
    ax.set_ylabel('Number of loci', fontsize=10)
    ax.set_ylim(bottom=0, top=y_max * 1.12)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def perm_distribution(
    max_lods: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
    chrom: str = '',
    bins: int = 30,
    obs_lods: Optional[np.ndarray] = None,
    ref_gene: str = '',
    ref_chr: str = '',
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
    ax.hist(vals, bins=bins, density=True,
            color=FOUNDER_COLORS['NZOJ'], alpha=0.55, label=f'Permutations (n={len(vals):,})')

    c, loc, scale = genextreme.fit(vals, f0=None)
    xr = np.linspace(vals.min() - 0.5, vals.max() + 1.5, 300)
    pdf_vals = genextreme.pdf(xr, c, loc, scale)
    ax.plot(xr, pdf_vals, color=FOUNDER_COLORS['NODJ'], linewidth=2.0, label='GEV fit')

    if threshold is not None:
        ax.axvline(threshold, color=FOUNDER_COLORS['PWKJ'], linewidth=1.5,
                   linestyle='--', label=f'95th pctile = {threshold:.2f}')
        # shade tail above threshold
        tail_x = xr[xr >= threshold]
        ax.fill_between(tail_x, genextreme.pdf(tail_x, c, loc, scale),
                        alpha=0.25, color=FOUNDER_COLORS['PWKJ'])
        n_above = int((vals >= threshold).sum())
        ax.text(threshold + 0.05 * (vals.max() - vals.min()),
                ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.1,
                f'{n_above}/{len(vals)}\nexceed thr',
                fontsize=10, color=FOUNDER_COLORS['PWKJ'], va='top')

    if obs_lods is not None:
        obs_f = obs_lods[np.isfinite(obs_lods)]
        ax.plot(obs_f, np.zeros_like(obs_f),
                '|', color=COLOR_CC_FALLBACK, alpha=0.5, markersize=5,
                label='observed genes', zorder=0)

    # info box in top-left corner (no title)
    _parts: list[str] = [f'n_perms = {len(vals):,}']
    if ref_gene:
        _parts.append(f'ref_gene = {ref_gene}')
    if ref_chr:
        _parts.append(f'ref_chr  = {ref_chr}')
    elif chrom:
        _parts.append(f'ref_chr  = {chrom}')
    ax.text(
        0.04, 0.97, '\n'.join(_parts),
        transform=ax.transAxes, ha='left', va='top',
        fontfamily='monospace', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=COLOR_NEUTRAL_DARK, alpha=0.88),
    )

    ax.set_xlabel('Max LOD Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def egene_lod_distribution(
    peak_lods: np.ndarray,
    is_egene: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
    chrom: str = '',
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
    ax.hist(below, bins=shared_bins, color=FOUNDER_COLORS['B6J'],
            alpha=0.70, label='below threshold')
    ax.hist(above, bins=shared_bins, color=FOUNDER_COLORS['NODJ'],
            alpha=0.85, label='eGenes')

    if threshold is not None:
        ax.axvline(threshold, color=FOUNDER_COLORS['PWKJ'], linewidth=1.5,
                   linestyle='--', label='GEV 95%')

    ax.set_xlabel('Peak LOD Score', fontsize=11)
    ax.set_ylabel('Gene count', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_ylim(bottom=0)
    pct = 100.0 * n_egene / n_total if n_total > 0 else 0.0
    info_lines = [
        f'{pct:.1f}%  eGenes',
        f'eGenes  = {n_egene:,}',
        f'total   = {n_total:,}',
    ]
    if threshold is not None:
        info_lines.append(f'FWER thr = {threshold:.2f}')
    ax.text(
        0.02, 0.97, '\n'.join(info_lines),
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=COLOR_NEUTRAL_DARK, alpha=0.88),
    )
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.set_title('eGene LOD Score Distribution', fontsize=11)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def mediation_profile(
    positions: np.ndarray,
    lod_base: np.ndarray,
    lod_mediated: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    gene: str = '',
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
    pos  = positions[order]
    base = lod_base[order]
    med  = lod_mediated[order]

    ax.plot(pos, base, color=FOUNDER_COLORS['NODJ'], linewidth=1.5, label='LOD Score (base)')
    ax.plot(pos, med,  color=FOUNDER_COLORS['NZOJ'], linewidth=1.5,
            linestyle='--', label='LOD Score (mediated)')
    drop = base - med
    ax.fill_between(pos, med, base, where=(base > med),
                    alpha=0.22, color=FOUNDER_COLORS['PWKJ'], label='LOD Score drop')

    # annotate peak drop position
    if drop.max() > 0:
        peak_idx = int(np.argmax(drop))
        peak_pos_drop = float(pos[peak_idx])
        peak_base_lod = float(base[peak_idx])
        peak_drop_lod = float(drop[peak_idx])
        prop_med = peak_drop_lod / max(peak_base_lod, 1e-9)
        ax.annotate(
            f'Δ={peak_drop_lod:.1f}  ({prop_med * 100:.0f}% mediated)',
            (peak_pos_drop, (float(base[peak_idx]) + float(med[peak_idx])) / 2),
            xytext=(6, 0), textcoords='offset points',
            fontsize=8, color=FOUNDER_COLORS['PWKJ'], va='center',
        )
        ax.axvline(peak_pos_drop, color=FOUNDER_COLORS['PWKJ'],
                   linewidth=0.8, linestyle=':', alpha=0.6)

    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)', fontsize=11)
    ax.set_ylabel('LOD Score', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    title = gene if gene else 'Mediation'
    if chrom:
        title += f'  chr{chrom}'
    ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def ci_strip(
    genes: list[str],
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    peak_mb: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
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
    cmap = plt.matplotlib.colormaps.get_cmap('RdYlBu_r')
    w_max = float(np.nanmax(widths))
    w_min = float(np.nanmin(widths))
    w_range = max(w_max - w_min, 1e-9)

    for i, (lo, hi, pk) in enumerate(zip(ci_lo, ci_hi, peak_mb)):
        norm_w = (float(hi - lo) - w_min) / w_range
        bar_color = cmap(norm_w)
        ax.plot([lo, hi], [i, i], color=bar_color, linewidth=3,
                solid_capstyle='round')
        ax.scatter(pk, i, color=COLOR_NEUTRAL_DARK, s=18, zorder=3)

    # median CI width reference line — vertical line at the median peak + offset
    ax.axvline(float(np.nanmean(peak_mb)) + med_width / 2,
               color=COLOR_NEUTRAL_DARK, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(float(np.nanmean(peak_mb)) + med_width / 2 + 0.1,
            n - 0.5, f'med width={med_width:.1f} Mb',
            fontsize=7, color=COLOR_NEUTRAL_DARK, va='top')

    # annotate widest-CI genes by name
    widest_idx = np.argsort(widths)[::-1][:min(3, n)]
    for wi in widest_idx:
        ax.text(float(ci_hi[wi]) + 0.1, wi,
                genes[wi], fontsize=6, va='center', color=FOUNDER_COLORS['PWKJ'])

    ax.set_yticks(range(n))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)', fontsize=11)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_title(f'Bootstrap CI  {n} eGenes', fontsize=11)
    # colorbar for CI width
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=w_min, vmax=w_max),
    )
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label('CI width (Mb)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    sns.despine(ax=ax)
    _set_font(ax)
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
    chrom: str = '',
    gene: str = '',
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
        ax.fill_between(haplo_positions[ord_h], haplo_lod[ord_h],
                        alpha=0.12, color=COLOR_CC_FALLBACK, zorder=1,
                        label='haplotype LOD')

    ord_s = np.argsort(positions)
    pos_s = positions[ord_s]
    lod_s = snp_lod[ord_s]
    bet_s = beta_dosage[ord_s]

    sc = ax.scatter(pos_s, lod_s, c=bet_s, cmap='coolwarm',
                    s=8, linewidths=0, alpha=0.7, zorder=3)
    cbar = ax.get_figure().colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('β dosage', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if peak_pos is not None:
        pk_idx = int(np.argmin(np.abs(pos_s - peak_pos)))
        pk_lod = float(lod_s[pk_idx])
        ax.scatter([peak_pos], [pk_lod], marker='*', s=120,
                   color=FOUNDER_COLORS['PWKJ'], zorder=5)

    if gene:
        ax.text(0.02, 0.96, gene, transform=ax.transAxes,
                fontsize=8, va='top', fontweight='bold',
                color=FOUNDER_COLORS['NODJ'])

    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('LOD Score', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_xlim(float(pos_s[0]), float(pos_s[-1]))
    ax.set_ylim(0, float(lod_s.max()) * 1.18)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def instability_lod_overlay(
    positions: np.ndarray,
    rop_lod: np.ndarray,
    inst_scores: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    gene: str = '',
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

    ax.fill_between(pos, rop, alpha=0.15, color=FOUNDER_COLORS['NODJ'], zorder=1)
    ax.plot(pos, rop, color=FOUNDER_COLORS['NODJ'], lw=1.2, zorder=2,
            label='ROP LOD')

    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, lw=0.8, ls='--', zorder=0)

    ax2 = ax.twinx()
    ax2.fill_between(pos, inst, alpha=0.10, color=_INST_COLOR, zorder=1)
    ax2.plot(pos, inst, color=_INST_COLOR, lw=1.0, ls='--', zorder=2,
             label='instability')
    ax2.set_ylabel('Instability Score', fontsize=11, color=_INST_COLOR)
    ax2.tick_params(labelsize=9, colors=_INST_COLOR)
    ax2.set_ylim(0, max(float(inst.max()) * 1.3, 0.2))

    if peak_pos is not None:
        ax.axvline(peak_pos, color=FOUNDER_COLORS['PWKJ'],
                   lw=0.8, ls=':', zorder=3)

    if gene:
        ax.text(0.02, 0.96, gene, transform=ax.transAxes,
                fontsize=8, va='top', fontweight='bold',
                color=FOUNDER_COLORS['NODJ'])

    # concordance info box
    from scipy.stats import spearmanr
    rho, _ = spearmanr(rop, inst)
    ax.text(
        0.98, 0.97,
        f'inst–LOD ρ = {rho:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        fontfamily='monospace', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=COLOR_NEUTRAL_DARK, alpha=0.88),
    )

    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('ROP LOD Score', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(0, float(rop.max()) * 1.18)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
              frameon=False, loc='upper left')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def mi_rop_overlay(
    positions: np.ndarray,
    rop_lod: np.ndarray,
    mi_lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    gene: str = '',
    peak_pos: Optional[float] = None,
    threshold: Optional[float] = None,
) -> plt.Axes:
    """Overlay MI LOD profile against ROP LOD on a shared position axis.

    Args:
        positions: Locus positions in Mb, shape (n_loci,).
        rop_lod: ROP LOD scores (additive haplotype scan), shape (n_loci,).
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
    rop = rop_lod[ord_i]
    mi  = mi_lod[ord_i]

    ax.fill_between(pos, rop, alpha=0.12, color=FOUNDER_COLORS['NODJ'], zorder=1)
    ax.plot(pos, rop, color=FOUNDER_COLORS['NODJ'], lw=1.2, zorder=2, label='ROP')
    ax.fill_between(pos, mi, alpha=0.08, color=_MI_COLOR, zorder=1)
    ax.plot(pos, mi, color=_MI_COLOR, lw=1.0, ls='--', zorder=2, label='MI')

    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, lw=0.8, ls='--', zorder=0)

    if peak_pos is not None:
        ax.axvline(peak_pos, color=FOUNDER_COLORS['PWKJ'],
                   lw=0.8, ls=':', zorder=3)

    if gene:
        ax.text(0.02, 0.96, gene, transform=ax.transAxes,
                fontsize=8, va='top', fontweight='bold',
                color=FOUNDER_COLORS['NODJ'])

    delta = float(np.max(mi) - np.max(rop))
    ax.text(
        0.98, 0.97,
        f'ΔLODpeak  = {delta:+.2f}\nROP peak  = {float(np.max(rop)):.2f}\n'
        f'MI  peak  = {float(np.max(mi)):.2f}',
        transform=ax.transAxes, ha='right', va='top',
        fontfamily='monospace', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=COLOR_NEUTRAL_DARK, alpha=0.88),
    )

    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('LOD Score', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(0, max(float(np.concatenate([rop, mi]).max()) * 1.2, 1.0))
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    sns.despine(ax=ax)
    _set_font(ax)
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
    from .compare import _COLOR_A as _CPY, _COLOR_B as _CR  # local import avoids circular

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5), dpi=300)

    chrs = [c for c in _CHR_ORDER if c in py_thresholds]
    extra = sorted(set(py_thresholds) - set(chrs))
    chrs = chrs + extra

    x = np.arange(len(chrs))
    w = 0.35 if r_thresholds else 0.6
    offset = -w / 2 if r_thresholds else 0.0

    py_vals = np.array([py_thresholds.get(c, float('nan')) for c in chrs])
    ax.bar(x + offset, py_vals, width=w, color=_CPY, alpha=0.8, label='Python')

    if r_thresholds:
        r_vals = np.array([r_thresholds.get(c, float('nan')) for c in chrs])
        ax.bar(x + w / 2, r_vals, width=w, color=_CR, alpha=0.8, label='R')

    if genome_wide_thr is not None:
        ax.axhline(genome_wide_thr, color=COLOR_NEUTRAL_DARK, lw=1.4, ls='--',
                   label=f'genome-wide={genome_wide_thr:.2f}')

    ax.set_xticks(x)
    ax.set_xticklabels([f'chr{c}' for c in chrs], rotation=45, ha='right',
                       fontsize=9)
    ax.set_xlabel('Chromosome', fontsize=13, fontweight='bold')
    ax.set_ylabel('FWER Threshold (LOD)', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax)
    _set_font(ax)
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

    chrs_list = list(dict.fromkeys(
        c for c in _CHR_ORDER if c in set(chroms)
    )) + sorted(set(chroms) - set(_CHR_ORDER))

    if chr_offsets is None:
        off = 0.0
        chr_offsets = {}
        for c in chrs_list:
            mask = [ch == c for ch in chroms]
            pks = peak_mb[mask] if any(mask) else np.array([0.0])
            chr_offsets[c] = off
            off += float(np.nanmax(pks)) + 15.0

    widths = ci_hi - ci_lo
    cmap = plt.matplotlib.colormaps.get_cmap('RdYlBu_r')
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
        ax.plot([x - half_w + off, x + half_w + off], [0, 0],
                color=bar_color, lw=0.4, alpha=0.3)

    for c in chrs_list:
        off = chr_offsets.get(c, 0.0)
        mask = np.array([ch == c for ch in chroms])
        if mask.any():
            mid = float(peak_mb[mask].mean()) + off
            xticks.append(mid)
            xlabels.append(f'chr{c}')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Peak LOD', fontsize=13, fontweight='bold')
    ax.set_xlabel('Genomic Position (Mb, cumulative)', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cb = ax.get_figure().colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label('CI width (Mb)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def scan_throughput(
    chr_labels: list[str],
    py_genes_per_min: np.ndarray,
    *,
    r_genes_per_min: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    py_total_label: str = '',
    r_total_label: str = '',
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

    ax.bar(x + offset, py_genes_per_min, width=w,
           color=_CPY, alpha=0.85,
           label=f'Python  {py_total_label}'.strip())

    if r_genes_per_min is not None:
        ax.bar(x + w / 2, r_genes_per_min, width=w,
               color=_CR, alpha=0.85,
               label=f'R (miQTL)  {r_total_label}'.strip())

    ax.set_xticks(x)
    ax.set_xticklabels([f'chr{c}' for c in chr_labels],
                       rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Chromosome', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (genes/min)', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax)
    _set_font(ax)
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
    ax.bar(x, mean_drop, color=FOUNDER_COLORS['NODJ'], alpha=0.85, edgecolor='none',
           yerr=yerr,
           error_kw={'linewidth': 0.8, 'ecolor': COLOR_NEUTRAL_DARK, 'capsize': 3})

    bar_max = float(np.nanmax(mean_drop)) if len(mean_drop) else 1.0
    if n_pairs is not None:
        for xi, np_val in enumerate(n_pairs):
            ax.text(xi, float(mean_drop[xi]) + bar_max * 0.02,
                    f'n={int(np_val)}',
                    ha='center', va='bottom', fontsize=6, color=COLOR_NEUTRAL_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(chrs, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Chromosome', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean LOD Score Drop (±1 SD)', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0, top=bar_max * 1.25)

    _info = f'outcomes  = {n_outcomes:,}' if n_outcomes else ''
    if _info:
        ax.text(0.98, 0.97, _info, transform=ax.transAxes,
                fontsize=10, va='top', ha='right', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor=COLOR_NEUTRAL_DARK, alpha=0.88))
    sns.despine(ax=ax)
    _set_font(ax)
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
        1, 2 if has_mem else 1,
        figsize=(9 if has_mem else 5, max(3, len(labels) * 0.45)),
        dpi=300,
    )
    ax_t = axes[0] if has_mem else axes

    ax_t.barh(labels, elapsed_s, color=FOUNDER_COLORS['NODJ'], alpha=0.85,
               edgecolor='none')
    ax_t.invert_yaxis()
    ax_t.set_xlabel('Elapsed (s)', fontsize=13, fontweight='bold')
    ax_t.tick_params(labelsize=11)
    sns.despine(ax=ax_t)

    if has_mem:
        axes[1].barh(labels, peak_heap_mb.astype(float), color=FOUNDER_COLORS['NZOJ'],
                     alpha=0.85, edgecolor='none')
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Peak heap (MB)', fontsize=13, fontweight='bold')
        axes[1].tick_params(labelsize=11)
        sns.despine(ax=axes[1])

    _set_font(ax_t)
    return fig


def chromosome_lod_profile(
    positions: np.ndarray,
    lod: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    threshold: Optional[float] = None,
    color: str = FOUNDER_COLORS['NODJ'],
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
            f'n={len(positions):,} loci',
            transform=ax.transAxes,
            fontsize=7,
            va='top',
            color=COLOR_NEUTRAL_DARK,
        )
    return ax


def peak_locus_zoom(
    positions: np.ndarray,
    lod: np.ndarray,
    peak_mb: float,
    *,
    window_mb: float = 2.0,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    threshold: Optional[float] = None,
    ci_lo_mb: Optional[float] = None,
    ci_hi_mb: Optional[float] = None,
    color: str = FOUNDER_COLORS['NODJ'],
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
            f'LOD={peak_lod_val:.1f}',
            ha='center',
            va='bottom',
            fontsize=7,
            color=FOUNDER_COLORS['PWKJ'],
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
    color: str = FOUNDER_COLORS['NODJ'],
    ylabel: str = 'LOD Score',
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
        {chr: gene_name} annotated in the title when not the top gene.

    Returns
    -------
    dict[str, plt.Figure]
        Keyed by chromosome string.
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    chr_vals = lod_df['chr'].astype(str)
    chrs = _ordered_chrs(chr_vals)

    out: dict[str, plt.Figure] = {}
    for c in chrs:
        lod_c = lod_df[chr_vals == c]
        if lod_c.empty:
            continue

        if pk_df is not None:
            pk_c = pk_df[pk_df['chr'].astype(str) == c]
            top_gene = (
                str(pk_c.loc[pk_c['peak_lod'].idxmax(), 'gene'])
                if not pk_c.empty
                else ''
            )
        else:
            gene_max = lod_c.groupby('gene')['lod'].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ''

        if not top_gene:
            continue
        lod_top = lod_c[lod_c['gene'] == top_gene].sort_values('pos_mb')
        if lod_top.empty:
            continue

        thr_val: Optional[float] = thr.get(c)
        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
        lod_profile(
            lod_top['pos_mb'].values,
            lod_top['lod'].values,
            ax=ax,
            chrom=c,
            threshold=thr_val,
            color=color,
            fill=True,
        )
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        title = f'{top_gene}  chr{c}'
        p1 = prio.get(c)
        if p1 and p1 != top_gene:
            p1_data = lod_c[lod_c['gene'] == p1].sort_values('pos_mb')
            if not p1_data.empty:
                p1_max = float(p1_data['lod'].max())
                ax.plot(
                    p1_data['pos_mb'].values,
                    p1_data['lod'].values,
                    color=FOUNDER_COLORS['PWKJ'],
                    linewidth=1.2,
                    alpha=0.80,
                    label=f'{p1}  LOD={p1_max:.1f}',
                    zorder=3,
                )
                ax.legend(fontsize=10, loc='upper right', framealpha=0.85)
                title += f'  /  {p1}'
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
        {chr: gene_name} forces the plotted gene per chromosome.

    Returns
    -------
    dict[str, plt.Figure]
    """
    prio = priority_genes or {}
    chr_vals = snp_df['chr'].astype(str)
    chrs = _ordered_chrs(chr_vals)

    out: dict[str, plt.Figure] = {}
    for c in chrs:
        snp_c = snp_df[chr_vals == c]
        if snp_c.empty:
            continue

        if prio.get(c) and prio[c] in snp_c['gene'].values:
            top_gene = prio[c]
        elif pk_df is not None:
            pk_c = pk_df[pk_df['chr'].astype(str) == c]
            top_gene = (
                str(pk_c.loc[pk_c['peak_lod'].idxmax(), 'gene'])
                if not pk_c.empty
                else ''
            )
        else:
            gene_max = snp_c.groupby('gene')['lod'].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ''

        if not top_gene:
            continue
        snp_g = snp_c[snp_c['gene'] == top_gene].sort_values('pos_mb')
        if snp_g.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
        ax.scatter(
            snp_g['pos_mb'].values,
            snp_g['lod'].values,
            color=_SNP_COLOR,
            s=10,
            alpha=0.80,
            linewidths=0,
            zorder=2,
        )
        if len(snp_g) > 0:
            top_row = snp_g.loc[snp_g['lod'].idxmax()]
            snp_label = str(top_row.get('snp', ''))
            if snp_label:
                ax.annotate(
                    snp_label,
                    xy=(top_row['pos_mb'], top_row['lod']),
                    xytext=(8, 6),
                    textcoords='offset points',
                    fontsize=10,
                    color=FOUNDER_COLORS['PWKJ'],
                    arrowprops=dict(
                        arrowstyle='-',
                        color=FOUNDER_COLORS['PWKJ'],
                        lw=0.8,
                    ),
                )
        ax.set_xlim(
            float(snp_g['pos_mb'].min()),
            float(snp_g['pos_mb'].max()),
        )
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f'Chr {c} position (Mb)', fontsize=13)
        ax.set_ylabel('LOD Score (SNP)', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_title(f'SNP fine-mapping  —  {top_gene}  chr{c}', fontsize=13)
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
    zoom_window_mb: float = 1.0,
) -> dict[str, plt.Figure]:
    """Per-chromosome qtl2-style locus panels (founder effects + LOD profile + zoom).

    qtl2 equivalent: ``plot_coefCC`` at peak locus stacked above ``plot_scan1``.
    The effect bar uses peak-locus betas from *eff_df* rather than a per-locus
    coefficient profile, because mosaiqtl stores only peak-locus betas per gene.
    When *tss_mb_map* is provided, a third panel shows a zoomed cis-window view
    with TSS and peak annotated relative to each other.

    Parameters
    ----------
    lod_df : pd.DataFrame
        Columns: gene, chr (str), pos_mb, lod.
    eff_df : pd.DataFrame, optional
        Columns: gene, chr (str), peak_mb, peak_lod, beta_A … beta_H.
        When provided, the top panel shows founder effect bars.
    founders : list[str], optional
        Display-name founder list matching beta column order (e.g. AJ, B6J …).
    thr_by_chr : dict, optional
        Per-chromosome GEV threshold.
    priority_genes : dict, optional
        {chr: gene_name}.
    tss_mb_map : dict, optional
        {gene_name: tss_mb}. When provided, adds a zoomed LOD panel per figure.
    zoom_window_mb : float
        Margin around the [TSS, peak] interval in the zoom panel. Default 1.0 Mb.

    Returns
    -------
    dict[str, plt.Figure]
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    chr_vals = lod_df['chr'].astype(str)
    chrs = _ordered_chrs(chr_vals)

    out: dict[str, plt.Figure] = {}
    for c in chrs:
        lod_c = lod_df[chr_vals == c]
        if lod_c.empty:
            continue

        if eff_df is not None:
            eff_c = eff_df[eff_df['chr'].astype(str) == c]
            top_gene = (
                str(eff_c.loc[eff_c['peak_lod'].idxmax(), 'gene'])
                if not eff_c.empty
                else ''
            )
        else:
            gene_max = lod_c.groupby('gene')['lod'].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ''

        if prio.get(c):
            candidates = (
                eff_df[eff_df['chr'].astype(str) == c]['gene'].values
                if eff_df is not None
                else lod_c['gene'].unique()
            )
            if prio[c] in candidates:
                top_gene = prio[c]

        if not top_gene:
            continue
        lod_top = lod_c[lod_c['gene'] == top_gene].sort_values('pos_mb')
        if lod_top.empty:
            continue

        peak_betas: Optional[np.ndarray] = None
        peak_pos: Optional[float] = None
        if eff_df is not None and founders is not None:
            eff_row_df = eff_df[
                (eff_df['chr'].astype(str) == c) & (eff_df['gene'] == top_gene)
            ]
            if not eff_row_df.empty:
                beta_cols = [b for b in eff_row_df.columns if b.startswith('beta_')]
                row0 = eff_row_df.iloc[0]
                peak_betas = np.array([float(row0[b]) for b in beta_cols])
                peak_pos = float(row0.get('peak_mb', lod_top['pos_mb'].iloc[0]))

        thr_val: Optional[float] = thr.get(c)
        tss = (tss_mb_map or {}).get(top_gene)
        fig = locus_panel(
            lod_top['pos_mb'].values,
            lod_top['lod'].values,
            founders or [],
            peak_betas=peak_betas,
            peak_pos=peak_pos,
            tss_mb=tss,
            zoom_window_mb=zoom_window_mb,
            threshold=thr_val,
            chrom=c,
            highlight_gene=top_gene,
        )
        out[c] = fig

    return out


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
    zoom_window_mb: float = 1.0,
    threshold: Optional[float] = None,
    chrom: str = '',
    highlight_gene: Optional[str] = None,
) -> plt.Figure:
    """qtl2-style locus view: founder effect bars + full LOD profile + zoom panel.

    Panels present depend on which optional arguments are supplied:
    - *peak_betas* adds the founder effect bar panel at the top.
    - *tss_mb* adds a zoomed LOD panel showing TSS and peak positions relative
      to each other, replacing the chromosome-wide view with a cis-window close-up.
    - Neither → single full-chromosome LOD panel.

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
        Peak locus position in Mb; annotated on both LOD panels.
    tss_mb : float, optional
        Gene TSS position in Mb; when provided, adds a zoomed cis-window panel
        showing TSS (dashed) and peak (dotted) relative to each other.
    zoom_window_mb : float
        Half-width of margin on each side of the [TSS, peak] interval in the zoom
        panel. Default 1.0 Mb.
    threshold : float, optional
        Permutation threshold; drawn as a dashed line on the LOD panels.
    chrom : str
        Chromosome label for the x-axis.
    highlight_gene : str, optional
        Gene name shown in the figure title.

    Returns
    -------
    plt.Figure
    """
    has_betas = peak_betas is not None
    has_zoom  = tss_mb is not None

    # ── build panel layout ────────────────────────────────────────────────────
    n_panels = (1 if has_betas else 0) + 1 + (1 if has_zoom else 0)
    ratios   = ([2] if has_betas else []) + [1] + ([1] if has_zoom else [])
    fig_h    = 2.5 * sum(ratios)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, fig_h), dpi=300,
        gridspec_kw={'height_ratios': ratios},
        squeeze=False,
    )
    axes_flat = axes.ravel()
    ax_idx = 0

    title = highlight_gene or ''
    if chrom:
        title = f'{title}  chr{chrom}' if title else f'chr{chrom}'

    # ── panel 1: founder effect bars ─────────────────────────────────────────
    if has_betas:
        ax_eff = axes_flat[ax_idx]; ax_idx += 1
        _effect_bars(peak_betas, founders, ax=ax_eff)
        ax_eff.set_xlabel('')
        if title:
            ax_eff.set_title(title, fontsize=9)
        sns.despine(ax=ax_eff)

    # ── panel 2: full-chromosome LOD profile ─────────────────────────────────
    ax_full = axes_flat[ax_idx]; ax_idx += 1
    lod_profile(positions, lod, ax=ax_full, chrom=chrom,
                threshold=threshold, peak_pos=peak_pos, fill=True,
                tss_mb=tss_mb)
    ax_full.set_title('')
    # gene track arrow at TSS in Panel 2 when tss_mb is provided
    if tss_mb is not None:
        _ylim2 = ax_full.get_ylim()
        _arrow_y = _ylim2[1] * 0.06
        ax_full.annotate(
            '', xy=(tss_mb + 0.3, _arrow_y), xytext=(tss_mb, _arrow_y),
            arrowprops=dict(arrowstyle='->', color=COLOR_CC_FALLBACK,
                            lw=1.0),
        )
        _gene_label = highlight_gene or ''
        if _gene_label:
            ax_full.text(tss_mb, _arrow_y + _ylim2[1] * 0.04,
                         _gene_label, fontsize=7, ha='left',
                         color=COLOR_CC_FALLBACK, va='bottom')
    if not has_betas and title:
        # bold corner label instead of set_title
        ax_full.text(0.02, 0.96, title, transform=ax_full.transAxes,
                     fontsize=9, va='top', fontweight='bold',
                     color=FOUNDER_COLORS['NODJ'])
    sns.despine(ax=ax_full)

    # ── panel 3: zoomed cis-window ────────────────────────────────────────────
    if has_zoom:
        ax_zoom = axes_flat[ax_idx]
        anchor  = peak_pos if peak_pos is not None else tss_mb
        lo = min(tss_mb, anchor) - zoom_window_mb       # type: ignore[arg-type]
        hi = max(tss_mb, anchor) + zoom_window_mb       # type: ignore[arg-type]

        mask = (positions >= lo) & (positions <= hi)
        pos_z = positions[mask]
        lod_z = lod[mask]

        if pos_z.size > 0:
            lod_profile(pos_z, lod_z, ax=ax_zoom, chrom=chrom,
                        threshold=threshold, peak_pos=None, fill=True)
        ax_zoom.set_xlim(lo, hi)

        ax_zoom.axvline(tss_mb, color=COLOR_NEUTRAL_DARK,
                        linewidth=0.9, linestyle='--', zorder=2, label='TSS')
        if peak_pos is not None:
            ax_zoom.axvline(peak_pos, color=FOUNDER_COLORS['PWKJ'],
                            linewidth=0.9, linestyle=':', zorder=2, label='peak')
        ax_zoom.set_xlabel(f'chr{chrom} position (Mb)')
        ax_zoom.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax_zoom)

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
        Columns: gene, chr (str), pos_mb, lod.
    pk_df : pd.DataFrame, optional
        Peak table for top-gene selection.
    thr_by_chr : dict, optional
        Per-chromosome permutation threshold.
    priority_genes : dict, optional
        {chr: gene_name} annotated in panel titles.
    n_cols : int
        Panels per row (default 4 → 4×5 for 20 chromosomes).

    Returns
    -------
    plt.Figure
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    chr_vals = lod_df['chr'].astype(str)
    chrs = _ordered_chrs(chr_vals)
    n_chrs = len(chrs)
    n_rows = (n_chrs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 2.2),
        dpi=300,
        squeeze=False,
    )

    for i, c in enumerate(chrs):
        ax = axes[i // n_cols, i % n_cols]
        lod_c = lod_df[chr_vals == c]

        if pk_df is not None:
            pk_c = pk_df[pk_df['chr'].astype(str) == c]
            top_gene = (
                str(pk_c.loc[pk_c['peak_lod'].idxmax(), 'gene'])
                if not pk_c.empty
                else ''
            )
        else:
            gene_max = lod_c.groupby('gene')['lod'].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ''

        if top_gene:
            lod_top = lod_c[lod_c['gene'] == top_gene].sort_values('pos_mb')
            if not lod_top.empty:
                lod_profile(
                    lod_top['pos_mb'].values,
                    lod_top['lod'].values,
                    ax=ax,
                    chrom='',
                    threshold=thr.get(c),
                    fill=True,
                    linewidth=0.8,
                )
                label = prio.get(c) or top_gene
                ax.set_title(f'chr{c}  {label[:14]}', fontsize=7)
                ax.set_xlabel(f'chr{c} (Mb)', fontsize=6)
                ax.tick_params(labelsize=5)
                sns.despine(ax=ax)
                continue
        ax.set_visible(False)

    for j in range(n_chrs, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle('LOD Score Profiles — Top Gene per Chromosome', fontsize=9, y=1.01)
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
        Columns: gene, chr (str), pos_mb, lod.
    eff_df : pd.DataFrame, optional
        Columns: gene, chr (str), peak_mb, peak_lod, beta_A … beta_H.
    founders : list[str], optional
        Display-name founders matching beta column order.
    thr_by_chr : dict, optional
        Per-chromosome GEV threshold.
    priority_genes : dict, optional
        {chr: gene_name}.
    n_cols : int
        Panels per row.

    Returns
    -------
    plt.Figure
    """
    thr = thr_by_chr or {}
    prio = priority_genes or {}
    show_effects = eff_df is not None and founders is not None
    chr_vals = lod_df['chr'].astype(str)
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
            eff_c = eff_df[eff_df['chr'].astype(str) == c]
            top_gene = (
                str(eff_c.loc[eff_c['peak_lod'].idxmax(), 'gene'])
                if not eff_c.empty
                else ''
            )
        else:
            gene_max = lod_c.groupby('gene')['lod'].max()
            top_gene = str(gene_max.idxmax()) if len(gene_max) > 0 else ''

        if prio.get(c):
            candidates = (
                eff_df[eff_df['chr'].astype(str) == c]['gene'].values
                if eff_df is not None
                else lod_c['gene'].unique()
            )
            if prio[c] in candidates:
                top_gene = prio[c]

        if not top_gene:
            continue
        lod_top = lod_c[lod_c['gene'] == top_gene].sort_values('pos_mb')
        if lod_top.empty:
            continue
        label = top_gene[:14]
        thr_val: Optional[float] = thr.get(c)

        if show_effects:
            eff_row_df = eff_df[
                (eff_df['chr'].astype(str) == c) & (eff_df['gene'] == top_gene)
            ]
            if not eff_row_df.empty:
                beta_cols = [b for b in eff_row_df.columns if b.startswith('beta_')]
                betas = np.array([float(eff_row_df.iloc[0][b]) for b in beta_cols])
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=outer[row, col],
                    height_ratios=[2, 1],
                    hspace=0.08,
                )
                ax_top = fig.add_subplot(inner[0])
                ax_bot = fig.add_subplot(inner[1])

                _effect_bars(betas, founders, ax=ax_top)  # type: ignore[arg-type]
                ax_top.set_xlabel('')
                ax_top.tick_params(labelsize=5)
                ax_top.set_title(f'chr{c}  {label}', fontsize=7)
                sns.despine(ax=ax_top)

                lod_profile(
                    lod_top['pos_mb'].values, lod_top['lod'].values,
                    ax=ax_bot, chrom='', threshold=thr_val,
                    fill=True, linewidth=0.8,
                )
                ax_bot.set_xlabel(f'chr{c} (Mb)', fontsize=6)
                ax_bot.tick_params(labelsize=5)
                ax_bot.set_title('')
                sns.despine(ax=ax_bot)
                continue

        # LOD-only fallback
        ax = fig.add_subplot(outer[row, col])
        lod_profile(
            lod_top['pos_mb'].values, lod_top['lod'].values,
            ax=ax, chrom='', threshold=thr_val, fill=True, linewidth=0.8,
        )
        ax.set_xlabel(f'chr{c} (Mb)', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(f'chr{c}  {label}', fontsize=7)
        sns.despine(ax=ax)

    for j in range(n_chrs, n_rows * n_cols):
        ax_empty = fig.add_subplot(outer[j // n_cols, j % n_cols])
        ax_empty.set_visible(False)

    title = 'Locus panels — top gene per chromosome'
    if show_effects:
        title += '  (founder effects + LOD)'
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
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 2.5),
        dpi=300,
        squeeze=False,
    )

    for i, c in enumerate(chrs):
        ax = axes[i // n_cols, i % n_cols]
        perm_distribution(perm_by_chr[c], ax=ax, threshold=thr_by_chr.get(c), chrom=c)
        ax.tick_params(labelsize=5)
        ax.set_xlabel('Max LOD Score', fontsize=6)
        ax.set_ylabel('Count', fontsize=6)
        try:
            legend = ax.get_legend()
            if legend:
                legend.prop.set_size(5)
        except AttributeError:
            pass
        sns.despine(ax=ax)

    for j in range(n_chrs, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle('Permutation null distributions by chromosome', fontsize=9, y=1.01)
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
    chr_arr = pk_df['chr'].astype(str).values
    pos_arr = pk_df['peak_mb'].values.astype(float)
    lod_arr = pk_df['peak_lod'].values.astype(float)
    gene_arr = pk_df['gene'].values if 'gene' in pk_df.columns else np.full(len(pk_df), '')
    chrs = _ordered_chrs(pd.Series(chr_arr))

    fig, (ax_mh, ax_bar) = plt.subplots(
        2, 1,
        figsize=(14, 5.5),
        dpi=300,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    egene_color = FOUNDER_COLORS['PWKJ']
    offset = 0.0
    xticks: list[float] = []
    egene_counts: list[int] = []
    # track global top gene across chromosomes for annotation
    top_x: Optional[float] = None
    top_y: float = 0.0
    top_gene_name: str = ''

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
        ax_mh.scatter(
            x[~eg_c], lod_c[~eg_c],
            color=COLOR_CC_FALLBACK, s=3, linewidths=0, zorder=2, alpha=0.45,
        )
        ax_mh.scatter(
            x[eg_c], lod_c[eg_c],
            color=egene_color, s=8, linewidths=0, zorder=3,
        )

        # track genome-wide peak for labeling
        if len(lod_c) > 0 and float(lod_c.max()) > top_y:
            top_y = float(lod_c.max())
            peak_i = int(np.argmax(lod_c))
            top_x = float(x[peak_i])
            top_gene_name = str(genes_c[peak_i]) if len(genes_c) > peak_i else ''

        if thr_by_chr and (thr_val := thr_by_chr.get(c)) is not None:
            ax_mh.hlines(
                thr_val,
                offset,
                offset + float(pos_c.max()),
                colors=COLOR_NEUTRAL_DARK,
                linewidth=0.7,
                linestyle='--',
                zorder=1,
            )

        xticks.append(float(offset + (float(pos_c.max()) - float(pos_c.min())) / 2.0))
        egene_counts.append(int(eg_c.sum()))
        offset += float(pos_c.max()) + _CHR_GAP_MB

    ax_mh.set_xlim(0, offset)
    ax_mh.set_ylim(bottom=0)
    ax_mh.set_xticks(xticks)
    ax_mh.set_xticklabels(chrs, fontsize=8)
    ax_mh.tick_params(axis='x', length=2)
    ax_mh.set_ylabel('Peak LOD Score', fontsize=11)
    n_genes = len(pk_df)
    n_eg = int(is_egene.sum())
    ax_mh.set_title(
        f'Genome-wide peak LOD  ({n_genes:,} genes,  {n_eg:,} eGenes  [{100*n_eg/max(n_genes,1):.1f}%])',
        fontsize=11,
    )
    # label the single genome-wide peak gene
    if top_x is not None and top_gene_name:
        ax_mh.annotate(
            top_gene_name, (top_x, top_y),
            xytext=(4, 4), textcoords='offset points',
            fontsize=8, color=egene_color, fontweight='bold',
        )
    # scan_stats summary box — top-right corner of Manhattan
    stat_lines: list[str] = []
    for key, label in [('n_genes', 'Genes'), ('n_egenes', 'eGenes'),
                       ('perm_n', 'Perms'), ('top_gene', 'Top gene'),
                       ('peak_chr', 'Peak chr')]:
        val = stats.get(key)
        if val is not None:
            stat_lines.append(f'{label}: {val}')
    if stat_lines:
        ax_mh.text(0.99, 0.98, '\n'.join(stat_lines),
                   transform=ax_mh.transAxes, fontsize=7,
                   va='top', ha='right', color=COLOR_NEUTRAL_DARK,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=COLOR_CC_FALLBACK, alpha=0.8))
    sns.despine(ax=ax_mh)

    ax_bar.bar(range(len(chrs)), egene_counts, color=egene_color, width=0.7)
    # count labels on top of bars
    bar_max = max(egene_counts) if egene_counts else 1
    for xi, cnt in enumerate(egene_counts):
        if cnt > 0:
            ax_bar.text(xi, cnt + bar_max * 0.02, str(cnt),
                        ha='center', va='bottom', fontsize=6)
    ax_bar.set_xticks(range(len(chrs)))
    ax_bar.set_xticklabels(chrs, fontsize=8)
    ax_bar.set_ylabel('eGenes', fontsize=10)
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
        1, n_panels,
        figsize=(n_panels * 4.0, 3.5),
        dpi=300,
        gridspec_kw={'width_ratios': width_ratios},
    )
    axes = axes if n_panels > 1 else [axes]

    chrs_ordered = [c for c in _CHR_ORDER if c in thr_by_chr]
    chrs_ordered += sorted(set(thr_by_chr) - set(_CHR_ORDER))
    thrs = [thr_by_chr[c] if thr_by_chr[c] is not None else 0.0
            for c in chrs_ordered]

    ax0 = axes[0]
    ax0.bar(
        range(len(chrs_ordered)),
        thrs,
        color=FOUNDER_COLORS['NZOJ'],
        width=0.7,
    )
    ax0.set_xticks(range(len(chrs_ordered)))
    ax0.set_xticklabels(chrs_ordered, fontsize=10, rotation=45, ha='right')
    ax0.set_ylabel('GEV Threshold (LOD)', fontsize=12)
    ax0.set_title('Permutation threshold per chromosome', fontsize=12)
    ax0.set_xlim(-0.5, len(chrs_ordered) - 0.5)
    ax0.set_ylim(bottom=0)
    ax0.tick_params(axis='y', labelsize=10)
    sns.despine(ax=ax0)

    if has_cis:
        ax1 = axes[1]
        finite = cis_dist_mb[np.isfinite(cis_dist_mb)]
        ax1.hist(finite, bins=40, color=FOUNDER_COLORS['NODJ'])
        ax1.set_xlabel('Cis distance (Mb)', fontsize=12)
        ax1.set_ylabel('eGene count', fontsize=12)
        ax1.set_title('eGene cis-distance', fontsize=12)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(labelsize=10)
        sns.despine(ax=ax1)
        ax_stats = axes[2]
    else:
        ax_stats = axes[1]

    stats = scan_stats or {}
    n_genes = stats.get('n_genes', len(pk_df))
    n_egenes = stats.get('n_egenes', 0)
    n_perms = stats.get('n_perms', 0)
    top_gene = stats.get('top_gene', '')
    peak_chr = stats.get('peak_chr', '')

    _perm_str = (f'{n_perms:,}  (FWER via GEV)'
                 if n_perms > 0 else 'no permutations run')
    _egene_method = 'FWER (GEV perm)' if n_perms > 0 else 'LOD threshold'

    text_lines = [
        f'Genes tested:  {n_genes:,}',
        f'eGenes:        {n_egenes:,}  ({_egene_method})',
        f'Perms:         {_perm_str}',
        '',
        f'Top gene:      {top_gene}',
        f'Peak chr:      chr{peak_chr}',
    ]
    ax_stats.text(
        0.05, 0.95,
        '\n'.join(text_lines),
        transform=ax_stats.transAxes,
        fontsize=10,
        va='top',
        ha='left',
        fontfamily=['Carlito', 'Calibri', 'DejaVu Sans'],
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor=COLOR_NEUTRAL_DARK,
            alpha=0.9,
        ),
    )
    ax_stats.set_axis_off()

    fig.tight_layout()
    return fig
