"""LOD profiles, Manhattan plots, region zooms, and SNP association plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..genotype.founders import COLOR_NEUTRAL_DARK, COLOR_CC_FALLBACK, FOUNDER_COLORS

_CHR_ORDER: list[str] = [str(i) for i in range(1, 20)] + ['X']
_CHR_GAP_MB: float = 10.0
_CHR_COLORS: tuple[str, str] = (FOUNDER_COLORS['NODJ'], COLOR_NEUTRAL_DARK)


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
        Permutation threshold — drawn as horizontal dashed line.
    peak_pos : float, optional
        Peak position in Mb — drawn as vertical dotted line.
    color : str
        Line color; must be imported from founders.py.
    linewidth : float
        Line width.
    fill : bool
        Shade area under the LOD curve.
    fill_alpha : float
        Alpha for the fill band.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), dpi=300)
    order = np.argsort(positions)
    positions = positions[order]
    lod = lod[order]
    ax.plot(positions, lod, color=color, linewidth=linewidth, zorder=2)
    if fill:
        ax.fill_between(positions, 0, lod, alpha=fill_alpha, color=color, zorder=1)
    ax.set_xlim(float(positions[0]), float(positions[-1]))
    ax.set_ylim(0, max(float(lod.max()) * 1.15, 1.0))
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)')
    ax.set_ylabel('LOD')
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
                   linestyle='--', zorder=1)
    if peak_pos is not None:
        ax.axvline(peak_pos, color=FOUNDER_COLORS['PWKJ'],
                   linewidth=0.8, linestyle=':', zorder=1)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def manhattan(
    scans: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    figsize: tuple[float, float] = (14, 3),
    threshold: Optional[float] = None,
    highlight_chrom: Optional[str] = None,
    highlight_color: str = FOUNDER_COLORS['PWKJ'],
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

    for ci, chrom in enumerate(chroms):
        pos, lod = scans[chrom]
        order = np.argsort(pos)
        pos   = pos[order]
        lod   = lod[order]
        x     = pos + offset
        color = (highlight_color if chrom == highlight_chrom
                 else _CHR_COLORS[ci % 2])
        ax.scatter(x, lod, color=color, s=1.2, linewidths=0, zorder=2, rasterized=True)
        xticks.append(float(offset + (pos[-1] - pos[0]) / 2))
        xlabels.append(chrom)
        offset += float(pos[-1]) + _CHR_GAP_MB

    ax.set_xlim(0, offset)
    ax.set_ylim(0, None)
    ax.set_ylabel('LOD')
    ax.set_xlabel('Chromosome')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis='x', length=2)
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
                   linestyle='--', zorder=1)
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
    ax.set_xlabel('LOD')
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
    ax.set_ylabel('Peak LOD')
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
    ax.set_xlabel('LMM LOD')
    ax.set_ylabel('NB-GLMM LOD')
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
    ax.bar(x, n_loci, color=FOUNDER_COLORS['NODJ'], width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(chr_labels, fontsize=7)
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Number of loci')
    ax.set_ylim(bottom=0)
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

    Returns
    -------
    plt.Axes
    """
    from scipy.stats import genextreme

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), dpi=300)

    vals = max_lods[np.isfinite(max_lods)]
    ax.hist(vals, bins=bins, density=True,
            color=FOUNDER_COLORS['NZOJ'], alpha=0.6, label='Permutation max-LOD')

    c, loc, scale = genextreme.fit(vals, f0=None)
    xr = np.linspace(vals.min() - 0.5, vals.max() + 1.0, 200)
    ax.plot(xr, genextreme.pdf(xr, c, loc, scale),
            color=FOUNDER_COLORS['NODJ'], linewidth=1.5, label='GEV fit')

    if threshold is not None:
        ax.axvline(threshold, color=FOUNDER_COLORS['PWKJ'], linewidth=1.2,
                   linestyle='--', label=f'95th pctile = {threshold:.2f}')

    ax.set_xlabel('Max LOD')
    ax.set_ylabel('Density')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    label = f'chr{chrom}  n={len(vals)}' if chrom else f'n={len(vals)}'
    ax.set_title(f'Permutation distribution  {label}', fontsize=9)
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
        _, ax = plt.subplots(figsize=(7, 4), dpi=300)

    below = peak_lods[~is_egene]
    above = peak_lods[is_egene]
    ax.hist(below, bins=bins, color=FOUNDER_COLORS['B6J'],
            alpha=0.8, label='below threshold')
    ax.hist(above, bins=bins, color=FOUNDER_COLORS['NODJ'],
            alpha=0.9, label=f'eGenes (n={is_egene.sum()})')

    if threshold is not None:
        ax.axvline(threshold, color=FOUNDER_COLORS['PWKJ'], linewidth=1.2,
                   linestyle='--', label=f'GEV 95% = {threshold:.2f}')

    ax.set_xlabel('Peak LOD')
    ax.set_ylabel('Number of eGenes')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    n_total = len(peak_lods)
    label = f'chr{chrom}  n={n_total}' if chrom else f'n={n_total}'
    ax.set_title(f'eGene calling  {label}', fontsize=9)
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
        _, ax = plt.subplots(figsize=(10, 3), dpi=300)

    order = np.argsort(positions)
    pos  = positions[order]
    base = lod_base[order]
    med  = lod_mediated[order]

    ax.plot(pos, base, color=FOUNDER_COLORS['NODJ'], linewidth=1.2, label='LOD base')
    ax.plot(pos, med,  color=FOUNDER_COLORS['NZOJ'], linewidth=1.2,
            linestyle='--', label='LOD mediated')
    ax.fill_between(pos, med, base, where=(base > med),
                    alpha=0.20, color=FOUNDER_COLORS['PWKJ'], label='LOD drop')

    ax.set_xlim(float(pos[0]), float(pos[-1]))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)')
    ax.set_ylabel('Mediation LOD')
    ax.legend(fontsize=8, frameon=False)
    title = 'Mediation'
    if gene:
        title += f'  {gene}'
    if chrom:
        title += f'  chr{chrom}'
    ax.set_title(title, fontsize=9)
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
        _, ax = plt.subplots(figsize=(8, max(3, n * 0.35)), dpi=300)

    for i, (lo, hi, pk) in enumerate(zip(ci_lo, ci_hi, peak_mb)):
        ax.plot([lo, hi], [i, i], color=FOUNDER_COLORS['NODJ'],
                linewidth=3, solid_capstyle='round')
        ax.scatter(pk, i, color=FOUNDER_COLORS['PWKJ'], s=30, zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)')
    ax.set_title(f'Bootstrap CI  {n} eGenes', fontsize=9)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


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
