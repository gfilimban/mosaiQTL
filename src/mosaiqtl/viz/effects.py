"""Founder effect visualizations: coefficient profiles, effect bars, and caterpillar plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..genotype.founders import (
    COLOR_NEUTRAL_DARK, COLOR_CC_FALLBACK, FOUNDER_COLORS, FOUNDER_ORDER,
)

_CHR_ORDER: list[str] = [str(i) for i in range(1, 20)] + ['X']
_CHR_GAP_MB: float = 10.0


def _set_font(ax: plt.Axes) -> None:
    """Apply Carlito/Calibri to all text elements of ax."""
    for item in ([ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily(['Carlito', 'Calibri', 'DejaVu Sans'])


def _recover_full_betas(betas: np.ndarray) -> np.ndarray:
    """Recover J-th founder via sum-to-zero: beta_J = -sum(betas[:J-1])."""
    return np.append(betas, -betas.sum())


def coef_profile(
    positions: np.ndarray,
    betas: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = '',
    linewidth: float = 1.1,
    alpha: float = 0.85,
    show_legend: bool = True,
    ylim: Optional[float] = None,
) -> plt.Axes:
    """Founder coefficient profiles along a chromosome.

    Plots all J founder haplotype effect estimates as colored lines across
    the chromosome.

    Parameters
    ----------
    positions : np.ndarray
        Marker positions in Mb, shape (n_loci,).
    betas : np.ndarray
        Haplotype effects, shape (n_loci, J) sum-to-zero, or (n_loci, J-1).
    founders : list[str]
        Founder names in FOUNDER_ORDER; controls line colors.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    chrom : str
        Chromosome label for x-axis.
    linewidth : float
        Line width per founder.
    alpha : float
        Line opacity.
    show_legend : bool
        Include founder color legend.
    ylim : float, optional
        Symmetric y-axis limit ±ylim. Clips blown-up betas at near-zero-prob
        loci (rare founders → huge coefficients). Recommended: 3× phenotype SD.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), dpi=300)

    order = np.argsort(positions)
    positions = positions[order]
    betas = betas[order]

    # accept (n_loci, J) sum-to-zero betas or legacy (n_loci, J-1)
    j = len(founders)
    if betas.shape[1] == j - 1:
        full = np.column_stack([betas, -betas.sum(axis=1)])
    else:
        full = betas  # already J columns

    for fi, fname in enumerate(founders):
        color = FOUNDER_COLORS.get(fname, COLOR_CC_FALLBACK)
        ax.plot(positions, full[:, fi], color=color, linewidth=linewidth,
                alpha=alpha, label=fname, zorder=2)

    ax.axhline(0, color=COLOR_NEUTRAL_DARK, linewidth=0.5, zorder=1)
    ax.set_xlim(float(positions[0]), float(positions[-1]))
    if ylim is not None:
        ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(f'Chr {chrom} position (Mb)' if chrom else 'Position (Mb)')
    ax.set_ylabel('Founder effect')
    if show_legend:
        ax.legend(fontsize=6, ncol=4, frameon=False, loc='upper right',
                  handlelength=1.2)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def coef_profile_genome(
    scans: dict[str, tuple[np.ndarray, np.ndarray]],
    founders: list[str] = FOUNDER_ORDER,
    *,
    figsize: tuple[float, float] = (14, 3),
    linewidth: float = 0.8,
    alpha: float = 0.75,
) -> plt.Figure:
    """Genome-wide founder coefficient profiles across all chromosomes.

    Parameters
    ----------
    scans : dict[str, tuple[np.ndarray, np.ndarray]]
        ``{chrom: (positions_Mb, betas)}`` where betas has shape (n_loci, J-1).
    founders : list[str]
        Founder names in FOUNDER_ORDER.
    figsize : tuple[float, float]
        Figure dimensions in inches.
    linewidth : float
        Line width per founder.
    alpha : float
        Line opacity.

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
    legend_drawn        = False

    for chrom in chroms:
        pos, betas = scans[chrom]
        full = np.column_stack([betas, -betas.sum(axis=1)])
        for fi, fname in enumerate(founders):
            color = FOUNDER_COLORS.get(fname, COLOR_CC_FALLBACK)
            label = fname if not legend_drawn else None
            ax.plot(pos + offset, full[:, fi], color=color,
                    linewidth=linewidth, alpha=alpha, label=label, zorder=2)
        legend_drawn = True
        xticks.append(float(offset + (pos[-1] - pos[0]) / 2))
        xlabels.append(chrom)
        offset += float(pos[-1]) + _CHR_GAP_MB

    ax.axhline(0, color=COLOR_NEUTRAL_DARK, linewidth=0.4, zorder=1)
    ax.set_xlim(0, offset)
    ax.set_ylabel('Founder effect')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis='x', length=2)
    ax.legend(fontsize=6, ncol=4, frameon=False, loc='upper right',
              handlelength=1.2)
    sns.despine(ax=ax)
    _set_font(ax)
    return fig


def effect_bars(
    betas_locus: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    se: Optional[np.ndarray] = None,
) -> plt.Axes:
    """Per-founder effect bar chart at a single locus.

    The J-th (reference) founder is shown at zero. If betas_locus has
    shape (J-1,) the reference is recovered via sum-to-zero.

    Parameters
    ----------
    betas_locus : np.ndarray
        Shape (J-1,) or (J,) founder effects at the peak locus.
    founders : list[str]
        Founder names; length must match betas or betas+1.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    se : np.ndarray, optional
        Standard errors same shape as betas_locus; drawn as error bars.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=300)

    J = len(founders)
    if len(betas_locus) == J - 1:
        full_b  = _recover_full_betas(betas_locus)
        full_se = (np.append(se, np.nan) if se is not None else None)
    else:
        full_b  = np.asarray(betas_locus[:J])
        full_se = (np.asarray(se[:J]) if se is not None else None)

    colors = [FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in founders]
    x      = np.arange(J)
    ax.bar(x, full_b, color=colors,
           yerr=full_se, capsize=3,
           error_kw={'linewidth': 0.8, 'ecolor': COLOR_NEUTRAL_DARK})
    ax.axhline(0, color=COLOR_NEUTRAL_DARK, linewidth=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(founders, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Effect (β)')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def effect_caterpillar(
    betas_locus: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    ci_wide: Optional[np.ndarray] = None,
    ci_narrow: Optional[np.ndarray] = None,
    connect: bool = True,
) -> plt.Axes:
    """Caterpillar plot of founder effects at a single locus with confidence intervals.

    Horizontal point-and-CI layout: each founder is a row, effect on x-axis.
    Wide CI (95%) and narrow CI (75%) drawn as nested intervals when provided.

    Parameters
    ----------
    betas_locus : np.ndarray
        Shape (J-1,) or (J,) founder effects.
    founders : list[str]
        Founder names.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    ci_wide : np.ndarray, optional
        Shape (J, 2) [lower, upper] 95% CI per founder.
    ci_narrow : np.ndarray, optional
        Shape (J, 2) [lower, upper] 75% CI per founder.
    connect : bool
        Draw a connecting line through founder midpoints.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=300)

    J = len(founders)
    if len(betas_locus) == J - 1:
        full_b = _recover_full_betas(betas_locus)
    else:
        full_b = np.asarray(betas_locus[:J])

    y      = np.arange(J)
    colors = [FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in founders]

    if connect:
        ax.plot(full_b, y, color=COLOR_NEUTRAL_DARK, linewidth=0.8,
                zorder=1, alpha=0.5)
    if ci_wide is not None:
        for i in range(J):
            ax.plot(ci_wide[i], [y[i], y[i]], color=colors[i],
                    linewidth=2.0, alpha=0.3, solid_capstyle='round', zorder=2)
    if ci_narrow is not None:
        for i in range(J):
            ax.plot(ci_narrow[i], [y[i], y[i]], color=colors[i],
                    linewidth=3.5, alpha=0.6, solid_capstyle='round', zorder=3)

    ax.scatter(full_b, y, color=colors, s=40, zorder=4,
               edgecolors='white', linewidths=0.5)
    ax.axvline(0, color=COLOR_NEUTRAL_DARK, linewidth=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(founders, fontsize=8)
    ax.set_xlabel('Effect (β)')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def dispersion_distribution(
    phi: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    phi_prior: Optional[float] = None,
    log_scale: bool = True,
) -> plt.Axes:
    """NB dispersion (φ) distribution across genes — Stage 1 calibration QC.

    Parameters
    ----------
    phi : np.ndarray
        Per-gene EB-shrunk dispersion estimates, shape (n_genes,).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    phi_prior : float, optional
        Global prior mean — drawn as vertical dashed line.
    log_scale : bool
        Plot log₁₀(φ) on x-axis; recommended since φ spans orders of magnitude.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=300)

    phi_valid = phi[np.isfinite(phi) & (phi > 0)]
    x      = np.log10(phi_valid) if log_scale else phi_valid
    xlabel = 'log₁₀(φ)' if log_scale else 'φ'

    sns.histplot(x, bins=50, color=FOUNDER_COLORS['NODJ'],
                 alpha=0.7, ax=ax, linewidth=0)
    if phi_prior is not None:
        prior_x = np.log10(phi_prior) if log_scale else phi_prior
        ax.axvline(prior_x, color=FOUNDER_COLORS['PWKJ'],
                   linewidth=1.0, linestyle='--', label=f'prior = {phi_prior:.3f}')
        ax.legend(fontsize=8, frameon=False)

    ax.set_xlim(left=float(x.min()))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Genes')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax
