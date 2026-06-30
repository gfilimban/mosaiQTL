"""Founder effect visualizations: coefficient profiles, effect bars, and caterpillar plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..genotype.founders import (
    COLOR_NEUTRAL_DARK, COLOR_NEUTRAL_LIGHT, COLOR_CC_FALLBACK,
    FOUNDER_COLORS, FOUNDER_ORDER,
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
    ylabel: str = 'Founder effect (RINT SD)',
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
    ylabel : str
        Y-axis label. Default assumes RINT-transformed phenotype.

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
    ax.set_ylabel(ylabel)
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
    ylabel: str = 'Founder effect (RINT SD)',
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
    ax.set_ylabel(ylabel)
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
    ylabel: str = 'Founder effect (RINT SD)',
    gene: str = '',
    locus: str = '',
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
    ylabel : str
        Y-axis label. Default assumes RINT-transformed phenotype.
    gene : str
        Gene name shown in title.
    locus : str
        Locus identifier shown in title alongside gene.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=300)

    J = len(founders)
    if len(betas_locus) == J - 1:
        full_b  = _recover_full_betas(betas_locus)
        full_se = (np.append(se, np.nan) if se is not None else None)
    else:
        full_b  = np.asarray(betas_locus[:J])
        full_se = (np.asarray(se[:J]) if se is not None else None)

    colors = [FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in founders]
    x = np.arange(J)
    # compute range first so text placement is correct
    y_abs_max = float(np.nanmax(np.abs(full_b))) * 1.2 or 1.0
    ax.bar(x, full_b, color=colors,
           yerr=full_se, capsize=3,
           error_kw={'linewidth': 0.8, 'ecolor': COLOR_NEUTRAL_DARK})
    # ±0.5 SD scale lines to calibrate magnitude (slightly more visible)
    for ref_level in [0.5, 1.0]:
        if ref_level < y_abs_max:
            for sign in [1, -1]:
                ax.axhline(sign * ref_level, color=COLOR_NEUTRAL_DARK,
                           linewidth=0.4, linestyle=':', alpha=0.6, zorder=0)
    # annotate the founder with the largest absolute effect
    top_idx = int(np.nanargmax(np.abs(full_b)))
    top_val = float(full_b[top_idx])
    top_offset = (0, 8) if top_val >= 0 else (0, -8)
    top_va     = 'bottom' if top_val >= 0 else 'top'
    ax.annotate(
        f'{top_val:+.2f}',
        xy=(top_idx, top_val),
        xytext=top_offset, textcoords='offset points',
        fontsize=10, fontweight='bold', ha='center', va=top_va,
        color=colors[top_idx],
    )
    # annotate the opposite-direction extreme only if meaningfully large (> 0.1)
    if top_val >= 0:
        opp_val = float(np.nanmin(full_b))
        opp_idx = int(np.nanargmin(full_b))
        if opp_val < -0.1:
            ax.annotate(
                f'{opp_val:+.2f}',
                xy=(opp_idx, opp_val),
                xytext=(0, -8), textcoords='offset points',
                fontsize=10, fontweight='bold', ha='center', va='top',
                color=colors[opp_idx],
            )
    else:
        opp_val = float(np.nanmax(full_b))
        opp_idx = int(np.nanargmax(full_b))
        if opp_val > 0.1:
            ax.annotate(
                f'{opp_val:+.2f}',
                xy=(opp_idx, opp_val),
                xytext=(0, 8), textcoords='offset points',
                fontsize=10, fontweight='bold', ha='center', va='bottom',
                color=colors[opp_idx],
            )
    ax.set_xticks(x)
    ax.set_xticklabels(founders, rotation=40, ha='right', fontsize=11,
                       fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-y_abs_max, y_abs_max)
    title = gene if gene else 'Founder effects'
    if locus:
        title += f'  @{locus}'
    ax.set_title(title, fontsize=13)
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
    xlabel: str = 'Founder effect (RINT SD)',
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
    xlabel : str
        X-axis label. Default assumes RINT-transformed phenotype.

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
    ax.set_xlabel(xlabel)
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
    ax.set_ylabel('Number of genes')
    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def haplotype_sex_profile(
    phenotype: np.ndarray,
    hap_probs: np.ndarray,
    sex: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    phenotype_label: str = 'Phenotype',
    chrom: str = '',
    position_mb: Optional[float] = None,
    gxe_result: Optional[dict] = None,
) -> plt.Axes:
    """Per-founder mean ± SEM phenotype split by sex — descriptive haplotype × sex profile.

    Visualises the interaction pattern identified by ``qtl_gxe_lrt``.  Each founder
    group is assigned by argmax of haplotype probabilities; two lines (male/female)
    show how the haplotype effect magnitude or direction differs between sexes.
    When ``gxe_result`` is provided (the dict returned by ``qtl_gxe_lrt``), the
    LMM-corrected overall significance is annotated on the plot.

    Parameters
    ----------
    phenotype : np.ndarray
        Shape (n_strains,) trait values.
    hap_probs : np.ndarray
        Shape (n_strains, J) founder probabilities at one locus.
    sex : np.ndarray
        Shape (n_strains,) binary sex (0 = male, 1 = female).
    founders : list[str]
        Founder names; length must equal hap_probs.shape[1].
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    phenotype_label : str
        Y-axis label.
    chrom : str
        Chromosome label for title annotation.
    position_mb : float, optional
        Locus position in Mb for title annotation.
    gxe_result : dict, optional
        Return value of ``qtl_gxe_lrt`` — must contain keys ``lod_gxe_sex`` and
        ``p_gxe_sex``.  These are LMM-based (kinship-corrected) and annotated as
        a text box in the lower-right corner.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

    mat      = np.asarray(hap_probs, dtype=float)
    assigned = np.argmax(mat, axis=1)
    sex_arr  = (np.asarray(sex, dtype=float) >= 0.5).astype(int)

    sex_labels    = {0: 'Male', 1: 'Female'}
    sex_markers   = {0: 'o', 1: 's'}
    sex_linestyle = {0: '-', 1: '--'}

    for sex_val in (0, 1):
        means: list[float] = []
        sems:  list[float] = []
        x_pos: list[int]   = []

        for fi in range(len(founders)):
            mask = (assigned == fi) & (sex_arr == sex_val)
            if mask.sum() < 2:
                means.append(float('nan'))
                sems.append(0.0)
            else:
                vals = phenotype[mask].astype(float)
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            x_pos.append(fi)

        means_arr = np.array(means, dtype=float)
        sems_arr  = np.array(sems,  dtype=float)
        valid     = np.isfinite(means_arr)

        color = COLOR_NEUTRAL_DARK if sex_val == 0 else COLOR_CC_FALLBACK
        ax.errorbar(
            np.array(x_pos)[valid],
            means_arr[valid],
            yerr=sems_arr[valid],
            color=color,
            marker=sex_markers[sex_val],
            linestyle=sex_linestyle[sex_val],
            linewidth=1.0,
            markersize=5,
            capsize=3,
            label=sex_labels[sex_val],
            zorder=3,
        )

    for fi, fname in enumerate(founders):
        color = FOUNDER_COLORS.get(fname, COLOR_CC_FALLBACK)
        ax.axvspan(fi - 0.4, fi + 0.4, alpha=0.08, color=color, zorder=0)

    ax.set_xticks(np.arange(len(founders)))
    ax.set_xticklabels(founders, rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-0.6, len(founders) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(phenotype_label)
    ax.legend(fontsize=8, frameon=False, loc='upper right')

    if gxe_result is not None:
        lod = gxe_result.get('lod_gxe_sex', float('nan'))
        p   = gxe_result.get('p_gxe_sex',   float('nan'))
        p_str = f'{p:.2e}' if (not np.isnan(p) and p < 0.001) else f'{p:.3f}'
        ann = f'LOD$_{{gxe}}$ = {lod:.2f}\np$_{{gxe}}$ = {p_str} (LMM)'
        ax.text(0.98, 0.04, ann, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLOR_NEUTRAL_LIGHT, alpha=0.9))

    title_parts = []
    if chrom:
        title_parts.append(f'Chr {chrom}')
    if position_mb is not None:
        title_parts.append(f'{position_mb:.2f} Mb')
    if title_parts:
        ax.set_title(' · '.join(title_parts), fontsize=8, pad=4)

    sns.despine(ax=ax)
    _set_font(ax)
    return ax


# ---------------------------------------------------------------------------
# Per-chromosome and grid summaries
# ---------------------------------------------------------------------------

def _eff_chr_sort_key(c: str) -> tuple[int, int]:
    """Chromosome sort key matching _CHR_ORDER (1–19 then X)."""
    try:
        return (0, _CHR_ORDER.index(c))
    except ValueError:
        try:
            return (1, int(c))
        except ValueError:
            return (2, 0)


def effects_per_chr(
    eff_df: pd.DataFrame,
    founders: list[str],
    *,
    priority_genes: Optional[dict[str, str]] = None,
) -> dict[str, plt.Figure]:
    """Per-chromosome founder effect bar figures for the top gene on each chromosome.

    Parameters
    ----------
    eff_df : pd.DataFrame
        Columns: gene (str), chr (str), peak_mb, peak_lod, beta_A … beta_H.
    founders : list[str]
        Display-name founders matching the beta column order.
    priority_genes : dict[str, str], optional
        {chr: gene_name} — used instead of top-peak gene when present.

    Returns
    -------
    dict[str, plt.Figure]
        Keyed by chromosome string.
    """
    prio = priority_genes or {}
    beta_cols = [b for b in eff_df.columns if b.startswith('beta_')]
    chr_vals = eff_df['chr'].astype(str)
    ordered = [c for c in _CHR_ORDER if c in set(chr_vals)]
    extra = sorted(set(chr_vals) - set(_CHR_ORDER))
    chrs = ordered + extra

    out: dict[str, plt.Figure] = {}
    for c in chrs:
        eff_c = eff_df[chr_vals == c]
        if eff_c.empty:
            continue

        gene = prio.get(c)
        if gene and gene in eff_c['gene'].values:
            row = eff_c[eff_c['gene'] == gene].iloc[0]
        else:
            row = eff_c.loc[eff_c['peak_lod'].idxmax()]
            gene = str(row['gene'])

        betas = np.array([float(row[b]) for b in beta_cols])
        fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
        effect_bars(betas, founders, ax=ax)
        ax.set_title(
            f'Founder effects  {gene}  chr{c}  LOD={row["peak_lod"]:.1f}',
            fontsize=9,
        )
        sns.despine(ax=ax)
        fig.tight_layout()
        out[c] = fig

    return out


def genome_effects_grid(
    eff_df: pd.DataFrame,
    founders: list[str],
    *,
    n_cols: int = 4,
) -> plt.Figure:
    """Grid of founder effect bars, one panel per chromosome (top gene).

    Parameters
    ----------
    eff_df : pd.DataFrame
        Columns: gene (str), chr (str), peak_lod, beta_A … beta_H.
    founders : list[str]
        Display-name founders matching the beta column order.
    n_cols : int
        Panels per row (default 4 → 4×5 for 20 chromosomes).

    Returns
    -------
    plt.Figure
    """
    beta_cols = [b for b in eff_df.columns if b.startswith('beta_')]
    chr_vals = eff_df['chr'].astype(str)
    ordered = [c for c in _CHR_ORDER if c in set(chr_vals)]
    extra = sorted(set(chr_vals) - set(_CHR_ORDER))
    chrs = ordered + extra
    n_chrs = len(chrs)
    n_rows = (n_chrs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.2),
        dpi=300,
        squeeze=False,
    )

    for i, c in enumerate(chrs):
        ax = axes[i // n_cols, i % n_cols]
        eff_c = eff_df[chr_vals == c]
        if eff_c.empty:
            ax.set_visible(False)
            continue

        row = eff_c.loc[eff_c['peak_lod'].idxmax()]
        betas = np.array([float(row[b]) for b in beta_cols])
        gene = str(row['gene'])

        effect_bars(betas, founders, ax=ax)
        ax.set_xlabel('')
        ax.tick_params(labelsize=5)
        ax.set_xticklabels(founders, rotation=60, ha='right', fontsize=5)
        ax.set_title(f'chr{c}  {gene[:12]}', fontsize=7)
        sns.despine(ax=ax)

    for j in range(n_chrs, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle('Founder effects — top gene per chromosome', fontsize=9, y=1.01)
    fig.tight_layout()
    return fig
