"""Haplotype probability heatmaps, phenotype-by-genotype plots, and kinship visualizations."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..genotype.founders import (
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_CC_FALLBACK,
    FOUNDER_COLORS,
    FOUNDER_ORDER,
)


def _set_font(ax: plt.Axes) -> None:
    """Apply Carlito/Calibri to all text elements of ax."""
    for item in (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontfamily(["Carlito", "Calibri", "DejaVu Sans"])


def prob_heatmap(
    hap_probs: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    phenotype: Optional[np.ndarray] = None,
    phenotype_id: str = "phenotype",
    gene: str = "",
    sample_labels: Optional[list[str]] = None,
    chrom: str = "",
    position_mb: Optional[float] = None,
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Haplotype dosage heatmap at a single locus, optionally sorted by phenotype.

    Founders are on the y-axis; samples (sorted ascending by phenotype) are on
    the x-axis.  A secondary x-axis on top shows phenotype values at five
    quantile positions, matching the miQTL ``prob.heatmap`` layout.

    Parameters
    ----------
    hap_probs : np.ndarray
        Shape (n_strains, J) founder probabilities at one locus.
    founders : list[str]
        Founder names; length must equal hap_probs.shape[1].
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    phenotype : np.ndarray, optional
        Shape (n_strains,) trait values; strains sorted ascending when provided.
    phenotype_id : str
        Phenotype name used in the x-axis label.
    gene : str
        eGene name; when provided, becomes the primary title element.
    sample_labels : list[str], optional
        Unused in current layout (kept for API compatibility).
    chrom : str
        Chromosome label for title.
    position_mb : float, optional
        Locus position in Mb for title annotation.
    cmap : str
        Colormap name for probability intensity.

    Returns
    -------
    plt.Axes
    """
    n_strains_in = hap_probs.shape[0]
    if ax is None:
        width = max(5.0, n_strains_in * 0.18)
        _, ax = plt.subplots(figsize=(width, 3.5), dpi=300)

    mat = np.asarray(hap_probs, dtype=float)  # (n_strains, J)
    pheno_sorted: Optional[np.ndarray] = None
    if phenotype is not None:
        order = np.argsort(phenotype)
        mat = mat[order]
        pheno_sorted = phenotype[order]

    # founders on y-axis (rows), strains on x-axis (columns) — matches miQTL layout
    mat = mat.T  # (J, n_strains)
    J = mat.shape[0]
    n_strains = mat.shape[1]

    im = ax.imshow(
        mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Haplotype Dose")

    # y-axis: founder names
    ax.set_yticks(np.arange(J))
    ax.set_yticklabels(founders, fontsize=7)

    # x-axis bottom: samples label; no individual ticks
    ax.set_xticks([])
    ax.set_xlabel("Strains (sorted by haplotype probability)", fontsize=8)

    # x-axis top: trait values at 5 quantile-spaced sample positions
    # — makes clear the numbers are trait values, not genomic coordinates
    if pheno_sorted is not None and n_strains >= 5:
        ax_top = ax.twiny()
        tick_cols = np.linspace(0, n_strains - 1, 5).astype(int)
        ax_top.set_xlim(-0.5, n_strains - 0.5)
        ax_top.set_xticks(tick_cols)
        ax_top.set_xticklabels(
            [f"{pheno_sorted[i]:.2f}" for i in tick_cols], fontsize=7
        )
        ax_top.tick_params(top=True, length=3)
        ax_top.set_xlabel(f"{phenotype_id} (trait value)", fontsize=7, labelpad=4)
        _set_font(ax_top)

    # title: gene name prominent; locus position secondary
    loc_parts = []
    if chrom:
        loc_parts.append(f"Chr {chrom}")
    if position_mb is not None:
        loc_parts.append(f"{position_mb:.2f} Mb")
    loc_str = " · ".join(loc_parts)
    if gene:
        title = f"{gene}  —  {loc_str}" if loc_str else gene
    else:
        title = loc_str
    if title:
        ax.set_title(title, fontsize=9, pad=4)

    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def pxg(
    phenotype: np.ndarray,
    hap_probs: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    phenotype_label: str = "Phenotype",
    chrom: str = "",
    position_mb: Optional[float] = None,
    jitter: float = 0.15,
    point_size: float = 18.0,
    alpha: float = 0.7,
) -> plt.Axes:
    """Phenotype-by-genotype plot: each sample assigned to its most-probable founder.

    Samples are grouped by the founder with highest haplotype probability. Points
    are jittered within each founder group and colored by founder.

    Parameters
    ----------
    phenotype : np.ndarray
        Shape (n_strains,) trait values.
    hap_probs : np.ndarray
        Shape (n_strains, J) founder probabilities at one locus.
    founders : list[str]
        Founder names; length must equal hap_probs.shape[1].
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    phenotype_label : str
        Y-axis label.
    chrom : str
        Chromosome label for title.
    position_mb : float, optional
        Locus position in Mb for title annotation.
    jitter : float
        Horizontal jitter spread (in x-unit fractions).
    point_size : float
        Scatter point size.
    alpha : float
        Point opacity.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

    mat = np.asarray(hap_probs, dtype=float)
    assigned = np.argmax(mat, axis=1)
    rng = np.random.default_rng(0)

    for fi, fname in enumerate(founders):
        mask = assigned == fi
        if not mask.any():
            continue
        color = FOUNDER_COLORS.get(fname, COLOR_CC_FALLBACK)
        x_vals = fi + rng.uniform(-jitter, jitter, size=mask.sum())
        sns.scatterplot(
            x=x_vals,
            y=phenotype[mask],
            color=color,
            s=point_size,
            alpha=alpha,
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
            label=fname,
            ax=ax,
        )

    ax.set_xticks(np.arange(len(founders)))
    ax.set_xticklabels(founders, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(phenotype_label)
    ax.set_xlim(-0.6, len(founders) - 0.4)

    title_parts = []
    if chrom:
        title_parts.append(f"Chr {chrom}")
    if position_mb is not None:
        title_parts.append(f"{position_mb:.2f} Mb")
    if title_parts:
        ax.set_title(" · ".join(title_parts), fontsize=8, pad=4)

    sns.despine(ax=ax)
    _set_font(ax)
    return ax


def pxg_sex_facet(
    phenotype: np.ndarray,
    hap_probs: np.ndarray,
    sex: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    phenotype_label: str = "Phenotype",
    chrom: str = "",
    position_mb: Optional[float] = None,
    jitter: float = 0.15,
    point_size: float = 18.0,
    alpha: float = 0.7,
) -> plt.Figure:
    """Sex-faceted PxG plot: two side-by-side panels (male left, female right).

    Each panel calls the same layout as ``pxg`` but restricted to one sex.
    Useful for visualising sex-differential haplotype effects at a peak locus.

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
    phenotype_label : str
        Y-axis label (left panel only).
    chrom : str
        Chromosome label for panel titles.
    position_mb : float, optional
        Locus position in Mb for panel titles.
    jitter : float
        Horizontal jitter spread.
    point_size : float
        Scatter point size.
    alpha : float
        Point opacity.

    Returns
    -------
    plt.Figure
        Figure with two Axes (male left, female right).
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.5), dpi=300, sharey=True, constrained_layout=True
    )

    sex_arr = np.asarray(sex, dtype=float)
    sex_masks = {0: sex_arr < 0.5, 1: sex_arr >= 0.5}
    sex_titles = {0: "Male", 1: "Female"}

    loc_parts = []
    if chrom:
        loc_parts.append(f"Chr {chrom}")
    if position_mb is not None:
        loc_parts.append(f"{position_mb:.2f} Mb")
    loc_str = " · ".join(loc_parts)

    for sex_val, ax in zip((0, 1), axes):
        mask = sex_masks[sex_val]
        ph_sub = phenotype[mask]
        hp_sub = np.asarray(hap_probs, dtype=float)[mask]

        pxg(
            ph_sub,
            hp_sub,
            founders,
            ax=ax,
            phenotype_label=phenotype_label if sex_val == 0 else "",
            jitter=jitter,
            point_size=point_size,
            alpha=alpha,
        )

        panel_title = sex_titles[sex_val]
        if loc_str:
            panel_title = f"{sex_titles[sex_val]}  —  {loc_str}"
        ax.set_title(panel_title, fontsize=8, pad=4)

    return fig


def kinship_heatmap(
    kinship: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    sample_labels: Optional[list[str]] = None,
    block_boundaries: Optional[list[int]] = None,
    block_labels: Optional[list[str]] = None,
    chrom: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Axes:
    """Kinship matrix heatmap with optional founder/F1/CC block annotations.

    Visualises the realized kinship matrix. Block boundaries allow annotation of
    strain-group structure (founders, F1s, CC strains) as vertical/horizontal lines.

    Parameters
    ----------
    kinship : np.ndarray
        Shape (n_strains, n_strains) symmetric kinship matrix.
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    sample_labels : list[str], optional
        Tick labels; suppressed when n_strains > 40.
    block_boundaries : list[int], optional
        Sample indices where group blocks begin (drawn as dividing lines).
    block_labels : list[str], optional
        Labels for each block; placed at block midpoints on both axes.
    chrom : str
        Chromosome label (e.g. 'LOCO chr 8') for title.
    cmap : str
        Colormap; 'RdBu_r' gives blue (low) → white → red (high) symmetry.
    vmin : float, optional
        Colormap minimum; defaults to matrix min.
    vmax : float, optional
        Colormap maximum; defaults to matrix max.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        n = kinship.shape[0]
        side = max(4.0, n * 0.06)
        _, ax = plt.subplots(figsize=(side, side), dpi=300)

    K = np.asarray(kinship, dtype=float)
    n = K.shape[0]
    v_abs = float(np.nanmax(np.abs(K)))
    v_lo = vmin if vmin is not None else -v_abs
    v_hi = vmax if vmax is not None else v_abs

    im = ax.imshow(
        K, aspect="equal", cmap=cmap, vmin=v_lo, vmax=v_hi, interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Kinship")

    if block_boundaries is not None:
        for b in block_boundaries:
            ax.axhline(b - 0.5, color=COLOR_NEUTRAL_DARK, linewidth=0.6, zorder=4)
            ax.axvline(b - 0.5, color=COLOR_NEUTRAL_DARK, linewidth=0.6, zorder=4)

    if block_boundaries is not None and block_labels is not None:
        bounds = list(block_boundaries) + [n]
        starts = [0] + list(block_boundaries)
        for label, start, end in zip(block_labels, starts, bounds):
            mid = (start + end) / 2.0 - 0.5
            ax.text(
                -0.5,
                mid,
                label,
                ha="right",
                va="center",
                fontsize=6,
                color=COLOR_NEUTRAL_DARK,
                fontfamily=["Carlito", "Calibri", "DejaVu Sans"],
            )

    if sample_labels is not None and n <= 40:
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(sample_labels, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(sample_labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if chrom:
        ax.set_title(f"Kinship — LOCO chr {chrom}", fontsize=8, pad=4)

    sns.despine(ax=ax, left=True, bottom=True)
    _set_font(ax)
    return ax


def epistasis_heatmap(
    interaction_lod: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    ax: Optional[plt.Axes] = None,
    vmax: Optional[float] = None,
    cmap: str = "Reds",
) -> plt.Axes:
    """Founder × founder epistasis interaction LOD heatmap (J × J grid).

    Displays pairwise founder interaction LOD scores at a given locus pair.
    The diagonal is suppressed (additive self-interaction is undefined).

    Parameters
    ----------
    interaction_lod : np.ndarray
        Shape (J, J) pairwise interaction LOD scores.
    founders : list[str]
        Founder names; length must equal interaction_lod.shape[0].
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    vmax : float, optional
        Colormap ceiling; defaults to matrix max.
    cmap : str
        Colormap; 'Reds' recommended (low = white, high = red).

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)

    J = len(founders)
    mat = np.asarray(interaction_lod, dtype=float).copy()
    np.fill_diagonal(mat, np.nan)

    v_hi = vmax if vmax is not None else float(np.nanmax(mat))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=COLOR_NEUTRAL_LIGHT)

    im = ax.imshow(
        mat, aspect="equal", cmap=cmap_obj, vmin=0.0, vmax=v_hi, interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Interaction LOD")

    ax.set_xticks(np.arange(J))
    ax.set_xticklabels(founders, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(J))
    ax.set_yticklabels(founders, fontsize=7)
    ax.set_xlabel("Founder (locus 2)")
    ax.set_ylabel("Founder (locus 1)")

    sns.despine(ax=ax, left=True, bottom=True)
    _set_font(ax)
    return ax


def kinship_distribution(
    K: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    chrom: str = "",
) -> plt.Axes:
    """KDE of diagonal vs off-diagonal kinship values.

    Parameters
    ----------
    K : np.ndarray
        Square kinship matrix, shape (n, n).
    ax : plt.Axes, optional
        Axes to draw on; created at dpi=300 if None.
    chrom : str
        Chromosome label for title.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), dpi=300)

    n = K.shape[0]
    diag = np.diag(K)
    tril = K[np.tril_indices(n, -1)]

    sns.kdeplot(diag, ax=ax, label="diagonal", color=FOUNDER_COLORS["NODJ"])
    sns.kdeplot(tril, ax=ax, label="off-diagonal", color=FOUNDER_COLORS["129S1J"])

    ax.set_xlabel("Kinship value")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    title = "Kinship distribution"
    if chrom:
        title += f"  chr{chrom}"
    ax.set_title(title, fontsize=9)
    sns.despine(ax=ax)
    _set_font(ax)
    return ax
