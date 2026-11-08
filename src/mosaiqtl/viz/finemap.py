"""Fine-mapping visualization: regional association plots and credible-set summaries.

Standard field views following LocusZoom conventions:
  regional_plot          — per-gene position × LOD, PIP-colored gradient, CS members outlined
  locus_finemap_panel    — IGV-style 3-tier figure: LOD profile + PIP track + gene track
  haplotype_snp_comparison — founder effect bars + REF/ALT allele coding at the CS lead SNP
  egene_resolution       — lead PIP vs haplotype LOD across all eGenes
  cs_size_distribution   — credible-set size histogram

Each function returns its own standalone figure or axes — none are nested inside each other.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._theme import (
    BRAND_PRIMARY,
    BRAND_SECONDARY,
    INK,
    WASH,
    CMAP_MAGNITUDE,
    KW_REFERENCE,
    FONT_STACK,
    apply_theme,
    info_box,
    set_style,
)
from ..genotype.cc import FOUNDER_COLORS, FOUNDER_ORDER, COLOR_NEUTRAL_LIGHT

__all__ = [
    "regional_plot",
    "locus_finemap_panel",
    "haplotype_snp_comparison",
    "egene_resolution",
    "cs_size_distribution",
]


def regional_plot(
    fm_df: pd.DataFrame,
    gene_id: str,
    *,
    ax: Optional[plt.Axes] = None,
    window_mb: Optional[float] = None,
    highlight_lead: bool = True,
) -> plt.Axes:
    """LocusZoom-style regional fine-mapping plot for a single eGene.

    Each SNP is a circle at (position, LOD).  Fill color encodes PIP
    continuously via the ``CMAP_MAGNITUDE`` colormap so SNP association
    strength and fine-mapping resolution are readable in one glance.
    In-CS SNPs carry a black edge ring; the lead SNP (highest PIP) is
    drawn as a larger diamond on top.

    Parameters
    ----------
    fm_df : pd.DataFrame
        Full finemapping table (``eqtl_finemapping.csv``).
    gene_id : str
        Ensembl gene ID to plot.
    ax : plt.Axes, optional
        Axes to draw on; created at (10, 3.5) if not provided.
    window_mb : float, optional
        Restrict x-axis to ±window_mb around the lead SNP.  None shows
        all SNPs.
    highlight_lead : bool
        Whether to draw the lead SNP with a distinct diamond marker.

    Returns
    -------
    plt.Axes
    """
    set_style()
    gene = fm_df[fm_df["gene_id"] == gene_id].copy()
    if gene.empty:
        raise ValueError(f"gene_id {gene_id!r} not found in fm_df")

    gene_sym = str(gene["gene_symbol"].iloc[0])
    chrom = str(gene["chr"].iloc[0])

    pos_mb = gene["pos"].values / 1e6
    lod = gene["lod"].values
    pip = gene["pip"].values
    cs_mask = (gene["CS"] == "T").values
    lead_mask = (gene["lead_snp"] == "T").values

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

    if window_mb is not None:
        lead_pos = float(pos_mb[lead_mask].mean()) if lead_mask.any() else float(pos_mb.mean())
        view = (pos_mb >= lead_pos - window_mb) & (pos_mb <= lead_pos + window_mb)
        pos_mb, lod, pip, cs_mask, lead_mask = (
            pos_mb[view], lod[view], pip[view], cs_mask[view], lead_mask[view]
        )

    # out-of-CS background layer (muted, no edge)
    # ax.scatter accepted: colormapped scatter (c=array, cmap=), no seaborn equivalent
    out_mask = ~cs_mask
    if out_mask.any():
        ax.scatter(
            pos_mb[out_mask], lod[out_mask],
            c=pip[out_mask], cmap=CMAP_MAGNITUDE, vmin=0, vmax=1,
            alpha=0.35, s=14, linewidths=0, zorder=2,
        )

    # in-CS layer (black edge ring marks credible-set membership)
    cs_only = cs_mask & ~lead_mask
    if cs_only.any():
        ax.scatter(
            pos_mb[cs_only], lod[cs_only],
            c=pip[cs_only], cmap=CMAP_MAGNITUDE, vmin=0, vmax=1,
            alpha=0.90, s=36, linewidths=0.7, edgecolors=INK, zorder=4,
        )

    # lead SNP — diamond, larger, same PIP colormap
    if highlight_lead and lead_mask.any():
        sc = ax.scatter(
            pos_mb[lead_mask], lod[lead_mask],
            c=pip[lead_mask], cmap=CMAP_MAGNITUDE, vmin=0, vmax=1,
            alpha=1.0, s=90, linewidths=0.9, edgecolors=INK,
            marker="D", zorder=6,
        )
        plt.colorbar(sc, ax=ax, label="PIP", fraction=0.025, pad=0.01)

    # haplotype-peak reference line
    hap_peak_lod = gene["hap_peak_lod"].iloc[0] if "hap_peak_lod" in gene.columns else None
    if hap_peak_lod is not None and not pd.isna(hap_peak_lod):
        ax.axhline(float(hap_peak_lod), **KW_REFERENCE)
        ax.text(
            pos_mb.min(),
            float(hap_peak_lod) * 1.01,
            "hap peak",
            fontsize=7, color=INK, va="bottom",
        )

    n_cs = int(cs_mask.sum())
    lead_pip = float(pip[lead_mask].max()) if lead_mask.any() else float("nan")
    info_box(ax, f"CS n={n_cs}  lead PIP={lead_pip:.2f}", corner="tr", fontsize=7)

    ax.set_xlabel(f"chr{chrom} position (Mb)")
    ax.set_ylabel("LOD")
    ax.set_title(gene_sym, fontsize=10)
    apply_theme(ax, from_zero=True)
    return ax


def locus_finemap_panel(
    fm_df: pd.DataFrame,
    gene_id: str,
    lod_positions: np.ndarray,
    lod_scores: np.ndarray,
    *,
    chrom: str = "",
    peak_pos_mb: Optional[float] = None,
    genes_df: Optional[pd.DataFrame] = None,
    highlight_gene: Optional[str] = None,
    window_mb: float = 1.0,
    threshold: Optional[float] = None,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    """IGV-style 3-tier standalone figure for a single fine-mapped eGene.

    Panel stack (top → bottom):

    1. **LOD profile** — haplotype-scan LOD across the cis window, with the
       permutation threshold and peak position annotated.
    2. **PIP track** — per-SNP PIP as a scatter plot; in-CS SNPs drawn in
       ``BRAND_PRIMARY``, non-CS in ``WASH``, lead SNP as a larger diamond.
    3. **Gene track** — genome-browser-style gene body track (requires
       *genes_df*).

    All three panels share the x-axis (Mb within the cis window).  This
    figure is standalone — it is never embedded in a larger panel figure.

    Parameters
    ----------
    fm_df : pd.DataFrame
        Full finemapping table (``eqtl_finemapping.csv``).
    gene_id : str
        Ensembl gene ID.
    lod_positions : np.ndarray
        Haplotype scan locus positions in Mb for *chrom*.
    lod_scores : np.ndarray
        LOD scores matching *lod_positions*.
    chrom : str
        Chromosome label (e.g. ``'19'``).
    peak_pos_mb : float, optional
        Peak locus position in Mb; vertical line on LOD + PIP panels.
    genes_df : pd.DataFrame, optional
        Gene annotation table with columns ``start_mb``, ``end_mb``,
        ``gene_symbol`` (or ``gene_id``), and optional ``strand``.
        Required for the gene track tier.
    highlight_gene : str, optional
        Gene name highlighted in the gene track.
    window_mb : float
        Half-width of the cis window shown on all panels.  Default 1.0 Mb.
    threshold : float, optional
        Permutation LOD threshold; drawn as a dotted line on the LOD panel.
    figsize : tuple[float, float]
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
    """
    set_style()

    gene = fm_df[fm_df["gene_id"] == gene_id].copy()
    if gene.empty:
        raise ValueError(f"gene_id {gene_id!r} not found in fm_df")

    gene_sym = str(gene["gene_symbol"].iloc[0])
    has_track = genes_df is not None

    n_panels = 3 if has_track else 2
    height_ratios = ([2.5, 1.5, 1.0] if has_track else [2.5, 1.5])[:n_panels]

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.06)
    ax_lod = fig.add_subplot(gs[0])
    ax_pip = fig.add_subplot(gs[1], sharex=ax_lod)
    ax_gene = fig.add_subplot(gs[2], sharex=ax_lod) if has_track else None

    lead_mask = (gene["lead_snp"] == "T").values
    if peak_pos_mb is None:
        peak_pos_mb = float((gene["pos"].values[lead_mask] / 1e6).mean()) if lead_mask.any() \
            else float(gene["pos"].mean() / 1e6)
    xlim = (peak_pos_mb - window_mb, peak_pos_mb + window_mb)

    # ── tier 1: LOD profile ───────────────────────────────────────────────────
    win_lod = (lod_positions >= xlim[0]) & (lod_positions <= xlim[1])
    if win_lod.any():
        sns.lineplot(
            x=lod_positions[win_lod], y=lod_scores[win_lod],
            color=BRAND_PRIMARY, linewidth=1.4, errorbar=None, ax=ax_lod,
        )
    if threshold is not None:
        ax_lod.axhline(threshold, linestyle=":", linewidth=1.0, color=INK, alpha=0.6)
    ax_lod.axvline(peak_pos_mb, color=BRAND_SECONDARY, linewidth=1.0, zorder=4)
    ax_lod.set_ylabel("LOD")
    ax_lod.set_title(gene_sym, fontsize=10)
    apply_theme(ax_lod, from_zero=True)
    plt.setp(ax_lod.get_xticklabels(), visible=False)

    # ── tier 2: PIP scatter ───────────────────────────────────────────────────
    pos_mb = gene["pos"].values / 1e6
    pip = gene["pip"].values
    cs_mask = (gene["CS"] == "T").values

    win = (pos_mb >= xlim[0]) & (pos_mb <= xlim[1])
    pos_w, pip_w, cs_w, lead_w = pos_mb[win], pip[win], cs_mask[win], lead_mask[win]

    out_w = ~cs_w
    if out_w.any():
        sns.scatterplot(
            x=pos_w[out_w], y=pip_w[out_w],
            color=WASH, s=14, alpha=0.5, linewidths=0, ax=ax_pip, legend=False, zorder=2,
        )
    cs_only_w = cs_w & ~lead_w
    if cs_only_w.any():
        sns.scatterplot(
            x=pos_w[cs_only_w], y=pip_w[cs_only_w],
            color=BRAND_PRIMARY, s=28, alpha=0.85, linewidths=0.6, edgecolors=INK,
            ax=ax_pip, legend=False, zorder=4,
        )
    if lead_w.any():
        sns.scatterplot(
            x=pos_w[lead_w], y=pip_w[lead_w],
            color=BRAND_SECONDARY, s=70, marker="D", linewidths=0.8, edgecolors=INK,
            ax=ax_pip, legend=False, zorder=6,
        )
    ax_pip.axvline(peak_pos_mb, color=BRAND_SECONDARY, linewidth=1.0, zorder=3)
    ax_pip.set_ylim(0, 1.05)
    ax_pip.set_ylabel("PIP")
    apply_theme(ax_pip, from_zero=True)
    if has_track:
        plt.setp(ax_pip.get_xticklabels(), visible=False)
    else:
        ax_pip.set_xlabel(f"chr{chrom} position (Mb)")

    # ── tier 3: gene track ────────────────────────────────────────────────────
    if has_track and ax_gene is not None:
        from .locus import _draw_gene_track
        _draw_gene_track(
            ax_gene, genes_df, xlim,
            highlight_gene=highlight_gene or gene_sym,
            peak_pos=peak_pos_mb,
        )
        ax_gene.set_xlabel(f"chr{chrom} position (Mb)")

    ax_lod.set_xlim(*xlim)
    fig.align_ylabels([ax_lod, ax_pip] + ([ax_gene] if ax_gene else []))
    fig.tight_layout(h_pad=0.2)
    return fig


def haplotype_snp_comparison(
    fm_peak_df: pd.DataFrame,
    gene_id: str,
    founder_betas: np.ndarray,
    founders: list[str] = FOUNDER_ORDER,
    *,
    figsize: tuple[float, float] = (11, 4),
) -> plt.Figure:
    """Two-panel standalone figure: founder effect bars + REF/ALT allele membership.

    Left panel shows the 8-founder haplotype effects from the scan.  Right
    panel shows, for each founder, whether they carry the REF or ALT allele
    at the credible-set lead SNP, colored by their canonical founder color.
    This directly validates the haplotype → biallelic compression.

    This figure is standalone — it is never embedded in a larger panel figure.

    Parameters
    ----------
    fm_peak_df : pd.DataFrame
        One-row-per-eGene peak table (``eqtl_finemapping_peak.csv``).
    gene_id : str
        Ensembl gene ID.
    founder_betas : np.ndarray
        Per-founder effect estimates at the peak locus, shape ``(J,)``.
    founders : list[str]
        Founder display names matching *founder_betas*.
    figsize : tuple[float, float]
        Figure dimensions in inches.

    Returns
    -------
    plt.Figure
    """
    set_style()

    row = fm_peak_df[fm_peak_df["gene_id"] == gene_id]
    if row.empty:
        raise ValueError(f"gene_id {gene_id!r} not found in fm_peak_df")
    row = row.iloc[0]

    gene_sym = str(row.get("gene_symbol", gene_id))
    ref_set: set[str] = set(str(row.get("ref_founders", "")).split(",")) - {""}
    alt_set: set[str] = set(str(row.get("alt_founders", "")).split(",")) - {""}
    effect_size = row.get("effect_size", np.nan)

    fig, (ax_bars, ax_allele) = plt.subplots(1, 2, figsize=figsize, dpi=300)

    # ── left: founder effect bars ─────────────────────────────────────────────
    from .effects import effect_bars as _effect_bars
    _effect_bars(
        betas_locus=founder_betas,
        founders=founders,
        ax=ax_bars,
        gene=gene_sym,
        locus=str(row.get("variant", "")),
    )

    # ── right: allele membership per founder ──────────────────────────────────
    # ax.barh with per-bar color list — accepted matplotlib exception (viz/CLAUDE.md)
    J = len(founders)
    bar_colors = [FOUNDER_COLORS.get(f, COLOR_NEUTRAL_LIGHT) for f in founders]
    bar_widths = [
        1.0 if f in alt_set else (0.35 if f in ref_set else 0.15)
        for f in founders
    ]
    y_pos = np.arange(J)
    ax_allele.barh(y_pos, bar_widths, color=bar_colors, alpha=0.85, edgecolor=INK, linewidth=0.5)

    if not pd.isna(effect_size):
        ax_allele.axvline(0, **KW_REFERENCE)
        ax_allele.text(
            0.02, J - 0.3,
            f"effect={effect_size:+.3f}",
            fontsize=7, color=INK, va="top",
        )

    ax_allele.set_yticks(y_pos)
    ax_allele.set_yticklabels(founders, fontfamily=FONT_STACK, fontsize=9)
    ax_allele.set_xlabel("Allele weight (ALT=1.0, REF=0.35)")
    ax_allele.set_title("Allele coding at lead SNP", fontsize=10)

    handles = [
        mpatches.Patch(facecolor=BRAND_SECONDARY, edgecolor=INK, label="ALT carrier"),
        mpatches.Patch(facecolor=WASH, edgecolor=INK, label="REF carrier"),
    ]
    ax_allele.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    apply_theme(ax_allele, from_zero=True)

    fig.suptitle(gene_sym, fontsize=11)
    fig.tight_layout()
    return fig


def egene_resolution(
    fm_peak_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    min_pip: float = 0.0,
) -> plt.Axes:
    """Lead-SNP PIP vs haplotype peak LOD for all fine-mapped eGenes.

    Shows how well Sanger SNP imputation resolves a locus: genes in the
    top-right corner have both strong haplotype signal and high posterior
    SNP confidence; genes in the lower-left have haplotype signal but
    diffuse SNP uncertainty.

    Parameters
    ----------
    fm_peak_df : pd.DataFrame
        One-row-per-eGene peak table (``eqtl_finemapping_peak.csv``).
    ax : plt.Axes, optional
        Axes to draw on; created at (5, 4.5) if not provided.
    min_pip : float
        Only show eGenes with lead PIP ≥ this value.

    Returns
    -------
    plt.Axes
    """
    set_style()
    df = fm_peak_df.dropna(subset=["pip", "hap_peak_lod"]).copy()
    if min_pip > 0:
        df = df[df["pip"] >= min_pip]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5), dpi=300)

    # colormapped scatter: pip encodes color (accepted matplotlib-only exception)
    sc = ax.scatter(
        df["hap_peak_lod"].values,
        df["pip"].values,
        c=df["pip"].values,
        cmap=CMAP_MAGNITUDE, vmin=0, vmax=1,
        s=18, alpha=0.55, linewidths=0, zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="PIP", fraction=0.03, pad=0.01)

    n = len(df)
    high_res = int((df["pip"] >= 0.5).sum())
    info_box(
        ax,
        f"n={n}  PIP≥0.5: {high_res} ({100 * high_res / max(n, 1):.0f}%)",
        corner="tl",
        fontsize=7,
    )

    ax.set_xlabel("Haplotype peak LOD")
    ax.set_ylabel("Lead-SNP PIP")
    ax.set_ylim(0, 1.05)
    apply_theme(ax, from_zero=True)
    return ax


def cs_size_distribution(
    fm_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    max_cs_size: int = 50,
) -> plt.Axes:
    """Credible-set size histogram: number of SNPs in each CS.

    Parameters
    ----------
    fm_df : pd.DataFrame
        Full finemapping table (``eqtl_finemapping.csv``).
    ax : plt.Axes, optional
        Axes to draw on; created at (5, 3.5) if not provided.
    max_cs_size : int
        Bin cap; CS sizes above this are grouped into the final bin.

    Returns
    -------
    plt.Axes
    """
    set_style()
    cs_rows = fm_df[fm_df["CS"] == "T"]
    sizes = cs_rows.groupby("gene_id").size()
    sizes_clipped = sizes.clip(upper=max_cs_size)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

    bins = np.arange(1, max_cs_size + 2) - 0.5
    sns.histplot(sizes_clipped, bins=bins, color=BRAND_PRIMARY, alpha=0.85, ax=ax, kde=False)

    median_sz = float(np.median(sizes))
    ax.axvline(median_sz, **KW_REFERENCE)
    ax.text(
        median_sz + 0.5, ax.get_ylim()[1] * 0.90,
        f"median={median_sz:.0f}",
        fontsize=7, color=INK,
    )

    n_singleton = int((sizes == 1).sum())
    pct = 100 * n_singleton / max(len(sizes), 1)
    info_box(ax, f"1-SNP CS: {n_singleton} ({pct:.0f}%)", corner="tr", fontsize=7)

    ax.set_xlabel("Credible-set size (SNPs)")
    ax.set_ylabel("eGenes")
    apply_theme(ax, from_zero=True)
    return ax
