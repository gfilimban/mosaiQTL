"""locus — IGV-style multi-panel figures for a single QTL locus."""

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
    COLOR_UTR,
)
from .effects import effect_bars as _effect_bars
from .genome import _ordered_chrs, lod_profile

__all__ = [
    'locus_panel',
    'locus_panels_per_chr',
    'locus_panel_grid',
]


def locus_panels_per_chr(
    lod_df: pd.DataFrame,
    eff_df: Optional[pd.DataFrame] = None,
    founders: Optional[list[str]] = None,
    thr_by_chr: Optional[dict[str, Optional[float]]] = None,
    *,
    priority_genes: Optional[dict[str, str]] = None,
    gene_list: Optional[list[str]] = None,
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
        {chr: gene_symbol}.  Used only when *gene_list* is ``None``; overrides
        the automatic top-per-chromosome selection for specified chromosomes.
    gene_list : list[str], optional
        Explicit list of gene symbols to produce panels for.  When provided,
        one figure is generated per gene and the returned dict is keyed by
        gene_symbol.  When ``None`` (default), one figure is generated per
        chromosome showing the highest-LOD eGene (or the *priority_genes*
        override) and the dict is keyed by chromosome string.
    tss_mb_map : dict, optional
        {gene_symbol: tss_mb}. When provided, adds a zoomed LOD panel per figure.
    near_tss_mb_map : dict, optional
        {gene_symbol: near_tss_mb}. TSS of the nearest gene to each QTL peak;
        shown as a dashed green marker in the zoom panel and gene track.
    genes_df : pd.DataFrame, optional
        Gene annotation table (``start_mb``, ``end_mb``, ``gene_symbol`` or
        ``gene_id``, optional ``strand``). When provided with *tss_mb_map*,
        adds a genome-browser-style gene body track below the zoom panel.
    zoom_window_mb : float
        Margin around the [TSS, peak] interval in the zoom panel. Default 1.0 Mb.
    gtf_path : str, optional
        Path to a ``.gtf`` or ``.gtf.gz`` file.  When provided, real
        exon/CDS/UTR intervals are loaded per panel and passed to the gene
        track — one GTF read per chromosome panel.

    Returns
    -------
    dict[str, plt.Figure]
        Keyed by chromosome string when *gene_list* is ``None``, or by
        gene_symbol when *gene_list* is provided.
    """
    thr = thr_by_chr or {}
    chr_vals = lod_df["chr"].astype(str)

    def _build_one(gene: str, chrom: str) -> Optional[plt.Figure]:
        lod_g = lod_df[(chr_vals == chrom) & (lod_df["gene_symbol"] == gene)]
        lod_g = lod_g.sort_values("pos_mb")
        if lod_g.empty:
            return None

        peak_betas: Optional[np.ndarray] = None
        peak_pos: Optional[float] = None
        peak_locus_id: Optional[str] = None
        if eff_df is not None and founders is not None:
            eff_row_df = eff_df[
                (eff_df["chr"].astype(str) == chrom) & (eff_df["gene_symbol"] == gene)
            ]
            if not eff_row_df.empty:
                beta_cols = [b for b in eff_row_df.columns if b.startswith("beta_")]
                row0 = eff_row_df.iloc[0]
                peak_betas = np.array([float(row0[b]) for b in beta_cols])
                peak_pos = float(row0.get("peak_mb", lod_g["pos_mb"].iloc[-1]))
                if "peak_locus" in eff_row_df.columns:
                    peak_locus_id = str(row0["peak_locus"])

        return locus_panel(
            lod_g["pos_mb"].values,
            lod_g["lod"].values,
            founders or [],
            peak_betas=peak_betas,
            peak_pos=peak_pos,
            tss_mb=(tss_mb_map or {}).get(gene),
            near_tss_mb=(near_tss_mb_map or {}).get(gene),
            genes_df=genes_df,
            zoom_window_mb=zoom_window_mb,
            threshold=thr.get(chrom),
            chrom=chrom,
            highlight_gene=gene,
            gtf_path=gtf_path,
            peak_locus=peak_locus_id,
        )

    out: dict[str, plt.Figure] = {}

    if gene_list is not None:
        # ── user-specified gene list: one figure per gene, keyed by gene_symbol ──
        for gene in gene_list:
            rows = lod_df[lod_df["gene_symbol"] == gene]
            if rows.empty:
                continue
            chrom = str(rows["chr"].iloc[0])
            fig = _build_one(gene, chrom)
            if fig is not None:
                out[gene] = fig
    else:
        # ── automatic: one figure per chromosome, highest-LOD eGene ─────────────
        prio = priority_genes or {}
        for c in _ordered_chrs(chr_vals):
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
            fig = _build_one(top_gene, c)
            if fig is not None:
                out[c] = fig

    return out


def _gene_track(
    ax: plt.Axes,
    genes_df: pd.DataFrame,
    xlim: tuple[float, float],
    *,
    highlight_gene: Optional[str] = None,
    peak_pos: Optional[float] = None,
    peak_locus: Optional[str] = None,
    near_tss_mb: Optional[float] = None,
    exon_df: Optional[pd.DataFrame] = None,
) -> None:
    """Draw a genome-browser-style gene body track on *ax*.

    When *exon_df* is supplied (from :func:`~mosaiqtl.io.annotation_io.load_exons_from_gtf`),
    each gene is drawn with its real transcript/exon/CDS/UTR structure:

    * Thin intron line spanning the full transcript.
    * UTR blocks (``feature == 'UTR'`` or exon portions not covered by CDS)
      at medium height.
    * CDS blocks (coding exons) at full height.
    * Directional arrows (›/‹) marking the strand in every intron gap.
    * An arrowhead at the transcription start site (5′ end).

    When *exon_df* is ``None``, only a thin gene-body line with direction
    arrows is drawn (an honest representation when annotation data is
    unavailable, rather than guessing at exon structure).

    *peak_pos* and *near_tss_mb* are drawn as vertical lines so their
    position relative to surrounding genes is immediately readable.

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
    # single-gene mode — x-axis spans the gene body; blocks drawn large and legible
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

        # block heights, tall enough to read clearly
        _HUTR = 0.38  # UTR blocks (GTF feature 'UTR')
        _HCDS = 0.62  # coding exon blocks (GTF feature 'CDS'; legend label: 'exon')

        # feature colors: CDS=dark blue, UTR=sky blue, intron/arrow=neutral
        _CLR_CDS = BRAND_PRIMARY  # dark blue — coding exons
        _CLR_UTR = COLOR_UTR  # sky blue — UTRs: transcribed but not translated
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

            # 3. directional arrow markers in intron gaps — these are plot
            #    decoration, not data points, so a plain marker is used
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

        # 4. TSS arrowhead — size proportional to gene length
        _adx = _gspan * 0.05
        if _str == "+":
            ax.annotate(
                "",
                xy=(_tss + _adx, _y),
                xytext=(_tss, _y),
                arrowprops=dict(arrowstyle="->", color=_clr, lw=1.5, mutation_scale=14),
                zorder=6,
                annotation_clip=False,
            )
        else:
            ax.annotate(
                "",
                xy=(_tss - _adx, _y),
                xytext=(_tss, _y),
                arrowprops=dict(arrowstyle="->", color=_clr, lw=1.5, mutation_scale=14),
                zorder=6,
                annotation_clip=False,
            )

        # 5. gene label above gene body
        _gene_label_fs = 14 if _gn == highlight_gene else 9
        ax.text(
            (_glo + _ghi) / 2,
            _y + _HCDS / 2 + 0.05,
            _gn,
            fontsize=_gene_label_fs,
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
                _peak_label = peak_locus if peak_locus else "peak"
                ax.text(
                    peak_pos,
                    -_HCDS / 2 - 0.08,
                    _peak_label,
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
    # multi-gene mode — row-packed, compact heights
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
        ax.axvline(peak_pos, color=BRAND_SECONDARY, lw=1.3, ls="-", zorder=8, alpha=0.9)
        _mg_peak_label = peak_locus if peak_locus else "peak"
        ax.text(
            peak_pos,
            y_top,
            _mg_peak_label,
            fontsize=6,
            color=BRAND_SECONDARY,
            ha="center",
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
    show_zoom: bool = True,
) -> plt.Figure:
    """Locus view: founder effect bars + full LOD profile + zoom + gene track.

    Panel stack (top → bottom) depends on which optional arguments are supplied:
    - *peak_betas* → founder effect bar panel at the top.
    - always → full-chromosome LOD profile.
    - *tss_mb* → zoomed cis-window LOD panel with TSS and peak annotated.
    - *genes_df* (requires *tss_mb*) → genome-browser-style gene body track
      sharing the zoom panel's x-axis; shows *peak_pos* (red) and
      *near_tss_mb* (green dashed) as vertical markers so the QTL locus is
      immediately readable relative to surrounding genes.

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
        provided together with *tss_mb*, adds a genome-browser-style gene body
        track below the zoom panel.
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
    has_zoom = tss_mb is not None and show_zoom
    # gene track needs tss_mb for the cis-scale x-axis; zoom panel is optional
    has_track = genes_df is not None and tss_mb is not None

    # ── pre-compute zoom bounds (zoom panel only) ─────────────────────────────
    zoom_lo: Optional[float] = None
    zoom_hi: Optional[float] = None
    if has_zoom:
        anchor = peak_pos if peak_pos is not None else tss_mb
        zoom_lo = min(tss_mb, anchor) - zoom_window_mb  # type: ignore[arg-type]
        zoom_hi = max(tss_mb, anchor) + zoom_window_mb  # type: ignore[arg-type]

    # ── gene-track bounds (independent of zoom panel) ─────────────────────────
    # When the zoom panel is shown, the gene track shares its x-axis (same
    # window).  When show_zoom=False (e.g. Lw scans where the scan IS the cis
    # window), the gene track spans the actual scan positions instead — it does
    # not need or use the zoom-panel bounds.
    track_lo: Optional[float] = None
    track_hi: Optional[float] = None
    if has_track:
        if has_zoom and zoom_lo is not None and zoom_hi is not None:
            track_lo, track_hi = zoom_lo, zoom_hi
        else:
            # scan positions define the relevant region
            track_lo = float(positions.min())
            track_hi = float(positions.max())

    # ── load exon/CDS/UTR annotation (needed before panel layout for height) ──
    _exon_df: Optional[pd.DataFrame] = None
    if (
        has_track
        and track_lo is not None
        and track_hi is not None
        and gtf_path
        and chrom
    ):
        try:
            from ..io.annotation_io import load_exons_from_gtf

            _exon_df = load_exons_from_gtf(gtf_path, chrom, track_lo, track_hi)
        except Exception as _exc:  # noqa: BLE001
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "locus_panel: GTF load failed (%s); falling back to gene-body bars",
                _exc,
            )

    # ── build panel layout: gene track | zoom | full chr | founder effects ──
    ratios: list[float] = (
        ([0.8] if has_track else [])
        + ([1.0] if has_zoom else [])
        + [1.0]
        + ([2.0] if has_betas else [])
    )
    fig_h = 2.0 * sum(ratios)

    fig, axes = plt.subplots(
        len(ratios),
        1,
        figsize=(16, fig_h),
        dpi=300,
        gridspec_kw={"height_ratios": ratios},
        squeeze=False,
    )
    axes_flat = axes.ravel()
    ax_idx = 0

    title = highlight_gene or ""
    if chrom:
        title = f"{title}  chr{chrom}" if title else f"chr{chrom}"

    # ── panel 1: gene body track (top) ───────────────────────────────────────
    if has_track and track_lo is not None and track_hi is not None:
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
            (track_lo, track_hi),
            highlight_gene=highlight_gene,
            peak_pos=peak_pos,
            peak_locus=peak_locus,
            near_tss_mb=near_tss_mb,
            exon_df=_exon_df,
        )
        ax_track.set_xlabel("")
        # capture gene-body xlim so the zoom panel can shade it as a reference span
        _gene_body_xlim: tuple[float, float] = ax_track.get_xlim()
    else:
        _gene_body_xlim = (zoom_lo, zoom_hi) if zoom_lo is not None else (0.0, 0.0)

    # ── panel 2: zoomed cis-window LOD — ±window_mb around peak ─────────────
    # x-axis = zoom_lo..zoom_hi so the LOD bump is visible.
    # gene body extent is shaded in blue so it's spatially anchored to the
    # gene track above. peak and TSS lines carry through from the full-chr
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
        annotate_peak=peak_pos is not None,
        locus_id=peak_locus or "",
    )
    ax_full.set_title("")

    # shade the zoom window only when a zoom panel exists (has_zoom)
    if has_zoom and zoom_lo is not None and zoom_hi is not None:
        ax_full.axvspan(zoom_lo, zoom_hi, alpha=0.10, color=INK, zorder=0)

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

    # ── panel 4: founder effect bars ─────────────────────────────────────────
    if has_betas:
        ax_eff = axes_flat[ax_idx]
        ax_idx += 1
        _effect_bars(peak_betas, founders, ax=ax_eff)
        ax_eff.set_xlabel("")
        sns.despine(ax=ax_eff)

    fig.tight_layout()
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
    """Grid of locus panels (founder effects + LOD), one per chromosome.

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
