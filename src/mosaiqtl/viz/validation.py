"""
validation — phase 2 r vs python qc plots (s1–s11)

all plots use seaborn over matplotlib where possible.
font: carlito. axes from zero. dpi=300.
never import this module from installed package — tests/ only.
"""

import numpy as np
import os
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless use
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from ..genotype.founders import (
    FOUNDER_COLORS,
    F1_COLORS,
    COLOR_NEUTRAL_DARK,
    COLOR_CC_FALLBACK,
)

logger = logging.getLogger("mosaiqtl")

# semantic aliases for validation scatter convention (R=red x-axis, Python=blue y-axis)
_COLOR_PY = FOUNDER_COLORS["NODJ"]  # python points — blue
_COLOR_R = FOUNDER_COLORS["PWKJ"]  # r reference line — red
_COLOR_TRUE = FOUNDER_COLORS["CASTJ"]  # true causal locus marker — green
_COLOR_DISP = FOUNDER_COLORS["WSBJ"]  # dispersion scatter — purple
_CHR_PALETTE = [FOUNDER_COLORS["NODJ"], FOUNDER_COLORS["B6J"]]  # alternating manhattan

ALL_STRAIN_COLORS = {**FOUNDER_COLORS, **F1_COLORS}


def set_style():
    """set consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Carlito", "DejaVu Sans", "Arial"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        }
    )


def plot_lod_profile(scan_result, true_params=None, save_path=None):
    """genome-wide lod profile (manhattan-style).

    parameters
    ----------
    scan_result : ScanResult
        output of scan.scan_h2lmm()
    true_params : dict or None
        if provided, marks the true causal locus
    save_path : str or None
        save to file if provided
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    df = scan_result.to_dataframe()
    chromosomes = sorted(
        df["chromosome"].unique(), key=lambda x: int(x) if x.isdigit() else 99
    )

    # assign cumulative x positions per chromosome
    chr_offsets = {}
    offset = 0
    for chrom in chromosomes:
        chr_offsets[chrom] = offset
        chr_mask = df["chromosome"] == chrom
        if chr_mask.sum() > 0:
            offset += df.loc[chr_mask, "position"].max() + 10  # gap between chr

    x_positions = np.array(
        [chr_offsets[row["chromosome"]] + row["position"] for _, row in df.iterrows()]
    )

    # alternate chr colors
    chr_color_map = {}
    palette = _CHR_PALETTE
    for i, chrom in enumerate(chromosomes):
        chr_color_map[chrom] = palette[i % 2]

    colors = [chr_color_map[c] for c in df["chromosome"]]

    # scatter lod scores
    ax.scatter(
        x_positions,
        df["lod"],
        c=colors,
        s=12,
        alpha=0.7,
        edgecolors="none",
        rasterized=True,
    )

    # mark peak
    peak = scan_result.peak_locus()
    peak_x = chr_offsets[peak["chromosome"]] + peak["position"]
    ax.axhline(y=peak["lod"], color=_COLOR_R, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.annotate(
        f'peak: chr{peak["chromosome"]}\nLOD={peak["lod"]:.1f}',
        xy=(peak_x, peak["lod"]),
        xytext=(peak_x, peak["lod"] + 1),
        fontsize=9,
        color=_COLOR_R,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=_COLOR_R, lw=0.8),
    )

    # mark true locus if provided
    if true_params is not None:
        true_chrom = true_params.get("causal_chrom", None)
        true_idx = true_params.get("causal_locus_idx", None)
        if true_chrom and true_idx is not None:
            true_mask = df["chromosome"] == true_chrom
            if true_mask.sum() > true_idx:
                true_pos = df.loc[true_mask, "position"].iloc[true_idx]
                true_x = chr_offsets[true_chrom] + true_pos
                ax.axvline(
                    x=true_x,
                    color=_COLOR_TRUE,
                    linewidth=1,
                    linestyle=":",
                    alpha=0.7,
                    label="true causal locus",
                )
                ax.legend(fontsize=9)

    # chr labels
    for chrom in chromosomes:
        chr_mask = df["chromosome"] == chrom
        if chr_mask.sum() > 0:
            mid_x = chr_offsets[chrom] + df.loc[chr_mask, "position"].median()
            ax.text(
                mid_x,
                -0.5,
                f"chr{chrom}",
                ha="center",
                fontsize=8,
                color=COLOR_NEUTRAL_DARK,
            )

    ax.set_ylim(bottom=0)
    ax.set_ylabel("LOD score")
    ax.set_xlabel("")
    ax.set_title("genome-wide LOD profile")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    sns.despine(ax=ax, bottom=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_kinship_heatmap(k_matrix, sample_ids=None, save_path=None):
    """kinship matrix heatmap.

    parameters
    ----------
    k_matrix : np.ndarray (n, n)
    sample_ids : list of str or None
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        k_matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "kinship", "shrink": 0.7},
    )

    # annotate structure blocks if sample_ids available
    if sample_ids is not None:
        # count founders, f1s, cc strains
        n_founders = sum(1 for s in sample_ids if s in FOUNDER_COLORS)
        n_f1 = sum(1 for s in sample_ids if s in F1_COLORS)

        if n_founders > 0:
            ax.axhline(y=n_founders, color="white", linewidth=1.5)
            ax.axvline(x=n_founders, color="white", linewidth=1.5)
        if n_founders + n_f1 > 0 and n_f1 > 0:
            ax.axhline(y=n_founders + n_f1, color="white", linewidth=1)
            ax.axvline(x=n_founders + n_f1, color="white", linewidth=1)

    ax.set_title("LOCO kinship matrix")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_founder_effects(
    scan_result,
    genomecache,
    peak_chrom=None,
    peak_idx=None,
    founder_names=None,
    save_path=None,
):
    """founder allele effects at the peak locus.

    parameters
    ----------
    scan_result : ScanResult
    genomecache : GenomeCache
    peak_chrom : str or None
        if none, uses the genome-wide peak
    peak_idx : int or None
    founder_names : list of str or None
    save_path : str or None
    """
    set_style()

    if peak_chrom is None:
        peak = scan_result.peak_locus()
        # find the index within that chromosome
        chr_mask = scan_result.chromosomes == peak["chromosome"]
        chr_indices = np.where(chr_mask)[0]
        chr_lods = scan_result.lod[chr_mask]
        peak_idx = chr_indices[np.argmax(chr_lods)]
        peak_chrom = peak["chromosome"]

    betas = scan_result.betas[peak_idx]
    j_minus_1 = len(betas)

    if founder_names is None:
        founder_names = genomecache.founders[:j_minus_1]

    # add the dropped founder (effect = 0, reference)
    all_names = list(founder_names) + [genomecache.founders[-1]]
    all_effects = np.append(betas, 0.0)

    # color by founder palette
    colors = [FOUNDER_COLORS.get(name, COLOR_CC_FALLBACK) for name in all_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        range(len(all_names)),
        all_effects,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axhline(y=0, color=COLOR_NEUTRAL_DARK, linewidth=0.5)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("founder effect ($\\hat{\\beta}_f$)")
    ax.set_title(f"founder haplotype effects at peak (chr{peak_chrom})")
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_pvalue_qq(scan_result, save_path=None):
    """qq plot of p-values (uniform null expectation).

    parameters
    ----------
    scan_result : ScanResult
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    pvals = scan_result.p_value[scan_result.p_value > 0]
    pvals = np.sort(pvals)
    n = len(pvals)

    observed = -np.log10(pvals)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    sns.scatterplot(
        x=expected,
        y=observed,
        s=10,
        alpha=0.6,
        color=_COLOR_PY,
        edgecolors="none",
        rasterized=True,
        ax=ax,
    )
    sns.lineplot(
        x=[0, max(expected)],
        y=[0, max(expected)],
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        label="$y = x$ (null)",
        errorbar=None,
        ax=ax,
    )

    # genomic inflation factor
    lambda_gc = np.median(pvals) / 0.5
    ax.text(
        0.05,
        0.95,
        f"$\\lambda_{{GC}}$ = {1/lambda_gc:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("expected $-\\log_{10}(p)$")
    ax.set_ylabel("observed $-\\log_{10}(p)$")
    ax.set_title("QQ plot")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_dispersion_vs_truth(phi_estimated, phi_true, save_path=None):
    """scatter plot of estimated vs true per-subject dispersion.

    parameters
    ----------
    phi_estimated : np.ndarray
    phi_true : np.ndarray
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    valid = ~np.isnan(phi_estimated) & ~np.isnan(phi_true)
    phi_e = phi_estimated[valid]
    phi_t = phi_true[valid]

    sns.scatterplot(
        x=phi_t,
        y=phi_e,
        s=20,
        alpha=0.6,
        color=_COLOR_DISP,
        edgecolors="white",
        linewidth=0.3,
        ax=ax,
    )

    # diagonal
    lims = [0, max(phi_t.max(), phi_e.max()) * 1.1]
    sns.lineplot(
        x=lims,
        y=lims,
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        label="$y = x$",
        errorbar=None,
        ax=ax,
    )

    # correlation
    corr = np.corrcoef(phi_t, phi_e)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"$r$ = {corr:.3f}\n$n$ = {valid.sum()}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("true $\\hat{\\phi}_i$")
    ax.set_ylabel("estimated $\\hat{\\phi}_i$")
    ax.set_title("per-subject dispersion estimation")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


# ----------------------------------------------------------------
# cross-language comparison (python vs r reference)
# ----------------------------------------------------------------


def plot_lod_correlation(py_scan_result, r_results_csv, save_path=None):
    """scatter of mosaiQTL LOD vs R reference LOD at every shared locus.

    parameters
    ----------
    py_scan_result : ScanResult
        mosaiqtl scan output
    r_results_csv : str
        path to r validation scan_results.csv
    save_path : str or None
    """
    import pandas as pd

    set_style()

    py_df = py_scan_result.to_dataframe()
    r_df = pd.read_csv(r_results_csv)

    # merge on locus_id
    merged = py_df.merge(
        r_df, left_on="locus_id", right_on="locus", suffixes=("_py", "_r")
    )

    if len(merged) == 0:
        logger.warning("no shared loci between python and r results")
        return None, None

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(
        x=merged["lod_r"],
        y=merged["lod_py"],
        s=12,
        alpha=0.6,
        color=_COLOR_PY,
        edgecolors="none",
        rasterized=True,
        ax=ax,
    )

    # diagonal
    lims = [0, max(merged["lod_r"].max(), merged["lod_py"].max()) * 1.1]
    sns.lineplot(
        x=lims,
        y=lims,
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        label="$y = x$",
        errorbar=None,
        ax=ax,
    )

    # correlation
    corr = np.corrcoef(merged["lod_r"], merged["lod_py"])[0, 1]
    max_diff = np.max(np.abs(merged["lod_py"] - merged["lod_r"]))
    ax.text(
        0.05,
        0.95,
        f"$r$ = {corr:.6f}\nmax $|\\Delta|$ = {max_diff:.4f}\n$n$ = {len(merged)}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Reference LOD (R)")
    ax.set_ylabel("mosaiQTL LOD (Python)")
    ax.set_title("cross-language LOD comparison")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_lod_difference(py_scan_result, r_results_csv, save_path=None):
    """residual plot: lod difference (python - r) by genomic position.

    parameters
    ----------
    py_scan_result : ScanResult
    r_results_csv : str
    save_path : str or None
    """
    import pandas as pd

    set_style()

    py_df = py_scan_result.to_dataframe()
    r_df = pd.read_csv(r_results_csv)

    merged = py_df.merge(
        r_df, left_on="locus_id", right_on="locus", suffixes=("_py", "_r")
    )

    if len(merged) == 0:
        return None, None

    fig, ax = plt.subplots(figsize=(12, 3))

    diff = merged["lod_py"] - merged["lod_r"]

    # color by chromosome
    chromosomes = sorted(
        merged["chromosome"].unique(), key=lambda x: int(x) if x.isdigit() else 99
    )
    palette = _CHR_PALETTE
    colors = [palette[chromosomes.index(c) % 2] for c in merged["chromosome"]]

    ax.scatter(
        range(len(diff)),
        diff,
        c=colors,
        s=8,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(y=0, color=_COLOR_R, linewidth=0.8, linestyle="--")

    # annotate stats
    ax.text(
        0.98,
        0.95,
        f"mean $\\Delta$ = {diff.mean():.4f}\n"
        f"sd $\\Delta$ = {diff.std():.4f}\n"
        f"max $|\\Delta|$ = {np.abs(diff).max():.4f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
    )

    ax.set_ylim(bottom=min(-0.1, diff.min() * 1.2), top=max(0.1, diff.max() * 1.2))
    ax.set_xlabel("locus index")
    ax.set_ylabel("$\\Delta$ LOD (Python $-$ R)")
    ax.set_title("LOD residuals")
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_founder_effects_comparison(
    py_scan_result, genomecache, r_effects_csv, save_path=None
):
    """side-by-side founder effects at peak: python vs r.

    parameters
    ----------
    py_scan_result : ScanResult
    genomecache : GenomeCache
    r_effects_csv : str
        path to r allele_effects.csv
    save_path : str or None
    """
    import pandas as pd

    set_style()

    # python: peak locus effects
    peak = py_scan_result.peak_locus()
    chr_mask = py_scan_result.chromosomes == peak["chromosome"]
    chr_indices = np.where(chr_mask)[0]
    peak_idx = chr_indices[np.argmax(py_scan_result.lod[chr_mask])]
    py_betas = py_scan_result.betas[peak_idx]

    j_minus_1 = len(py_betas)
    founder_names = genomecache.founders[:j_minus_1]
    ref_founder = genomecache.founders[-1]

    py_all = np.append(py_betas, 0.0)
    all_names = list(founder_names) + [ref_founder]

    # r: read effects for the same gene and peak locus
    r_effects = None
    if os.path.exists(r_effects_csv):
        r_df = pd.read_csv(r_effects_csv)
        # match the peak locus
        r_match = r_df[r_df["locus"] == peak["locus_id"]]
        if len(r_match) > 0:
            r_effects = r_match.set_index("founder")["effect"]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(all_names))
    width = 0.35

    # python bars
    colors_py = [FOUNDER_COLORS.get(n, COLOR_CC_FALLBACK) for n in all_names]
    ax.bar(
        x - width / 2,
        py_all,
        width,
        color=colors_py,
        edgecolor="white",
        linewidth=0.5,
        label="mosaiQTL (Python)",
    )

    # r bars (if available)
    if r_effects is not None:
        r_all = np.array([r_effects.get(n, 0.0) for n in all_names])
        ax.bar(
            x + width / 2,
            r_all,
            width,
            color=colors_py,
            edgecolor=COLOR_NEUTRAL_DARK,
            linewidth=0.5,
            alpha=0.5,
            label="Reference (R)",
        )

    ax.axhline(y=0, color=COLOR_NEUTRAL_DARK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("founder effect ($\\hat{\\beta}_f$)")
    ax.set_title(f'founder effects comparison at peak (chr{peak["chromosome"]})')
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_validation_panel(
    scan_result,
    k_matrix,
    genomecache,
    true_params=None,
    phi_estimated=None,
    phi_true=None,
    save_dir="validation/qc_plots",
):
    """generate full qc panel for validation.

    generates two sets of plots:
      1. internal qc (always): lod profile, kinship, founder effects,
         qq plot, dispersion scatter
      2. cross-language comparison (if r output exists): lod correlation,
         lod residuals, founder effects comparison

    also exports python scan results as csv for r comparison.

    parameters
    ----------
    scan_result : ScanResult
    k_matrix : np.ndarray
    genomecache : GenomeCache
    true_params : dict or None
    phi_estimated : np.ndarray or None
    phi_true : np.ndarray or None
    save_dir : str
    """
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # --- export python results for r comparison ---
    py_df = scan_result.to_dataframe()
    py_csv = os.path.join(save_dir, "py_scan_results.csv")
    py_df.to_csv(py_csv, index=False)
    logger.info(f"exported python scan results: {py_csv}")

    # also export h2 per chromosome
    if hasattr(scan_result, "h2_null") and scan_result.h2_null:
        h2_rows = [
            {"chromosome": c, "h2_null": h} for c, h in scan_result.h2_null.items()
        ]
        pd.DataFrame(h2_rows).to_csv(
            os.path.join(save_dir, "py_h2_null.csv"), index=False
        )

    # --- 1. internal qc plots (always generated) ---
    plot_lod_profile(
        scan_result,
        true_params=true_params,
        save_path=os.path.join(save_dir, "lod_profile.png"),
    )
    plot_kinship_heatmap(
        k_matrix,
        sample_ids=list(genomecache.sample_ids),
        save_path=os.path.join(save_dir, "kinship_heatmap.png"),
    )
    plot_founder_effects(
        scan_result,
        genomecache,
        save_path=os.path.join(save_dir, "founder_effects.png"),
    )
    plot_pvalue_qq(scan_result, save_path=os.path.join(save_dir, "pvalue_qq.png"))
    if phi_estimated is not None and phi_true is not None:
        plot_dispersion_vs_truth(
            phi_estimated,
            phi_true,
            save_path=os.path.join(save_dir, "dispersion_scatter.png"),
        )

    # --- 2. cross-language comparison (if r output available) ---
    r_output_dir = os.path.join(os.path.dirname(save_dir), "r_data", "r_output")
    r_scan_csv = os.path.join(r_output_dir, "scan_results.csv")
    r_effects_csv = os.path.join(r_output_dir, "allele_effects.csv")

    if os.path.exists(r_scan_csv):
        logger.info("r output detected — generating cross-language comparison plots")

        plot_lod_correlation(
            scan_result,
            r_scan_csv,
            save_path=os.path.join(save_dir, "cross_lod_correlation.png"),
        )
        plot_lod_difference(
            scan_result,
            r_scan_csv,
            save_path=os.path.join(save_dir, "cross_lod_residuals.png"),
        )
        if os.path.exists(r_effects_csv):
            plot_founder_effects_comparison(
                scan_result,
                genomecache,
                r_effects_csv,
                save_path=os.path.join(save_dir, "cross_founder_effects.png"),
            )
    else:
        logger.info(
            f"no r output at {r_scan_csv} — skipping cross-language plots. "
            f"run: Rscript validation/r_reference/run_validation.R validation/r_data/"
        )

    plt.close("all")
    logger.info(f"qc panel saved to {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# S6/S7/S9/S11 — scan-type comparison helpers
# Each function overlays Python and R output for a specific scan type.
# Pattern: Python ScanResult + R CSV (same column schema as plot_lod_correlation).
# Works for both LMM and NB outputs since both return ScanResult.
# ─────────────────────────────────────────────────────────────────────────────


def plot_permutation_comparison(
    py_thresholds, r_thresholds_csv, alpha=0.05, save_path=None
):
    """compare permutation-derived LOD thresholds: Python vs R (S6).

    parameters
    ----------
    py_thresholds : dict  {alpha: lod_threshold}  from get_gev_thresholds()
    r_thresholds_csv : str  path to r thresholds CSV (columns: alpha, threshold)
    alpha : float  significance level to annotate
    save_path : str or None
    """
    import pandas as pd

    set_style()

    r_df = pd.read_csv(r_thresholds_csv)
    fig, ax = plt.subplots(figsize=(6, 4))

    py_thresh = py_thresholds.get(alpha, float("nan"))

    r_match = r_df[np.isclose(r_df["alpha"].values, alpha)]
    r_thresh = (
        float(r_match["threshold"].values[0]) if len(r_match) > 0 else float("nan")
    )

    ax.bar(
        ["Python", "R"], [py_thresh, r_thresh], color=[_COLOR_PY, _COLOR_R], width=0.5
    )
    ax.set_ylabel("LOD threshold")
    ax.set_title(f"permutation threshold comparison (α = {alpha})")
    sns.despine(ax=ax)

    delta = abs(py_thresh - r_thresh)
    ax.text(
        0.98,
        0.95,
        f"|Δ| = {delta:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"saved: {save_path}")
    return fig, ax


def plot_mi_scan_comparison(py_mi_result, r_mi_csv, save_path=None):
    """compare MI-averaged LOD scores: Python vs R reference (S7).

    mirrors miQTL multi.imput.lmmbygls — R averages LOD over n_imp draws.

    parameters
    ----------
    py_mi_result : ScanResult  from scan_h2lmm_mi()
    r_mi_csv : str  path to r MI scan CSV (columns: locus, lod)
    save_path : str or None
    """
    import pandas as pd

    set_style()

    py_df = py_mi_result.to_dataframe()
    r_df = pd.read_csv(r_mi_csv)
    merged = py_df.merge(
        r_df, left_on="locus_id", right_on="locus", suffixes=("_py", "_r")
    )
    if len(merged) == 0:
        logger.warning("plot_mi_scan_comparison: no shared loci")
        return None, None

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        x=merged["lod_r"],
        y=merged["lod_py"],
        s=12,
        alpha=0.6,
        color=_COLOR_PY,
        edgecolors="none",
        rasterized=True,
        ax=ax,
    )
    lims = [0, max(merged["lod_r"].max(), merged["lod_py"].max()) * 1.1]
    sns.lineplot(
        x=lims,
        y=lims,
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        errorbar=None,
        ax=ax,
    )
    corr = np.corrcoef(merged["lod_r"], merged["lod_py"])[0, 1]
    ax.text(
        0.05,
        0.95,
        f"$r$ = {corr:.6f}\n$n$ = {len(merged)}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("MI LOD (R)")
    ax.set_ylabel("MI LOD (Python)")
    ax.set_title("multiple-imputation scan comparison (S7)")
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_mediation_comparison(
    py_mediation_df, r_mediation_csv, mediator_col="mediator_lod", save_path=None
):
    """compare mediation LOD scores: Python vs R reference (S9).

    mirrors miQTL mediation.scan.qr / extract.mediation.qr.

    parameters
    ----------
    py_mediation_df : pd.DataFrame  from fixef_mediation_scan(); columns: locus_id, lod
    r_mediation_csv : str  path to r mediation CSV
    mediator_col : str  LOD column in r CSV
    save_path : str or None
    """
    import pandas as pd

    set_style()

    r_df = pd.read_csv(r_mediation_csv)
    merged = py_mediation_df.merge(
        r_df, left_on="locus_id", right_on="locus", suffixes=("_py", "_r")
    )
    if len(merged) == 0:
        logger.warning("plot_mediation_comparison: no shared loci")
        return None, None

    py_col = (
        f"{mediator_col}_py" if f"{mediator_col}_py" in merged.columns else mediator_col
    )
    r_col = (
        f"{mediator_col}_r" if f"{mediator_col}_r" in merged.columns else mediator_col
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        x=merged[r_col],
        y=merged[py_col],
        s=12,
        alpha=0.6,
        color=_COLOR_PY,
        edgecolors="none",
        rasterized=True,
        ax=ax,
    )
    lims = [0, max(merged[r_col].max(), merged[py_col].max()) * 1.1]
    sns.lineplot(
        x=lims,
        y=lims,
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        errorbar=None,
        ax=ax,
    )
    corr = np.corrcoef(merged[r_col], merged[py_col])[0, 1]
    ax.text(
        0.05, 0.95, f"$r$ = {corr:.6f}", transform=ax.transAxes, fontsize=10, va="top"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("mediation LOD (R)")
    ax.set_ylabel("mediation LOD (Python)")
    ax.set_title("mediation scan comparison (S9)")
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_instability_comparison(py_instability, r_instability_csv, save_path=None):
    """compare instability LOD scores: Python vs R reference (S11).

    mirrors miQTL instability.lm.scan — bootstrap mean LOD over subsamples.

    parameters
    ----------
    py_instability : ScanResult  from instability_lm_scan()
    r_instability_csv : str  path to r instability CSV (columns: locus, lod)
    save_path : str or None
    """
    import pandas as pd

    set_style()

    py_df = py_instability.to_dataframe()
    r_df = pd.read_csv(r_instability_csv)
    merged = py_df.merge(
        r_df, left_on="locus_id", right_on="locus", suffixes=("_py", "_r")
    )
    if len(merged) == 0:
        logger.warning("plot_instability_comparison: no shared loci")
        return None, None

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        x=merged["lod_r"],
        y=merged["lod_py"],
        s=12,
        alpha=0.6,
        color=_COLOR_PY,
        edgecolors="none",
        rasterized=True,
        ax=ax,
    )
    lims = [0, max(merged["lod_r"].max(), merged["lod_py"].max()) * 1.1]
    sns.lineplot(
        x=lims,
        y=lims,
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        errorbar=None,
        ax=ax,
    )
    corr = np.corrcoef(merged["lod_r"], merged["lod_py"])[0, 1]
    ax.text(
        0.05, 0.95, f"$r$ = {corr:.6f}", transform=ax.transAxes, fontsize=10, va="top"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("instability LOD (R)")
    ax.set_ylabel("instability LOD (Python)")
    ax.set_title("instability scan comparison (S11)")
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# NB-specific validation (no R reference — internal consistency checks)
# ─────────────────────────────────────────────────────────────────────────────


def plot_phi_distribution(phi_shrunk, phi_raw=None, save_path=None):
    """histogram of EB-shrunk dispersion estimates across genes (NB Stage 1).

    parameters
    ----------
    phi_shrunk : np.ndarray  EB-shrunk φ per gene (from fit_null_sc_native)
    phi_raw : np.ndarray or None  raw per-gene φ before shrinkage
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(float(np.nanpercentile(phi_shrunk, 99)), 1e-3), 40)
    sns.histplot(
        phi_shrunk[~np.isnan(phi_shrunk)],
        bins=bins,
        color=_COLOR_PY,
        alpha=0.7,
        label="shrunk φ",
        ax=ax,
        linewidth=0,
    )
    if phi_raw is not None:
        sns.histplot(
            phi_raw[~np.isnan(phi_raw)],
            bins=bins,
            color=_COLOR_R,
            alpha=0.4,
            label="raw φ",
            ax=ax,
            linewidth=0,
        )
        ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_xlabel("overdispersion φ")
    ax.set_ylabel("gene count")
    ax.set_title("NB Stage 1: dispersion distribution")
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_nb_vs_lmm_comparison(nb_scan_result, lmm_scan_result, save_path=None):
    """scatter of NB score LOD vs Gaussian LMM LOD at each locus.

    validates that the NB and LMM scans agree in direction and roughly
    in magnitude when the trait is approximately Gaussian. high correlation
    is expected for genes with moderate dispersion.

    parameters
    ----------
    nb_scan_result  : ScanResult  from scan_nb_gene()
    lmm_scan_result : ScanResult  from scan_h2lmm()
    save_path : str or None
    """
    set_style()

    nb_df = nb_scan_result.to_dataframe()
    lmm_df = lmm_scan_result.to_dataframe()
    merged = nb_df.merge(lmm_df, on="locus_id", suffixes=("_nb", "_lmm"))
    if len(merged) == 0:
        logger.warning("plot_nb_vs_lmm_comparison: no shared loci")
        return None, None

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(
        x=merged["lod_lmm"],
        y=merged["lod_nb"],
        s=12,
        alpha=0.6,
        color=_COLOR_PY,
        edgecolors="none",
        rasterized=True,
        ax=ax,
    )
    lims = [0, max(merged["lod_lmm"].max(), merged["lod_nb"].max()) * 1.1]
    sns.lineplot(
        x=lims,
        y=lims,
        linestyle="--",
        color=_COLOR_R,
        linewidth=1,
        alpha=0.5,
        label="y = x",
        errorbar=None,
        ax=ax,
    )
    if len(merged) > 2:
        corr = np.corrcoef(merged["lod_lmm"], merged["lod_nb"])[0, 1]
        ax.text(
            0.05,
            0.95,
            f"$r$ = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("LMM LOD")
    ax.set_ylabel("NB score LOD")
    ax.set_title("NB vs LMM LOD comparison")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
