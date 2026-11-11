"""Shared plot styling: colors, fonts, and helpers used across every figure.

The colors defined here stand for the package itself — used for scan data,
significance thresholds, reference lines, info boxes, and chromosome bands.
They are never used to represent a specific founder strain.

Founder-strain colors are a separate palette (genotype/cc.py) and are
only used where the plot is specifically about founder identity — coefficient
profiles, effect bars, caterpillar plots, probability heatmaps, PxG plots,
and epistasis grids.
"""

from __future__ import annotations

import logging as _logging

import matplotlib.pyplot as plt
import seaborn as sns

from ..genotype.cc import (
    COLOR_CC_FALLBACK,
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_THRESHOLD,
)

# ── 1. identity colors ────────────────────────────────────────────────────
BRAND_PRIMARY: str = "#3B6EA5"  # single-series lines, KDE fills, primary dots
BRAND_SECONDARY: str = "#B5423A"  # series B in any A-vs-B comparison
BRAND_ACCENT: str = COLOR_THRESHOLD  # significance-threshold lines only
INK: str = COLOR_NEUTRAL_DARK  # reference/identity/zero lines, borders
MUTED: str = COLOR_CC_FALLBACK  # de-emphasised series, marginal fills
WASH: str = COLOR_NEUTRAL_LIGHT  # below-threshold / low-expression dots
SURFACE: str = "#FFFFFF"  # figure background, info-box fill

# two shades of the same hue for alternating chromosomes, so the bands read
# as a visual rhythm rather than as meaningful categories
CHR_BANDS: tuple[str, str] = (BRAND_PRIMARY, "#8FB0CE")

# gene-track feature colors — UTR uses a lighter shade of the CDS blue so the
# visual hierarchy reads: CDS (dark blue) > UTR (sky blue) > intron (neutral)
COLOR_UTR: str = "#6BAED6"  # sky blue — transcribed but not translated

# ── 2. semantic colormaps ─────────────────────────────────────────────────
CMAP_MAGNITUDE: str = "crest"  # unsigned magnitude: LOD score, PIP, −log10(p)
CMAP_SIGNED: str = (
    "coolwarm"  # signed effect size (beta); needs a symmetric color scale
)
CMAP_DENSITY: str = "YlOrRd"  # unsigned density: haplotype probability, |delta-beta|

# ── 3. one style = one meaning ────────────────────────────────────────────
KW_THRESHOLD: dict = dict(color=BRAND_ACCENT, ls=":", lw=1.4, alpha=0.85, zorder=4)
KW_REFERENCE: dict = dict(color=INK, ls="--", lw=1.2, alpha=0.70, zorder=3)
KW_DOT: dict = dict(s=5, alpha=0.40, linewidths=0, rasterized=True, zorder=2)
KW_DOT_MUTED: dict = dict(
    s=4,
    alpha=0.30,
    linewidths=0,
    rasterized=True,
    zorder=1,
    color=WASH,
)
KW_DOT_HILITE: dict = dict(s=60, linewidths=0.6, edgecolor=SURFACE, zorder=6)
KW_BOX: dict = dict(
    boxstyle="round,pad=0.35",
    facecolor=SURFACE,
    edgecolor=INK,
    alpha=0.90,
)

FONT_STACK: list[str] = ["Carlito", "Calibri", "DejaVu Sans"]


def set_style() -> None:
    """Apply package-wide plot style. Call once at the start of every plotting function."""
    # Carlito/Calibri are preferred but not installed on every system; DejaVu Sans
    # is the guaranteed fallback. Suppress the font-not-found warning so it does
    # not pollute logs on HPC nodes where only DejaVu is available.
    _logging.getLogger('matplotlib.font_manager').setLevel(_logging.ERROR)
    sns.set_style("ticks")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def apply_theme(ax: plt.Axes, *, from_zero: bool = True) -> plt.Axes:
    """Apply fonts, remove top/right axis lines, and start the y-axis at zero.

    Call this at the end of every plotting function.
    """
    for item in (
        [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontfamily(FONT_STACK)
    sns.despine(ax=ax)
    if from_zero:
        ax.set_ylim(bottom=0)
    return ax


def info_box(
    ax: plt.Axes,
    text: str,
    corner: str = "tl",
    fontsize: float = 8,
) -> None:
    """Monospace stat box anchored to a corner of the axes.

    Parameters
    ----------
    ax : plt.Axes
        Axes to annotate.
    text : str
        Box text; newlines allowed.
    corner : str
        One of 'tl', 'tr', 'bl', 'br'.
    fontsize : float
        Font size in points.
    """
    coords = {
        "tl": (0.03, 0.97, "left", "top"),
        "tr": (0.97, 0.97, "right", "top"),
        "bl": (0.03, 0.03, "left", "bottom"),
        "br": (0.97, 0.03, "right", "bottom"),
    }
    x, y, ha, va = coords[corner]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va=va,
        family="monospace",
        bbox=KW_BOX,
        zorder=5,
    )
