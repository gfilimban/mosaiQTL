"""_theme — mosaiQTL package visual identity (Layer 1).

Layer 1 (this module): colors that mean "mosaiQTL". Used for scan data,
thresholds, reference lines, info boxes, chromosome bands. NEVER a founder color.

Layer 2 (genotype/founders.py): FOUNDER_COLORS / F1_COLORS / cc_strain_palette.
Used ONLY when the visual encoding IS founder identity — coef_profile, effect_bars,
caterpillar, prob_heatmap, pxg, epistasis_grid. Importing FOUNDER_COLORS outside
viz/effects.py and viz/haplotypes.py is a bug.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from ..genotype.founders import (
    COLOR_CC_FALLBACK,
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_THRESHOLD,
)

# ── 1. identity colors ────────────────────────────────────────────────────
BRAND_PRIMARY: str = "#3B6EA5"  # single-series lines, KDE fills, primary dots
BRAND_SECONDARY: str = "#B5423A"  # series B in any A-vs-B comparison
BRAND_ACCENT: str = COLOR_THRESHOLD  # '#E8870A' — thresholds ONLY
INK: str = COLOR_NEUTRAL_DARK  # '#444444' — reference/identity/zero lines, borders
MUTED: str = COLOR_CC_FALLBACK  # '#888888' — de-emphasised series, marginal fills
WASH: str = COLOR_NEUTRAL_LIGHT  # '#E8E8E8' — below-threshold / low-expr dots
SURFACE: str = "#FFFFFF"  # figure face, info-box fill

# Chromosome alternation: two values of one hue so bands never read as categories
CHR_BANDS: tuple[str, str] = (BRAND_PRIMARY, "#8FB0CE")

# ── 2. semantic colormaps ─────────────────────────────────────────────────
CMAP_MAGNITUDE: str = "crest"  # unsigned magnitude: LOD, PIP, −log₁₀p
CMAP_SIGNED: str = "coolwarm"  # signed effect (beta) — requires symmetric norm
CMAP_DENSITY: str = "YlOrRd"  # unsigned density: haplotype prob, |Δβ|

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
    """Package-wide rcParams. Call once at the top of every figure-producing function."""
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
    """Font stack + despine + axes-from-zero. Call at the end of every plot function."""
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
