"""
founders — canonical strain constants for CC/DO mice

single source of truth for all founder order, colors, and strain palettes.
no other module in mosaiqtl defines hex color strings — import from here.

axis order rule (enforced in all plots):
    Founders (FOUNDER_ORDER) → F1s (F1_COLORS key order) → CC strains (alphanumeric)

CCPopulation is the default Population instance for 8-founder CC mice.
import it from here rather than constructing it at call sites.
"""

from __future__ import annotations

from typing import Optional

from mosaiqtl.models.config import Population

FOUNDER_ORDER: list[str] = [
    "AJ",
    "B6J",
    "129S1J",
    "NODJ",
    "NZOJ",
    "CASTJ",
    "PWKJ",
    "WSBJ",
]

FOUNDER_CODE: dict[str, str] = {
    "AJ": "A",
    "B6J": "B",
    "129S1J": "C",
    "NODJ": "D",
    "NZOJ": "E",
    "CASTJ": "F",
    "PWKJ": "G",
    "WSBJ": "H",
}

FOUNDER_COLORS: dict[str, str] = {
    "AJ": "#F4C245",
    "B6J": "#C0BFBF",
    "129S1J": "#DA9CC1",
    "NODJ": "#4F6EAF",
    "NZOJ": "#52A5DB",
    "CASTJ": "#55AF5B",
    "PWKJ": "#D83026",
    "WSBJ": "#683C91",
}

F1_COLORS: dict[str, str] = {
    "B6AF1J": "#DAC082",
    "B6129S1F1J": "#CDADC0",
    "B6CASTF1J": "#8AB78D",
    "B6NODF1J": "#8796B7",
    "B6NZOF1J": "#89B2CD",
    "B6PWKF1J": "#CC7772",
    "B6WSBF1J": "#947DA8",
}

# canonical plot axis order: founders → F1s → CC strains (caller appends CC alphanumeric)
STRAIN_ORDER: list[str] = FOUNDER_ORDER + list(F1_COLORS.keys())

# neutral greys used in plots — defined here so no module ever hardcodes hex
COLOR_NEUTRAL_DARK: str = "#444444"  # axis lines, tick labels, reference bars
COLOR_CC_FALLBACK: str = "#888888"  # CC strains without ancestry probability data
COLOR_NEUTRAL_LIGHT: str = "#E8E8E8"  # NaN/masked cells in heatmaps
COLOR_THRESHOLD: str = "#E8870A"  # LOD threshold lines + quadrant annotations


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' hex string to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(  # type: ignore[return-value]
        int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)  # noqa: E203
    )


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) floats in [0, 1] to '#RRGGBB' hex string."""
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


def _blend_hex(color_a: str, color_b: str, t: float = 0.5) -> str:
    """linear RGB blend; t=0 → color_a, t=1 → color_b."""
    ra, ga, ba = _hex_to_rgb(color_a)
    rb, gb, bb = _hex_to_rgb(color_b)
    return _rgb_to_hex(ra + t * (rb - ra), ga + t * (gb - ga), ba + t * (bb - ba))


def cc_strain_palette(
    cc_strains: list[str],
    haplotype_probs: Optional[dict[str, list[float]]] = None,
) -> dict[str, str]:
    """ancestry-aware color palette for CC strains.

    parameters
    ----------
    cc_strains : list[str]
        CC strain identifiers (e.g. ['CC001', 'CC002', ...])
    haplotype_probs : dict[str, list[float]], optional
        per-strain genome-wide mean haplotype probabilities (length 8, FOUNDER_ORDER).
        when provided, colors are blended between the two dominant founder colors
        weighted by their relative proportions.
        when None, a fixed grey (#888888) is returned for all CC strains.

    returns
    -------
    dict[str, str]
        mapping strain name → hex color string
    """
    palette: dict[str, str] = {}
    for strain in sorted(cc_strains):
        if haplotype_probs is None or strain not in haplotype_probs:
            palette[strain] = COLOR_CC_FALLBACK
            continue

        probs = list(haplotype_probs[strain])
        if len(probs) != len(FOUNDER_ORDER):
            palette[strain] = COLOR_CC_FALLBACK
            continue

        # find two dominant founders
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        i1, p1 = indexed[0]
        i2, p2 = indexed[1]
        total = p1 + p2
        if total < 1e-9:
            palette[strain] = COLOR_CC_FALLBACK
            continue

        t = p2 / total  # blend weight toward second-dominant founder
        color = _blend_hex(
            FOUNDER_COLORS[FOUNDER_ORDER[i1]],
            FOUNDER_COLORS[FOUNDER_ORDER[i2]],
            t,
        )
        palette[strain] = color

    return palette


# ---------------------------------------------------------------------------
# CCPopulation — default Population instance for 8-founder CC/DO mice (mm39)
# ---------------------------------------------------------------------------
CCPopulation: Population = Population(
    name="cc",
    founders=FOUNDER_ORDER,
    colors=FOUNDER_COLORS,
    f1_colors=F1_COLORS,
    has_sex_chromosomes=True,
    sex_chromosome="chrX",
    x_donors=[],  # set by caller when sex-chromosome QTL handling is needed
    genotyping_error_prob=0.002,
    n_gen=1,
    inbred=True,
    loco_sufficient=True,
)


def display_founders(cache_founders: list[str]) -> list[str]:
    """Map HAPPY cache founder codes (A–H) to FOUNDER_ORDER display names.

    Parameters
    ----------
    cache_founders :
        Founder codes as stored in the genome cache (e.g. ``['A', 'B', ...]``).

    Returns
    -------
    list[str]
        Display names aligned to FOUNDER_ORDER (e.g. ``['AJ', 'B6J', ...]``).
    """
    _inv = {v: k for k, v in FOUNDER_CODE.items()}
    return [_inv.get(f, _inv.get(f.upper(), f)) for f in cache_founders]


__all__ = [
    "FOUNDER_ORDER",
    "FOUNDER_CODE",
    "FOUNDER_COLORS",
    "F1_COLORS",
    "STRAIN_ORDER",
    "COLOR_NEUTRAL_DARK",
    "COLOR_CC_FALLBACK",
    "COLOR_NEUTRAL_LIGHT",
    "COLOR_THRESHOLD",
    "display_founders",
    "cc_strain_palette",
    "CCPopulation",
]
