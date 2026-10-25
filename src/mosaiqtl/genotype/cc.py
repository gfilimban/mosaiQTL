"""cc — canonical strain constants for CC/DO mice.

Single source of truth for founder order, colors, F1 colors, and CC
strain palettes. No other module defines hex color strings — import from here.

Axis order: founders (FOUNDER_ORDER) → F1s (F1_COLORS key order) → CC strains
(alphanumeric).

CCPopulation is the default population definition for 8-founder CC mice.
"""

from __future__ import annotations

from typing import Optional

from mosaiqtl.models.config import Population
from .colors import blend_hex_oklab

FOUNDER_ORDER: list[str] = [
    'AJ',
    'B6J',
    '129S1J',
    'NODJ',
    'NZOJ',
    'CASTJ',
    'PWKJ',
    'WSBJ',
]

FOUNDER_CODE: dict[str, str] = {
    'AJ': 'A',
    'B6J': 'B',
    '129S1J': 'C',
    'NODJ': 'D',
    'NZOJ': 'E',
    'CASTJ': 'F',
    'PWKJ': 'G',
    'WSBJ': 'H',
}

FOUNDER_COLORS: dict[str, str] = {
    'AJ': '#F4C245',
    'B6J': '#C0BFBF',
    '129S1J': '#DA9CC1',
    'NODJ': '#4F6EAF',
    'NZOJ': '#52A5DB',
    'CASTJ': '#55AF5B',
    'PWKJ': '#D83026',
    'WSBJ': '#683C91',
}

F1_COLORS: dict[str, str] = {
    'B6AF1J': '#DAC082',
    'B6129S1F1J': '#CDADC0',
    'B6CASTF1J': '#8AB78D',
    'B6NODF1J': '#8796B7',
    'B6NZOF1J': '#89B2CD',
    'B6PWKF1J': '#CC7772',
    'B6WSBF1J': '#947DA8',
}

# canonical plot axis order: founders, then F1s
STRAIN_ORDER: list[str] = FOUNDER_ORDER + list(F1_COLORS.keys())

COLOR_NEUTRAL_DARK: str = '#444444'
COLOR_CC_FALLBACK: str = '#888888'
COLOR_NEUTRAL_LIGHT: str = '#E8E8E8'
COLOR_THRESHOLD: str = '#E8870A'


def cc_strain_palette(
    cc_strains: list[str],
    haplotype_probs: Optional[dict[str, list[float]]] = None,
    n_top: int = 2,
) -> dict[str, str]:
    """Ancestry-aware color palette for CC strains using OKLab blending.

    Blends the top ``n_top`` founders by ancestry proportion in OKLab space,
    renormalised to sum to 1. Using only the dominant founders avoids the
    washed-out convergence that arises when all 8 minor contributions pull
    every strain toward the same muted mid-tone.

    Parameters
    ----------
    cc_strains :
        CC strain identifiers (e.g. ['CC001', 'CC002', ...]).
    haplotype_probs :
        Per-strain genome-wide mean founder ancestry proportions, one value
        per founder in FOUNDER_ORDER order. None → fallback grey for all.
    n_top :
        Number of dominant founders to include in the blend (default 3).

    Returns
    -------
    dict[str, str]
        Strain name → hex color string.
    """
    founder_hexes = [FOUNDER_COLORS[f] for f in FOUNDER_ORDER]
    palette: dict[str, str] = {}
    for strain in sorted(cc_strains):
        if haplotype_probs is None or strain not in haplotype_probs:
            palette[strain] = COLOR_CC_FALLBACK
            continue
        probs = list(haplotype_probs[strain])
        if len(probs) != len(FOUNDER_ORDER) or sum(probs) < 1e-9:
            palette[strain] = COLOR_CC_FALLBACK
            continue
        # keep only the top-n_top founders by proportion
        top = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:n_top]
        idxs, weights = zip(*top)
        colors = [founder_hexes[i] for i in idxs]
        palette[strain] = blend_hex_oklab(list(colors), list(weights))
    return palette


CCPopulation: Population = Population(
    name='cc',
    founders=FOUNDER_ORDER,
    colors=FOUNDER_COLORS,
    f1_colors=F1_COLORS,
    has_sex_chromosomes=True,
    sex_chromosome='chrX',
    x_donors=[],
    genotyping_error_prob=0.002,
    n_gen=1,
    inbred=True,
    loco_sufficient=True,
)


def display_founders(cache_founders: list[str]) -> list[str]:
    """Map genome cache founder letter codes (A-H) to full strain names.

    Parameters
    ----------
    cache_founders :
        Founder codes as stored in the genome cache (e.g. ['A', 'B', ...]).

    Returns
    -------
    list[str]
        Display names aligned to FOUNDER_ORDER (e.g. ['AJ', 'B6J', ...]).
    """
    _inv = {v: k for k, v in FOUNDER_CODE.items()}
    return [_inv.get(f, _inv.get(f.upper(), f)) for f in cache_founders]


def strip_j(name: str) -> str:
    """Human-readable display name: strips trailing J colony suffix.

    AJ is exempt — the J is part of its strain identity, not a suffix.

    Examples
    --------
    >>> strip_j('PWKJ')
    'PWK'
    >>> strip_j('AJ')
    'AJ'
    """
    if name == 'AJ':
        return 'AJ'
    if name.endswith('J'):
        return name[:-1]
    return name


__all__ = [
    'FOUNDER_ORDER',
    'FOUNDER_CODE',
    'FOUNDER_COLORS',
    'F1_COLORS',
    'STRAIN_ORDER',
    'strip_j',
    'COLOR_NEUTRAL_DARK',
    'COLOR_CC_FALLBACK',
    'COLOR_NEUTRAL_LIGHT',
    'COLOR_THRESHOLD',
    'display_founders',
    'cc_strain_palette',
    'CCPopulation',
]
