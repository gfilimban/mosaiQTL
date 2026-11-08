"""colors — perceptually uniform color utilities.

Blending uses OKLab (Ottosson 2020), where Euclidean distance equals
perceived difference. Converting to OKLab before a weighted average
ensures haplotype-blended CC strain colors land perceptually between
their founder contributions rather than biased by sRGB gamma.
"""

from __future__ import annotations

import numpy as np

__all__ = ['hex_to_rgb', 'rgb_to_hex', 'blend_hex_oklab']

# OKLab matrices — Ottosson 2020 (bottosson.github.io/posts/oklab/)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=float)
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=float)
_M2_INV = np.linalg.inv(_M2)
_M1_INV = np.linalg.inv(_M1)


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) in [0, 1]."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) in [0, 1] to '#RRGGBB'."""
    return '#{:02X}{:02X}{:02X}'.format(round(r * 255), round(g * 255), round(b * 255))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


def _hex_to_oklab(hex_color: str) -> np.ndarray:
    rgb_lin = np.array([_srgb_to_linear(c) for c in hex_to_rgb(hex_color)])
    return _M2 @ np.cbrt(_M1 @ rgb_lin)


def _oklab_to_hex(lab: np.ndarray) -> str:
    lms = (_M2_INV @ lab) ** 3
    rgb_lin = _M1_INV @ lms
    r, g, b = (_linear_to_srgb(float(c)) for c in rgb_lin)
    return rgb_to_hex(r, g, b)


def blend_hex_oklab(colors: list[str], weights: list[float]) -> str:
    """Weighted blend of hex colors in OKLab space.

    Parameters
    ----------
    colors :
        Hex color strings ('#RRGGBB').
    weights :
        Non-negative weights; normalised internally.

    Returns
    -------
    str
        Blended '#RRGGBB' color.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total < 1e-9:
        raise ValueError('weights sum to zero')
    w /= total
    lab = sum(wi * _hex_to_oklab(c) for c, wi in zip(colors, w))
    return _oklab_to_hex(lab)  # type: ignore[arg-type]
