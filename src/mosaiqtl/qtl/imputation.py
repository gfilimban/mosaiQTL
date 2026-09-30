"""imputation — Bayesian SNP fine-mapping from imputed association statistics.

Layer 3 of the Sanger fine-mapping stack.

Takes per-SNP association statistics (LOD scores + effect signs) produced by
the haplotype-imputed scan (qtl/snp_scan.py or interpolated from the
haplotype scan via genotype/interpolation.py) and computes:

  1. Wakefield approximate Bayes factors (ABF) per SNP.
  2. Posterior inclusion probabilities (PIP) under the single-causal-variant
     model.
  3. A highest-PIP credible set capturing a user-specified coverage level
     (default 95 %).

"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mosaiqtl")

__all__ = [
    "wakefield_abf",
    "abf_pip",
    "credible_set",
    "lod_beta_to_z",
    "fine_map_region",
]

_LOG10_TO_LN: float = np.log(10.0)  # 2×this × LOD = LRT chi²


def wakefield_abf(
    z: np.ndarray,
    se: np.ndarray | float,
    prior_var: float = 0.04,
) -> np.ndarray:
    """Compute Wakefield approximate Bayes factors for a set of SNPs.

    Parameters
    ----------
    z : np.ndarray
        Shape ``(n_snps,)``.  Signed z-scores (effect / SE).
    se : np.ndarray or float
        Standard errors.  Scalar or ``(n_snps,)``; broadcast over z.
    prior_var : float, optional
        Prior effect variance *W*.  Default 0.04 (prior SD = 0.2) for
        cis-eQTL on standardized phenotypes.

    Returns
    -------
    np.ndarray
        Shape ``(n_snps,)``.  ABF values ≥ 0.

    Notes
    -----
    ``log_ABF = 0.5 × log(V/(V+W)) + z² × W / (2(V+W))``,
    ``V = SE²``, ``W = prior_var``.  Computed in log-space.
    """
    z = np.asarray(z, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)  # broadcasts naturally in arithmetic
    V = se**2
    r = prior_var / (V + prior_var)
    log_abf = 0.5 * np.log1p(-r) + 0.5 * z**2 * r
    return np.exp(log_abf)


def abf_pip(
    abf: np.ndarray,
    prior_prob: float | None = None,
) -> np.ndarray:
    """Compute posterior inclusion probabilities from ABFs.

    Assumes a single-causal-variant model with uniform prior over all SNPs.

    Parameters
    ----------
    abf : np.ndarray
        Shape ``(n_snps,)``.  ABF values from :func:`wakefield_abf`.
    prior_prob : float or None, optional
        Scalar prior per SNP.  When ``None``, uses ``1/n_snps``.  Because
        the prior is a scalar (uniform), it cancels identically in the
        normalization and does not affect the result.

    Returns
    -------
    np.ndarray
        Shape ``(n_snps,)``.  PIP values in (0, 1).
    """
    abf = np.asarray(abf, dtype=np.float64)
    log_abf = np.log(np.maximum(abf, 1e-300))
    log_z = log_abf.max()
    w = np.exp(log_abf - log_z)
    return w / w.sum()


def credible_set(
    pip: np.ndarray,
    coverage: float = 0.95,
) -> np.ndarray:
    """Return indices of the smallest credible set covering *coverage*.

    Parameters
    ----------
    pip : np.ndarray
        Shape ``(n_snps,)``.  PIP values from :func:`abf_pip`.
    coverage : float, optional
        Target cumulative posterior probability (default 0.95).

    Returns
    -------
    np.ndarray
        Integer indices in original array order whose combined PIP
        ≥ *coverage*.
    """
    pip = np.asarray(pip, dtype=np.float64)
    order = np.argsort(-pip)  # descending; avoids [::-1] copy
    cumsum = np.cumsum(pip[order])
    n_in = int(np.searchsorted(cumsum, coverage, side="left")) + 1
    return np.sort(order[: min(n_in, len(pip))])


def lod_beta_to_z(
    lod: np.ndarray,
    beta_sign: np.ndarray,
) -> np.ndarray:
    """Convert LOD scores and effect sign to signed z-scores.

    Under the 1-df LRT: ``chi² = 2 ln(10) × LOD``, ``|z| = √chi²``.

    Parameters
    ----------
    lod : np.ndarray
        Shape ``(n_snps,)``.  LOD scores ≥ 0; negatives clamped.
    beta_sign : np.ndarray
        Shape ``(n_snps,)``.  Sign of the dosage effect.  The raw
        ``beta_dosage`` column may be passed directly.

    Returns
    -------
    np.ndarray
        Shape ``(n_snps,)``.  Signed z-scores.
    """
    lod = np.asarray(lod, dtype=np.float64)
    z_abs = np.sqrt(np.maximum(lod, 0.0) * (2.0 * _LOG10_TO_LN))
    sign = np.sign(np.asarray(beta_sign, dtype=np.float64))
    sign = np.where(sign == 0, 1.0, sign)
    return sign * z_abs


def fine_map_region(
    lod: np.ndarray,
    beta_dosage: np.ndarray,
    n: int,
    snp_ids: np.ndarray | None = None,
    pos_mb: np.ndarray | None = None,
    prior_var: float = 0.04,
    coverage: float = 0.95,
) -> pd.DataFrame:
    """Run single-causal-variant Bayesian fine-mapping for one genomic region.

    Parameters
    ----------
    lod : np.ndarray
        Shape ``(n_snps,)``.  LOD scores at each SNP.
    beta_dosage : np.ndarray
        Shape ``(n_snps,)``.  Dosage effect (sign only used for z-direction).
    n : int
        Sample size; SE estimated as ``1 / √n``.
    snp_ids : np.ndarray or None, optional
        String SNP identifiers.  Defaults to integer indices.
    pos_mb : np.ndarray or None, optional
        Physical positions in Mb, stored in the result.
    prior_var : float, optional
        Wakefield prior effect variance (default 0.04).
    coverage : float, optional
        Credible-set coverage level (default 0.95).

    Returns
    -------
    pd.DataFrame
        Columns: ``snp``, [``pos_mb``], ``lod``, ``z``, ``abf``, ``pip``,
        ``in_cs``.  Sorted by ``pip`` descending.
    """
    lod = np.asarray(lod, dtype=np.float64)
    beta_dosage = np.asarray(beta_dosage, dtype=np.float64)
    n_snps = len(lod)
    se = 1.0 / np.sqrt(max(n, 1))

    z = lod_beta_to_z(lod, beta_dosage)
    abf = wakefield_abf(z, se, prior_var=prior_var)
    pip = abf_pip(abf)
    cs_idx = credible_set(pip, coverage=coverage)
    in_cs = np.zeros(n_snps, dtype=bool)
    in_cs[cs_idx] = True

    data: dict = {
        "snp": snp_ids if snp_ids is not None else np.arange(n_snps),
        "lod": lod,
        "z": z,
        "abf": abf,
        "pip": pip,
        "in_cs": in_cs,
    }
    if pos_mb is not None:
        data["pos_mb"] = np.asarray(pos_mb, dtype=np.float64)

    cols = (
        ["snp"]
        + (["pos_mb"] if pos_mb is not None else [])
        + ["lod", "z", "abf", "pip", "in_cs"]
    )
    df = (
        pd.DataFrame(data)[cols]
        .sort_values("pip", ascending=False)
        .reset_index(drop=True)
    )

    logger.debug(
        "fine_map_region: %d SNPs, %d in %d%% CS, top PIP=%.3f",
        n_snps,
        int(in_cs.sum()),
        int(coverage * 100),
        float(pip.max()),
    )
    return df
