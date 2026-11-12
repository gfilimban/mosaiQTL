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
    "merge_ld_groups",
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
    with np.errstate(divide='ignore', invalid='ignore'):
        # Inf → 700 (float64 overflow from a very large z; treat as strong evidence).
        # NaN → log(1e-300) (failed scan; near-zero weight, not dominant — a NaN
        # ABF also fails np.isfinite and would otherwise map to 700, collapsing
        # every other SNP's PIP to ~0).
        log_abf = np.where(
            np.isnan(abf), np.log(1e-300),
            np.where(np.isfinite(abf), np.log(np.maximum(abf, 1e-300)), 700.0),
        )
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


def merge_ld_groups(
    rep_patterns: list,
    r2_matrix: np.ndarray,
    snp_patterns: "list[tuple]",
    r2_threshold: float = 0.8,
) -> "dict[int, int]":
    """Merge LD-redundant founder-pattern representatives into LD groups.

    Greedy algorithm: sorts representatives by their maximum off-diagonal r²
    (most correlated first) and merges any pair with r² ≥ *r2_threshold* into
    the group of the higher-LOD representative.  Applied after the imputed SNP
    scan where ``rep_patterns`` are the unique 8-bit founder allele patterns and
    ``r2_matrix`` is the (n_reps × n_reps) pairwise dosage correlation matrix.

    Parameters
    ----------
    rep_patterns : list of tuple
        Unique founder allele patterns in the order they appear in
        ``r2_matrix``.  Length = n_reps.
    r2_matrix : np.ndarray
        Shape ``(n_reps, n_reps)``.  Pairwise r² between representative
        dosage vectors.
    snp_patterns : list of tuple
        Per-SNP founder allele pattern (same length as the SNP scan result).
    r2_threshold : float
        Merge threshold; two representatives are merged when r² ≥ this value.
        Default 0.8 is conservative; 0.5 is the commonly used minimum.

    Returns
    -------
    dict
        Maps each SNP index (0-based position in *snp_patterns*) to a group
        integer.  SNPs in the same group share a merged LD haplotype.
    """
    n_reps = len(rep_patterns)
    pat_to_rep_idx = {p: i for i, p in enumerate(rep_patterns)}

    # union-find for representative merging
    parent = list(range(n_reps))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            if r2_matrix[i, j] >= r2_threshold:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[rj] = ri

    # canonical group id = root of union-find tree
    rep_to_group = {i: _find(i) for i in range(n_reps)}

    # assign each SNP to its group via its founder pattern
    snp_to_group = {}
    for snp_idx, pat in enumerate(snp_patterns):
        rep_idx = pat_to_rep_idx.get(pat)
        snp_to_group[snp_idx] = rep_to_group[rep_idx] if rep_idx is not None else snp_idx
    return snp_to_group


def fine_map_region(
    lod: np.ndarray,
    beta_dosage: np.ndarray,
    se_dosage: np.ndarray,
    snp_ids: np.ndarray | None = None,
    pos_mb: np.ndarray | None = None,
    ld_groups: "dict[int, int] | None" = None,
    prior_var: float = 0.04,
    coverage: float = 0.95,
) -> pd.DataFrame:
    """Run single-causal-variant Bayesian fine-mapping for one genomic region.

    Without ``ld_groups`` the prior is uniform over all n_snps variants and
    PIP_j = ABF_j / Σ ABF_k.  With large windows containing many correlated
    SNPs the individual PIPs become uniformly small.

    With ``ld_groups`` (from :func:`merge_ld_groups`) the prior is uniform
    over haplotype LD groups rather than individual SNPs.  ABF is computed
    from the highest-LOD representative of each group; posterior mass is
    distributed equally among group members.  The credible set is grown
    group-by-group from highest to lowest group PIP, so all CS members fall
    within a single LD haplotype block.

    Parameters
    ----------
    lod : np.ndarray
        Shape ``(n_snps,)``.  LOD scores at each SNP.
    beta_dosage : np.ndarray
        Shape ``(n_snps,)``.  Per-SNP dosage effect from the association scan.
    se_dosage : np.ndarray
        Shape ``(n_snps,)``.  Wald standard error of each ``beta_dosage``.
        The Wald z-score is ``beta_dosage / se_dosage``; SNPs with a
        non-finite or non-positive SE are excluded from the posterior.
    snp_ids : np.ndarray or None, optional
        String SNP identifiers.  Defaults to integer indices.
    pos_mb : np.ndarray or None, optional
        Physical positions in Mb, stored in the result.
    ld_groups : dict or None, optional
        Maps each SNP index → integer group id from :func:`merge_ld_groups`.
        When provided, ABF is run on group representatives; all members of a
        group receive equal share of the group's posterior PIP.
    prior_var : float, optional
        Wakefield prior effect variance *W*, in the units of ``beta_dosage``.
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
    se_dosage = np.asarray(se_dosage, dtype=np.float64)
    n_snps = len(lod)

    # SNPs without a finite LOD, effect, or positive SE have no valid Wald
    # statistic; they get a placeholder (z=0, SE=1) here and NaN ABF below.
    valid_mask = (
        np.isfinite(lod)
        & np.isfinite(beta_dosage)
        & np.isfinite(se_dosage)
        & (se_dosage > 0)
    )
    beta_valid = np.where(valid_mask, beta_dosage, 0.0)
    se_valid = np.where(valid_mask, se_dosage, 1.0)

    z = beta_valid / se_valid
    abf = wakefield_abf(z, se_valid, prior_var=prior_var)
    # ABF for invalid SNPs is set to NaN so abf_pip assigns them near-zero weight.
    abf = np.where(valid_mask, abf, np.nan)

    if ld_groups is not None:
        # ── LD-aware ABF ────────────────────────────────────────────────────
        # group SNPs; pick best (highest ABF) representative per group
        group_best_abf: dict[int, float] = {}
        group_members: dict[int, list] = {}
        for snp_idx, gid in ld_groups.items():
            group_members.setdefault(gid, []).append(snp_idx)
            if gid not in group_best_abf or abf[snp_idx] > group_best_abf[gid]:
                group_best_abf[gid] = abf[snp_idx]

        # ABF / PIP over groups (n_groups ≤ 256 for 8-founder CC data)
        group_ids = list(group_best_abf.keys())
        group_abf_arr = np.array([group_best_abf[g] for g in group_ids])
        group_pip_arr = abf_pip(group_abf_arr)  # sums to 1 over groups
        group_pip_map = dict(zip(group_ids, group_pip_arr))

        # distribute each group's PIP equally among members
        pip = np.zeros(n_snps, dtype=np.float64)
        for gid, members in group_members.items():
            per_member = group_pip_map[gid] / len(members)
            for mi in members:
                pip[mi] = per_member

        # credible set: include all members of groups until coverage reached,
        # processing groups from highest to lowest PIP
        in_cs = np.zeros(n_snps, dtype=bool)
        cumulative = 0.0
        for gid in sorted(group_ids, key=lambda g: -group_pip_map[g]):
            if cumulative >= coverage:
                break
            cumulative += group_pip_map[gid]
            for mi in group_members[gid]:
                in_cs[mi] = True
    else:
        # ── standard ABF (no LD information) ────────────────────────────────
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

    n_groups = len(set(ld_groups.values())) if ld_groups else n_snps
    logger.debug(
        "fine_map_region: %d SNPs (%d LD groups), %d in %d%% CS, top PIP=%.3f",
        n_snps,
        n_groups,
        int(in_cs.sum()),
        int(coverage * 100),
        float(pip.max()),
    )
    return df
