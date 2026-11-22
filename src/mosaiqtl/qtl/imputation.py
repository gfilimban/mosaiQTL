"""imputation — Bayesian SNP fine-mapping from imputed association statistics.

Layer 3 of the Sanger fine-mapping stack.

Takes per-SNP association statistics (LOD scores + effect signs) produced by
the haplotype-imputed scan (qtl/snp_scan.py or interpolated from the
haplotype scan via genotype/interpolation.py) and computes:

  1. Approximate Bayes factors per SNP, kept as natural logs (``log_abf``,
     association over null).
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
    "compute_log_abf",
    "log_abf_to_pip",
    "credible_set",
    "lod_beta_to_z",
    "fine_map_region",
    "merge_ld_groups",
    "pattern_credible_sets",
    "format_founder_pattern",
]

_LOG10_TO_LN: float = np.log(10.0)  # 2×this × LOD = LRT chi²


def compute_log_abf(
    z: np.ndarray,
    se: np.ndarray | float,
    prior_var: float = 0.04,
) -> np.ndarray:
    r"""Natural log of the approximate Bayes factor, association over null.

    Parameters
    ----------
    z : np.ndarray
        Shape ``(n_snps,)``.  Signed z-scores (effect / SE).
    se : np.ndarray or float
        Standard errors.  Scalar or ``(n_snps,)``; broadcast over z.
    prior_var : float, optional
        Prior effect variance $W$.  Default 0.04 (prior SD = 0.2) for
        cis-eQTL on standardized phenotypes.

    Returns
    -------
    np.ndarray
        Shape ``(n_snps,)``.  $\log\mathrm{ABF}$; positive favours association.

    Notes
    -----
    $\log\mathrm{ABF} = \tfrac12\log\frac{V}{V+W} + \frac{z^2 W}{2(V+W)}$ with
    $V = \mathrm{SE}^2$.
    """
    z = np.asarray(z, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    r = prior_var / (se**2 + prior_var)
    return 0.5 * np.log1p(-r) + 0.5 * z**2 * r


def log_abf_to_pip(log_abf: np.ndarray) -> np.ndarray:
    r"""Posterior inclusion probabilities from log Bayes factors.

    Single-causal-variant model with a uniform prior over the SNPs, so
    $\mathrm{PIP}_j = \mathrm{softmax}(\log\mathrm{ABF})_j$, evaluated by
    subtracting the maximum so large values cannot overflow.

    Parameters
    ----------
    log_abf : np.ndarray
        Shape ``(n_snps,)``.  NaN marks a SNP with no valid statistic and gets
        zero weight.

    Returns
    -------
    np.ndarray
        Shape ``(n_snps,)``.  Probabilities summing to 1, or all zeros when no
        SNP has a finite $\log\mathrm{ABF}$.
    """
    x = np.where(np.isnan(log_abf), -np.inf, np.asarray(log_abf, dtype=np.float64))
    top = x.max()
    if not np.isfinite(top):
        return np.zeros_like(x)
    w = np.exp(x - top)
    return w / w.sum()


def credible_set(
    pip: np.ndarray,
    coverage: float = 0.95,
) -> np.ndarray:
    """Return indices of the smallest credible set covering *coverage*.

    Parameters
    ----------
    pip : np.ndarray
        Shape ``(n_snps,)``.  Posterior inclusion probabilities.
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
    patterns: "np.ndarray | None" = None,
    tss_mb: "float | None" = None,
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
        Prior effect variance $W$, in the units of ``beta_dosage``.
    coverage : float, optional
        Credible-set coverage level (default 0.95).
    patterns : np.ndarray or None, optional
        Shape ``(n_snps,)``.  Founder allele pattern of each SNP as a string
        (see :func:`format_founder_pattern`); stored as the ``pattern`` column.
    tss_mb : float or None, optional
        Strand-aware transcription start site in Mb; the last tie-break of the
        row order is the distance of a variant from it.

    Returns
    -------
    pd.DataFrame
        Columns: ``snp``, [``pos_mb``], [``pattern``], [``ld_group``, with
        ``ld_groups``], ``lod``, ``z``,
        ``log_abf`` (natural log, association over null; NaN without a valid
        Wald statistic), ``pip``, ``in_cs``.  The first row is the lead
        variant.  Rows are ordered by ``pip`` descending, then ``lod``
        descending, then distance to ``tss_mb`` ascending, then input order.
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
    log_abf = np.where(valid_mask, compute_log_abf(z, se_valid, prior_var=prior_var), np.nan)

    if ld_groups is not None:
        # ── LD-aware ABF ────────────────────────────────────────────────────
        # a group's evidence is its best member; invalid members rank at -inf
        rank = np.where(valid_mask, log_abf, -np.inf)
        group_best: dict[int, float] = {}
        group_members: dict[int, list] = {}
        for snp_idx, gid in ld_groups.items():
            group_members.setdefault(gid, []).append(snp_idx)
            group_best[gid] = max(group_best.get(gid, -np.inf), rank[snp_idx])

        group_ids = list(group_best)
        group_pip_arr = log_abf_to_pip(np.array([group_best[g] for g in group_ids]))
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
        pip = log_abf_to_pip(log_abf)
        cs_idx = credible_set(pip, coverage=coverage)
        in_cs = np.zeros(n_snps, dtype=bool)
        in_cs[cs_idx] = True

    data: dict = {
        "snp": snp_ids if snp_ids is not None else np.arange(n_snps),
        "lod": lod,
        "z": z,
        "log_abf": log_abf,
        "pip": pip,
        "in_cs": in_cs,
    }
    if pos_mb is not None:
        data["pos_mb"] = np.asarray(pos_mb, dtype=np.float64)
    if patterns is not None:
        data["pattern"] = np.asarray(patterns, dtype=object)
    if ld_groups is not None:
        data["ld_group"] = np.array([ld_groups[i] for i in range(n_snps)], dtype=np.int64)

    cols = (
        ["snp"]
        + (["pos_mb"] if pos_mb is not None else [])
        + (["pattern"] if patterns is not None else [])
        + (["ld_group"] if ld_groups is not None else [])
        + ["lod", "z", "log_abf", "pip", "in_cs"]
    )
    lod_key = np.where(np.isfinite(lod), lod, -np.inf)
    if tss_mb is not None and pos_mb is not None:
        dist_key = np.abs(np.asarray(pos_mb, dtype=np.float64) - tss_mb)
        dist_key = np.where(np.isfinite(dist_key), dist_key, np.inf)
    else:
        dist_key = np.zeros(n_snps)
    # lexsort is stable and its last key is primary
    order = np.lexsort((dist_key, -lod_key, -pip))
    df = pd.DataFrame(data)[cols].iloc[order].reset_index(drop=True)

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


def format_founder_pattern(pattern: "tuple[int, ...]") -> str:
    """Render a founder allele pattern as a string of 0/1 digits, one per founder.

    Digit $j$ is 1 when founder $j$ carries the alternative allele and 0 when it
    carries the reference allele; ``.`` marks a missing call.  Founders follow the
    genotype-cache order.
    """
    return "".join("." if x < 0 else str(int(x)) for x in pattern)


LD_GROUP_COLS: list[str] = [
    "ld_group", "group_pip", "patterns", "n_patterns", "n_variants",
    "span_bp", "best_lod_variant", "best_lod", "in_cs",
]


def ld_group_summary(fm_df: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    r"""Summarise LD-aware fine-mapping posteriors at the LD-group level.

    Variants whose founder patterns have $r^2 \ge$ the merge threshold form one
    LD group and are not separated by the data.  The group posterior is the sum
    of its members' $\mathrm{PIP}$ and equals the softmax of the group
    $\log \mathrm{ABF}$ under a prior that is uniform over groups.  Membership
    of the credible set is taken from the member rows, so the table agrees with
    the variant-level ``in_cs``.

    Parameters
    ----------
    fm_df : pd.DataFrame
        Output of :func:`fine_map_region` called with ``ld_groups`` and
        ``patterns``; must hold every variant of the region.
    tol : float, optional
        Allowed deviation of the total posterior from 1.

    Returns
    -------
    pd.DataFrame
        One row per LD group, columns :data:`LD_GROUP_COLS`, ranked by
        ``group_pip`` descending (``ld_group`` = 1 is the top group; ties keep
        input order).  ``patterns`` is the comma-joined list of distinct founder
        patterns in the group.  ``span_bp`` is <NA> without ``pos_mb``.  Empty
        when no variant has a valid Wald statistic.

    Raises
    ------
    ValueError
        If the variant posteriors do not sum to 1 within ``tol``.
    """
    total = float(fm_df["pip"].sum())
    if total == 0.0:
        return pd.DataFrame(columns=LD_GROUP_COLS)
    if abs(total - 1.0) > tol:
        raise ValueError(f"variant PIPs sum to {total:.12g}, expected 1 within {tol:g}")

    by_group = fm_df.groupby("ld_group", sort=False)
    out = by_group.agg(
        group_pip=("pip", "sum"),
        n_variants=("snp", "size"),
        n_patterns=("pattern", "nunique"),
        best_lod=("lod", "max"),
        in_cs=("in_cs", "max"),
    )
    out["patterns"] = by_group["pattern"].agg(lambda p: ",".join(dict.fromkeys(p)))
    out["best_lod_variant"] = fm_df.loc[by_group["lod"].idxmax(), ["ld_group", "snp"]].set_index(
        "ld_group"
    )["snp"]
    if "pos_mb" in fm_df.columns:
        spans = by_group["pos_mb"].agg(lambda p: p.max() - p.min())
        out["span_bp"] = (spans * 1e6).round().astype("Int64")
    else:
        out["span_bp"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out = out.sort_values("group_pip", ascending=False, kind="stable").reset_index(drop=True)
    out["ld_group"] = np.arange(1, len(out) + 1)
    return out[LD_GROUP_COLS]
