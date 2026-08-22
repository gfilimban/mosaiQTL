"""results_io — serialisation helpers for QTL scan outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mosaiqtl.models.records import ScanResult


def pos_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_val: str,
    right_val: str,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
    round_dp: int = 3,
) -> pd.DataFrame:
    """Inner-join two scan DataFrames on (chr, rounded pos_mb).

    Deduplicates to the highest-value row per (chr, pos) before joining to
    avoid many-to-many matches when multiple genes share a locus position.

    Parameters
    ----------
    left, right:
        DataFrames that each contain at least ``chr``, ``pos_mb``, and
        the column named by ``left_val`` / ``right_val``.
    left_val, right_val:
        Column names to extract from each DataFrame.
    left_label, right_label:
        Output column names.  When omitted: uses ``left_val`` / ``right_val``
        directly when they differ; adds ``_left`` / ``_right`` suffixes when
        both names are the same.
    round_dp:
        Decimal places to round ``pos_mb`` before joining (default 3).

    Returns
    -------
    DataFrame with columns ``_chr``, ``_pos``, and the two value columns.
    """
    left = left.copy()
    right = right.copy()
    left["_chr"] = left["chr"].astype(str)
    right["_chr"] = right["chr"].astype(str)
    left["_pos"] = left["pos_mb"].round(round_dp)
    right["_pos"] = right["pos_mb"].round(round_dp)

    same_name = (left_val == right_val) and left_label is None and right_label is None
    left_out = left_label or (left_val + ("_left" if same_name else ""))
    right_out = right_label or (right_val + ("_right" if same_name else ""))

    left_d = (
        left[["_chr", "_pos", left_val]]
        .rename(columns={left_val: left_out})
        .sort_values(left_out, ascending=False)
        .drop_duplicates(["_chr", "_pos"])
    )
    right_d = (
        right[["_chr", "_pos", right_val]]
        .rename(columns={right_val: right_out})
        .sort_values(right_out, ascending=False)
        .drop_duplicates(["_chr", "_pos"])
    )
    return pd.merge(left_d, right_d, on=["_chr", "_pos"]).dropna()


def scan_result_to_df(
    scan: "ScanResult",
    gene_id: str,
    gene_symbol: str = "",
) -> pd.DataFrame:
    """Convert a ScanResult to a per-locus DataFrame.

    Parameters
    ----------
    scan:
        Completed scan result from ``scan_h2lmm`` or equivalent.
    gene_id:
        ENSMUSG ID (version-stripped) — stable join key written to the
        ``gene_id`` column.
    gene_symbol:
        Display name (e.g. 'Ankrd39') written to the ``gene_symbol`` column.
        Empty string when not available.

    Returns
    -------
    DataFrame with columns: gene_id, gene_symbol, chr, locus, pos_mb, lod.
    """
    return pd.DataFrame(
        {
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "chr": np.asarray(scan.chromosomes),
            "locus": np.asarray(scan.locus_ids),
            "pos_mb": np.asarray(scan.positions),
            "lod": np.asarray(scan.lod),
        }
    )
