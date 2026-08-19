"""Unit tests for io/results_io.py — pos_merge helper.

Covers module: src/mosaiqtl/io/results_io.py
Function:      pos_merge(left, right, left_val, right_val, ...)

Design goals tested:
  - Happy-path inner join on (chr, pos_mb)
  - Deduplication to highest-value row per (chr, pos) before join
  - Column naming: same-name → _left/_right suffixes; explicit labels win;
    different names → no suffix
  - round_dp controls join precision
  - chr coerced to str regardless of dtype in input
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.io.results_io import pos_merge

# ── helpers ───────────────────────────────────────────────────────────────────


def _df(chr_vals, pos_vals, col_name, col_vals):
    """Build a minimal DataFrame suitable for pos_merge."""
    return pd.DataFrame(
        {
            "chr": chr_vals,
            "pos_mb": pos_vals,
            col_name: col_vals,
        }
    )


# ── happy path ────────────────────────────────────────────────────────────────


class TestBasicJoin:
    def test_basic_join(self):
        """Rows sharing (chr, pos_mb) appear in the result."""
        left = _df(["1"], [10.123], "lod_py", [5.0])
        right = _df(["1"], [10.123], "lod_r", [4.8])
        result = pos_merge(left, right, "lod_py", "lod_r")
        assert len(result) == 1
        np.testing.assert_allclose(result["lod_py"].iloc[0], 5.0, rtol=1e-6)
        np.testing.assert_allclose(result["lod_r"].iloc[0], 4.8, rtol=1e-6)

    def test_inner_join_no_unmatched(self):
        """Rows not matching on (chr, pos_mb) are absent from the result."""
        left = _df(["1", "2"], [10.0, 20.0], "lod", [3.0, 4.0])
        right = _df(["1"], [10.0], "lod", [2.9])
        result = pos_merge(left, right, "lod", "lod")
        # only chr=1 pos=10.0 matches; chr=2 pos=20.0 is absent
        assert len(result) == 1
        assert result["_chr"].iloc[0] == "1"


# ── column naming ─────────────────────────────────────────────────────────────


class TestColumnNaming:
    def test_same_val_name_suffixes(self):
        """left_val == right_val and no labels → _left/_right suffixes."""
        left = _df(["1"], [5.0], "lod", [3.0])
        right = _df(["1"], [5.0], "lod", [2.8])
        result = pos_merge(left, right, "lod", "lod")
        assert "lod_left" in result.columns
        assert "lod_right" in result.columns
        assert "lod" not in result.columns

    def test_explicit_labels(self):
        """left_label / right_label override automatic naming."""
        left = _df(["1"], [5.0], "lod", [3.0])
        right = _df(["1"], [5.0], "lod", [2.8])
        result = pos_merge(left, right, "lod", "lod", left_label="py", right_label="r")
        assert "py" in result.columns
        assert "r" in result.columns
        assert "lod_left" not in result.columns
        assert "lod_right" not in result.columns

    def test_different_val_names_no_suffix(self):
        """left_val != right_val and no labels → columns keep original names."""
        left = _df(["1"], [5.0], "lod_mediated", [3.0])
        right = _df(["1"], [5.0], "lod", [2.8])
        result = pos_merge(left, right, "lod_mediated", "lod")
        assert "lod_mediated" in result.columns
        assert "lod" in result.columns
        assert "lod_mediated_left" not in result.columns
        assert "lod_right" not in result.columns


# ── deduplication ─────────────────────────────────────────────────────────────


class TestDeduplication:
    def test_dedup_keeps_highest(self):
        """Multiple rows at same (chr, pos) in left: highest value wins."""
        left = pd.DataFrame(
            {
                "chr": ["1", "1"],
                "pos_mb": [10.0, 10.0],
                "lod": [2.0, 7.5],  # 7.5 should be kept
            }
        )
        right = _df(["1"], [10.0], "lod", [6.0])
        result = pos_merge(left, right, "lod", "lod")
        assert len(result) == 1
        np.testing.assert_allclose(result["lod_left"].iloc[0], 7.5, rtol=1e-6)


# ── rounding precision ────────────────────────────────────────────────────────


class TestRoundDp:
    def test_round_dp_joins_at_coarse_precision(self):
        """Two positions that round to the same value at dp=2 do join.

        10.001 rounds to 10.0 at dp=2; 10.004 also rounds to 10.0 at dp=2.
        They are distinct at dp=3 (10.001 vs 10.004), so they should NOT join
        at dp=3 but SHOULD join at dp=2.
        """
        left = _df(["1"], [10.001], "lod", [3.0])
        right = _df(["1"], [10.004], "lod", [2.9])
        result = pos_merge(left, right, "lod", "lod", round_dp=2)
        assert len(result) == 1

    def test_round_dp_no_join_at_fine_precision(self):
        """Same two positions do NOT join at dp=3 (10.001 != 10.004 at 3dp)."""
        left = _df(["1"], [10.001], "lod", [3.0])
        right = _df(["1"], [10.004], "lod", [2.9])
        result = pos_merge(left, right, "lod", "lod", round_dp=3)
        assert len(result) == 0


# ── chr dtype coercion ────────────────────────────────────────────────────────


class TestChrAsStr:
    def test_chr_int_in_one_df_still_joins(self):
        """chr as int in left, str in right — both coerced to str; join succeeds."""
        left = pd.DataFrame(
            {
                "chr": [1],  # integer dtype
                "pos_mb": [20.0],
                "lod": [4.0],
            }
        )
        right = _df(["1"], [20.0], "lod", [3.8])
        result = pos_merge(left, right, "lod", "lod")
        assert len(result) == 1
        # _chr column should be string '1'
        assert result["_chr"].iloc[0] == "1"
