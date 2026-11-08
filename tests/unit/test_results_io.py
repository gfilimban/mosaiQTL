"""Unit tests for io/results_io.py.

Covers module: src/mosaiqtl/io/results_io.py

Functions:
  pos_merge(left, right, left_val, right_val, ...)
  build_eqtl_scan_df(scan, genomecache, ...)
  write_eqtl_scan(scan, genomecache, out_path, ...)
  build_eqtl_finemapping_df(fm_df, gene_id, chr_, ...)
  write_eqtl_finemapping(...)
  build_eqtl_finemapping_peak_df(fm_df, ...)
  write_eqtl_finemapping_peak(...)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.io.results_io import (
    build_eqtl_finemapping_df,
    build_eqtl_finemapping_peak_df,
    pos_merge,
    write_eqtl_finemapping,
    write_eqtl_finemapping_peak,
)
from mosaiqtl.pipeline import (
    _FOUNDER_EFFECT_COLS,
    EQTL_FINEMAPPING_COLS,
    EQTL_FINEMAPPING_PEAK_COLS,
)

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


# ── build_eqtl_scan_df / write_eqtl_scan ─────────────────────────────────────
#
# These tests use the tiny_cache fixture (2 chromosomes, 15 loci each) and
# fabricated ScanResult objects to stay self-contained and fast.


class TestBuildEqtlScanDf:
    """build_eqtl_scan_df: EQTL_SCAN_COLS schema, peak_locus flag, NaN policy."""

    @staticmethod
    def _make_sr(lod, loci, pos_mb, chrom="1"):
        """Build a minimal ScanResult for tests."""
        from mosaiqtl.models.records import ScanResult

        return ScanResult(
            gene_id="g1",
            lod=np.asarray(lod, dtype=float),
            loci=loci,
            pos_mb=np.asarray(pos_mb, dtype=float),
            chr=chrom,
        )

    def test_columns_match_schema(self, tiny_cache):
        """Output columns are exactly EQTL_SCAN_COLS, in schema order."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df
        from mosaiqtl.pipeline import EQTL_SCAN_COLS

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr = self._make_sr(
            lod=[1.0] * len(lids),
            loci=list(lids),
            pos_mb=list(pos),
            chrom="1",
        )
        df = build_eqtl_scan_df({"g1": sr}, tiny_cache)
        assert list(df.columns) == EQTL_SCAN_COLS

    def test_row_count(self, tiny_cache):
        """One row per scanned locus per gene."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr = self._make_sr(
            lod=[1.0] * len(lids),
            loci=list(lids),
            pos_mb=list(pos),
        )
        df = build_eqtl_scan_df({"g1": sr}, tiny_cache)
        assert len(df) == len(lids)

    def test_peak_locus_flag(self, tiny_cache):
        """Exactly one 'T' per gene at the argmax-LOD locus; all others 'F'."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df

        lids, pos, _ = tiny_cache.get_chromosome("1")
        lod_vals = [float(i) for i in range(len(lids))]  # peak at last locus
        sr = self._make_sr(lod=lod_vals, loci=list(lids), pos_mb=list(pos))
        df = build_eqtl_scan_df({"g1": sr}, tiny_cache)

        assert (df["peak_locus"] == "T").sum() == 1
        assert (df["peak_locus"] == "F").sum() == len(lids) - 1
        # the 'T' row should have the highest lod
        peak_row = df[df["peak_locus"] == "T"].iloc[0]
        assert float(peak_row["lod"]) == max(lod_vals)

    def test_sanger_cols_absent_from_scan(self, tiny_cache):
        """Sanger allele columns are not part of EQTL_SCAN_COLS schema."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df, _SANGER_COLS

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr = self._make_sr(lod=[1.0] * len(lids), loci=list(lids), pos_mb=list(pos))
        df = build_eqtl_scan_df({"g1": sr}, tiny_cache)
        for col in _SANGER_COLS:
            assert col not in df.columns, f"Sanger col {col!r} should not appear in scan output"

    def test_pval_nominal_decreases_with_lod(self, tiny_cache):
        """pval_nominal is a monotone decreasing function of LOD."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df

        lids, pos, _ = tiny_cache.get_chromosome("1")
        # two-row toy with known LOD ordering
        sr = self._make_sr(
            lod=[0.5, 2.0],
            loci=list(lids[:2]),
            pos_mb=list(pos[:2]),
        )
        df = build_eqtl_scan_df({"g1": sr}, tiny_cache)
        df_sorted = df.sort_values("lod")
        pvals = df_sorted["pval_nominal"].values
        assert pvals[0] > pvals[1], "higher LOD should give smaller p-value"

    def test_gene_symbol_map_applied(self, tiny_cache):
        """gene_symbol_map is used when supplied."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr = self._make_sr(lod=[1.0], loci=[lids[0]], pos_mb=[pos[0]])
        df = build_eqtl_scan_df(
            {"g1": sr}, tiny_cache, gene_symbol_map={"g1": "Ankrd39"}
        )
        assert (df["gene_symbol"] == "Ankrd39").all()

    def test_empty_scan_returns_schema_columns(self, tiny_cache):
        """An empty scan dict returns an empty DataFrame with EQTL_SCAN_COLS."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df
        from mosaiqtl.pipeline import EQTL_SCAN_COLS

        df = build_eqtl_scan_df({}, tiny_cache)
        assert list(df.columns) == EQTL_SCAN_COLS
        assert len(df) == 0

    def test_multi_gene_row_count(self, tiny_cache):
        """Rows from multiple genes are concatenated (total = sum of per-gene loci)."""
        from mosaiqtl.io.results_io import build_eqtl_scan_df

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr1 = self._make_sr(lod=[1.0] * 3, loci=list(lids[:3]), pos_mb=list(pos[:3]))
        sr2_obj = self._make_sr(
            lod=[0.5] * 2, loci=list(lids[:2]), pos_mb=list(pos[:2])
        )
        sr2_obj = type(sr1)(
            gene_id="g2",
            lod=np.array([0.5, 0.5]),
            loci=list(lids[:2]),
            pos_mb=list(pos[:2]),
            chr="1",
        )
        df = build_eqtl_scan_df({"g1": sr1, "g2": sr2_obj}, tiny_cache)
        assert len(df) == 5
        assert set(df["gene_id"].unique()) == {"g1", "g2"}


class TestWriteEqtlScan:
    """write_eqtl_scan: atomically writes EQTL_SCAN_COLS CSV."""

    def test_writes_csv_file(self, tmp_path, tiny_cache):
        """Output file exists and is non-empty after a successful write."""
        from mosaiqtl.io.results_io import write_eqtl_scan
        from mosaiqtl.models.records import ScanResult

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr = ScanResult(
            gene_id="g1",
            lod=np.ones(len(lids)),
            loci=list(lids),
            pos_mb=pos,
            chr="1",
        )
        out = tmp_path / "eqtl_scan.csv"
        write_eqtl_scan({"g1": sr}, tiny_cache, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_roundtrip_columns(self, tmp_path, tiny_cache):
        """CSV read back has exactly EQTL_SCAN_COLS columns."""
        from mosaiqtl.io.results_io import write_eqtl_scan
        from mosaiqtl.models.records import ScanResult
        from mosaiqtl.pipeline import EQTL_SCAN_COLS

        lids, pos, _ = tiny_cache.get_chromosome("1")
        sr = ScanResult(
            gene_id="g1",
            lod=np.ones(len(lids)),
            loci=list(lids),
            pos_mb=pos,
            chr="1",
        )
        out = tmp_path / "eqtl_scan.csv"
        write_eqtl_scan({"g1": sr}, tiny_cache, out)
        reloaded = pd.read_csv(out)
        assert list(reloaded.columns) == EQTL_SCAN_COLS


# ── finemapping writer fixtures ───────────────────────────────────────────────


def _fm_df(
    n_cs: int = 4, n_non_cs: int = 2, include_pos_mb: bool = True
) -> pd.DataFrame:
    """Minimal fine_map_region-style DataFrame with in_cs column."""
    rng = np.random.default_rng(42)
    n = n_cs + n_non_cs
    pip = rng.dirichlet(np.ones(n))
    # sort descending so first CS row has highest PIP (mirrors fine_map_region)
    order = np.argsort(-pip)
    pip = pip[order]
    in_cs = np.zeros(n, dtype=bool)
    in_cs[:n_cs] = True
    df: dict = {
        "snp": [f"rs{i}" for i in range(n)],
        "lod": rng.uniform(1.0, 6.0, n),
        "z": rng.normal(0, 2, n),
        "abf": rng.uniform(0.5, 3.0, n),
        "pip": pip,
        "in_cs": in_cs,
    }
    if include_pos_mb:
        df["pos_mb"] = np.linspace(100.0, 100.5, n)
    return pd.DataFrame(df)


# ── build_eqtl_finemapping_df ─────────────────────────────────────────────────


class TestBuildEqtlFinemappingDf:
    def test_output_columns(self):
        df = build_eqtl_finemapping_df(
            _fm_df(), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        # intermediate df carries EQTL_FINEMAPPING_COLS + _FOUNDER_EFFECT_COLS so
        # build_eqtl_finemapping_peak_df can lift founder betas into the peak file.
        # write_eqtl_finemapping trims to EQTL_FINEMAPPING_COLS at write time.
        assert list(df.columns) == EQTL_FINEMAPPING_COLS + _FOUNDER_EFFECT_COLS

    def test_only_cs_members_included(self):
        df = build_eqtl_finemapping_df(
            _fm_df(n_cs=3, n_non_cs=5), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        assert len(df) == 3

    def test_lead_snp_marks_first_row(self):
        df = build_eqtl_finemapping_df(
            _fm_df(n_cs=4), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        assert df["lead_snp"].iloc[0] == "T"
        assert (df["lead_snp"].iloc[1:] == "F").all()

    def test_hap_peak_fields_populated(self):
        df = build_eqtl_finemapping_df(
            _fm_df(),
            "ENSMUSG0001",
            "8",
            100_200_000,
            "UNC1234",
            5.2,
            ci_lo_bp=99_000_000,
            ci_hi_bp=101_000_000,
        )
        assert int(df["hap_peak_pos"].iloc[0]) == 100_200_000
        assert df["hap_peak_variant"].iloc[0] == "UNC1234"
        assert float(df["hap_peak_lod"].iloc[0]) == pytest.approx(5.2)
        assert int(df["ci_lo_bp"].iloc[0]) == 99_000_000
        assert int(df["ci_hi_bp"].iloc[0]) == 101_000_000

    def test_pip_method_and_cs_flag(self):
        df = build_eqtl_finemapping_df(
            _fm_df(), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        assert (df["pip_method"] == "ABF").all()
        assert (df["CS"] == "T").all()

    def test_pos_column_when_pos_mb_present(self):
        df = build_eqtl_finemapping_df(
            _fm_df(include_pos_mb=True), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        assert df["pos"].notna().all()

    def test_pos_column_nan_when_pos_mb_absent(self):
        df = build_eqtl_finemapping_df(
            _fm_df(include_pos_mb=False),
            "ENSMUSG0001",
            "8",
            100_200_000,
            "UNC1234",
            5.2,
        )
        assert df["pos"].isna().all()

    def test_alleles_df_populates_sanger_cols(self):
        fm = _fm_df(n_cs=2)
        snps = fm.loc[fm["in_cs"], "snp"].tolist()
        alleles = pd.DataFrame(
            {
                "snp": snps,
                "ref": ["A", "C"],
                "alt": ["T", "G"],
            }
        )
        df = build_eqtl_finemapping_df(
            fm, "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2, alleles_df=alleles
        )
        assert df["ref"].notna().all()
        assert df["alt"].notna().all()

    def test_no_alleles_df_sanger_cols_nan(self):
        df = build_eqtl_finemapping_df(
            _fm_df(), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        assert df["ref"].isna().all()
        assert df["alt"].isna().all()

    def test_gene_id_and_symbol_populated(self):
        df = build_eqtl_finemapping_df(
            _fm_df(),
            "ENSMUSG0001",
            "8",
            100_200_000,
            "UNC1234",
            5.2,
            gene_symbol="Piezo1",
        )
        assert (df["gene_id"] == "ENSMUSG0001").all()
        assert (df["gene_symbol"] == "Piezo1").all()


# ── write_eqtl_finemapping ────────────────────────────────────────────────────


class TestWriteEqtlFinemapping:
    def test_file_created_and_schema_correct(self, tmp_path):
        out = tmp_path / "finemapping.csv"
        write_eqtl_finemapping(
            _fm_df(), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2, out
        )
        assert out.exists()
        assert list(pd.read_csv(out).columns) == EQTL_FINEMAPPING_COLS


# ── build_eqtl_finemapping_peak_df ───────────────────────────────────────────


class TestBuildEqtlFinemappingPeakDf:
    @pytest.fixture
    def fm_built(self):
        return build_eqtl_finemapping_df(
            _fm_df(n_cs=3),
            "ENSMUSG0001",
            "8",
            100_200_000,
            "UNC1234",
            5.2,
            gene_symbol="Piezo1",
        )

    def test_output_columns(self, fm_built):
        df = build_eqtl_finemapping_peak_df(fm_built)
        assert list(df.columns) == EQTL_FINEMAPPING_PEAK_COLS

    def test_single_row(self, fm_built):
        df = build_eqtl_finemapping_peak_df(fm_built)
        assert len(df) == 1

    def test_lead_snp_column_dropped(self, fm_built):
        df = build_eqtl_finemapping_peak_df(fm_built)
        assert "lead_snp" not in df.columns

    def test_founder_effects_populated(self):
        effects = {
            f"effect_{f}": float(i)
            for i, f in enumerate(
                ["AJ", "B6J", "129S1J", "NODJ", "NZOJ", "CASTJ", "PWKJ", "WSBJ"]
            )
        }
        fm = build_eqtl_finemapping_df(
            _fm_df(n_cs=3), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2,
            founder_effects=effects,
        )
        df = build_eqtl_finemapping_peak_df(fm)
        assert float(df["effect_AJ"].iloc[0]) == pytest.approx(0.0)
        assert float(df["effect_B6J"].iloc[0]) == pytest.approx(1.0)

    def test_founder_effects_nan_when_none(self, fm_built):
        # fm_built built without founder_effects → NaN
        df = build_eqtl_finemapping_peak_df(fm_built)
        assert df["effect_AJ"].isna().all()

    def test_sig_values_populated(self):
        sig = {
            "pval_perm": 0.01,
            "fdr": 0.03,
            "fdr_method": "qvalue",
            "significance": "T",
        }
        fm = build_eqtl_finemapping_df(
            _fm_df(n_cs=3), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2,
            sig_values=sig,
        )
        df = build_eqtl_finemapping_peak_df(fm)
        assert float(df["fdr"].iloc[0]) == pytest.approx(0.03)
        assert df["fdr_method"].iloc[0] == "qvalue"

    def test_sig_values_nan_when_none(self, fm_built):
        # fm_built built without sig_values → NaN
        df = build_eqtl_finemapping_peak_df(fm_built)
        assert df["pval_perm"].isna().all()


# ── write_eqtl_finemapping_peak ───────────────────────────────────────────────


class TestWriteEqtlFinemappingPeak:
    def test_file_created_and_schema_correct(self, tmp_path):
        fm_built = build_eqtl_finemapping_df(
            _fm_df(n_cs=2), "ENSMUSG0001", "8", 100_200_000, "UNC1234", 5.2
        )
        out = tmp_path / "finemapping_peak.csv"
        write_eqtl_finemapping_peak(fm_built, out)
        assert out.exists()
        assert list(pd.read_csv(out).columns) == EQTL_FINEMAPPING_PEAK_COLS
