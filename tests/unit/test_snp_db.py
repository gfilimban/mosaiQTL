"""Unit tests for genotype/snp_db.py — build_founder_allele_gm and load_founder_alleles."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.snp_db import build_founder_allele_gm, load_founder_alleles
from mosaiqtl.models.config import Population

# ── minimal test fixtures ─────────────────────────────────────────────────────

_FOUNDERS_4 = ["FA", "FB", "FC", "FD"]
_COLORS_4 = {"FA": "#111111", "FB": "#222222", "FC": "#333333", "FD": "#444444"}

_POP4 = Population(
    name="test4",
    founders=_FOUNDERS_4,
    colors=_COLORS_4,
)


def _make_cache(chroms=("1",), n_loci=5, n_strains=5, founders=None):
    """Minimal GenomeCache for snp_db tests (probs unused — pass dummy).

    n_loci=5 → positions [10, 30, 50, 70, 90] Mb so test SNPs at round Mb
    values hit a locus exactly and pass the default max_dist_mb=0.5 filter.
    """
    founders = founders or _FOUNDERS_4
    n_f = len(founders)
    data = {}
    for c in chroms:
        lids = np.array([f"rs{c}_{i:03d}" for i in range(n_loci)])
        pos = np.linspace(10.0, 90.0, n_loci)  # Mb: [10, 30, 50, 70, 90]
        probs = np.ones((n_loci, n_strains, n_f)) / n_f
        data[c] = {"locus_ids": lids, "positions": pos, "probs": probs}
    sample_ids = np.array([f"S{i}" for i in range(n_strains)])
    return GenomeCache(sample_ids, founders, data)


def _make_gm_csv(tmp_path, founders, rows):
    """Write a minimal founder-genotype CSV.

    rows: list of (marker, chr, pos_bp, allele_per_founder...)
    """
    cols = ["marker", "chr", "pos"] + founders
    df = pd.DataFrame(rows, columns=cols)
    p = tmp_path / "gm.csv"
    df.to_csv(p, index=False)
    return p


# ── build ─────────────────────────────────────────────────────────────────────


class TestBuildFounderAlleleDb:
    def test_creates_hdf5_file(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "A", "T", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        assert out.exists()

    def test_metadata_founders_stored(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "A", "T", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        with h5py.File(out, "r") as hf:
            stored = [
                f.decode() if isinstance(f, bytes) else f
                for f in np.array(hf["metadata"]["founders"])
            ]
        assert stored == _FOUNDERS_4

    def test_chrom_group_created(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "A", "T", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        with h5py.File(out, "r") as hf:
            assert "1" in hf

    def test_alleles_shape_n_founders(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "A", "T", "T"),
                ("snp2", "1", 70_000_000, "C", "G", "C", "G"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        with h5py.File(out, "r") as hf:
            shape = np.array(hf["1"]["alleles"]).shape
        assert shape == (2, 4)

    def test_returns_summary_dict(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "A", "T", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        summary = build_founder_allele_gm(csv, cache, out)
        assert set(summary.keys()) >= {
            "n_raw",
            "n_monomorphic",
            "n_multiallelic",
            "n_too_far",
            "n_written",
        }

    def test_monomorphic_filtered(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp_mono", "1", 50_000_000, "A", "A", "A", "A"),  # monomorphic
                ("snp_bi", "1", 70_000_000, "A", "T", "A", "T"),  # biallelic — kept
            ],
        )
        out = tmp_path / "alleles.h5"
        summary = build_founder_allele_gm(csv, cache, out)
        assert summary["n_monomorphic"] == 1
        assert summary["n_written"] == 1

    def test_multiallelic_filtered(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                (
                    "snp_multi",
                    "1",
                    50_000_000,
                    "A",
                    "C",
                    "G",
                    "T",
                ),  # 4 alleles — dropped
                ("snp_bi", "1", 70_000_000, "A", "T", "A", "T"),  # kept
            ],
        )
        out = tmp_path / "alleles.h5"
        summary = build_founder_allele_gm(csv, cache, out)
        assert summary["n_multiallelic"] == 1
        assert summary["n_written"] == 1

    def test_snp_too_far_filtered(self, tmp_path):
        cache = _make_cache(n_loci=5)  # positions at 10,30,50,70,90 Mb
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                (
                    "snp_close",
                    "1",
                    50_000_000,
                    "A",
                    "T",
                    "A",
                    "T",
                ),  # 50 Mb → 50 Mb locus = 0 dist
                (
                    "snp_far",
                    "1",
                    1_000_000,
                    "A",
                    "T",
                    "A",
                    "T",
                ),  # 1 Mb → nearest 10 Mb = 9 Mb away
            ],
        )
        out = tmp_path / "alleles.h5"
        summary = build_founder_allele_gm(csv, cache, out, max_dist_mb=0.5)
        assert summary["n_too_far"] == 1
        assert summary["n_written"] == 1

    def test_population_founders_used(self, tmp_path):
        cache = _make_cache(founders=_FOUNDERS_4)
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "T", "A", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out, population=_POP4)
        with h5py.File(out, "r") as hf:
            stored = [
                f.decode() if isinstance(f, bytes) else f
                for f in np.array(hf["metadata"]["founders"])
            ]
        assert stored == list(_POP4.founders)

    def test_explicit_founder_cols_override(self, tmp_path):
        # build with explicit override — CSV has different column names
        founders_alt = ["X1", "X2", "X3", "X4"]
        cache = _make_cache(founders=founders_alt)
        csv = _make_gm_csv(
            tmp_path,
            founders_alt,
            [
                ("snp1", "1", 50_000_000, "A", "T", "A", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out, founder_cols=founders_alt)
        with h5py.File(out, "r") as hf:
            stored = [
                f.decode() if isinstance(f, bytes) else f
                for f in np.array(hf["metadata"]["founders"])
            ]
        assert stored == founders_alt

    def test_build_idempotent(self, tmp_path):
        cache = _make_cache()
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "T", "A", "T"),
            ],
        )
        out = tmp_path / "alleles.h5"
        s1 = build_founder_allele_gm(csv, cache, out)
        s2 = build_founder_allele_gm(csv, cache, out)
        assert s1["n_written"] == s2["n_written"]

    def test_two_chromosomes(self, tmp_path):
        cache = _make_cache(chroms=("1", "2"))
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "T", "A", "T"),
                ("snp2", "2", 50_000_000, "C", "G", "C", "G"),
            ],
        )
        out = tmp_path / "alleles.h5"
        summary = build_founder_allele_gm(csv, cache, out)
        assert summary["n_written"] == 2
        assert set(summary["per_chr"].keys()) == {"1", "2"}


# ── load ──────────────────────────────────────────────────────────────────────


class TestLoadFounderAlleles:
    def _build(self, tmp_path, rows=None):
        cache = _make_cache()
        if rows is None:
            rows = [
                ("snp1", "1", 50_000_000, "A", "T", "A", "T"),
                ("snp2", "1", 70_000_000, "C", "G", "C", "G"),
            ]
        csv = _make_gm_csv(tmp_path, _FOUNDERS_4, rows)
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        return out

    def test_returns_dataframe(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        assert isinstance(df, pd.DataFrame)

    def test_standard_columns_present(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        for col in ("marker", "chr", "pos_mb", "pos_bp", "locus_id"):
            assert col in df.columns

    def test_founder_columns_present(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        for f in _FOUNDERS_4:
            assert f in df.columns

    def test_row_count(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        assert len(df) == 2

    def test_chrom_filter(self, tmp_path):
        cache = _make_cache(chroms=("1", "2"))
        csv = _make_gm_csv(
            tmp_path,
            _FOUNDERS_4,
            [
                ("snp1", "1", 50_000_000, "A", "T", "A", "T"),
                ("snp2", "2", 50_000_000, "C", "G", "C", "G"),
            ],
        )
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        df = load_founder_alleles(out, chrom="1")
        assert len(df) == 1
        assert df["chr"].iloc[0] == "1"

    def test_invalid_chrom_raises(self, tmp_path):
        out = self._build(tmp_path)
        with pytest.raises(KeyError):
            load_founder_alleles(out, chrom="chrZ")

    def test_allele_values_are_strings(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        import pandas as pd

        for f in _FOUNDERS_4:
            assert pd.api.types.is_string_dtype(df[f])

    def test_pos_mb_float(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        assert df["pos_mb"].dtype == np.float64 or df["pos_mb"].dtype.kind == "f"

    def test_pos_bp_int(self, tmp_path):
        out = self._build(tmp_path)
        df = load_founder_alleles(out)
        assert df["pos_bp"].dtype.kind in ("i", "u")

    def test_round_trip_allele_values(self, tmp_path):
        cache = _make_cache()
        # biallelic: A/T partition (FA,FB carry A; FC,FD carry T) — 2 distinct alleles
        rows = [("snp1", "1", 50_000_000, "A", "A", "T", "T")]
        csv = _make_gm_csv(tmp_path, _FOUNDERS_4, rows)
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        df = load_founder_alleles(out)
        assert df["FA"].iloc[0] == "A"
        assert df["FB"].iloc[0] == "A"
        assert df["FC"].iloc[0] == "T"
        assert df["FD"].iloc[0] == "T"

    def test_locus_id_linked(self, tmp_path):
        cache = _make_cache()
        rows = [("snp1", "1", 50_000_000, "A", "T", "A", "T")]
        csv = _make_gm_csv(tmp_path, _FOUNDERS_4, rows)
        out = tmp_path / "alleles.h5"
        build_founder_allele_gm(csv, cache, out)
        df = load_founder_alleles(out)
        # locus_id must be a non-empty string matching a cache locus pattern
        lid = df["locus_id"].iloc[0]
        assert isinstance(lid, str) and len(lid) > 0
        assert lid.startswith("rs1_")  # matches _make_cache naming: 'rs{chrom}_{i:03d}'


class TestBuildFounderAlleleMgpCoordinates:
    """VCF POS is 1-based; SPDI positions are 0-based."""

    def test_spdi_is_zero_based_and_pos_bp_is_one_based(self, tmp_path):
        pysam = pytest.importorskip("pysam")
        from mosaiqtl.genotype.snp_db import build_founder_allele_mgp

        vcf = tmp_path / "t.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=200000000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tAJ\tPWK\n"
            "1\t10000001\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t1/1\n"
        )
        gz = tmp_path / "t.vcf.gz"
        pysam.tabix_compress(str(vcf), str(gz), force=True)
        out = tmp_path / "alleles.h5"
        cache = _make_cache(founders=["A", "B"])
        build_founder_allele_mgp(
            gz, cache, out, vcf_sample_map={"AJ": "AJ", "PWK": "PWK"}
        )
        df = load_founder_alleles(out)
        assert df["marker"].iloc[0] == "NC_000067.7:10000000:A:G"
        assert int(df["pos_bp"].iloc[0]) == 10000001
