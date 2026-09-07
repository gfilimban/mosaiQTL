"""Unit tests for io/annotation_io.py — load_exons_from_gtf GTF parsing."""

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.io.annotation_io import load_exons_from_gtf

# ── minimal GENCODE-style GTF fixture ──────────────────────────────────────────
# Mirrors real GENCODE attribute formatting: 'gene_name' (not 'gene_symbol')
# carries the display symbol; 'tag "Ensembl_canonical"' marks the canonical
# transcript. Two genes on chr1, one canonical and one not, to exercise both
# the gene_name→gene_symbol remap and the canonical_only filter together.
_GTF_LINES = [
    "##description: test fixture\n",
    "chr1\tHAVANA\tgene\t1000000\t1005000\t.\t+\t.\t"
    'gene_id "ENSMUSG00000000001.1"; gene_name "Foo1";\n',
    "chr1\tHAVANA\ttranscript\t1000000\t1005000\t.\t+\t.\t"
    'gene_id "ENSMUSG00000000001.1"; transcript_id "ENSMUST00000000001.1"; '
    'gene_name "Foo1"; tag "basic"; tag "Ensembl_canonical";\n',
    "chr1\tHAVANA\texon\t1000000\t1001000\t.\t+\t.\t"
    'gene_id "ENSMUSG00000000001.1"; transcript_id "ENSMUST00000000001.1"; '
    'gene_name "Foo1"; tag "basic"; tag "Ensembl_canonical";\n',
    "chr1\tHAVANA\tCDS\t1000200\t1000800\t.\t+\t.\t"
    'gene_id "ENSMUSG00000000001.1"; transcript_id "ENSMUST00000000001.1"; '
    'gene_name "Foo1"; tag "basic"; tag "Ensembl_canonical";\n',
    # second, non-canonical transcript of the same gene — dropped when
    # canonical_only=True (the default)
    "chr1\tHAVANA\ttranscript\t1000000\t1006000\t.\t+\t.\t"
    'gene_id "ENSMUSG00000000001.1"; transcript_id "ENSMUST00000000002.1"; '
    'gene_name "Foo1"; tag "basic";\n',
    # unrelated gene, well outside the query window
    "chr1\tHAVANA\tgene\t9000000\t9005000\t.\t-\t.\t"
    'gene_id "ENSMUSG00000000002.1"; gene_name "Bar1";\n',
]


@pytest.fixture()
def gtf_path(tmp_path):
    """Write the fixture GTF (gzipped, as real GENCODE files ship) to disk."""
    path = tmp_path / "test.gtf.gz"
    with gzip.open(path, "wt") as fh:
        fh.writelines(_GTF_LINES)
    return path


def test_gene_symbol_reads_gene_name_attribute(gtf_path):
    """GTF attribute key is 'gene_name', not 'gene_symbol' — must be remapped.

    Regression test: load_exons_from_gtf previously read attrs.get('gene_symbol'),
    which is never a real GTF attribute key, so gene_symbol always came back as
    an empty string. That silently broke every downstream per-gene exon lookup
    in viz/genome.py's _gene_track (dict keyed by gene_symbol), producing plain
    backbone+chevron rendering with no exon/CDS blocks regardless of a valid
    --gtf path.
    """
    df = load_exons_from_gtf(gtf_path, "1", 1.0, 1.01)
    assert not df.empty
    assert set(df["gene_symbol"]) == {"Foo1"}
    assert "" not in set(df["gene_symbol"])


def test_chrom_prefix_normalization(gtf_path):
    """Bare '1' and 'chr1' both match the GTF's 'chr1' column."""
    df_bare = load_exons_from_gtf(gtf_path, "1", 1.0, 1.01)
    df_pref = load_exons_from_gtf(gtf_path, "chr1", 1.0, 1.01)
    assert not df_bare.empty
    assert sorted(df_bare["feature"]) == sorted(df_pref["feature"])


def test_window_outside_any_feature_returns_empty(gtf_path):
    """A window with no overlapping features returns an empty, correctly-columned frame."""
    df = load_exons_from_gtf(gtf_path, "1", 5.0, 5.01)
    assert df.empty
    assert list(df.columns) == [
        "gene_symbol",
        "gene_id",
        "feature",
        "start_mb",
        "end_mb",
        "strand",
    ]


def test_canonical_only_drops_noncanonical_transcript(gtf_path):
    """canonical_only=True (default) keeps only the Ensembl_canonical transcript."""
    df = load_exons_from_gtf(gtf_path, "1", 1.0, 1.01, canonical_only=True)
    tx = df[df["feature"] == "transcript"]
    assert len(tx) == 1
    assert float(tx.iloc[0]["end_mb"]) == pytest.approx(1.005)


def test_canonical_only_false_keeps_all_transcripts(gtf_path):
    """canonical_only=False keeps every transcript, not just the tagged canonical one."""
    df = load_exons_from_gtf(gtf_path, "1", 1.0, 1.01, canonical_only=False)
    tx = df[df["feature"] == "transcript"]
    assert len(tx) == 2


def test_cds_and_exon_features_present(gtf_path):
    """Both exon and CDS rows are returned for a gene with coding structure."""
    df = load_exons_from_gtf(gtf_path, "1", 1.0, 1.01)
    assert {"exon", "CDS"}.issubset(set(df["feature"]))
