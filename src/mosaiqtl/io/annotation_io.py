"""GTF annotation I/O — exon/CDS interval loading for gene-track visualisation."""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Union

import pandas as pd

log = logging.getLogger(__name__)

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def _parse_attrs(attr_str: str) -> dict[str, str]:
    """Parse GTF attribute string into a dict."""
    return dict(_ATTR_RE.findall(attr_str))


def load_exons_from_gtf(
    gtf_path: Union[str, Path],
    chrom: str,
    start_mb: float,
    end_mb: float,
    *,
    canonical_only: bool = True,
) -> pd.DataFrame:
    """Load exon and CDS intervals for all genes overlapping a genomic window.

    Streams the GTF without loading it fully into memory.  When
    *canonical_only* is True (default), only exon records tagged
    ``Ensembl_canonical`` are kept — one transcript per gene.

    ``chrom`` is normalised internally so callers may pass either ``'8'`` or
    ``'chr8'`` regardless of the current internal chromosome convention.

    Parameters
    ----------
    gtf_path : str or Path
        Path to ``.gtf`` or ``.gtf.gz`` annotation file.
    chrom : str
        Chromosome identifier, with or without ``'chr'`` prefix.
    start_mb : float
        Window start in Mb (inclusive).
    end_mb : float
        Window end in Mb (inclusive).
    canonical_only : bool
        Keep only records tagged ``Ensembl_canonical`` (default True).

    Returns
    -------
    pd.DataFrame
        Columns: ``gene_symbol``, ``gene_id``, ``feature``, ``start_mb``,
        ``end_mb``, ``strand``.  ``feature`` is one of ``'transcript'``,
        ``'exon'``, ``'CDS'``, or ``'UTR'``.  ``transcript`` rows give the
        authoritative full-transcript backbone extent; ``exon`` rows span the
        full exon (UTR + CDS); ``CDS`` rows mark the coding portion; ``UTR``
        rows mark the non-coding flanks.  Rows are not clipped to the window —
        full intervals are returned so callers can render partial overlaps
        correctly.
    """
    gtf_chrom = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    start_bp = int(start_mb * 1e6)
    end_bp = int(end_mb * 1e6)

    rows: list[dict] = []
    path = Path(gtf_path)
    opener = gzip.open if path.suffix == ".gz" else open

    # transcript records are collected regardless of canonical_only so we
    # always have the authoritative backbone extent; they are filtered to
    # canonical at the gene level after collection.
    _transcript_rows: list[dict] = []

    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[0] != gtf_chrom:
                continue
            feature = fields[2]
            if feature not in ("exon", "CDS", "UTR", "transcript"):
                continue
            feat_start = int(fields[3])
            feat_end = int(fields[4])
            # skip features completely outside the window
            if feat_end < start_bp or feat_start > end_bp:
                continue
            is_canonical = "Ensembl_canonical" in fields[8]
            if canonical_only and not is_canonical:
                continue
            attrs = _parse_attrs(fields[8])
            row = {
                "gene_symbol": attrs.get("gene_symbol", ""),
                "gene_id": attrs.get("gene_id", ""),
                "feature": feature,
                "start_mb": feat_start / 1e6,
                "end_mb": feat_end / 1e6,
                "strand": fields[6],
            }
            if feature == "transcript":
                _transcript_rows.append(row)
            else:
                rows.append(row)

    # For genes that have exon/CDS/UTR rows, inject their transcript backbone.
    # This gives the renderer an authoritative full-transcript extent even when
    # the outermost exons don't reach the annotated transcript start/end.
    if rows and _transcript_rows:
        genes_with_data = {r["gene_symbol"] for r in rows}
        for tr in _transcript_rows:
            if tr["gene_symbol"] in genes_with_data:
                rows.append(tr)

    if not rows:
        log.debug(
            "load_exons_from_gtf: no records found for %s:%g–%g Mb",
            gtf_chrom,
            start_mb,
            end_mb,
        )
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "gene_id",
                "feature",
                "start_mb",
                "end_mb",
                "strand",
            ],
        )

    df = pd.DataFrame(rows)
    log.debug(
        "load_exons_from_gtf: %d records (exon/CDS/UTR/transcript) for %d genes in %s:%g–%g Mb",
        len(df),
        df["gene_symbol"].nunique(),
        gtf_chrom,
        start_mb,
        end_mb,
    )
    return df
