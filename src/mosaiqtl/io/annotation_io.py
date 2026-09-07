"""gene annotation loading — exon/CDS intervals for drawing gene tracks in plots."""

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
    """parse a GTF attribute string (key "value" pairs) into a dict."""
    return dict(_ATTR_RE.findall(attr_str))


def load_exons_from_gtf(
    gtf_path: Union[str, Path],
    chrom: str,
    start_mb: float,
    end_mb: float,
    *,
    canonical_only: bool = True,
) -> pd.DataFrame:
    """load exon and CDS intervals for all genes overlapping a genomic window.

    reads the annotation file line by line rather than loading it fully into
    memory. when *canonical_only* is True (default), only the one canonical
    transcript per gene is kept, rather than every alternative transcript.

    ``chrom`` accepts either ``'8'`` or ``'chr8'`` regardless of which
    convention the caller uses elsewhere.

    parameters
    ----------
    gtf_path : str or Path
        path to ``.gtf`` or ``.gtf.gz`` annotation file.
    chrom : str
        chromosome identifier, with or without ``'chr'`` prefix.
    start_mb : float
        window start in Mb (inclusive).
    end_mb : float
        window end in Mb (inclusive).
    canonical_only : bool
        keep only the canonical transcript per gene (default True).

    returns
    -------
    pd.DataFrame
        columns: ``gene_symbol``, ``gene_id``, ``feature``, ``start_mb``,
        ``end_mb``, ``strand``. ``feature`` is one of ``'transcript'``,
        ``'exon'``, ``'CDS'``, or ``'UTR'``. ``transcript`` rows give each
        gene's full-length backbone; ``exon`` rows span the full exon
        (coding + untranslated); ``CDS`` rows mark just the coding portion;
        ``UTR`` rows mark the non-coding flanks. rows are not clipped to the
        window — full intervals are returned so callers can render partial
        overlaps correctly.
    """
    gtf_chrom = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    start_bp = int(start_mb * 1e6)
    end_bp = int(end_mb * 1e6)

    rows: list[dict] = []
    path = Path(gtf_path)
    opener = gzip.open if path.suffix == ".gz" else open

    # collect transcript records regardless of canonical_only, so the full
    # gene backbone extent is always available; filtering happens afterward
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
                # GTF stores the display symbol under the 'gene_name' attribute key
                # (GENCODE convention); 'gene_symbol' is not a GTF attribute at all
                # and always parsed as empty, silently disabling every downstream
                # per-gene exon lookup.
                "gene_symbol": attrs.get("gene_name", ""),
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

    # add the transcript backbone row for any gene that has exon/CDS/UTR
    # rows, so plots can show the full transcript extent even where the
    # outermost exons fall short of the annotated start/end
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
