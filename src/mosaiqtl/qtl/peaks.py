"""peaks — peak detection and LOD support intervals for QTL scan results."""

import numpy as np
import pandas as pd

from ..utils import chr_sort_key

__all__ = [
    'grab_locus_from_scan',
    'lod_drop_ci',
    'find_peaks',
]


def grab_locus_from_scan(scan_result):
    """return the locus id with the highest LOD score."""
    return scan_result.locus_ids[np.argmax(scan_result.lod)]


def lod_drop_ci(scan_result, chrom, drop=1.5):
    """confidence interval around the peak on one chromosome, defined as the
    span of loci whose LOD stays within `drop` LOD units of the peak.

    returns (lo_mb, hi_mb, peak_mb).
    """
    mask = scan_result.chromosomes == chrom
    pos = scan_result.positions[mask]
    lod = scan_result.lod[mask]
    if len(lod) == 0:
        return np.nan, np.nan, np.nan
    pi = np.argmax(lod)
    thresh = lod[pi] - drop
    above = np.where(lod >= thresh)[0]
    return float(pos[above[0]]), float(pos[above[-1]]), float(pos[pi])


def find_peaks(scan_result, threshold, drop=1.5, allow_multiple=False, min_lod_gap=1.5):
    """find significant QTL peaks in a genome-wide scan.

    parameters
    ----------
    scan_result   : ScanResult
    threshold     : float  minimum LOD to report a peak
    drop          : float  LOD drop for CI bounds (default 1.5)
    allow_multiple: bool   if True, report secondary peaks per chromosome
        that are above threshold AND at least min_lod_gap LOD below the
        primary peak (i.e. separated local maxima).  default False returns
        only the highest-LOD peak per chromosome.
    min_lod_gap   : float  minimum LOD separation between multiple peaks
        on the same chromosome (only used when allow_multiple=True)

    returns
    -------
    pd.DataFrame  columns: phenotype_id, chr, locus_id, pos_mb, lod,
                            ci_lo_mb, ci_hi_mb
        one row per peak, sorted by chr then pos_mb.
        empty DataFrame (same columns) when no locus exceeds threshold.
    """
    from .scan import ScanResult  # noqa: PLC0415

    _COLS = ("phenotype_id", "chr", "locus_id", "pos_mb", "lod", "ci_lo_mb", "ci_hi_mb")
    rows = []

    for chrom in sorted(np.unique(scan_result.chromosomes), key=chr_sort_key):
        mask = scan_result.chromosomes == chrom
        lod_c = scan_result.lod[mask]
        pos_c = scan_result.positions[mask]
        lids_c = scan_result.locus_ids[mask]

        if lod_c.max() < threshold:
            continue

        if not allow_multiple:
            # single best peak per chromosome
            pi = int(np.argmax(lod_c))
            lo, hi, pk = lod_drop_ci(scan_result, chrom, drop)
            rows.append(
                (
                    scan_result.phenotype_id,
                    chrom,
                    lids_c[pi],
                    float(pos_c[pi]),
                    float(lod_c[pi]),
                    lo,
                    hi,
                )
            )
        else:
            # all local maxima above threshold separated by >= min_lod_gap
            # a point is a local max if it is >= both neighbours
            n = len(lod_c)
            for i in range(n):
                if lod_c[i] < threshold:
                    continue
                left_ok = (i == 0) or (lod_c[i] >= lod_c[i - 1])
                right_ok = (i == n - 1) or (lod_c[i] >= lod_c[i + 1])
                if not (left_ok and right_ok):
                    continue
                # check separation from any already-accepted peak on this chr
                too_close = any(
                    abs(pos_c[i] - r[3]) < 1e-9  # same position
                    or (lod_c[i] < r[4] - min_lod_gap)  # not sep enough
                    for r in rows
                    if r[1] == chrom
                )
                if too_close:
                    continue
                # build a temporary ScanResult view for this chromosome to
                # get the CI (lod_drop_ci operates on a ScanResult)
                _sr_c = ScanResult(
                    locus_ids=lids_c,
                    chromosomes=np.array([chrom] * n),
                    positions=pos_c,
                    lod=lod_c,
                    p_value=scan_result.p_value[mask],
                    betas=scan_result.betas[mask],
                    h2_null={chrom: scan_result.h2_null.get(chrom, 0.0)},
                    phenotype_id=scan_result.phenotype_id,
                )
                lo, hi, pk = lod_drop_ci(_sr_c, chrom, drop)
                rows.append(
                    (
                        scan_result.phenotype_id,
                        chrom,
                        lids_c[i],
                        float(pos_c[i]),
                        float(lod_c[i]),
                        lo,
                        hi,
                    )
                )

    return pd.DataFrame(rows, columns=list(_COLS))
