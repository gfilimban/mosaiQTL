"""thresholds — genome-wide significance thresholds from permutation nulls.

Fits a GEV to the per-permutation max-LOD distribution and reads
per-chromosome null tables to produce FWER-controlled LOD thresholds.
"""

import numpy as np
import logging
from pathlib import Path
from scipy.stats import genextreme

from mosaiqtl.io.tables import read_table

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'get_gev_thresholds',
    'load_per_chr_thresholds',
]


def get_gev_thresholds(
    max_lods: 'np.ndarray',
    percentile: float = 0.95,
) -> 'tuple[float, dict] | tuple[None, None]':
    """fit a GEV distribution to the permutation peak-LOD distribution.

    fitting a GEV curve gives a more stable threshold than an empirical
    quantile when n_perms is small (100-500).

    parameters
    ----------
    max_lods :
        per-permutation maximum LOD scores.
    percentile :
        significance percentile (default ``0.95`` = 5% genome-wide FWE).

    returns
    -------
    tuple[float, dict]
        LOD threshold at the requested percentile and a parameter dict with
        keys ``c``, ``loc``, ``scale``, ``threshold``, ``percentile``,
        ``n_perms``, ``empirical_quantile``.
    tuple[None, None]
        when there are fewer than 10 finite values or the GEV fit fails.
    """
    finite = max_lods[np.isfinite(max_lods)]
    if len(finite) < 10:
        logger.warning(
            'get_gev_thresholds: only %d finite values — need ≥10; returning None',
            len(finite),
        )
        return None, None
    try:
        c, loc, scale = genextreme.fit(finite)
        threshold = float(genextreme.ppf(percentile, c, loc=loc, scale=scale))
    except Exception as exc:
        _emp = float(np.quantile(finite, percentile))
        logger.warning(
            'get_gev_thresholds: GEV fit failed (%s) — returning None '
            '(empirical %.0f%% quantile was %.3f)',
            exc,
            percentile * 100,
            _emp,
        )
        return None, None

    params = {
        'c': float(c),
        'loc': float(loc),
        'scale': float(scale),
        'threshold': threshold,
        'percentile': percentile,
        'n_perms': len(finite),
        'empirical_quantile': float(np.quantile(finite, percentile)),
    }
    logger.debug(
        'get_gev_thresholds: n=%d  shape=%.4f  loc=%.4f  scale=%.4f  threshold=%.3f',
        len(finite),
        float(c),
        float(loc),
        float(scale),
        threshold,
    )
    return threshold, params


def load_per_chr_thresholds(
    perm_dir: 'Path',
    percentile: float = 0.95,
) -> 'dict[str, float | None]':
    """load per-chromosome null distributions and return GEV thresholds.

    reads ``perm_max_lods_chr{chrom}.tsv`` files written by
    :func:`~mosaiqtl.qtl.permutation.run_threshold_scans`, fits a GEV to
    each chromosome's max-LOD null, and returns per-chromosome FWER
    thresholds at *percentile*.

    useful when ``threshold.tsv`` was written with only a ``genome_wide``
    summary row (checkpointed scans) and per-chromosome bars are needed for
    diagnostic plots.

    parameters
    ----------
    perm_dir :
        directory containing ``perm_max_lods_chr*.tsv`` files (typically
        the ``core/`` subdirectory of a scan output).
    percentile :
        significance percentile; default ``0.95`` (5% FWER).

    returns
    -------
    dict[str, float | None]
        ``{chrom: threshold}``; value is ``None`` when the GEV fit fails.
    """
    perm_dir = Path(perm_dir)
    result: 'dict[str, float | None]' = {}
    for pf in sorted(perm_dir.glob('perm_max_lods_chr*.tsv')):
        chrom = pf.stem.replace('perm_max_lods_chr', '')
        try:
            nulls = read_table(pf)['max_lod'].values
            thr, _ = get_gev_thresholds(nulls, percentile=percentile)
            result[chrom] = thr
        except Exception:
            result[chrom] = None
    return result
