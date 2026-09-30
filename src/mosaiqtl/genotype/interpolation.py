"""interpolation — haplotype probability interpolation and biallelic collapse.

Layer 1 — interpolate_haplotype_probs():
    maps founder haplotype probabilities from the sparse GigaMUGA marker grid
    to arbitrary Sanger SNP positions using piecewise linear interpolation.
    linear interpolation preserves the probability simplex: weights are
    non-negative and sum to 1 at every SNP × strain pair.

Layer 2 — collapse_to_dosage() + snp_allele_stats():
    collapses the (n_snps, n_strains, n_founders) probability tensor to a
    (n_snps, n_strains) expected ALT-allele dosage using a pre-built binary
    alt_mask (see genotype/snp_db.py::make_alt_mask).  snp_allele_stats()
    derives allele frequency, MAF, and MAC from the dosage matrix.

primary use: Layers 1–2 of the Sanger fine-mapping stack.  consumers (Layer 2
biallelic collapse, Layer 3 ABF/SuSiE-RSS) call this function to obtain
haplotype probabilities at each Sanger SNP position before computing
REF/ALT dosage or posterior inclusion probabilities.
"""

from __future__ import annotations

import logging

import numpy as np

from .genomecache import GenomeCache

logger = logging.getLogger('mosaiqtl')


def interpolate_haplotype_probs(
    cache: GenomeCache,
    snp_positions_bp: np.ndarray,
    chrom: str,
    gmap: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate haplotype probabilities from marker grid to SNP positions.

    Performs piecewise linear interpolation of the (n_loci, n_strains,
    n_founders) probability tensor onto an arbitrary set of SNP positions.
    SNPs outside the marker grid are clamped to the nearest boundary marker
    (no extrapolation), preserving the probability simplex.

    Parameters
    ----------
    cache : GenomeCache
        Haplotype probability cache.  Must contain *chrom*.
    snp_positions_bp : np.ndarray
        1-D integer or float array of Sanger SNP positions in base pairs.
        Must be on the same chromosome as *chrom*.  Need not be sorted.
    chrom : str
        Chromosome identifier (e.g. ``'1'``, ``'X'``).  Must match the
        key used in *cache* (no ``chr`` prefix required by this function;
        normalise in the caller if needed).
    gmap : np.ndarray or None, optional
        If provided, a ``(2, n_loci)`` float array whose first row holds
        marker physical positions in Mb (matching ``cache.get_chromosome``
        order) and whose second row holds the corresponding genetic map
        positions in cM.  When given, interpolation is performed in
        genetic-map space (cM) rather than physical-position (Mb) space,
        which reduces interpolation error across recombination hotspots.
        Positions in *snp_positions_bp* are converted to cM via linear
        interpolation of *gmap* before the main interpolation step.
        When ``None`` (default), physical-position interpolation is used.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_snps, n_strains, n_founders)`` giving
        the interpolated haplotype probabilities at each SNP position.
        Probabilities are non-negative and sum to 1 per (SNP, strain) pair
        within floating-point tolerance.

    Raises
    ------
    ValueError
        If *chrom* is not present in *cache*.
    ValueError
        If *gmap* is provided but its shape or physical-position row does
        not match ``cache.get_chromosome(chrom)`` positions.
    """
    if chrom not in cache.chromosomes:
        raise ValueError(
            f'chromosome {chrom!r} not found in cache '
            f'(available: {cache.chromosomes})'
        )

    _lids, marker_mb, probs = cache.get_chromosome(chrom)
    # probs: (n_loci, n_strains, n_founders)

    snp_mb = np.asarray(snp_positions_bp, dtype=np.float64) / 1e6

    if gmap is not None:
        gmap = np.asarray(gmap, dtype=np.float64)
        if gmap.shape != (2, len(marker_mb)):
            raise ValueError(
                f'gmap shape {gmap.shape} does not match '
                f'(2, {len(marker_mb)}) expected for {chrom!r}'
            )
        gmap_phys_mb, gmap_cm = gmap[0], gmap[1]
        # interpolate SNP bp → cM via genetic map
        snp_cm = _linear_interp_1d(gmap_phys_mb, gmap_cm, snp_mb)
        # use cM coordinates for the main interpolation
        grid = gmap_cm
        query = snp_cm
    else:
        grid = marker_mb
        query = snp_mb

    # ── find bracket markers for each SNP ────────────────────────────────
    # idx: index of the right-hand marker; clamp so left = idx-1 is valid
    n_loci = len(grid)
    idx = np.searchsorted(grid, query, side='right')
    idx = np.clip(idx, 1, n_loci - 1)

    left = idx - 1

    dx = grid[idx] - grid[left]             # (n_snps,)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.where(dx > 0, (query - grid[left]) / dx, 0.0)  # (n_snps,)
    t = np.clip(t, 0.0, 1.0)

    # ── segment loop over marker intervals ───────────────────────────────
    # iterate over the L-1 unique bracket pairs (≤ ~4000) rather than
    # over all S SNPs; peak temporary memory per iteration is
    # O((S/L) × n_strains × n_founders) vs O(S × ...) for full fancy-indexing.
    n_snps = len(query)
    n_strains, n_founders = probs.shape[1], probs.shape[2]
    interp = np.empty((n_snps, n_strains, n_founders), dtype=np.float64)
    p_left = probs[0]                       # (n_strains, n_founders); reused across loop
    for seg in range(1, n_loci):
        p_right = probs[seg]
        mask = left == seg - 1
        if not mask.any():
            p_left = p_right
            continue
        t_seg = t[mask, np.newaxis, np.newaxis]   # (k, 1, 1)
        interp[mask] = (1.0 - t_seg) * p_left + t_seg * p_right
        p_left = p_right

    logger.debug(
        'interpolated %d SNPs on chr%s using %d markers (space: %s)',
        len(snp_mb), chrom, n_loci,
        'genetic-map (cM)' if gmap is not None else 'physical (Mb)',
    )
    return interp


def collapse_to_dosage(
    probs: np.ndarray,
    alt_mask: np.ndarray,
) -> np.ndarray:
    """Collapse haplotype probabilities to per-strain ALT-allele dosage.

    For each SNP and each strain, dosage is the expected number of ALT-allele
    haplotypes: ``dosage[s, i] = Σ_f probs[s, i, f] × alt_mask[s, f]``.

    Because CC strains are haploid-equivalent (one recombinant inbred line per
    ``sample_id``), dosage lies in [0, 1], not [0, 2].

    Memory note: iterates over n_founders (≤ 8) rather than materialising a
    full ``(n_snps, n_strains, n_founders)`` intermediate.  Peak extra
    allocation is one ``(n_snps, n_strains)`` accumulator.

    Parameters
    ----------
    probs : np.ndarray
        Float64 array of shape ``(n_snps, n_strains, n_founders)``; output of
        :func:`interpolate_haplotype_probs`.
    alt_mask : np.ndarray
        Float64 array of shape ``(n_snps, n_founders)`` with values in
        ``{0.0, 1.0}``; output of
        :func:`~mosaiqtl.genotype.snp_db.make_alt_mask`.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_snps, n_strains)``.  Values in [0, 1].

    Raises
    ------
    ValueError
        If ``probs`` and ``alt_mask`` have inconsistent SNP or founder axes.
    """
    n_snps, n_strains, n_founders = probs.shape
    if alt_mask.shape != (n_snps, n_founders):
        raise ValueError(
            f'alt_mask shape {alt_mask.shape} does not match '
            f'(n_snps={n_snps}, n_founders={n_founders})'
        )
    dosage = np.zeros((n_snps, n_strains), dtype=np.float64)
    for f in range(n_founders):
        col = alt_mask[:, f]
        if not col.any():
            continue
        # probs[:, :, f] is (n_snps, n_strains); col[:, np.newaxis] is (n_snps, 1)
        dosage += probs[:, :, f] * col[:, np.newaxis]
    return dosage


def snp_allele_stats(
    dosage: np.ndarray,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute allele frequency, MAF, and MAC from a dosage matrix.

    All three statistics are computed across strains (axis 1), assuming equal
    strain weights.  For a proper population-weighted estimate, the caller
    should pass a pre-weighted dosage matrix.

    Parameters
    ----------
    dosage : np.ndarray
        Float64 array of shape ``(n_snps, n_strains)``; output of
        :func:`collapse_to_dosage`.
    n_samples : int or None, optional
        Total number of biological samples (used for MAC).  When ``None``,
        ``n_strains`` (``dosage.shape[1]``) is used as a proxy.

    Returns
    -------
    af : np.ndarray
        Shape ``(n_snps,)``.  Mean ALT allele frequency across strains.
    maf : np.ndarray
        Shape ``(n_snps,)``.  Minor allele frequency: ``min(af, 1 - af)``.
    mac : np.ndarray
        Shape ``(n_snps,)``.  Minor allele count: ``maf × n_samples``,
        rounded to the nearest integer.
    """
    n_eff = n_samples if n_samples is not None else dosage.shape[1]
    af = dosage.mean(axis=1)                  # (n_snps,)
    maf = np.minimum(af, 1.0 - af)
    mac = np.round(maf * n_eff).astype(np.int64)
    return af, maf, mac


def _linear_interp_1d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Piecewise linear 1-D interpolation with boundary clamping.

    Parameters
    ----------
    x_grid : np.ndarray
        Strictly increasing 1-D knot positions.
    y_grid : np.ndarray
        Values at knot positions; same length as *x_grid*.
    x_query : np.ndarray
        Query positions.  Out-of-range queries are clamped to the nearest
        boundary value.

    Returns
    -------
    np.ndarray
        Interpolated values at each query position; same shape as *x_query*.
    """
    n = len(x_grid)
    idx = np.searchsorted(x_grid, x_query, side='right')
    idx = np.clip(idx, 1, n - 1)
    left, right = idx - 1, idx
    dx = x_grid[right] - x_grid[left]
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.where(dx > 0, (x_query - x_grid[left]) / dx, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t) * y_grid[left] + t * y_grid[right]
