"""
records — result dataclasses for QTL scan outputs

all dataclasses use __slots__ for memory efficiency.
field naming rules:
  VQTLResult.scan_result      — dispersion scan  (NOT .vqtl_scan)
  VQTLResult.mean_scan_result — mean eQTL        (NOT .mean_scan)
  ScanResult.chr / QTLResult.chr — always str, never int

locus classification:
  'cis'   — peak within cis_window_mb of gene TSS (same chr)
  'local' — same chromosome, outside cis window
  'trans' — different chromosome from gene
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class ScanResult:
    """per-gene genome-wide scan output.

    attributes
    ----------
    lod : np.ndarray
        LOD scores, shape (n_loci,)
    loci : list[str]
        locus identifiers, length n_loci
    pos_mb : np.ndarray
        physical positions in Mb, shape (n_loci,)
    chr : str
        chromosome identifier — always str, never int
    """

    lod:    np.ndarray
    loci:   list[str]
    pos_mb: np.ndarray
    chr:    str


@dataclass(slots=True)
class FounderEffect:
    """haplotype effect estimate for one founder at one locus.

    attributes
    ----------
    founder : str
        founder strain name (from FOUNDER_ORDER)
    estimate : float
        effect size (log-scale for NB, linear for LMM)
    se : float
        standard error of estimate
    ci_lo : float
        lower bound of 95% confidence interval
    ci_hi : float
        upper bound of 95% confidence interval
    """

    founder:  str
    estimate: float
    se:       float
    ci_lo:    float
    ci_hi:    float


@dataclass(slots=True)
class VQTLResult:
    """dispersion QTL scan output paired with mean eQTL scan.

    attributes
    ----------
    scan_result : ScanResult
        dispersion (vQTL) scan — access as .scan_result, not .vqtl_scan
    mean_scan_result : ScanResult
        mean eQTL scan — access as .mean_scan_result, not .mean_scan
    phi_estimates : np.ndarray
        per-subject dispersion estimates, shape (n_subjects,)
    phi_method : str
        dispersion estimation method: 'moments' or 'mle'
    gene_id : str
        gene identifier
    """

    scan_result:      ScanResult
    mean_scan_result: ScanResult
    phi_estimates:    np.ndarray
    phi_method:       str
    gene_id:          str


@dataclass(slots=True)
class QTLResult:
    """eGene-level summary for one gene × cell-type combination.

    attributes
    ----------
    gene_id : str
    gene_name : str
    cell_type : str
    gene_chr : str
        chromosome where the gene TSS is located — always str, never int
    gene_tss_mb : float
        gene TSS position in Mb
    peak_locus : str
        locus identifier at the LOD peak (e.g. 'UNC2373664')
    peak_chr : str
        chromosome of peak locus — differs from gene_chr for trans
    peak_pos_mb : float
        peak locus position in Mb
    lod : float
        peak LOD score
    lod_threshold : float
        permutation-derived significance threshold
    p_value : float
    q_value : float
        BH-FDR corrected q-value
    n_perms : int
        number of permutations used to derive lod_threshold
    locus_type : Literal['cis', 'local', 'trans']
        'cis'   = same chr, within cis_window_mb of TSS
        'local' = same chr, outside cis window
        'trans' = different chromosome from gene
    distance_to_tss_mb : float
        signed distance peak → TSS in Mb; positive = downstream,
        negative = upstream; np.nan for trans
    cis_window_mb : float
        half-width of cis window used for locus_type assignment (Mb)
    founder_effects : dict[str, FounderEffect]
        per-founder effects at peak locus, keyed by founder name
    variance_explained : float
        proportion of phenotypic variance explained at peak locus
    model : str
        scan model used: 'nb-glmm' or 'lmm'
    n_samples : int
        number of samples used in the scan
    """

    gene_id:            str
    gene_name:          str
    cell_type:          str
    gene_chr:           str
    gene_tss_mb:        float
    peak_locus:         str
    peak_chr:           str
    peak_pos_mb:        float
    lod:                float
    lod_threshold:      float
    p_value:            float
    q_value:            float
    n_perms:            int
    locus_type:         Literal['cis', 'local', 'trans']
    distance_to_tss_mb: float
    cis_window_mb:      float
    founder_effects:    dict[str, FounderEffect]
    variance_explained: float
    model:              str
    n_samples:          int
