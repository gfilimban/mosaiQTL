"""
records — result dataclasses for QTL scan outputs

all dataclasses use __slots__ for memory efficiency.
field naming rules:
  VQTLResult.scan_result      — dispersion scan  (NOT .vqtl_scan)
  VQTLResult.mean_scan_result — mean eQTL        (NOT .mean_scan)
  ScanResult.chr / QTLResult.gene_chr / QTLResult.peak_chr — always str, never int

locus classification:
  'cis'   — peak within cis_window_mb of gene TSS (same chr)
  'local' — same chromosome, outside cis window
  'trans' — different chromosome from gene

axis coverage:
  Tier 1 (affects result interpretation) → QTLResult fields
  Tier 2 (scan parameters for reproduction) → ProvenanceRecord only

  DataGranularity: 'single-cell' covers both scRNA-seq and snRNA-seq —
    the capture technology does not affect QTL result interpretation.
    CLI --granular accepts 'single-nuclei' as alias for 'single-cell'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

DataGranularity = Literal['single-cell', 'pseudobulk', 'bulk']
HaplotypeModel  = Literal['additive', 'full', 'mi']
TestStatistic   = Literal['score', 'lrt', 'f_test']
ScanModel       = Literal['nb-glmm', 'lmm']
# Population is str (not Literal) — open to any multiparent population or organism.
# current values: 'cc', 'do', 'mpp', 'f2', 'structured'
#   'mpp'        — any generic MPP; MAGIC (19 founders) is the reference example
#   'structured' — arbitrary structured population; human GWAS cohorts are the
#                  reference example (no haplotype probs; kinship from SNP GRM)


@dataclass(slots=True)
class LMMResult:
    """LMM fit result at a single locus (internal; used by lmm.py).

    attributes
    ----------
    lod      : float  LOD = (n/2) log10(RSS_0/RSS_1) — ML denominator n
    p_value  : float  F-test p-value
    h2       : float  heritability fixed from null model
    f_stat   : float  F statistic
    betas    : np.ndarray (J,)    haplotype effects at this locus (sum ≈ 0)
    sigma2_g : float  genetic variance component
    sigma2_e : float  residual variance component
    rss_null : float  weighted RSS under null
    rss_alt  : float  weighted RSS under alternative
    """

    lod:      float
    p_value:  float
    h2:       float
    f_stat:   float
    betas:    np.ndarray
    sigma2_g: float
    sigma2_e: float
    rss_null: float
    rss_alt:  float


@dataclass(slots=True)
class ScanResult:
    """single-chromosome scan output for one gene.

    attributes
    ----------
    gene_id : str
        gene identifier — required for standalone serialization
    lod : np.ndarray
        LOD scores, shape (n_loci,)
    loci : list[str]
        locus identifiers, length n_loci
    pos_mb : np.ndarray
        physical positions in Mb, shape (n_loci,)
    chr : str
        chromosome identifier — always str, never int
    """

    gene_id: str
    lod:     np.ndarray
    loci:    list[str]
    pos_mb:  np.ndarray
    chr:     str


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
    z_score : float
        Wald z-statistic (estimate / se)
    p_value : float
        two-sided Wald p-value for this founder's effect
    ci_lo : float
        lower bound of 95% confidence interval
    ci_hi : float
        upper bound of 95% confidence interval
    """

    founder:  str
    estimate: float
    se:       float
    z_score:  float
    p_value:  float
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
    cell_type : str or None
        cell type label for single-cell/pseudobulk; None for bulk
    data_granularity : DataGranularity
        aggregation level — 'single-cell' covers sc and sn RNA-seq
    model : ScanModel
    n_samples : int
    """

    scan_result:      ScanResult
    mean_scan_result: ScanResult
    phi_estimates:    np.ndarray
    phi_method:       str
    gene_id:          str
    cell_type:        Optional[str]
    data_granularity: DataGranularity
    model:            ScanModel
    n_samples:        int


@dataclass(slots=True)
class ProvenanceRecord:
    """full scan configuration for DESIGN-06 reproducibility.

    contains all Tier 2 axes (scan parameters) needed to exactly
    reproduce a result. attach to serialized output via QTLResult.provenance.

    attributes
    ----------
    git_hash : str
        repository commit hash at scan time
    seed : int
        random seed used for permutations
    timestamp : str
        ISO-8601 timestamp of scan completion
    scan_window : str
        window scanned: 'cis', 'chr', 'genome', or 'loci=<set>'
    phenotype_input : str
        phenotype representation: 'raw', 'normalized', or 'averaged'
    offset : str
        library size offset: 'from_libsize', 'from_file', or 'none'
    covariates : str
        covariate specification passed to --covar
    n_perms : int
        number of permutations run
    cis_window_mb : float
        cis window half-width in Mb
    """

    git_hash:        str
    seed:            int
    timestamp:       str
    scan_window:     str
    phenotype_input: str
    offset:          str
    covariates:      str
    n_perms:         int
    cis_window_mb:   float


@dataclass(slots=True)
class QTLResult:
    """eGene-level summary for one gene × grouping combination.

    tier-1 scan axes (affect result interpretation) are stored here.
    tier-2 axes (reproduction parameters) are in provenance.

    attributes
    ----------
    gene_id : str
    gene_name : str
    cell_type : str or None
        cell type label for single-cell/pseudobulk; None for bulk
    data_granularity : DataGranularity
        aggregation level — 'single-cell' covers sc and sn RNA-seq
    population : str
        design population; validated against SUPPORTED_POPULATIONS in ScanConfig.
        examples: 'cc', 'do', 'mpp' (e.g. MAGIC), 'f2', 'structured' (e.g. human cohort)
    model : ScanModel
        statistical model: 'nb-glmm' or 'lmm'
    test_statistic : TestStatistic
        test used: 'score', 'lrt', or 'f_test' — affects p_value derivation
    haplotype : HaplotypeModel
        haplotype encoding: 'additive', 'full', or 'mi'
        determines meaning of founder_effects
    gene_chr : str
        chromosome of gene TSS — always str, never int
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
    n_loci_tested : int
        number of loci scanned; context for multiple testing correction
    n_perms : int
        number of permutations used to derive lod_threshold
    n_samples : int
        number of samples used in the scan
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
    provenance : ProvenanceRecord
        full scan configuration for exact reproduction (DESIGN-06)
    """

    gene_id:            str
    gene_name:          str
    cell_type:          Optional[str]
    data_granularity:   DataGranularity
    population:         str
    model:              ScanModel
    test_statistic:     TestStatistic
    haplotype:          HaplotypeModel
    gene_chr:           str
    gene_tss_mb:        float
    peak_locus:         str
    peak_chr:           str
    peak_pos_mb:        float
    lod:                float
    lod_threshold:      float
    p_value:            float
    q_value:            float
    n_loci_tested:      int
    n_perms:            int
    n_samples:          int
    locus_type:         Literal['cis', 'local', 'trans']
    distance_to_tss_mb: float
    cis_window_mb:      float
    founder_effects:    dict[str, FounderEffect]
    variance_explained: float
    provenance:         ProvenanceRecord
