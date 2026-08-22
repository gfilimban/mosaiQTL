"""
pipeline — user-facing layer API

three-layer analysis framework for cis-eQTL mapping in multiparent populations.
users select how far to go: discovery only, add characterization, or full
mechanistic resolution. within Layer 3, each subsection is independently
optional — pass only the modules and inputs you have.

    Layer 1 — QTL Discovery         pipeline.discover()
    Layer 2 — QTL Characterization  pipeline.characterize()
    Layer 3 — Mechanistic Resolution pipeline.resolve()

quick start:
    from mosaiqtl import pipeline

    # Layer 1 only
    disc = pipeline.discover(counts, cache, covariates, sample_ids)

    # Layer 1 + 2
    disc  = pipeline.discover(counts, cache, covariates, sample_ids)
    char  = pipeline.characterize(disc, cache, covariates)

    # Full pipeline
    disc  = pipeline.discover(counts, cache, covariates, sample_ids)
    char  = pipeline.characterize(disc, cache, covariates)
    res   = pipeline.resolve(disc, cache,
                modules=['snp_catalogue', 'conditional_scan'],
                snp_catalogue_kwargs={'mgp_vcf': '/path/to/mgp.vcf.gz'},
                conditional_scan_kwargs={'n_signals': 3})
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from scipy import stats as _sp_stats

from .qtl.scan import scan_h2lmm, scan_multi_gene
from .qtl.mediation import MEDIATION_COLS, FIXEF_MEDIATION_COLS
from .qtl.snp_scan import SNP_SCAN_COLS
from .qtl.permutation import CI_COLS

from .genotype import kinship as kin_module

logger = logging.getLogger("mosaiqtl")


# ─────────────────────────────────────────────────────────────────────────────
# canonical output schemas
# gene_id is the stable join key (ENSMUSG ID, version-stripped).
# gene_symbol is display-only (e.g. 'Ankrd39'); not guaranteed unique.
# ─────────────────────────────────────────────────────────────────────────────

LOD_COLS: list[str] = ["gene_id", "chr", "locus", "pos_mb", "lod"]

ALLELE_EFFECTS_BASE_COLS: list[str] = [
    "gene_id",
    "gene_symbol",
    "chr",
    "start_mb",
    "peak_locus",
    "peak_mb",
    "peak_lod",
    "h2",
    "n_valid",
    "rss0",
    "rss1",
    "r2_peak",
    "sigma2_e",
    "tau2",
]

PEAK_LOD_COLS: list[str] = [
    "gene_id",
    "gene_symbol",
    "chr",
    "start_mb",
    "near_tss_mb",
    "peak_locus",
    "peak_mb",
    "cis_dist_mb",
    "peak_lod",
    "h2",
    "n_valid",
    "rss0",
    "rss1",
    "r2_peak",
    "sigma2_e",
    "tau2",
]

EGENE_SUMMARY_COLS: list[str] = [
    "gene_id",
    "gene_symbol",
    "chr",
    "start_mb",
    "near_tss_mb",
    "peak_locus",
    "peak_mb",
    "cis_dist_mb",
    "peak_lod",
    "h2",
    "perm_n",
    "perm_chr",
    "chi2_haplotype",
    "p_haplotype",
    "in_r_egenes",
]

# ─────────────────────────────────────────────────────────────────────────────
# eQTL output schemas
# Column order: locus identity → alleles → haplotype freq → gene → association
#               → haplotype effects → fine-mapping → significance → run metadata
# All coordinates are integer bp (GRCm39). Mb appears only in cis_dist_mb
# (signed diagnostic quantity, not a coordinate).
# Allele annotation is haplotype-resolution throughout.
# ─────────────────────────────────────────────────────────────────────────────

# ── shared building blocks ────────────────────────────────────────────────────

# Shared locus + gene block used by both haplotype scan and Sanger fine-mapping.
# hard_call_dosage excluded from fine-mapping (N/A at Sanger resolution).
_LOCUS_COLS: list[str] = [
    "variant",  # SPDI: {refseq_accn}:{pos_bp-1}:{ref}:{effect_allele}
    "chr",  # e.g. 'chr8'
    "pos",  # integer bp (GRCm39)
    "ref",  # B6J allele = GRCm39 reference allele (VCF convention)
    "effect_allele",  # non-ref allele for this haplotype group contrast
    "other_allele",  # complement of effect_allele
    "hard_call_dosage",  # constant 'F' — scan uses haplotype probs, not hard calls
    "ref_founders",  # comma-joined founders carrying ref, e.g. 'AJ,B6J,129S1J,NODJ'
    "alt_founders",  # comma-joined founders carrying effect_allele
    "af",  # mean haplotype prob for alt_founders across CC strains
    "maf",  # min(af, 1-af)
    "mac",  # round(maf × n_strains) — strain level
    "ma_sample",  # round(maf × n_sample) — replicate mouse level
    "gene_id",  # ENSMUSG ID, version suffix stripped (join key)
    "gene_symbol",  # gene symbol, e.g. 'Ankrd39' (display only)
    "sd_phenotype",  # sqrt(rss0 / (n_valid - 1)) — SD after covariate regression
    "n_sample",  # n_valid from null model fit
    "lod",  # joint LOD across all 8 founders (7 df score test)
    "h2_null_chr",  # LOCO null heritability (per-chromosome constant)
    "pval_nominal",  # chi² p-value (7 df)
    "log10p_nominal",  # -log10(pval_nominal)
    "cis_dist_mb",  # signed distance TSS → locus (Mb); only Mb column in outputs
    "effect_size",  # haplotype contrast: c_k^T β_founders
    "se_effect_size",  # sqrt(c_k^T Var(β) c_k)
]

# Per-founder haplotype effects — only at peak/lead-SNP locus (one row per eGene)
_FOUNDER_EFFECT_COLS: list[str] = [
    "effect_AJ",
    "effect_B6J",
    "effect_129S1J",
    "effect_NODJ",
    "effect_NZOJ",
    "effect_CASTJ",
    "effect_PWKJ",
    "effect_WSBJ",
]

# Gene-level significance — carried from haplotype scan, not re-estimated at Sanger
_SIG_COLS: list[str] = [
    "pval_perm",  # gene-level permutation p-value
    "fdr",  # BH FDR across all scanned genes (SWE-09: Storey q-value planned)
    "fdr_method",  # constant 'BH'
    "significance",  # 'T' if passed LOD threshold AND fdr < 0.05
]

# Run-level metadata — constant for the whole file; peak/submission rows only
# (not replicated on every per-locus scan row)
_RUN_COLS: list[str] = [
    "interaction",  # 'F' for all main-effect rows
    "interaction_context",  # empty; OLS4 term ID only for dedicated interaction analyses
    "conditional",  # empty
    "conditional_method",  # empty
]

# ── schema definitions ────────────────────────────────────────────────────────

# Haplotype scan — all cis loci × all scanned genes (intermediate, per-chromosome)
EQTL_SCAN_COLS: list[str] = _LOCUS_COLS + ["peak_locus"]

# Haplotype peak — filtered to peak_locus == 'T', peak_locus dropped,
# per-founder effects + significance + run metadata appended.
EQTL_PEAK_COLS: list[str] = _LOCUS_COLS + _FOUNDER_EFFECT_COLS + _SIG_COLS + _RUN_COLS

# Sanger fine-mapping checkpoint — CS members + lead SNP per eGene.
# hard_call_dosage excluded (N/A at Sanger resolution).
# lead_snp == 'T' marks the highest-LOD SNP per eGene; 'F' for other CS members.
EQTL_FINEMAPPING_COLS: list[str] = [
    c for c in _LOCUS_COLS if c != "hard_call_dosage"
] + [
    "PIP",  # posterior inclusion probability
    "PIP_method",  # 'ABF' or 'SuSiE-RSS'
    "CS",  # 'T' if in the 95% credible set for this gene
    "lead_snp",  # 'T' for the highest-LOD Sanger SNP per eGene; 'F' for other CS members
    "ci_lo_bp",  # LOD-drop CI lower bound (bp) — haplotype scan window context
    "ci_hi_bp",  # LOD-drop CI upper bound (bp)
    "hap_peak_pos",  # GigaMUGA peak position (bp) — back-link to _eqtl_peak.csv
    "hap_peak_variant",  # SPDI of GigaMUGA peak
    "hap_peak_lod",  # LOD at GigaMUGA peak
]

# Sanger fine-mapping peak — submission file (one row per eGene).
# Derived from EQTL_FINEMAPPING_COLS filtered to lead_snp == 'T', lead_snp dropped,
# then per-founder effects + significance + run metadata appended.
EQTL_FINEMAPPING_PEAK_COLS: list[str] = (
    [c for c in EQTL_FINEMAPPING_COLS if c != "lead_snp"]
    + _FOUNDER_EFFECT_COLS
    + _SIG_COLS
    + _RUN_COLS
)

# Summary stats — one row per chromosome + one 'genome' row per tissue × cell type.
# git_hash populated from ProvenanceRecord.git_hash at write time.
EQTL_STATS_COLS: list[str] = [
    "tissue",
    "celltype",
    "chr",
    "lod_threshold",
    "gev_mu",
    "gev_sigma",
    "gev_xi",
    "n_genes_scanned",
    "n_egenes",
    "n_loci",
    "n_perms",
    # global row only (chr == 'genome')
    "fdr_threshold",
    "scan_window_mb",
    "scan_seed",
    "gencode_version",
    "ensembl_release",
    "git_hash",
]

__all__ = [
    "LOD_COLS",
    "ALLELE_EFFECTS_BASE_COLS",
    "PEAK_LOD_COLS",
    "EGENE_SUMMARY_COLS",
    "MEDIATION_COLS",
    "FIXEF_MEDIATION_COLS",
    "SNP_SCAN_COLS",
    "CI_COLS",
    "EQTL_SCAN_COLS",
    "EQTL_PEAK_COLS",
    "EQTL_FINEMAPPING_COLS",
    "EQTL_FINEMAPPING_PEAK_COLS",
    "EQTL_STATS_COLS",
    "call_egenes",
    "finalize_egene_summary",
    "discover",
    "characterize",
    "resolve",
    "DiscoveryResult",
    "CharacterizationResult",
    "ResolutionResult",
]


def finalize_egene_summary(
    egenes_df: pd.DataFrame,
    n_founders: int | None = None,
) -> pd.DataFrame:
    """Add chi2/p_haplotype stats and apply canonical EGENE_SUMMARY_COLS order.

    Parameters
    ----------
    egenes_df : DataFrame produced by call_egenes — must contain peak_lod column.
    n_founders : number of founders (df = n_founders - 1 for chi2 test). When
        None (default), derived from the number of beta_{founder} columns present
        in egenes_df, making this function population-agnostic.

    Returns
    -------
    DataFrame with columns reindexed to EGENE_SUMMARY_COLS (present columns only).
    """
    out = egenes_df.copy()
    if n_founders is None:
        beta_cols = [c for c in out.columns if c.startswith("beta_")]
        n_founders = len(beta_cols) if beta_cols else 8
        if not beta_cols:
            logger.warning(
                "finalize_egene_summary: no beta_ columns found; "
                "defaulting n_founders=8 for chi2 df — pass n_founders explicitly "
                "for non-CC populations"
            )
    if "peak_lod" in out.columns:
        out["chi2_haplotype"] = 2.0 * np.log(10) * out["peak_lod"]
        out["p_haplotype"] = _sp_stats.chi2.sf(
            out["chi2_haplotype"].values, df=n_founders - 1
        )
    return out.reindex(columns=[c for c in EGENE_SUMMARY_COLS if c in out.columns])


# ─────────────────────────────────────────────────────────────────────────────
# eGene utilities
# ─────────────────────────────────────────────────────────────────────────────


def call_egenes(
    peak_lod_df: pd.DataFrame,
    threshold: float,
    effects_df: "pd.DataFrame | None" = None,
    lod_col: str = "peak_lod",
) -> pd.DataFrame:
    """Apply a LOD threshold and return one row per called eGene.

    Parameters
    ----------
    peak_lod_df : DataFrame with columns [gene_id, gene_symbol, peak_lod,
        chr, peak_locus, peak_mb].  gene_id is the stable ENSMUSG join key;
        gene_symbol is the display name.  Positional columns (chr, peak_locus,
        peak_mb) should be present at scan time so downstream GxE/effect steps
        are not blocked by a missing effects file.
    threshold   : LOD threshold (e.g. from get_gev_thresholds)
    effects_df  : optional allele-effects DataFrame — used only to fill
        chr/peak_locus/peak_mb when those columns are absent from peak_lod_df
        (legacy fallback); merge is performed on gene_id
    lod_col     : name of the peak LOD column

    Returns
    -------
    DataFrame with columns [gene_id, gene_symbol, chr, peak_locus, peak_mb,
    peak_lod, ...] — one row per called eGene; positional columns NaN only if
    absent from both inputs
    """
    egenes = peak_lod_df[peak_lod_df[lod_col] >= threshold].copy()
    pos_cols = ("chr", "peak_locus", "peak_mb")
    missing = [c for c in pos_cols if c not in egenes.columns]
    if missing and effects_df is not None and len(effects_df) > 0:
        fill_cols = [c for c in ("gene_id", *missing) if c in effects_df.columns]
        if len(fill_cols) > 1:
            egenes = egenes.merge(effects_df[fill_cols], on="gene_id", how="left")
    return egenes.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    """output of pipeline.discover() — Layer 1."""

    scan: dict  # {gene_id: ScanResult}
    model: str  # 'nb' or 'lmm'
    genomecache: object  # GenomeCache reference
    kinship_decomp: Optional[dict] = None  # populated when model='lmm'
    dispersion: Optional[dict] = None  # {gene_id: float}, model='nb' only


@dataclass
class CharacterizationResult:
    """output of pipeline.characterize() — Layer 2."""

    allele_effects: Optional[dict] = None  # {gene_id: np.ndarray (J-1,)}
    vqtl: Optional[dict] = None  # {gene_id: ScanResult}
    mediation: Optional[dict] = None  # {gene_id: DataFrame}
    discovery: Optional[DiscoveryResult] = None


@dataclass
class ResolutionResult:
    """output of pipeline.resolve() — Layer 3."""

    snp_catalogue: Optional[dict] = None  # {gene_id: DataFrame of candidate SNPs}
    conditional_scan: Optional[dict] = None  # {gene_id: list[ScanResult]}
    ase: Optional[dict] = None  # {gene_id: DataFrame allele counts}
    sqtl: Optional[dict] = None  # {gene_id: ScanResult on PSI phenotype}
    modules_run: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — QTL Discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover(
    phenotype_matrix,
    gene_ids,
    genomecache,
    covariates=None,
    sample_ids=None,
    model="nb",
    lib_sizes=None,
    cis_loci_dict=None,
    n_cores=1,
    kinship_decomp=None,
):
    """Layer 1 — genome-wide haplotype QTL scan for a set of genes.

    parameters
    ----------
    phenotype_matrix : np.ndarray (n, G)  raw counts (nb) or normalized (lmm)
    gene_ids         : list of str  length G
    genomecache     : GenomeCache
    covariates       : np.ndarray (n, p) or None  intercept + batch + PCs
    sample_ids       : array-like (n,) or None
    model            : 'nb' (default) or 'lmm'
        'nb'  — NB GLM on raw counts; no kinship; primary mosaiQTL model
        'lmm' — gaussian lmm scan with loco kinship correction
    lib_sizes        : np.ndarray (n,) or None  (nb only) log-offset
    cis_loci_dict    : dict {gene_id: set} or None  restrict scan to cis window
    n_cores          : int
    kinship_decomp   : dict or None  pre-computed LOCO kinship (lmm only);
                       computed internally if None

    returns
    -------
    DiscoveryResult
    """
    if model == "nb":
        # counts_cells: (n_genes, n_cells); sample_ids: per-cell strain ids
        from .qtl.nb import scan_nb as _scan_nb

        scan_results = _scan_nb(
            phenotype_matrix,
            genomecache,
            sample_ids,
            gene_ids=gene_ids,
            covariates_cells=covariates,
            lib_sizes=lib_sizes,
            cis_loci_dict=cis_loci_dict,
            n_cores=n_cores,
        )
        return DiscoveryResult(scan=scan_results, model="nb", genomecache=genomecache)

    elif model == "lmm":
        if kinship_decomp is None:
            kd_raw = kin_module.calc_kinship(genomecache, method="loco")
            kinship_decomp = kin_module.decomp_kinship(kd_raw)
        scan_results = scan_multi_gene(
            phenotype_matrix,
            gene_ids,
            genomecache,
            covariates=covariates,
            kinship_decomp=kinship_decomp,
            sample_ids=sample_ids,
        )
        return DiscoveryResult(
            scan=scan_results,
            model="lmm",
            genomecache=genomecache,
            kinship_decomp=kinship_decomp,
        )
    else:
        raise ValueError(f'model must be "nb" or "lmm", got {model!r}')


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — QTL Characterization
# ─────────────────────────────────────────────────────────────────────────────


def characterize(
    discovery_result,
    covariates=None,
    modules=("allele_effects", "vqtl", "mediation"),
    allele_effects_kwargs=None,
    vqtl_kwargs=None,
    mediation_kwargs=None,
):
    """Layer 2 — characterize discovered QTLs.

    parameters
    ----------
    discovery_result : DiscoveryResult  from pipeline.discover()
    covariates       : np.ndarray (n, p) or None
    modules          : iterable of str  subset of:
        'allele_effects' — founder haplotype effect sizes at peak loci
        'vqtl'          — dispersion QTL scan (transcriptional noise)
        'mediation'     — LOD-drop mediation analysis
    *_kwargs         : module-specific keyword arguments (dicts or None)

    returns
    -------
    CharacterizationResult
    """
    res = CharacterizationResult(discovery=discovery_result)

    if "allele_effects" in modules:
        # extract betas at peak locus from each ScanResult;
        # LRT mode needed for NB to get non-zero betas
        ae = {}
        for gid, sr in discovery_result.scan.items():
            if len(sr.lod) == 0:
                continue
            pi = int(np.argmax(sr.lod))
            ae[gid] = sr.betas[pi]  # (J-1,) — zeros for NB score-test mode
        res.allele_effects = ae
        logger.info(f"allele effects: {len(ae)} genes")

    if "vqtl" in modules:
        # vqtl requires per-cell counts and subject_ids (not available from
        # pseudobulk DiscoveryResult). call scan_vqtl() directly with cell-
        # level data; pass vqtl_kwargs={'kinship_decomp': kd} for efficiency.
        logger.info("vqtl: call scan_vqtl() directly with cell-level counts")
        res.vqtl = {}

    if "mediation" in modules:
        # mediation requires explicit outcome + mediator pair — skip auto-loop
        # users should call mediation_scan_qr() directly per gene pair
        logger.info("mediation: call mediation_scan_qr() directly per gene pair")
        res.mediation = {}

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Mechanistic Resolution
# ─────────────────────────────────────────────────────────────────────────────

_LAYER3_MODULES = ("snp_catalogue", "conditional_scan", "ase", "sqtl")


def resolve(
    discovery_result,
    covariates=None,
    modules=("snp_catalogue", "conditional_scan"),
    snp_catalogue_kwargs=None,
    conditional_scan_kwargs=None,
    ase_kwargs=None,
    sqtl_kwargs=None,
):
    """Layer 3 — Mechanistic Resolution.

    maps QTL intervals to candidate SNPs, decomposes multi-signal loci,
    validates via allele-specific expression in F1 hybrids, and integrates
    splicing QTLs from long-read data. each subsection is independently
    optional — run only what your data support.

    parameters
    ----------
    discovery_result : DiscoveryResult
    covariates       : np.ndarray (n, p) or None
    modules          : iterable of str, any subset of:

        'snp_catalogue'    [3a] — founder-divergent SNP candidates per QTL interval.
                                  requires snp_catalogue_kwargs['mgp_vcf'].

        'conditional_scan' [3b] — forward-stepwise multi-signal decomposition.
                                  conditions on peak BLUP, rescans residuals.
                                  kwargs: n_signals (int, default 3).

        'ase'              [3c] — allele-specific expression via kallisto in F1 hybrids.
                                  requires ase_kwargs['f1_fastq_dirs'] and
                                  ase_kwargs['founder_fastas'].
                                  only meaningful for heterozygous F1 crosses —
                                  inbred CC strains are skipped automatically.

        'sqtl'             [3d] — splicing QTL scan on PSI or isoform phenotypes.
                                  requires sqtl_kwargs['psi_matrix'] and
                                  sqtl_kwargs['gene_ids'].

    snp_catalogue_kwargs : dict or None
        mgp_vcf      : str   path to MGP r9 VCF (bgzipped + tabix-indexed)
        window_mb    : float cis window in Mb around QTL peak (default 10.0)
        founders     : list  subset of 8 founders to query (default all 8)

    conditional_scan_kwargs : dict or None
        n_signals    : int   max independent signals to extract (default 3)
        lod_drop     : float LOD drop for secondary signal threshold (default 1.5)

    ase_kwargs : dict or None
        f1_fastq_dirs  : dict  {f1_cross_name: [r1_path, r2_path]}
                               e.g. {'B6CASTF1J': ['/path/r1.fastq.gz', ...]}
        founder_fastas : dict  {founder_name: fasta_path}  MGP pseudogenome fastas
        gtf_path       : str   shared GTF annotation path
        tool           : str   'kallisto' (default) — pseudoalignment via kallisto|bus
        threads        : int   kallisto threads (default 8)

    sqtl_kwargs : dict or None
        psi_matrix   : np.ndarray (n, J)  percent-spliced-in per junction
        junction_ids : list of str        length J
        gene_map     : dict {junction_id: gene_id}

    returns
    -------
    ResolutionResult
    """
    res = ResolutionResult()
    cache = discovery_result.genomecache
    unknown = set(modules) - set(_LAYER3_MODULES)
    if unknown:
        raise ValueError(f"unknown modules: {unknown}. valid: {_LAYER3_MODULES}")

    if "snp_catalogue" in modules:
        res.snp_catalogue = _run_snp_catalogue(
            discovery_result, cache, **(snp_catalogue_kwargs or {})
        )
        res.modules_run.append("snp_catalogue")

    if "conditional_scan" in modules:
        res.conditional_scan = _run_conditional_scan(
            discovery_result, cache, covariates, **(conditional_scan_kwargs or {})
        )
        res.modules_run.append("conditional_scan")

    if "ase" in modules:
        res.ase = _run_ase_kallisto(discovery_result, **(ase_kwargs or {}))
        res.modules_run.append("ase")

    if "sqtl" in modules:
        res.sqtl = _run_sqtl(discovery_result, cache, covariates, **(sqtl_kwargs or {}))
        res.modules_run.append("sqtl")

    logger.info(f"Layer 3 complete: {res.modules_run}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 internal implementations
# ─────────────────────────────────────────────────────────────────────────────


def _run_snp_catalogue(disc, cache, mgp_vcf=None, window_mb=10.0, founders=None):
    """3a — SNP catalogue: MGP overlay onto haplotype QTL intervals.

    for each gene with a peak QTL, queries the MGP VCF within window_mb
    of the peak, filters to SNPs private to founders with the top haplotype
    effect, and returns a DataFrame of candidates ranked by founder divergence.

    requires: mgp_vcf (bgzipped + tabix-indexed Sanger MGP r9 VCF).
    """
    if mgp_vcf is None:
        logger.warning("snp_catalogue: mgp_vcf not provided — skipping")
        return {}
    from .qtl.snp_scan import build_snp_catalogue

    return build_snp_catalogue(
        disc.scan, cache, mgp_vcf, window_mb=window_mb, founders=founders
    )


def _run_conditional_scan(disc, cache, covariates, n_signals=3, lod_drop_threshold=1.5):
    """3b — conditional scan: forward-stepwise multi-signal decomposition.

    for each gene with a significant peak, conditions on the peak-locus BLUP
    and rescans the same cis window for secondary signals. repeats up to
    n_signals times or until no signal exceeds lod_drop.

    rationale: CC haplotype blocks average ~10 Mb; a single QTL interval
    often contains multiple independent signals. conditioning on peak BLUP
    (rather than peak-locus residual) reduces false secondary signals from
    LD within the block.
    """
    from .qtl.epistasis import scan_conditional

    results = {}
    for gid, sr in disc.scan.items():
        if len(sr.lod) == 0:
            continue
        signals = scan_conditional(
            sr,
            cache,
            covariates,
            n_signals=n_signals,
            lod_drop_threshold=lod_drop_threshold,
            kinship_decomp=disc.kinship_decomp,
        )
        if signals:
            results[gid] = signals
    return results


def _run_ase_kallisto(
    disc,
    f1_fastq_dirs=None,
    founder_fastas=None,
    gtf_path=None,
    tool="kallisto",
    threads=8,
):
    """3c — ASE in F1 hybrids via kallisto|bustools.

    for each F1 cross in f1_fastq_dirs, builds a two-founder transcriptome
    index (dam × sire from founder_fastas + gtf), runs kallisto bus, then
    separates reads by allele of origin using bustools.

    only F1 hybrid samples (heterozygous) are processed. inbred CC strains
    are skipped because homozygosity precludes allele-specific assignment.

    approach:
      1. build founder-specific transcriptomes from MGP pseudogenomes + GTF
         using kb ref with the founder genome FASTA as d-list
      2. concatenate dam + sire transcriptomes into one two-allele index
      3. run kallisto bus → BUS file
      4. bustools sort → bustools capture (transcript-to-allele t2g)
      5. bustools count per allele → two count matrices (dam / sire)
      6. compute allelic imbalance: AI = (dam − sire) / (dam + sire)

    parameters
    ----------
    f1_fastq_dirs  : dict {cross_name: [r1, r2]}
        e.g. {'B6CASTF1J': ['/r1.fastq.gz', '/r2.fastq.gz']}
        cross_name must encode parental founders as 'B6{FOUNDER}F1J'
        to allow automatic lookup in founder_fastas.
    founder_fastas : dict {founder_name: fasta_path}
        founder genome FASTAs from MGP pseudogenomes (one per CC founder).
    gtf_path       : str  shared genome annotation GTF.
    tool           : str  'kallisto' (default). future: 'alevin_fry'.
    threads        : int  kallisto thread count.

    returns
    -------
    dict {cross_name: DataFrame}
        columns: gene_id, dam_counts, sire_counts, ai, n_cells
    """
    if f1_fastq_dirs is None or founder_fastas is None or gtf_path is None:
        logger.warning(
            "ase: f1_fastq_dirs, founder_fastas, and gtf_path required — skipping"
        )
        return {}
    if tool != "kallisto":
        raise NotImplementedError(f'ase tool {tool!r} not implemented; use "kallisto"')
    from .qtl.ase import run_kallisto_ase

    return run_kallisto_ase(f1_fastq_dirs, founder_fastas, gtf_path, threads=threads)


def _run_sqtl(
    disc, cache, covariates, psi_matrix=None, junction_ids=None, gene_map=None
):
    """3d — sQTL scan: haplotype scan on PSI (percent-spliced-in) phenotypes.

    reuses the same scan engine (scan_nb or scan_h2lmm) with PSI ratios
    as the phenotype instead of expression counts. compatible with both
    short-read (LeafCutter junctions) and long-read (SQANTI3 isoforms).

    PSI is bounded [0,1] — use LMM (after RINT) or beta-regression;
    NB GLM is inappropriate for PSI. model defaults to LMM.

    parameters
    ----------
    psi_matrix   : np.ndarray (n, J)  PSI per junction per sample
    junction_ids : list of str        length J
    gene_map     : dict {junction_id: gene_id}  for grouping output

    returns
    -------
    dict {junction_id: ScanResult}
    """
    if psi_matrix is None or junction_ids is None:
        logger.warning("sqtl: psi_matrix and junction_ids required — skipping")
        return {}
    if disc.kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(cache, method="loco")
        kd = kin_module.decomp_kinship(kd_raw)
    else:
        kd = disc.kinship_decomp
    results = {}
    for j, jid in enumerate(junction_ids):
        psi_j = psi_matrix[:, j]
        if np.all(np.isnan(psi_j)):
            continue
        results[jid] = scan_h2lmm(
            psi_j, cache, covariates, kinship_decomp=kd, phenotype_id=jid
        )
    return results
