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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pathlib import Path
    from .genotype.genomecache import GenomeCache

import pandas as pd
from scipy import stats as _sp_stats

from .qtl.scan import scan_h2lmm, scan_multi_gene
from .qtl.mediation import MEDIATION_COLS, FIXEF_MEDIATION_COLS
from .qtl.snp_scan import SNP_SCAN_COLS
from .qtl.ci import CI_COLS

from .genotype import kinship as kin_module

logger = logging.getLogger("mosaiqtl")


# ─────────────────────────────────────────────────────────────────────────────
# canonical output schemas
# gene_id is the stable join key (ENSMUSG ID, version-stripped).
# gene_symbol is display-only (e.g. 'Ankrd39'); not guaranteed unique.
# ─────────────────────────────────────────────────────────────────────────────

LOD_COLS: list[str] = ["gene_id", "gene_symbol", "chr", "locus", "pos_mb", "lod"]

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
    "pval_perm",
    "fdr",
    "fdr_method",
    "significance",
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
#
# Two source sets feed the composite schemas below:
#
#   _SCAN_COLS   — produced by the haplotype scan; available immediately after
#                  scan_chromosome() returns. No allele DB lookup required.
#
#   _ALLELE_COLS — joined from sanger_alleles.h5 (built from the MGP VCF by
#                  genotype/snp_db.py) after the scan. variant (SPDI), rsid,
#                  ref/alt alleles, founder carrier sets, dosage mode, allele
#                  frequencies, and dosage regression betas are all Sanger-derived
#                  and are NOT available at haplotype-scan time.
#
# Join key: (chr, pos) — GigaMUGA grid position matches Sanger SNP position after
# interpolation (genotype/interpolation.py). _SCAN_COLS carries chr+pos; _ALLELE_COLS
# carries variant+rsid which encode the same coordinate in SPDI form.

# Columns produced by the haplotype scan itself — no Sanger annotation needed.
_SCAN_COLS: list[str] = [
    # ── locus coordinate ──────────────────────────────────────────────────────
    "chr",            # chromosome with 'chr' prefix (GRCm39/mm39)
    "pos",            # integer bp (GRCm39/mm39 — mm39-grounded, not lifted)
    # ── gene identity ─────────────────────────────────────────────────────────
    "gene_id",        # ENSMUSG ID (version suffix stripped); IGVF spec field 'gene'
    "gene_symbol",    # gene symbol e.g. 'Ankrd39' (display only)
    # ── association statistics ─────────────────────────────────────────────────
    "lod",            # joint 7-df LOD score (8 founders)
    "h2_null_chr",    # LOCO null heritability (per-chromosome constant)
    "pval_nominal",   # chi²(7 df) p-value
    "log10p_nominal", # -log10(pval_nominal)
    "cis_dist_mb",    # signed distance TSS → locus (Mb)
    # ── model diagnostics ─────────────────────────────────────────────────────
    "sd_phenotype",   # sqrt(rss0 / (n_valid-1)); ~1 for log-norm pseudobulk counts
    "n_sample",       # n_valid from LMM null fit
    # ── scan bookkeeping ──────────────────────────────────────────────────────
    "peak_locus",     # 'T' for the argmax-LOD row per gene; 'F' otherwise
]

# Columns derived from Sanger/GigaMUGA allele DB — joined at peak-calling or
# fine-mapping time. effect_size / se_effect_size are dosage betas from the
# Sanger imputed biallelic regression, not from the 7-df haplotype scan.
_ALLELE_COLS: list[str] = [
    # ── locus identifier (Sanger-derived) ─────────────────────────────────────
    "variant",        # SPDI: {refseq_accn}:{pos_bp-1}:{ref}:{alt} (mm39-grounded)
    "rsid",           # dbSNP rsID from mgp_REL2021_snps_founders.rsID.vcf.gz; NaN if absent
    # ── VCF alleles ───────────────────────────────────────────────────────────
    "ref",            # GRCm39 reference allele (B6J is the mm39 reference strain)
    "alt",            # VCF alternative allele
    # ── statistical allele coding ─────────────────────────────────────────────
    "effect_allele",  # allele coded as the dosage predictor in the biallelic regression
    "other_allele",   # complementary allele (not always == ref; coding is data-driven)
    # ── founder carrier sets ──────────────────────────────────────────────────
    "ref_founders",   # comma-joined founders carrying ref allele e.g. 'AJ,B6J,129S1J'
    "alt_founders",   # comma-joined founders carrying alt allele
    # ── dosage mode and allele frequencies ────────────────────────────────────
    "dosage",         # 'T' = haplotype-prob imputed dosage; 'F' = hard calls (GigaMUGA peak)
    "af",             # allele frequency of effect_allele across 48 strains
    "maf",            # min(af, 1-af)
    "mac",            # round(maf × 48) at strain level (48 = 33 CC + 8 founders + 7 F1)
    "ma_sample",      # round(maf × n_sample) at replicate mouse level
    # ── dosage regression outputs (Sanger only; NaN at GigaMUGA haplotype level)
    "effect_size",    # dosage beta from Sanger biallelic regression
    "se_effect_size", # SE of dosage beta
]

# Per-founder haplotype effects — estimated at peak/lead-SNP locus (one row per eGene).
# 8 betas sum to ~0 per locus (contrast parameterisation from effects.py).
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

# Gene-level significance — from permutation calibration and Storey FDR; one value
# per gene, repeated on all per-locus rows for that gene.
_SIG_COLS: list[str] = [
    "pval_perm",    # Beta-calibrated per-gene permutation p-value
    "fdr",          # Storey q-value across all genes that cleared the Stage-1 GEV gate
    "fdr_method",   # constant 'qvalue'
    "significance", # 'T' if fdr <= fdr_threshold
]

_NULL_MODEL_COLS: list[str] = [
    "h2_null_chr",
    "n_sample",
    "sd_phenotype",
    "cis_dist_mb",
]

# Empty: main-effect-only analyses carry no interaction, interaction_context,
# conditional, or conditional_method columns.
_RUN_COLS: list[str] = []

# ── schema definitions ────────────────────────────────────────────────────────

# Haplotype scan — all cis loci × all scanned genes (per-chromosome intermediate).
# _ALLELE_COLS absent: sanger_alleles.h5 lookup happens post-scan at peak-calling time.
EQTL_SCAN_COLS: list[str] = _SCAN_COLS

# Haplotype peak — one row per eGene.
EQTL_PEAK_COLS: list[str] = [
    *_SCAN_COLS,            # chr, pos, gene_id, gene_symbol, LOD stats, model diagnostics
    *_FOUNDER_EFFECT_COLS,  # 8 per-founder betas at peak locus (effects.py)
    *_SIG_COLS,             # gene-level significance: pval_perm, fdr, fdr_method, significance
]

# Sanger fine-mapping — one row per credible-set member per eGene.
# Column order: variant identity → alleles → gene → fine-mapping posteriors
#               → association → significance → allele carriers/freq
#               → dosage regression → haplotype context → null model metadata.
# _FOUNDER_EFFECT_COLS excluded: founder betas are haplotype-peak properties, not per-SNP.
EQTL_FINEMAPPING_COLS: list[str] = [
    # variant identity (Sanger SPDI → coordinate)
    "variant", "rsid", "chr", "pos", "ref", "alt",
    # gene context
    "gene_id", "gene_symbol", "cis_dist_mb",
    # fine-mapping posteriors (the primary result)
    "pip", "CS", "lead_snp", "pip_method",
    # association statistics
    "lod", "pval_nominal", "log10p_nominal",
    # gene-level significance
    "pval_perm", "fdr", "fdr_method", "significance",
    # allele coding (effect/other allele + founder carrier sets)
    "effect_allele", "other_allele", "ref_founders", "alt_founders",
    # dosage, frequency, effect
    "dosage", "af", "maf", "mac", "ma_sample", "effect_size", "se_effect_size",
    # haplotype-scan back-reference
    "hap_peak_variant", "hap_peak_pos", "hap_peak_lod", "ci_lo_bp", "ci_hi_bp",
    # null model metadata
    "h2_null_chr", "sd_phenotype", "n_sample",
]

# Lead SNP PIP below this value indicates the locus did not resolve to a credible
# causal SNP — the haplotype signal is real but no Sanger SNP in the CI window
# captures it (missing variant, complex allele pattern, or low LD in CC strains).
CS_RESOLVED_PIP_THRESHOLD: float = 0.10

# Sanger fine-mapping peak — one row per eGene (highest-PIP CS member).
# lead_snp dropped (redundant for a single-row-per-gene file); cs_resolved flags
# whether the locus resolved to a credible SNP candidate (lead pip >= threshold);
# founder betas added here (haplotype-peak property, not per-SNP).
_FM_PEAK_BASE: list[str] = [c for c in EQTL_FINEMAPPING_COLS if c != "lead_snp"]
_pip_idx = _FM_PEAK_BASE.index("pip") + 1
EQTL_FINEMAPPING_PEAK_COLS: list[str] = (
    _FM_PEAK_BASE[:_pip_idx]
    + ["cs_resolved"]
    + _FM_PEAK_BASE[_pip_idx:]
    + _FOUNDER_EFFECT_COLS
)

# Imputed SNP association scan inside each eGene's LOD-drop CI — the input to
# Sanger fine-mapping.  Haplotype-peak context columns, then the per-SNP scan
# statistics (SNP_SCAN_COLS minus the duplicated gene_id/chr).
IMPUTED_SNP_COLS: list[str] = [
    "gene_id",
    "gene_symbol",
    "chr",
    "hap_peak_pos",
    "hap_peak_variant",
    "hap_peak_lod",
    "ci_lo_bp",
    "ci_hi_bp",
] + [c for c in SNP_SCAN_COLS if c not in ("gene_id", "chr")] + [
    "rsid",
    "ref",
    "alt",
    "effect_allele",
    "other_allele",
    "ref_founders",
    "alt_founders",
    "dosage",
    "af",
    "maf",
    "effect_size",
    "se_effect_size",
]

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

SCAN_TYPE_TO_WINDOW: dict[str, str] = {'L': 'local', 'C': 'cis', 'G': 'genome'}

__all__ = [
    "SCAN_TYPE_TO_WINDOW",
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
    "CS_RESOLVED_PIP_THRESHOLD",
    "EQTL_FINEMAPPING_COLS",
    "EQTL_FINEMAPPING_PEAK_COLS",
    "EQTL_STATS_COLS",
    "IMPUTED_SNP_COLS",
    "call_egenes",
    "calibrate_egenes_storey",
    "cap_egene_ci",
    "enrich_peak_lod",
    "finalize_egene_summary",
    "prepare_egenes_for_finemap",
    "impute_egene",
    "impute_egenes",
    "finemap_imputed",
    "finemap_egenes",
    "discover",
    "characterize",
    "resolve",
    "DiscoveryResult",
    "CharacterizationResult",
    "ResolutionResult",
]


def enrich_peak_lod(
    peak_lod_df: pd.DataFrame,
    gene_info_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join gene_symbol and start_mb from gene_info and compute cis_dist_mb.

    Parameters
    ----------
    peak_lod_df :
        Output of a scan merge step (PEAK_LOD_COLS subset).
    gene_info_df :
        Gene annotation table with at minimum ``gene_id`` and ``start`` columns.

    Returns
    -------
    pd.DataFrame
        ``peak_lod_df`` enriched with ``gene_symbol``, ``start_mb``, and
        ``cis_dist_mb`` columns (where data is available).
    """
    gi_cols = ['gene_id']
    if 'gene_symbol' in gene_info_df.columns and 'gene_symbol' not in peak_lod_df.columns:
        gi_cols.append('gene_symbol')
    if 'start' in gene_info_df.columns:
        gi_cols.append('start')
    gi_sub = gene_info_df[gi_cols].drop_duplicates('gene_id')
    if 'start' in gi_sub.columns:
        gi_sub = gi_sub.assign(start_mb=gi_sub['start'] / 1e6).drop(columns=['start'])
    out = peak_lod_df.merge(gi_sub, on='gene_id', how='left')
    if 'start_mb' in out.columns and 'peak_mb' in out.columns:
        out['cis_dist_mb'] = out['peak_mb'] - out['start_mb']
    return out


def prepare_egenes_for_finemap(egenes_df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw scan columns to the names expected by :func:`finemap_egenes`.

    ``peak_lod.csv`` carries raw scan column names (``h2``, ``n_valid``,
    ``rss0``).  The fine-mapping pipeline reads ``h2_null_chr``, ``n_sample``,
    and ``sd_phenotype``.  This function adds the aliased columns when the
    source columns are present and the target columns are absent.

    Parameters
    ----------
    egenes_df :
        eGene DataFrame, typically loaded from a merged ``peak_lod.csv`` or
        ``eqtl_summary.csv``.

    Returns
    -------
    pd.DataFrame
        Copy of ``egenes_df`` with alias columns added.
    """
    df = egenes_df.copy()
    if 'h2_null_chr' not in df.columns and 'h2' in df.columns:
        df['h2_null_chr'] = df['h2']
    if 'n_sample' not in df.columns and 'n_valid' in df.columns:
        df['n_sample'] = df['n_valid']
    if ('sd_phenotype' not in df.columns
            and 'rss0' in df.columns
            and 'n_valid' in df.columns):
        df['sd_phenotype'] = np.sqrt(
            df['rss0'] / (df['n_valid'] - 1).clip(lower=1)
        )
    return df


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


def cap_egene_ci(
    egene_df: pd.DataFrame,
    scan_type: str,
    window_mb: float,
    gene_tss_df: 'pd.DataFrame | None' = None,
    gene_info_df: 'pd.DataFrame | None' = None,
) -> pd.DataFrame:
    """Cap ci_lo_bp / ci_hi_bp to ±window_mb from TSS for L-mode scans.

    Returns the DataFrame unchanged for C/G modes or when window_mb == 0.
    TSS source priority: gene_tss_df (strand-aware, joined via gene_symbol) >
    gene_info_df start/end/strand > start_mb column > peak_mb fallback.

    Parameters
    ----------
    egene_df : DataFrame with gene_id, peak_mb, ci_lo_bp, ci_hi_bp columns.
    scan_type : 'L', 'C', or 'G'.
    window_mb : half-window in Mb for L-mode scans.
    gene_tss_df : optional strand-aware TSS table (gene_name + tss_mb, or
        gene_name + start/end/strand from which tss_mb is derived).
    gene_info_df : optional gene info table (gene_id, gene_symbol, start, end,
        strand); used as secondary TSS source and for gene_id→gene_symbol lookup
        when gene_symbol is absent from egene_df.
    """
    if scan_type != 'L' or window_mb <= 0:
        return egene_df.copy()

    df = egene_df.copy()
    _tss_mb: 'pd.Series | None' = None

    if gene_tss_df is not None:
        _gtss = gene_tss_df.copy()
        _cols = set(_gtss.columns)
        if {'start', 'end', 'strand'} <= _cols:
            _gtss['tss_mb'] = np.where(
                _gtss['strand'] == '-', _gtss['end'] / 1e6, _gtss['start'] / 1e6
            )
        elif 'start' in _cols and 'tss_mb' not in _cols:
            _gtss['tss_mb'] = _gtss['start'] / 1e6
        if 'gene_name' in _gtss.columns and 'tss_mb' in _gtss.columns:
            _sym_to_tss = _gtss.groupby('gene_name')['tss_mb'].mean()
            if 'gene_symbol' in df.columns:
                _tss_mb = df['gene_symbol'].map(_sym_to_tss)
            elif gene_info_df is not None and 'gene_symbol' in gene_info_df.columns:
                _gid_to_sym = gene_info_df.set_index('gene_id')['gene_symbol']
                _tss_mb = df['gene_id'].map(_gid_to_sym).map(_sym_to_tss)

    if _tss_mb is None or _tss_mb.isna().all():
        if gene_info_df is not None and 'start' in gene_info_df.columns:
            _has_strand = 'strand' in gene_info_df.columns
            _gi_tss = gene_info_df.set_index('gene_id').apply(
                lambda r: (float(r['end']) / 1e6
                           if _has_strand and r.get('strand') == '-'
                           else float(r['start']) / 1e6),
                axis=1,
            )
            _tss_mb = df['gene_id'].map(_gi_tss)
        elif 'start_mb' in df.columns:
            _tss_mb = df['start_mb']

    _peak_mb = df['peak_mb'] if 'peak_mb' in df.columns else pd.Series(0.0, index=df.index)
    _tss_fill = _tss_mb if _tss_mb is not None else _peak_mb
    _tss_bp = (_tss_fill.fillna(_peak_mb) * 1e6).round().astype('Int64')
    _half = int(window_mb * 1e6)
    _cap_lo = (_tss_bp - _half).clip(lower=0)
    _cap_hi = _tss_bp + _half

    if 'ci_lo_bp' in df.columns:
        df['ci_lo_bp'] = df['ci_lo_bp'].fillna(_cap_lo).combine(_cap_lo, max)
    if 'ci_hi_bp' in df.columns:
        df['ci_hi_bp'] = df['ci_hi_bp'].fillna(_cap_hi).combine(_cap_hi, min)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# eGene utilities
# ─────────────────────────────────────────────────────────────────────────────


def call_egenes(
    peak_lod_df: pd.DataFrame,
    threshold: float,
    effects_df: "pd.DataFrame | None" = None,
    lod_col: str = "peak_lod",
) -> pd.DataFrame:
    """Apply a genome-wide LOD threshold — the coarse Stage-1 promotion gate.

    A raw threshold comparison is cheap but treats every gene as though it
    shares one null distribution, which is not statistically valid for a
    final FDR call. Genes cleared here are candidates for the expensive
    per-gene Beta+Storey calibration in calibrate_egenes_storey(), which
    produces the real significance call; this function alone is not a
    final eGene list.

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
    peak_lod, ...] — one row per gene clearing the Stage-1 gate; positional
    columns NaN only if absent from both inputs
    """
    egenes = peak_lod_df[peak_lod_df[lod_col] >= threshold].copy()
    pos_cols = ("chr", "peak_locus", "peak_mb")
    missing = [c for c in pos_cols if c not in egenes.columns]
    if missing and effects_df is not None and len(effects_df) > 0:
        fill_cols = [c for c in ("gene_id", *missing) if c in effects_df.columns]
        if len(fill_cols) > 1:
            egenes = egenes.merge(effects_df[fill_cols], on="gene_id", how="left")
    return egenes.reset_index(drop=True)


def calibrate_egenes_storey(
    egenes_df: pd.DataFrame,
    pval_perm: "dict[str, float] | pd.Series",
    gene_col: str = "gene_id",
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Populate the significance block (_SIG_COLS) via per-gene Beta + Storey.

    Stage-2 of eGene calling: takes genes that already cleared call_egenes'
    Stage-1 threshold gate, attaches their per-gene Beta-calibrated
    permutation p-value (produced upstream, one call to
    qtl.permutation.calibrate_gene_pvalue per gene), estimates π₀ and
    q-values across them via Storey's method, and gates the final
    significance call on q ≤ fdr_threshold rather than the raw LOD.

    Parameters
    ----------
    egenes_df     : DataFrame from call_egenes — must contain gene_col.
    pval_perm     : dict or Series {gene_col value: p_adj} — the calibrated
        p-value for each gene, as returned by
        qtl.permutation.calibrate_gene_pvalue.
    gene_col      : join key present in both egenes_df and pval_perm.
    fdr_threshold : q-value cutoff for significance (default 0.05).

    Returns
    -------
    DataFrame — egenes_df with pval_perm, fdr, fdr_method, significance
    columns added/overwritten (_SIG_COLS).
    """
    from .qtl import fdr as fdr_module

    out = egenes_df.copy()
    pv = pval_perm if isinstance(pval_perm, pd.Series) else pd.Series(pval_perm)
    out["pval_perm"] = out[gene_col].map(pv)

    q_values, pi0 = fdr_module.storey_qvalue(out["pval_perm"].to_numpy())
    out["fdr"] = q_values
    out["fdr_method"] = "qvalue"
    out["significance"] = np.where(out["fdr"] <= fdr_threshold, "T", "F")

    logger.info(
        "calibrate_egenes_storey: n_genes=%d  pi0=%.4f  n_egenes(fdr<=%.2f)=%d",
        len(out),
        pi0,
        fdr_threshold,
        int((out["significance"] == "T").sum()),
    )
    return out


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
    scan_window="genome",
    gene_chr_map=None,
    gene_pos_mb_map=None,
    window_mb=10.0,
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
        'lmm' — gaussian lmm scan with loco kinship correction; validated port
    lib_sizes        : np.ndarray (n,) or None  (nb only) log-offset
    cis_loci_dict    : dict {gene_id: set} or None  (nb only) explicit locus
                       subset per gene, passed straight to qtl/nb.py::scan_nb
    n_cores          : int
    kinship_decomp   : dict or None  pre-computed LOCO kinship (lmm only);
                       computed internally if None
    scan_window      : str  'genome' | 'cis' | 'local'  (lmm only; default
                       'genome'), passed straight to scan_multi_gene
    gene_chr_map     : dict {gene_id: str} or None  (lmm only) required for
                       scan_window='cis' or 'local'
    gene_pos_mb_map  : dict {gene_id: float} or None  (lmm only) TSS in Mb;
                       required for scan_window='local'
    window_mb        : float  (lmm only) half-window in Mb for
                       scan_window='local' (default 10.0)

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
            gene_chr_map=gene_chr_map,
            covariates=covariates,
            kinship_decomp=kinship_decomp,
            sample_ids=sample_ids,
            scan_window=scan_window,
            gene_pos_mb_map=gene_pos_mb_map,
            window_mb=window_mb,
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
    from .genotype.snp_db import build_snp_catalogue

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


# ── Layer 4 — Sanger fine-mapping ────────────────────────────────────────────


def impute_egene(
    gene_id: str,
    chr_: str,
    phenotype: np.ndarray,
    genomecache: "GenomeCache",
    alleles_df: pd.DataFrame,
    hap_peak_pos: int,
    hap_peak_variant: str,
    hap_peak_lod: float,
    ci_lo_bp: int,
    ci_hi_bp: int,
    covariates: Optional[np.ndarray] = None,
    kinship_decomp: Optional[dict] = None,
    sample_ids: Optional[list] = None,
    founder_order: Optional[list] = None,
    ref_founder: str = "B6J",
    gene_symbol: str = "",
) -> Optional[pd.DataFrame]:
    """Imputed SNP association scan for one eGene within its LOD-drop CI.

    Filters alleles_df to the CI window, collapses each SNP to a biallelic
    founder ALT mask, imputes per-strain dosages from the haplotype
    probabilities and tests each dosage with the LMM.

    Args:
        gene_id: Ensembl gene ID.
        chr_: Chromosome key matching genomecache.
        phenotype: Expression vector (n,), sample-level when sample_ids is
            given, else in cache strain order.
        genomecache: Loaded GenomeCache; probabilities load lazily per chr.
        alleles_df: Output of load_founder_alleles for chr_.
        hap_peak_pos: Haplotype-scan peak position (bp).
        hap_peak_variant: Locus ID of the haplotype peak.
        hap_peak_lod: LOD at the haplotype peak.
        ci_lo_bp: LOD-drop CI lower bound (bp).
        ci_hi_bp: LOD-drop CI upper bound (bp).
        covariates: Covariate matrix (n, p) aligned with phenotype, or None.
        kinship_decomp: LOCO kinship decomposition keyed by chromosome.
        sample_ids: Genotype ID per element of phenotype; replicate inputs
            are reduced to strain means before the scan.
        founder_order: Canonical founder names; cache founders when None.
        ref_founder: REF founder for the biallelic collapse.
        gene_symbol: Display gene name carried into the output.

    Returns:
        DataFrame in IMPUTED_SNP_COLS layout, or None when no polymorphic SNP
        falls within the CI window.
    """
    from .genotype.snp_db import make_alt_mask
    from .qtl.snp_scan import imputed_snp_scan_h2lmm

    window = alleles_df[
        (alleles_df["pos_bp"] >= ci_lo_bp) & (alleles_df["pos_bp"] <= ci_hi_bp)
    ].reset_index(drop=True)
    if window.empty:
        logger.debug(
            "impute_egene: %s — no SNPs in CI window [%d, %d]",
            gene_id,
            ci_lo_bp,
            ci_hi_bp,
        )
        return None

    f_order = founder_order if founder_order is not None else list(genomecache.founders)
    alt_mask = make_alt_mask(window, f_order, ref_founder=ref_founder)
    snp_genos = pd.DataFrame(alt_mask, index=window["marker"].values, columns=f_order)
    snp_genos["chr"] = window["chr"].values
    snp_genos["pos_mb"] = window["pos_mb"].values

    snp_df = imputed_snp_scan_h2lmm(
        phenotype,
        genomecache,
        snp_genos,
        covariates,
        kinship_decomp,
        phenotype_id=gene_id,
        gene_symbol=gene_symbol,
        sample_ids=sample_ids,
    )
    if snp_df is None or snp_df.empty:
        return None

    # extract LD data before IMPUTED_SNP_COLS slice drops attrs and internal cols
    _ld_data = snp_df.attrs.get("_ld")
    _snp_patterns = (
        snp_df["_founder_pattern"].tolist()
        if "_founder_pattern" in snp_df.columns
        else None
    )

    snp_df = snp_df.assign(
        gene_symbol=gene_symbol,
        chr=str(chr_),
        hap_peak_pos=int(hap_peak_pos),
        hap_peak_variant=hap_peak_variant,
        hap_peak_lod=float(hap_peak_lod),
        ci_lo_bp=int(ci_lo_bp),
        ci_hi_bp=int(ci_hi_bp),
    )

    ref_f = ref_founder if ref_founder in f_order else (f_order[0] if f_order else None)
    if ref_f is not None and set(f_order).issubset(set(window.columns)):
        ann_rows: list[dict] = []
        for _, arow in window.iterrows():
            ref_allele = str(arow.get(ref_f, ""))
            f_alleles = {f: str(arow.get(f, "")) for f in f_order}
            valid = {f: a for f, a in f_alleles.items() if a and a != "N"}
            alt_set = {a for a in valid.values() if a != ref_allele}
            alt_allele = next(iter(alt_set), "")
            ann_rows.append(
                {
                    "snp": arow["marker"],
                    "rsid": arow.get("rsid", pd.NA),
                    "ref": ref_allele,
                    "alt": alt_allele,
                    # dosage predictor is always the non-REF allele (make_alt_mask codes
                    # 1 = "differs from ref_founder"); holds for any ref_founder choice
                    "effect_allele": alt_allele,
                    "other_allele": ref_allele,
                    "dosage": "T",   # haplotype-probability imputed (not hard calls)
                    "ref_founders": ",".join(
                        f for f in f_order if valid.get(f) == ref_allele
                    ),
                    "alt_founders": ",".join(
                        f for f in f_order if valid.get(f) != ref_allele and f in valid
                    ),
                }
            )
        ann_df = pd.DataFrame(ann_rows)
        _snp_idx = snp_df.set_index("snp")
        # af/maf/mac/ma_sample: mean haplotype-probability-weighted dosage
        # across the CC sample, giving the realized ALT allele frequency in
        # this cohort at each SNP position.
        _dosage_mean = ann_df["snp"].map(_snp_idx["dosage_mean"])
        _maf_col = _dosage_mean.apply(
            lambda x: min(x, 1.0 - x) if pd.notna(x) else float("nan")
        )
        _n_col = ann_df["snp"].map(_snp_idx["n"])
        ann_df["af"]        = _dosage_mean
        ann_df["maf"]       = _maf_col
        ann_df["mac"]       = (_maf_col * _n_col).round().astype("Int64")
        ann_df["ma_sample"] = ann_df["mac"]
        ann_df["effect_size"] = ann_df["snp"].map(_snp_idx["beta_dosage"])
        # Wald inversion: se = |beta| / |z|, z = sqrt(2·ln10·lod)
        _beta_abs = ann_df["snp"].map(_snp_idx["beta_dosage"]).abs()
        _lod_map = ann_df["snp"].map(_snp_idx["lod"]).clip(lower=0.0)
        _z_abs = np.sqrt(2.0 * np.log(10) * _lod_map)
        ann_df["se_effect_size"] = (_beta_abs / _z_abs.where(_z_abs > 1e-10)).where(
            _beta_abs > 1e-10, other=np.nan
        )
        snp_df = snp_df.merge(ann_df, on="snp", how="left")
    else:
        for col in ("ref", "alt", "ref_founders", "alt_founders", "af", "maf",
                    "mac", "ma_sample", "effect_size", "se_effect_size"):
            snp_df[col] = pd.NA

    result_df = snp_df[IMPUTED_SNP_COLS].copy()
    # attach LD payload on the result so impute_egenes can accumulate it
    if _ld_data is not None and _snp_patterns is not None:
        result_df.attrs["_ld"] = _ld_data
        result_df.attrs["_snp_patterns"] = _snp_patterns
    return result_df


def impute_egenes(
    egene_df: pd.DataFrame,
    phenotype_matrix: np.ndarray,
    gene_ids: list,
    genomecache: "GenomeCache",
    alleles_h5: "Path | str",
    covariates: Optional[np.ndarray] = None,
    kinship_decomp: Optional[dict] = None,
    sample_ids: Optional[list] = None,
    founder_order: Optional[list] = None,
    ref_founder: str = "B6J",
    n_cores: int = 1,
) -> pd.DataFrame:
    """Imputed SNP association scan for every eGene within its LOD-drop CI.

    Founder alleles are loaded once per chromosome; eGenes on chromosomes
    absent from the allele database are skipped with a warning.

    Args:
        egene_df: One row per eGene with gene_id, chr, pos, lod, variant,
            ci_lo_bp and ci_hi_bp (optional gene_symbol).
        phenotype_matrix: Expression matrix (n_genes, n_samples).
        gene_ids: Gene IDs matching axis 0 of phenotype_matrix.
        genomecache: Loaded GenomeCache.
        alleles_h5: Founder allele database built by snp_db.
        covariates: Covariate matrix (n_samples, p) or None.
        kinship_decomp: LOCO kinship decomposition keyed by chromosome.
        sample_ids: Genotype ID per sample column of phenotype_matrix.
        founder_order: Canonical founder names; cache founders when None.
        ref_founder: REF founder for the biallelic collapse.

    Returns:
        DataFrame in IMPUTED_SNP_COLS layout (empty with those columns when
        no SNP was tested).
    """
    from .genotype.snp_db import load_founder_alleles

    _required = {"gene_id", "chr", "pos", "lod", "variant", "ci_lo_bp", "ci_hi_bp"}
    _missing = _required - set(egene_df.columns)
    if _missing:
        raise ValueError(f"egene_df missing required columns: {sorted(_missing)}")

    gene_idx = {g: i for i, g in enumerate(gene_ids)}
    f_order = founder_order if founder_order is not None else list(genomecache.founders)
    eg = egene_df.dropna(subset=["ci_lo_bp", "ci_hi_bp"]).copy()
    eg["chr"] = eg["chr"].astype(str)

    n_total = len(eg)
    n_done = 0
    frames: list = []
    ld_by_gene: dict = {}  # gene_id → {ld: ..., snp_patterns: ...}
    for chrom, chrom_df in eg.groupby("chr", sort=False):
        chrom = str(chrom)
        logger.info(
            "impute_egenes: loading alleles for chr%s (%d eGenes)", chrom, len(chrom_df)
        )
        try:
            alleles_chr = load_founder_alleles(alleles_h5, chrom=chrom)
        except KeyError:
            logger.warning("impute_egenes: chr%s not in alleles file — skipping", chrom)
            continue

        for _, row in chrom_df.iterrows():
            gid = str(row["gene_id"])
            if gid not in gene_idx:
                logger.warning("impute_egenes: %s not in gene_ids — skipping", gid)
                continue
            n_done += 1
            sym = row.get("gene_symbol", gid)
            _label = sym if (sym and not pd.isna(sym) and sym != gid) else gid
            logger.info(
                "impute_egenes: chr%s [%d/%d] %s",
                chrom, n_done, n_total, _label,
            )
            result = impute_egene(
                gene_id=gid,
                chr_=chrom,
                phenotype=phenotype_matrix[gene_idx[gid]],
                genomecache=genomecache,
                alleles_df=alleles_chr,
                hap_peak_pos=int(row["pos"]),
                hap_peak_variant=str(row["variant"]),
                hap_peak_lod=float(row["lod"]),
                ci_lo_bp=int(row["ci_lo_bp"]),
                ci_hi_bp=int(row["ci_hi_bp"]),
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                sample_ids=sample_ids,
                founder_order=f_order,
                ref_founder=ref_founder,
                gene_symbol="" if pd.isna(sym) else str(sym),
            )
            if result is not None and not result.empty:
                _ld = result.attrs.get("_ld")
                _sp = result.attrs.get("_snp_patterns")
                if _ld is not None and _sp is not None:
                    ld_by_gene[gid] = {"ld": _ld, "snp_patterns": _sp}
                result.attrs.clear()
                frames.append(result)

    if not frames:
        return pd.DataFrame(columns=IMPUTED_SNP_COLS), {}
    return pd.concat(frames, ignore_index=True), ld_by_gene


def finemap_imputed(
    imputed_df: pd.DataFrame,
    out_dir: "Path | str",
    prior_var: float = 0.04,
    coverage: float = 0.95,
    tag: str = "",
    egene_df: Optional[pd.DataFrame] = None,
    ld_by_gene: Optional[dict] = None,
) -> tuple:
    """Wakefield-ABF credible sets from an imputed SNP scan, written to disk.

    Args:
        imputed_df: Output of impute_egenes (IMPUTED_SNP_COLS layout).
        out_dir: Directory for eqtl_finemapping{_tag}.csv and
            eqtl_finemapping_peak{_tag}.csv.
        prior_var: Wakefield prior variance on the standardised effect.
        coverage: Credible-set coverage.
        tag: Suffix appended to output filenames.
        egene_df: One row per eGene; supplies ``_FOUNDER_EFFECT_COLS``
            (``effect_{founder}`` columns), ``_SIG_COLS``
            (``pval_perm``, ``fdr``, ``fdr_method``, ``significance``), and
            ``_NULL_MODEL_COLS`` (``h2_null_chr``, ``n_sample``,
            ``sd_phenotype``, ``cis_dist_mb``).
            These are broadcast to every credible-set row for the matching
            ``gene_id``.  Missing rows are filled with ``pd.NA``.
            Accepts ``eqtl_peak.csv`` (``effect_`` prefix) directly; also
            accepts a DataFrame with ``beta_`` prefix (renamed on the fly).

    Returns:
        Tuple (finemapping_df, finemapping_peak_df).
    """
    from pathlib import Path as _Path
    from .qtl.imputation import fine_map_region
    from .io.results_io import (
        _atomic_write_csv,
        build_eqtl_finemapping_df,
        build_eqtl_finemapping_peak_df,
    )

    _founder_names = [c[len("effect_"):] for c in _FOUNDER_EFFECT_COLS]

    egene_idx: dict = {}
    if egene_df is not None:
        eg = egene_df.copy()
        for f in _founder_names:
            if f"beta_{f}" in eg.columns and f"effect_{f}" not in eg.columns:
                eg[f"effect_{f}"] = eg[f"beta_{f}"]
        egene_idx = {str(g): row for g, row in eg.set_index("gene_id").iterrows()}

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _fm_genes = imputed_df["gene_id"].unique()
    _fm_total = len(_fm_genes)
    logger.info("finemap_imputed: ABF credible sets for %d genes", _fm_total)

    fm_frames: list = []
    fp_frames: list = []
    for _fm_i, (gid, g) in enumerate(imputed_df.groupby("gene_id", sort=False), 1):
        first = g.iloc[0]
        _sym = first.get("gene_symbol", gid)
        _flabel = _sym if (_sym and not pd.isna(_sym) and _sym != gid) else gid
        logger.info(
            "finemap_imputed: [%d/%d] %s (%d SNPs)",
            _fm_i, _fm_total, _flabel, len(g),
        )
        # build LD groups from per-gene founder-pattern r² when available
        _ld_groups: Optional[dict] = None
        _gene_ld = (ld_by_gene or {}).get(str(gid))
        if _gene_ld is not None:
            from .qtl.imputation import merge_ld_groups as _merge_ld
            _ld_groups = _merge_ld(
                rep_patterns=_gene_ld["ld"]["rep_patterns"],
                r2_matrix=_gene_ld["ld"]["r2_matrix"],
                snp_patterns=_gene_ld["snp_patterns"],
            )

        fm_df = fine_map_region(
            lod=g["lod"].to_numpy(dtype=float),
            beta_dosage=g["beta_dosage"].to_numpy(dtype=float),
            n=int(first["n"]),
            snp_ids=g["snp"].to_numpy(),
            pos_mb=g["pos_mb"].to_numpy(dtype=float),
            ld_groups=_ld_groups,
            prior_var=prior_var,
            coverage=coverage,
        )
        sym = first.get("gene_symbol", "")

        founder_effects: Optional[dict] = None
        sig_values: Optional[dict] = None
        null_vals: Optional[dict] = None
        eg_row = egene_idx.get(str(gid))
        if eg_row is not None:
            founder_effects = {
                fc: float(eg_row[fc])
                for fc in _FOUNDER_EFFECT_COLS
                if fc in eg_row.index and not pd.isna(eg_row[fc])
            }
            sig_values = {
                c: eg_row[c]
                for c in _SIG_COLS
                if c in eg_row.index and not pd.isna(eg_row[c])
            }
            null_vals = {
                c: eg_row[c]
                for c in _NULL_MODEL_COLS
                if c in eg_row.index and not pd.isna(eg_row[c])
            }

        _alleles_ann: Optional[pd.DataFrame] = None
        _ann_cols = [c for c in (
            "rsid", "ref", "alt", "effect_allele", "other_allele",
            "ref_founders", "alt_founders", "dosage",
            "af", "maf", "effect_size", "se_effect_size",
        ) if c in g.columns]
        if _ann_cols and "snp" in g.columns:
            _cs_snps = set(fm_df[fm_df["in_cs"]]["snp"].tolist()) if "snp" in fm_df.columns else set()
            if _cs_snps:
                _ann_sub = g[g["snp"].isin(_cs_snps)][["snp"] + _ann_cols].copy()
                if not _ann_sub.empty:
                    _alleles_ann = _ann_sub

        result = build_eqtl_finemapping_df(
            fm_df,
            str(gid),
            str(first["chr"]),
            int(first["hap_peak_pos"]),
            str(first["hap_peak_variant"]),
            float(first["hap_peak_lod"]),
            ci_lo_bp=int(first["ci_lo_bp"]),
            ci_hi_bp=int(first["ci_hi_bp"]),
            gene_symbol="" if pd.isna(sym) else str(sym),
            alleles_df=_alleles_ann,
            founder_effects=founder_effects,
            sig_values=sig_values,
            null_model_values=null_vals,
        )
        if result.empty:
            continue
        fm_frames.append(result)
        peak_row = build_eqtl_finemapping_peak_df(result)
        if not peak_row.empty:
            fp_frames.append(peak_row)

    fm_out = (
        pd.concat(fm_frames, ignore_index=True)
        if fm_frames
        else pd.DataFrame(columns=EQTL_FINEMAPPING_COLS)
    )
    fp_out = (
        pd.concat(fp_frames, ignore_index=True)
        if fp_frames
        else pd.DataFrame(columns=EQTL_FINEMAPPING_PEAK_COLS)
    )

    suffix = f"_{tag}" if tag else ""
    # fm_out intermediate carries _FOUNDER_EFFECT_COLS for build_eqtl_finemapping_peak_df;
    # trim to EQTL_FINEMAPPING_COLS (per-SNP schema) before writing to disk.
    _atomic_write_csv(fm_out.reindex(columns=EQTL_FINEMAPPING_COLS), out_dir / f"eqtl_finemapping{suffix}.csv")
    _atomic_write_csv(fp_out, out_dir / f"eqtl_finemapping_peak{suffix}.csv")
    logger.info(
        "finemap_imputed: %d CS SNPs across %d eGenes written to %s",
        len(fm_out),
        len(fp_out),
        out_dir,
    )
    return fm_out, fp_out


def finemap_egenes(
    egene_df: pd.DataFrame,
    phenotype_matrix: np.ndarray,
    gene_ids: list,
    genomecache: "GenomeCache",
    alleles_h5: "Path | str",
    out_dir: "Path | str",
    covariates: Optional[np.ndarray] = None,
    kinship_decomp: Optional[dict] = None,
    sample_ids: Optional[list] = None,
    founder_order: Optional[list] = None,
    ref_founder: str = "B6J",
    prior_var: float = 0.04,
    coverage: float = 0.95,
    tag: str = "",
) -> tuple:
    """Imputed SNP scan plus Wakefield-ABF fine-mapping for every eGene.

    Composition of impute_egenes and finemap_imputed.

    Args:
        egene_df: One row per eGene with gene_id, chr, pos, lod, variant,
            ci_lo_bp and ci_hi_bp.
        phenotype_matrix: Expression matrix (n_genes, n_samples).
        gene_ids: Gene IDs matching axis 0 of phenotype_matrix.
        genomecache: Loaded GenomeCache.
        alleles_h5: Founder allele database built by snp_db.
        out_dir: Output directory.
        covariates: Covariate matrix (n_samples, p) or None.
        kinship_decomp: LOCO kinship decomposition keyed by chromosome.
        sample_ids: Genotype ID per sample column of phenotype_matrix.
        founder_order: Canonical founder names; cache founders when None.
        ref_founder: REF founder for the biallelic collapse.
        prior_var: Wakefield prior variance.
        coverage: Credible-set coverage.
        tag: Suffix appended to output filenames.

    Returns:
        Tuple (finemapping_df, finemapping_peak_df).
    """
    imputed, ld_by_gene = impute_egenes(
        egene_df,
        phenotype_matrix,
        gene_ids,
        genomecache,
        alleles_h5,
        covariates=covariates,
        kinship_decomp=kinship_decomp,
        sample_ids=sample_ids,
        founder_order=founder_order,
        ref_founder=ref_founder,
    )
    return finemap_imputed(
        imputed, out_dir, prior_var=prior_var, coverage=coverage, tag=tag,
        egene_df=egene_df, ld_by_gene=ld_by_gene,
    )
