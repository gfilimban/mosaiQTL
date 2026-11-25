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

import os
import re
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
from .qtl.imputation import LD_GROUP_COLS

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
    "cis_dist_gene_tss_mb",
    "peak_lod",
    "peak_at_edge",
    "h2",
    "h2_at_boundary",
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
    "cis_dist_gene_tss_mb",
    "peak_lod",
    "peak_at_edge",
    "ci_lo",
    "ci_hi",
    "ci_width_mb",
    "ci_truncated",
    "h2",
    "perm_n",
    "chi2_haplotype",
    "p_haplotype",
    "pval_perm",
    "qval_storey",
    "qval_bh",
    "lfdr",
    "pi0",
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
    "af",             # mean imputed dosage of the effect allele over the gene's n_sample scanned samples
    "maf",            # min(af, 1-af)
    "mac",            # round(maf × n_sample): minor-allele count over the scanned samples
    "ma_sample",      # equal to mac
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

# Call rule -> value written to fdr_method.
_CALL_RULES: dict[str, str] = {"qvalue": "qvalue", "bh": "BH", "lfdr": "lfdr"}

# Gene-level significance — from permutation calibration and Storey FDR; one value
# per gene, repeated on all per-locus rows for that gene.
_SIG_COLS: list[str] = [
    "pval_perm",    # Beta-calibrated per-gene permutation p-value
    "qval_storey",  # Storey q-value across all tested genes
    "qval_bh",      # Benjamini-Hochberg adjusted p-value across all tested genes
    "lfdr",         # local false discovery rate: posterior probability the gene is null
    "pi0",          # estimated null fraction used by qval_storey and lfdr (one value per table)
    "fdr",          # the value selected by the call rule (qval_storey, qval_bh or lfdr)
    "fdr_method",   # 'qvalue', 'BH' or 'lfdr'
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
    "pip", "log_abf", "CS", "lead_snp", "pip_method",
    # association statistics
    "lod", "pval_nominal", "log10p_nominal",
    # gene-level significance
    "pval_perm", "qval_storey", "qval_bh", "lfdr", "pi0", "fdr", "fdr_method", "significance",
    # allele coding (effect/other allele + founder carrier sets)
    "effect_allele", "other_allele", "ref_founders", "alt_founders",
    # dosage, frequency, effect
    "dosage", "af", "maf", "mac", "ma_sample", "effect_size", "se_effect_size",
    # haplotype-scan back-reference
    "hap_peak_variant", "hap_peak_pos", "hap_peak_lod", "ci_lo_bp", "ci_hi_bp",
    # null model metadata
    "h2_null_chr", "sd_phenotype", "n_sample",
]

# LD-group fine-mapping: one row per LD group per eGene, ranked by group_pip (1 = top).
# group_pip sums to 1 over a gene's groups; in_cs marks the groups in the credible set.
EQTL_FINEMAPPING_GROUP_COLS: list[str] = ["gene_id", *LD_GROUP_COLS]

# The peak file lists only eGenes whose lead variant has pip above this value. Under the
# single-causal-variant model pip is the posterior probability that the variant is the
# causal one; variants that share a founder allele pattern have identical ABF and split
# that probability, so a single variant seldom reaches 0.5 in a dense window and 0.25
# keeps the variants carrying at least a quarter of the posterior. Every other eGene
# stays in the per-variant and LD-group tables.
PEAK_PIP_THRESHOLD: float = 0.25

# Sanger fine-mapping peak — one row per resolved eGene (its lead variant).
# lead_snp dropped (redundant for a single-row-per-gene file);
# founder betas added here (haplotype-peak property, not per-SNP).
_FM_PEAK_BASE: list[str] = [c for c in EQTL_FINEMAPPING_COLS if c != "lead_snp"]
EQTL_FINEMAPPING_PEAK_COLS: list[str] = _FM_PEAK_BASE + _FOUNDER_EFFECT_COLS

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
    "mac",
    "ma_sample",
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
    "scanned_chromosomes",
    "validate_finemap_frames",
    "MEDIATION_COLS",
    "FIXEF_MEDIATION_COLS",
    "SNP_SCAN_COLS",
    "CI_COLS",
    "EQTL_SCAN_COLS",
    "EQTL_PEAK_COLS",
    "PEAK_PIP_THRESHOLD",
    "EQTL_FINEMAPPING_COLS",
    "EQTL_FINEMAPPING_PEAK_COLS",
    "EQTL_FINEMAPPING_GROUP_COLS",
    "EQTL_STATS_COLS",
    "IMPUTED_SNP_COLS",
    "call_egenes",
    "calibrate_egenes",
    "call_significant_egenes",
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
    gene_tss_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join gene_symbol and TSS anchors from the annotation tables and compute cis_dist_mb.

    Genes join on the unversioned ``gene_id``, so versioned and unversioned
    tables match. ``start_mb`` is the gene's strand-aware TSS from ``gene_info_df``
    (``start`` on ``+``, ``end`` on ``-``) and is the anchor of the scan window.
    ``near_tss_mb`` is the strand-aware transcript TSS nearest the peak from
    ``gene_tss_df``. ``cis_dist_mb = peak_mb - near_tss_mb`` where a transcript
    TSS exists, else ``peak_mb - start_mb``, multiplied by -1 for minus-strand
    genes so that negative always means upstream of the TSS.
    ``cis_dist_gene_tss_mb`` is the same signed distance measured from one fixed
    TSS per gene, the most upstream transcript TSS in ``gene_tss_df``, so it does
    not change with the peak and is comparable across cell types.

    Parameters
    ----------
    peak_lod_df :
        Output of a scan merge step (PEAK_LOD_COLS subset).
    gene_info_df :
        Gene annotation table with ``gene_id`` and ``start``; ``end`` and ``strand``
        give the strand-aware TSS.
    gene_tss_df :
        Per-transcript table with ``gene_id``, ``start``, ``end`` and ``strand``,
        or None when no transcript table is available.

    Returns
    -------
    pd.DataFrame
        ``peak_lod_df`` enriched with ``gene_symbol``, ``start_mb``,
        ``near_tss_mb``, ``cis_dist_mb`` and ``cis_dist_gene_tss_mb`` (where data
        is available).
    """
    from .utils import (
        build_tss_lookup,
        gene_tss_anchor,
        nearest_transcript_tss_mb,
        strip_gene_version,
    )

    key = strip_gene_version(peak_lod_df['gene_id'])
    gi = gene_info_df.assign(gene_id=strip_gene_version(gene_info_df['gene_id']))
    gi = gi.drop_duplicates('gene_id')
    gi_cols = ['gene_id']
    if 'gene_symbol' in gi.columns and 'gene_symbol' not in peak_lod_df.columns:
        gi_cols.append('gene_symbol')
    gi_sub = gi[gi_cols].rename(columns={'gene_id': '_key'})
    if 'start' in gi.columns:
        if not {'end', 'strand'} <= set(gi.columns):
            logger.warning(
                'enrich_peak_lod: gene_info lacks end/strand; start_mb is the '
                'gene start, not the TSS, for minus-strand genes'
            )
        gi_sub = gi_sub.assign(start_mb=gi_sub['_key'].map(build_tss_lookup(gi)))
    out = peak_lod_df.assign(_key=key).merge(gi_sub, on='_key', how='left')
    out = out.drop(columns='_key')
    if 'peak_mb' not in out.columns:
        return out
    if gene_tss_df is not None:
        out['near_tss_mb'] = nearest_transcript_tss_mb(out, gene_tss_df)
        fixed = gene_tss_anchor(gene_tss_df).reindex(key.to_numpy())
        out['cis_dist_gene_tss_mb'] = (
            (out['peak_mb'].to_numpy() - fixed['tss_mb'].to_numpy()) * fixed['sign'].to_numpy()
        )
    anchor = out['near_tss_mb'] if 'near_tss_mb' in out.columns else None
    if 'start_mb' in out.columns:
        anchor = out['start_mb'] if anchor is None else anchor.fillna(out['start_mb'])
    if anchor is not None:
        sign = 1.0
        if 'strand' in gi.columns:
            sign = np.where(key.map(gi.set_index('gene_id')['strand']).eq('-'), -1.0, 1.0)
        out['cis_dist_mb'] = (out['peak_mb'] - anchor) * sign
    n_miss = int(out['cis_dist_mb'].isna().sum()) if 'cis_dist_mb' in out.columns else 0
    if n_miss:
        logger.warning('enrich_peak_lod: %d of %d genes have no TSS anchor', n_miss, len(out))
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


SUPPORT_INTERVAL_COLS: tuple[str, ...] = ("ci_lo", "ci_hi", "ci_width_mb", "ci_truncated")


def attach_support_interval(summary_df: pd.DataFrame, ci_df: pd.DataFrame) -> pd.DataFrame:
    """Join each eGene's support interval onto the eGene summary.

    Parameters
    ----------
    summary_df : eGene summary with a ``gene_id`` column.
    ci_df : support-interval table (``CI_COLS``), one row per eGene.

    Returns
    -------
    DataFrame with the rows of ``summary_df`` and the ``SUPPORT_INTERVAL_COLS`` filled
    for eGenes present in ``ci_df`` (missing otherwise), in ``EGENE_SUMMARY_COLS`` order.
    """
    ci = ci_df[["gene_id", *SUPPORT_INTERVAL_COLS]].copy()
    ci["gene_id"] = ci["gene_id"].astype(str)
    ci = ci.drop_duplicates("gene_id")
    ci["ci_truncated"] = ci["ci_truncated"].astype("boolean")
    out = summary_df.drop(columns=[c for c in SUPPORT_INTERVAL_COLS if c in summary_df.columns])
    out = out.assign(gene_id=out["gene_id"].astype(str)).merge(ci, on="gene_id", how="left")
    return out.reindex(columns=[c for c in EGENE_SUMMARY_COLS if c in out.columns])


def select_submission_genes(peak_df: pd.DataFrame, fdr_max: float) -> pd.DataFrame:
    r"""Rows of the calibrated peak table whose Benjamini-Hochberg q-value is at or below ``fdr_max``.

    The calibration and the q-values are computed over every tested gene before this
    selection, so ``qval_bh`` does not depend on it, whatever the eGene call rule is.
    Choosing ``fdr_max`` above the eGene calling threshold keeps genes that were not
    called but are plausible eQTLs. Genes with a missing ``qval_bh`` are dropped.

    Parameters
    ----------
    peak_df : peak table carrying the calibration block, including ``qval_bh``.
    fdr_max : largest ``qval_bh`` kept.
    """
    return peak_df.loc[peak_df["qval_bh"] <= fdr_max].copy()


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
    per-gene Beta+Storey calibration in calibrate_egenes(), which
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


_LOCAL_TAG_RE = re.compile(r'^Lw(\d+(?:\.\d+)?)$')


def _calibrate_pvalues_loop(
    egenes_df: pd.DataFrame,
    genomecache: "GenomeCache",
    pheno_mat: np.ndarray,
    gene_idx: "dict[str, int]",
    covariates: "np.ndarray | None",
    kinship_loco: dict,
    tss_map: "dict[str, float]",
    sym_map: "dict[str, str]",
    scan_local: bool,
    window_mb: float,
    n_perms: int,
    use_rint: bool,
    z_mat: "np.ndarray | None",
    valid_mask: "np.ndarray | None",
    seed: int,
) -> pd.DataFrame:
    """per-gene calibrated p-values from a null built the way the real scan is built.

    returns a frame indexed by gene_id with pval_perm and the fitted Beta shape
    parameters beta_a and beta_b of the gene's null p-values.

    phenotype, covariates and kinship are at sample level: y and x are restricted
    to genotyped samples, K_n = Z K_m Z^T is eigendecomposed once per chromosome
    (genes are grouped by chromosome), and each gene's null reuses its own
    heritability, RINT setting and locus window. each gene draws from its own
    seed stream so results do not depend on gene order. genes whose calibration
    cannot be computed get NaN and are counted in one summary warning.
    """
    import zlib

    from .qtl import lmm as lmm_module
    from .qtl.fdr import calibrate_gene_pvalue as _cal
    from .qtl.permutation import KinshipSingularError

    valid = (valid_mask if valid_mask is not None
             else np.ones(pheno_mat.shape[1], dtype=bool))
    z_sub = z_mat[valid] if z_mat is not None else None
    x = (covariates[valid] if covariates is not None
         else np.ones((int(valid.sum()), 1)))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((len(x), 1)), x])

    nan3 = (float('nan'),) * 3
    fits: dict[str, tuple[float, float, float]] = {}
    n_failed = 0
    n_total = len(egenes_df)
    n_done = 0
    for chrom, grp in egenes_df.groupby(egenes_df['chr'].astype(str), sort=False):
        k_m = kinship_loco[chrom]
        k_n = z_sub @ k_m @ z_sub.T if z_sub is not None else k_m
        ev, evc = np.linalg.eigh(k_n)
        decomp = {chrom: {'values': ev, 'vectors': evc}}
        loco = {chrom: k_n}
        chr_data = lmm_module.precompute_chromosome(
            genomecache, chrom, ev, evc, z_sub, 'additive')
        for _, row in grp.iterrows():
            n_done += 1
            gid = str(row['gene_id'])
            sym = sym_map.get(gid, gid)
            logger.info('  calibrate [%d/%d] %s chr%s', n_done, n_total, sym, chrom)
            if gid not in gene_idx:
                fits[gid] = nan3
                n_failed += 1
                continue
            gene_seed = int(np.random.SeedSequence(
                [seed, zlib.crc32(gid.encode())]).generate_state(1)[0])
            try:
                p_adj, a_hat, b_hat, _ = _cal(
                    y_gene=pheno_mat[gene_idx[gid]][valid],
                    genomecache=genomecache,
                    chrom=chrom,
                    peak_lod=float(row['peak_lod']),
                    covariates=x,
                    kinship_decomp=decomp,
                    kinship_loco=loco,
                    n_perms=n_perms,
                    gene_pos_mb=tss_map.get(gid) if scan_local else None,
                    window_mb=window_mb if scan_local else None,
                    use_rint=use_rint,
                    z_mat=z_sub,
                    seed=gene_seed,
                    chr_data=chr_data,
                )
                fits[gid] = (float(p_adj), float(a_hat), float(b_hat))
            except (ValueError, np.linalg.LinAlgError, KinshipSingularError) as exc:
                logger.debug('calibration failed for %s: %s', sym, exc)
                fits[gid] = nan3
                n_failed += 1
        del k_n, ev, evc, decomp, loco, chr_data
    if n_failed:
        logger.warning('calibration produced no p-value for %d / %d genes',
                       n_failed, n_total)
    return pd.DataFrame.from_dict(
        fits, orient='index', columns=['pval_perm', 'beta_a', 'beta_b']
    )


def calibrate_egenes(
    egenes_df: pd.DataFrame,
    pval_perm: "dict[str, float] | pd.Series | None" = None,
    gene_col: str = "gene_id",
    fdr_threshold: float = 0.05,
    *,
    genomecache: "GenomeCache | None" = None,
    pheno_mat: "np.ndarray | None" = None,
    gene_idx: "dict[str, int] | None" = None,
    covariates: "np.ndarray | None" = None,
    kinship_loco: "dict | None" = None,
    stag: str = "C",
    gene_info_df: "pd.DataFrame | None" = None,
    n_perms: int = 200,
    use_rint: bool = True,
    z_mat: "np.ndarray | None" = None,
    valid_mask: "np.ndarray | None" = None,
    seed: int = 1,
    call_rule: str = "bh",
    lod_threshold: "float | None" = None,
) -> pd.DataFrame:
    """eGene calling: per-gene Beta calibration, FDR over all tested genes, LOD gate.

    Accepts either pre-computed per-gene p-values (``pval_perm``) or the scan
    inputs needed to compute them on the fly.  The q-value is estimated over
    every row of ``egenes_df``, so the frame should hold all tested genes.
    When computing on the fly, each gene's null is simulated at sample level
    from its own heritability and the per-chromosome LOCO kinship expanded
    through ``z_mat``, and is restricted to the same locus window as the real
    scan — decoded from the scan tag ``stag`` — so the null and real
    distributions share one LOD scale.

    The scan tag encodes the window policy: ``'Lw<N>'`` (e.g. ``'Lw1'``) is a
    local ±N Mb window; anything else uses the whole chromosome.  Drawing the
    null over a wider window than the real scan inflates null max-LODs and
    drives calibrated p-values toward 1, collapsing FDR power.

    Parameters
    ----------
    egenes_df      : tested-gene table — must contain gene_col and chr,
                     peak_lod columns.
    pval_perm      : pre-computed {gene_id: p_adj} dict or Series.  When
                     ``None``, the remaining keyword arguments must supply
                     the scan inputs for on-the-fly calibration.
    gene_col       : join key in egenes_df (default ``'gene_id'``).
    fdr_threshold  : q-value cutoff (default 0.05).
    genomecache    : GenomeCache; required when pval_perm is None.
    pheno_mat      : (n_genes, n_samples) expression matrix; required when
                     pval_perm is None.
    gene_idx       : {gene_id: row_index_in_pheno_mat}; required when
                     pval_perm is None.
    covariates     : (n_samples, p) covariate matrix, or None.
    kinship_loco   : {chrom: np.ndarray} strain-level LOCO kinship matrices.
    stag           : scan tag used to derive the window restriction.
    gene_info_df   : annotation table (gene_id, start, end, chr) for TSS
                     lookup; used when stag is a local-window tag.
    n_perms        : parametric null permutations per gene.
    use_rint       : apply RINT to y_gene before scanning.
    z_mat          : (n_samples, n_strains) sample-to-strain design matrix.
    valid_mask     : (n_strains,) bool mask.
    seed           : base RNG seed; each gene derives its own stream.
    call_rule      : 'bh' (Benjamini-Hochberg, default), 'qvalue' (Storey) or
                     'lfdr' (local false discovery rate); selects which value
                     is written to fdr and thresholded.
    lod_threshold  : permutation LOD threshold. When given, a gene is called
                     significant only if fdr <= fdr_threshold and peak_lod is
                     above it; the q-values still use every tested gene.

    Returns
    -------
    DataFrame — egenes_df with pval_perm, perm_n (null draws per gene; the
    smallest attainable p is about 1/perm_n), qval_storey, qval_bh, lfdr, pi0,
    fdr, fdr_method, significance added. All three values are written (lfdr is
    NaN when fewer than fdr.LFDR_MIN_GENES genes were tested); only fdr and
    significance follow call_rule and lod_threshold.
    """
    from .qtl import fdr as fdr_module
    from .utils import build_tss_lookup

    if pval_perm is None:
        if genomecache is None or pheno_mat is None or gene_idx is None:
            raise ValueError(
                'genomecache, pheno_mat, and gene_idx are required '
                'when pval_perm is not pre-computed'
            )

        _m = _LOCAL_TAG_RE.match(str(stag))
        _scan_local = _m is not None
        _window_mb  = float(_m.group(1)) if _m else 0.0
        if not _scan_local and str(stag) != 'C':
            raise ValueError(
                f"calibration supports C (cis) and L (local, tag 'Lw<window>') scans; "
                f"got {stag!r}"
            )

        tss_map: dict[str, float] = {}
        if _scan_local:
            if gene_info_df is not None:
                try:
                    tss_map = build_tss_lookup(gene_info_df)
                except Exception as exc:
                    logger.warning('build_tss_lookup failed: %s', exc)
            for _col in ('tss_mb', 'near_tss_mb', 'start_mb'):
                if _col in egenes_df.columns:
                    for _, _r in egenes_df.iterrows():
                        _gid, _v = str(_r[gene_col]), _r[_col]
                        if pd.notna(_v) and _gid not in tss_map:
                            tss_map[_gid] = float(_v)
                    break

        sym_map: dict[str, str] = {}
        if 'gene_symbol' in egenes_df.columns:
            sym_map = dict(zip(egenes_df[gene_col].astype(str),
                               egenes_df['gene_symbol'].astype(str)))
        elif gene_info_df is not None and 'gene_symbol' in gene_info_df.columns:
            sym_map = dict(zip(gene_info_df['gene_id'].astype(str),
                               gene_info_df['gene_symbol'].astype(str)))

        fits = _calibrate_pvalues_loop(
            egenes_df, genomecache, pheno_mat, gene_idx,
            covariates, kinship_loco,
            tss_map, sym_map, _scan_local, _window_mb,
            n_perms, use_rint, z_mat, valid_mask, seed,
        )
        pval_perm = fits['pval_perm']
    else:
        fits = None

    out = egenes_df.copy()
    pv = pval_perm if isinstance(pval_perm, pd.Series) else pd.Series(pval_perm)
    out["pval_perm"] = out[gene_col].map(pv)
    if fits is not None:
        for col in ("beta_a", "beta_b"):
            out[col] = out[gene_col].map(fits[col])
    out["perm_n"] = n_perms

    if call_rule not in _CALL_RULES:
        raise ValueError(f"call_rule must be one of {sorted(_CALL_RULES)}, got {call_rule!r}")
    pv_arr = out["pval_perm"].to_numpy()
    q_storey, pi0 = fdr_module.qvalue(pv_arr)
    q_bh = fdr_module.bh(pv_arr)
    if int(np.isfinite(pv_arr).sum()) >= fdr_module.LFDR_MIN_GENES:
        local = fdr_module.lfdr(pv_arr, pi0=pi0)
    elif call_rule == "lfdr":
        raise ValueError(
            f"call_rule 'lfdr' needs at least {fdr_module.LFDR_MIN_GENES} tested genes"
        )
    else:
        local = np.full(len(pv_arr), np.nan)
    out["qval_storey"] = q_storey
    out["qval_bh"] = q_bh
    out["lfdr"] = local
    out["pi0"] = pi0
    out["fdr"] = {"qvalue": q_storey, "bh": q_bh, "lfdr": local}[call_rule]
    out["fdr_method"] = _CALL_RULES[call_rule]
    called = (out["fdr"] <= fdr_threshold).to_numpy()
    n_fdr = int(called.sum())
    if lod_threshold is not None:
        called = called & (out["peak_lod"] > lod_threshold).to_numpy()
    out["significance"] = np.where(called, "T", "F")

    logger.info(
        "calibrate_egenes: n_genes=%d  pi0=%.4f  rule=%s  "
        "fdr<=%.2f: %d  significant (lod>%s): %d  (storey=%d  bh=%d)",
        len(out), pi0, call_rule, fdr_threshold, n_fdr,
        "-" if lod_threshold is None else f"{lod_threshold:.3f}", int(called.sum()),
        int((q_storey <= fdr_threshold).sum()), int((q_bh <= fdr_threshold).sum()),
    )
    return out


def call_significant_egenes(
    peak_df: pd.DataFrame,
    *,
    genomecache: "GenomeCache",
    pheno_mat: np.ndarray,
    gene_idx: "dict[str, int]",
    covariates: "np.ndarray | None",
    kinship_decomp: dict,
    kinship_loco: dict,
    sample_ids: "list[str]",
    stag: str,
    gene_info_df: "pd.DataFrame | None",
    fdr_threshold: float,
    n_perms: int,
    use_rint: bool,
    z_mat: "np.ndarray | None",
    valid_mask: "np.ndarray | None",
    seed: int,
    call_rule: str = "bh",
    lod_threshold: "float | None" = None,
) -> pd.DataFrame:
    """Call eGenes over every tested gene: Beta calibration, FDR, LOD gate, effects.

    The q-value is estimated from the p-values of all genes with a peak, because
    its pi0 estimate and rank-based error count assume null p-values are uniform
    over the full tested set. Selecting genes by their peak LOD first would
    concentrate the surviving null p-values near zero. Founder effects are
    computed afterwards, only for genes called significant.

    Parameters
    ----------
    peak_df        : per-gene peak table with gene_id, chr, peak_lod, peak_locus.
        Rows missing gene_id, chr or peak_lod are not tested.
    genomecache    : GenomeCache holding the haplotype probabilities.
    pheno_mat      : (n_genes, n_samples) expression matrix.
    gene_idx       : {gene_id: row index in pheno_mat}.
    covariates     : (n_samples, p) covariate matrix, or None.
    kinship_decomp : {chrom: {'values', 'vectors'}} strain-level decomposition.
    kinship_loco   : {chrom: ndarray} strain-level LOCO kinship.
    sample_ids     : genotype-level ID of each column of pheno_mat.
    stag           : scan tag giving the locus window (C or Lw<window>).
    gene_info_df   : annotation table with gene_id and gene_symbol / coordinates.
    fdr_threshold  : FDR cutoff on the value the call rule selects.
    n_perms        : parametric null draws per gene.
    use_rint       : apply RINT to the phenotype before scanning.
    z_mat, valid_mask : sample-to-strain design matrix and strain validity mask.
    seed           : base seed; each gene derives its own stream from it.
    call_rule      : 'bh' (Benjamini-Hochberg, default), 'qvalue' (Storey) or 'lfdr'.
    lod_threshold  : permutation LOD threshold a significant gene's peak LOD must
        exceed; ``None`` applies the FDR cutoff alone.

    Returns
    -------
    DataFrame with one row per tested gene: the peak columns plus pval_perm,
    qval_storey, qval_bh, lfdr, pi0, fdr, fdr_method, significance, and
    beta_<founder> columns filled for significant genes.
    """
    from .qtl.effects import get_allele_effects

    tested = (peak_df.dropna(subset=["gene_id", "chr", "peak_lod"])
              .reset_index(drop=True))
    called = calibrate_egenes(
        tested,
        fdr_threshold=fdr_threshold,
        genomecache=genomecache,
        pheno_mat=pheno_mat,
        gene_idx=gene_idx,
        covariates=covariates,
        kinship_loco=kinship_loco,
        stag=stag,
        gene_info_df=gene_info_df,
        n_perms=n_perms,
        use_rint=use_rint,
        z_mat=z_mat,
        valid_mask=valid_mask,
        seed=seed,
        call_rule=call_rule,
        lod_threshold=lod_threshold,
    )

    significant = called[called["significance"] == "T"]
    logger.info("allele effects: %d significant eGenes", len(significant))
    rows: list[dict] = []
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        gid, chrom = str(row["gene_id"]), str(row["chr"])
        locus = str(row.get("peak_locus", ""))
        if gid not in gene_idx or not locus or locus == "nan":
            continue
        logger.info("  effects [%d/%d] %s chr%s", i, len(significant), gid, chrom)
        effects = get_allele_effects(
            phenotype=pheno_mat[gene_idx[gid]],
            genomecache=genomecache,
            locus_id=locus,
            chrom=chrom,
            kinship_decomp=kinship_decomp,
            covariates=covariates,
            sample_ids=sample_ids,
            use_rint=use_rint,
        )
        rows.append({"gene_id": gid,
                     **{f"beta_{fe.founder}": fe.estimate for fe in effects}})
    if rows:
        eff_df = pd.DataFrame(rows)
        called = called.merge(eff_df, on="gene_id", how="left")
    return called


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
    *,
    use_rint: bool,
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
        use_rint: Rank-inverse normal transform of the phenotype before the
            SNP scan; must equal the value the haplotype scan used.
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
        use_rint=use_rint,
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
            alt_counts: dict[str, int] = {}
            for a in valid.values():
                if a != ref_allele:
                    alt_counts[a] = alt_counts.get(a, 0) + 1
            alt_allele = (
                min(alt_counts, key=lambda a: (-alt_counts[a], a)) if alt_counts else ""
            )
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
        # the fitted standard error of the dosage effect; where it is missing,
        # invert the likelihood-ratio statistic: se = |beta| / sqrt(2 ln10 lod)
        _beta_abs = ann_df["snp"].map(_snp_idx["beta_dosage"]).abs()
        _lod_map = ann_df["snp"].map(_snp_idx["lod"]).clip(lower=0.0)
        _z_abs = np.sqrt(2.0 * np.log(10) * _lod_map)
        _se_wald = (_beta_abs / _z_abs.where(_z_abs > 1e-10)).where(
            _beta_abs > 1e-10, other=np.nan
        )
        ann_df["se_effect_size"] = ann_df["snp"].map(_snp_idx["se_dosage"]).fillna(_se_wald)
        snp_df = snp_df.merge(ann_df, on="snp", how="left")

    result_df = snp_df.reindex(columns=IMPUTED_SNP_COLS)
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
    *,
    use_rint: bool,
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
        use_rint: Rank-inverse normal transform of the phenotype before the
            SNP scan; must equal the value the haplotype scan used.

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
                use_rint=use_rint,
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


def scanned_chromosomes(scan_dir: 'str | os.PathLike[str]') -> list[str]:
    """Return the unprefixed chromosome labels with a complete per-chromosome scan output.

    A chromosome is complete when ``lod_chr{N}.csv``, ``peak_stats_chr{N}.csv`` and
    ``timing_chr{N}.json`` all exist, in biological chromosome order.

    Raises
    ------
    FileNotFoundError
        If no ``lod_chr*.csv`` exists.
    ValueError
        If any chromosome with a LOD file lacks its peak-statistics or timing file.
    """
    from pathlib import Path

    from .utils import chr_sort_key

    directory = Path(scan_dir)
    chroms = [p.stem[len('lod_chr'):] for p in directory.glob('lod_chr*.csv')]
    if not chroms:
        raise FileNotFoundError(f'no lod_chr*.csv in {directory}')
    expected = {
        'peak_stats_chr': '.csv',
        'timing_chr': '.json',
    }
    missing = [
        f'{prefix}{chrom}{ext}'
        for chrom in chroms
        for prefix, ext in expected.items()
        if not (directory / f'{prefix}{chrom}{ext}').exists()
    ]
    if missing:
        raise ValueError(f'incomplete per-chromosome scan output, missing: {", ".join(sorted(missing))}')
    return sorted(chroms, key=chr_sort_key)


def validate_finemap_frames(
    fm_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    input_genes: set[str],
) -> None:
    """Check the fine-mapping tables before they are written.

    Every output gene must be one of ``input_genes``; ``(gene_id, variant)`` must be
    unique in ``fm_df``; ``peak_df`` must hold one row per gene, and only genes
    present in ``fm_df``.

    Raises
    ------
    ValueError
        Naming the first violated invariant and up to five offending keys.
    """
    def _sample(keys: object) -> str:
        return ', '.join(map(str, list(keys)[:5]))

    fm_genes = set(fm_df['gene_id'].astype(str))
    unknown = fm_genes - input_genes
    if unknown:
        raise ValueError(f'fine-mapping output has genes not in the input: {_sample(unknown)}')
    dup = fm_df.duplicated(['gene_id', 'variant'], keep=False)
    if dup.any():
        keys = fm_df.loc[dup, ['gene_id', 'variant']].drop_duplicates().itertuples(index=False)
        raise ValueError(f'duplicate (gene_id, variant) rows in fine-mapping output: {_sample(keys)}')
    peak_dup = peak_df['gene_id'].duplicated(keep=False)
    if peak_dup.any():
        raise ValueError(
            f'peak output has several rows per gene: {_sample(peak_df.loc[peak_dup, "gene_id"].unique())}'
        )
    orphan = set(peak_df['gene_id'].astype(str)) - fm_genes
    if orphan:
        raise ValueError(f'peak output has genes absent from the per-variant table: {_sample(orphan)}')


def finemap_imputed(
    imputed_df: pd.DataFrame,
    out_dir: "Path | str",
    prior_var: float = 0.04,
    coverage: float = 0.95,
    tag: str = "",
    egene_df: Optional[pd.DataFrame] = None,
    ld_by_gene: Optional[dict] = None,
) -> tuple:
    """ABF-based credible sets from an imputed SNP scan, written to disk.

    Args:
        imputed_df: Output of impute_egenes (IMPUTED_SNP_COLS layout).
        out_dir: Directory for eqtl_finemapping{_tag}.csv and
            eqtl_finemapping_peak{_tag}.csv.
        prior_var: Prior effect variance on the standardised effect.
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
    from .qtl.imputation import (
        fine_map_region,
        format_founder_pattern,
        ld_group_summary,
    )
    from .io.atomic import atomic_write_csv
    from .io.results_io import (
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
    grp_frames: list = []
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
        _patterns: Optional[np.ndarray] = None
        _gene_ld = (ld_by_gene or {}).get(str(gid))
        if _gene_ld is not None:
            _patterns = np.array(
                [format_founder_pattern(p) for p in _gene_ld["snp_patterns"]], dtype=object
            )
            from .qtl.imputation import merge_ld_groups as _merge_ld
            _ld_groups = _merge_ld(
                rep_patterns=_gene_ld["ld"]["rep_patterns"],
                r2_matrix=_gene_ld["ld"]["r2_matrix"],
                snp_patterns=_gene_ld["snp_patterns"],
            )

        eg_row = egene_idx.get(str(gid))
        _tss_mb: Optional[float] = None
        if eg_row is not None:
            for _tc in ("near_tss_mb", "start_mb"):
                if _tc in eg_row.index and not pd.isna(eg_row[_tc]):
                    _tss_mb = float(eg_row[_tc])
                    break

        fm_df = fine_map_region(
            lod=g["lod"].to_numpy(dtype=float),
            beta_dosage=g["beta_dosage"].to_numpy(dtype=float),
            se_dosage=g["se_dosage"].to_numpy(dtype=float),
            snp_ids=g["snp"].to_numpy(),
            pos_mb=g["pos_mb"].to_numpy(dtype=float),
            ld_groups=_ld_groups,
            prior_var=prior_var,
            coverage=coverage,
            patterns=_patterns,
            tss_mb=_tss_mb,
        )
        if _ld_groups is not None:
            _grp_df = ld_group_summary(fm_df)
            if not _grp_df.empty:
                grp_frames.append(_grp_df.assign(gene_id=str(gid)))
        else:
            logger.warning(
                "finemap_imputed: no LD data for %s; variant-level posteriors only, "
                "no LD-group row written", _flabel,
            )
        sym = first.get("gene_symbol", "")

        founder_effects: Optional[dict] = None
        sig_values: Optional[dict] = None
        null_vals: Optional[dict] = None
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
            "af", "maf", "mac", "ma_sample", "effect_size", "se_effect_size",
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
        if not peak_row.empty and float(peak_row["pip"].iloc[0]) > PEAK_PIP_THRESHOLD:
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

    validate_finemap_frames(fm_out, fp_out, set(map(str, _fm_genes)))

    suffix = f"_{tag}" if tag else ""
    # fm_out intermediate carries _FOUNDER_EFFECT_COLS for build_eqtl_finemapping_peak_df;
    # trim to EQTL_FINEMAPPING_COLS (per-SNP schema) before writing to disk.
    atomic_write_csv(fm_out.reindex(columns=EQTL_FINEMAPPING_COLS), out_dir / f"eqtl_finemapping{suffix}.csv")
    atomic_write_csv(fp_out, out_dir / f"eqtl_finemapping_peak{suffix}.csv")
    grp_out = (
        pd.concat(grp_frames, ignore_index=True)
        if grp_frames
        else pd.DataFrame(columns=EQTL_FINEMAPPING_GROUP_COLS)
    )
    atomic_write_csv(
        grp_out.reindex(columns=EQTL_FINEMAPPING_GROUP_COLS),
        out_dir / f"eqtl_finemapping_groups{suffix}.csv",
    )
    n_fm_genes = fm_out['gene_id'].nunique()
    logger.info(
        "finemap_imputed: %d CS SNPs across %d eGenes written to %s; "
        "%d of %d eGenes have a lead pip <= %.2f and are not in the peak file",
        len(fm_out),
        n_fm_genes,
        out_dir,
        n_fm_genes - len(fp_out),
        n_fm_genes,
        PEAK_PIP_THRESHOLD,
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
    *,
    use_rint: bool,
) -> tuple:
    """Imputed SNP scan plus ABF fine-mapping for every eGene.

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
        use_rint: Rank-inverse normal transform of the phenotype before the
            SNP scan; must equal the value the haplotype scan used.
        prior_var: Prior effect variance.
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
        use_rint=use_rint,
    )
    return finemap_imputed(
        imputed, out_dir, prior_var=prior_var, coverage=coverage, tag=tag,
        egene_df=egene_df, ld_by_gene=ld_by_gene,
    )
