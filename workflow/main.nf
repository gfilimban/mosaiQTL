#!/usr/bin/env nextflow
/*
 * mosaiQTL — genome-wide LMM QTL scan pipeline
 *
 * Single cell type:
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/gastroc_next_params.yaml -resume
 *
 * Multiple cell types (celltypes_dir × all chrs):
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/ctxhipp_next_params.yaml -resume
 *   # or override cell type list on the CLI:
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/ctxhipp_next_params.yaml \
 *       --celltypes 'glutamatergic_neuron,oligodendrocyte' -resume
 *
 * Stub run (DAG validation, no compute):
 *   nextflow run workflow/main.nf -stub-run -params-file config/default_params.yaml
 *
 * params.kinship_loco is optional:
 *   - if provided, KINSHIP_LOCO is skipped and the file is used directly
 *   - if null (default), KINSHIP_LOCO builds it from params.genomecache
 *
 * Cell type input (two modes, mutually exclusive — celltypes_dir takes precedence):
 *   params.celltypes_dir  directory; each subdir named after a cell type and containing
 *                         counts_lognorm.h5 and gene_info.csv (or use params.gene_info globally)
 *   params.phenotypes     path to a single phenotype HDF5 (single-CT mode, legacy)
 *
 * publishDir layout:
 *   {data_dir}/kinship_loco.h5                              — shared, written once
 *   {output_dir}/{celltype}/null_fit.csv                    — LMM null fit per gene
 *   {output_dir}/{celltype}/lod_chr{c}.csv                  — per-chr LOD
 *   {output_dir}/{celltype}/lod_genome.csv                  — merged genome-wide LOD
 *   {output_dir}/{celltype}/peak_lod.csv                    — per-gene peak locus
 *   {output_dir}/{celltype}/perm_max_lods.csv               — permutation max-LOD (all chrs)
 *   {output_dir}/{celltype}/perm_max_lods_chr{c}.csv        — per-chr permutation max-LOD
 *   {output_dir}/{celltype}/allele_effects.csv              — founder haplotype effects
 *   {output_dir}/{celltype}/eqtl_summary.csv                — eGene calling summary
 *   {output_dir}/{celltype}/eqtl_peak*.csv                  — submission eQTL peak table (haplotype-level, NaN CI)
 *   {output_dir}/{celltype}/eqtl_peak_ci*.csv              — eQTL peak with LOD-drop CI (ci_lo_bp/ci_hi_bp filled)
 *   {output_dir}/{celltype}/eqtl_stats*.csv                 — per-chr GEV stats
 *   {output_dir}/{celltype}/run_metadata.json               — version + git hash + provenance
 *   {output_dir}/{celltype}/eqtl_finemapping*.csv           — ABF credible sets
 *   {output_dir}/{celltype}/eqtl_finemapping_peak*.csv      — per-gene fine-mapping peak
 */

include { KINSHIP_LOCO  } from '../modules/local/kinship_loco'
include { LMM_SCAN      } from '../modules/local/lmm_scan'
include { PERMUTE       } from '../modules/local/permute'
include { MERGE_RESULTS } from '../modules/local/merge_results'
include { ESTIMATE      } from '../modules/local/estimate'
include { REPORT        } from '../modules/local/report'
include { CHARACTERISE  } from '../modules/local/characterise'
include { FINEMAP       } from '../modules/local/finemap'


// ── helpers ───────────────────────────────────────────────────────────────────

def _require(param, name) {
    if (param == null) {
        error "params.${name} is required — set it in your params YAML or on the CLI"
    }
    return file(param)
}

def chr_channel() {
    def all_chrs = (1..19).collect { it.toString() } + ['X']
    if (params.chroms == 'all') {
        return Channel.from(all_chrs)
    }
    return Channel.from(params.chroms.split(',').collect { it.trim() })
}


// ── cell-type channel ─────────────────────────────────────────────────────────
//
// Returns a channel of tuples: [celltype_name, phenotypes_path, gene_info_path]
//
// Mode A — celltypes_dir (multi-CT): discover subdirs automatically or filter
//   by params.celltypes.  Each subdir must contain counts_lognorm.h5.
//   gene_info is per-CT if {ct}/gene_info.csv exists, else falls back to
//   params.gene_info (global).
//
// Mode B — params.phenotypes (single-CT legacy): wraps in a one-element channel
//   named by params.celltype (default 'default').

def ct_channel(gene_info_global) {
    if (params.celltypes_dir) {
        def ctdir = file(params.celltypes_dir)

        def ct_ch
        if (params.celltypes && params.celltypes != 'all') {
            ct_ch = Channel.from(
                params.celltypes.split(',').collect { it.trim() }
            ).map { ct ->
                def pheno    = file("${params.celltypes_dir}/${ct}/counts_lognorm.h5")
                def gi_local = file("${params.celltypes_dir}/${ct}/gene_info.csv")
                def gi       = gi_local.exists() ? gi_local : gene_info_global
                if (!pheno.exists()) {
                    error "counts_lognorm.h5 not found for cell type '${ct}' at ${pheno}"
                }
                [ct, pheno, gi]
            }
        } else {
            ct_ch = Channel.fromPath("${params.celltypes_dir}/*/counts_lognorm.h5")
                .map { pheno ->
                    def ct       = pheno.parent.name
                    def gi_local = file("${pheno.parent}/gene_info.csv")
                    def gi       = gi_local.exists() ? gi_local : gene_info_global
                    [ct, pheno, gi]
                }
        }
        return ct_ch

    } else if (params.phenotypes) {
        def ct    = params.celltype ?: 'default'
        def pheno = file(params.phenotypes)
        return Channel.value([ct, pheno, gene_info_global])

    } else {
        error "Either params.celltypes_dir or params.phenotypes must be set"
    }
}


// ── main workflow ─────────────────────────────────────────────────────────────

workflow {

    // derive paths from data_dir when not set explicitly in the params file
    def _kinship_path  = params.kinship_loco   ?: (params.data_dir ? "${params.data_dir}/kinship_loco.h5"           : null)
    def _metadata_path = params.metadata       ?: (params.data_dir ? "${params.data_dir}/igvf_sample_metadata.csv"  : null)
    def _sanger_path   = params.sanger_alleles ?: (params.data_dir ? "${params.data_dir}/sanger_alleles.h5"         : null)

    genomecache      = _require(params.genomecache, 'genomecache')
    gene_info_global = _require(params.gene_info,   'gene_info')
    sanger_alleles   = _require(_sanger_path, 'sanger_alleles')

    // kinship: shared across all cell types — built once, then reused
    if (_kinship_path) {
        kinship = Channel.value(file(_kinship_path))
    } else {
        kinship = KINSHIP_LOCO(genomecache).kinship_h5
    }

    // cell-type channel — emits [celltype, phenotypes, gene_info]
    ct_ch = ct_channel(gene_info_global)

    // ── Stage 1a: LMM scan (CT × chr grid) ───────────────────────────────────
    chroms = chr_channel()
    ct_chr = ct_ch.combine(chroms)
    // ct_chr: [celltype, phenotypes, gene_info, chrom]

    LMM_SCAN(genomecache, kinship, ct_chr)
    // LMM_SCAN.out: [celltype, lod_chr.csv]

    // ── Stage 1b: permutation thresholds (once per CT, parallel with scan) ───
    // Uses pre-built kinship directly — no NB null fit required
    permute_input = ct_ch.map { ct, pheno, gi -> [ct, gi, pheno] }

    PERMUTE(genomecache, kinship, permute_input)
    // PERMUTE.out: [celltype, perm_max_lods.csv, [perm_max_lods_chr*.csv]]

    // ── Stage 2: merge per-chr LOD → genome + peak per CT ────────────────────
    LMM_SCAN.out
        .groupTuple()
        | MERGE_RESULTS
    // MERGE_RESULTS.out: [celltype, lod_genome.csv, peak_lod.csv]

    // ── Stage 3: founder haplotype effects at peak loci ───────────────────────
    // Join peak_lod with phenotypes from ct_ch on celltype
    estimate_input = MERGE_RESULTS.out
        .map { ct, lod, peak -> [ct, peak] }
        .join(ct_ch.map { ct, pheno, gi -> [ct, pheno] })
        .map { ct, peak, pheno -> [ct, peak, pheno] }

    ESTIMATE(genomecache, kinship, estimate_input)
    // ESTIMATE.out: [celltype, allele_effects.csv]

    // ── Stage 4: eGene calling + submission files ─────────────────────────────
    // Collect all perm files into a list so REPORT's work dir has them all
    perm_files_ch = PERMUTE.out
        .map { ct, perm_lods, perm_chrs ->
            [ct, [perm_lods, perm_chrs].flatten()]
        }

    report_input = MERGE_RESULTS.out
        .map { ct, lod, peak -> [ct, peak] }
        .join(perm_files_ch)
        .join(ESTIMATE.out)
        .map { ct, peak, perms, effects -> [ct, peak, perms, effects] }

    REPORT(report_input)
    // REPORT.out: [celltype, eqtl_summary.csv, [eqtl_peak*.csv], [eqtl_stats*.csv], run_metadata.json]

    // ── Stage 5: LOD-drop CI for eGenes ──────────────────────────────────────
    // CHARACTERISE rescans each eGene's chromosome to fill ci_lo_bp/ci_hi_bp.
    // Takes eqtl_peak*.csv from REPORT; outputs eqtl_peak_ci*.csv for FINEMAP.
    char_input = REPORT.out
        .map { ct, summary, peaks, stats, metadata ->
            def peak_file = peaks instanceof List ? peaks[0] : peaks
            [ct, peak_file]
        }
        .join(ct_ch.map { ct, pheno, gi -> [ct, pheno] })
        .map { ct, eqtl_peak, pheno -> [ct, eqtl_peak, pheno] }

    CHARACTERISE(genomecache, kinship, char_input)
    // CHARACTERISE.out: [celltype, eqtl_peak_ci*.csv]

    // ── Stage 6: Sanger SNP fine-mapping ─────────────────────────────────────
    // eqtl_peak_ci*.csv from CHARACTERISE has ci_lo_bp/ci_hi_bp populated.
    finemap_input = CHARACTERISE.out
        .map { ct, ci_peak ->
            def peak_file = ci_peak instanceof List ? ci_peak[0] : ci_peak
            [ct, peak_file]
        }
        .join(ct_ch.map { ct, pheno, gi -> [ct, pheno] })
        .map { ct, eqtl_peak_ci, pheno -> [ct, eqtl_peak_ci, pheno] }

    FINEMAP(genomecache, kinship, sanger_alleles, finemap_input)
    // FINEMAP.out: [celltype, [eqtl_finemapping*.csv], [eqtl_finemapping_peak*.csv]]

}
