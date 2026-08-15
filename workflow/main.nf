#!/usr/bin/env nextflow
/*
 * mosaiQTL — genome-wide LMM QTL scan pipeline
 *
 * Single cell type:
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/gastroc_params.yaml -resume
 *
 * Multiple cell types (celltypes_dir × all chrs):
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/ctxhipp_params.yaml -resume
 *   # or override cell type list on the CLI:
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/ctxhipp_params.yaml \
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
 *                         pseudobulk.h5 and gene_info.csv (or use params.gene_info globally)
 *   params.phenotypes     path to a single phenotype HDF5 (single-CT mode, legacy)
 *
 * publishDir layout:
 *   {output_dir}/{celltype}/lod/lod_chr{c}.csv   — per-chr LOD
 *   {output_dir}/{celltype}/lod_genome.csv        — merged genome-wide LOD
 *   {output_dir}/kinship_loco.h5                  — shared, written once
 */

nextflow.enable.dsl = 2

include { KINSHIP_LOCO  } from '../modules/local/kinship_loco'
include { LMM_SCAN      } from '../modules/local/lmm_scan'
include { MERGE_RESULTS } from '../modules/local/merge_results'


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
//   by params.celltypes.  Each subdir must contain pseudobulk.h5.
//   gene_info is per-CT if {ct}/gene_info.csv exists, else falls back to
//   params.gene_info (global).
//
// Mode B — params.phenotypes (single-CT legacy): wraps in a one-element channel
//   named by params.celltype (default 'default').

def ct_channel(gene_info_global) {
    if (params.celltypes_dir) {
        def ctdir = file(params.celltypes_dir)

        // collect candidate directories
        def ct_ch
        if (params.celltypes && params.celltypes != 'all') {
            ct_ch = Channel.from(
                params.celltypes.split(',').collect { it.trim() }
            ).map { ct ->
                def pheno   = file("${params.celltypes_dir}/${ct}/pseudobulk.h5")
                def gi_local = file("${params.celltypes_dir}/${ct}/gene_info.csv")
                def gi      = gi_local.exists() ? gi_local : gene_info_global
                if (!pheno.exists()) {
                    error "pseudobulk.h5 not found for cell type '${ct}' at ${pheno}"
                }
                [ct, pheno, gi]
            }
        } else {
            // discover all subdirs that contain pseudobulk.h5
            ct_ch = Channel.fromPath("${params.celltypes_dir}/*/pseudobulk.h5")
                .map { pheno ->
                    def ct      = pheno.parent.name
                    def gi_local = file("${pheno.parent}/gene_info.csv")
                    def gi      = gi_local.exists() ? gi_local : gene_info_global
                    [ct, pheno, gi]
                }
        }
        return ct_ch

    } else if (params.phenotypes) {
        // legacy single-CT mode
        def ct   = params.celltype ?: 'default'
        def pheno = file(params.phenotypes)
        return Channel.value([ct, pheno, gene_info_global])

    } else {
        error "Either params.celltypes_dir or params.phenotypes must be set"
    }
}


// ── main workflow ─────────────────────────────────────────────────────────────

workflow {

    genomecache      = _require(params.genomecache, 'genomecache')
    gene_info_global = _require(params.gene_info,   'gene_info')

    // kinship: shared across all cell types — built once, then reused
    if (params.kinship_loco) {
        kinship = Channel.value(file(params.kinship_loco))
    } else {
        kinship = KINSHIP_LOCO(genomecache).kinship_h5
    }

    // cell-type × chromosome grid
    ct_ch  = ct_channel(gene_info_global)
    chroms = chr_channel()

    // combine: every (celltype, pheno, gene_info, chrom) tuple → one LMM_SCAN job
    ct_chr = ct_ch.combine(chroms)
    //   ct_chr emits: [celltype, pheno_path, gene_info_path, chrom]

    LMM_SCAN(
        genomecache,
        kinship,
        ct_chr,
    )
    // LMM_SCAN.out emits: [celltype, lod_csv]

    // merge per-celltype: group all per-chr CSVs by CT, then concatenate
    LMM_SCAN.out
        .groupTuple()
        | MERGE_RESULTS

}
