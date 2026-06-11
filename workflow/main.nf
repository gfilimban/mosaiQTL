#!/usr/bin/env nextflow
/*
 * mosaiQTL — genome-wide LMM QTL scan pipeline
 *
 * Run:
 *   nextflow run main.nf -profile local  -params-file config/default_params.yaml
 *   nextflow run main.nf -profile slurm  -params-file config/default_params.yaml -resume
 *   nextflow run main.nf -stub-run       -params-file config/default_params.yaml
 */

nextflow.enable.dsl = 2

include { LMM_SCAN      } from '../modules/local/lmm_scan'
include { MERGE_RESULTS } from '../modules/local/merge_results'


// ── input validation ──────────────────────────────────────────────────────────

def _require(param, name) {
    if (param == null) {
        error "params.${name} is required — set it in your params YAML or on the CLI"
    }
    return file(param)
}


// ── chromosome channel ────────────────────────────────────────────────────────

def chr_channel() {
    def all_chrs = (1..19).collect { it.toString() } + ['X']
    if (params.chroms == 'all') {
        return Channel.from(all_chrs)
    }
    return Channel.from(params.chroms.split(',').collect { it.trim() })
}


// ── main workflow ─────────────────────────────────────────────────────────────

workflow {

    genomecache  = _require(params.genomecache,  'genomecache')
    kinship_loco = _require(params.kinship_loco, 'kinship_loco')
    phenotypes   = _require(params.phenotypes,   'phenotypes')
    gene_info    = _require(params.gene_info,    'gene_info')

    chroms = chr_channel()

    LMM_SCAN(
        genomecache,
        kinship_loco,
        phenotypes,
        gene_info,
        chroms,
    )

    MERGE_RESULTS(LMM_SCAN.out.collect())

}
