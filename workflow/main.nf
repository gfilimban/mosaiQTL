#!/usr/bin/env nextflow
/*
 * mosaiQTL — genome-wide LMM QTL scan pipeline
 *
 * Run (gastroc example):
 *   nextflow run workflow/main.nf -profile slurm \
 *       -params-file config/gastroc_params.yaml -resume
 *
 * General:
 *   nextflow run workflow/main.nf -profile local  -params-file config/default_params.yaml
 *   nextflow run workflow/main.nf -stub-run        -params-file config/default_params.yaml
 *
 * params.kinship_loco is optional:
 *   - if provided, KINSHIP_LOCO is skipped and the file is used directly
 *   - if null (default), KINSHIP_LOCO builds it from params.genomecache
 */

nextflow.enable.dsl = 2

include { KINSHIP_LOCO  } from '../modules/local/kinship_loco'
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

    genomecache = _require(params.genomecache, 'genomecache')
    phenotypes  = _require(params.phenotypes,  'phenotypes')
    gene_info   = _require(params.gene_info,   'gene_info')

    // kinship: use pre-built file if provided, otherwise build it
    if (params.kinship_loco) {
        kinship = Channel.value(file(params.kinship_loco))
    } else {
        kinship = KINSHIP_LOCO(genomecache).kinship_h5
    }

    chroms = chr_channel()

    LMM_SCAN(
        genomecache,
        kinship,
        phenotypes,
        gene_info,
        chroms,
    )

    MERGE_RESULTS(LMM_SCAN.out.collect())

}
