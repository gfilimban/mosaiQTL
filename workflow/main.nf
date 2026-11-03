#!/usr/bin/env nextflow

include { BUILD_CACHE   } from '../modules/local/build_cache'
include { KINSHIP_LOCO  } from '../modules/local/kinship_loco'
include { LMM_SCAN      } from '../modules/local/lmm_scan'
include { MERGE_RESULTS } from '../modules/local/merge_results'
include { PERMUTE       } from '../modules/local/permute'
include { CALL_EGENES   } from '../modules/local/call_egenes'
include { FINEMAP_CHR   } from '../modules/local/finemap_chr'
include { FINEMAP_MERGE } from '../modules/local/finemap_merge'
include { REPORT        } from '../modules/local/report'

// ── helpers ───────────────────────────────────────────────────────────────────

def stag() {
    params.scan_type == 'L' ? "Lw${(params.window_mb as float).toInteger()}" : params.scan_type
}

def chr_list() {
    params.chroms == 'all'
        ? (1..19).collect { it.toString() } + ['X']
        : params.chroms.split(',').collect { it.trim() }
}

// Returns channel of [celltype, phenotypes, gene_info, metadata]
def ct_channel(gi_global, meta) {
    if (params.celltypes_dir) {
        def cts = (params.celltypes && params.celltypes != 'all')
            ? Channel.from(params.celltypes.split(',').collect { it.trim() })
            : Channel.fromPath("${params.celltypes_dir}/*/counts_lognorm.h5")
                     .map { it.parent.name }
        return cts.map { ct ->
            def pheno = file("${params.celltypes_dir}/${ct}/counts_lognorm.h5")
            def gi_ct = file("${params.celltypes_dir}/${ct}/gene_info.csv").exists()
                ? file("${params.celltypes_dir}/${ct}/gene_info.csv")
                : gi_global
            if (!pheno.exists()) error "counts_lognorm.h5 missing for '${ct}'"
            [ct, pheno, gi_ct, meta]
        }
    }
    return Channel.value([params.celltype ?: 'default', file(params.phenotypes ?: 'NO_PHENOTYPES'), gi_global, meta])
}

// Process script blocks run from work/<hash>/ — resolve output_dir to an
// absolute path so tee log paths are valid from any working directory.
params.output_dir = file(params.output_dir as String).toAbsolutePath().toString()

// ── workflow ──────────────────────────────────────────────────────────────────

workflow {

    new File("${params.output_dir}/logs").mkdirs()

    scan_tag   = stag()
    gi_global  = file(params.gene_info      ?: 'NO_GENE_INFO')
    gene_tss   = Channel.value(params.gene_tss ? file(params.gene_tss) : file('NO_GENE_TSS'))
    meta       = file(params.metadata       ?: 'NO_METADATA')
    sanger_h5  = params.sanger_alleles ? Channel.value(file(params.sanger_alleles)) : Channel.value(file('NO_SANGER'))
    gtf_ch     = params.gtf ? Channel.value(file(params.gtf)) : Channel.value(file('NO_GTF'))

    // cache — skip build if igvf_cache.h5 already present in ref/
    def cf = file("${launchDir}/ref/igvf_cache.h5")
    gc = cf.exists()
        ? Channel.value(cf)
        : BUILD_CACHE(Channel.value(file(params.gigamuga_csv ?: 'NO_GIGAMUGA'))).cache_h5

    // kinship — skip build if igvf_kinship_loco.h5 already present in ref/
    def kf = file("${launchDir}/ref/igvf_kinship_loco.h5")
    kinship = kf.exists()
        ? Channel.value(kf)
        : KINSHIP_LOCO(gc).kinship_h5

    // cell-type channel with scan_tag attached: [ct, stag, pheno, gi, meta]
    ct_ch = ct_channel(gi_global, meta)
                .map { ct, pheno, gi, m -> [ct, scan_tag, pheno, gi, m] }

    // ── LMM scan (CT × chr) ───────────────────────────────────────────────────
    lmm_in = ct_ch
        .combine(Channel.from(chr_list()))
        .map { ct, stag, pheno, gi, m, chrom -> [ct, chrom, pheno, gi, m] }

    LMM_SCAN(gc, kinship, lmm_in)

    // ── Merge per-chr → genome-wide (per CT) ─────────────────────────────────
    merge_in = LMM_SCAN.out
        .map { ct, chrom, lod, peak, timing -> [ct, lod, peak, timing] }
        .groupTuple()
        .join(ct_ch.map { ct, stag, pheno, gi, m -> [ct, stag, gi] })
        .map { ct, lods, peaks, timings, stag, gi -> [ct, stag, lods, peaks, timings, gi] }

    MERGE_RESULTS(merge_in)

    // ── Permutation (per CT, after merge — needs peak_lod for rep gene) ───────
    perm_in = MERGE_RESULTS.out
        .map { ct, stag, lod, peak, stats -> [ct, stag, peak] }
        .join(ct_ch.map { ct, stag, pheno, gi, m -> [ct, pheno, m] })
        .combine(gene_tss)
        .map { ct, stag, peak, pheno, m, tss -> [ct, stag, pheno, peak, m, tss] }

    PERMUTE(gc, kinship, perm_in)

    // ── eGene calling + FDR ───────────────────────────────────────────────────
    call_in = MERGE_RESULTS.out
        .map { ct, stag, lod, peak, stats -> [ct, stag, peak] }
        .join(PERMUTE.out.map { ct, stag, perm, perm_chrs, gev -> [ct, perm] })
        .join(ct_ch.map { ct, stag, pheno, gi, m -> [ct, pheno, gi, m] })
        .map { ct, stag, peak, perm, pheno, gi, m -> [ct, stag, peak, perm, pheno, gi, m] }

    CALL_EGENES(gc, kinship, call_in)

    // ── Fine-mapping (parallel by chr — 20 FINEMAP_CHR jobs per CT) ─────────
    // peak_lod.csv carries null model cols + beta_{founder} cols for egene enrichment
    // use by:[0,1] to join on (ct, stag) composite key so stag is not duplicated
    finemap_base = CALL_EGENES.out
        .map { ct, stag, peak, pval, summary -> [ct, stag, summary] }
        .join(MERGE_RESULTS.out.map { ct, stag, lod, peak_lod, stats -> [ct, stag, peak_lod] }, by: [0, 1])
        .join(ct_ch.map { ct, stag, pheno, gi, m -> [ct, pheno, m] })
        .map { ct, stag, summary, peak_lod, pheno, m -> [ct, stag, summary, peak_lod, pheno, m] }

    finemap_chr_in = finemap_base
        .combine(Channel.from(chr_list()))
        .map { ct, stag, summary, peak_lod, pheno, m, chrom ->
               [ct, stag, chrom, summary, peak_lod, pheno, m] }

    FINEMAP_CHR(gc, kinship, sanger_h5, finemap_chr_in)

    finemap_merge_in = FINEMAP_CHR.out
        .map { ct, stag, chrom, fm, fm_peak -> [ct, stag, fm, fm_peak] }
        .groupTuple(by: [0, 1])

    FINEMAP_MERGE(finemap_merge_in)

    // ── Report (all scan visualisations) ─────────────────────────────────────
    report_in = CALL_EGENES.out
        .map { ct, stag, peak, pval, summary -> [ct, stag, peak, summary, pval] }
        .join(PERMUTE.out.map { ct, stag, perm, perm_chrs, gev -> [ct, perm, perm_chrs] })
        .join(MERGE_RESULTS.out.map { ct, stag, lod, peak, stats -> [ct, lod] })
        .join(ct_ch.map { ct, stag, pheno, gi, m -> [ct, gi] })
        .map { ct, stag, peak, summary, pval, perm, perm_chrs, lod_genome, gi ->
               [ct, stag, peak, summary, pval, perm, perm_chrs, gi, lod_genome] }

    REPORT(gc, kinship, gtf_ch, report_in)
}
