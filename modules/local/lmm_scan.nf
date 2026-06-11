/*
 * LMM_SCAN — per-chromosome LMM QTL scan (array job)
 *
 * One process per chromosome; each writes lod_chr{chrom}.csv to output_dir.
 * Downstream: MERGE_RESULTS collects all per-chr CSVs.
 */

process LMM_SCAN {

    label 'process_medium'
    tag   "chr${chrom}"

    publishDir "${params.output_dir}/lod", mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    path  phenotypes
    path  gene_info
    val   chrom

    output:
    path "lod_chr${chrom}.csv"

    stub:
    """
    touch lod_chr${chrom}.csv
    """

    script:
    """
    python ${projectDir}/workflow/scripts/run_lmm_scan_chr.py \\
        --cache       ${genomecache} \\
        --kinship     ${kinship_loco} \\
        --phenotypes  ${phenotypes} \\
        --gene-info   ${gene_info} \\
        --chr         ${chrom} \\
        --scan-window ${params.scan_window} \\
        --window-mb   ${params.window_mb} \\
        --output-dir  .
    """
}
