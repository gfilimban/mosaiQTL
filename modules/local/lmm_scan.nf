/*
 * LMM_SCAN — per-(celltype, chromosome) LMM QTL scan
 *
 * Input  : [celltype, phenotypes_path, gene_info_path, chrom]  (from ct_chr channel)
 * Output : [celltype, lod_chr{chrom}.csv]
 *
 * Runs one job per (celltype × chromosome) pair.  20 chrs × N cell types = N×20
 * parallel SLURM array jobs; kinship is a shared value channel (built once).
 */

process LMM_SCAN {

    label 'process_medium'
    tag   "${celltype}/chr${chrom}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), path(phenotypes), path(gene_info), val(chrom)

    output:
    tuple val(celltype), path("lod_chr${chrom}.csv")

    script:
    """
    python ${projectDir}/scripts/run_lmm_scan_chr.py \\
        --cache       ${genomecache} \\
        --kinship     ${kinship_loco} \\
        --phenotypes  ${phenotypes} \\
        --gene-info   ${gene_info} \\
        --chr         ${chrom} \\
        --metadata    ${params.metadata ?: params.data_dir + '/igvf_sample_metadata.csv'} \\
        --covariates  '${params.covariates}' \\
        --scan-window ${params.scan_window} \\
        --window-mb   ${params.window_mb} \\
        --output-dir  .
    """

    stub:
    """
    touch lod_chr${chrom}.csv
    """
}
