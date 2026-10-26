/*
 * ESTIMATE — founder haplotype effects at each gene's peak locus
 *
 * Input  : path kinship_loco (shared)
 *          tuple [celltype, peak_lod.csv, phenotypes]
 * Output : tuple [celltype, allele_effects.csv]
 *
 * One job per cell type, fires after MERGE_RESULTS.
 */

process ESTIMATE {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), path(peak_lod), path(phenotypes)

    output:
    tuple val(celltype), path('allele_effects.csv')

    script:
    """
    python ${projectDir}/scripts/run_estimate.py \\
        --cache       ${genomecache} \\
        --scan-result ${peak_lod} \\
        --phenotypes  ${phenotypes} \\
        --kinship     ${kinship_loco} \\
        --metadata    ${params.metadata} \\
        --covariates  '${params.covariates}' \\
        --output      allele_effects.csv
    """

    stub:
    """
    touch allele_effects.csv
    """
}
