/*
 * ESTIMATE — founder haplotype effects at each gene's peak locus
 *
 * Input  : [celltype, peak_lod.csv, phenotypes_path]
 * Output : [celltype, allele_effects.csv]
 *
 * One job per cell type, fires after MERGE_RESULTS.
 */

process ESTIMATE {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    tuple val(celltype), path(peak_lod), path(phenotypes)

    output:
    tuple val(celltype), path('allele_effects.csv')

    script:
    """
    python ${projectDir}/workflow/scripts/run_estimate.py \\
        --cache       ${genomecache} \\
        --scan-result ${peak_lod} \\
        --counts      ${phenotypes} \\
        --metadata    ${params.metadata} \\
        --covariates  '${params.covariates}' \\
        --output      allele_effects.csv
    """

    stub:
    """
    touch allele_effects.csv
    """
}
