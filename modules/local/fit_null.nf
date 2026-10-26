/*
 * FIT_NULL — fit LMM null model per gene
 *
 * Input  : [celltype, phenotypes_path, gene_info_path]
 * Output : [celltype, null_fit.csv]
 *
 * One job per cell type.  null_fit.csv feeds PERMUTE.
 */

process FIT_NULL {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), path(phenotypes), path(gene_info)

    output:
    tuple val(celltype), path('null_fit.csv')

    script:
    """
    python ${projectDir}/scripts/run_fit_null.py \\
        --cache       ${genomecache} \\
        --counts      ${phenotypes} \\
        --metadata    ${params.metadata} \\
        --covariates  '${params.covariates}' \\
        --output      null_fit.csv
    """

    stub:
    """
    touch null_fit.csv
    """
}
