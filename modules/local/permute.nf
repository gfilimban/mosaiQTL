/*
 * PERMUTE — permutation threshold via GEV fit on max-LOD distribution
 *
 * Input  : [celltype, null_fit.csv, gene_info_path, phenotypes_path]
 * Output : [celltype, perm_max_lods.csv, perm_max_lods_chr*.csv (glob)]
 *
 * One job per cell type, fires after FIT_NULL.
 */

process PERMUTE {

    label 'process_long'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    tuple val(celltype), path(null_fit), path(gene_info), path(phenotypes)

    output:
    tuple val(celltype), path('perm_max_lods.csv'), path('perm_max_lods_chr*.csv')

    script:
    """
    python ${projectDir}/workflow/scripts/run_permute.py \\
        --cache       ${genomecache} \\
        --counts      ${phenotypes} \\
        --metadata    ${params.metadata} \\
        --null-fit    ${null_fit} \\
        --gene-info   ${gene_info} \\
        --n-perms     ${params.n_perms} \\
        --scan-window local \\
        --window-mb   ${params.window_mb} \\
        --covariates  '${params.covariates}' \\
        --output      .
    """

    stub:
    """
    touch perm_max_lods.csv
    touch perm_max_lods_chr1.csv
    """
}
