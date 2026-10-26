/*
 * PERMUTE — LMM permutation threshold via GEV fit on max-LOD distribution
 *
 * Input  : path kinship_loco (shared)
 *          tuple [celltype, gene_info, phenotypes]
 * Output : tuple [celltype, perm_max_lods.csv, perm_max_lods_chr*.csv (glob)]
 *
 * One job per cell type. Uses pre-built kinship from KINSHIP_LOCO — no NB null fit needed.
 */

process PERMUTE {

    label 'process_long'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), path(gene_info), path(phenotypes)

    output:
    tuple val(celltype), path('perm_max_lods.csv'), path('perm_max_lods_chr*.csv')

    script:
    """
    python ${projectDir}/scripts/run_permute.py \\
        --cache       ${genomecache} \\
        --phenotypes  ${phenotypes} \\
        --kinship     ${kinship_loco} \\
        --metadata    ${params.metadata ?: params.data_dir + '/igvf_sample_metadata.csv'} \\
        --gene-info   ${gene_info} \\
        --n-perms     ${params.n_perms} \\
        --scan-window genome \\
        --window-mb   ${params.window_mb} \\
        --output      .
    """

    stub:
    """
    touch perm_max_lods.csv
    touch perm_max_lods_chr1.csv
    """
}
