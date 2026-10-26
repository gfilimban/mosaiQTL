/*
 * FINEMAP — Sanger SNP fine-mapping for eGenes (Wakefield ABF credible sets)
 *
 * Input  : path kinship_loco (shared)
 *          tuple [celltype, eqtl_peak.csv, phenotypes]
 * Output : tuple [celltype, eqtl_finemapping*.csv, eqtl_finemapping_peak*.csv]
 *
 * Requires sanger_alleles.h5 in params.data_dir (shared across all cell types).
 * One job per cell type, fires after REPORT.
 */

process FINEMAP {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    path  sanger_alleles
    tuple val(celltype), path(eqtl_peak), path(phenotypes)

    output:
    tuple val(celltype), path('eqtl_finemapping*.csv'), path('eqtl_finemapping_peak*.csv')

    script:
    """
    python ${projectDir}/scripts/run_finemap.py \\
        --egenes      ${eqtl_peak} \\
        --phenotypes  ${phenotypes} \\
        --kinship     ${kinship_loco} \\
        --metadata    ${params.metadata} \\
        --covariates  '${params.covariates}' \\
        --cache       ${genomecache} \\
        --alleles     ${sanger_alleles} \\
        --celltype    ${celltype} \\
        --output-dir  .
    """

    stub:
    """
    touch eqtl_finemapping.csv
    touch eqtl_finemapping_peak.csv
    """
}
