/*
 * CHARACTERISE — LOD-drop CI for eGenes (fills ci_lo_bp / ci_hi_bp)
 *
 * Input  : path kinship_loco (shared)
 *          tuple [celltype, eqtl_peak.csv, phenotypes]
 * Output : tuple [celltype, eqtl_peak_ci.csv]
 *
 * One job per cell type, fires after REPORT and before FINEMAP.
 * Rescans each eGene's chromosome with LMM to compute 1.5-LOD support
 * intervals, then converts Mb → bp for the IGVF submission schema.
 */

process CHARACTERISE {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), path(eqtl_peak), path(phenotypes)

    output:
    tuple val(celltype), path('eqtl_peak_ci*.csv')

    script:
    """
    python ${projectDir}/scripts/run_characterise.py \\
        --egenes      ${eqtl_peak} \\
        --phenotypes  ${phenotypes} \\
        --kinship     ${kinship_loco} \\
        --cache       ${genomecache} \\
        --metadata    ${params.metadata ?: params.data_dir + '/igvf_sample_metadata.csv'} \\
        --covariates  '${params.covariates}' \\
        --output-dir  .
    """

    stub:
    """
    touch eqtl_peak_ci.csv
    """
}
