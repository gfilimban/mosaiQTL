process CALL_EGENES {

    label 'process_long'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/core" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), val(scan_tag),
          path(peak_lod, stageAs: 'merged_peak_lod.input.csv'), path(perm_max_lods),
          path(phenotypes), path(gene_info), path(metadata)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_peak_lod_all.csv'),
          path('*_pval_perm.json'),
          path('*_haplotype_eqtl_summary.csv'),
          path('*_haplotype_peak_lod.csv')

    script:
    def tis    = params.tissue ?: 'run'
    def ct_log = celltype.replaceAll(/\s+/, '_')
    def tag    = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    def rint   = (params.use_rint != null && !params.use_rint) ? 0 : 1
    def call_rule = params.call_rule ?: 'bh'
    def submission_fdr = params.submission_fdr ?: 0.1
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    python ${projectDir}/scripts/run_call_egenes.py \\
        ${peak_lod} ${perm_max_lods} ${phenotypes} ${gene_info} \\
        ${genomecache} ${kinship_loco} ${metadata} \\
        '${celltype}' ${scan_tag} '${params.covariates}' \\
        ${params.fdr_threshold} ${params.n_perms_cal} ${params.seed} . ${params.tissue} ${rint} ${call_rule} ${submission_fdr} \\
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/call_egenes.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_peak_lod_all.csv ${tag}_pval_perm.json ${tag}_haplotype_eqtl_summary.csv ${tag}_haplotype_peak_lod.csv
    """
}
