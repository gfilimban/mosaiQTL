process FINEMAP_MERGE {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/finemap" }, mode: 'copy',
               saveAs: { fn -> fn.endsWith('_haplotype_eqtl_summary.csv') ? null : fn }
    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/core" }, mode: 'copy',
               pattern: '*_haplotype_eqtl_summary.csv'

    input:
    tuple val(celltype), val(scan_tag),
          path(fm_files),
          path(fm_peak_files),
          path(fm_group_files),
          path(ci_files),
          path(call_summary, stageAs: 'call_summary.input.csv')

    output:
    tuple val(celltype), val(scan_tag),
          path('*_eqtl_finemapping.csv.gz'),
          path('*_eqtl_finemapping_peak.csv'),
          path('*_eqtl_finemapping_groups.csv'),
          path('*_eqtl_ci.csv'),
          path('*_haplotype_eqtl_summary.csv')

    script:
    def tis    = params.tissue ?: 'run'
    def ct_log = celltype.replaceAll(/\s+/, '_')
    def tag    = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    python ${projectDir}/scripts/run_finemap_merge.py \\
        . '${tag}' . ${call_summary} \\
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/finemap_merge.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_eqtl_finemapping.csv.gz ${tag}_eqtl_finemapping_peak.csv ${tag}_eqtl_finemapping_groups.csv ${tag}_eqtl_ci.csv ${tag}_haplotype_eqtl_summary.csv
    """
}
