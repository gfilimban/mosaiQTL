process FINEMAP_MERGE {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/finemap" }, mode: 'copy',
               saveAs: { fn -> fn.endsWith('_haplotype_eqtl_summary.tsv') ? null : fn }
    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/core" }, mode: 'copy',
               pattern: '*_haplotype_eqtl_summary.tsv'

    input:
    tuple val(celltype), val(scan_tag),
          path(fm_files),
          path(fm_peak_files),
          path(fm_group_files),
          path(ci_files),
          path(call_summary, stageAs: 'call_summary.input.tsv')

    output:
    tuple val(celltype), val(scan_tag),
          path('*_eqtl_finemapping.tsv.gz'),
          path('*_eqtl_finemapping_peak.tsv'),
          path('*_eqtl_finemapping_groups.tsv'),
          path('*_eqtl_ci.tsv'),
          path('*_haplotype_eqtl_summary.tsv')

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
    touch ${tag}_eqtl_finemapping.tsv.gz ${tag}_eqtl_finemapping_peak.tsv ${tag}_eqtl_finemapping_groups.tsv ${tag}_eqtl_ci.tsv ${tag}_haplotype_eqtl_summary.tsv
    """
}
