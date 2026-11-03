process MERGE_RESULTS {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}/${scan_tag}" }, mode: 'copy'

    input:
    tuple val(celltype), val(scan_tag),
          path(lod_csvs), path(peak_stats_csvs), path(timing_jsons),
          path(gene_info)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_lod_genome.csv'),
          path('*_peak_lod.csv'),
          path('*_scan_stats.csv')

    script:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${launchDir}/${params.output_dir}/logs
    python ${projectDir}/scripts/run_merge.py \\
        . ${gene_info} '${celltype}' ${scan_tag} . ${params.tissue} \\
        2>&1 | tee ${launchDir}/${params.output_dir}/logs/${tag}_merge.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_lod_genome.csv ${tag}_peak_lod.csv ${tag}_scan_stats.csv
    """
}
