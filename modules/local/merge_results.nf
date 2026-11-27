process MERGE_RESULTS {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/core" }, mode: 'copy',
               saveAs: { fn -> fn.endsWith('_peak_lod.tsv') ? null : fn }

    input:
    path  gene_tss
    tuple val(celltype), val(scan_tag),
          path(lod_tables), path(peak_stats_tables), path(timing_jsons),
          path(gene_info)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_lod_genome.tsv'),
          path('*_peak_lod.tsv'),
          path('*_scan_stats.tsv')

    script:
    def tis    = params.tissue ?: 'run'
    def ct_log = celltype.replaceAll(/\s+/, '_')
    def tag    = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    def tss_arg = gene_tss.name != 'NO_GENE_TSS' ? gene_tss.name : '-'
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    python ${projectDir}/scripts/run_merge.py \\
        . ${gene_info} '${celltype}' ${scan_tag} . ${params.tissue} ${tss_arg} \\
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/merge.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_lod_genome.tsv ${tag}_peak_lod.tsv ${tag}_scan_stats.tsv
    """
}
