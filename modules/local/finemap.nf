process FINEMAP {

    label 'process_long'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/finemap" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    path  sanger_alleles
    tuple val(celltype), val(scan_tag),
          path(eqtl_summary), path(peak_lod), path(phenotypes), path(metadata)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_eqtl_finemapping.csv'),
          path('*_eqtl_finemapping_peak.csv')

    script:
    def sanger = sanger_alleles.name != 'NO_SANGER' ? sanger_alleles.name : ''
    def tis    = params.tissue ?: 'run'
    def ct_log = celltype.replaceAll(/\s+/, '_')
    def tag    = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    # schema_ver=${params.finemap_schema_ver ?: '1'}
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    python ${projectDir}/scripts/run_finemap.py \\
        ${eqtl_summary} ${genomecache} ${kinship_loco} ${phenotypes} ${metadata} \\
        '${celltype}' ${scan_tag} '${params.covariates}' \\
        . ${sanger} ${peak_lod} ${params.tissue} \\
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/finemap.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_eqtl_finemapping.csv ${tag}_eqtl_finemapping_peak.csv
    """
}
