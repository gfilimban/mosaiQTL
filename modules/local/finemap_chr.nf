process FINEMAP_CHR {

    label 'process_medium'
    tag   "${celltype}:chr${chrom}"

    input:
    path  genomecache
    path  kinship_loco
    path  sanger_alleles
    tuple val(celltype), val(scan_tag), val(chrom),
          path(eqtl_summary), path(peak_lod), path(phenotypes), path(metadata)

    output:
    tuple val(celltype), val(scan_tag), val(chrom),
          path('*_eqtl_finemapping.csv'),
          path('*_eqtl_finemapping_peak.csv')

    script:
    def sanger = sanger_alleles.name != 'NO_SANGER' ? sanger_alleles.name : ''
    def tag    = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${launchDir}/${params.output_dir}/logs
    python ${projectDir}/scripts/run_finemap.py \\
        ${eqtl_summary} ${genomecache} ${kinship_loco} ${phenotypes} ${metadata} \\
        '${celltype}' ${scan_tag} '${params.covariates}' \\
        . ${sanger} ${peak_lod} ${params.tissue} ${chrom} \\
        2>&1 | tee ${launchDir}/${params.output_dir}/logs/${tag}_chr${chrom}_finemap.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_chr${chrom}_eqtl_finemapping.csv ${tag}_chr${chrom}_eqtl_finemapping_peak.csv
    """
}
