process FINEMAP_CHR {

    label 'process_ultra'
    tag   "${celltype}:chr${chrom}"

    input:
    path  genomecache
    path  kinship_loco
    path  sanger_alleles
    tuple val(celltype), val(scan_tag), val(chrom),
          path(eqtl_summary), path(peak_lod), path(phenotypes), path(metadata)

    output:
    tuple val(celltype), val(scan_tag), val(chrom),
          path('*_eqtl_finemapping.tsv'),
          path('*_eqtl_finemapping_peak.tsv'),
          path('*_eqtl_finemapping_groups.tsv'),
          path('*_eqtl_ci.tsv')

    script:
    def sanger  = sanger_alleles.name != 'NO_SANGER' ? sanger_alleles.name : '-'
    def tissue  = params.tissue ?: '-'
    def rint    = params.use_rint ? 1 : 0
    def tis     = params.tissue ?: 'run'
    def ct_log  = celltype.replaceAll(/\s+/, '_')
    def tag     = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    python ${projectDir}/scripts/run_finemap.py \\
        ${eqtl_summary} ${genomecache} ${kinship_loco} ${phenotypes} ${metadata} \\
        '${celltype}' ${scan_tag} '${params.covariates}' \\
        . ${sanger} ${peak_lod} ${tissue} ${chrom} \\
        '${params.scan_type}' ${params.window_mb} ${rint} \\
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/finemap_chr${chrom}.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_chr${chrom}_eqtl_finemapping.tsv ${tag}_chr${chrom}_eqtl_finemapping_peak.tsv ${tag}_chr${chrom}_eqtl_finemapping_groups.tsv ${tag}_chr${chrom}_eqtl_ci.tsv
    """
}
