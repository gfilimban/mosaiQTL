process PERMUTE {

    label 'process_long'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/core" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), val(scan_tag), path(phenotypes), path(peak_lod), path(metadata)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_perm_max_lods.tsv'),
          path('*_perm_max_lods_chr*.tsv'),
          path('*_perm_gev_params.json')

    script:
    def rint    = params.use_rint ? 1 : 0
    def tis     = params.tissue ?: 'run'
    def tissue_q = "'${params.tissue ?: ''}'"
    def ct_log  = celltype.replaceAll(/\s+/, '_')
    def tag     = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    python ${projectDir}/scripts/run_permute.py \\
        ${genomecache} ${kinship_loco} ${phenotypes} ${peak_lod} ${metadata} \\
        '${celltype}' ${scan_tag} '${params.covariates}' \\
        ${params.n_perms} ${params.seed} ${params.perm_method} . \\
        '${params.scan_type}' ${params.window_mb} ${tissue_q} ${rint} \\
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/permute.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_perm_max_lods.tsv ${tag}_perm_max_lods_chr1.tsv ${tag}_perm_gev_params.json
    """
}
