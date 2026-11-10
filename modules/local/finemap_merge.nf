process FINEMAP_MERGE {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype.replaceAll(/\s+/, '_')}/${scan_tag}/finemap" }, mode: 'copy'

    input:
    tuple val(celltype), val(scan_tag),
          path(fm_files),
          path(fm_peak_files)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_eqtl_finemapping.csv'),
          path('*_eqtl_finemapping_peak.csv')

    script:
    def tis    = params.tissue ?: 'run'
    def ct_log = celltype.replaceAll(/\s+/, '_')
    def tag    = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
    {
    for pat in '*_eqtl_finemapping.csv' '*_eqtl_finemapping_peak.csv'; do
        base=\${pat#\\*_}
        out="${tag}_\${base}"
        files=( \$(ls \${pat} 2>/dev/null | sort -V) )
        if [ \${#files[@]} -eq 0 ]; then
            touch "\${out}"
        else
            head -1 "\${files[0]}" > "\${out}"
            for f in "\${files[@]}"; do tail -n +2 "\${f}"; done >> "\${out}"
        fi
        echo "merged \${#files[@]} files -> \${out}"
    done
    } 2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/finemap_merge.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_eqtl_finemapping.csv ${tag}_eqtl_finemapping_peak.csv
    """
}
