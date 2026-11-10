process FINEMAP_MERGE {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}/${scan_tag}" }, mode: 'copy'

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
    mkdir -p ${params.output_dir}/logs/${tis}/${scan_tag}/${ct_log}
    python - <<'PYEOF' 2>&1 | tee ${params.output_dir}/logs/${tis}/${scan_tag}/${ct_log}/finemap_merge.log
import glob, pandas as pd
from pathlib import Path

tag = '${tag}'

for suffix, out_name in [('*_eqtl_finemapping.csv', f'{tag}_eqtl_finemapping.csv'),
                          ('*_eqtl_finemapping_peak.csv', f'{tag}_eqtl_finemapping_peak.csv')]:
    files = sorted(glob.glob(suffix))
    if not files:
        open(out_name, 'w').close()
        continue
    # stream one chr at a time so peak memory is O(1 chr) not O(20 chr) + concat copy
    for i, f in enumerate(files):
        df = pd.read_csv(f, dtype={'gene_id': str, 'chr': str})
        df.to_csv(out_name, mode='a', header=(i == 0), index=False)
        del df
PYEOF
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_eqtl_finemapping.csv ${tag}_eqtl_finemapping_peak.csv
    """
}
