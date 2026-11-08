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
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    python - <<'PYEOF'
import glob, pandas as pd
from pathlib import Path

tag = '${tag}'

for suffix, out_name in [('*_eqtl_finemapping.csv', f'{tag}_eqtl_finemapping.csv'),
                          ('*_eqtl_finemapping_peak.csv', f'{tag}_eqtl_finemapping_peak.csv')]:
    files = sorted(glob.glob(suffix))
    if not files:
        open(out_name, 'w').close()
        continue
    frames = [pd.read_csv(f, dtype={'gene_id': str, 'chr': str}) for f in files]
    pd.concat(frames, ignore_index=True).to_csv(out_name, index=False)
PYEOF
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_eqtl_finemapping.csv ${tag}_eqtl_finemapping_peak.csv
    """
}
