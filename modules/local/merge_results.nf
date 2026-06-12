/*
 * MERGE_RESULTS — concatenate per-chromosome LOD CSVs into lod_genome.csv
 */

process MERGE_RESULTS {

    label 'process_single'
    tag   'merge'

    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path lod_csvs

    output:
    path 'lod_genome.csv'

    stub:
    """
    touch lod_genome.csv
    """

    script:
    """
    python - <<'PYEOF'
import glob
import pandas as pd

csvs = sorted(glob.glob('lod_chr*.csv'))
if not csvs:
    raise FileNotFoundError('no per-chr LOD CSVs found')

# natural sort: chr1 < chr2 < ... < chr19 < chrX
def _chr_key(path):
    stem = path.split('lod_chr')[1].replace('.csv', '')
    return (0, int(stem)) if stem.isdigit() else (1, stem)

pd.concat(
    [pd.read_csv(f) for f in sorted(csvs, key=_chr_key)],
    ignore_index=True,
).to_csv('lod_genome.csv', index=False)
PYEOF
    """
}
