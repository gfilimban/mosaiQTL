/*
 * MERGE_RESULTS — concatenate per-chromosome LOD CSVs into lod_genome.csv
 *
 * Input  : [celltype, [lod_chr*.csv, ...]]  (groupTuple output from LMM_SCAN)
 * Output : [celltype, lod_genome.csv]
 *
 * One job per cell type, fires after all its chr jobs complete.
 * Published to {output_dir}/{celltype}/lod_genome.csv.
 */

process MERGE_RESULTS {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    tuple val(celltype), path(lod_csvs), path(timing_jsons)

    output:
    tuple val(celltype), path('lod_genome.csv'), path('peak_lod.csv'), path('scan_stats.csv')

    script:
    """
    python ${projectDir}/scripts/merge_lod_csvs.py \
        --celltype ${celltype} \
        --output   lod_genome.csv \
        ${lod_csvs}
    """

    stub:
    """
    touch lod_genome.csv peak_lod.csv scan_stats.csv
    """
}
