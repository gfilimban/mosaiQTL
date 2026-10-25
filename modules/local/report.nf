/*
 * REPORT — submission file writer: eqtl_peak.csv + eqtl_stats.csv
 *
 * Input  : [celltype, egenes_csv, allele_effects_csv, perm_stats_csv, lod_genome_csv,
 *            eqtl_finemapping_csv, eqtl_finemapping_peak_csv]
 * Output : [celltype, eqtl_peak.csv, eqtl_stats.csv]
 *
 * Runs after FINEMAP; writes the full EQTL_PEAK_COLS + EQTL_STATS_COLS schemas.
 */

process REPORT {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    tuple val(celltype),
          path(egenes_csv),
          path(allele_effects_csv),
          path(perm_stats_csv),
          path(lod_genome_csv),
          path(eqtl_finemapping_csv),
          path(eqtl_finemapping_peak_csv)

    output:
    tuple val(celltype), path('eqtl_peak.csv'), path('eqtl_stats.csv')

    script:
    """
    python ${projectDir}/workflow/scripts/run_report.py \\
        --egenes            ${egenes_csv} \\
        --allele-effects    ${allele_effects_csv} \\
        --perm-stats        ${perm_stats_csv} \\
        --lod               ${lod_genome_csv} \\
        --finemapping       ${eqtl_finemapping_csv} \\
        --finemapping-peak  ${eqtl_finemapping_peak_csv} \\
        --celltype          ${celltype} \\
        --output-dir        .
    """

    stub:
    """
    touch eqtl_peak.csv
    touch eqtl_stats.csv
    """
}
