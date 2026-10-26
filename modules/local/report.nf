/*
 * REPORT — eGene calling + submission files (eqtl_summary, eqtl_peak, eqtl_stats)
 *
 * Input  : [celltype, peak_lod.csv, perm_dir/, allele_effects.csv]
 * Output : [celltype, eqtl_summary.csv, eqtl_peak*.csv, eqtl_stats*.csv,
 *           run_metadata.json]
 *
 * run_metadata.json captures mosaiqtl package version, git hash of the repo,
 * celltype label, Nextflow run name, and UTC timestamp for provenance.
 *
 * Calls `mosaiqtl report`.  Fires after PERMUTE + ESTIMATE both complete.
 */

process REPORT {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    tuple val(celltype),
          path(peak_lod),
          path(perm_files),   // list: perm_max_lods.csv + perm_max_lods_chr*.csv — staged in work dir; script uses --perm-dir .
          path(allele_effects),
          path(scan_stats)

    output:
    tuple val(celltype),
          path('eqtl_summary.csv'),
          path('eqtl_peak*.csv'),
          path('eqtl_stats*.csv'),
          path('run_metadata.json'),
          path('scan_info.json'),
          path('sample_breakdown.csv')

    script:
    def meta_path  = params.metadata ?: (params.data_dir ? "${params.data_dir}/igvf_sample_metadata.csv" : '')
    def pheno_path = params.celltypes_dir ? "${params.celltypes_dir}/${celltype}/counts_lognorm.h5"
                                          : (params.phenotypes ?: '')
    """
    python ${projectDir}/scripts/run_report.py \\
        --scan-result    ${peak_lod} \\
        --perm-dir       . \\
        --allele-effects ${allele_effects} \\
        --scan-stats     ${scan_stats} \\
        --output-dir     . \\
        --celltype       '${celltype}' \\
        --project-dir    '${projectDir}' \\
        --nf-run-name    '${workflow.runName}' \\
        --metadata       '${meta_path}' \\
        --phenotypes     '${pheno_path}'
    """

    stub:
    """
    touch eqtl_summary.csv eqtl_peak.csv eqtl_stats.csv run_metadata.json scan_info.json sample_breakdown.csv
    """
}
