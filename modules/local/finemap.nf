/*
 * FINEMAP — Sanger SNP fine-mapping for eGenes
 *
 * Input  : [celltype, egenes_csv, counts_h5, lod_genome_csv]
 * Output : [celltype, eqtl_finemapping.csv, eqtl_finemapping_peak.csv]
 *
 * Requires sanger_alleles.h5 in params.data_dir (shared across all cell types).
 * One job per cell type, fires after MERGE_RESULTS completes for that CT.
 */

process FINEMAP {

    label 'process_medium'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}" }, mode: 'copy'

    input:
    path  sanger_alleles
    tuple val(celltype), path(egenes_csv), path(counts_h5), path(lod_genome_csv)

    output:
    tuple val(celltype), path('eqtl_finemapping.csv'), path('eqtl_finemapping_peak.csv')

    script:
    """
    python ${projectDir}/workflow/scripts/run_finemap.py \\
        --egenes         ${egenes_csv} \\
        --counts         ${counts_h5} \\
        --lod            ${lod_genome_csv} \\
        --sanger-alleles ${sanger_alleles} \\
        --celltype       ${celltype} \\
        --output-dir     .
    """

    stub:
    """
    touch eqtl_finemapping.csv
    touch eqtl_finemapping_peak.csv
    """
}
