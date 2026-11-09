process REPORT {

    label 'process_single'
    tag   "${celltype}"

    publishDir { "${params.output_dir}/${celltype}/${scan_tag}" }, mode: 'copy'

    input:
    path  genomecache
    path  kinship_loco
    path  gtf
    tuple val(celltype), val(scan_tag),
          path(peak_lod), path(eqtl_summary),
          path(pval_perm_json), path(perm_max_lods),
          path(perm_max_lods_chrs),
          path(gene_info), path(lod_genome)

    output:
    tuple val(celltype), val(scan_tag),
          path('*_eqtl_stats.csv'),
          path('*_genome_summary.png'),
          path('*_genome_summary_share.png'),
          path('*_genome_diagnostic.png'),
          path('*_genome_diagnostic_share.png'),
          path('*_perm_distribution.png'),
          path('*_perm_distribution_share.png'),
          path('*_egene_lod_dist.png'),
          path('*_egene_lod_dist_share.png'),
          path('run_metadata.json')
    path '*_loci_per_chr*.png',        optional: true
    path '*_lod_panel_grid*.png',      optional: true
    path '*_perm_panel_grid*.png',     optional: true
    path '*_cis_distance_panel*.png',  optional: true
    path '*_cis_histogram*.png',       optional: true
    path '*_effects_grid*.png',        optional: true
    path '*_kinship_heatmap*.png',     optional: true
    path '*_kinship_dist*.png',        optional: true
    path '*_lod_profile_chr*.png',     optional: true
    path '*_perm_dist_chr*.png',       optional: true
    path '*_effects_chr*.png',         optional: true
    path '*_haplotype_heatmap*.png',   optional: true
    path '*_manhattan*.png',           optional: true
    path '*_locus_panel_grid*.png',    optional: true
    path '*_locus_panel_chr*.png',     optional: true

    script:
    def gtf_arg = gtf.name != 'NO_GTF' ? gtf.name : '-'
    def tag     = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs
    python ${projectDir}/scripts/run_report.py \\
        ${peak_lod} ${eqtl_summary} ${pval_perm_json} ${perm_max_lods} \\
        ${gene_info} ${lod_genome} ${kinship_loco} ${genomecache} \\
        '${celltype}' ${scan_tag} . ${gtf_arg} ${params.tissue} \\
        2>&1 | tee ${params.output_dir}/logs/${tag}_report.log
    """

    stub:
    def tag = params.tissue ? "${params.tissue}_${celltype}_${scan_tag}" : "${celltype}_${scan_tag}"
    """
    touch ${tag}_eqtl_stats.csv \\
          ${tag}_genome_summary.png ${tag}_genome_summary_share.png \\
          ${tag}_genome_diagnostic.png ${tag}_genome_diagnostic_share.png \\
          ${tag}_perm_distribution.png ${tag}_perm_distribution_share.png \\
          ${tag}_egene_lod_dist.png ${tag}_egene_lod_dist_share.png \\
          ${tag}_manhattan.png ${tag}_manhattan_share.png \\
          ${tag}_haplotype_heatmap.png ${tag}_haplotype_heatmap_share.png \\
          ${tag}_locus_panel_grid.png ${tag}_locus_panel_grid_share.png \\
          run_metadata.json
    """
}
