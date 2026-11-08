/*
 * LMM_SCAN — per-(celltype, chromosome) LMM QTL scan.
 * Parallelised: one SLURM job per (celltype × chromosome).
 */

process LMM_SCAN {

    label 'process_long'
    tag   "${celltype}/chr${chrom}"

    input:
    path  genomecache
    path  kinship_loco
    tuple val(celltype), val(chrom), path(phenotypes), path(gene_info), path(metadata)

    output:
    tuple val(celltype), val(chrom),
          path("lod_chr${chrom}.csv"),
          path("peak_stats_chr${chrom}.csv"),
          path("timing_chr${chrom}.json")

    script:
    def cov     = params.covariates ?: ''
    def stype   = params.scan_type  ?: 'L'
    def wmb     = params.window_mb  ?: 1.0
    def rint    = (params.use_rint != null && !params.use_rint) ? 0 : 1
    def ct_log  = celltype.replaceAll(/\s+/, '_')
    """
    set -o pipefail
    mkdir -p ${launchDir}/${params.output_dir}/logs
    python ${projectDir}/scripts/run_lmm_scan_chr.py \\
        ${genomecache} \\
        ${kinship_loco} \\
        ${phenotypes} \\
        ${gene_info} \\
        ${metadata} \\
        ${chrom} \\
        ${stype} \\
        ${wmb} \\
        '${cov}' \\
        . \\
        ${rint} \\
        2>&1 | tee ${launchDir}/${params.output_dir}/logs/${ct_log}_lmm_scan_chr${chrom}.log
    """

    stub:
    """
    touch lod_chr${chrom}.csv peak_stats_chr${chrom}.csv timing_chr${chrom}.json
    """
}
