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
    tuple val(celltype), val(scan_tag), val(chrom), path(phenotypes), path(gene_info), path(metadata)

    output:
    tuple val(celltype), val(chrom),
          path("lod_chr${chrom}.tsv"),
          path("peak_stats_chr${chrom}.tsv"),
          path("timing_chr${chrom}.json")

    script:
    def cov      = params.covariates ?: ''
    def stype    = params.scan_type
    def wmb      = params.window_mb
    def rint     = (params.use_rint != null && !params.use_rint) ? 0 : 1
    def ct_log   = celltype.replaceAll(/\s+/, '_')
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${scan_tag}/${ct_log}
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
        2>&1 | tee ${params.output_dir}/logs/${scan_tag}/${ct_log}/lmm_scan_chr${chrom}.log
    """

    stub:
    """
    touch lod_chr${chrom}.tsv peak_stats_chr${chrom}.tsv timing_chr${chrom}.json
    """
}
