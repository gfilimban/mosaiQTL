/*
 * KINSHIP_LOCO — build per-chromosome LOCO kinship HDF5 from genome cache.
 *
 * {projectDir}/ref/kinship_loco.h5 is the single canonical copy. The workflow
 * invokes this process only when that file is absent; publishDir then places
 * the freshly built file in ref/, so every later run reuses it without a rebuild.
 */

process KINSHIP_LOCO {

    label 'process_medium'
    tag   'kinship_loco'

    publishDir "${launchDir}/ref", mode: 'copy'

    input:
    path genomecache

    output:
    path 'igvf_kinship_loco.h5', emit: kinship_h5

    script:
    def tis = params.tissue ?: 'run'
    """
    set -o pipefail
    mkdir -p ${params.output_dir}/logs/${tis}
    python ${projectDir}/scripts/build_kinship_loco.py \\
        ${genomecache} \\
        igvf_kinship_loco.h5 \\
        2>&1 | tee ${params.output_dir}/logs/${tis}/kinship_loco.log
    """

    stub:
    """
    touch igvf_kinship_loco.h5
    """
}
