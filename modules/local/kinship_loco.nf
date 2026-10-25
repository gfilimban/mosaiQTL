/*
 * KINSHIP_LOCO — compute per-chromosome LOCO kinship from genome cache
 *
 * Run once per dataset; output can be cached with -resume.
 */

process KINSHIP_LOCO {

    label 'process_medium'
    tag   'kinship_loco'

    publishDir "${params.data_dir}", mode: 'copy'

    input:
    path genomecache

    output:
    path 'kinship_loco.h5', emit: kinship_h5

    script:
    """
    python ${projectDir}/workflow/scripts/build_kinship_loco.py \\
        --cache  ${genomecache} \\
        --output kinship_loco.h5
    """

    stub:
    """
    touch kinship_loco.h5
    """
}
