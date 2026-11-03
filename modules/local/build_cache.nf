process BUILD_CACHE {

    label 'process_long'
    tag   'igvf_cache'

    publishDir "${launchDir}/ref", mode: 'copy'

    input:
    path gigamuga

    output:
    path 'igvf_cache.h5', emit: cache_h5

    script:
    """
    python ${projectDir}/scripts/build_cache.py \\
        ${gigamuga} .
    """

    stub:
    """
    touch igvf_cache.h5
    """
}
