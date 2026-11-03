"""Build LOCO kinship HDF5 from a genome cache — called by Nextflow KINSHIP_LOCO process.

Usage: python build_kinship_loco.py <cache.h5> <output_filename.h5>

<output_filename.h5> is written in the current working directory.
"""
import sys, time
import logging
import numpy as np
import h5py
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    cache_path  = Path(sys.argv[1])
    out_path    = Path(sys.argv[2])

    if out_path.exists():
        log.info('kinship file already exists — skipping: %s', out_path)
        return

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship

    gc = GenomeCache.from_hdf5(cache_path)
    sample_ids = list(gc.sample_ids)
    chroms = list(gc.chromosomes)
    t0 = time.perf_counter()
    log.info('START kinship_loco: %d strains × %d chrs', len(sample_ids), len(chroms))

    kinship_loco = calc_kinship(gc, method='loco')
    decomp       = decomp_kinship(kinship_loco)

    with h5py.File(out_path, 'w') as fh:
        fh.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        for _ci, (chrom, d) in enumerate(decomp.items(), 1):
            grp = fh.create_group(f'chr{chrom}')
            grp.create_dataset('values',  data=d['values'])
            grp.create_dataset('vectors', data=d['vectors'])
            log.info('  [%d/%d] chr%s done', _ci, len(chroms), chrom)

    log.info('DONE kinship_loco: wrote %s  %.1fs', out_path, time.perf_counter() - t0)


if __name__ == '__main__':
    main()
