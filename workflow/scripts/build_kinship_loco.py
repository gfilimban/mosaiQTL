"""Build LOCO kinship HDF5 from a genome cache — called by Nextflow KINSHIP_LOCO process."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


@click.command()
@click.option('--cache', required=True, type=click.Path(exists=True),
              help='Genome cache HDF5.')
@click.option('--output', required=True, type=click.Path(),
              help='Output kinship_loco.h5 path.')
def main(cache: str, output: str) -> None:
    """Compute per-chromosome LOCO kinship and eigendecompose into HDF5.

    Output layout: /chr{c}/values (eigenvalues), /chr{c}/vectors (eigenvectors),
    /sample_ids (strain labels).
    """
    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship

    log.info('loading genome cache from %s', cache)
    gc = GenomeCache.from_hdf5(Path(cache))
    log.info('cache: %d strains, %d chromosomes', len(gc.sample_ids), len(gc.chromosomes))

    log.info('computing LOCO kinship …')
    loco = calc_kinship(gc, method='loco')

    log.info('eigendecomposing %d matrices …', len(loco))
    decomp = decomp_kinship(loco)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as f:
        sids = np.array([s.encode() for s in gc.sample_ids])
        f.create_dataset('sample_ids', data=sids)
        for chrom, d in decomp.items():
            grp = f.create_group(f'chr{chrom}')
            grp.create_dataset('values',  data=d['values'].astype(np.float64))
            grp.create_dataset('vectors', data=d['vectors'].astype(np.float64))
            log.info('  chr%s: top-3 λ=%s', chrom, d['values'][:3])

    log.info('wrote %s', out)


if __name__ == '__main__':
    main()
