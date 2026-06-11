"""Per-chromosome LMM QTL scan — called by Nextflow LMM_SCAN process."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def _load_phenotypes(
    h5_path: Path,
    chrom: str,
    gene_info: pd.DataFrame,
    scan_window: str,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str]]:
    """Load phenotype matrix for genes on chrom (cis) or all genes (genome).

    Returns
    -------
    pheno_mat : (n_genes, n_samples) float32
    gene_ids  : list of gene IDs, length n_genes
    sample_ids : list of sample IDs, length n_samples
    gene_chr_map : {gene_id: chrom} for cis filtering
    """
    with h5py.File(h5_path, 'r') as f:
        # accept both canonical names (gene_ids/sample_ids) and legacy (genes/samples)
        _sid_key = 'sample_ids' if 'sample_ids' in f else 'samples'
        _gid_key = 'gene_ids' if 'gene_ids' in f else 'genes'
        sample_ids: list[str] = [s.decode() if isinstance(s, bytes) else s
                                  for s in f[_sid_key][:]]
        all_gene_ids: list[str] = [g.decode() if isinstance(g, bytes) else g
                                    for g in f[_gid_key][:]]
        gene_chr_map = dict(zip(gene_info['gene_id'], gene_info['chr'].astype(str)))

        if scan_window == 'cis':
            chrom_genes = gene_info.loc[
                gene_info['chr'].astype(str) == str(chrom), 'gene_id'
            ].tolist()
            indices = [i for i, g in enumerate(all_gene_ids) if g in set(chrom_genes)]
        else:
            indices = list(range(len(all_gene_ids)))

        if not indices:
            log.warning('no genes for chr%s (scan_window=%s)', chrom, scan_window)
            return (
                np.empty((0, len(sample_ids)), dtype=np.float32),
                [],
                sample_ids,
                gene_chr_map,
            )

        counts: np.ndarray = f['counts'][np.array(indices), :].astype(np.float32)
        gene_ids = [all_gene_ids[i] for i in indices]

    log.info('loaded %d genes × %d samples for chr%s', len(gene_ids), len(sample_ids), chrom)
    return counts, gene_ids, sample_ids, gene_chr_map


def _load_kinship(
    h5_path: Path,
    chrom: str,
    sample_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Load LOCO kinship eigendecomposition for chrom from HDF5.

    Expected HDF5 layout (written by kinship_loco.py or equivalent):
        /chr{c}/values   — eigenvalues  (n_strains,)
        /chr{c}/vectors  — eigenvectors (n_strains, n_strains)
        /sample_ids      — strain-level sample IDs

    Returns
    -------
    dict mapping chrom -> {'values': ..., 'vectors': ...}
    """
    key = f'chr{chrom}'
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f'kinship HDF5 missing group {key!r}')
        values = f[key]['values'][:]
        vectors = f[key]['vectors'][:]
    return {chrom: {'values': values, 'vectors': vectors}}


@click.command()
@click.option('--cache', required=True, type=click.Path(exists=True),
              help='Genome cache HDF5.')
@click.option('--kinship', required=True, type=click.Path(exists=True),
              help='LOCO kinship HDF5.')
@click.option('--phenotypes', required=True, type=click.Path(exists=True),
              help='Phenotype HDF5 (counts, gene_ids, sample_ids datasets).')
@click.option('--gene-info', required=True, type=click.Path(exists=True),
              help='Gene info CSV (gene_id, chr, start, end).')
@click.option('--chr', 'chrom', required=True,
              help='Chromosome to scan (e.g. 8 or X).')
@click.option('--scan-window', default='cis', show_default=True,
              type=click.Choice(['cis', 'genome', 'local']),
              help='Scan window type.')
@click.option('--window-mb', default=1.0, show_default=True, type=float,
              help='Half-window for local scan (Mb).')
@click.option('--output-dir', default='.', show_default=True,
              type=click.Path(),
              help='Directory to write lod_chr{chrom}.csv.')
def main(
    cache: str,
    kinship: str,
    phenotypes: str,
    gene_info: str,
    chrom: str,
    scan_window: str,
    window_mb: float,
    output_dir: str,
) -> None:
    """Per-chromosome LMM QTL scan for mosaiQTL Nextflow pipeline."""
    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.qtl.scan import scan_multi_gene

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f'lod_chr{chrom}.csv'

    log.info('starting chr%s scan', chrom)

    gi = pd.read_csv(gene_info)
    pheno_mat, gene_ids, sample_ids, gene_chr_map = _load_phenotypes(
        Path(phenotypes), chrom, gi, scan_window
    )

    if len(gene_ids) == 0:
        log.warning('no genes for chr%s — writing empty CSV', chrom)
        pd.DataFrame(columns=['gene', 'chr', 'locus', 'pos_mb', 'lod']).to_csv(
            out_csv, index=False
        )
        return

    gc = GenomeCache.from_hdf5(Path(cache))
    kinship_decomp = _load_kinship(Path(kinship), chrom, sample_ids)

    scan_multi_gene(
        pheno_mat,
        gene_ids,
        gc,
        gene_chr_map=gene_chr_map,
        kinship_decomp=kinship_decomp,
        sample_ids=sample_ids,
        use_rint=True,
        scan_window=scan_window,
        window_mb=window_mb,
        output_dir=out,
    )

    log.info('wrote %s', out_csv)


if __name__ == '__main__':
    main()
