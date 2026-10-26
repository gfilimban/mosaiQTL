"""Per-chromosome LMM QTL scan — called by Nextflow LMM_SCAN process."""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
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


def _write_timing(
    out: Path, chrom: str, n_genes: int,
    elapsed_s: float, peak_heap_mb: float, peak_rss_mb: float,
) -> None:
    (out / f'timing_chr{chrom}.json').write_text(json.dumps({
        'chr': chrom,
        'n_genes': n_genes,
        'elapsed_s': round(elapsed_s, 3),
        'peak_heap_mb': round(peak_heap_mb, 1),
        'peak_rss_mb': round(peak_rss_mb, 1),
    }))


def _load_phenotypes(
    h5_path: Path,
    chrom: str,
    gene_info: pd.DataFrame,
    scan_window: str,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str]]:
    """Load phenotype matrix for genes on chrom (cis/local) or all genes (genome).

    Returns
    -------
    pheno_mat : (n_genes, n_samples) float32
    gene_ids  : list of gene IDs, length n_genes
    sample_ids : list of sample IDs, length n_samples
    gene_chr_map : {gene_id: chrom} for cis/local filtering
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

        if scan_window in ('cis', 'local'):
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


_SW_MAP: dict[str, str] = {'L': 'local', 'C': 'cis', 'G': 'genome'}


@click.command()
@click.option('--cache', required=True, type=click.Path(exists=True))
@click.option('--kinship', required=True, type=click.Path(exists=True))
@click.option('--phenotypes', required=True, type=click.Path(exists=True))
@click.option('--gene-info', required=True, type=click.Path(exists=True))
@click.option('--chr', 'chrom', required=True)
@click.option('--metadata', required=True, type=click.Path(exists=True))
@click.option('--covariates', default='sex,dataset', show_default=True)
@click.option('--scan-type', default='L', show_default=True,
              type=click.Choice(['L', 'C', 'G']),
              help='L = local ±window_mb | C = chromosome-wide | G = genome-wide')
@click.option('--window-mb', default=1.0, show_default=True, type=float)
@click.option('--output-dir', default='.', show_default=True, type=click.Path())
def main(
    cache: str,
    kinship: str,
    phenotypes: str,
    gene_info: str,
    chrom: str,
    metadata: str,
    covariates: str,
    scan_type: str,
    window_mb: float,
    output_dir: str,
) -> None:
    """Per-chromosome LMM QTL scan for mosaiQTL Nextflow pipeline."""
    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.io.covariates import build_covariate_matrix
    from mosaiqtl.qtl.scan import scan_multi_gene

    _sw = _SW_MAP[scan_type]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f'lod_chr{chrom}.csv'

    log.info('starting chr%s scan (scan_type=%s)', chrom, scan_type)

    t_start = time.perf_counter()
    tracemalloc.start()

    gi = pd.read_csv(gene_info)
    pheno_mat, gene_ids, sample_ids, gene_chr_map = _load_phenotypes(
        Path(phenotypes), chrom, gi, _sw
    )

    if len(gene_ids) == 0:
        log.warning('no genes for chr%s — writing empty CSV', chrom)
        pd.DataFrame(columns=['gene_id', 'chr', 'locus', 'pos_mb', 'lod']).to_csv(
            out_csv, index=False
        )
        elapsed_s = time.perf_counter() - t_start
        _, peak_heap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        _write_timing(out, chrom, 0, elapsed_s, peak_heap / 1024 ** 2, 0.0)
        return

    meta = pd.read_csv(metadata)
    meta_geno_map: dict[str, str] = dict(zip(meta['sample_id'], meta['genotype']))
    geno_ids = [meta_geno_map.get(s, s) for s in sample_ids]

    # build covariate matrix aligned to scan sample order
    cov_cols = [c for c in covariates.split(',') if c.strip()]
    covariates_arr: 'np.ndarray | None' = None
    if cov_cols:
        meta_scan = meta[meta['sample_id'].isin(sample_ids)].drop_duplicates('sample_id')
        cov_df = build_covariate_matrix(meta_scan, covariates=cov_cols)
        cov_aligned = cov_df.reindex(sample_ids)
        if not cov_aligned.isna().any().any():
            covariates_arr = cov_aligned.values

    gc = GenomeCache.from_hdf5(Path(cache))
    kinship_decomp = _load_kinship(Path(kinship), chrom, sample_ids)

    gene_symbol_map = (
        dict(zip(gi['gene_id'].astype(str), gi['gene_symbol'].astype(str)))
        if 'gene_symbol' in gi.columns else None
    )

    # strand-aware gene boundary: end/1e6 for − strand genes, start/1e6 for + strand
    gene_pos_mb_map: 'dict[str, float] | None' = None
    if _sw == 'local':
        _has_strand = 'strand' in gi.columns
        gene_pos_mb_map = {}
        for _, row in gi.iterrows():
            gid = str(row['gene_id'])
            if _has_strand and row['strand'] == '-':
                gene_pos_mb_map[gid] = float(row['end']) / 1e6
            else:
                gene_pos_mb_map[gid] = float(row['start']) / 1e6

    scan_multi_gene(
        pheno_mat.T,
        gene_ids,
        gc,
        gene_chr_map=gene_chr_map,
        kinship_decomp=kinship_decomp,
        sample_ids=geno_ids,
        covariates=covariates_arr,
        use_rint=True,
        scan_window=_sw,
        gene_pos_mb_map=gene_pos_mb_map,
        window_mb=window_mb,
        output_dir=out,
        gene_symbol_map=gene_symbol_map,
    )

    elapsed_s = time.perf_counter() - t_start
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    from mosaiqtl.utils import get_rss_mb
    _write_timing(out, chrom, len(gene_ids), elapsed_s, peak_heap / 1024 ** 2, get_rss_mb())
    log.info('wrote %s (%.1fs)', out_csv, elapsed_s)


if __name__ == '__main__':
    main()
