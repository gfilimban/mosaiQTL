"""Per-chromosome LMM QTL scan — called by Nextflow LMM_SCAN process.

Usage: python run_lmm_scan_chr.py <cache.h5> <kinship.h5> <phenotypes.h5>
           <gene_info.csv> <metadata.csv> <chrom>
           <scan_type:L|C|G> <window_mb> <covariates> <output_dir>
           [use_rint:0|1]  (default 1)

use_rint=1 is correct only for LMM on log-normalised expression (the current
scan path).  Set to 0 for Gaussian phenotypes that are already approximately
normal (e.g. body weight), or when comparing against a reference that did not
apply RINT.  The NB-GLMM path does not use this flag at all — RINT on raw
counts invalidates the negative-binomial likelihood.
"""
import sys, logging, time, tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    cache_path, kinship_path, pheno_path, gene_info_path, meta_path = [Path(sys.argv[i]) for i in range(1, 6)]
    chrom       = sys.argv[6]
    scan_type   = sys.argv[7]
    window_mb   = float(sys.argv[8])
    covariates  = sys.argv[9]
    out         = Path(sys.argv[10])
    use_rint    = sys.argv[11] != '0' if len(sys.argv) > 11 else True

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.atomic import atomic_write_json, atomic_write_table
    from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
    from mosaiqtl.pipeline import ALLELE_EFFECTS_BASE_COLS, LOD_COLS, PEAK_LOD_COLS, SCAN_TYPE_TO_WINDOW
    from mosaiqtl.qtl.scan import scan_multi_gene_lmm
    from mosaiqtl.utils import build_tss_lookup, get_rss_mb

    scan_window = SCAN_TYPE_TO_WINDOW[scan_type]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tracemalloc.start()

    gi = pd.read_csv(gene_info_path)
    gene_chr_map = dict(zip(gi['gene_id'].astype(str), gi['chr'].astype(str)))

    if scan_window in ('cis', 'local'):
        genes_wanted = gi.loc[gi['chr'].astype(str) == str(chrom), 'gene_id'].tolist()
    else:
        genes_wanted = gi['gene_id'].tolist()

    pheno_mat, gene_ids, sample_ids = load_expression_h5(pheno_path, genes_wanted)

    lod_tsv  = out / f'lod_chr{chrom}.tsv'
    lod_part = out / f'.lod_chr{chrom}.tsv.part'
    peak_tsv = out / f'peak_stats_chr{chrom}.tsv'

    log.info('START chr%s: %d genes  scan=%s  window_mb=%.1f  samples=%d',
             chrom, len(gene_ids), scan_window, window_mb, len(sample_ids))

    if len(gene_ids) == 0:
        log.warning('no genes for chr%s', chrom)
        atomic_write_table(pd.DataFrame(columns=LOD_COLS), lod_tsv)
        atomic_write_table(pd.DataFrame(columns=PEAK_LOD_COLS), peak_tsv)
        atomic_write_json({'chr': chrom, 'n_genes': 0, 'elapsed_s': 0.0}, out / f'timing_chr{chrom}.json')
        return

    meta     = pd.read_csv(meta_path)
    geno_map = build_sample_geno_map(meta)
    geno_ids = [geno_map.get(s, s) for s in sample_ids]

    cov_cols = [c for c in covariates.split(',') if c.strip()]
    cov_arr  = build_cov_array(meta, sample_ids, cov_cols)

    gc             = GenomeCache.from_hdf5(cache_path)
    kinship_decomp = load_kinship_decomp(kinship_path, chroms=[chrom])

    gene_meta = gi[['gene_id', 'gene_symbol']].copy() if 'gene_symbol' in gi.columns else None

    gene_pos_mb_map: 'dict[str, float] | None' = None
    if scan_window == 'local':
        gene_pos_mb_map = {
            gid: tss for gid, tss in build_tss_lookup(gi).items()
            if gid in set(gene_ids)
        }

    expr_cache = {gene_ids[i]: pheno_mat[i] for i in range(len(gene_ids))}

    peak_stats = scan_multi_gene_lmm(
        expr_cache=expr_cache,
        gene_chr_map=gene_chr_map,
        genomecache=gc,
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        sample_ids=geno_ids,
        use_rint=use_rint,
        scan_window=scan_window,
        gene_pos_mb_map=gene_pos_mb_map,
        window_mb=window_mb,
        lod_path=lod_part,
        gene_meta=gene_meta,
    )
    if lod_part.exists():
        lod_part.replace(lod_tsv)
    else:
        atomic_write_table(pd.DataFrame(columns=LOD_COLS), lod_tsv)

    # include beta_{founder} cols alongside PEAK_LOD_COLS so run_finemap.py
    # can enrich egenes_df from peak_lod.tsv without a separate file
    _beta_cols = [c for c in peak_stats.columns if c.startswith('beta_')]
    present = [c for c in PEAK_LOD_COLS if c in peak_stats.columns] + _beta_cols
    atomic_write_table(peak_stats[present], peak_tsv)

    elapsed = time.perf_counter() - t0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    atomic_write_json({
        'chr': chrom, 'n_genes': len(gene_ids),
        'elapsed_s': round(elapsed, 3),
        'peak_heap_mb': round(peak_heap / 1024 ** 2, 1),
        'peak_rss_mb': round(get_rss_mb(), 1),
    }, out / f'timing_chr{chrom}.json')
    log.info('DONE chr%s: %d genes  %.1fs  heap=%.1fMB  rss=%.1fMB',
             chrom, len(gene_ids), elapsed,
             round(peak_heap / 1024 ** 2, 1), round(get_rss_mb(), 1))


if __name__ == '__main__':
    main()
