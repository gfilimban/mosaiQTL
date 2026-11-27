"""Permutation threshold scan — called by Nextflow PERMUTE process.

Usage: python run_permute.py <cache.h5> <kinship.h5> <phenotypes.h5>
           <peak_lod.tsv> <metadata.csv> <celltype>
           <scan_tag> <covariates>
           <n_perms> <seed> <perm_method> <output_dir>
           <scan_type> <window_mb> <tissue> <use_rint 1|0>

<scan_type>: L (local ±window_mb), C (cis-chr), or G (genome-wide).
  L: local null — representative gene's chr + ±window_mb around its TSS (start_mb).
     Calibrated to the actual number of loci each gene competes against; correctly
     lower threshold than genome-wide.  Matches validate_tissue.py S5 L-mode logic.
  C/G: genome-wide null — all chromosomes, no window restriction.
<window_mb>: half-window in Mb around TSS; required for scan_type=L; pass 0 otherwise.

Representative gene is selected by pick_rep_gene() from mosaiqtl.qtl.permutation:
  median h² across all genes (or chromosome-restricted for L mode) from peak_lod.tsv.
  h² is the LMM heritability written to peak_lod.tsv by scan_multi_gene; using the
  median-h² gene keeps the null LOD distribution representative of the gene population.

Writes: {tag}_perm_max_lods.tsv, {tag}_perm_max_lods_chr{c}.tsv, {tag}_perm_gev_params.json
"""
import sys, logging, time
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    (cache_arg, kinship_arg, pheno_arg, peak_lod_arg, meta_arg, celltype, stag, covariates,
     n_perms_arg, seed_arg, perm_method, out_arg, scan_type, window_arg, tissue,
     rint_arg) = sys.argv[1:]

    cache_path, kinship_path, pheno_path, peak_lod_tsv, meta_path, out = (
        Path(a) for a in (cache_arg, kinship_arg, pheno_arg, peak_lod_arg, meta_arg, out_arg)
    )
    n_perms, seed = int(n_perms_arg), int(seed_arg)
    window_mb, use_rint = float(window_arg), rint_arg != '0'

    tag  = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.atomic import atomic_write_json, atomic_write_table
    from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
    from mosaiqtl.io.tables import read_table
    from mosaiqtl.qtl.permutation import generate_null_outcomes, pick_rep_gene, run_threshold_scans
    from mosaiqtl.qtl.thresholds import get_gev_thresholds
    from mosaiqtl.utils import build_z_matrix, strip_chr_prefix

    log.info('START permute: tag=%s  n_perms=%d  seed=%d  method=%s  scan_type=%s',
             tag, n_perms, seed, perm_method, scan_type)
    t0 = time.perf_counter()

    peak_df     = read_table(peak_lod_tsv, dtype={'chr': str})
    peak_df['chr'] = strip_chr_prefix(peak_df['chr'])
    rep_id, rep_chr, rep_tss_mb, h2_med = pick_rep_gene(peak_df, chrom=None)

    perm_chrom: 'str | None' = None
    perm_tss_mb: 'float | None' = None
    if scan_type == 'L' and window_mb > 0:
        perm_chrom  = rep_chr
        perm_tss_mb = rep_tss_mb
        log.info('L-mode local null: chr%s  TSS=%.3fMb  ±%.1fMb',
                 perm_chrom, perm_tss_mb, window_mb)
    elif scan_type == 'C':
        perm_chrom = rep_chr
        log.info('C-mode chr-wide null: chr%s', perm_chrom)
    elif scan_type == 'G':
        log.info('G-mode genome-wide null')
    else:
        raise ValueError(
            f"scan_type must be 'C', 'G', or 'L' with window_mb > 0; "
            f'got scan_type={scan_type!r} window_mb={window_mb}'
        )

    counts, _, sample_ids = load_expression_h5(pheno_path, [rep_id])
    y = counts[0].astype(np.float64)

    gc             = GenomeCache.from_hdf5(cache_path)
    kinship_decomp = load_kinship_decomp(kinship_path)

    meta_df  = pd.read_csv(meta_path)
    geno_map = build_sample_geno_map(meta_df)
    geno_ids = [geno_map.get(s, s) for s in sample_ids]

    cov_cols = [c for c in covariates.split(',') if c.strip()]
    cov_arr  = build_cov_array(meta_df, sample_ids, cov_cols)

    Z, _ = build_z_matrix(geno_ids, list(gc.sample_ids))

    log.info('generating %d null outcomes …', n_perms)
    null_outcomes = generate_null_outcomes(y, n_perms=n_perms, seed=seed)
    log.info('running threshold scans …')
    max_lods, chr_null_lods = run_threshold_scans(
        null_outcomes, gc,
        phenotype=y,
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        permutation_method=perm_method,
        z_mat=Z,
        ref_chrom=rep_chr,
        chrom=perm_chrom,
        gene_pos_mb=perm_tss_mb,
        window_mb=window_mb if perm_chrom is not None else None,
        seed=seed,
        use_rint=use_rint,
    )

    atomic_write_table(pd.DataFrame({'max_lod': max_lods}), out / f'{tag}_perm_max_lods.tsv')

    gev_params: dict = {}
    for chrom, lods in sorted(chr_null_lods.items()):
        atomic_write_table(pd.DataFrame({'max_lod': lods}), out / f'{tag}_perm_max_lods_chr{chrom}.tsv')
        threshold, params = get_gev_thresholds(np.array(lods))
        if params is not None:
            gev_params[chrom] = params

    atomic_write_json(gev_params, out / f'{tag}_perm_gev_params.json')
    log.info('DONE %s: %d perms  %d chrs  %.1fs', tag, n_perms, len(chr_null_lods), time.perf_counter() - t0)


if __name__ == '__main__':
    main()
