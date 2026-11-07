"""Permutation threshold scan — called by Nextflow PERMUTE process.

Usage: python run_permute.py <cache.h5> <kinship.h5> <phenotypes.h5>
           <peak_lod.csv> <metadata.csv> <celltype>
           <scan_tag> <covariates>
           <n_perms> <seed> <perm_method> <output_dir>
           <scan_type> <window_mb> [tissue]

<scan_type>: L (local ±window_mb), C (cis-chr), or G (genome-wide).
  L: local null — representative gene's chr + ±window_mb around its TSS (start_mb).
     Calibrated to the actual number of loci each gene competes against; correctly
     lower threshold than genome-wide.  Matches validate_tissue.py S5 L-mode logic.
  C/G: genome-wide null — all chromosomes, no window restriction.
<window_mb>: half-window in Mb around TSS; required for scan_type=L; pass 0 otherwise.

Representative gene is selected by pick_rep_gene() from mosaiqtl.qtl.permutation:
  median h² across all genes (or chromosome-restricted for L mode) from peak_lod.csv.
  h² is the LMM heritability written to peak_lod.csv by scan_multi_gene; using the
  median-h² gene keeps the null LOD distribution representative of the gene population.

Writes: {tag}_perm_max_lods.csv, {tag}_perm_max_lods_chr{c}.csv, {tag}_perm_gev_params.json
"""
import sys, json, logging, time
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    cache_path   = Path(sys.argv[1])
    kinship_path = Path(sys.argv[2])
    pheno_path   = Path(sys.argv[3])
    peak_lod_csv = Path(sys.argv[4])
    meta_path    = Path(sys.argv[5])
    celltype     = sys.argv[6]
    stag         = sys.argv[7]
    covariates   = sys.argv[8]
    n_perms      = int(sys.argv[9])
    seed         = int(sys.argv[10])
    perm_method  = sys.argv[11]
    out          = Path(sys.argv[12])
    scan_type      = sys.argv[13] if len(sys.argv) > 13 else 'G'
    window_mb      = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0
    tissue         = sys.argv[15] if len(sys.argv) > 15 else ''
    gene_tss_path  = Path(sys.argv[16]) if len(sys.argv) > 16 else None

    tag  = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.covariates import build_covariate_matrix, build_sample_geno_map
    from mosaiqtl.qtl.permutation import generate_null_outcomes, pick_rep_gene, run_threshold_scans
    from mosaiqtl.qtl.thresholds import get_gev_thresholds

    log.info('START permute: tag=%s  n_perms=%d  seed=%d  method=%s  scan_type=%s',
             tag, n_perms, seed, perm_method, scan_type)
    t0 = time.perf_counter()

    peak_df     = pd.read_csv(peak_lod_csv, dtype={'chr': str})
    gene_tss_df = pd.read_csv(gene_tss_path, dtype={'chr': str}) if gene_tss_path else None

    rep_id, rep_chr, rep_tss_mb, h2_med = pick_rep_gene(
        peak_df, chrom=None, gene_tss_df=gene_tss_df,
    )

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
    else:
        log.info('G-mode genome-wide null')

    counts, _, sample_ids = load_expression_h5(pheno_path, [rep_id])
    y = counts[0].astype(np.float64)

    gc             = GenomeCache.from_hdf5(cache_path)
    kinship_decomp = load_kinship_decomp(kinship_path)

    meta_df  = pd.read_csv(meta_path)
    geno_map = build_sample_geno_map(meta_df)
    geno_ids = [geno_map.get(s, s) for s in sample_ids]

    cov_arr: 'np.ndarray | None' = None
    cov_cols = [c for c in covariates.split(',') if c.strip()]
    if cov_cols:
        meta_scan   = meta_df[meta_df['sample_id'].isin(sample_ids)].drop_duplicates('sample_id')
        cov_df      = build_covariate_matrix(meta_scan, covariates=cov_cols)
        cov_aligned = cov_df.reindex(sample_ids)
        missing     = cov_aligned.index[cov_aligned.isna().any(axis=1)].tolist()
        if missing:
            raise ValueError(
                f'covariates {cov_cols} missing for {len(missing)} sample(s): {missing}'
            )
        cov_arr = cov_aligned.values

    log.info('generating %d null outcomes …', n_perms)
    null_outcomes = generate_null_outcomes(y, n_perms=n_perms, seed=seed)
    log.info('running threshold scans …')
    max_lods, chr_null_lods = run_threshold_scans(
        null_outcomes, gc,
        phenotype=y,
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        permutation_method=perm_method,
        sample_ids=geno_ids,
        chrom=perm_chrom,
        gene_pos_mb=perm_tss_mb,
        window_mb=window_mb if perm_chrom is not None else None,
        seed=seed,
    )

    pd.DataFrame({'max_lod': max_lods}).to_csv(out / f'{tag}_perm_max_lods.csv', index=False)

    gev_params: dict = {}
    for chrom, lods in sorted(chr_null_lods.items()):
        pd.DataFrame({'max_lod': lods}).to_csv(out / f'{tag}_perm_max_lods_chr{chrom}.csv', index=False)
        threshold, params = get_gev_thresholds(np.array(lods))
        if params is not None:
            gev_params[chrom] = params

    (out / f'{tag}_perm_gev_params.json').write_text(json.dumps(gev_params, indent=2))
    log.info('DONE %s: %d perms  %d chrs  %.1fs', tag, n_perms, len(chr_null_lods), time.perf_counter() - t0)


if __name__ == '__main__':
    main()
