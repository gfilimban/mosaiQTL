"""Sanger SNP fine-mapping — called by Nextflow FINEMAP process.

Usage: python run_finemap.py <eqtl_summary.csv> <cache.h5> <kinship.h5>
           <phenotypes.h5> <metadata.csv> <celltype>
           <scan_tag> <covariates> <output_dir>
           <sanger_alleles.h5|-> <peak_lod.csv|-> <tissue|-> <chrom|->
           <scan_type> <window_mb> <gene_tss.csv|->

A '-' marks an absent optional argument.

Each eGene's chromosome is rescanned to obtain its 1.5-LOD support interval;
SNPs inside that interval are imputed from founder alleles and fine-mapped
with approximate Bayes factor credible sets.
sanger_alleles.h5 is optional; header-only outputs are written when absent.
Writes:
  {tag}_eqtl_finemapping.csv      — credible-set SNPs per eGene
  {tag}_eqtl_finemapping_peak.csv — credible-set lead SNP per eGene
"""
import sys, logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    egene_csv    = Path(sys.argv[1])
    cache_path   = Path(sys.argv[2])
    kinship_path = Path(sys.argv[3])
    pheno_path   = Path(sys.argv[4])
    meta_path    = Path(sys.argv[5])
    celltype     = sys.argv[6]
    stag         = sys.argv[7]
    covariates   = sys.argv[8]
    out          = Path(sys.argv[9])

    def _opt(i: int) -> 'str | None':
        return None if sys.argv[i] == '-' else sys.argv[i]

    sanger_arg, peak_lod_arg, tissue_arg, chrom_arg = (_opt(i) for i in range(10, 14))
    scan_type = sys.argv[14]
    window_mb = float(sys.argv[15])
    gene_tss_arg = _opt(16)

    sanger_path = Path(sanger_arg) if sanger_arg else None
    peak_lod_path = Path(peak_lod_arg) if peak_lod_arg else None
    gene_tss_path = Path(gene_tss_arg) if gene_tss_arg else None
    tissue = tissue_arg or ''
    chrom_filter = chrom_arg or ''

    tag      = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    chr_sfx  = f'_chr{chrom_filter}' if chrom_filter else ''
    out.mkdir(parents=True, exist_ok=True)
    fm_out      = out / f'{tag}{chr_sfx}_eqtl_finemapping.csv'
    fm_peak_out = out / f'{tag}{chr_sfx}_eqtl_finemapping_peak.csv'

    from mosaiqtl.pipeline import (
        cap_egene_ci, EQTL_FINEMAPPING_COLS, EQTL_FINEMAPPING_PEAK_COLS,
        prepare_egenes_for_finemap,
    )
    from mosaiqtl.utils import strip_chr_prefix

    egenes_df = pd.read_csv(egene_csv, dtype={'gene_id': str, 'chr': str})
    egenes_df['chr'] = strip_chr_prefix(egenes_df['chr'])
    if chrom_filter:
        egenes_df = egenes_df[egenes_df['chr'] == chrom_filter].reset_index(drop=True)
        log.info('chr filter: %s  →  %d eGenes', chrom_filter, len(egenes_df))

    # enrich egenes_df with null model cols + founder betas from peak_lod.csv
    # (peak_lod.csv carries beta_{founder} cols written by run_lmm_scan_chr.py)
    if peak_lod_path is not None and peak_lod_path.exists():
        aux = pd.read_csv(peak_lod_path, dtype={'gene_id': str})
        extra = [c for c in aux.columns if c not in egenes_df.columns and c != 'gene_id']
        if extra:
            egenes_df = egenes_df.merge(aux[['gene_id'] + extra], on='gene_id', how='left')
            log.info('enriched egenes_df from peak_lod.csv: +%d cols', len(extra))

    egenes_df = prepare_egenes_for_finemap(egenes_df)

    log.info('START finemap: tag=%s  %d eGenes  sanger=%s',
             tag, len(egenes_df), sanger_path or 'none')
    if len(egenes_df) == 0 or sanger_path is None:
        log.info('no eGenes or no sanger_alleles.h5 — writing header-only outputs')
        pd.DataFrame(columns=EQTL_FINEMAPPING_COLS).to_csv(fm_out, index=False)
        pd.DataFrame(columns=EQTL_FINEMAPPING_PEAK_COLS).to_csv(fm_peak_out, index=False)
        return

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
    from mosaiqtl.pipeline import finemap_egenes
    from mosaiqtl.qtl.ci import ci_scan_multi_gene

    gc             = GenomeCache.from_hdf5(cache_path)
    kinship_decomp = load_kinship_decomp(kinship_path)

    egene_gene_ids = egenes_df['gene_id'].astype(str).tolist()
    pheno_mat_eg, gene_ids_eg, sample_ids = load_expression_h5(pheno_path, egene_gene_ids)
    pheno_mat_eg = pheno_mat_eg.astype(np.float64)
    gene_idx = {g: i for i, g in enumerate(gene_ids_eg)}

    meta_df  = pd.read_csv(meta_path)
    geno_map = build_sample_geno_map(meta_df)
    geno_ids = [geno_map.get(s, s) for s in sample_ids]

    cov_cols = [c for c in covariates.split(',') if c.strip()]
    cov_arr  = build_cov_array(meta_df, sample_ids, cov_cols)

    expr_cache = {
        gid: pheno_mat_eg[gene_idx[gid]]
        for gid in egene_gene_ids
        if gid in gene_idx
    }

    # ── 1.5-LOD support interval per eGene ────────────────────────────────────
    log.info('support intervals for %d eGenes', len(expr_cache))
    ci_df = ci_scan_multi_gene(
        expr_cache,
        dict(zip(egenes_df['gene_id'], egenes_df['chr'])),
        dict(zip(egenes_df['gene_id'], egenes_df['peak_mb'].astype(float))),
        gc,
        peak_lod_map=dict(zip(egenes_df['gene_id'], egenes_df['peak_lod'].astype(float))),
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        sample_ids=geno_ids,
    )
    del expr_cache
    if ci_df.empty:
        log.warning('no support intervals computed — writing header-only outputs')
        pd.DataFrame(columns=EQTL_FINEMAPPING_COLS).to_csv(fm_out, index=False)
        pd.DataFrame(columns=EQTL_FINEMAPPING_PEAK_COLS).to_csv(fm_peak_out, index=False)
        return

    _ci_mb = ci_df[['gene_id', 'ci_lo', 'ci_hi']].copy()
    _ci_mb['gene_id'] = _ci_mb['gene_id'].astype(str)
    _eg_with_ci = egenes_df.merge(
        _ci_mb.assign(
            ci_lo_bp=(_ci_mb['ci_lo'] * 1e6).round().astype('Int64'),
            ci_hi_bp=(_ci_mb['ci_hi'] * 1e6).round().astype('Int64'),
        )[['gene_id', 'ci_lo_bp', 'ci_hi_bp']],
        on='gene_id', how='left',
    )
    _gtss = (pd.read_csv(gene_tss_path)
             if gene_tss_path is not None and gene_tss_path.exists() else None)
    _eg_capped = cap_egene_ci(_eg_with_ci, scan_type, window_mb, gene_tss_df=_gtss)
    ci_bp = _eg_capped[['gene_id', 'ci_lo_bp', 'ci_hi_bp']].copy()
    if scan_type == 'L' and window_mb > 0:
        log.info(
            'L-mode CI cap ±%.1f Mb applied: %d eGenes  mean CI %.2f Mb',
            window_mb, len(ci_bp),
            float((ci_bp['ci_hi_bp'] - ci_bp['ci_lo_bp']).mean() / 1e6),
        )

    fm_in = egenes_df.assign(
        pos=(egenes_df['peak_mb'] * 1e6).round().astype('Int64'),
        lod=egenes_df['peak_lod'],
        variant=egenes_df['peak_locus'],
    ).merge(ci_bp, on='gene_id', how='left')

    # ── imputed SNP scan + ABF credible sets ──────────────────────────────────
    log.info('Sanger fine-mapping: %d eGenes via %s', len(fm_in), sanger_path)
    _, fm_peak_df = finemap_egenes(
        fm_in,
        pheno_mat_eg,
        gene_ids_eg,
        gc,
        sanger_path,
        out,
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        sample_ids=geno_ids,
    )
    (out / 'eqtl_finemapping.csv').rename(fm_out)
    (out / 'eqtl_finemapping_peak.csv').rename(fm_peak_out)
    log.info('wrote %s + %s (%d eGenes)', fm_out, fm_peak_out, len(fm_peak_df))


if __name__ == '__main__':
    main()
