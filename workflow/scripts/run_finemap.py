"""Sanger SNP fine-mapping — called by Nextflow FINEMAP process.

Usage: python run_finemap.py <eqtl_summary.tsv> <cache.h5> <kinship.h5>
           <phenotypes.h5> <metadata.csv> <celltype>
           <scan_tag> <covariates> <output_dir>
           <sanger_alleles.h5|-> <peak_lod.tsv|-> <tissue|-> <chrom|->
           <scan_type> <window_mb> <use_rint 1|0>

A '-' marks an absent optional argument.

Each eGene's scan window (±window_mb of its strand-aware TSS, the eGene table's start_mb)
is rescanned to obtain its contiguous 1.5-LOD support interval;
SNPs inside that interval are imputed from founder alleles and fine-mapped
with approximate Bayes factor credible sets.
sanger_alleles.h5 is optional; header-only fine-mapping outputs are written when absent.
Writes:
  {tag}_eqtl_finemapping.tsv      — credible-set SNPs per eGene
  {tag}_eqtl_finemapping_peak.tsv — credible-set lead SNP per eGene
  {tag}_eqtl_finemapping_groups.tsv — one row per LD group per eGene (group PIP, credible-set flag)
  {tag}_eqtl_ci.tsv              — 1.5-LOD support interval per eGene (Mb, with ci_truncated);
                                    written whether or not sanger_alleles.h5 is present
"""
import sys, logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    (egene_arg, cache_arg, kinship_arg, pheno_arg, meta_arg, celltype, stag, covariates, out_arg,
     sanger_arg, peak_lod_arg, tissue_arg, chrom_arg, scan_type, window_arg, rint_arg) = sys.argv[1:]

    egene_tsv, cache_path, kinship_path, pheno_path, meta_path, out = (
        Path(a) for a in (egene_arg, cache_arg, kinship_arg, pheno_arg, meta_arg, out_arg)
    )
    sanger_path, peak_lod_path = (
        None if a == '-' else Path(a) for a in (sanger_arg, peak_lod_arg)
    )
    tissue = '' if tissue_arg == '-' else tissue_arg
    chrom_filter = '' if chrom_arg == '-' else chrom_arg
    window_mb = float(window_arg)
    use_rint = rint_arg != '0'

    tag      = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    chr_sfx  = f'_chr{chrom_filter}' if chrom_filter else ''
    out.mkdir(parents=True, exist_ok=True)
    fm_out      = out / f'{tag}{chr_sfx}_eqtl_finemapping.tsv'
    fm_peak_out = out / f'{tag}{chr_sfx}_eqtl_finemapping_peak.tsv'
    fm_group_out = out / f'{tag}{chr_sfx}_eqtl_finemapping_groups.tsv'
    ci_out     = out / f'{tag}{chr_sfx}_eqtl_ci.tsv'

    from mosaiqtl.io.atomic import atomic_write_table
    from mosaiqtl.io.tables import read_table
    from mosaiqtl.pipeline import (
        CI_COLS, EQTL_FINEMAPPING_COLS, EQTL_FINEMAPPING_GROUP_COLS,
        EQTL_FINEMAPPING_PEAK_COLS, prepare_egenes_for_finemap,
    )
    from mosaiqtl.utils import add_chr_prefix, strip_chr_prefix

    egenes_df = read_table(egene_tsv, dtype={'gene_id': str, 'chr': str})
    egenes_df['chr'] = strip_chr_prefix(egenes_df['chr'])
    if chrom_filter:
        egenes_df = egenes_df[egenes_df['chr'] == chrom_filter].reset_index(drop=True)
        log.info('chr filter: %s  →  %d eGenes', chrom_filter, len(egenes_df))

    # enrich egenes_df with null model cols + founder betas from peak_lod.tsv
    # (peak_lod.tsv carries beta_{founder} cols written by run_lmm_scan_chr.py)
    if peak_lod_path is not None and peak_lod_path.exists():
        aux = read_table(peak_lod_path, dtype={'gene_id': str})
        extra = [c for c in aux.columns if c not in egenes_df.columns and c != 'gene_id']
        if extra:
            egenes_df = egenes_df.merge(aux[['gene_id'] + extra], on='gene_id', how='left')
            log.info('enriched egenes_df from peak_lod.tsv: +%d cols', len(extra))

    egenes_df = prepare_egenes_for_finemap(egenes_df)

    log.info('START finemap: tag=%s  %d eGenes  sanger=%s',
             tag, len(egenes_df), sanger_path or 'none')
    if len(egenes_df) == 0:
        log.info('no eGenes on this chromosome — writing header-only outputs')
        atomic_write_table(pd.DataFrame(columns=CI_COLS), ci_out)
        atomic_write_table(pd.DataFrame(columns=EQTL_FINEMAPPING_COLS), fm_out)
        atomic_write_table(pd.DataFrame(columns=EQTL_FINEMAPPING_PEAK_COLS), fm_peak_out)
        atomic_write_table(pd.DataFrame(columns=EQTL_FINEMAPPING_GROUP_COLS), fm_group_out)
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
    windowed = scan_type == 'L' and window_mb > 0
    if windowed and 'start_mb' not in egenes_df.columns:
        raise ValueError('L-mode fine-mapping needs start_mb (strand-aware TSS) in the eGene table')
    ci_df = ci_scan_multi_gene(
        expr_cache,
        dict(zip(egenes_df['gene_id'], egenes_df['chr'])),
        dict(zip(egenes_df['gene_id'], egenes_df['peak_mb'].astype(float))),
        gc,
        peak_lod_map=dict(zip(egenes_df['gene_id'], egenes_df['peak_lod'].astype(float))),
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        sample_ids=geno_ids,
        use_rint=use_rint,
        gene_pos_mb_map=(
            egenes_df.dropna(subset=['start_mb']).set_index('gene_id')['start_mb'].astype(float).to_dict()
            if windowed else None
        ),
        window_mb=window_mb if windowed else None,
        gene_meta=egenes_df[['gene_id', 'gene_symbol']],
    )
    del expr_cache
    atomic_write_table(ci_df.assign(chr=add_chr_prefix(ci_df['chr'].astype(str)))[CI_COLS], ci_out)
    if ci_df.empty or sanger_path is None:
        log.warning('no support intervals or no sanger_alleles.h5 — writing header-only fine-mapping outputs')
        atomic_write_table(pd.DataFrame(columns=EQTL_FINEMAPPING_COLS), fm_out)
        atomic_write_table(pd.DataFrame(columns=EQTL_FINEMAPPING_PEAK_COLS), fm_peak_out)
        atomic_write_table(pd.DataFrame(columns=EQTL_FINEMAPPING_GROUP_COLS), fm_group_out)
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
    ci_bp = _eg_with_ci[['gene_id', 'ci_lo_bp', 'ci_hi_bp']].copy()
    if windowed:
        log.info(
            'L-mode support intervals inside ±%.1f Mb of the TSS: %d eGenes  mean CI %.2f Mb  truncated by window %d',
            window_mb, len(ci_bp),
            float((ci_bp['ci_hi_bp'] - ci_bp['ci_lo_bp']).mean() / 1e6),
            int(ci_df['ci_truncated'].sum()),
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
        use_rint=use_rint,
    )
    (out / 'eqtl_finemapping.tsv').rename(fm_out)
    (out / 'eqtl_finemapping_peak.tsv').rename(fm_peak_out)
    (out / 'eqtl_finemapping_groups.tsv').rename(fm_group_out)
    log.info('wrote %s + %s + %s (%d eGenes)', fm_out, fm_peak_out, fm_group_out, len(fm_peak_df))


if __name__ == '__main__':
    main()
