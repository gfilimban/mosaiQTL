"""eGene calling + FDR — called by Nextflow CALL_EGENES process.

Usage: python run_call_egenes.py <peak_lod.tsv> <perm_max_lods.tsv>
           <phenotypes.h5> <gene_info.csv> <cache.h5> <kinship.h5>
           <metadata.csv> <celltype> <scan_tag>
           <covariates> <fdr_threshold> <n_perms_cal> <seed>
           <output_dir> [tissue] [use_rint:0|1]  (default 1)
           [call_rule:bh|qvalue|lfdr]  (default bh)
           [submission_fdr]  (default 0.1)

use_rint must match the value used during the scan — allele effects are
estimated on the same transformed phenotype so betas remain on the scan scale.
Set to 0 when the scan ran without RINT (Gaussian phenotypes).  NB-GLMM path
does not use this flag.

<perm_max_lods.tsv> is the genome-wide max-LOD null written by PERMUTE; it
supplies the permutation LOD threshold a significant eGene's peak LOD must exceed.
<peak_lod.tsv> is read only; outputs are always new files.
Every gene with a peak is calibrated and enters the q-values; a gene is an eGene when
the call-rule q-value is at most fdr_threshold and its peak LOD exceeds the permutation
threshold. Founder effects are computed for eGenes only.
Writes:
  {tag}_peak_lod_all.tsv — all tested genes with is_egene, pval_perm, qval_storey, qval_bh, lfdr, pi0, fdr
  {tag}_pval_perm.json   — per-gene calibrated p-values {gene_id: float}
  {tag}_haplotype_peak_lod.tsv — the rows of {tag}_peak_lod_all.tsv with qval_bh <= submission_fdr
  {tag}_haplotype_eqtl_summary.tsv — significant eGenes only, with founder effects + q-value;
    the support-interval columns are added by FINEMAP_MERGE
All four are written (empty / header-only) when no gene has a peak.
"""
import sys, logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    peak_lod_tsv   = Path(sys.argv[1])
    perm_tsv       = Path(sys.argv[2])
    pheno_path     = Path(sys.argv[3])
    gene_info_path = Path(sys.argv[4])
    cache_path     = Path(sys.argv[5])
    kinship_path   = Path(sys.argv[6])
    meta_path      = Path(sys.argv[7])
    celltype       = sys.argv[8]
    stag           = sys.argv[9]
    covariates     = sys.argv[10]
    fdr_threshold  = float(sys.argv[11])
    n_perms_cal    = int(sys.argv[12])
    seed           = int(sys.argv[13])
    out            = Path(sys.argv[14])
    tissue         = sys.argv[15] if len(sys.argv) > 15 else ''
    use_rint       = sys.argv[16] != '0' if len(sys.argv) > 16 else True
    call_rule      = sys.argv[17] if len(sys.argv) > 17 else 'bh'
    submission_fdr = float(sys.argv[18]) if len(sys.argv) > 18 else 0.1

    tag  = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import kinship_from_decomp, load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.atomic import atomic_write_json, atomic_write_table
    from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
    from mosaiqtl.io.tables import read_table
    from mosaiqtl.pipeline import (
        EGENE_SUMMARY_COLS, call_significant_egenes, finalize_egene_summary,
        select_submission_genes,
    )
    from mosaiqtl.qtl.thresholds import get_gev_thresholds
    from mosaiqtl.utils import add_chr_prefix, build_z_matrix, strip_chr_prefix

    peak_out   = out / f'{tag}_peak_lod_all.tsv'
    pval_out    = out / f'{tag}_pval_perm.json'
    summary_out = out / f'{tag}_haplotype_eqtl_summary.tsv'
    submission_out = out / f'{tag}_haplotype_peak_lod.tsv'
    if peak_out.resolve() == peak_lod_tsv.resolve():
        raise ValueError(
            f'output {peak_out} would overwrite input {peak_lod_tsv}; '
            'stage the input peak_lod under a different name'
        )

    log.info('START call_egenes: tag=%s  fdr=%.2f  n_perms_cal=%d  seed=%d',
             tag, fdr_threshold, n_perms_cal, seed)

    # ── threshold ──────────────────────────────────────────────────────────────
    max_lods = read_table(perm_tsv)['max_lod'].to_numpy()
    threshold, _ = get_gev_thresholds(max_lods)
    if threshold is None:
        threshold = float(np.quantile(max_lods[np.isfinite(max_lods)], 0.95))
        log.warning('GEV fit failed — empirical 95th: %.3f', threshold)
    log.info('reference genome-wide LOD threshold: %.3f', threshold)

    # ── gene info (symbol map + TSS for calibration window) ───────────────────
    gene_info_df = pd.read_csv(gene_info_path, dtype={'gene_id': str})
    if 'gene' in gene_info_df.columns and 'gene_id' not in gene_info_df.columns:
        gene_info_df = gene_info_df.rename(columns={'gene': 'gene_id'})

    # ── tested genes: every gene with a peak enters calibration ───────────────
    peak_df   = read_table(peak_lod_tsv, dtype={'gene_id': str, 'chr': str})
    peak_df['chr'] = strip_chr_prefix(peak_df['chr'])
    tested_df = peak_df.dropna(subset=['gene_id', 'chr', 'peak_lod'])
    log.info('%d / %d genes tested', len(tested_df), len(peak_df))
    if len(tested_df) == 0:
        empty_peak = peak_df.assign(is_egene=False)
        empty_peak['chr'] = add_chr_prefix(empty_peak['chr'])
        atomic_write_table(empty_peak, peak_out)
        atomic_write_table(empty_peak.iloc[0:0], submission_out)
        atomic_write_json({}, pval_out)
        atomic_write_table(pd.DataFrame(columns=EGENE_SUMMARY_COLS), summary_out)
        log.info('no tested genes — wrote empty %s outputs', tag)
        return

    # ── shared objects ─────────────────────────────────────────────────────────
    gc             = GenomeCache.from_hdf5(cache_path)
    kinship_decomp = load_kinship_decomp(kinship_path)
    kinship_loco   = {c: kinship_from_decomp(d) for c, d in kinship_decomp.items()}

    tested_ids = tested_df['gene_id'].astype(str).tolist()
    pheno_mat, gene_ids_loaded, sample_ids = load_expression_h5(pheno_path, tested_ids)
    pheno_mat = pheno_mat.astype(np.float64)
    gene_idx = {g: i for i, g in enumerate(gene_ids_loaded)}

    meta_df  = pd.read_csv(meta_path)
    geno_map = build_sample_geno_map(meta_df)
    geno_ids = [geno_map.get(s, s) for s in sample_ids]

    cov_cols = [c for c in covariates.split(',') if c.strip()]
    cov_arr  = build_cov_array(meta_df, sample_ids, cov_cols)

    strain_ids_ordered = list(gc.sample_ids)
    Z, valid_mask = build_z_matrix(geno_ids, strain_ids_ordered)

    # ── calibrate all tested genes → Storey q-value → effects for significant ─
    log.info('calibration: %d tested genes  n_perms_cal=%d  stag=%s',
             len(tested_df), n_perms_cal, stag)
    called_df = call_significant_egenes(
        tested_df,
        genomecache=gc,
        pheno_mat=pheno_mat,
        gene_idx=gene_idx,
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        kinship_loco=kinship_loco,
        sample_ids=geno_ids,
        stag=stag,
        gene_info_df=gene_info_df,
        fdr_threshold=fdr_threshold,
        n_perms=n_perms_cal,
        use_rint=use_rint,
        z_mat=Z,
        valid_mask=valid_mask,
        seed=seed,
        call_rule=call_rule,
        lod_threshold=threshold,
    )
    significant_df = called_df[called_df['significance'] == 'T']

    # ── write pval_perm.json (every tested gene) ──────────────────────────────
    pval_perm = dict(zip(called_df['gene_id'].astype(str),
                         called_df['pval_perm'].astype(float)))
    atomic_write_json(pval_perm, pval_out)

    # ── annotate full peak_lod with is_egene flag → updated peak_lod.tsv ─────
    significant_ids = set(significant_df['gene_id'].astype(str))
    peak_df = peak_df.copy()
    peak_df['is_egene'] = peak_df['gene_id'].astype(str).isin(significant_ids)
    adj_cols = [c for c in ('pval_perm', 'beta_a', 'beta_b', 'perm_n', 'qval_storey', 'qval_bh', 'lfdr', 'pi0', 'fdr', 'fdr_method')
                if c in called_df.columns]
    adj_df = called_df[['gene_id', *adj_cols]].assign(
        gene_id=lambda d: d['gene_id'].astype(str)
    ).drop_duplicates('gene_id')
    peak_df['gene_id'] = peak_df['gene_id'].astype(str)
    peak_df = peak_df.drop(columns=[c for c in adj_cols if c in peak_df.columns]).merge(
        adj_df, on='gene_id', how='left'
    )
    peak_out_df = peak_df.copy()
    peak_out_df['chr'] = add_chr_prefix(peak_out_df['chr'])
    atomic_write_table(peak_out_df, peak_out)
    log.info('wrote %s (%d genes, %d eGenes)', peak_out, len(peak_df),
             peak_df['is_egene'].sum())

    submission_df = select_submission_genes(peak_out_df, submission_fdr)
    atomic_write_table(submission_df, submission_out)
    log.info('wrote %s (%d of %d tested genes with qval_bh <= %g)', submission_out,
             len(submission_df), len(peak_out_df), submission_fdr)

    final_df = finalize_egene_summary(significant_df, n_founders=len(gc.founders))
    summary_df = final_df.copy()
    if 'chr' in summary_df.columns:
        summary_df['chr'] = add_chr_prefix(summary_df['chr'])
    atomic_write_table(summary_df, summary_out)
    log.info('wrote %s (%d eGenes)', summary_out, len(final_df))


if __name__ == '__main__':
    main()
