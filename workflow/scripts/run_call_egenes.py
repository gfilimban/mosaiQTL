"""eGene calling + FDR — called by Nextflow CALL_EGENES process.

Usage: python run_call_egenes.py <peak_lod.csv> <perm_max_lods.csv>
           <phenotypes.h5> <gene_info.csv> <cache.h5> <kinship.h5>
           <metadata.csv> <celltype> <scan_tag>
           <covariates> <fdr_threshold> <n_perms_cal> <seed>
           <output_dir> [tissue] [use_rint:0|1]  (default 1)

use_rint must match the value used during the scan — allele effects are
estimated on the same transformed phenotype so betas remain on the scan scale.
Set to 0 when the scan ran without RINT (Gaussian phenotypes).  NB-GLMM path
does not use this flag.

<perm_max_lods.csv> is the genome-wide max-LOD null written by PERMUTE.
<peak_lod.csv> is read only; outputs are always new files.
Writes:
  {tag}_peak_lod.csv     — all genes with is_egene flag added
  {tag}_pval_perm.json   — per-gene calibrated p-values {gene_id: float}
  {tag}_eqtl_summary.csv — eGenes only, with founder effects + Storey q-value
All three are written (empty / header-only) when no gene clears Stage-1.
"""
import sys, json, logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    peak_lod_csv   = Path(sys.argv[1])
    perm_csv       = Path(sys.argv[2])
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

    tag  = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import kinship_from_decomp, load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
    from mosaiqtl.pipeline import (
        EGENE_SUMMARY_COLS, normalize_gene_ids, call_egenes,
        calibrate_egenes_storey, finalize_egene_summary,
    )
    from mosaiqtl.qtl.effects import get_allele_effects
    from mosaiqtl.qtl.thresholds import get_gev_thresholds
    from mosaiqtl.utils import build_z_matrix

    peak_out    = out / f'{tag}_peak_lod.csv'
    pval_out    = out / f'{tag}_pval_perm.json'
    summary_out = out / f'{tag}_eqtl_summary.csv'
    if peak_out.resolve() == peak_lod_csv.resolve():
        raise ValueError(
            f'output {peak_out} would overwrite input {peak_lod_csv}; '
            'stage the input peak_lod under a different name'
        )

    log.info('START call_egenes: tag=%s  fdr=%.2f  n_perms_cal=%d  seed=%d',
             tag, fdr_threshold, n_perms_cal, seed)

    # ── threshold ──────────────────────────────────────────────────────────────
    max_lods = pd.read_csv(perm_csv)['max_lod'].to_numpy()
    threshold, _ = get_gev_thresholds(max_lods)
    if threshold is None:
        threshold = float(np.quantile(max_lods[np.isfinite(max_lods)], 0.95))
        log.warning('GEV fit failed — empirical 95th: %.3f', threshold)
    log.info('LOD threshold: %.3f', threshold)

    # ── gene info (symbol map + TSS for calibration window) ───────────────────
    gene_info_df = pd.read_csv(gene_info_path, dtype={'gene_id': str})
    if 'gene' in gene_info_df.columns and 'gene_id' not in gene_info_df.columns:
        gene_info_df = gene_info_df.rename(columns={'gene': 'gene_id'})

    # ── Stage-1 gate ───────────────────────────────────────────────────────────
    peak_df   = pd.read_csv(peak_lod_csv, dtype={'gene_id': str, 'chr': str})
    peak_df   = normalize_gene_ids(peak_df)
    egenes_df = call_egenes(peak_df, threshold)
    log.info('Stage-1: %d / %d eGenes', len(egenes_df), len(peak_df))
    if len(egenes_df) == 0:
        peak_df.assign(is_egene=False).to_csv(peak_out, index=False)
        pval_out.write_text(json.dumps({}))
        pd.DataFrame(columns=EGENE_SUMMARY_COLS).to_csv(summary_out, index=False)
        log.info('no eGenes — wrote empty %s outputs', tag)
        return

    # ── shared objects ─────────────────────────────────────────────────────────
    gc             = GenomeCache.from_hdf5(cache_path)
    kinship_decomp = load_kinship_decomp(kinship_path)
    kinship_loco   = {c: kinship_from_decomp(d) for c, d in kinship_decomp.items()}

    egene_ids = egenes_df['gene_id'].astype(str).tolist()
    pheno_mat_eg, gene_ids_eg, sample_ids = load_expression_h5(pheno_path, egene_ids)
    pheno_mat_eg = pheno_mat_eg.astype(np.float64)
    gene_idx = {g: i for i, g in enumerate(gene_ids_eg)}

    meta_df  = pd.read_csv(meta_path)
    geno_map = build_sample_geno_map(meta_df)
    geno_ids = [geno_map.get(s, s) for s in sample_ids]

    cov_cols = [c for c in covariates.split(',') if c.strip()]
    cov_arr  = build_cov_array(meta_df, sample_ids, cov_cols)

    strain_ids_ordered = list(gc.sample_ids)
    Z, valid_mask = build_z_matrix(geno_ids, strain_ids_ordered)

    sym_lookup: dict[str, str] = {}
    if 'gene_symbol' in gene_info_df.columns:
        sym_lookup = dict(zip(gene_info_df['gene_id'].astype(str),
                              gene_info_df['gene_symbol'].astype(str)))

    # ── allele effects at peak locus ───────────────────────────────────────────
    log.info('allele effects: %d eGenes', len(egenes_df))
    effects_rows: list[dict] = []
    for _eff_i, (_, row) in enumerate(egenes_df.iterrows(), 1):
        gid   = str(row['gene_id'])
        chrom = str(row['chr'])
        locus = str(row.get('peak_locus', ''))
        sym   = sym_lookup.get(gid, str(row.get('gene_symbol', gid)))
        log.info('  effects [%d/%d] %s chr%s', _eff_i, len(egenes_df), sym, chrom)
        if gid not in gene_idx or not locus:
            continue
        fxe = get_allele_effects(
            phenotype=pheno_mat_eg[gene_idx[gid]],
            genomecache=gc,
            locus_id=locus,
            chrom=chrom,
            kinship_decomp=kinship_decomp,
            covariates=cov_arr,
            sample_ids=geno_ids,
            use_rint=use_rint,
        )
        erow: dict = {'gene_id': gid, 'gene_symbol': row.get('gene_symbol', '')}
        for fe in fxe:
            erow[f'beta_{fe.founder}'] = fe.estimate
        effects_rows.append(erow)

    if effects_rows:
        eff_df    = pd.DataFrame(effects_rows)
        merge_cols = [c for c in eff_df.columns if c not in egenes_df.columns or c == 'gene_id']
        egenes_df = egenes_df.merge(eff_df[merge_cols], on='gene_id', how='left')

    # ── per-gene Beta calibration → Storey FDR ────────────────────────────────
    log.info('Beta calibration: %d eGenes  n_perms_cal=%d  stag=%s',
             len(egenes_df), n_perms_cal, stag)
    egenes_df = calibrate_egenes_storey(
        egenes_df,
        fdr_threshold=fdr_threshold,
        genomecache=gc,
        pheno_mat=pheno_mat_eg,
        gene_idx=gene_idx,
        covariates=cov_arr,
        kinship_decomp=kinship_decomp,
        kinship_loco=kinship_loco,
        stag=stag,
        gene_info_df=gene_info_df,
        n_perms=n_perms_cal,
        use_rint=use_rint,
        z_mat=Z,
        valid_mask=valid_mask,
        seed=seed,
    )

    # ── write pval_perm.json ───────────────────────────────────────────────────
    pval_perm = dict(zip(egenes_df['gene_id'].astype(str),
                         egenes_df['pval_perm'].astype(float)))
    pval_out.write_text(json.dumps(pval_perm, indent=2))

    # ── annotate full peak_lod with is_egene flag → updated peak_lod.csv ─────
    egene_ids = set(egenes_df['gene_id'].astype(str))
    peak_df   = peak_df.copy()
    peak_df['is_egene'] = peak_df['gene_id'].astype(str).isin(egene_ids)
    peak_df.to_csv(peak_out, index=False)
    log.info('wrote %s (%d genes, %d eGenes)', peak_out, len(peak_df), peak_df['is_egene'].sum())

    final_df = finalize_egene_summary(egenes_df)
    final_df.to_csv(summary_out, index=False)
    log.info('wrote %s (%d eGenes)', summary_out, len(final_df))


if __name__ == '__main__':
    main()
