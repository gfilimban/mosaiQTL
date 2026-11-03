"""eGene calling + FDR — called by Nextflow CALL_EGENES process.

Usage: python run_call_egenes.py <peak_lod.csv> <perm_max_lods.csv>
           <phenotypes.h5> <gene_info.csv> <cache.h5> <kinship.h5>
           <metadata.csv> <celltype> <scan_tag>
           <covariates> <fdr_threshold> <n_perms_cal> <seed>
           <output_dir> [tissue]

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

    tag  = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.kinship import kinship_from_decomp, load_kinship_decomp
    from mosaiqtl.io.anndata_io import load_expression_h5
    from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
    from mosaiqtl.pipeline import (
        EGENE_SUMMARY_COLS, call_egenes, calibrate_egenes_storey, finalize_egene_summary,
    )
    from mosaiqtl.qtl.effects import get_allele_effects
    from mosaiqtl.qtl.fdr import calibrate_gene_pvalue
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

    # ── Stage-1 gate ───────────────────────────────────────────────────────────
    peak_df   = pd.read_csv(peak_lod_csv, dtype={'gene_id': str, 'chr': str})
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

    # calibrate_gene_pvalue operates at strain level; reduce sample phenotype/covariates
    strain_ids_ordered = list(gc.sample_ids)
    Z, valid_mask = build_z_matrix(geno_ids, strain_ids_ordered)
    Z_sub = Z[valid_mask]                              # (n_valid, m)
    rep_counts = Z_sub.sum(axis=0)                     # (m,) — replicates per strain

    def _strain_mean_vec(vec: np.ndarray) -> np.ndarray:
        s = Z_sub.T @ vec[valid_mask]
        return np.where(rep_counts > 0, s / rep_counts, 0.0)

    cov_strain: 'np.ndarray | None' = None
    if cov_arr is not None:
        cov_strain = np.stack(
            [_strain_mean_vec(cov_arr[:, j]) for j in range(cov_arr.shape[1])], axis=1
        )

    # ── allele effects at peak locus ───────────────────────────────────────────
    log.info('allele effects: %d eGenes', len(egenes_df))
    effects_rows: list[dict] = []
    for _eff_i, (_, row) in enumerate(egenes_df.iterrows(), 1):
        gid   = str(row['gene_id'])
        chrom = str(row['chr'])
        locus = str(row.get('peak_locus', ''))
        log.info('  effects [%d/%d] %s chr%s', _eff_i, len(egenes_df), gid, chrom)
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
            use_rint=True,
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
    log.info('Beta calibration: %d eGenes  n_perms_cal=%d', len(egenes_df), n_perms_cal)
    pval_perm: dict[str, float] = {}
    for _cal_i, (_, row) in enumerate(egenes_df.iterrows(), 1):
        gid   = str(row['gene_id'])
        chrom = str(row['chr'])
        log.info('  calibrate [%d/%d] %s chr%s', _cal_i, len(egenes_df), gid, chrom)
        if gid not in gene_idx:
            continue
        try:
            y_strain = _strain_mean_vec(pheno_mat_eg[gene_idx[gid]])
            p_adj, _, _, _ = calibrate_gene_pvalue(
                y_gene=y_strain,
                genomecache=gc,
                chrom=chrom,
                peak_lod=float(row['peak_lod']),
                covariates=cov_strain,
                kinship_decomp=kinship_decomp,
                kinship_loco=kinship_loco,
                n_perms=n_perms_cal,
                seed=seed,
            )
            pval_perm[gid] = float(p_adj)
        except Exception as exc:
            log.warning('calibrate_gene_pvalue failed for %s: %s', gid, exc)

    # ── write pval_perm.json ───────────────────────────────────────────────────
    pval_out.write_text(json.dumps(pval_perm, indent=2))

    if pval_perm:
        egenes_df = calibrate_egenes_storey(egenes_df, pval_perm, fdr_threshold=fdr_threshold)

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
