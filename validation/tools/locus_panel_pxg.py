"""locus_panel_pxg.py — produce locus panels + standalone PxG figures for a gene list.

Edit GENES_OF_INTEREST, CELLTYPE, and SCAN_TAG at the top, then run:
    conda run -n bioinfo python validation/tools/locus_panel_pxg.py

Outputs per gene under validation/ctxhipp_data/locus_panels/<celltype>_<tag>/:
  locus_<gene>_chr<chr>.png   — IGV-style locus panel (LOD + zoom + effects)
  pxg_<gene>_chr<chr>.png     — standalone phenotype-by-genotype figure
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.founders import FOUNDER_ORDER, F1_COLORS
from mosaiqtl.viz.locus import locus_panel, locus_panels_per_chr
from mosaiqtl.viz.haplotypes import pxg_figure

# ── USER CONFIG ───────────────────────────────────────────────────────────────
CELLTYPE   = 'glutamatergic_neuron'
SCAN_TAG   = 'Lw1'
GENES_OF_INTEREST = [
    'Neurl1a',
    'Kcnq5',
    # add more gene symbols here
]

NOVA_DIR  = pathlib.Path('validation/ctxhipp_data/nova/celltypes')
CACHE_H5  = pathlib.Path('data/cache.h5')
OUT_DIR   = pathlib.Path(f'validation/ctxhipp_data/locus_panels/{CELLTYPE}_{SCAN_TAG}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD SHARED DATA ──────────────────────────────────────────────────────────
print('loading genome cache …')
gc = GenomeCache.from_hdf5(str(CACHE_H5))
cache_strains = list(gc.sample_ids)   # canonical strain order in the cache

ct_dir   = NOVA_DIR / CELLTYPE
# full per-locus LOD scan (gene_id, gene_symbol, chr, locus, pos_mb, lod)
lod_f    = ct_dir / SCAN_TAG / 'core' / f'lod_{CELLTYPE}_{SCAN_TAG}.csv'
# per-founder betas at peak locus
eff_f    = ct_dir / SCAN_TAG / 'core' / f'allele_effects_{CELLTYPE}_{SCAN_TAG}.csv'
# significant eGenes (includes start_mb / near_tss_mb)
egene_f  = ct_dir / SCAN_TAG / 'eqtl_summary.csv'

lod_df   = pd.read_csv(lod_f, dtype={'chr': str})
eff_df   = pd.read_csv(eff_f)
egene_df = pd.read_csv(egene_f)

# founders from the cache in display order
founders = list(FOUNDER_ORDER)        # AJ … WSBJ

# strain → group mapping for PxG marker shapes
f1_strains = list(F1_COLORS.keys())
cc_strains = [s for s in cache_strains
              if s not in founders and s not in f1_strains]
strain_groups: dict[str, str] = {}
for s in founders:   strain_groups[s] = 'founder'
for s in f1_strains: strain_groups[s] = 'f1'
for s in cc_strains: strain_groups[s] = 'cc'

# per-chromosome GEV thresholds (optional — pass None to skip threshold line)
thr_by_chr: dict[str, float] = {}
thr_dir = ct_dir / SCAN_TAG
for f in thr_dir.glob('perm_max_lods_chr*.csv'):
    chrom = f.stem.replace('perm_max_lods_chr', '')
    try:
        from mosaiqtl.qtl.thresholds import get_gev_thresholds
        perm = pd.read_csv(f)['max_lod'].values
        tval, _ = get_gev_thresholds(perm)
        if tval is not None:
            thr_by_chr[chrom] = float(tval)
    except Exception:
        pass


# ── PER-GENE LOOP ─────────────────────────────────────────────────────────────
for gene in GENES_OF_INTEREST:
    row = egene_df[egene_df['gene_symbol'] == gene]
    if row.empty:
        print(f'  {gene}: not in eqtl_summary — skipping')
        continue
    row = row.iloc[0]
    chrom      = str(row['chr'])
    peak_mb    = float(row['peak_mb'])
    peak_locus = str(row['peak_locus']) if 'peak_locus' in row else None

    # LOD profile for this gene
    lod_g = lod_df[lod_df['gene_symbol'] == gene].sort_values('pos_mb')
    if lod_g.empty:
        print(f'  {gene}: no LOD data — skipping')
        continue

    # founder betas
    beta_cols  = [f'beta_{f}' for f in founders]
    eff_row    = eff_df[eff_df['gene_symbol'] == gene]
    peak_betas = None
    if not eff_row.empty and all(c in eff_row.columns for c in beta_cols):
        peak_betas = eff_row.iloc[0][beta_cols].values.astype(float)

    # haplotype posteriors at the peak locus — shape (n_strains, n_founders)
    loci_ids, pos_mb_arr, probs = gc.get_chromosome(chrom)
    locus_idx = np.where(loci_ids == peak_locus)[0] if peak_locus else None
    hap_probs = None
    if locus_idx is not None and len(locus_idx) > 0:
        hap_probs = probs[locus_idx[0], :, :]   # (n_strains, n_founders)

    # expression means per strain from lognorm counts
    import h5py
    expr_mean: np.ndarray | None = None
    counts_h5 = ct_dir / 'counts_lognorm.h5'
    if counts_h5.exists() and 'gene_id' in row.index:
        with h5py.File(counts_h5, 'r') as fh:
            all_genes   = [g.decode() for g in fh['genes'][:]]
            all_samples = [s.decode() for s in fh['samples'][:]]
            if row['gene_id'] in all_genes:
                gi = all_genes.index(row['gene_id'])
                expr_vec = fh['counts'][gi, :].astype(float)
                sample_strains = [s.split('_')[1] if '_' in s else s
                                  for s in all_samples]
                df_e = pd.DataFrame({'strain': sample_strains, 'expr': expr_vec})
                smean = df_e.groupby('strain')['expr'].mean()
                expr_mean = np.array([smean.get(s, np.nan) for s in cache_strains])

    tss_mb   = float(row['start_mb'])
    near_tss = float(row['near_tss_mb']) if pd.notna(row.get('near_tss_mb')) else None

    # ── locus panel ───────────────────────────────────────────────────────────
    fig_locus = locus_panel(
        lod_g['pos_mb'].values,
        lod_g['lod'].values,
        founders,
        peak_betas=peak_betas,
        peak_pos=peak_mb,
        tss_mb=tss_mb,
        near_tss_mb=near_tss,
        threshold=thr_by_chr.get(chrom),
        chrom=chrom,
        highlight_gene=gene,
        peak_locus=peak_locus,
    )
    out_locus = OUT_DIR / f'locus_{gene}_chr{chrom}.png'
    fig_locus.savefig(out_locus, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_locus)
    print(f'  {gene} (chr{chrom}): locus panel → {out_locus.name}')

    # ── standalone PxG figure ─────────────────────────────────────────────────
    if hap_probs is not None and expr_mean is not None and (~np.isnan(expr_mean)).any():
        fig_pxg = pxg_figure(
            expr_mean,
            hap_probs,
            founders,
            strain_groups=strain_groups,
            phenotype_label='mean log-norm expression',
            gene_symbol=gene,
            chrom=chrom,
            position_mb=peak_mb,
        )
        out_pxg = OUT_DIR / f'pxg_{gene}_chr{chrom}.png'
        fig_pxg.savefig(out_pxg, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_pxg)
        print(f'  {gene} (chr{chrom}): PxG figure  → {out_pxg.name}')
    else:
        print(f'  {gene}: PxG skipped (no expression or haplotype data)')

print('done')
