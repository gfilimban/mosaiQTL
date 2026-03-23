# validate_vs_r.py
# compare mosaiqtl scan output against r miqtl lod scores
#
# reads pre-existing files produced by run_validation.R:
#   output/validation/lod_r.csv        — r miqtl lod scores (gene × locus)
#   output/validation/pheno_48_used.csv — the exact pheno_48 r used
#   output/genomecache/                 — happy genome cache
#   input/pmap.csv                      — physical map (marker, chr, pos_mb)
#
# run order:
#   1. Rscript validation/r_reference/run_validation.R 8 5
#   2. python validation/validate_vs_r.py [--chr 8] [--n_genes 5]
#
# pass threshold: pearson r > 0.999 and max|Δlod| < 0.01 per gene

import numpy as np
import pandas as pd
import os, sys, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import scan as scan_module
from mosaiqtl.utils import rint

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
log = logging.getLogger('validate')

# ── paths ─────────────────────────────────────────────────────────────────────
base_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'
cache_dir  = os.path.join(base_dir, 'output', 'genomecache')
val_dir    = os.path.join(pkg_dir, 'validation')  # where run_validation.R writes
pmap_path  = os.path.join(base_dir, 'input',  'pmap.csv')

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--chr',     default='8', help='chromosome (must match what R used)')
parser.add_argument('--n_genes', default=5,   type=int)
args   = parser.parse_args()
chr_id = args.chr

# ── load r reference lod scores ───────────────────────────────────────────────
lod_r_path = os.path.join(val_dir, 'r_data', 'lod_r.csv')
if not os.path.exists(lod_r_path):
    log.error(f'lod_r.csv not found at {lod_r_path}')
    log.error('run first: Rscript validation/r_reference/run_validation.R')
    sys.exit(1)

lod_r      = pd.read_csv(lod_r_path)
gene_list  = lod_r['gene_id'].unique()[:args.n_genes].tolist()
log.info(f'r reference: {lod_r.shape[0]} rows | genes: {gene_list}')

# ── load pheno_48 (exact matrix r used) ──────────────────────────────────────
pheno_path = os.path.join(val_dir, 'r_data', 'pheno_48_used.csv')
if not os.path.exists(pheno_path):
    log.error(f'pheno_48_used.csv not found at {pheno_path}')
    log.error('run_validation.R exports this automatically')
    sys.exit(1)

pheno_48 = pd.read_csv(pheno_path, index_col=0)   # rownames = subject.name
pheno_48.index.name = 'subject.name'
pheno_48 = pheno_48.reset_index()
log.info(f'pheno_48: {pheno_48.shape[0]} strains × {pheno_48.shape[1]} cols')

# ── load genome cache ──────────────────────────────────────────────────────────
if not os.path.isdir(cache_dir):
    log.error(f'genome cache not found: {cache_dir}')
    sys.exit(1)

log.info(f'loading cache from {cache_dir}')
cache = GenomeCache.from_happy_dir(cache_dir, model='additive')
log.info(f'cache: {len(cache.chromosomes)} chromosomes | '
         f'{len(cache.sample_ids)} strains | {cache.n_founders} founders')

# ── load pmap ─────────────────────────────────────────────────────────────────
pmap = pd.read_csv(pmap_path)
if 'marker' in pmap.columns:
    pmap = pmap.rename(columns={'marker': 'locus', 'pos': 'pos_mb'})
# strip .RData suffix defensively — scan.h2lmm() strips it on the R side too
pmap['locus'] = pmap['locus'].str.replace('.RData', '', regex=False)
chr_pmap = pmap[pmap['chr'].astype(str) == str(chr_id)].copy()
log.info(f'pmap chr{chr_id}: {len(chr_pmap)} loci')

# ── build covariate matrix ────────────────────────────────────────────────────
# batch (dummy-encoded, drop first) + pc1..pc10 — same as scan_eqtl.R formula
pc_cols = [f'pc{i}' for i in range(1, 11)]
batch_dummies = pd.get_dummies(pheno_48['batch'], prefix='batch', drop_first=True)
covar = pd.concat([
    pd.Series(np.ones(len(pheno_48)), name='intercept'),
    batch_dummies.astype(float),
    pheno_48[pc_cols].astype(float),
], axis=1)

# align pheno_48 rows to cache strain order
cache_strains = list(cache.sample_ids)
pheno_48 = pheno_48.set_index('subject.name').reindex(cache_strains).reset_index()
covar    = covar.reindex(pheno_48.index)

log.info(f'covariate matrix: {covar.shape}')

# ── compute loco kinship + eigendecompose ─────────────────────────────────────
log.info('computing loco kinship...')
kinship_dict = kin_module.calc_kinship(cache, method='loco')
kinship_decomp = kin_module.decomp_kinship({chr_id: kinship_dict[chr_id]})
log.info(f'kinship chr{chr_id}: {kinship_dict[chr_id].shape}')

# ── scan each gene and compare ────────────────────────────────────────────────
# for each gene: get the exact set of loci r used, scan those, correlate lods

all_pass    = True
comparisons = []

for gene in gene_list:
    if gene not in pheno_48.columns:
        log.warning(f'{gene}: not in pheno_48 — skipping')
        continue

    y = pheno_48[gene].values.astype(float)

    # get the loci r actually scanned for this gene (from lod_r.csv)
    r_gene = lod_r[lod_r['gene_id'] == gene][['locus', 'lod']].rename(
        columns={'lod': 'lod_r'})
    if r_gene.empty:
        log.warning(f'{gene}: no r loci found — skipping')
        continue

    # tss = midpoint of r loci positions (guarantees same cis window)
    r_positions = chr_pmap.loc[
        chr_pmap['locus'].isin(r_gene['locus']), 'pos_mb']
    gene_tss = float(r_positions.mean()) if len(r_positions) else \
               float(chr_pmap['pos_mb'].mean())

    log.info(f'scanning {gene} | {len(r_gene)} loci | tss≈{gene_tss:.2f} Mb')

    try:
        py_result = scan_module.scan_h2lmm(
            phenotype      = y,
            genome_cache   = cache,
            covariates     = covar.values,
            kinship_decomp = kinship_decomp,
            use_rint       = False,  # pheno_48 already rint'd by scan_eqtl.R
            phenotype_id   = gene,
        )
    except Exception as e:
        log.error(f'  scan_h2lmm failed for {gene}: {e}')
        all_pass = False
        continue

    py_df = py_result.to_dataframe()[['locus_id', 'lod']].rename(
        columns={'locus_id': 'locus', 'lod': 'lod_py'})

    merged = r_gene.merge(py_df, on='locus', how='inner').dropna()

    if len(merged) < 3:
        log.warning(f'  {gene}: only {len(merged)} matching loci — skipping')
        all_pass = False
        continue

    r_corr = float(np.corrcoef(merged['lod_r'], merged['lod_py'])[0, 1])
    max_d  = float((merged['lod_r'] - merged['lod_py']).abs().max())
    passed = r_corr > 0.999 and max_d < 0.01

    status = 'PASS' if passed else 'FAIL'
    log.info(f'  {status}  {gene:25s}  r={r_corr:.6f}  '
             f'max|Δlod|={max_d:.6f}  n={len(merged)}')

    if not passed:
        all_pass = False

    merged['gene_id'] = gene
    comparisons.append(merged)

# ── save and summarise ────────────────────────────────────────────────────────
if comparisons:
    out = pd.concat(comparisons, ignore_index=True)
    out_path = os.path.join(val_dir, 'lod_comparison.csv')
    os.makedirs(val_dir, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info(f'\nsaved: {out_path}  ({len(out)} rows)')

print()
print('=' * 55)
print('  ALL GENES PASS' if all_pass else '  SOME GENES FAILED — see log above')
print('  threshold: r > 0.999 and max|Δlod| < 0.01')
print('=' * 55)

sys.exit(0 if all_pass else 1)
