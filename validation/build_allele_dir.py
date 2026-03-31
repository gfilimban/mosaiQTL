# build_allele_dir.py
# builds miQTL-compatible allele_dir from GigaMUGA founder genotype CSV.
#
# allele_dir format (miQTL imputed.snp.scan.h2lmm):
#   allele_dir/chrN.alleles  — one flat text file per chromosome
#   9 lines per SNP:
#     line 1 : SNP_ID
#     lines 2-9 : one founder dosage per line (0/1/2), in A-H HAPPY order
#       A=AJ B=B6J C=129S1J D=NODJ E=NZOJ F=CASTJ G=PWKJ H=WSBJ
#   values: 0=ref homozygous, 1=het, 2=alt homozygous
#
# input: GM_ccfo_mm39.csv — same file used by GenomeCache.from_gigamuga()
#   genotype encoding: A/B/H where A=ref, H=het, B=alt
#
# usage:
#   python validation/build_allele_dir.py
#   python validation/build_allele_dir.py --chrs 8,19
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import os, sys, argparse, logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('build_allele_dir')

ap = argparse.ArgumentParser()
ap.add_argument('--chrs', default=None,
                help='comma-separated chromosomes e.g. 8,19 (default: all)')
ap.add_argument('--min_maf', default=0.0, type=float,
                help='minimum minor allele frequency across founders (default: 0)')
args = ap.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
data_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
gm_path    = os.path.join(data_dir, 'input', 'GM_ccfo_mm39.csv')
pmap_path  = os.path.join(data_dir, 'input', 'pmap.csv')   # unused; kept for snp_genos reference
allele_dir = os.path.join(data_dir, 'output', 'allele_dir')
snp_out    = os.path.join(data_dir, 'input',  'snp_genos.csv')

# founder columns in GigaMUGA CSV, in A-H HAPPY order
FOUNDER_COLS = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']

# ── load GigaMUGA ─────────────────────────────────────────────────────────────
log.info(f'loading GigaMUGA from {gm_path}...')
gm = pd.read_csv(gm_path, low_memory=False)
log.info(f'  {len(gm)} SNPs x {len(gm.columns)} cols')

# set marker column as index (SNP ids); rename pos->pos_mb
if 'marker' in gm.columns: gm = gm.set_index('marker')
if 'pos'    in gm.columns: gm = gm.rename(columns={'pos': 'pos_mb'})
gm['chr'] = gm['chr'].astype(str)
log.info(f'  SNP index: {gm.index[:3].tolist()}...')

# filter chromosomes if requested
if args.chrs:
    keep = [c.strip() for c in args.chrs.split(',')]
    gm   = gm[gm['chr'].isin(keep)]
    log.info(f'  filtered to chr {keep}: {len(gm)} SNPs')

# ── encode founder nucleotide calls → 0/1/2 dosage ───────────────────────────
# founders are inbred: single letter = homozygous (T means TT)
# per SNP: determine ref (most common non-missing allele) and alt allele
# ref homozygous=0, alt homozygous=2, missing/N=0 (impute as ref)
f_cols = [f for f in FOUNDER_COLS if f in gm.columns]
if len(f_cols) < 2:
    log.error(f'founder columns not found. available: {list(gm.columns[:20])}')
    sys.exit(1)
log.info(f'  encoding {len(f_cols)} founders: {f_cols}')

def calls_to_dosage(row):
    '''convert nucleotide founder calls to 0/1/2 per SNP row'''
    calls = row[f_cols].values
    # clean: uppercase, treat empty string / 'N' / '0' as missing
    cleaned = []
    for c in calls:
        if pd.isna(c): cleaned.append(None)
        else:
            s = str(c).strip().upper()
            cleaned.append(None if s in ('', 'N', '0', 'NA') else s)
    # determine alleles from non-missing calls
    alleles = [c for c in cleaned if c is not None]
    unique  = list(dict.fromkeys(alleles))   # preserve order, deduplicate
    if len(unique) == 0:
        return pd.Series([0.0] * len(f_cols), index=f_cols)
    ref = max(set(alleles), key=alleles.count)  # most common = ref
    alt = next((a for a in unique if a != ref), None)
    dosages = []
    for c in cleaned:
        if c is None:        dosages.append(0.0)   # missing -> ref
        elif c == ref:       dosages.append(0.0)
        elif c == alt:       dosages.append(2.0)
        else:                dosages.append(0.0)   # unexpected allele -> ref
    return pd.Series(dosages, index=f_cols)

gm[f_cols] = gm.apply(calls_to_dosage, axis=1)

# sanity: count polymorphic SNPs (any founder != 0)
n_poly = (gm[f_cols].max(axis=1) > 0).sum()
log.info(f'  polymorphic SNPs (any founder alt): {n_poly} / {len(gm)}')

# ── filter by MAF ─────────────────────────────────────────────────────────────
if args.min_maf > 0:
    dosages = gm[f_cols].values
    maf     = np.nanmean(dosages, axis=1) / 2.0
    maf     = np.minimum(maf, 1 - maf)
    gm      = gm[maf >= args.min_maf]
    log.info(f'  after MAF>={args.min_maf} filter: {len(gm)} SNPs')

# ── save snp_genos.csv for validate_port.py S8 ───────────────────────────────
snp_df = gm[['chr', 'pos_mb'] + f_cols].copy()
snp_df[f_cols] = snp_df[f_cols].fillna(0).astype(int)  # nan (missing) -> 0 (ref)
snp_df.to_csv(snp_out)
log.info(f'saved snp_genos.csv: {len(snp_df)} SNPs -> {snp_out}')

# ── write allele_dir/chrN.alleles flat text files ────────────────────────────
# format: 9 lines per SNP — SNP_ID then 8 founder dosages, one per line
os.makedirs(allele_dir, exist_ok=True)
log.info(f'writing allele_dir to {allele_dir}...')

for chrom, chr_df in gm.groupby('chr'):
    out_path  = os.path.join(allele_dir, f'chr{chrom}.alleles')
    n_written = 0
    with open(out_path, 'w') as fh:
        for snp_id, row in chr_df.iterrows():
            dosages = [row.get(f, np.nan) for f in f_cols]
            dosages = [0.0 if np.isnan(d) else d for d in dosages]  # nan -> 0 (ref)
            fh.write(snp_id + '\n')       # line 1: SNP id
            for d in dosages:
                fh.write(f'{int(d)}\n')   # lines 2-9: one dosage per founder
            n_written += 1
    log.info(f'  chr{chrom}: {n_written} SNPs -> {out_path}')

log.info('done')
