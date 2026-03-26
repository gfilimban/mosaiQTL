# build_allele_dir.py
# builds miQTL-compatible allele_dir from GigaMUGA founder genotype CSV.
#
# allele_dir format (miQTL HAPPY):
#   allele_dir/chrN/SNP_ID.RData
#   each file: named numeric vector, one value per founder (0/1/2 dosage)
#
# input: GM_ccfo_mm39.csv — same file used by GenomeCache.from_gigamuga()
#   rows = SNPs, columns include founders (AJ, B6J, 129S1J, NODJ, NZOJ, CASTJ, PWKJ, WSBJ)
#   genotype encoding: A/B/H where A=ref homozygous, B=alt homozygous, H=het
#   converted to dosage: A=0, H=1, B=2
#
# usage:
#   python validation/build_allele_dir.py
#   python validation/build_allele_dir.py --chrs 8,19  # specific chromosomes only
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
data_dir    = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir     = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'
gm_path     = os.path.join(data_dir, 'input', 'GM_ccfo_mm39.csv')
pmap_path   = os.path.join(data_dir, 'input', 'pmap.csv')
allele_dir  = os.path.join(data_dir, 'output', 'allele_dir')
snp_out     = os.path.join(data_dir, 'input',  'snp_genos.csv')

# founder column names in GigaMUGA CSV
FOUNDER_COLS = ['AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ']

# ── load GigaMUGA ─────────────────────────────────────────────────────────────
log.info(f'loading GigaMUGA from {gm_path}...')
gm = pd.read_csv(gm_path, low_memory=False)
log.info(f'  {len(gm)} SNPs × {len(gm.columns)} cols')

# load pmap for chr and position
pmap = pd.read_csv(pmap_path)
if 'marker' in pmap.columns: pmap = pmap.rename(columns={'marker':'locus'})
if 'pos'    in pmap.columns: pmap = pmap.rename(columns={'pos':'pos_mb'})
pmap['locus'] = pmap['locus'].str.replace(r'\.RData$','',regex=True)

# ── align SNP ids ─────────────────────────────────────────────────────────────
# GigaMUGA rows indexed by SNP id (first column or index)
if gm.index.name is None or gm.index.dtype == object:
    snp_col = gm.columns[0]
    gm      = gm.set_index(snp_col)
log.info(f'  SNP index set: {gm.index[:3].tolist()}...')

# subset to SNPs that have chr/pos info
gm = gm.loc[gm.index.isin(pmap['locus'])]
gm = gm.merge(pmap[['locus','chr','pos_mb']].set_index('locus'),
               left_index=True, right_index=True, how='left')
gm['chr'] = gm['chr'].astype(str)
log.info(f'  {len(gm)} SNPs with pmap coordinates')

# filter by chromosome if requested
if args.chrs:
    chrs_keep = [c.strip() for c in args.chrs.split(',')]
    gm = gm[gm['chr'].isin(chrs_keep)]
    log.info(f'  filtered to chr {chrs_keep}: {len(gm)} SNPs')

# ── encode founder genotypes: A/H/B → 0/1/2 ──────────────────────────────────
def encode_dosage(val):
    """A/H/B or homref/het/homalt string → dosage 0/1/2."""
    if pd.isna(val): return np.nan
    v = str(val).strip().upper()
    if v in ('A','0','AA','HOMREF','REF'): return 0.0
    if v in ('H','1','AB','HET'):          return 1.0
    if v in ('B','2','BB','HOMALT','ALT'): return 2.0
    return np.nan

f_cols_present = [f for f in FOUNDER_COLS if f in gm.columns]
if len(f_cols_present) < 2:
    log.error(f'founder columns not found. available: {list(gm.columns[:20])}')
    sys.exit(1)
log.info(f'  encoding {len(f_cols_present)} founders: {f_cols_present}')

for f in f_cols_present:
    gm[f] = gm[f].apply(encode_dosage)

# ── filter by MAF ─────────────────────────────────────────────────────────────
if args.min_maf > 0:
    dosages = gm[f_cols_present].values
    maf     = np.nanmean(dosages, axis=1) / 2.0
    maf     = np.minimum(maf, 1 - maf)
    gm      = gm[maf >= args.min_maf]
    log.info(f'  after MAF≥{args.min_maf} filter: {len(gm)} SNPs')

# ── save snp_genos.csv (for Python validate_port.py s8) ──────────────────────
snp_df = gm[['chr','pos_mb'] + f_cols_present].copy()
snp_df.to_csv(snp_out)
log.info(f'saved snp_genos.csv: {len(snp_df)} SNPs → {snp_out}')

# ── write allele_dir (for R imputed.snp.scan.h2lmm) ──────────────────────────
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    has_rpy2 = True
except ImportError:
    has_rpy2 = False
    log.warning('rpy2 not available — skipping allele_dir .RData write')
    log.warning('allele_dir can still be built in R with:')
    log.warning('  source("validation/build_allele_dir.R")')

if has_rpy2:
    log.info(f'writing allele_dir to {allele_dir}...')
    os.makedirs(allele_dir, exist_ok=True)
    n_written = 0; n_skip = 0

    for chrom, chr_df in gm.groupby('chr'):
        chr_dir = os.path.join(allele_dir, f'chr{chrom}')
        os.makedirs(chr_dir, exist_ok=True)

        for snp_id, row in chr_df.iterrows():
            rdata_path = os.path.join(chr_dir, f'{snp_id}.RData')
            if os.path.exists(rdata_path):
                n_skip += 1; continue
            geno_vec = np.array([float(row.get(f, np.nan)) for f in f_cols_present])
            geno_vec = np.nan_to_num(geno_vec, nan=0.0)
            rv       = ro.FloatVector(geno_vec)
            rv.names = ro.StrVector(f_cols_present)
            env      = ro.r['new.env']()
            env[snp_id] = rv
            ro.r['save'](ro.StrVector([snp_id]), file=rdata_path, envir=env)
            n_written += 1

        log.info(f'  chr{chrom}: {len(chr_df)} SNPs  written={n_written}  skipped={n_skip}')

    log.info(f'allele_dir complete: {n_written} files written  {n_skip} skipped')
    log.info(f'  path: {allele_dir}')
else:
    # write an R fallback script
    r_script = f'''# build_allele_dir.R — R fallback for allele_dir construction
# run if rpy2 is unavailable: Rscript validation/build_allele_dir.R
snp_csv    <- "{snp_out}"
allele_dir <- "{allele_dir}"
founders   <- c({", ".join([f'"{f}"' for f in f_cols_present])})
df         <- read.csv(snp_csv, row.names=1, stringsAsFactors=FALSE)
df$chr     <- as.character(df$chr)
for (chr in unique(df$chr)) {{
  chr_df  <- df[df$chr == chr, ]
  chr_dir <- file.path(allele_dir, paste0('chr', chr))
  dir.create(chr_dir, recursive=TRUE, showWarnings=FALSE)
  for (snp_id in rownames(chr_df)) {{
    out_path <- file.path(chr_dir, paste0(snp_id, '.RData'))
    if (file.exists(out_path)) next
    geno_vec <- as.numeric(chr_df[snp_id, founders])
    geno_vec[is.na(geno_vec)] <- 0
    names(geno_vec) <- founders
    assign(snp_id, geno_vec)
    save(list=snp_id, file=out_path)
    rm(list=snp_id)
  }}
  message('  chr', chr, ': done')
}}
cat('allele_dir built:', allele_dir, '\\n')
'''
    r_out = os.path.join(os.path.dirname(__file__), 'build_allele_dir.R')
    open(r_out, 'w').write(r_script)
    log.info(f'wrote R fallback: {r_out}')
    log.info('run with: Rscript validation/build_allele_dir.R')

log.info('done')
