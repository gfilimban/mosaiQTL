# build_allele_dir.py
# builds miQTL-compatible allele_dir from GigaMUGA founder genotype CSV.
#
# allele_dir format (miQTL extract.imputed.design.matrix.from.doqtl.genotype):
#   allele_dir/chrN.alleles — one flat text file per chromosome
#   4 lines per SNP:
#     line 1  : SNP_ID
#     line 2  : ref_allele\tf1\tf2\t...\tf8   (1=has ref, 0=does not)
#     line 3  : alt_allele\tf1\tf2\t...\tf8   (1=has alt, 0=does not)
#     line 4  : NA\t0\t0\t...\t0
#   miQTL greps: grep -w -A 3 'SNP_ID' chrN.alleles
#   then parses 3-row matrix, drops col1 (allele label), uses which.max per
#   founder column to assign founder allele
#
# input: GM_ccf1fo_mm39.csv
#   founders are inbred: single nucleotide call = homozygous
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
args = ap.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
data_dir   = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
gm_path    = os.path.join(data_dir, 'input', 'GM_ccf1fo_mm39.csv')
allele_dir = os.path.join(data_dir, 'output', 'allele_dir')
snp_out    = os.path.join(data_dir, 'input',  'snp_genos.csv')

# founder columns in GigaMUGA CSV, in A-H HAPPY order
FOUNDER_COLS  = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
HAPPY_LETTERS = ['A',  'B',   'C',      'D',    'E',    'F',     'G',    'H']

# ── load GigaMUGA ─────────────────────────────────────────────────────────────
log.info(f'loading GigaMUGA from {gm_path}...')
gm = pd.read_csv(gm_path, low_memory=False)
log.info(f'  {len(gm)} SNPs x {len(gm.columns)} cols')

# set marker as index; rename pos->pos_mb
if 'marker' in gm.columns: gm = gm.set_index('marker')
if 'pos'    in gm.columns: gm = gm.rename(columns={'pos': 'pos_mb'})
gm['chr'] = gm['chr'].astype(str)
log.info(f'  SNP index: {gm.index[:3].tolist()}...')

# filter chromosomes if requested
if args.chrs:
    keep = [c.strip() for c in args.chrs.split(',')]
    gm   = gm[gm['chr'].isin(keep)]
    log.info(f'  filtered to chr {keep}: {len(gm)} SNPs')

f_cols = [f for f in FOUNDER_COLS if f in gm.columns]
if len(f_cols) < 2:
    log.error(f'founder columns not found. available: {list(gm.columns[:20])}')
    sys.exit(1)
log.info(f'  founders: {f_cols}')

# ── save snp_genos.csv with HAPPY letter columns for validate_port.py S8 ─────
snp_df = gm[['chr', 'pos_mb'] + f_cols].copy()
snp_df.columns = ['chr', 'pos_mb'] + HAPPY_LETTERS[:len(f_cols)]
snp_df.to_csv(snp_out)
log.info(f'saved snp_genos.csv: {len(snp_df)} SNPs -> {snp_out}')

# ── write allele_dir/chrN.alleles ─────────────────────────────────────────────
# 4 lines per SNP: SNP_ID / ref_row / alt_row / NA_row
# each data row: allele_label\tf1\tf2\t...\tf8
# founders are inbred (single letter = homozygous); missing/N -> ref
os.makedirs(allele_dir, exist_ok=True)
log.info(f'writing allele_dir to {allele_dir}...')

for chrom, chr_df in gm.groupby('chr'):
    out_path  = os.path.join(allele_dir, f'chr{chrom}.alleles')
    n_written = 0
    with open(out_path, 'w') as fh:
        for snp_id, row in chr_df.iterrows():
            calls = [str(row.get(f, '')).strip().upper() for f in f_cols]
            # treat missing/N as None
            calls = [None if c in ('', 'N', 'NAN', 'NA') else c for c in calls]
            valid = [c for c in calls if c is not None]
            if not valid:
                # all missing — monomorphic ref, skip SNP
                continue
            # determine ref (most common) and alt allele
            ref = max(set(valid), key=valid.count)
            alts = [c for c in set(valid) if c != ref]
            alt = alts[0] if alts else None

            # build per-founder indicator rows (0/1)
            ref_row = [1 if c == ref else 0 for c in calls]
            alt_row = [1 if (alt and c == alt) else 0 for c in calls]
            na_row  = [0] * len(f_cols)

            fh.write(snp_id + '\n')
            fh.write(ref + '\t' + '\t'.join(map(str, ref_row)) + '\n')
            fh.write((alt if alt else 'NA') + '\t' + '\t'.join(map(str, alt_row)) + '\n')
            fh.write('NA\t' + '\t'.join(map(str, na_row)) + '\n')
            n_written += 1

    log.info(f'  chr{chrom}: {n_written} SNPs -> {out_path}')

log.info('done')
