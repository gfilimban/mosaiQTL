# build_allele_dir.R
# builds miQTL-compatible allele_dir from snp_genos.csv.
# run AFTER build_allele_dir.py has written snp_genos.csv,
# OR as a standalone replacement if rpy2 is unavailable.
#
# allele_dir format: one .RData file per SNP, each containing a named
# numeric vector of founder allele dosages (0=ref, 1=het, 2=alt).
#
# usage:
#   Rscript validation/build_allele_dir.R
# ─────────────────────────────────────────────────────────────────────────────
data_dir   <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
snp_csv    <- file.path(data_dir, 'input',  'snp_genos.csv')
allele_dir <- file.path(data_dir, 'output', 'allele_dir')
founders   <- c('AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ')

if (!file.exists(snp_csv))
  stop('snp_genos.csv not found: ', snp_csv,
       '\nrun python validation/build_allele_dir.py first')

message('[load] reading ', snp_csv)
df      <- read.csv(snp_csv, row.names=1, stringsAsFactors=FALSE)
df$chr  <- as.character(df$chr)
message('  ', nrow(df), ' SNPs × ', ncol(df), ' cols')

# subset founders to those present in the CSV
founders <- founders[founders %in% colnames(df)]
message('  founders: ', paste(founders, collapse=', '))

chrs    <- sort(unique(df$chr))
n_total <- 0; n_skip  <- 0

for (chr in chrs) {
  chr_df  <- df[df$chr == chr, , drop=FALSE]
  chr_dir <- file.path(allele_dir, paste0('chr', chr))
  dir.create(chr_dir, recursive=TRUE, showWarnings=FALSE)

  for (snp_id in rownames(chr_df)) {
    out_path <- file.path(chr_dir, paste0(snp_id, '.RData'))
    if (file.exists(out_path)) { n_skip <- n_skip + 1; next }
    geno_vec           <- as.numeric(chr_df[snp_id, founders])
    geno_vec[is.na(geno_vec)] <- 0
    names(geno_vec)    <- founders
    assign(snp_id, geno_vec)
    save(list=snp_id, file=out_path)
    rm(list=snp_id)
    n_total <- n_total + 1
  }
  message('  chr', chr, ': ', nrow(chr_df), ' SNPs  written=', n_total,
          '  skipped=', n_skip)
}

cat('\n[done] allele_dir:', allele_dir, '\n')
cat('  total written:', n_total, '  skipped (existing):', n_skip, '\n')
