# build_allele_dir.R
# builds miQTL-compatible allele_dir from snp_genos.csv.
#
# allele_dir format (miQTL imputed.snp.scan.h2lmm):
#   allele_dir/chrN.alleles — one flat text file per chromosome
#   9 lines per SNP: SNP_ID then 8 founder dosages (0/1/2), one per line
#   founder order: AJ B6J 129S1J NODJ NZOJ CASTJ PWKJ WSBJ (A-H HAPPY order)
#
# usage: Rscript validation/build_allele_dir.R
# ─────────────────────────────────────────────────────────────────────────────
data_dir   <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
snp_csv    <- file.path(data_dir, 'input',  'snp_genos.csv')
allele_dir <- file.path(data_dir, 'output', 'allele_dir')
founders   <- c('A','B','C','D','E','F','G','H')   # HAPPY single-letter codes (A=AJ..H=WSBJ)
full_names <- c('AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ')

if (!file.exists(snp_csv))
  stop('snp_genos.csv not found: ', snp_csv,
       '\nrun: python validation/build_allele_dir.py first')

message('[load] reading ', snp_csv)
df      <- read.csv(snp_csv, row.names=1, stringsAsFactors=FALSE,
                    check.names=FALSE)       # preserve 129S1J without mangling
df$chr  <- as.character(df$chr)
message('  ', nrow(df), ' SNPs x ', ncol(df), ' cols')

founders <- founders[founders %in% colnames(df)]
message('  founders: ', paste(founders, collapse=', '))

# rename full founder names to HAPPY single-letter codes if needed
present_full <- full_names[full_names %in% colnames(df)]
if (length(present_full) > 0) {
  colnames(df)[match(present_full, colnames(df))] <-
    founders[match(present_full, full_names)]
  message('  renamed columns to HAPPY letters')
}

dir.create(allele_dir, recursive=TRUE, showWarnings=FALSE)
n_total <- 0

for (chr in sort(unique(df$chr))) {
  chr_df   <- df[df$chr == chr, , drop=FALSE]
  out_path <- file.path(allele_dir, paste0('chr', chr, '.alleles'))
  con      <- file(out_path, 'w')
  for (snp_id in rownames(chr_df)) {
    dosages <- as.integer(round(as.numeric(chr_df[snp_id, founders])))
    dosages[is.na(dosages)] <- 0L          # nan -> 0 (ref)
    writeLines(c(snp_id, as.character(dosages)), con)  # 9 lines: id + 8 dosages
    n_total <- n_total + 1
  }
  close(con)
  message('  chr', chr, ': ', nrow(chr_df), ' SNPs -> ', out_path)
}

cat('\n[done] allele_dir:', allele_dir, '\n')
cat('  total SNPs written:', n_total, '\n')
