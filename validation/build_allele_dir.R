# build_allele_dir.R
# builds miQTL-compatible allele_dir from snp_genos.csv.
#
# allele_dir format (miQTL extract.imputed.design.matrix.from.doqtl.genotype):
#   allele_dir/chrN.alleles — one flat text file per chromosome
#   4 lines per SNP:
#     line 1 : SNP_ID
#     line 2 : ref_allele\t0/1 per founder
#     line 3 : alt_allele\t0/1 per founder  (or NA if monomorphic)
#     line 4 : NA\t0 per founder
#   miQTL reads via: grep -w -A 3 'SNP_ID' chrN.alleles
#
# usage: Rscript validation/build_allele_dir.R
# ─────────────────────────────────────────────────────────────────────────────
data_dir   <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
gm_path    <- file.path(data_dir, 'input', 'GM_ccfo_mm39.csv')
allele_dir <- file.path(data_dir, 'output', 'allele_dir')

# founder columns in A-H HAPPY order
founders   <- c('AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ')

message('[load] reading ', gm_path)
gm         <- read.csv(gm_path, row.names=1, stringsAsFactors=FALSE,
                       check.names=FALSE)
gm$chr     <- as.character(gm$chr)
if ('pos' %in% colnames(gm)) colnames(gm)[colnames(gm)=='pos'] <- 'pos_mb'
message('  ', nrow(gm), ' SNPs x ', ncol(gm), ' cols')

f_cols <- founders[founders %in% colnames(gm)]
message('  founders: ', paste(f_cols, collapse=', '))

dir.create(allele_dir, recursive=TRUE, showWarnings=FALSE)
n_total <- 0; n_skip <- 0

for (chr in sort(unique(gm$chr))) {
  chr_df   <- gm[gm$chr == chr, , drop=FALSE]
  out_path <- file.path(allele_dir, paste0('chr', chr, '.alleles'))
  con      <- file(out_path, 'w')

  for (snp_id in rownames(chr_df)) {
    calls <- toupper(trimws(as.character(chr_df[snp_id, f_cols])))
    calls[calls %in% c('', 'N', 'NA', 'NAN')] <- NA

    valid <- calls[!is.na(calls)]
    if (length(valid) == 0) { n_skip <- n_skip + 1; next }

    # ref = most common allele; alt = second allele if present
    ref   <- names(sort(table(valid), decreasing=TRUE))[1]
    alts  <- setdiff(unique(valid), ref)
    alt   <- if (length(alts) > 0) alts[1] else NA

    ref_row <- as.integer(!is.na(calls) & calls == ref)
    alt_row <- if (!is.na(alt)) as.integer(!is.na(calls) & calls == alt) else rep(0L, length(f_cols))
    na_row  <- rep(0L, length(f_cols))

    writeLines(snp_id, con)
    writeLines(paste(c(ref,  ref_row), collapse='\t'), con)
    writeLines(paste(c(if (!is.na(alt)) alt else 'NA', alt_row), collapse='\t'), con)
    writeLines(paste(c('NA', na_row),  collapse='\t'), con)
    n_total <- n_total + 1
  }
  close(con)
  message('  chr', chr, ': ', nrow(chr_df), ' SNPs  written=', n_total, '  skipped=', n_skip)
}

cat('\n[done] allele_dir:', allele_dir, '\n')
cat('  total written:', n_total, '  skipped:', n_skip, '\n')
