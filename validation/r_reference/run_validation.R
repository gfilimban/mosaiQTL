#!/usr/bin/env Rscript
# ---------------------------------------------------------------
# r reference for mosaiqtl validation
#
# mirrors the scan_eqtl.R pipeline exactly:
#   - reads exported haplotype probs, kinship, pheno_df, pmap
#   - collapses pheno_df to 48 strain means (path a)
#   - runs scan.h2lmm() with additive model, rop, loco kinship
#   - saves lod scores and kinship for python comparison
#
# usage:
#   Rscript validation/r_reference/run_validation.R validation/r_data/
# ---------------------------------------------------------------

library(miqtl)
suppressPackageStartupMessages(library(dplyr))

args       <- commandArgs(trailingOnly = TRUE)
data_dir   <- ifelse(length(args) > 0, args[1], 'validation/r_data')
output_dir <- file.path(data_dir, 'r_output')
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat('[init] reading simulated data from:', data_dir, '\n')

# ── read phenotype ────────────────────────────────────────────────────────────
pheno_df <- read.csv(file.path(data_dir, 'pheno_df.csv'),
                     stringsAsFactors = FALSE, check.names = FALSE)
cat('  pheno_df:', nrow(pheno_df), 'samples\n')

# ── read physical map ──────────────────────────────────────────────────────────
pmap <- read.csv(file.path(data_dir, 'pmap.csv'), stringsAsFactors = FALSE,
                 check.names = FALSE)
colnames(pmap)[colnames(pmap) == 'marker'] <- 'locus'
colnames(pmap)[colnames(pmap) == 'pos']    <- 'pos_mb'
rownames(pmap) <- pmap$locus

# ── read kinship (loco) ───────────────────────────────────────────────────────
k_dir   <- file.path(data_dir, 'kinship')
k_files <- list.files(k_dir, pattern = 'kinship_chr.*\\.csv', full.names = TRUE)
kinship_loco <- list()
for (kf in k_files) {
  chrom <- gsub('.*kinship_chr(.*)\\.csv', '\\1', kf)
  k_mat <- as.matrix(read.csv(kf, row.names = 1, check.names = FALSE))
  kinship_loco[[chrom]] <- k_mat
}
cat('  loaded kinship for chromosomes:', paste(names(kinship_loco), collapse = ', '), '\n')

# ── reconstruct haplotype probabilities ───────────────────────────────────────
# read flat csv per chromosome, build 3d arrays for scan.h2lmm()
# (in real pipeline this comes from the happy genome cache)
hap_dir     <- file.path(data_dir, 'haplotypes')
chromosomes <- gsub('chr(.*)\\.csv', '\\1', list.files(hap_dir, pattern = 'chr.*\\.csv'))
founders    <- NULL

# we need to write a temporary genome cache directory for scan.h2lmm()
temp_cache <- file.path(output_dir, 'genomecache', 'additive')

for (chrom in chromosomes) {
  df <- read.csv(file.path(hap_dir, paste0('chr', chrom, '.csv')),
                 stringsAsFactors = FALSE, check.names = FALSE)

  if (is.null(founders)) {
    # founder names = columns after locus, pos_mb, subject
    founders <- setdiff(colnames(df), c('locus', 'pos_mb', 'subject'))
  }

  subjects <- unique(df$subject)
  loci     <- unique(df$locus)

  # create cache directory
  cache_chr_dir <- file.path(temp_cache, paste0('chr', chrom), 'data')
  dir.create(cache_chr_dir, recursive = TRUE, showWarnings = FALSE)

  for (loc in loci) {
    loc_df  <- df[df$locus == loc, ]
    # build 48 x 8 matrix (subjects x founders)
    mat <- as.matrix(loc_df[, founders])
    rownames(mat) <- loc_df$subject

    # save as .RData (matching happy cache format)
    # assign to variable with locus name, then save
    assign(loc, mat)
    save(list = loc, file = file.path(cache_chr_dir, paste0(loc, '.RData')))
    rm(list = loc)
  }
}

genomecache_path <- file.path(output_dir, 'genomecache')
cat('  built temp genome cache at:', genomecache_path, '\n')

# ── collapse to 48 strain means (matching scan_eqtl.R) ─────────────────────
gene_names <- grep('^gene', colnames(pheno_df), value = TRUE)
pc_names   <- grep('^pc', colnames(pheno_df), value = TRUE)
avg_cols   <- c(gene_names, pc_names)

pheno_48 <- aggregate(pheno_df[, avg_cols],
                      by  = list(subject.name = pheno_df$subject.name),
                      FUN = mean, na.rm = TRUE)
pheno_48$batch     <- tapply(pheno_df$batch, pheno_df$subject.name,
                             `[`, 1)[pheno_48$subject.name]
rownames(pheno_48) <- pheno_48$subject.name

cat('  pheno_48:', nrow(pheno_48), 'strains x', ncol(pheno_48), 'columns\n')

# diagnostic: verify name consistency across data structures
cat('\n[diagnostic] name consistency check:\n')
cat('  pheno_48 subject.name (first 5): ',
    paste(head(pheno_48$subject.name, 5), collapse = ', '), '\n')
cat('  kinship rownames (first 5):      ',
    paste(head(rownames(kinship_loco[['1']]), 5), collapse = ', '), '\n')

# check genome cache
test_cache_dir <- file.path(genomecache_path, 'additive', 'chr1', 'data')
if (dir.exists(test_cache_dir)) {
  test_rdata <- list.files(test_cache_dir)[1]
  test_env <- new.env()
  load(file.path(test_cache_dir, test_rdata), envir = test_env)
  test_mat <- test_env[[ls(test_env)[1]]]
  cat('  cache rownames (first 5):         ',
      paste(head(rownames(test_mat), 5), collapse = ', '), '\n')
  cat('  cache dims:                       ',
      paste(dim(test_mat), collapse = ' x '), '\n')

  # check overlap
  overlap <- intersect(pheno_48$subject.name, rownames(test_mat))
  cat('  name overlap (pheno vs cache):    ', length(overlap), '/',
      nrow(pheno_48), '\n')
  if (length(overlap) == 0) {
    cat('  *** WARNING: no overlapping names — scans will fail\n')
  }
} else {
  cat('  cache dir not found: ', test_cache_dir, '\n')
}
cat('\n')

# ── scan each gene per chromosome ─────────────────────────────────────────────
n_pcs         <- 10
cis_window_mb <- 0.5
results_list  <- list()
effects_list  <- list()
h2_list       <- list()

for (gene in gene_names) {
  cat('  scanning:', gene, '\n')

  for (scan_chr in chromosomes) {
    # cis loci
    chr_loci <- pmap[pmap$chr == scan_chr, ]
    # use midpoint as "tss" (no real annotation in simulated data)
    gene_tss <- mean(chr_loci$pos_mb)
    cis_loci <- chr_loci[abs(chr_loci$pos_mb - gene_tss) <= cis_window_mb, 'locus']

    if (length(cis_loci) == 0) next

    # kinship
    k_chr <- kinship_loco[[scan_chr]]

    # formula (no sex after averaging)
    scan_formula <- as.formula(
      paste0('`', gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse = ' + '))
    )

    scan_result <- tryCatch(
      scan.h2lmm(
        genomecache           = genomecache_path,
        data                  = pheno_48,
        pheno.id              = 'subject.name',
        geno.id               = 'subject.name',
        formula               = scan_formula,
        K                     = k_chr,
        use.multi.impute      = FALSE,
        model                 = 'additive',
        just.these.loci       = cis_loci,
        return.allele.effects = TRUE
      ),
      error = function(e) {
        cat('  *** scan failed:', gene, 'chr', scan_chr, '-', e$message, '\n')
        NULL
      }
    )

    if (is.null(scan_result)) next

    pos_vals <- pmap$pos_mb[match(scan_result$loci, pmap$locus)]

    # per-locus results
    results_list[[paste(gene, scan_chr, sep = '_')]] <- data.frame(
      gene_id = gene,
      locus   = scan_result$loci,
      chr     = scan_chr,
      pos_mb  = pos_vals,
      p_value = scan_result$p.value,
      lod     = scan_result$LOD,
      stringsAsFactors = FALSE
    )

    # allele effects at peak locus for this gene-chr pair
    peak_idx <- which.max(scan_result$LOD)
    if (!is.null(scan_result$allele.effects) && length(peak_idx) > 0) {
      effects <- scan_result$allele.effects[peak_idx, ]
      effects_list[[paste(gene, scan_chr, sep = '_')]] <- data.frame(
        gene_id  = gene,
        chr      = scan_chr,
        locus    = scan_result$loci[peak_idx],
        founder  = names(effects),
        effect   = as.numeric(effects),
        stringsAsFactors = FALSE
      )
    }

    # h2 under null
    if (!is.null(scan_result$h2)) {
      h2_list[[paste(gene, scan_chr, sep = '_')]] <- data.frame(
        gene_id = gene,
        chr     = scan_chr,
        h2_null = scan_result$h2,
        stringsAsFactors = FALSE
      )
    }
  }
}

# ── combine and save ─────────────────────────────────────────────────────────
if (length(results_list) == 0) {
  cat('\n[error] no scan results collected — all scans failed.\n')
  cat('  diagnostic info:\n')
  cat('  pheno_48 subject.names: ', paste(head(pheno_48$subject.name, 5), collapse = ', '), '...\n')
  cat('  k_chr rownames: ', paste(head(rownames(kinship_loco[['1']]), 5), collapse = ', '), '...\n')
  # check if cache loci exist
  test_cache <- file.path(genomecache_path, 'additive', 'chr1', 'data')
  if (dir.exists(test_cache)) {
    cat('  cache loci (chr1): ', paste(head(list.files(test_cache), 3), collapse = ', '), '...\n')
  } else {
    cat('  cache dir not found: ', test_cache, '\n')
  }
  stop('no results to combine')
}

results_all           <- do.call(rbind, results_list)
rownames(results_all) <- NULL

# lead locus per gene
lead_per_gene <- results_all %>%
  group_by(gene_id) %>%
  slice_min(p_value, n = 1, with_ties = FALSE) %>%
  ungroup()

write.csv(results_all,
          file.path(output_dir, 'scan_results.csv'),
          row.names = FALSE)
write.csv(lead_per_gene,
          file.path(output_dir, 'lead_per_gene.csv'),
          row.names = FALSE)

# save allele effects
if (length(effects_list) > 0) {
  effects_all <- do.call(rbind, effects_list)
  rownames(effects_all) <- NULL
  write.csv(effects_all,
            file.path(output_dir, 'allele_effects.csv'),
            row.names = FALSE)
}

# save h2 estimates
if (length(h2_list) > 0) {
  h2_all <- do.call(rbind, h2_list)
  rownames(h2_all) <- NULL
  write.csv(h2_all,
            file.path(output_dir, 'h2_null.csv'),
            row.names = FALSE)
}

cat('\n[results]\n')
for (i in seq_len(nrow(lead_per_gene))) {
  row <- lead_per_gene[i, ]
  cat(sprintf('  %s: chr%s pos=%.1f lod=%.2f p=%.2e\n',
              row$gene_id, row$chr, row$pos_mb, row$lod, row$p_value))
}

cat('\n[done] r validation outputs saved to:', output_dir, '\n')
