# run_validation.R
# r reference for mosaiqtl cross-language validation
#
# two modes (auto-detected):
#   real data  — reads pheno_df.rds, kinship_loco.rds, genomecache from
#                crsp cc gastrocnemius dataset. preferred.
#   simulation — reads exported csvs from validation/r_data/ (written by
#                validate_vs_r.py in simulation mode). used locally / ci.
#
# outputs (always): scan_results.csv, lead_per_gene.csv, allele_effects.csv
#
# usage:
#   Rscript validation/r_reference/run_validation.R [data_dir]
#   # real mode (on crsp, no args needed):
#   Rscript validation/r_reference/run_validation.R
#   # simulation mode (local):
#   Rscript validation/r_reference/run_validation.R validation/r_data/

suppressPackageStartupMessages({
  lib_path <- '/data/homezvol2/gfilimba/R/x86_64-pc-linux-gnu-library/4.4'
  .libPaths(c(lib_path, .libPaths()))
  library(miqtl)
  library(dplyr)
})

# ── detect mode ───────────────────────────────────────────────────────────────
REAL_BASE  <- '/share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc'
REAL_CACHE <- file.path(REAL_BASE, 'output', 'genomecache')
REAL_PHENO <- file.path(REAL_BASE, 'output', 'pheno_df.rds')
REAL_KIN   <- file.path(REAL_BASE, 'output', 'kinship_loco.rds')
REAL_PMAP  <- file.path(REAL_BASE, 'input', 'pmap.csv')

USE_REAL <- dir.exists(REAL_CACHE) && file.exists(REAL_PHENO)

args     <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else 'validation/r_data'

if (USE_REAL) {
  cat('[mode] real data — cc gastrocnemius dataset\n')
  output_dir <- file.path(REAL_BASE, 'output', 'validation', 'r_output')
} else {
  cat('[mode] simulation — reading from:', data_dir, '\n')
  output_dir <- file.path(data_dir, 'r_output')
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
cat('[output]', output_dir, '\n\n')

# ── load data ─────────────────────────────────────────────────────────────────
if (USE_REAL) {
  # real: read rds + real genomecache directly
  pheno_df     <- readRDS(REAL_PHENO)
  kinship_loco <- readRDS(REAL_KIN)
  pheno_df$sex <- as.factor(pheno_df$sex)
  genomecache_path <- REAL_CACHE

  pmap <- read.csv(REAL_PMAP, stringsAsFactors = FALSE, check.names = FALSE)
  if ('marker' %in% colnames(pmap)) colnames(pmap)[colnames(pmap) == 'marker'] <- 'locus'
  if ('pos'    %in% colnames(pmap)) colnames(pmap)[colnames(pmap) == 'pos']    <- 'pos_mb'
  # strip .RData suffix — scan.h2lmm() appends it internally
  pmap$locus    <- sub('\\.RData$', '', pmap$locus)
  rownames(pmap) <- pmap$locus

  # identify gene columns (not metadata / covariates)
  meta_cols <- c('sample_id', 'subject.name', 'sex', 'batch',
                 paste0('pc', 1:10))
  all_genes <- setdiff(colnames(pheno_df), meta_cols)
  gene_names <- head(all_genes, 5)
  n_pcs      <- 10
  pc_cols    <- paste0('pc', 1:n_pcs)
  chromosomes <- as.character(sort(as.integer(
    intersect(unique(pmap$chr), names(kinship_loco)))))

  # collapse 415 replicates → 48 strain means (path a — miqtl original)
  # sex excluded after averaging m + f replicates
  pheno_48 <- aggregate(
    pheno_df[, c(gene_names, pc_cols)],
    by  = list(subject.name = pheno_df$subject.name),
    FUN = mean, na.rm = TRUE
  )
  pheno_48$batch     <- tapply(as.character(pheno_df$batch),
                               pheno_df$subject.name, `[`, 1)[pheno_48$subject.name]
  rownames(pheno_48) <- pheno_48$subject.name

  cat('[data] pheno_df:', nrow(pheno_df), 'samples\n')
  cat('[data] pheno_48:', nrow(pheno_48), 'strains\n')
  cat('[data] genes:',    paste(gene_names, collapse = ', '), '\n')
  cat('[data] chrs:',     paste(chromosomes, collapse = ', '), '\n\n')

} else {
  # simulation: read csvs exported by validate_vs_r.py
  pheno_df <- read.csv(file.path(data_dir, 'pheno_df.csv'),
                       stringsAsFactors = FALSE, check.names = FALSE)
  cat('[data] pheno_df:', nrow(pheno_df), 'samples\n')

  pmap <- read.csv(file.path(data_dir, 'pmap.csv'),
                   stringsAsFactors = FALSE, check.names = FALSE)
  colnames(pmap)[colnames(pmap) == 'marker'] <- 'locus'
  colnames(pmap)[colnames(pmap) == 'pos']    <- 'pos_mb'
  rownames(pmap) <- pmap$locus

  k_dir   <- file.path(data_dir, 'kinship')
  k_files <- list.files(k_dir, pattern = 'kinship_chr.*\\.csv', full.names = TRUE)
  kinship_loco <- list()
  for (kf in k_files) {
    chrom <- gsub('.*kinship_chr(.*)\\.csv', '\\1', kf)
    kinship_loco[[chrom]] <- as.matrix(
      read.csv(kf, row.names = 1, check.names = FALSE))
  }
  cat('[data] kinship chrs:', paste(names(kinship_loco), collapse = ', '), '\n')

  # build temp genome cache from exported haplotype csvs
  hap_dir     <- file.path(data_dir, 'haplotypes')
  chromosomes <- gsub('chr(.*)\\.csv', '\\1',
                      list.files(hap_dir, pattern = 'chr.*\\.csv'))
  founders    <- NULL
  temp_cache  <- file.path(output_dir, 'genomecache', 'additive')

  for (chrom in chromosomes) {
    df <- read.csv(file.path(hap_dir, paste0('chr', chrom, '.csv')),
                   stringsAsFactors = FALSE, check.names = FALSE)
    if (is.null(founders))
      founders <- setdiff(colnames(df), c('locus', 'pos_mb', 'subject'))

    cache_chr_dir <- file.path(temp_cache, paste0('chr', chrom), 'data')
    dir.create(cache_chr_dir, recursive = TRUE, showWarnings = FALSE)

    for (loc in unique(df$locus)) {
      loc_df <- df[df$locus == loc, ]
      mat    <- as.matrix(loc_df[, founders])
      rownames(mat) <- loc_df$subject
      assign(loc, mat)
      save(list = loc, file = file.path(cache_chr_dir, paste0(loc, '.RData')))
      rm(list = loc)
    }
  }
  genomecache_path <- file.path(output_dir, 'genomecache')
  cat('[data] built temp genome cache at:', genomecache_path, '\n')

  gene_names <- grep('^gene', colnames(pheno_df), value = TRUE)
  pc_cols    <- paste0('pc', 1:10)
  n_pcs      <- 10

  pheno_48 <- aggregate(
    pheno_df[, c(gene_names, pc_cols)],
    by  = list(subject.name = pheno_df$subject.name),
    FUN = mean, na.rm = TRUE
  )
  pheno_48$batch     <- tapply(pheno_df$batch, pheno_df$subject.name,
                               `[`, 1)[pheno_48$subject.name]
  rownames(pheno_48) <- pheno_48$subject.name
  cat('[data] pheno_48:', nrow(pheno_48), 'strains\n\n')
}

# ── name consistency check ────────────────────────────────────────────────────
cat('[name check]\n')
k_chr1    <- kinship_loco[[chromosomes[1]]]
n_overlap <- length(intersect(pheno_48$subject.name, rownames(k_chr1)))
cat('  pheno_48 strains:', nrow(pheno_48), '\n')
cat('  kinship strains:', nrow(k_chr1), '\n')
cat('  overlap:', n_overlap, '/ 48\n')
if (n_overlap < 48)
  cat('  [warning] only', n_overlap, 'strains overlap — scans may fail\n')
cat('\n')

# ── cis-eqtl scan ─────────────────────────────────────────────────────────────
cis_window_mb <- 0.5
results_list  <- list()
effects_list  <- list()
h2_list       <- list()

for (gene in gene_names) {
  for (scan_chr in chromosomes) {
    chr_pmap  <- pmap[pmap$chr == scan_chr, ]
    gene_tss  <- mean(chr_pmap$pos_mb)   # midpoint as proxy (sim) or real tss
    cis_loci  <- chr_pmap$locus[abs(chr_pmap$pos_mb - gene_tss) <= cis_window_mb]

    if (length(cis_loci) == 0) next

    k_chr         <- kinship_loco[[scan_chr]]
    scan_formula  <- as.formula(
      paste0('`', gene, '` ~ batch + ', paste(paste0('pc', 1:n_pcs), collapse = ' + '))
    )

    cat(sprintf('  scanning: %-20s chr%s  %d loci\n', gene, scan_chr, length(cis_loci)))

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
        # print full message — no silent failures during validation
        message(sprintf('  [error] %s chr%s: %s', gene, scan_chr, conditionMessage(e)))
        NULL
      }
    )

    if (is.null(scan_result)) next

    pos_mb <- pmap[scan_result$loci, 'pos_mb']
    key    <- paste(gene, scan_chr, sep = '_')

    results_list[[key]] <- data.frame(
      gene_id = gene, locus = scan_result$loci, chr = scan_chr,
      pos_mb = pos_mb, lod = scan_result$LOD, p_value = scan_result$p.value,
      stringsAsFactors = FALSE
    )

    if (!is.null(scan_result$allele.effects)) {
      peak_idx <- which.max(scan_result$LOD)
      effects  <- scan_result$allele.effects[peak_idx, ]
      effects_list[[key]] <- data.frame(
        gene_id = gene, chr = scan_chr,
        locus   = scan_result$loci[peak_idx],
        founder = names(effects), effect = as.numeric(effects),
        stringsAsFactors = FALSE
      )
    }
  }
}

# ── combine and save ───────────────────────────────────────────────────────────
if (length(results_list) == 0)
  stop('[error] no scan results — check errors above')

results_all <- do.call(rbind, results_list)
rownames(results_all) <- NULL

lead_per_gene <- results_all %>%
  group_by(gene_id) %>%
  slice_min(p_value, n = 1, with_ties = FALSE) %>%
  ungroup() %>% as.data.frame()

write.csv(results_all,  file.path(output_dir, 'scan_results.csv'),  row.names = FALSE)
write.csv(lead_per_gene, file.path(output_dir, 'lead_per_gene.csv'), row.names = FALSE)
if (length(effects_list) > 0)
  write.csv(do.call(rbind, effects_list),
            file.path(output_dir, 'allele_effects.csv'), row.names = FALSE)

cat('\n[results]\n')
for (i in seq_len(nrow(lead_per_gene))) {
  r <- lead_per_gene[i, ]
  cat(sprintf('  %-20s chr%-3s pos=%6.2f Mb  lod=%5.2f  p=%s\n',
              r$gene_id, r$chr, r$pos_mb, r$lod,
              format(r$p_value, scientific = TRUE, digits = 3)))
}
cat('\n[done] outputs saved to:', output_dir, '\n')
