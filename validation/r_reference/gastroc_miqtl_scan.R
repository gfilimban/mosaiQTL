# gastroc_miqtl_scan.R — full miQTL workflow on gastroc whole-tissue pseudobulk
#
# prerequisite: run validation/convert_gastroc_h5ad.py first to produce
#   validation/gastroc_data/counts_lognorm.h5
#   validation/gastroc_data/metadata.csv
#
# sections:
#   1.  cache info              → cache_info.csv, founders.csv, sample_ids.csv
#   2.  haplotype probs         → haplotype_probs.csv
#   3.  LOCO kinship summary    → kinship_summary.csv, kinship_chr8_elements.csv
#                                 plots/kinship_chr8.png
#   S1–S3 combined timing       → timing_setup.csv
#   4.  LMM scan (ROP, LOCO)    → lod_genome.csv, allele_effects_chr8.csv
#                                 timing_chr.csv, timing_genome.csv
#                                 plots/lod_profile_<gene>.png
#                                 plots/allele_effects_<gene>.png
#                                 plots/prob_heatmap_<gene>.png
#   5.  permutation threshold   → perm_max_lods.csv, thresholds.csv
#   6.  cache collapse          → cache_collapse.csv
#   7.  MI scan (top gene)      → mi_lod.csv
#   8.  SNP scan (gated)        → snp_lod.csv
#   9.  mediation               → mediation.csv
#  10.  bootstrap CI            → ci.csv, ci_summary.csv
#  11.  instability scan        → instability.csv
#
# benchmarking:
#   per-chromosome: timing_chr.csv  (elapsed per chr × all genes)
#   genome-wide:    timing_genome.csv (total elapsed × all genes × all chr)
#
# usage:
#   Rscript validation/r_reference/gastroc_miqtl_scan.R
#   Rscript validation/r_reference/gastroc_miqtl_scan.R 1,2,3,4
# ─────────────────────────────────────────────────────────────────────────────

lib_path <- Sys.getenv('R_LIB_PATH', unset = '')
if (nchar(lib_path) > 0 && dir.exists(lib_path)) .libPaths(c(lib_path, .libPaths()))

for (pkg in c('miqtl', 'rhdf5', 'evir')) library(pkg, character.only = TRUE)

# ── crossprod fix ─────────────────────────────────────────────────────────────
if (!is.primitive(crossprod) || typeof(crossprod) != 'builtin') {
  assign('crossprod',  base::crossprod,  envir = globalenv())
  assign('tcrossprod', base::tcrossprod, envir = globalenv())
}

# ── RINT (Blom formula) ───────────────────────────────────────────────────────
rint_blom <- function(x) {
  n     <- sum(!is.na(x))
  ranks <- rank(x, na.last = 'keep', ties.method = 'average')
  qnorm((ranks - 0.375) / (n + 0.25))
}

# ── args ──────────────────────────────────────────────────────────────────────
# usage (local):  Rscript gastroc_miqtl_scan.R [sections] [chr] [scan_type]
# usage (HPC):    Rscript gastroc_miqtl_scan.R 4 8 C   # S4, chr 8, cis scan
# scan_type:
#   C        — cis: scan each gene only against its own chromosome (default)
#   G        — genome-wide: scan all genes against every chromosome
#   L[:Mb]   — local: scan each gene against loci within ±Mb of its TSS (default ±1 Mb)
#              e.g. L:2 = ±2 Mb window
args         <- commandArgs(trailingOnly = TRUE)
run_sections <- if (length(args) >= 1 && nchar(args[1]) > 0 && args[1] != 'all') {
                  as.integer(strsplit(args[1], ',')[[1]])
                } else c(1:5, 7:11)
chr_arg      <- if (length(args) >= 2 && nchar(args[2]) > 0) args[2] else NULL
hpc_mode     <- !is.null(chr_arg)

# parse scan type (arg 3): C / G / L[:window_mb]
scan_type_raw  <- if (length(args) >= 3 && nchar(args[3]) > 0) toupper(args[3]) else 'C'
scan_window_mb <- 1.0
if (startsWith(scan_type_raw, 'L')) {
  scan_type <- 'L'
  parts <- strsplit(scan_type_raw, ':')[[1]]
  if (length(parts) >= 2) scan_window_mb <- as.numeric(parts[2])
} else {
  scan_type <- scan_type_raw   # 'C' or 'G'
}
cat(sprintf('[args] sections=%s  chr=%s  scan=%s%s\n',
            paste(run_sections, collapse=','),
            if (is.null(chr_arg)) 'all' else chr_arg,
            scan_type,
            if (scan_type == 'L') sprintf(' (±%.1f Mb)', scan_window_mb) else ''))

run <- function(s) s %in% run_sections

# Peak R heap during a timed section: gc(reset=TRUE) before the section resets
# R's internal max-used counter; peak_heap_mb() after reads it back.
# Column layout of gc() is stable: 'max used' followed immediately by '(Mb)'.
peak_heap_mb <- function() {
  g  <- gc(verbose = FALSE)
  cn <- colnames(g)
  sum(g[, which(cn == 'max used') + 1L])
}

# Current process RSS in MB (macOS/Linux).  Returns NA on error.
get_rss_mb <- function() {
  tryCatch({
    line <- system(sprintf('ps -o rss= -p %d', Sys.getpid()), intern = TRUE)
    round(as.numeric(trimws(line)) / 1024, 1)
  }, error = function(e) NA_real_)
}

# Append one timing row to the shared list (used by S1–S3).
record_timing <- function(lst, section, component, n_items,
                          t_sys, heap_mb, rss_mb) {
  lst[[section]] <- data.frame(
    section      = section,
    component    = component,
    n_items      = as.integer(n_items),
    elapsed_s    = round(as.numeric(t_sys['elapsed']),    3),
    user_s       = round(as.numeric(t_sys['user.self']),  3),
    sys_s        = round(as.numeric(t_sys['sys.self']),   3),
    peak_heap_mb = round(heap_mb, 1),
    peak_rss_mb  = rss_mb,
    r_version    = paste(R.version$major, R.version$minor, sep = '.'),
    miqtl_version = tryCatch(as.character(packageVersion('miqtl')),
                             error = function(e) NA_character_),
    git_commit   = tryCatch(trimws(system('git rev-parse --short HEAD', intern = TRUE)),
                            error = function(e) NA_character_),
    timestamp    = format(Sys.time(), '%Y-%m-%dT%H:%M:%S'),
    stringsAsFactors = FALSE
  )
  lst
}

# ── paths ─────────────────────────────────────────────────────────────────────
miqtl_dir_local <- file.path(path.expand('~'), 'Documents/journey/learning/graduate',
                              'lab/projects/igvf/analysis/qtl/miqtl')
miqtl_dir_hpc   <- '/share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc'
miqtl_dir       <- if (dir.exists(miqtl_dir_local)) miqtl_dir_local else miqtl_dir_hpc
genomecache      <- file.path(miqtl_dir, 'output/genomecache')
genomecache_orig <- genomecache   # always points to CRSP; staging may reassign genomecache
kinship_path <- file.path(miqtl_dir, 'output/kinship_loco.rds')
allele_dir   <- file.path(miqtl_dir, 'output/allele_dir')
cat(sprintf('[paths] miqtl_dir   = %s\n', miqtl_dir))
cat(sprintf('[paths] genomecache = %s  (exists=%s)\n', genomecache, dir.exists(genomecache)))
cat(sprintf('[paths] kinship     = %s  (exists=%s)\n', kinship_path, file.exists(kinship_path)))

# ── stage chr-specific genomecache to local tmpdir ────────────────────────────
# copies only the chr being scanned + symlinks other chrs back to CRSP.
# eliminates per-locus NFS latency during scan.h2lmm (5,291 loci × n_genes).
if (hpc_mode) {
  local_gc   <- file.path(tempdir(), 'genomecache')
  chr_dir    <- paste0('chr', chr_arg)
  # discover model subdirs (additive, full, etc.) and stage chr_dir within each
  model_dirs <- list.dirs(genomecache, recursive = FALSE, full.names = FALSE)
  staged_key <- file.path(local_gc, model_dirs[1], chr_dir)
  if (!dir.exists(staged_key)) {
    cat(sprintf('[stage] copying %s across model dirs → %s\n', chr_dir, local_gc))
    t0 <- proc.time()[['elapsed']]
    dir.create(local_gc, recursive = TRUE, showWarnings = FALSE)
    # top-level metadata files (must create local_gc first)
    system(sprintf("find '%s' -maxdepth 1 -type f | xargs -I{} cp {} '%s/' 2>/dev/null",
                   genomecache, local_gc))
    for (md in model_dirs) {
      chr_src   <- file.path(genomecache, md, chr_dir)
      chr_dst   <- file.path(local_gc, md, chr_dir)
      dir.create(chr_dst, recursive = TRUE, showWarnings = FALSE)
      if (dir.exists(chr_src)) {
        system(sprintf("cp -r '%s/.' '%s/'", chr_src, chr_dst))
      }
      # symlink all other chr dirs within this model back to CRSP
      other_chrs <- list.dirs(file.path(genomecache, md),
                               recursive = FALSE, full.names = FALSE)
      for (oc in setdiff(other_chrs, chr_dir)) {
        tgt <- file.path(local_gc, md, oc)
        if (!file.exists(tgt))
          file.symlink(file.path(genomecache, md, oc), tgt)
      }
    }
    cat(sprintf('[stage] done in %.0fs\n', proc.time()[['elapsed']] - t0))
  } else {
    cat(sprintf('[stage] using cached %s\n', staged_key))
  }
  genomecache <- local_gc
}

out_dir      <- 'validation/gastroc_data'
r_data_dir   <- file.path(out_dir, 'r_data')   # all R outputs go here
# HPC per-chr jobs write into a chr-specific subdirectory under r_data/
s4_out_dir   <- if (hpc_mode) file.path(r_data_dir, paste0('chr', chr_arg)) else r_data_dir
plot_dir     <- file.path(out_dir, 'plots')
pheno_rds    <- file.path(out_dir, 'gastroc_pheno.rds')
dir.create(r_data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(s4_out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir,   recursive = TRUE, showWarnings = FALSE)
if (hpc_mode) cat(sprintf('[hpc] chr=%s  out=%s\n', chr_arg, s4_out_dir))

# ── load / build phenotype data.frame ────────────────────────────────────────
if (file.exists(pheno_rds)) {
  cat('[load] reading cached', pheno_rds, '\n')
  pheno_df <- readRDS(pheno_rds)
} else {
  h5_path   <- file.path(out_dir, 'counts_lognorm.h5')
  meta_path <- file.path(out_dir, 'metadata.csv')
  stopifnot(file.exists(h5_path), file.exists(meta_path))

  cat('[build] reading', h5_path, '\n')
  counts_mat <- h5read(h5_path, '/counts')      # (n_genes × n_samples) float32
  genes      <- as.character(h5read(h5_path, '/genes'))
  samples    <- as.character(h5read(h5_path, '/samples'))

  # rhdf5 transposes C-order arrays — correct orientation to (n_genes × n_samples)
if (nrow(counts_mat) == length(samples) && ncol(counts_mat) == length(genes)) {
  counts_mat <- t(counts_mat)
}
  
  rownames(counts_mat) <- make.names(genes, unique = TRUE)
  colnames(counts_mat) <- samples
  cat(sprintf('[build] %d genes × %d samples\n', nrow(counts_mat), ncol(counts_mat)))

  meta     <- read.csv(meta_path, stringsAsFactors = FALSE)
  expr_df  <- as.data.frame(t(counts_mat))
  expr_df$sample_id <- samples
  pheno_df <- merge(meta, expr_df, by = 'sample_id', all.x = TRUE)
  saveRDS(pheno_df, pheno_rds)
  cat('[build] saved', pheno_rds, '\n')
}

meta_cols <- c('sample_id', 'genotype', 'sex', 'dataset')
gene_cols <- setdiff(colnames(pheno_df), meta_cols)
n_genes   <- length(gene_cols)
cat(sprintf('[ready] %d samples  %d genes\n', nrow(pheno_df), n_genes))

# focal gene: used for S4 plots, S5 permutation gene, and downstream section scaffolding
focal_gene <- 'Piezo1'
if (!focal_gene %in% gene_cols)
  stop(sprintf('[focal] focal_gene "%s" not found in gene columns', focal_gene))
cat(sprintf('[focal] focal_gene = %s\n', focal_gene))

# Founders are the biological reference group for dataset covariate
pheno_df$dataset <- relevel(factor(pheno_df$dataset), ref = 'Founder')

# ── gene annotations: chr per gene (cis filter) ───────────────────────────────
gene_info_path <- file.path(out_dir, 'gene_info.csv')
if (file.exists(gene_info_path)) {
  gene_info  <- read.csv(gene_info_path, stringsAsFactors = FALSE)
  gene_info$gene_col <- make.names(gene_info$gene_id, unique = TRUE)
  gene_chr_map <- setNames(as.character(gene_info$chr), gene_info$gene_col)
  # TSS in Mb for L scan (use start column; convert from bp)
  gene_tss_mb  <- if ('start' %in% colnames(gene_info)) {
    setNames(as.numeric(gene_info$start) / 1e6, gene_info$gene_col)
  } else NULL
  cat(sprintf('[gene_info] %d genes  chr annotated=%d  TSS annotated=%d\n',
              nrow(gene_info),
              sum(!is.na(gene_chr_map)),
              if (!is.null(gene_tss_mb)) sum(!is.na(gene_tss_mb)) else 0L))
  if (scan_type == 'L' && is.null(gene_tss_mb))
    stop('[gene_info] L scan requires start column in gene_info.csv — rerun convert_gastroc_h5ad.py')
} else {
  gene_chr_map <- NULL
  gene_tss_mb  <- NULL
  if (scan_type != 'G')
    cat('[gene_info] gene_info.csv not found — falling back to G scan (all genes × all chr)\n')
  scan_type <- 'G'
}

# ── canonical strain order: Founders → F1s → CC (alphanumeric) ───────────────
FOUNDER_ORDER <- c('AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ')
F1_ORDER      <- c('B6AF1J', 'B6129S1F1J', 'B6CASTF1J', 'B6NODF1J',
                   'B6NZOF1J', 'B6PWKF1J', 'B6WSBF1J')
NON_CC_ORDER  <- c(FOUNDER_ORDER, F1_ORDER)

# returns integer index vector that sorts `strains` into canonical order
canonical_order <- function(strains) {
  non_cc_rank <- match(strains, NON_CC_ORDER)          # NA for CC strains
  cc_rank     <- length(NON_CC_ORDER) +
                 rank(ifelse(is.na(non_cc_rank), strains, ''),
                      ties.method = 'first') *
                 as.integer(is.na(non_cc_rank))
  order(ifelse(!is.na(non_cc_rank), non_cc_rank, cc_rank))
}

# strain-level mean phenotype (for prob.heatmap which works at genotype resolution)
strain_means <- aggregate(pheno_df[, gene_cols],
                          by = list(genotype = pheno_df$genotype),
                          FUN = mean, na.rm = TRUE)
# sort into canonical order so prob.heatmap y-axis matches Python
strain_means <- strain_means[canonical_order(strain_means$genotype), ]
rownames(strain_means) <- NULL

# RINT (Blom) at each data level independently.
# Strain-level scans (S5, S10, S11) receive RINT(mean(raw)) — biologically
# correct: normalize the genotypic value distribution across strains.
# Sample-level scans (S4, S7, S8, S9) receive RINT(sample-level) because
# miQTL reads phenotype from pheno_df directly and reduces internally.
for (g in gene_cols) {
  pheno_df[[g]]     <- rint_blom(pheno_df[[g]])
  strain_means[[g]] <- rint_blom(strain_means[[g]])
}
cat(sprintf('[rint] Blom RINT applied to %d gene columns\n', length(gene_cols)))

# ── LOCO kinship ──────────────────────────────────────────────────────────────
cat('[load] reading', kinship_path, '\n')
kinship_loco <- readRDS(kinship_path)
all_chrs     <- names(kinship_loco)
cat(sprintf('[load] %d chromosomes  kinship %d×%d\n',
            length(all_chrs), nrow(kinship_loco[[1]]), ncol(kinship_loco[[1]])))

# local runs default to chr 19 (one chr, fast); HPC uses the per-chr arg
local_default_chr <- '19'
target_chrs <- if (hpc_mode) chr_arg else local_default_chr
cat(sprintf('[target] scan chromosomes: %s\n', paste(target_chrs, collapse = ',')))

# Sort pheno_df rows by genotype so make.processed.data's include.subjects order
# (unique first-appearance in data) equals alphabetical, matching model.matrix Z columns.
# Without this, Z @ K @ Z^T scrambles the kinship assignment when pheno.id != geno.id.
pheno_df <- pheno_df[order(pheno_df$genotype), ]
cat('[fix] pheno_df sorted by genotype; first-appearance order now alphabetical\n')

# ─────────────────────────────────────────────────────────────────────────────
# S1 — cache info
# ─────────────────────────────────────────────────────────────────────────────
setup_timing <- list()

if (run(1)) {
  cat('\n[S1] cache info\n')
  gc(reset = TRUE, verbose = FALSE)
  t_s1 <- system.time({
    rd         <- DiploprobReader$new(genomecache_orig)
    founders   <- rd$getFounders()
    sample_ids <- rd$getSubjects()
    chrs       <- rd$getChromList()

    chr_info <- do.call(rbind, lapply(chrs, function(ch) {
      loci <- rd$getLoci(ch)
      mb   <- rd$getLocusStart(loci, scale = 'Mb')
      data.frame(chr = ch, n_loci = length(loci),
                 min_mb = min(mb), max_mb = max(mb), stringsAsFactors = FALSE)
    }))
    write.csv(chr_info,                          file.path(out_dir, 'cache_info.csv'),  row.names = FALSE)
    write.csv(data.frame(founder  = founders),   file.path(out_dir, 'founders.csv'),    row.names = FALSE)
    write.csv(data.frame(genotype = sample_ids[canonical_order(sample_ids)]),
              file.path(out_dir, 'genotype_ids.csv'), row.names = FALSE)
    cat(sprintf('  %d chromosomes  %d founders  %d genotypes\n',
                nrow(chr_info), length(founders), length(sample_ids)))
  })
  heap_s1 <- peak_heap_mb()
  rss_s1  <- get_rss_mb()
  setup_timing <- record_timing(setup_timing, 'S1', 'cache_info',
                                nrow(chr_info), t_s1, heap_s1, rss_s1)
  cat(sprintf('  [timing] %.2fs  heap=%.0fMB  rss=%.0fMB\n',
              t_s1['elapsed'], heap_s1, rss_s1))
}

# ─────────────────────────────────────────────────────────────────────────────
# S2 — haplotype probabilities (5 evenly spaced loci on target chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run(2)) {
  s2_chr <- target_chrs[1]
  cat(sprintf('\n[S2] haplotype probs  chr%s (5 loci)\n', s2_chr))
  gc(reset = TRUE, verbose = FALSE)
  t_s2 <- system.time({
    rd    <- DiploprobReader$new(genomecache_orig)
    loci8 <- rd$getLoci(s2_chr)
    loci8 <- loci8[round(seq(1, length(loci8), length.out = 5))]

    hap_rows <- do.call(rbind, lapply(loci8, function(loc) {
      mat <- rd$getLocusMatrix(loc, model = 'additive')   # (n_strains × 8)
      df  <- as.data.frame(mat)
      colnames(df) <- rd$getFounders()
      df$locus    <- loc
      df$genotype <- rownames(mat)
      df[canonical_order(df$genotype), ]   # canonical strain order within each locus
    }))
    write.csv(hap_rows, file.path(out_dir, 'haplotype_probs.csv'), row.names = FALSE)
    cat(sprintf('  exported %d rows\n', nrow(hap_rows)))
  })
  heap_s2 <- peak_heap_mb()
  rss_s2  <- get_rss_mb()
  setup_timing <- record_timing(setup_timing, 'S2', 'haplotype_probe',
                                nrow(hap_rows), t_s2, heap_s2, rss_s2)
  cat(sprintf('  [timing] %.2fs  heap=%.0fMB  rss=%.0fMB\n',
              t_s2['elapsed'], heap_s2, rss_s2))
}

# ─────────────────────────────────────────────────────────────────────────────
# S3 — LOCO kinship summary + heatmap
# ─────────────────────────────────────────────────────────────────────────────
if (run(3)) {
  cat('\n[S3] kinship summary\n')
  gc(reset = TRUE, verbose = FALSE)
  t_s3 <- system.time({
  k_summary <- do.call(rbind, lapply(all_chrs, function(ch) {
    K <- kinship_loco[[ch]]
    data.frame(chr          = ch,
               mean_diag    = mean(diag(K)),
               mean_offdiag = mean(K[lower.tri(K)]),
               stringsAsFactors = FALSE)
  }))
  write.csv(k_summary, file.path(out_dir, 'kinship_summary.csv'), row.names = FALSE)

  # ── derive K_genome and K^(c) from LOCO matrices ──────────────────────────
  cache_info_path <- file.path(out_dir, 'cache_info.csv')
  stopifnot(file.exists(cache_info_path))   # requires S1 to have run first
  cache_info <- read.csv(cache_info_path, stringsAsFactors = FALSE)
  cache_info$chr <- as.character(cache_info$chr)
  L      <- sum(cache_info$n_loci)
  n_chr  <- nrow(cache_info)

  # K_genome = sum_c[ K^(-c) * (L - L_c) ] / ( (n_chr - 1) * L )
  K_genome <- Reduce('+', lapply(cache_info$chr, function(ch) {
    L_c <- cache_info$n_loci[cache_info$chr == ch]
    kinship_loco[[ch]] * (L - L_c)
  })) / ((n_chr - 1) * L)

  plot_kinship_chr <- function(K, chr_label, title, filename) {
    ord   <- canonical_order(rownames(K))
    K     <- K[ord, ord]
    n_str <- nrow(K)
    side  <- max(1800L, as.integer(n_str * 36L))
    pal   <- colorRampPalette(c('#2166ac', 'white', '#d6604d'))(100)
    cb_w  <- as.integer(side * 0.14)
    png(file.path(plot_dir, filename), width = side + cb_w, height = side, res = 300)
    layout(matrix(c(1L, 2L), nrow = 1L), widths = c(side, cb_w))
    par(mar = c(7, 7, 3, 0.5))
    image(K[, n_str:1], main = title, axes = FALSE, col = pal)
    axis(1, at = seq(0, 1, length.out = n_str), labels = rownames(K),      las = 2, cex.axis = 0.35)
    axis(2, at = seq(0, 1, length.out = n_str), labels = rev(rownames(K)), las = 2, cex.axis = 0.35)
    lh      <- par('cin')[2]
    cb_vmar <- 3 * (side / 300) / (8 * lh)
    par(mar = c(cb_vmar, 0.3, cb_vmar, 3))
    image(matrix(seq(0, 1, length.out = 100L), nrow = 1L), col = pal, axes = FALSE)
    axis(4, at = c(0, 0.5, 1), labels = c('0', '0.5', '1'), las = 2, cex.axis = 0.6)
    mtext('kinship', side = 4, line = 2, cex = 0.7)
    dev.off()
  }

  # K^(c) = ( K_genome * L - K^(-c) * (L - L_c) ) / L_c
  # K^(c)_ij = (1/L_c) sum_{l in c} sum_f H_ilf * H_jlf
  # measures haplotype sharing between strains attributable to chr c loci only
  for (k_chr in names(kinship_loco)) {
    L_c   <- cache_info$n_loci[cache_info$chr == k_chr]
    K_c   <- (K_genome * L - kinship_loco[[k_chr]] * (L - L_c)) / L_c
    title <- bquote(K^{(c)} ~ '— chr' ~ .(k_chr))
    plot_kinship_chr(K_c, k_chr, title,
                     sprintf('kinship_chr%s_contribution.png', k_chr))
  }
  # ── target-chr full K matrix elements (for Python element-level scatter) ────
  k_elem_chr <- target_chrs[1]
  if (k_elem_chr %in% names(kinship_loco)) {
    K8  <- kinship_loco[[k_elem_chr]]
    ord <- canonical_order(rownames(K8))
    K8  <- K8[ord, ord]
    strain_names <- rownames(K8)
    n_str <- nrow(K8)
    elem_df <- data.frame(
      strain_i = rep(strain_names, each  = n_str),
      strain_j = rep(strain_names, times = n_str),
      k_value  = as.vector(K8),
      stringsAsFactors = FALSE
    )
    elem_fname <- sprintf('kinship_chr%s_elements.csv', k_elem_chr)
    write.csv(elem_df, file.path(out_dir, elem_fname), row.names = FALSE)
    cat(sprintf('  %s written (%d×%d = %d elements)\n',
                elem_fname, n_str, n_str, nrow(elem_df)))
  }

  cat(sprintf('  %d chromosomes summarised\n', nrow(k_summary)))
  })   # end system.time S3
  heap_s3 <- peak_heap_mb()
  rss_s3  <- get_rss_mb()
  setup_timing <- record_timing(setup_timing, 'S3', 'kinship_summary',
                                nrow(k_summary), t_s3, heap_s3, rss_s3)
  cat(sprintf('  [timing] %.2fs  heap=%.0fMB  rss=%.0fMB\n',
              t_s3['elapsed'], heap_s3, rss_s3))
  write.csv(do.call(rbind, setup_timing),
            file.path(out_dir, 'timing_setup.csv'), row.names = FALSE)
  cat(sprintf('  timing_setup.csv written (%d sections)\n', length(setup_timing)))
}

# ── n_loci per chr (for throughput tracking in S5–S11) ───────────────────────
{
  ci_path   <- file.path(out_dir, 'cache_info.csv')
  chr_nloci <- if (file.exists(ci_path)) {
    ci <- read.csv(ci_path, stringsAsFactors = FALSE)
    setNames(as.integer(ci$n_loci), as.character(ci$chr))
  } else NULL
}

# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM scan (ROP, LOCO) — full genome, all genes, timed per-chr + genome-wide
# ─────────────────────────────────────────────────────────────────────────────
lod_genome_path    <- file.path(out_dir, 'lod_genome.csv')
effects_path       <- file.path(out_dir, sprintf('allele_effects_chr%s.csv',
                                                  paste(target_chrs, collapse = '-')))
timing_chr_path    <- file.path(out_dir, 'timing_chr.csv')
timing_genome_path <- file.path(out_dir, 'timing_genome.csv')

peak_lod_df        <- NULL
chr_scan_results   <- list()

if (run(4)) {
  scan_chrs <- target_chrs
  cat(sprintf('\n[S4] LOCO scan  %d genes × chr %s%s\n',
              n_genes,
              paste(scan_chrs, collapse = ','),
              if (hpc_mode) '  [HPC per-chr mode]' else '  [sequential]'))

  all_lod_rows       <- list()
  all_effect_rows    <- list()
  total_genes_scanned <- 0L   # cumulative cis-genes across all scanned chromosomes

  progress_every <- 25L   # print a progress line every N genes

  gc(reset = TRUE, verbose = FALSE)
  chr_t <- system.time({
    for (ch in scan_chrs) {
      K_ch    <- kinship_loco[[ch]]
      collect_effects <- TRUE   # collect allele effects for every scanned chromosome
      rd_ch_pre  <- DiploprobReader$new(genomecache)
      loci_ch    <- rd_ch_pre$getLoci(ch)
      n_loci_ch  <- length(loci_ch)
      # loci positions in Mb — needed for L scan window filter
      loci_mb_ch <- if (scan_type == 'L') {
        setNames(rd_ch_pre$getLocusStart(loci_ch, scale = 'Mb'), loci_ch)
      } else NULL

      # gene filter per scan type
      chr_genes <- switch(scan_type,
        'G' = gene_cols,
        'C' = , 'L' = {   # both C and L restrict to cis-genes first
          if (!is.null(gene_chr_map))
            gene_cols[!is.na(gene_chr_map[gene_cols]) & gene_chr_map[gene_cols] == ch]
          else gene_cols
        }
      )
      n_genes_ch <- length(chr_genes)
      total_genes_scanned <- total_genes_scanned + n_genes_ch
      gene_i     <- 0L
      chr_start  <- proc.time()[['elapsed']]
      cat(sprintf('  [chr%s]  %d loci  %d %s-genes\n', ch, n_loci_ch, n_genes_ch,
                  switch(scan_type, 'C'='cis', 'G'='genome', 'L'=sprintf('local(±%.1fMb)', scan_window_mb))))
      if (n_genes_ch == 0L && !is.null(gene_chr_map) && scan_type != 'G') {
        ex <- head(names(gene_chr_map)[!is.na(gene_chr_map)], 3)
        gc <- head(gene_cols, 3)
        cat(sprintf('  [warn] 0 genes — gene_chr_map keys (e.g. %s) may not match gene_cols (e.g. %s)\n',
                    paste(ex, collapse=', '), paste(gc, collapse=', ')))
      }

      for (gene in chr_genes) {
        gene_i <- gene_i + 1L
        if (gene_i %% progress_every == 0L || gene_i == 1L) {
          elapsed_so_far <- proc.time()[['elapsed']] - chr_start
          cat(sprintf('    chr%s  gene %d/%d  %s  %.1fs\n',
                      ch, gene_i, n_genes_ch, gene, elapsed_so_far))
        }
        # L scan: restrict loci to ±scan_window_mb of gene TSS
        just_loci <- NULL
        if (scan_type == 'L' && !is.null(loci_mb_ch) && !is.null(gene_tss_mb)) {
          tss <- gene_tss_mb[gene]
          if (!is.na(tss)) {
            in_win <- abs(loci_mb_ch - tss) <= scan_window_mb
            just_loci <- loci_ch[in_win]
          }
          if (length(just_loci) == 0L) { gene_i <- gene_i + 1L; next }
        }

        f   <- as.formula(paste(gene, '~ sex + dataset'))
        res <- scan.h2lmm(
          genomecache           = genomecache,
          data                  = pheno_df,
          pheno.id              = 'sample_id',
          geno.id               = 'genotype',
          formula               = f,
          K                     = K_ch,
          use.multi.impute      = FALSE,
          return.allele.effects = collect_effects,
          chr                   = ch,
          just.these.loci       = just_loci
        )

        if (collect_effects) chr_scan_results[[gene]] <- res

        lod_vec <- res$LOD
        all_lod_rows[[length(all_lod_rows) + 1]] <- data.frame(
          gene   = gene,
          chr    = ch,
          locus  = names(lod_vec),
          pos_mb = res$pos[['Mb']],
          lod    = as.numeric(lod_vec),
          stringsAsFactors = FALSE
        )

        if (collect_effects && !is.null(res$allele.effects)) {
          eff_mat  <- res$allele.effects          # [8 founders × n_loci]
          founders <- rownames(eff_mat)
          loci_ae  <- colnames(eff_mat)
          pos_vec  <- setNames(res$pos[['Mb']], names(res$LOD))
          all_effect_rows[[length(all_effect_rows) + 1]] <- data.frame(
            gene    = gene,
            locus   = rep(loci_ae,    each  = length(founders)),
            pos_mb  = rep(pos_vec[loci_ae], each  = length(founders)),
            founder = rep(founders,   times = length(loci_ae)),
            effect  = as.vector(eff_mat),
            stringsAsFactors = FALSE
          )
        }
      }
      cat(sprintf('  chr%s  done  %d loci  %.1fs total\n', ch, n_loci_ch,
                  proc.time()[['elapsed']] - chr_start))
    }
  })

  lod_df     <- do.call(rbind, all_lod_rows)
  effects_df <- do.call(rbind, all_effect_rows)

  # ── per-chromosome timing row (one row per chr scanned by this job) ──────
  peak_mb_s4  <- peak_heap_mb()   # R-heap peak during S4 scan
  rd_s4      <- DiploprobReader$new(genomecache)
  peak_rss_mb <- tryCatch({
    pid  <- Sys.getpid()
    line <- system(sprintf('ps -o rss= -p %d', pid), intern = TRUE)
    round(as.numeric(trimws(line)) / 1024, 1)
  }, error = function(e) NA_real_)

  timing_row <- data.frame(
    chr           = paste(scan_chrs, collapse = ','),
    n_samples     = nrow(pheno_df),
    n_genes       = total_genes_scanned,
    n_loci        = sum(sapply(scan_chrs, function(ch) length(rd_s4$getLoci(ch)))),
    scan_type     = scan_type,
    model         = 'additive_rop',
    elapsed_s     = as.numeric(chr_t['elapsed']),
    user_s        = as.numeric(chr_t['user.self']),
    sys_s         = as.numeric(chr_t['sys.self']),
    genes_per_min = round(total_genes_scanned / chr_t['elapsed'] * 60, 1),
    peak_heap_mb  = peak_mb_s4,
    peak_rss_mb   = peak_rss_mb,
    mode          = if (hpc_mode) 'hpc_per_chr' else 'sequential',
    r_version     = paste(R.version$major, R.version$minor, sep = '.'),
    miqtl_version = as.character(packageVersion('miqtl')),
    git_commit    = tryCatch(
                      trimws(system('git rev-parse --short HEAD', intern = TRUE)),
                      error = function(e) NA_character_),
    timestamp     = format(Sys.time(), '%Y-%m-%dT%H:%M:%S'),
    row.names     = NULL
  )

  # in HPC mode write to chr subdir; merge_chr_outputs.R stacks and computes
  # genome-wide wall-clock = max(elapsed_s) and total_cpu = sum(elapsed_s)
  write.csv(lod_df,     file.path(s4_out_dir, 'lod.csv'),     row.names = FALSE)
  write.csv(effects_df, file.path(s4_out_dir, 'allele_effects.csv'), row.names = FALSE)
  write.csv(timing_row, file.path(s4_out_dir, 'timing.csv'),  row.names = FALSE)

  if (!hpc_mode) {
    # sequential: also write to top-level paths used by S5-S11
    write.csv(lod_df,     lod_genome_path, row.names = FALSE)
    write.csv(effects_df, effects_path,    row.names = FALSE)
    write.csv(timing_row, timing_chr_path, row.names = FALSE)
    # genome-wide timing is identical to chr timing when sequential
    timing_genome <- data.frame(
      n_samples     = nrow(pheno_df),
      n_genes       = total_genes_scanned,
      n_chrs        = length(all_chrs),
      scan_type     = scan_type,
      model         = 'additive_rop',
      wall_clock_s  = as.numeric(chr_t['elapsed']),
      total_cpu_s   = as.numeric(chr_t['elapsed']),
      genes_per_min = round(total_genes_scanned / chr_t['elapsed'] * 60, 1),
      peak_heap_mb  = peak_mb_s4,
      peak_rss_mb   = peak_rss_mb,
      mode          = 'sequential',
      r_version     = paste(R.version$major, R.version$minor, sep = '.'),
      miqtl_version = as.character(packageVersion('miqtl')),
      git_commit    = timing_row$git_commit,
      timestamp     = timing_row$timestamp,
      row.names     = NULL
    )
    write.csv(timing_genome, timing_genome_path, row.names = FALSE)
    cat(sprintf('\n  sequential: %.1fs  %.1f genes/min\n',
                chr_t['elapsed'], timing_genome$genes_per_min))
  } else {
    cat(sprintf('\n  chr%s done: %.1fs  %.1f genes/min\n',
                chr_arg, chr_t['elapsed'], timing_row$genes_per_min))
    cat('  genome-wide wall-clock computed by merge_chr_outputs.R\n')
  }

  lod_genome  <- lod_df
  if (nrow(lod_genome) > 0L) {
    peak_lod_df <- do.call(rbind, lapply(unique(lod_genome$gene), function(g) {
      sub <- lod_genome[lod_genome$gene == g, ]
      sub[which.max(sub$lod), ]
    }))
    peak_lod_df <- peak_lod_df[order(-peak_lod_df$lod), ]
  } else {
    peak_lod_df <- NULL
    cat('  [warn] no LOD rows — check cis filter\n')
  }
  if (!hpc_mode)
    cat(sprintf('  top hit: %s  chr%s  LOD=%.2f\n',
                peak_lod_df$gene[1], peak_lod_df$chr[1], peak_lod_df$lod[1]))

  # ── plots: sequential mode only (HPC jobs write lod.csv then exit here) ──────
  if (!hpc_mode) {
    top_gene <- if (focal_gene %in% peak_lod_df$gene) focal_gene else peak_lod_df$gene[1]
    cat(sprintf('\n[S4] building scan object for %s (chr %s)\n', top_gene, paste(scan_chrs, collapse=',')))
    gene_chr_scans <- list()
    for (ch in scan_chrs) {
      f_top  <- as.formula(paste(top_gene, '~ sex + dataset'))
      gene_chr_scans[[ch]] <- scan.h2lmm(
        genomecache      = genomecache,
        data             = pheno_df,
        pheno.id         = 'sample_id',
        geno.id          = 'genotype',
        formula          = f_top,
        K                = kinship_loco[[ch]],
        use.multi.impute = FALSE,
        chr              = ch
      )
    }
    combined_lod   <- unlist(lapply(gene_chr_scans, function(r) as.numeric(r$LOD)))
    combined_pval  <- unlist(lapply(gene_chr_scans, function(r) r$p.value))
    combined_chr   <- unlist(lapply(gene_chr_scans, function(r) r$chr))
    combined_mb    <- unlist(lapply(gene_chr_scans, function(r) r$pos$Mb))
    combined_loci  <- unlist(lapply(gene_chr_scans, function(r) names(r$LOD)))
    names(combined_lod) <- combined_loci

    genome_scan_obj <- list(
      LOD     = combined_lod,
      p.value = combined_pval,
      chr     = combined_chr,
      pos     = list(Mb = combined_mb, cM = combined_mb),
      loci    = combined_loci,
      n       = nrow(pheno_df)
    )

    # LOD profile plot
    peak_mb   <- combined_mb[which.max(combined_lod)]
    threshold <- if (file.exists(file.path(out_dir, 'thresholds.csv'))) {
                   read.csv(file.path(out_dir, 'thresholds.csv'))$threshold[1]
                 } else NULL

    png(file.path(plot_dir, sprintf('lod_profile_%s.png', top_gene)),
        width = 3000L, height = 1200L, res = 300L)
    genome.plotter.whole(
      scan.list        = setNames(list(genome_scan_obj), top_gene),
      use.lod          = TRUE,
      main             = top_gene,
      hard.thresholds  = threshold,
      thresholds.col   = 'red',
      mark.locus       = combined_loci[which.max(combined_lod)],
      mark.locus.col   = 'red'
    )
    dev.off()

    # allele effects plot (chr 8 scan)
    if (!is.null(chr_scan_results[[top_gene]]) &&
        !is.null(chr_scan_results[[top_gene]]$allele.effects)) {
      png(file.path(plot_dir, sprintf('allele_effects_%s.png', top_gene)),
          width = 3000L, height = 1200L, res = 300L)
      allele.plotter.whole(scan.object = chr_scan_results[[top_gene]],
                           main        = sprintf('%s — chr %s', top_gene,
                                                 paste(scan_chrs, collapse = ',')))
      dev.off()
    }

    # prob heatmap at peak locus (strain-level mean phenotype)
    peak_locus <- combined_loci[which.max(combined_lod)]
    sm_df      <- strain_means[, c('genotype', top_gene)]
    colnames(sm_df)[2] <- 'pheno'

    png(file.path(plot_dir, sprintf('prob_heatmap_%s.png', top_gene)),
        width = 3000L, height = 1500L, res = 300L)
    prob.heatmap(marker         = peak_locus,
                 genomecache    = genomecache,
                 phenotype      = 'pheno',
                 phenotype.data = sm_df,
                 merge.by       = 'genotype',
                 alternative.phenotype.label = top_gene)
    dev.off()
  }

} else if (file.exists(lod_genome_path)) {
  cat('[S4] loading cached lod_genome.csv\n')
  lod_genome  <- read.csv(lod_genome_path, stringsAsFactors = FALSE)
  peak_lod_df <- do.call(rbind, lapply(unique(lod_genome$gene), function(g) {
    sub <- lod_genome[lod_genome$gene == g, ]
    sub[which.max(sub$lod), ]
  }))
  peak_lod_df <- peak_lod_df[order(-peak_lod_df$lod), ]
} else if (!is.null(chr_arg) && file.exists(file.path(s4_out_dir, 'lod.csv'))) {
  # single-chr restart: S4 wrote chr{N}/lod.csv but not the genome-wide CSV
  cat(sprintf('[S4] loading cached chr%s/lod.csv\n', chr_arg))
  lod_genome  <- read.csv(file.path(s4_out_dir, 'lod.csv'), stringsAsFactors = FALSE)
  peak_lod_df <- do.call(rbind, lapply(unique(lod_genome$gene), function(g) {
    sub <- lod_genome[lod_genome$gene == g, ]
    sub[which.max(sub$lod), ]
  }))
  peak_lod_df <- peak_lod_df[order(-peak_lod_df$lod), ]
}

top_gene <- if (!is.null(peak_lod_df) && focal_gene %in% peak_lod_df$gene) focal_gene else if (!is.null(peak_lod_df)) peak_lod_df$gene[1] else gene_cols[1]
top_chr  <- if (!is.null(peak_lod_df)) as.character(peak_lod_df$chr[1]) else all_chrs[1]

# ─────────────────────────────────────────────────────────────────────────────
# S5 — permutation threshold: one per chromosome, using that chr's top gene
# ─────────────────────────────────────────────────────────────────────────────
thresholds_chr_path <- file.path(out_dir, 'thresholds_chr.csv')
n_genes_s5 <- NA_integer_; n_loci_s5 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s5 <- proc.time()

if (run(5)) {
  stopifnot(!is.null(peak_lod_df))
  cat(sprintf('\n[S5] permutation thresholds — all chromosomes (top gene per chr)\n'))

  thr_rows <- list()
  for (ch in target_chrs) {
    # try genes in descending LOD order until one produces a valid perm run
    sub_ch <- peak_lod_df[as.character(peak_lod_df$chr) == ch, ]
    if (nrow(sub_ch) == 0L) next
    sub_ch <- sub_ch[order(-sub_ch$lod), ]

    perm_max_lods <- NULL
    thr_gene      <- NULL
    for (gi in seq_len(nrow(sub_ch))) {
      cand_gene <- sub_ch$gene[gi]
      n_obs <- sum(!is.na(pheno_df[[cand_gene]]))
      if (n_obs < 20L) next   # too few observations for a stable null fit

      # Permutation at strain-mean level using scan.h2lmm with pheno.id='genotype'.
      # miQTL scans at m=48 strain-mean resolution; LODs on m-scale match the S4
      # scan which also uses K (m×m) with strain-mean phenotype internally.
      # miQTL's helper functions crash when passed the expanded K; direct loop used.
      K_ch <- kinship_loco[[ch]]
      # PSD guard on m×m K_ch
      K_eigs    <- eigen(K_ch, symmetric = TRUE, only.values = TRUE)$values
      K_min_eig <- min(K_eigs)
      K_psd_tol <- max(abs(K_eigs)) * nrow(K_ch)^2 * .Machine$double.eps
      if (K_min_eig < -K_psd_tol) {
        cat(sprintf('  chr%s  gene=%s  [skip] near-singular LOCO kinship (min_eig=%.2e)\n',
                    ch, cand_gene, K_min_eig))
        next
      }

      n_perm_s5      <- 250L
      perm_max_lods_ <- numeric(n_perm_s5)
      y_gene         <- strain_means[[cand_gene]]   # RINT(mean(raw)) — strain-level
      set.seed(1)
      for (pi in seq_len(n_perm_s5)) {
        pheno_perm              <- strain_means
        pheno_perm[[cand_gene]] <- sample(y_gene)
        scan_perm <- scan.h2lmm(
          genomecache           = genomecache,
          data                  = pheno_perm,
          pheno.id              = 'genotype',
          geno.id               = 'genotype',
          formula               = as.formula(paste(cand_gene, '~ 1')),
          K                     = K_ch,
          use.multi.impute      = FALSE,
          return.allele.effects = FALSE,
          chr                   = ch,
          use.fix.par           = TRUE
        )
        perm_max_lods_[pi] <- max(scan_perm$LOD, na.rm = TRUE)
        if (pi %% 50L == 0L)
          cat(sprintf('    perm %d/%d\n', pi, n_perm_s5))
      }
      perm_max_lods <- perm_max_lods_
      thr_gene      <- cand_gene
      break
    }

    if (is.null(thr_gene)) {
      cat(sprintf('  chr%s  [warn] no valid gene found for permutation — skipping\n', ch))
      next
    }

    cat(sprintf('  chr%s  gene=%s\n', ch, thr_gene))
    # get.gev.thresholds uses evir::qgev(1-percentile) → lower 5th percentile (wrong
    # direction for LOD thresholds). Use evir directly: qgev(0.95) = upper 95th pct,
    # matching Python's genextreme.ppf(0.95).
    max_lods_ch   <- perm_max_lods[is.finite(perm_max_lods)]
    gev_fit_ch    <- evir::gev(max_lods_ch)
    threshold     <- evir::qgev(0.95, xi    = gev_fit_ch$par.est[['xi']],
                                      sigma = gev_fit_ch$par.est[['sigma']],
                                      mu    = gev_fit_ch$par.est[['mu']])
    empirical_p95 <- as.numeric(quantile(perm_max_lods, 0.95, na.rm = TRUE))

    write.csv(data.frame(perm_index = seq_along(perm_max_lods),
                         max_lod    = perm_max_lods),
              file.path(out_dir, sprintf('perm_max_lods_chr%s.csv', ch)), row.names = FALSE)
    thr_rows[[ch]] <- data.frame(chr = ch, gene = thr_gene,
                                 threshold = threshold, empirical_p95 = empirical_p95,
                                 stringsAsFactors = FALSE)
    cat(sprintf('    GEV=%.3f  empirical=%.3f\n', threshold, empirical_p95))
  }
  thr_df <- do.call(rbind, thr_rows)
  write.csv(thr_df, thresholds_chr_path, row.names = FALSE)
  cat(sprintf('  thresholds written for %d chromosomes\n', nrow(thr_df)))
  n_genes_s5 <- nrow(thr_df)
  n_loci_s5  <- if (!is.null(chr_nloci)) sum(chr_nloci[as.character(thr_df$chr)], na.rm = TRUE) else NA_integer_
}

# ── significant genes: peak LOD > per-chr threshold ──────────────────────────
sig_genes_df <- NULL
if (!is.null(peak_lod_df) && file.exists(thresholds_chr_path)) {
  thr_df  <- read.csv(thresholds_chr_path, stringsAsFactors = FALSE)
  thr_map <- setNames(thr_df$threshold, as.character(thr_df$chr))
  sig_genes_df <- peak_lod_df[
    mapply(function(lod, ch) {
      thr <- thr_map[as.character(ch)]
      !is.na(thr) && lod > thr
    }, peak_lod_df$lod, peak_lod_df$chr), ]
  cat(sprintf('[sig] %d / %d genes pass per-chr threshold\n',
              nrow(sig_genes_df), nrow(peak_lod_df)))
  write.csv(sig_genes_df, file.path(out_dir, 'egenes.csv'), row.names = FALSE)
  cat(sprintf('[sig] egenes.csv written (%d rows)\n', nrow(sig_genes_df)))
} else if (!is.null(peak_lod_df)) {
  sig_genes_df <- peak_lod_df   # no threshold yet — use all
}

# ─────────────────────────────────────────────────────────────────────────────
# S6 — cache collapse (genome-wide, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s5 <- peak_heap_mb()
rss_mb_s5  <- get_rss_mb()
n_genes_s6 <- NA_integer_; n_loci_s6 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s6 <- proc.time()
if (run(6)) {
  cat('\n[S6] cache collapse\n')
  collapsed_dir <- file.path(out_dir, 'genomecache_collapsed')
  if (!dir.exists(collapsed_dir)) {
    collapse.genomecache(
      genomecache  = genomecache,
      output.cache = collapsed_dir,
      criterion    = 'l2.norm',
      tol          = 0.02
    )
  }
  rd_full <- DiploprobReader$new(genomecache)
  rd_coll <- DiploprobReader$new(collapsed_dir)
  coll_summary <- do.call(rbind, lapply(rd_full$getChromList(), function(ch) {
    n_full <- length(rd_full$getLoci(ch))
    n_coll <- length(rd_coll$getLoci(ch))
    data.frame(chr = ch, n_loci_full = n_full, n_loci_collapsed = n_coll,
               pct_removed = round(100 * (1 - n_coll / n_full), 1),
               stringsAsFactors = FALSE)
  }))
  write.csv(coll_summary, file.path(out_dir, 'cache_collapse.csv'), row.names = FALSE)
  cat(sprintf('  %.1f%% loci removed overall\n',
              100 * (1 - sum(coll_summary$n_loci_collapsed) / sum(coll_summary$n_loci_full))))
  n_loci_s6 <- sum(coll_summary$n_loci_full)
}

# ─────────────────────────────────────────────────────────────────────────────
# S7 — MI scan: all significant genes on their peak chromosome
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s6 <- peak_heap_mb()
rss_mb_s6  <- get_rss_mb()
n_genes_s7 <- NA_integer_; n_loci_s7 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s7 <- proc.time()
if (run(7)) {
  stopifnot(!is.null(sig_genes_df))
  cat(sprintf('\n[S7] MI scan — %d significant genes\n', nrow(sig_genes_df)))
  set.seed(1)   # matches Python scan_h2lmm_mi(seed=1)
  mi_rows <- list()
  for (i in seq_len(nrow(sig_genes_df))) {
    gene <- sig_genes_df$gene[i]
    ch   <- as.character(sig_genes_df$chr[i])
    cat(sprintf('  [%d/%d] %s  chr%s\n', i, nrow(sig_genes_df), gene, ch))
    mi_res <- scan.h2lmm(
      genomecache      = genomecache, data = pheno_df,
      pheno.id         = 'sample_id', geno.id = 'genotype',
      formula          = as.formula(paste(gene, '~ sex + dataset')),
      K                = kinship_loco[[ch]],
      use.multi.impute = TRUE, num.imp = 11, chr = ch
    )
    mi_rows[[length(mi_rows) + 1]] <- data.frame(
      gene   = gene, chr = ch,
      locus  = names(mi_res$LOD),
      pos_mb = mi_res$pos[['Mb']],
      lod    = as.numeric(mi_res$LOD),
      stringsAsFactors = FALSE
    )
  }
  mi_df <- do.call(rbind, mi_rows)
  write.csv(mi_df, file.path(out_dir, 'mi_lod_genome.csv'), row.names = FALSE)
  cat(sprintf('  %d rows written\n', nrow(mi_df)))
  n_genes_s7 <- length(unique(mi_df$gene))
  n_loci_s7  <- if (!is.null(chr_nloci)) sum(chr_nloci[as.character(sig_genes_df$chr)], na.rm = TRUE) else NA_integer_
}

# ─────────────────────────────────────────────────────────────────────────────
# S8 — SNP scan: significant genes on their peak chromosome (gated)
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s7 <- peak_heap_mb()
rss_mb_s7  <- get_rss_mb()
n_genes_s8 <- NA_integer_; n_loci_s8 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s8 <- proc.time()
if (run(8)) {
  if (!dir.exists(allele_dir)) {
    cat('\n[S8] SKIPPED — allele_dir not found:', allele_dir, '\n')
    n_genes_s8 <- 0L; n_loci_s8 <- 0L
  } else {
    stopifnot(!is.null(sig_genes_df))
    cat(sprintf('\n[S8] SNP scan — %d significant genes\n', nrow(sig_genes_df)))
    # imputed.snp.scan.h2lmm hardcodes SUBJECT.NAME; alias column once before loop.
    pheno_df_s8 <- pheno_df
    pheno_df_s8$SUBJECT.NAME <- pheno_df_s8$sample_id
    snp_rows <- list()
    for (i in seq_len(nrow(sig_genes_df))) {
      gene <- sig_genes_df$gene[i]
      ch   <- as.character(sig_genes_df$chr[i])
      cat(sprintf('  [%d/%d] %s  chr%s\n', i, nrow(sig_genes_df), gene, ch))
      # Restrict to SNPs actually in the alleles file. Lines without tabs are bare
      # SNP_IDs (line 1 of each 4-line block); sparcc pseudomarkers like
      # UNC6_to_UNC15185 are absent from the alleles file and not removed by
      # imputed.snp.scan.h2lmm's ^c[X0-9]+\.loc regex, causing grep to crash.
      allele_file_ch <- file.path(allele_dir, paste0('chr', ch, '.alleles'))
      snp_ids_ch <- if (file.exists(allele_file_ch)) {
        lines_ch <- readLines(allele_file_ch, warn = FALSE)
        lines_ch[!grepl('\t', lines_ch, fixed = TRUE)]
      } else NULL
      snp_res <- imputed.snp.scan.h2lmm(
        genomecache     = genomecache, data = pheno_df_s8,
        formula         = as.formula(paste(gene, '~ sex + dataset')),
        K               = kinship_loco[[ch]],
        allele.dir      = allele_dir, chr = ch,
        just.these.loci = snp_ids_ch
      )
      snp_rows[[length(snp_rows) + 1]] <- data.frame(
        gene   = gene, chr = ch,
        snp    = names(snp_res$LOD),
        pos_mb = snp_res$pos[['Mb']],
        lod    = as.numeric(snp_res$LOD),
        stringsAsFactors = FALSE
      )
    }
    snp_df <- do.call(rbind, snp_rows)
    write.csv(snp_df, file.path(out_dir, 'snp_lod_genome.csv'), row.names = FALSE)
    cat(sprintf('  %d rows written\n', nrow(snp_df)))
    n_genes_s8 <- length(unique(snp_df$gene))
    n_loci_s8  <- nrow(snp_df)
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S9 — mediation: significant genes, all other sig genes as candidate mediators
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s8 <- peak_heap_mb()
rss_mb_s8  <- get_rss_mb()
n_genes_s9 <- NA_integer_; n_loci_s9 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s9 <- proc.time()
if (run(9)) {
  stopifnot(!is.null(sig_genes_df), !is.null(peak_lod_df))
  cat(sprintf('\n[S9] mediation — %d significant genes\n', nrow(sig_genes_df)))

  # build gene_data and renamed pheno_df once for all outcomes
  gene_col_map <- setNames(paste0('gene_', gene_cols), gene_cols)
  med_df       <- pheno_df
  colnames(med_df)[match(gene_cols, colnames(med_df))] <- gene_col_map

  # gene_data requires actual gene coordinates (TSS/start), not QTL peak positions
  if (!is.null(gene_chr_map) && file.exists(gene_info_path)) {
    gi <- read.csv(gene_info_path, stringsAsFactors = FALSE)
    gi$gene_col <- make.names(gi$gene_id, unique = TRUE)
    gene_data <- data.frame(
      chr       = as.character(gi$chr),
      start     = as.integer(gi$start),
      end       = as.integer(gi$end),
      gene_name = paste0('gene_', gi$gene_col),
      stringsAsFactors = FALSE
    )
    gene_data <- gene_data[!is.na(gene_data$start), ]
  } else {
    stop('[S9] gene_info.csv required for mediation — rerun convert_gastroc_h5ad.py')
  }

  med_rows <- list()
  for (i in seq_len(nrow(sig_genes_df))) {
    gene <- sig_genes_df$gene[i]
    ch   <- as.character(sig_genes_df$chr[i])
    # peak locus for this gene on its peak chromosome
    sub_g      <- lod_genome[lod_genome$gene == gene & lod_genome$chr == ch, ]
    peak_locus <- sub_g$locus[which.max(sub_g$lod)]
    cat(sprintf('  [%d/%d] %s  chr%s  locus=%s\n', i, nrow(sig_genes_df), gene, ch, peak_locus))

    med_qr <- extract.mediation.expression.qr(
      genomecache = genomecache, id = 'sample_id',
      locus = peak_locus, data = med_df,
      gene.data = gene_data, formula = as.formula('~ sex + dataset'), chr = ch
    )
    med_res <- mediation.scan.qr(
      mediation.qr.object = med_qr, data = med_df,
      phenotype = gene_col_map[[gene]], chr = ch, id = 'sample_id'
    )
    med_res$outcome_gene <- gene
    med_rows[[length(med_rows) + 1]] <- med_res
  }
  if (length(med_rows) > 0L) {
    med_df_out <- do.call(rbind, med_rows)
    write.csv(med_df_out, file.path(out_dir, 'mediation_genome.csv'), row.names = FALSE)
    cat(sprintf('  %d rows written\n', nrow(med_df_out)))
  }
  n_genes_s9 <- nrow(sig_genes_df)
  n_loci_s9  <- if (!is.null(chr_nloci)) sum(chr_nloci[as.character(sig_genes_df$chr)], na.rm = TRUE) else NA_integer_
}

# ─────────────────────────────────────────────────────────────────────────────
# S10 — bootstrap positional CI: all significant genes
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s9 <- peak_heap_mb()
rss_mb_s9  <- get_rss_mb()
n_genes_s10 <- NA_integer_; n_loci_s10 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s10 <- proc.time()
if (run(10)) {
  stopifnot(!is.null(sig_genes_df))
  cat(sprintf('\n[S10] bootstrap CI — %d significant genes\n', nrow(sig_genes_df)))
  ci_rows <- list(); ci_sum_rows <- list()

  for (i in seq_len(nrow(sig_genes_df))) {
    gene <- sig_genes_df$gene[i]
    ch   <- as.character(sig_genes_df$chr[i])
    cat(sprintf('  [%d/%d] %s  chr%s\n', i, nrow(sig_genes_df), gene, ch))
    # Bootstrap CI at strain level (pheno.id == geno.id = 'genotype').
    # generate.sample.outcomes.matrix with pheno.id != geno.id stores the expanded
    # n×n K in fit0$K; reduce.large.K then returns n×n (duplicate-name bug) and
    # chol() fails on the rank-deficient matrix. Strain-level scan keeps K at m×m.
    K_g         <- kinship_loco[[ch]]
    K_eigs_g    <- eigen(K_g, symmetric = TRUE, only.values = TRUE)$values
    K_min_eig_g <- min(K_eigs_g)
    K_psd_tol_g <- max(abs(K_eigs_g)) * nrow(K_g)^2 * .Machine$double.eps
    if (K_min_eig_g < -K_psd_tol_g) {
      cat(sprintf('    [skip] near-singular LOCO kinship (min_eig=%.2e) — CI skipped\n',
                  K_min_eig_g))
      next
    }

    f_strain <- as.formula(paste(gene, '~ 1'))
    rop_g <- scan.h2lmm(
      genomecache = genomecache, data = strain_means,
      pheno.id = 'genotype', geno.id = 'genotype',
      formula = f_strain, K = K_g,
      use.multi.impute = FALSE, return.allele.effects = FALSE, chr = ch
    )
    peak_locus_g <- grab.locus.from.scan(rop_g)
    rop_g_alt <- scan.h2lmm(
      genomecache = genomecache, data = strain_means,
      pheno.id = 'genotype', geno.id = 'genotype',
      formula = f_strain, K = K_g,
      use.multi.impute = FALSE, return.allele.effects = FALSE,
      chr = ch, just.these.loci = peak_locus_g
    )
    w <- rep(1, length(rop_g_alt$fit1$y)); names(w) <- names(rop_g_alt$fit1$y)
    rop_g_alt$fit1$weights <- w

    boot_out  <- generate.sample.outcomes.matrix(
      scan.object = rop_g_alt, model.type = 'alt',
      method = 'bootstrap', num.samples = 200, seed = 1
    )
    ci_result <- run.positional.scans(
      sim.object = boot_out, genomecache = genomecache, data = strain_means
    )
    peak_mb_g <- rop_g$pos[['Mb']][which.max(rop_g$LOD)]
    ci_mb_g   <- ci_result$peak.loci.pos$Mb
    ci_rows[[length(ci_rows) + 1]] <- data.frame(
      gene = gene, chr = ch, boot_index = seq_along(ci_mb_g), peak_mb = ci_mb_g
    )
    ci_sum_rows[[length(ci_sum_rows) + 1]] <- data.frame(
      gene     = gene, chr = ch,
      peak_mb  = peak_mb_g,
      ci_lo_mb = as.numeric(quantile(ci_mb_g, 0.025)),
      ci_hi_mb = as.numeric(quantile(ci_mb_g, 0.975))
    )
    cat(sprintf('    peak=%.2f Mb  95%% CI [%.2f, %.2f]\n',
                peak_mb_g,
                ci_sum_rows[[length(ci_sum_rows)]]$ci_lo_mb,
                ci_sum_rows[[length(ci_sum_rows)]]$ci_hi_mb))
  }
  write.csv(do.call(rbind, ci_rows),     file.path(out_dir, 'ci_genome.csv'),         row.names = FALSE)
  write.csv(do.call(rbind, ci_sum_rows), file.path(out_dir, 'ci_summary_genome.csv'), row.names = FALSE)
  cat(sprintf('  CI written for %d genes\n', length(ci_sum_rows)))
  n_genes_s10 <- length(ci_sum_rows)
  n_loci_s10  <- if (!is.null(chr_nloci)) sum(chr_nloci[as.character(sig_genes_df$chr)], na.rm = TRUE) else NA_integer_
}

# ─────────────────────────────────────────────────────────────────────────────
# S11 — instability scan: all genes, cis (same loop structure as S4)
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s10 <- peak_heap_mb()
rss_mb_s10  <- get_rss_mb()
n_genes_s11 <- NA_integer_; n_loci_s11 <- NA_integer_
gc(reset = TRUE, verbose = FALSE)
t_s11 <- proc.time()
if (run(11)) {
  cat('\n[S11] instability scan — all genes, cis\n')
  inst_rows <- list()

  for (ch in target_chrs) {
    chr_genes_11 <- switch(scan_type,
      'G' = gene_cols,
      'C' = , 'L' = if (!is.null(gene_chr_map))
                      gene_cols[!is.na(gene_chr_map[gene_cols]) & gene_chr_map[gene_cols] == ch]
                    else gene_cols
    )
    if (length(chr_genes_11) == 0L) next
    cat(sprintf('  chr%s  %d genes\n', ch, length(chr_genes_11)))

    for (gene in chr_genes_11) {
      f_g <- as.formula(paste(gene, '~ sex + dataset'))
      simple_obj <- generate.simple.sample.outcomes.matrix(
        formula = f_g, data = pheno_df, pheno.id = 'sample_id',
        method = 'permutation', num.samples = 100, seed = 1
      )
      inst_res <- instability.lm.scan(
        simple.sample.object = simple_obj,
        genomecache = genomecache, chr = ch, use.ROP = TRUE
      )
      inst_pmat <- inst_res$full.results$p.values
      inst_negp <- apply(-log10(inst_pmat), 2, median, na.rm = TRUE)
      inst_rows[[length(inst_rows) + 1]] <- data.frame(
        gene        = gene,
        locus       = colnames(inst_pmat),
        chr         = inst_res$full.results$chr,
        pos_mb      = inst_res$full.results$pos$Mb,
        neg_log10_p = inst_negp,
        stringsAsFactors = FALSE
      )
    }
  }
  inst_df <- do.call(rbind, inst_rows)
  write.csv(inst_df, file.path(out_dir, 'instability_genome.csv'), row.names = FALSE)
  cat(sprintf('  %d rows written\n', nrow(inst_df)))
  n_genes_s11 <- length(unique(inst_df$gene))
  n_loci_s11  <- length(unique(inst_df$locus))
}

# ─────────────────────────────────────────────────────────────────────────────
# timing summary — S4 per-chromosome + S5-S11 section wall-clock
# ─────────────────────────────────────────────────────────────────────────────
peak_mb_s11 <- peak_heap_mb()
rss_mb_s11  <- get_rss_mb()
t_end <- proc.time()

if (run(4) && file.exists(timing_chr_path) && file.exists(timing_genome_path)) {
  tc <- read.csv(timing_chr_path)
  tg <- read.csv(timing_genome_path)
  cat('\n── per-chromosome timing ───────────────────────────────────────────\n')
  print(tc[order(as.integer(gsub('[^0-9]', '', tc$chr))), c('chr', 'n_loci', 'elapsed_s')],
        row.names = FALSE)
  cat('\n── genome-wide timing ──────────────────────────────────────────────\n')
  print(tg[, c('n_genes', 'n_chrs', 'wall_clock_s', 'total_cpu_s', 'genes_per_min')],
        row.names = FALSE)
  cat('────────────────────────────────────────────────────────────────────\n')
}

# write per-section wall-clock + peak heap (S5-S11) to timing_sections.csv
# t_sN is set just before each section guard regardless of whether it ran;
# elapsed since t_sN captures the wall time of that section (0 if skipped).
# peak_mb_sN is captured immediately after the section closes; gc(reset=TRUE)
# before each section resets R's max-used counter so the reading is per-section.
elapsed_s  <- function(t0, t1) as.numeric(t1['elapsed']   - t0['elapsed'])
user_s_fn  <- function(t0, t1) as.numeric(t1['user.self'] - t0['user.self'])
sys_s_fn   <- function(t0, t1) as.numeric(t1['sys.self']  - t0['sys.self'])

sec_rows <- list(
  data.frame(section='S5_permutation',    elapsed_s=elapsed_s(t_s5,  t_s6),  user_s=user_s_fn(t_s5,  t_s6),  sys_s=sys_s_fn(t_s5,  t_s6),  peak_heap_mb=peak_mb_s5,  peak_rss_mb=rss_mb_s5,  n_genes=n_genes_s5,  n_loci=n_loci_s5,  n_perms=if(exists('n_perm_s5')) n_perm_s5 else NA_integer_,  stringsAsFactors=FALSE),
  data.frame(section='S6_cache_collapse', elapsed_s=elapsed_s(t_s6,  t_s7),  user_s=user_s_fn(t_s6,  t_s7),  sys_s=sys_s_fn(t_s6,  t_s7),  peak_heap_mb=peak_mb_s6,  peak_rss_mb=rss_mb_s6,  n_genes=n_genes_s6,  n_loci=n_loci_s6,  n_perms=NA_integer_,  stringsAsFactors=FALSE),
  data.frame(section='S7_mi_scan',        elapsed_s=elapsed_s(t_s7,  t_s8),  user_s=user_s_fn(t_s7,  t_s8),  sys_s=sys_s_fn(t_s7,  t_s8),  peak_heap_mb=peak_mb_s7,  peak_rss_mb=rss_mb_s7,  n_genes=n_genes_s7,  n_loci=n_loci_s7,  n_perms=NA_integer_,  stringsAsFactors=FALSE),
  data.frame(section='S8_snp_scan',       elapsed_s=elapsed_s(t_s8,  t_s9),  user_s=user_s_fn(t_s8,  t_s9),  sys_s=sys_s_fn(t_s8,  t_s9),  peak_heap_mb=peak_mb_s8,  peak_rss_mb=rss_mb_s8,  n_genes=n_genes_s8,  n_loci=n_loci_s8,  n_perms=NA_integer_,  stringsAsFactors=FALSE),
  data.frame(section='S9_mediation',      elapsed_s=elapsed_s(t_s9,  t_s10), user_s=user_s_fn(t_s9,  t_s10), sys_s=sys_s_fn(t_s9,  t_s10), peak_heap_mb=peak_mb_s9,  peak_rss_mb=rss_mb_s9,  n_genes=n_genes_s9,  n_loci=n_loci_s9,  n_perms=NA_integer_,  stringsAsFactors=FALSE),
  data.frame(section='S10_bootstrap_ci',  elapsed_s=elapsed_s(t_s10, t_s11), user_s=user_s_fn(t_s10, t_s11), sys_s=sys_s_fn(t_s10, t_s11), peak_heap_mb=peak_mb_s10, peak_rss_mb=rss_mb_s10, n_genes=n_genes_s10, n_loci=n_loci_s10, n_perms=200L,         stringsAsFactors=FALSE),
  data.frame(section='S11_instability',   elapsed_s=elapsed_s(t_s11, t_end), user_s=user_s_fn(t_s11, t_end), sys_s=sys_s_fn(t_s11, t_end), peak_heap_mb=peak_mb_s11, peak_rss_mb=rss_mb_s11, n_genes=n_genes_s11, n_loci=n_loci_s11, n_perms=NA_integer_,  stringsAsFactors=FALSE)
)
sec_df <- do.call(rbind, sec_rows)
sec_df$chr           <- chr_arg
sec_df$scan_type     <- scan_type
sec_df$r_version     <- paste(R.version$major, R.version$minor, sep = '.')
sec_df$miqtl_version <- tryCatch(as.character(packageVersion('miqtl')), error = function(e) NA_character_)
sec_df$git_commit    <- tryCatch(trimws(system('git rev-parse --short HEAD', intern = TRUE)), error = function(e) NA_character_)
sec_df$timestamp     <- format(Sys.time(), '%Y-%m-%dT%H:%M:%S')

write.csv(sec_df, file.path(out_dir, 'timing_sections.csv'), row.names = FALSE)

cat('\n── section timing (S5-S11) ─────────────────────────────────────────\n')
for (i in seq_len(nrow(sec_df))) {
  ng <- if (!is.na(sec_df$n_genes[i])) sec_df$n_genes[i] else 0L
  nl <- if (!is.na(sec_df$n_loci[i]))  sec_df$n_loci[i]  else 0L
  cat(sprintf('  %-24s  %6.0f s (cpu %5.0f s)  heap %5.0f MB  rss %5.0f MB  %d genes  %d loci  (%s)\n',
              sec_df$section[i], sec_df$elapsed_s[i], sec_df$user_s[i],
              sec_df$peak_heap_mb[i], sec_df$peak_rss_mb[i],
              ng, nl,
              if (sec_df$elapsed_s[i] > 0) 'ran' else 'skipped'))
}
cat('────────────────────────────────────────────────────────────────────\n')

# ── final eGene / eQTL summary ───────────────────────────────────────────────
cat('\n── FINAL SUMMARY ───────────────────────────────────────────────────────\n')
n_total <- if (!is.null(peak_lod_df)) nrow(peak_lod_df) else NA_integer_
n_egene <- if (!is.null(sig_genes_df) && file.exists(thresholds_chr_path))
             nrow(sig_genes_df) else NA_integer_
cat(sprintf('  genes with peak LOD:  %s\n', if (is.na(n_total)) 'n/a' else n_total))
cat(sprintf('  eGenes (LOD > thr):   %s\n', if (is.na(n_egene)) 'n/a (S5 not run)' else n_egene))
# eQTL count = eGenes here (one peak locus per gene; multi-signal scan not implemented)
cat(sprintf('  eQTL (= eGene peaks): %s\n', if (is.na(n_egene)) 'n/a' else n_egene))
if (!is.na(n_egene) && n_egene > 0) {
  top5 <- head(sig_genes_df[order(-sig_genes_df$lod), ], 5)
  cat('  top eGenes by LOD:\n')
  for (i in seq_len(nrow(top5)))
    cat(sprintf('    %d. %-22s chr%s  LOD=%.2f  pos=%.2f Mb\n',
                i, top5$gene[i], top5$chr[i], top5$lod[i], top5$pos_mb[i]))
}
# write machine-readable summary
write.csv(data.frame(
  n_genes_scanned = n_total,
  n_egenes        = n_egene,
  n_eqtl          = n_egene,
  scan_type       = scan_type,
  n_chrs          = length(scan_chrs),
  timestamp       = format(Sys.time(), '%Y-%m-%dT%H:%M:%S'),
  stringsAsFactors = FALSE
), file.path(out_dir, 'run_summary.csv'), row.names = FALSE)
cat('  run_summary.csv written\n')
cat('────────────────────────────────────────────────────────────────────────\n')

cat('\ndone\n')
