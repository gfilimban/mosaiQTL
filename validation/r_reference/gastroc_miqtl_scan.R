# gastroc_miqtl_scan.R — full miQTL workflow on gastroc whole-tissue pseudobulk
#
# prerequisite: run validation/convert_gastroc_h5ad.py first to produce
#   validation/gastroc_data/counts_lognorm.h5
#   validation/gastroc_data/metadata.csv
#
# sections:
#   1.  cache info              → cache_info.csv, founders.csv, sample_ids.csv
#   2.  haplotype probs         → haplotype_probs.csv
#   3.  LOCO kinship summary    → kinship_summary.csv
#                                 plots/kinship_chr8.png
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

lib_path <- Sys.getenv('R_LIB_PATH',
                       unset = '/Users/ghassan/miniconda3/envs/bioinfo/lib/R/library')
if (dir.exists(lib_path)) .libPaths(c(lib_path, .libPaths()))

for (pkg in c('miqtl', 'rhdf5')) library(pkg, character.only = TRUE)

# ── crossprod fix ─────────────────────────────────────────────────────────────
if (!is.primitive(crossprod) || typeof(crossprod) != 'builtin') {
  assign('crossprod',  base::crossprod,  envir = globalenv())
  assign('tcrossprod', base::tcrossprod, envir = globalenv())
}

# ── args ──────────────────────────────────────────────────────────────────────
# usage (local):  Rscript gastroc_miqtl_scan.R [sections]
# usage (HPC):    Rscript gastroc_miqtl_scan.R 4 8   # S4 on chr 8 only
#   per-chr HPC jobs write to gastroc_data/chr{N}/; merge_chr_outputs.R combines them
args         <- commandArgs(trailingOnly = TRUE)
run_sections <- if (length(args) >= 1 && nchar(args[1]) > 0) {
                  as.integer(strsplit(args[1], ',')[[1]])
                } else seq_len(11)
chr_arg      <- if (length(args) >= 2 && nchar(args[2]) > 0) args[2] else NULL
hpc_mode     <- !is.null(chr_arg)   # TRUE when scanning a single chromosome on HPC
run <- function(s) s %in% run_sections

# ── paths ─────────────────────────────────────────────────────────────────────
miqtl_dir    <- file.path(path.expand('~'), 'Documents/journey/learning/graduate',
                           'lab/projects/igvf/analysis/qtl/miqtl')
genomecache  <- file.path(miqtl_dir, 'output/genomecache')
kinship_path <- file.path(miqtl_dir, 'output/kinship_loco.rds')
allele_dir   <- file.path(miqtl_dir, 'output/allele_dir')
out_dir      <- 'validation/gastroc_data'
# HPC per-chr jobs write into a chr-specific subdirectory
s4_out_dir   <- if (hpc_mode) file.path(out_dir, paste0('chr', chr_arg)) else out_dir
plot_dir     <- file.path(out_dir, 'plots')
pheno_rds    <- file.path(out_dir, 'gastroc_pheno.rds')
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

# strain-level mean phenotype (for prob.heatmap which works at genotype resolution)
strain_means <- aggregate(pheno_df[, gene_cols],
                          by = list(genotype = pheno_df$genotype),
                          FUN = mean, na.rm = TRUE)

# ── LOCO kinship ──────────────────────────────────────────────────────────────
cat('[load] reading', kinship_path, '\n')
kinship_loco <- readRDS(kinship_path)
all_chrs     <- names(kinship_loco)
cat(sprintf('[load] %d chromosomes  kinship %d×%d\n',
            length(all_chrs), nrow(kinship_loco[[1]]), ncol(kinship_loco[[1]])))

# ─────────────────────────────────────────────────────────────────────────────
# S1 — cache info
# ─────────────────────────────────────────────────────────────────────────────
if (run(1)) {
  cat('\n[S1] cache info\n')
  rd         <- DiploprobReader$new(genomecache)
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
  write.csv(data.frame(founder = founders),    file.path(out_dir, 'founders.csv'),    row.names = FALSE)
  write.csv(data.frame(sample  = sample_ids),  file.path(out_dir, 'sample_ids.csv'), row.names = FALSE)
  cat(sprintf('  %d chromosomes  %d founders  %d strains\n',
              nrow(chr_info), length(founders), length(sample_ids)))
}

# ─────────────────────────────────────────────────────────────────────────────
# S2 — haplotype probabilities (5 evenly spaced loci on chr 8)
# ─────────────────────────────────────────────────────────────────────────────
if (run(2)) {
  cat('\n[S2] haplotype probs  chr8 (5 loci)\n')
  rd    <- DiploprobReader$new(genomecache)
  loci8 <- rd$getLoci('8')
  loci8 <- loci8[round(seq(1, length(loci8), length.out = 5))]

  hap_rows <- do.call(rbind, lapply(loci8, function(loc) {
    mat <- rd$getLocusMatrix(loc, model = 'additive')   # (n_strains × 8)
    df  <- as.data.frame(mat)
    colnames(df) <- rd$getFounders()
    df$locus      <- loc
    df$genotype   <- rownames(mat)
    df
  }))
  write.csv(hap_rows, file.path(out_dir, 'haplotype_probs.csv'), row.names = FALSE)
  cat(sprintf('  exported %d rows\n', nrow(hap_rows)))
}

# ─────────────────────────────────────────────────────────────────────────────
# S3 — LOCO kinship summary + heatmap
# ─────────────────────────────────────────────────────────────────────────────
if (run(3)) {
  cat('\n[S3] kinship summary\n')
  k_summary <- do.call(rbind, lapply(all_chrs, function(ch) {
    K <- kinship_loco[[ch]]
    data.frame(chr          = ch,
               mean_diag    = mean(diag(K)),
               mean_offdiag = mean(K[lower.tri(K)]),
               stringsAsFactors = FALSE)
  }))
  write.csv(k_summary, file.path(out_dir, 'kinship_summary.csv'), row.names = FALSE)

  K8    <- kinship_loco[['8']]
  n_str <- nrow(K8)
  side  <- max(1800L, as.integer(n_str * 36L))   # ~36 px per strain at res=300
  png(file.path(plot_dir, 'kinship_chr8.png'), width = side, height = side, res = 300)
  image(K8[n_str:1, ], main = 'LOCO kinship — chr 8', axes = FALSE,
        col = colorRampPalette(c('#2166ac', 'white', '#d6604d'))(100))
  axis(1, at = seq(0, 1, length.out = n_str), labels = rownames(K8),        las = 2, cex.axis = 0.35)
  axis(2, at = seq(0, 1, length.out = n_str), labels = rev(rownames(K8)),   las = 2, cex.axis = 0.35)
  dev.off()
  cat(sprintf('  %d chromosomes summarised\n', nrow(k_summary)))
}

# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM scan (ROP, LOCO) — full genome, all genes, timed per-chr + genome-wide
# ─────────────────────────────────────────────────────────────────────────────
lod_genome_path    <- file.path(out_dir, 'lod_genome.csv')
effects_path       <- file.path(out_dir, 'allele_effects_chr8.csv')
timing_chr_path    <- file.path(out_dir, 'timing_chr.csv')
timing_genome_path <- file.path(out_dir, 'timing_genome.csv')

peak_lod_df       <- NULL
chr8_scan_results <- list()

if (run(4)) {
  scan_chrs <- if (hpc_mode) chr_arg else all_chrs
  cat(sprintf('\n[S4] LOCO scan  %d genes × chr %s%s\n',
              n_genes,
              paste(scan_chrs, collapse = ','),
              if (hpc_mode) '  [HPC per-chr mode]' else '  [sequential]'))

  all_lod_rows    <- list()
  all_effect_rows <- list()

  chr_t <- system.time({
    for (ch in scan_chrs) {
      K_ch    <- kinship_loco[[ch]]
      on_chr8 <- ch == '8'

      for (gene in gene_cols) {
        f   <- as.formula(paste(gene, '~ dataset'))
        res <- scan.h2lmm(
          genomecache           = genomecache,
          data                  = pheno_df,
          pheno.id              = 'sample_id',
          geno.id               = 'genotype',
          formula               = f,
          K                     = K_ch,
          use.multi.impute      = FALSE,
          return.allele.effects = on_chr8,
          chr                   = ch
        )

        if (on_chr8) chr8_scan_results[[gene]] <- res

        lod_vec <- res$LOD
        all_lod_rows[[length(all_lod_rows) + 1]] <- data.frame(
          gene   = gene,
          chr    = ch,
          locus  = names(lod_vec),
          pos_mb = res$pos[['Mb']],
          lod    = as.numeric(lod_vec),
          stringsAsFactors = FALSE
        )

        if (on_chr8 && !is.null(res$allele.effects)) {
          for (loc in names(res$allele.effects)) {
            eff <- res$allele.effects[[loc]]
            all_effect_rows[[length(all_effect_rows) + 1]] <- data.frame(
              gene    = gene,
              locus   = loc,
              pos_mb  = res$pos[['Mb']][names(res$LOD) == loc],
              founder = names(eff),
              effect  = as.numeric(eff),
              stringsAsFactors = FALSE
            )
          }
        }
      }
      rd_ch  <- DiploprobReader$new(genomecache)
      n_loci <- length(rd_ch$getLoci(ch))
      cat(sprintf('  chr%s  %d loci  %.1fs\n', ch, n_loci,
                  proc.time()[['elapsed']]))
    }
  })

  lod_df     <- do.call(rbind, all_lod_rows)
  effects_df <- do.call(rbind, all_effect_rows)

  # ── per-chromosome timing row (one row per chr scanned by this job) ──────
  rd_s4  <- DiploprobReader$new(genomecache)
  timing_row <- data.frame(
    chr         = paste(scan_chrs, collapse = ','),
    n_genes     = n_genes,
    n_loci      = sum(sapply(scan_chrs, function(ch) length(rd_s4$getLoci(ch)))),
    elapsed_s   = as.numeric(chr_t['elapsed']),
    user_s      = as.numeric(chr_t['user.self']),
    sys_s       = as.numeric(chr_t['sys.self']),
    genes_per_min = round(n_genes / chr_t['elapsed'] * 60, 1),
    mode        = if (hpc_mode) 'hpc_per_chr' else 'sequential',
    row.names   = NULL
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
      n_genes           = n_genes,
      n_chrs            = length(all_chrs),
      wall_clock_s      = as.numeric(chr_t['elapsed']),
      total_cpu_s       = as.numeric(chr_t['elapsed']),
      genes_per_min     = round(n_genes / chr_t['elapsed'] * 60, 1),
      mode              = 'sequential',
      row.names         = NULL
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
  peak_lod_df <- do.call(rbind, lapply(gene_cols, function(g) {
    sub <- lod_genome[lod_genome$gene == g, ]
    sub[which.max(sub$lod), ]
  }))
  peak_lod_df <- peak_lod_df[order(-peak_lod_df$lod), ]
  if (!hpc_mode)
    cat(sprintf('  top hit: %s  chr%s  LOD=%.2f\n',
                peak_lod_df$gene[1], peak_lod_df$chr[1], peak_lod_df$lod[1]))

  # ── build combined genome-wide scan object for top gene (genome.plotter.whole)
  top_gene <- peak_lod_df$gene[1]
  cat(sprintf('\n[S4] building genome-wide scan object for %s\n', top_gene))
  gene_chr_scans <- list()
  for (ch in all_chrs) {
    f_top  <- as.formula(paste(top_gene, '~ dataset'))
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
    loci    = combined_loci
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
  if (!is.null(chr8_scan_results[[top_gene]]) &&
      !is.null(chr8_scan_results[[top_gene]]$allele.effects)) {
    png(file.path(plot_dir, sprintf('allele_effects_%s.png', top_gene)),
        width = 3000L, height = 1200L, res = 300L)
    allele.plotter.whole(scan.object = chr8_scan_results[[top_gene]],
                         main        = sprintf('%s — chr 8', top_gene))
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

} else if (file.exists(lod_genome_path)) {
  cat('[S4] loading cached lod_genome.csv\n')
  lod_genome  <- read.csv(lod_genome_path, stringsAsFactors = FALSE)
  peak_lod_df <- do.call(rbind, lapply(unique(lod_genome$gene), function(g) {
    sub <- lod_genome[lod_genome$gene == g, ]
    sub[which.max(sub$lod), ]
  }))
  peak_lod_df <- peak_lod_df[order(-peak_lod_df$lod), ]
}

top_gene <- if (!is.null(peak_lod_df)) peak_lod_df$gene[1] else gene_cols[1]
top_chr  <- if (!is.null(peak_lod_df)) as.character(peak_lod_df$chr[1]) else all_chrs[1]

# ─────────────────────────────────────────────────────────────────────────────
# S5 — permutation threshold (top gene, top chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run(5)) {
  cat(sprintf('\n[S5] permutation threshold  gene=%s  chr=%s\n', top_gene, top_chr))
  f_top <- as.formula(paste(top_gene, '~ dataset'))
  rop   <- scan.h2lmm(
    genomecache           = genomecache,
    data                  = pheno_df,
    pheno.id              = 'sample_id',
    geno.id               = 'genotype',
    formula               = f_top,
    K                     = kinship_loco[[top_chr]],
    use.multi.impute      = FALSE,
    return.allele.effects = FALSE,
    chr                   = top_chr
  )

  # named weights required for generate.sample.outcomes.matrix
  w        <- rep(1, length(rop$fit0$y))
  names(w) <- names(rop$fit0$y)
  rop$fit0$weights <- w

  perm_out <- generate.sample.outcomes.matrix(
    scan.object = rop,
    model.type  = 'null',
    method      = 'permutation',
    num.samples = 1000,
    seed        = 1
  )
  perm_scans <- run.threshold.scans(
    sim.threshold.object = perm_out,
    keep.full.scans      = FALSE,
    genomecache          = genomecache,
    data                 = pheno_df,
    chr                  = top_chr
  )

  # get.gev.thresholds returns a scalar
  threshold <- get.gev.thresholds(
    threshold.scans = perm_scans,
    use.lod         = TRUE,
    percentile      = 0.95,
    type            = 'max'
  )
  empirical_p95 <- quantile(perm_scans$max.statistics$LOD, 0.95)

  write.csv(data.frame(perm_index = seq_along(perm_scans$max.statistics$LOD),
                       max_lod    = perm_scans$max.statistics$LOD),
            file.path(out_dir, 'perm_max_lods.csv'), row.names = FALSE)
  write.csv(data.frame(threshold     = threshold,
                       empirical_p95 = as.numeric(empirical_p95)),
            file.path(out_dir, 'thresholds.csv'), row.names = FALSE)
  cat(sprintf('  GEV threshold (95th) = %.3f  empirical = %.3f\n',
              threshold, empirical_p95))
}

# ─────────────────────────────────────────────────────────────────────────────
# S6 — cache collapse
# ─────────────────────────────────────────────────────────────────────────────
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
}

# ─────────────────────────────────────────────────────────────────────────────
# S7 — multiple imputation scan (top gene, top chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run(7)) {
  cat(sprintf('\n[S7] MI scan  gene=%s  chr=%s\n', top_gene, top_chr))
  f_top  <- as.formula(paste(top_gene, '~ dataset'))
  mi_res <- scan.h2lmm(
    genomecache      = genomecache,
    data             = pheno_df,
    pheno.id         = 'sample_id',
    geno.id          = 'genotype',
    formula          = f_top,
    K                = kinship_loco[[top_chr]],
    use.multi.impute = TRUE,
    num.imp          = 11,
    chr              = top_chr
  )
  mi_lod <- data.frame(locus  = names(mi_res$LOD),
                       pos_mb = mi_res$pos[['Mb']],
                       lod    = as.numeric(mi_res$LOD),
                       stringsAsFactors = FALSE)
  write.csv(mi_lod, file.path(out_dir, 'mi_lod.csv'), row.names = FALSE)
  cat(sprintf('  MI peak LOD = %.3f\n', max(mi_lod$lod)))
}

# ─────────────────────────────────────────────────────────────────────────────
# S8 — SNP scan (gated: allele_dir required)
# ─────────────────────────────────────────────────────────────────────────────
if (run(8)) {
  if (!dir.exists(allele_dir)) {
    cat('\n[S8] SKIPPED — allele_dir not found:', allele_dir, '\n')
  } else {
    cat(sprintf('\n[S8] SNP scan  gene=%s  chr=%s\n', top_gene, top_chr))
    f_top   <- as.formula(paste(top_gene, '~ dataset'))
    snp_res <- imputed.snp.scan.h2lmm(
      genomecache  = genomecache,
      data         = pheno_df,
      pheno.id     = 'sample_id',
      geno.id      = 'genotype',
      formula      = f_top,
      K            = kinship_loco[[top_chr]],
      allele.dir   = allele_dir,
      chr          = top_chr
    )
    snp_lod <- data.frame(snp    = names(snp_res$LOD),
                          pos_mb = snp_res$pos[['Mb']],
                          lod    = as.numeric(snp_res$LOD),
                          stringsAsFactors = FALSE)
    write.csv(snp_lod, file.path(out_dir, 'snp_lod.csv'), row.names = FALSE)
    cat(sprintf('  SNP peak LOD = %.3f\n', max(snp_lod$lod)))
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S9 — mediation (gene-gene; gene_data built from peak_lod_df — no GTF needed)
# ─────────────────────────────────────────────────────────────────────────────
if (run(9)) {
  cat(sprintf('\n[S9] mediation  outcome=%s  chr=%s\n', top_gene, top_chr))
  stopifnot(!is.null(peak_lod_df), 'run S4 first or provide lod_genome.csv' = TRUE)

  # gene_data: chr, start_bp, end_bp, gene_name — derived from peak LOD positions
  # col 4 (gene_name) must match the gene_* column suffix after renaming
  gene_data <- data.frame(
    chr       = peak_lod_df$chr,
    start     = as.integer(peak_lod_df$pos_mb * 1e6),
    end       = as.integer(peak_lod_df$pos_mb * 1e6) + 1L,
    gene_name = peak_lod_df$gene,
    stringsAsFactors = FALSE
  )

  # rename gene columns to gene_* so extract.mediation.expression.qr can find them
  gene_col_map <- setNames(paste0('gene_', gene_cols), gene_cols)
  med_df       <- pheno_df
  colnames(med_df)[match(gene_cols, colnames(med_df))] <- gene_col_map
  gene_data$gene_name <- paste0('gene_', gene_data$gene_name)   # match col names

  # peak locus (condition locus for mediation)
  peak_locus <- if (exists('combined_loci')) {
                  combined_loci[which.max(combined_lod)]
                } else {
                  lod_tmp <- read.csv(lod_genome_path, stringsAsFactors = FALSE)
                  sub_tmp <- lod_tmp[lod_tmp$gene == top_gene & lod_tmp$chr == top_chr, ]
                  sub_tmp$locus[which.max(sub_tmp$lod)]
                }

  f_med  <- as.formula('~ dataset')
  med_qr <- extract.mediation.expression.qr(
    genomecache = genomecache,
    id          = 'sample_id',
    locus       = peak_locus,
    data        = med_df,
    gene.data   = gene_data,
    formula     = f_med,
    chr         = top_chr
  )
  med_res <- mediation.scan.qr(
    mediation.qr.object = med_qr,
    data                = med_df,
    phenotype           = gene_col_map[[top_gene]],
    chr                 = top_chr,
    id                  = 'sample_id'
  )
  write.csv(med_res, file.path(out_dir, 'mediation.csv'), row.names = FALSE)
  cat(sprintf('  %d mediators tested\n', nrow(med_res)))
}

# ─────────────────────────────────────────────────────────────────────────────
# S10 — bootstrap positional CI (top gene, top chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run(10)) {
  cat(sprintf('\n[S10] bootstrap CI  gene=%s  chr=%s\n', top_gene, top_chr))
  f_top <- as.formula(paste(top_gene, '~ dataset'))

  # step 1: full scan to identify peak locus
  rop10 <- scan.h2lmm(
    genomecache           = genomecache,
    data                  = pheno_df,
    pheno.id              = 'sample_id',
    geno.id               = 'genotype',
    formula               = f_top,
    K                     = kinship_loco[[top_chr]],
    use.multi.impute      = FALSE,
    return.allele.effects = FALSE,
    chr                   = top_chr
  )
  peak_locus_10 <- grab.locus.from.scan(rop10)

  # step 2: re-run at peak locus only to get fit1 (required by model.type='alt')
  rop10_alt <- scan.h2lmm(
    genomecache           = genomecache,
    data                  = pheno_df,
    pheno.id              = 'sample_id',
    geno.id               = 'genotype',
    formula               = f_top,
    K                     = kinship_loco[[top_chr]],
    use.multi.impute      = FALSE,
    return.allele.effects = FALSE,
    chr                   = top_chr,
    just.these.loci       = peak_locus_10
  )

  w        <- rep(1, length(rop10_alt$fit0$y))
  names(w) <- names(rop10_alt$fit0$y)
  rop10_alt$fit0$weights <- w

  boot_out <- generate.sample.outcomes.matrix(
    scan.object = rop10_alt,
    model.type  = 'alt',
    method      = 'bootstrap',
    num.samples = 200,
    seed        = 1
  )
  ci_result <- run.positional.scans(
    sim.object  = boot_out,
    genomecache = genomecache,
    data        = pheno_df
  )

  peak_mb    <- rop10$pos[['Mb']][which.max(rop10$LOD)]
  ci_mb      <- ci_result$peak.loci.pos$Mb
  ci_df      <- data.frame(boot_index = seq_along(ci_mb), peak_mb = ci_mb)
  ci_summary <- data.frame(ci_lo_mb = quantile(ci_mb, 0.025),
                            ci_hi_mb = quantile(ci_mb, 0.975),
                            peak_mb  = peak_mb)
  write.csv(ci_df,      file.path(out_dir, 'ci.csv'),         row.names = FALSE)
  write.csv(ci_summary, file.path(out_dir, 'ci_summary.csv'), row.names = FALSE)
  cat(sprintf('  peak = %.2f Mb  95%% CI [%.2f, %.2f] Mb\n',
              peak_mb, ci_summary$ci_lo_mb, ci_summary$ci_hi_mb))
}

# ─────────────────────────────────────────────────────────────────────────────
# S11 — instability scan (top gene, top chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run(11)) {
  cat(sprintf('\n[S11] instability scan  gene=%s  chr=%s\n', top_gene, top_chr))
  f_top <- as.formula(paste(top_gene, '~ dataset'))

  # generate.simple.sample.outcomes.matrix: OLS-based (no K), formula + data + pheno.id
  simple_obj <- generate.simple.sample.outcomes.matrix(
    formula     = f_top,
    data        = pheno_df,
    pheno.id    = 'sample_id',
    method      = 'permutation',
    num.samples = 100,
    seed        = 1
  )
  inst_res <- instability.lm.scan(
    simple.sample.object = simple_obj,
    genomecache          = genomecache,
    chr                  = top_chr,
    use.ROP              = TRUE
  )
  # instability returns p.values (F-test), not LOD; convert to -log10(p)
  inst_pmat  <- inst_res$full.results$p.values        # (n_scans × n_loci)
  inst_loci  <- colnames(inst_pmat)
  inst_negp  <- apply(-log10(inst_pmat), 2, median, na.rm = TRUE)
  inst_df    <- data.frame(
    locus  = inst_loci,
    chr    = inst_res$full.results$chr,
    pos_mb = inst_res$full.results$pos$Mb,
    neg_log10_p = inst_negp,
    stringsAsFactors = FALSE
  )
  write.csv(inst_df, file.path(out_dir, 'instability.csv'), row.names = FALSE)
  cat(sprintf('  instability peak -log10(p) = %.3f\n', max(inst_df$neg_log10_p)))
}

# ─────────────────────────────────────────────────────────────────────────────
# timing summary
# ─────────────────────────────────────────────────────────────────────────────
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

cat('\ndone\n')
