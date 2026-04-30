# merge_chr_outputs.R
# combines per-chromosome S4 outputs from SLURM array jobs into unified files.
# run after all per-chr gastroc_miqtl_scan.R jobs complete.
#
# HPC pattern:
#   Rscript gastroc_miqtl_scan.R 4 1   → gastroc_data/chr1/{lod,allele_effects,timing}.csv
#   Rscript gastroc_miqtl_scan.R 4 2   → gastroc_data/chr2/{lod,allele_effects,timing}.csv
#   ...
#   Rscript merge_chr_outputs.R         → gastroc_data/{lod_genome,allele_effects_chr8,
#                                                        timing_chr,timing_genome}.csv
# ─────────────────────────────────────────────────────────────────────────────
pkg_dir  <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'
out_dir  <- file.path(pkg_dir, 'validation', 'gastroc_data')

chrs <- c(as.character(1:19), 'X')

# ── sentinel check: abort if any chromosome hasn't completed S4 ──────────────
sentinel <- 'lod.csv'
missing_chrs <- c()
for (chr in chrs) {
  p <- file.path(out_dir, paste0('chr', chr), sentinel)
  if (!file.exists(p) || file.size(p) == 0) {
    message('  [sentinel] chr', chr, ': MISSING ', sentinel)
    missing_chrs <- c(missing_chrs, as.character(chr))
  } else {
    message('  [sentinel] chr', chr, ': ok')
  }
}
if (length(missing_chrs)) {
  stop('[merge] aborting — ', length(missing_chrs), ' chromosome(s) missing sentinel\n',
       '  missing: chr ', paste(missing_chrs, collapse=', chr '), '\n',
       '  rerun:   Rscript validation/r_reference/gastroc_miqtl_scan.R 4 <chr>')
}
message('[merge] all ', length(chrs), ' chromosomes complete — proceeding\n')

# ── merge LOD scores → lod_genome.csv ────────────────────────────────────────
message('[merge] lod.csv → lod_genome.csv')
lod_parts <- list()
for (chr in chrs) {
  p  <- file.path(out_dir, paste0('chr', chr), 'lod.csv')
  df <- read.csv(p, stringsAsFactors=FALSE)
  lod_parts[[chr]] <- df
}
lod_genome <- do.call(rbind, lod_parts)
rownames(lod_genome) <- NULL
write.csv(lod_genome, file.path(out_dir, 'lod_genome.csv'), row.names=FALSE)
message('  lod_genome.csv: ', nrow(lod_genome), ' rows from ', length(chrs), ' chromosomes')

# ── merge allele effects (chr 8 only) → allele_effects_chr8.csv ──────────────
message('[merge] allele_effects.csv → allele_effects_chr8.csv')
eff_path <- file.path(out_dir, 'chr8', 'allele_effects.csv')
if (file.exists(eff_path) && file.size(eff_path) > 0) {
  eff_df <- read.csv(eff_path, stringsAsFactors=FALSE)
  write.csv(eff_df, file.path(out_dir, 'allele_effects_chr8.csv'), row.names=FALSE)
  message('  allele_effects_chr8.csv: ', nrow(eff_df), ' rows')
} else {
  message('  allele_effects_chr8.csv: MISSING — chr 8 job may not have run with return.allele.effects=TRUE')
}

# ── merge + summarise timing ──────────────────────────────────────────────────
message('[merge] timing.csv → timing_chr.csv + timing_genome.csv')
timing_parts <- list()
for (chr in chrs) {
  p <- file.path(out_dir, paste0('chr', chr), 'timing.csv')
  if (file.exists(p)) {
    df      <- read.csv(p, stringsAsFactors=FALSE)
    df$chr  <- chr          # overwrite the comma-joined chr field with single chr
    timing_parts[[chr]] <- df
  }
}
timing_chr <- do.call(rbind, timing_parts)
rownames(timing_chr) <- NULL

# re-derive genes_per_min per row from elapsed_s (handles any rounding in original)
n_genes_per_row <- timing_chr$n_genes
timing_chr$genes_per_min <- round(n_genes_per_row / timing_chr$elapsed_s * 60, 1)

write.csv(timing_chr, file.path(out_dir, 'timing_chr.csv'), row.names=FALSE)
message('  timing_chr.csv: ', nrow(timing_chr), ' rows')

# genome-wide: wall-clock = max(elapsed) across chr (parallel); total_cpu = sum
n_genes      <- timing_chr$n_genes[1]        # same for every chr
wall_clock_s <- max(timing_chr$elapsed_s)
total_cpu_s  <- sum(timing_chr$elapsed_s)
timing_genome <- data.frame(
  n_genes       = n_genes,
  n_chrs        = length(chrs),
  wall_clock_s  = round(wall_clock_s, 1),
  total_cpu_s   = round(total_cpu_s,  1),
  genes_per_min = round(n_genes / wall_clock_s * 60, 1),
  speedup       = round(total_cpu_s / wall_clock_s, 2),   # ≈ n_chrs if perfectly parallel
  mode          = 'hpc_parallel'
)
write.csv(timing_genome, file.path(out_dir, 'timing_genome.csv'), row.names=FALSE)

# ── print benchmark summary ───────────────────────────────────────────────────
cat('\n── per-chromosome timing ────────────────────────────────────────────\n')
tc_ordered <- timing_chr[order(suppressWarnings(as.integer(timing_chr$chr))), ]
print(tc_ordered[, c('chr', 'n_loci', 'elapsed_s', 'genes_per_min')], row.names=FALSE)
cat('\n── genome-wide timing (HPC parallel) ───────────────────────────────\n')
print(timing_genome[, c('n_genes', 'n_chrs', 'wall_clock_s', 'total_cpu_s',
                         'genes_per_min', 'speedup')], row.names=FALSE)
cat('─────────────────────────────────────────────────────────────────────\n')
cat(sprintf('[merge] done.  outputs in %s\n', out_dir))
