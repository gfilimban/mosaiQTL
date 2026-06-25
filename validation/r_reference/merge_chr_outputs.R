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
pkg_dir  <- Sys.getenv('MOSAIQTL_DIR',
             unset = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl')
out_dir  <- file.path(pkg_dir, 'validation', 'gastroc_data')
r_data   <- file.path(out_dir, 'r_data')

chrs <- c(as.character(1:19), 'X')

# ── sentinel check: abort if any chromosome hasn't completed S4 ──────────────
sentinel <- 'lod.csv'
missing_chrs <- c()
for (chr in chrs) {
  p <- file.path(r_data, paste0('chr', chr), sentinel)
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

# ── merge peak LOD → peak_lod.csv (one row per gene, genome-wide top hit) ────
# lod_genome.csv (full locus-level) is NOT merged here — too large (3.5M rows × 20 chr).
# allele_effects_genome.csv is NOT merged here — ~6 GB per chr, structure not yet optimised.
# Both remain in per-chr r_data/chr{N}/ for downstream use.
message('[merge] peak_lod.csv → peak_lod.csv (genome-wide)')
peak_parts <- list()
for (chr in chrs) {
  p <- file.path(r_data, paste0('chr', chr), 'peak_lod.csv')
  if (file.exists(p) && file.size(p) > 0) {
    peak_parts[[chr]] <- read.csv(p, stringsAsFactors=FALSE)
  }
}
if (length(peak_parts) > 0) {
  peak_all <- do.call(rbind, peak_parts)
  rownames(peak_all) <- NULL
  # keep best locus per gene (cis scan → one chr per gene, but be safe)
  peak_lod_df <- do.call(rbind, lapply(unique(peak_all$gene), function(g) {
    sub <- peak_all[peak_all$gene == g, ]
    sub[which.max(sub$lod), ]
  }))
  peak_lod_df <- peak_lod_df[order(-peak_lod_df$lod), ]
  rownames(peak_lod_df) <- NULL
  write.csv(peak_lod_df, file.path(r_data, 'peak_lod.csv'), row.names=FALSE)
  message('  peak_lod.csv: ', nrow(peak_lod_df), ' genes  top=',
          peak_lod_df$gene[1], ' LOD=', round(peak_lod_df$lod[1], 2))
} else {
  message('  peak_lod.csv: no per-chr peak_lod.csv found — skipping')
  peak_lod_df <- NULL
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
