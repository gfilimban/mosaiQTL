# merge_chr_outputs.R
# combines per-chromosome CSVs from SLURM array jobs into unified files.
# run after run_validation_slurm.sh completes all chr jobs.
# ─────────────────────────────────────────────────────────────────────────────
pkg_dir <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'
r_data  <- file.path(pkg_dir, 'validation', 'port_validation', 'r_data')

# CSVs that are per-chromosome and need stacking.
# S4: lod_r, allele_effects_r | S5: thresholds_r, perm_lods_r | S7: mi_lod_r
# S8: snp_lod_r | S9: mediation_r | S10: ci_r, boot_pos_r | S11: fixef_r
per_chr_files <- c(
  'lod_r.csv',
  'allele_effects_r.csv',
  'mi_lod_r.csv',
  'snp_lod_r.csv',
  'thresholds_r.csv',
  'perm_lods_r.csv',
  'mediation_r.csv',
  'ci_r.csv',
  'boot_pos_r.csv',
  'fixef_r.csv'
)

# CSVs that are genome-level (written by setup job) — just check they exist
genome_files <- c(
  'cache_info_r.csv', 'founders_r.csv', 'sample_ids_r.csv',
  'haplotype_r.csv', 'kinship_r.csv', 'collapsed_cache_summary.csv',
  'pmap_r.csv', 'pheno_48_used.csv'
)

chrs <- c(1:19, 'X')

# ── sentinel check: abort if any chromosome hasn't completed S10 ──────────────
# ci_r.csv is written at the end of S10 (positional CI) — the last section
# that commonly times out. a missing sentinel means the chr job did not finish.
sentinel <- 'ci_r.csv'
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
       '  rerun:   bash validation/rerun_failed_chrs.sh ',
       paste(missing_chrs, collapse=','))
}
message('[merge] all ', length(chrs), ' chromosomes complete — proceeding\n')

message('[merge] combining per-chr outputs')

for (fn in per_chr_files) {
  parts <- list()
  for (chr in chrs) {
    p <- file.path(r_data, paste0('chr', chr), fn)
    if (file.exists(p)) {
      df <- tryCatch(read.csv(p, stringsAsFactors=FALSE),
                     error=function(e) NULL)
      if (!is.null(df) && nrow(df) > 0) parts[[as.character(chr)]] <- df
    }
  }
  if (length(parts)) {
    merged <- do.call(rbind, parts); rownames(merged) <- NULL
    out_path <- file.path(r_data, fn)
    write.csv(merged, out_path, row.names=FALSE)
    message('  ', fn, ': ', nrow(merged), ' rows from ',
            length(parts), ' chromosomes')
  } else {
    message('  ', fn, ': no data found across chromosomes')
  }
}

# genome-level: use per-chr first chromosome that has them
# (setup job writes them to r_data/ directly — just verify)
message('\n[merge] checking genome-level outputs')
for (fn in genome_files) {
  p <- file.path(r_data, fn)
  if (file.exists(p)) {
    df <- read.csv(p)
    message('  ', fn, ': ok (', nrow(df), ' rows)')
  } else {
    message('  ', fn, ': MISSING — re-run setup step')
  }
}

# also merge per-chr kinship matrices (one file per chr)
message('\n[merge] checking kinship matrices')
for (chr in chrs) {
  kf <- file.path(r_data, paste0('kinship_mat_r_chr', chr, '.csv'))
  # kinship matrices written by setup job to r_data/ directly
  # check per-chr subdir as fallback
  if (!file.exists(kf)) {
    kf2 <- file.path(r_data, paste0('chr', chr),
                     paste0('kinship_mat_r_chr', chr, '.csv'))
    if (file.exists(kf2)) file.copy(kf2, kf)
  }
  message('  kinship chr', chr, ': ',
          if (file.exists(kf)) 'ok' else 'MISSING')
}

cat('\n[merge] done. outputs in', r_data, '\n')
