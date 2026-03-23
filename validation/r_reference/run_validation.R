# run_validation.R
# cross-language validation of mosaiqtl against r miqtl
#
# this script is a direct derivative of scan_eqtl.R — same inputs, same
# scan.h2lmm() call, same parameters. the only difference is scope:
# it runs on a small subset of genes and one chromosome, then exports
# lod scores for comparison against mosaiqtl python output.
#
# since the pipeline has already produced all real data outputs, this
# script reads them directly:
#   output/pheno_48.rds      — 48-strain means (pre-collapsed, no rebuild needed)
#   output/kinship_loco.rds  — one 48×48 loco matrix per chromosome
#   output/genomecache/      — happy genome cache, additive model
#   input/pmap.csv           — physical map (marker, chr, pos_mb)
#   input/gencode.vM32.gtf.gz — gene tss coordinates (optional; falls back to
#                               midpoint if absent, e.g. when running locally)
#
# outputs:
#   output/validation/lod_r.csv          — lod + p_value per gene × locus
#   output/validation/lead_loci.csv      — top locus per gene
#   output/validation/allele_effects.csv — per-founder effects at peak locus
#
# usage (on crsp):
#   Rscript validation/r_reference/run_validation.R [chr] [n_genes]
#   Rscript validation/r_reference/run_validation.R 8 5
#
# ─────────────────────────────────────────────────────────────────────────────

# ── library paths (identical to scan_eqtl.R) ─────────────────────────────────
lib_path <- '/data/homezvol2/gfilimba/R/x86_64-pc-linux-gnu-library/4.4'
if (dir.exists(lib_path)) {
  .libPaths(c(lib_path, .libPaths()))
  library(miqtl,       lib.loc = lib_path)
  library(dplyr,       lib.loc = lib_path)
  library(rtracklayer, lib.loc = lib_path)
} else {
  library(miqtl); library(dplyr)
}

# ── args ──────────────────────────────────────────────────────────────────────
args    <- commandArgs(trailingOnly = TRUE)
scan_chr <- if (length(args) >= 1) args[1] else '8'
n_genes  <- if (length(args) >= 2) as.integer(args[2]) else 5

message('[init] chr=', scan_chr, '  n_genes=', n_genes)

# ── paths (identical to scan_eqtl.R) ─────────────────────────────────────────
base_dir         <- '/share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc'
genomecache_path <- file.path(base_dir, 'output', 'genomecache')
kinship_path     <- file.path(base_dir, 'output', 'kinship_loco.rds')
pheno_48_path    <- file.path(base_dir, 'output', 'pheno_48.rds')  # pre-built by pipeline
pheno_df_path    <- file.path(base_dir, 'output', 'pheno_df.rds')  # fallback if pheno_48 absent
gtf_path         <- file.path(base_dir, 'input',  'gencode.vM32.gtf.gz')
pmap_path        <- file.path(base_dir, 'input',  'pmap.csv')
output_dir       <- file.path(base_dir, 'output', 'validation')
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ── load inputs ───────────────────────────────────────────────────────────────
message('[load] reading inputs...')

# pheno_48 — pre-collapsed by pipeline; use directly if it exists, otherwise
# rebuild from pheno_df the same way scan_eqtl.R does it
if (file.exists(pheno_48_path)) {
  pheno_48 <- readRDS(pheno_48_path)
  message('  pheno_48 loaded: ', nrow(pheno_48), ' strains × ', ncol(pheno_48), ' cols')
} else {
  message('  pheno_48.rds not found — collapsing pheno_df (same as scan_eqtl.R)')
  pheno_df       <- readRDS(pheno_df_path)
  pheno_df$batch <- as.character(pheno_df$batch)
  pheno_df$sex   <- as.character(pheno_df$sex)
  meta_cols      <- c('sample_id', 'subject.name', 'sex', 'batch', paste0('pc', 1:10))
  gene_cols_all  <- setdiff(colnames(pheno_df), meta_cols)
  cols_to_avg    <- c(gene_cols_all, paste0('pc', 1:10))
  pheno_48 <- aggregate(pheno_df[, cols_to_avg],
                        by  = list(subject.name = pheno_df$subject.name),
                        FUN = mean, na.rm = TRUE)
  pheno_48$batch     <- tapply(pheno_df$batch, pheno_df$subject.name,
                               `[`, 1)[pheno_48$subject.name]
  rownames(pheno_48) <- pheno_48$subject.name
  message('  pheno_48 built: ', nrow(pheno_48), ' strains × ', ncol(pheno_48), ' cols')
}

# kinship — identical to scan_eqtl.R
kinship_loco <- readRDS(kinship_path)
k_chr <- kinship_loco[[scan_chr]]
message('  kinship chr', scan_chr, ': ', paste(dim(k_chr), collapse = ' × '))

# pmap — identical column rename from scan_eqtl.R
pmap_flat <- read.csv(pmap_path, stringsAsFactors = FALSE)
colnames(pmap_flat)[colnames(pmap_flat) == 'marker'] <- 'locus'
colnames(pmap_flat)[colnames(pmap_flat) == 'pos']    <- 'pos_mb'
# strip .RData suffix if present — scan.h2lmm() appends it; double extension = file not found
pmap_flat$locus    <- sub('\\.RData$', '', pmap_flat$locus)
rownames(pmap_flat) <- pmap_flat$locus
chr_loci <- pmap_flat[pmap_flat$chr == scan_chr, ]
message('  pmap chr', scan_chr, ': ', nrow(chr_loci), ' loci')

# gene tss — from gtf if present (same as scan_eqtl.R); otherwise chr midpoint fallback
if (file.exists(gtf_path)) {
  gtf       <- as.data.frame(import(gtf_path))
  gtf_genes <- gtf[gtf$type == 'gene', ]
  gene_annot <- data.frame(
    gene_id = gtf_genes$gene_name,
    chr     = gsub('^chr', '', as.character(gtf_genes$seqnames)),
    tss_mb  = ifelse(gtf_genes$strand == '+',
                     gtf_genes$start, gtf_genes$end) / 1e6,
    stringsAsFactors = FALSE
  )
  gene_annot <- gene_annot[gene_annot$chr == scan_chr, ]
  message('  gtf genes on chr', scan_chr, ': ', nrow(gene_annot))
  use_gtf <- TRUE
} else {
  message('  gtf not found — using chr midpoint as tss proxy (ok for validation)')
  use_gtf <- FALSE
}

# ── identify genes to scan ────────────────────────────────────────────────────
meta_cols  <- c('subject.name', 'batch', paste0('pc', 1:10))
all_genes  <- setdiff(colnames(pheno_48), meta_cols)

if (use_gtf) {
  # real mode: same intersection as scan_eqtl.R
  gene_names <- head(intersect(all_genes, gene_annot$gene_id), n_genes)
} else {
  # fallback: first n_genes from pheno_48 (simulation / local)
  gene_names <- head(all_genes, n_genes)
}

message('[scan] genes: ', paste(gene_names, collapse = ', '))

# ── scan loop (identical to scan_eqtl.R) ─────────────────────────────────────
cis_window_mb <- 0.5   # ±500 kb — same as scan_eqtl.R
n_pcs         <- 10
results_list  <- list()
effects_list  <- list()

for (gene in gene_names) {
  # tss lookup — gtf if available, else midpoint fallback
  if (use_gtf) {
    gene_tss <- gene_annot$tss_mb[gene_annot$gene_id == gene][1]
    if (is.na(gene_tss)) next
  } else {
    gene_tss <- mean(chr_loci$pos_mb)
  }

  cis_loci <- chr_loci[abs(chr_loci$pos_mb - gene_tss) <= cis_window_mb, 'locus']
  if (length(cis_loci) == 0) next

  # formula — same as scan_eqtl.R (backtick-wrap, no sex, batch + pc1..pc10)
  scan_formula <- as.formula(
    paste0('`', gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse = ' + '))
  )

  message('  scanning: ', sprintf('%-25s', gene),
          ' | ', length(cis_loci), ' loci')

  # scan.h2lmm() call — identical to scan_eqtl.R
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
      message('  scan failed: ', gene, ' — ', conditionMessage(e))
      NULL
    }
  )

  if (is.null(scan_result)) next

  # pos_mb lookup — match() instead of direct indexing (same defensive approach
  # as scan_eqtl.R in case some cache markers are absent from pmap)
  pos_mb_vals <- pmap_flat$pos_mb[match(scan_result$loci, pmap_flat$locus)]

  results_list[[gene]] <- data.frame(
    gene_id = gene,
    locus   = scan_result$loci,
    chr     = scan_chr,
    pos_mb  = pos_mb_vals,
    lod     = scan_result$LOD,
    p_value = scan_result$p.value,
    stringsAsFactors = FALSE
  )

  # allele effects at peak locus
  if (!is.null(scan_result$allele.effects)) {
    peak_idx <- which.max(scan_result$LOD)
    effects  <- scan_result$allele.effects[peak_idx, ]
    effects_list[[gene]] <- data.frame(
      gene_id = gene, chr = scan_chr,
      locus   = scan_result$loci[peak_idx],
      founder = names(effects), effect = as.numeric(effects),
      stringsAsFactors = FALSE
    )
  }
}

# ── save outputs ───────────────────────────────────────────────────────────────
if (length(results_list) == 0)
  stop('[error] no scan results — check errors above')

lod_results  <- do.call(rbind, results_list); rownames(lod_results) <- NULL
effects_all  <- if (length(effects_list) > 0) do.call(rbind, effects_list) else NULL

lead_loci <- lod_results %>%
  group_by(gene_id) %>%
  slice_min(p_value, n = 1, with_ties = FALSE) %>%
  ungroup() %>% as.data.frame()

write.csv(lod_results, file.path(output_dir, 'lod_r.csv'),     row.names = FALSE)
write.csv(lead_loci,   file.path(output_dir, 'lead_loci.csv'), row.names = FALSE)
if (!is.null(effects_all))
  write.csv(effects_all, file.path(output_dir, 'allele_effects.csv'), row.names = FALSE)

# also export the pheno_48 slice used here — python reads it for exact alignment
write.csv(pheno_48, file.path(output_dir, 'pheno_48_used.csv'))

cat('\n[results]\n')
for (i in seq_len(nrow(lead_loci))) {
  r <- lead_loci[i, ]
  cat(sprintf('  %-25s chr%-3s pos=%6.2f Mb  lod=%5.2f  p=%s\n',
              r$gene_id, r$chr, r$pos_mb, r$lod,
              format(r$p_value, scientific = TRUE, digits = 3)))
}

cat('\n[saved to', output_dir, ']\n')
cat('  lod_r.csv         :', nrow(lod_results), 'rows\n')
cat('  lead_loci.csv     :', nrow(lead_loci), 'rows\n')
cat('  pheno_48_used.csv : for python alignment\n')
cat('\n[done]\n')
