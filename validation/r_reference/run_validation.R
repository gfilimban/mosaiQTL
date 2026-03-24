# run_validation.R
# cross-language validation of mosaiqtl against r miqtl
#
# a scoped derivative of scan_eqtl.R — same inputs, same scan.h2lmm()
# call, same parameters. runs on a small gene/chromosome subset and
# exports lod scores for comparison against mosaiqtl python output.
#
# inputs (real pipeline outputs from founders_f1_genomecache.ipynb):
#   $data_dir/genomecache/       — happy genome cache, additive model
#   $data_dir/kinship_loco.rds   — one 48×48 loco matrix per chromosome
#   $data_dir/pheno_48.rds       — pre-collapsed 48-strain means
#   $data_dir/pheno_df.rds       — fallback if pheno_48.rds absent
#
# outputs (written to validation/r_data/ inside mosaiqtl package):
#   lod_r.csv          — lod + p_value per gene × locus
#   lead_loci.csv      — top locus per gene
#   allele_effects.csv — per-founder effects at peak locus
#   pheno_48_used.csv  — exact pheno_48 slice used (for python alignment)
#
# usage:
#   Rscript validation/r_reference/run_validation.R [chr] [n_genes]
#   Rscript validation/r_reference/run_validation.R 8 5
# ─────────────────────────────────────────────────────────────────────────────

# ── library paths ─────────────────────────────────────────────────────────────
lib_path <- '/data/homezvol2/gfilimba/R/x86_64-pc-linux-gnu-library/4.4'
if (dir.exists(lib_path)) {
  .libPaths(c(lib_path, .libPaths()))
  library(miqtl,       lib.loc = lib_path)
  library(dplyr,       lib.loc = lib_path)
  library(rtracklayer, lib.loc = lib_path)
} else {
  library(miqtl); library(dplyr)
  if (requireNamespace('rtracklayer', quietly = TRUE)) library(rtracklayer)
}

# ── args ──────────────────────────────────────────────────────────────────────
args    <- commandArgs(trailingOnly = TRUE)
# chr:     optional; default = all chromosomes (same as scan_eqtl.R which loops all)
# n_genes: optional cap for quick testing; omit to scan all genes
scan_chrs <- if (length(args) >= 1) args[1] else 'all'
n_genes   <- if (length(args) >= 2) as.integer(args[2]) else Inf

message('[init] chr=', scan_chrs,
        '  n_genes=', if (is.infinite(n_genes)) 'all' else n_genes)

# ── paths ─────────────────────────────────────────────────────────────────────
# data_dir: where real pipeline outputs live (from founders_f1_genomecache.ipynb)
# pkg_dir:  mosaiqtl package root — outputs always go to validation/r_data/ here
data_dir <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir  <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

genomecache_path <- file.path(data_dir, 'output', 'genomecache')
kinship_path     <- file.path(data_dir, 'output', 'kinship_loco.rds')
pheno_48_path    <- file.path(data_dir, 'output', 'results', 'pheno_48.rds')
pheno_df_path    <- file.path(data_dir, 'output', 'pheno_df.rds')
pmap_path        <- file.path(data_dir, 'input',  'pmap.csv')
gtf_path         <- file.path(data_dir, 'input',  'gencode.vM32.gtf.gz')
output_dir       <- file.path(pkg_dir,  'validation', 'r_data')

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
message('[paths]')
message('  data:   ', data_dir)
message('  output: ', output_dir)

# ── load inputs ───────────────────────────────────────────────────────────────
message('[load]')

# pheno_48 — use pre-built file if present; otherwise collapse pheno_df the
# same way scan_eqtl.R does (aggregate mean per strain, tapply first for batch)
if (file.exists(pheno_48_path)) {
  pheno_48 <- readRDS(pheno_48_path)
  # ensure rownames are set to subject.name for kinship indexing
  if (!identical(rownames(pheno_48), pheno_48$subject.name))
    rownames(pheno_48) <- pheno_48$subject.name
  message('  pheno_48.rds: ', nrow(pheno_48), ' strains × ', ncol(pheno_48), ' cols')
} else {
  message('  pheno_48.rds absent — collapsing pheno_df (same as scan_eqtl.R)')
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

# kinship — 48×48 loco, strain labels as rownames
kinship_loco <- readRDS(kinship_path)
# kinship loaded per-chromosome inside the loop below

# pmap — same column rename as scan_eqtl.R
pmap_flat <- read.csv(pmap_path, stringsAsFactors = FALSE)
colnames(pmap_flat)[colnames(pmap_flat) == 'marker'] <- 'locus'
colnames(pmap_flat)[colnames(pmap_flat) == 'pos']    <- 'pos_mb'
pmap_flat$locus    <- sub('\\.RData$', '', pmap_flat$locus)  # strip suffix if present
rownames(pmap_flat) <- pmap_flat$locus
# chr_loci and k_chr set per-chromosome inside the loop below
message('  pmap total loci: ', nrow(pmap_flat))

# gene tss — from gtf (same as scan_eqtl.R) with chr midpoint fallback
use_gtf <- FALSE
if (file.exists(gtf_path) && requireNamespace('rtracklayer', quietly = TRUE)) {
  gtf        <- as.data.frame(rtracklayer::import(gtf_path))
  gtf_genes  <- gtf[gtf$type == 'gene', ]
  gene_annot <- data.frame(
    gene_id = gtf_genes$gene_name,
    chr     = gsub('^chr', '', as.character(gtf_genes$seqnames)),
    tss_mb  = ifelse(gtf_genes$strand == '+',
                     gtf_genes$start, gtf_genes$end) / 1e6,
    stringsAsFactors = FALSE
  )
  # gene_annot kept for all chromosomes; subset per chr inside loop
  use_gtf    <- TRUE
  message('  gtf genes loaded: ', nrow(gene_annot))
} else {
  message('  gtf not found — using chr midpoint as tss (ok for validation)')
}

# ── select genes ──────────────────────────────────────────────────────────────
meta_cols <- c('subject.name', 'batch', paste0('pc', 1:10))
all_genes <- setdiff(colnames(pheno_48), meta_cols)
all_genes <- head(all_genes, n_genes)

# chromosomes to scan — all available in kinship if not specified
if (scan_chrs == 'all') {
  chromosomes <- names(kinship_loco)
} else {
  chromosomes <- strsplit(scan_chrs, ',')[[1]]
}
message('[scan] chromosomes: ', paste(sort(chromosomes), collapse = ', '))
message('[scan] genes in pheno_48: ', length(all_genes))

# ── scan loop (identical to scan_eqtl.R — loops chromosomes then genes) ───────
cis_window_mb <- 0.5   # ±500 kb — matches scan_eqtl.R
n_pcs         <- 10
results_list  <- list()
effects_list  <- list()

for (scan_chr in chromosomes) {
  message('\n[chr', scan_chr, ']')

  k_chr    <- kinship_loco[[scan_chr]]
  chr_loci <- pmap_flat[pmap_flat$chr == scan_chr, ]
  message('  loci: ', nrow(chr_loci))

  # genes on this chromosome (requires gtf); fall back to all genes if no gtf
  if (use_gtf) {
    chr_gene_annot <- gene_annot[gene_annot$chr == scan_chr, ]
    gene_names     <- intersect(all_genes, chr_gene_annot$gene_id)
  } else {
    chr_gene_annot <- NULL
    gene_names     <- all_genes
  }
  message('  genes: ', length(gene_names))

  for (gene in gene_names) {
    gene_tss <- if (use_gtf) {
    chr_gene_annot$tss_mb[chr_gene_annot$gene_id == gene][1]
  } else {
    mean(chr_loci$pos_mb)
  }
  if (is.na(gene_tss)) next

  cis_loci <- chr_loci[abs(chr_loci$pos_mb - gene_tss) <= cis_window_mb, 'locus']
  if (length(cis_loci) == 0) next

  scan_formula <- as.formula(
    paste0('`', gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse = ' + '))
  )

  if (length(gene_names) <= 20)   # only print gene names for small runs
    message('  ', sprintf('%-25s', gene), ' | ', length(cis_loci), ' loci')

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
      message('  scan failed: ', gene, ' — ', conditionMessage(e)); NULL
    }
  )

  if (is.null(scan_result)) next

  pos_mb_vals <- pmap_flat$pos_mb[match(scan_result$loci, pmap_flat$locus)]

  results_list[[paste0(gene, '_', scan_chr)]] <- data.frame(
    gene_id = gene, locus = scan_result$loci, chr = scan_chr,
    pos_mb = pos_mb_vals, lod = scan_result$LOD, p_value = scan_result$p.value,
    stringsAsFactors = FALSE
  )

  ae <- scan_result$allele.effects
  if (!is.null(ae) && length(scan_result$LOD) > 0) {
    peak_idx <- which.max(scan_result$LOD)
    # allele.effects can be a matrix (n_loci x 8) or named vector (single locus)
    if (is.matrix(ae) && nrow(ae) >= peak_idx) {
      effects <- ae[peak_idx, ]
    } else if (!is.matrix(ae)) {
      effects <- ae
    } else {
      effects <- NULL
    }
    if (!is.null(effects) && length(effects) > 0 && length(scan_result$loci) > 0) {
  effects_list[[paste0(gene, '_', scan_chr)]] <- data.frame(
        gene_id = gene, chr = scan_chr,
        locus   = scan_result$loci[min(peak_idx, length(scan_result$loci))],
        founder = names(effects), effect = as.numeric(effects),
        stringsAsFactors = FALSE
      )
    }
  }
}
}  # end chromosome loop

# ── save to validation/r_data/ ────────────────────────────────────────────────
if (length(results_list) == 0) stop('[error] no scan results — check errors above')

lod_results <- do.call(rbind, results_list); rownames(lod_results) <- NULL
lead_loci   <- lod_results %>%
  group_by(gene_id) %>% slice_min(p_value, n = 1, with_ties = FALSE) %>%
  ungroup() %>% as.data.frame()

write.csv(lod_results, file.path(output_dir, 'lod_r.csv'),     row.names = FALSE)
write.csv(lead_loci,   file.path(output_dir, 'lead_loci.csv'), row.names = FALSE)
# ensure batch column is present before writing (may be absent in pre-built pheno_48.rds)
if (!('batch' %in% colnames(pheno_48)) && exists('pheno_df')) {
  pheno_48$batch <- tapply(pheno_df$batch, pheno_df$subject.name, `[`, 1)[pheno_48$subject.name]
}
write.csv(pheno_48, file.path(output_dir, 'pheno_48_used.csv'), row.names = FALSE)  # python alignment

if (length(effects_list) > 0)
  write.csv(do.call(rbind, effects_list),
            file.path(output_dir, 'allele_effects.csv'), row.names = FALSE)

cat('\n[results]\n')
for (i in seq_len(nrow(lead_loci))) {
  r <- lead_loci[i, ]
  cat(sprintf('  %-25s chr%-3s pos=%6.2f Mb  lod=%5.2f  p=%s\n',
              r$gene_id, r$chr, r$pos_mb, r$lod,
              format(r$p_value, scientific = TRUE, digits = 3)))
}
cat('\n[saved to', output_dir, ']\n')
cat('  lod_r.csv, lead_loci.csv, allele_effects.csv, pheno_48_used.csv\n')
cat('\n[done]\n')
