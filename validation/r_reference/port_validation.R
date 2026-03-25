# port_validation.R
# R reference outputs for mosaiQTL Phase 2 full-port validation
# runs miQTL equivalents of each Phase 2 feature and writes CSVs
# that validate_port.py compares against Python implementations.
#
# sections:
#   4. collapse.genomecache()        → collapsed_cache_summary.csv
#   5. get.gev.thresholds()          → thresholds_r.csv
#   6. scan.h2lmm(use.multi.impute)  → mi_lod_r.csv
#   7. imputed.snp.scan.h2lmm()      → snp_lod_r.csv
#   8. mediation.scan.qr()           → mediation_r.csv
#   9. run.positional.scans()        → ci_r.csv
#  10. instability.lm.scan() +
#      fixef.eQTL.scan()             → fixef_r.csv
#
# sections 1-3 (haplotype, kinship, LMM scan) are covered by run_validation.R.
#
# usage:
#   Rscript validation/r_reference/port_validation.R [chr] [n_genes]
#   Rscript validation/r_reference/port_validation.R 8 20
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
args      <- commandArgs(trailingOnly = TRUE)
scan_chrs <- if (length(args) >= 1) args[1] else 'all'
n_genes   <- if (length(args) >= 2) as.integer(args[2]) else Inf

message('[init] chr=', scan_chrs,
        '  n_genes=', if (is.infinite(n_genes)) 'all' else n_genes)

# ── paths ─────────────────────────────────────────────────────────────────────
data_dir <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir  <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

genomecache_path <- file.path(data_dir, 'output', 'genomecache')
kinship_path     <- file.path(data_dir, 'output', 'kinship_loco.rds')
pheno_48_path    <- file.path(data_dir, 'output', 'results', 'pheno_48.rds')
pheno_df_path    <- file.path(data_dir, 'output', 'pheno_df.rds')
pmap_path        <- file.path(data_dir, 'input',  'pmap.csv')
gtf_path         <- file.path(data_dir, 'input',  'gencode.vM32.gtf.gz')
snp_db_path      <- file.path(data_dir, 'input',  'snp_db.rds')    # for SNP scan section
output_dir       <- file.path(pkg_dir,  'validation', 'port_validation', 'r_data')

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
message('[paths]  output: ', output_dir)

# ── shared inputs (same as run_validation.R) ──────────────────────────────────
message('[load]')
if (file.exists(pheno_48_path)) {
  pheno_48 <- readRDS(pheno_48_path)
  if (!identical(rownames(pheno_48), pheno_48$subject.name))
    rownames(pheno_48) <- pheno_48$subject.name
} else {
  pheno_df       <- readRDS(pheno_df_path)
  pheno_df$batch <- as.character(pheno_df$batch)
  meta_cols      <- c('sample_id', 'subject.name', 'sex', 'batch', paste0('pc', 1:10))
  gene_cols_all  <- setdiff(colnames(pheno_df), meta_cols)
  pheno_48 <- aggregate(pheno_df[, c(gene_cols_all, paste0('pc', 1:10))],
                        by=list(subject.name=pheno_df$subject.name), FUN=mean, na.rm=TRUE)
  pheno_48$batch     <- tapply(pheno_df$batch, pheno_df$subject.name, `[`, 1)[pheno_48$subject.name]
  rownames(pheno_48) <- pheno_48$subject.name
}
message('  pheno_48: ', nrow(pheno_48), ' strains × ', ncol(pheno_48), ' cols')

kinship_loco <- readRDS(kinship_path)
pmap_flat    <- read.csv(pmap_path, stringsAsFactors=FALSE)
colnames(pmap_flat)[colnames(pmap_flat)=='marker'] <- 'locus'
colnames(pmap_flat)[colnames(pmap_flat)=='pos']    <- 'pos_mb'
pmap_flat$locus    <- sub('\\.RData$', '', pmap_flat$locus)
rownames(pmap_flat) <- pmap_flat$locus

# GTF for gene TSS
use_gtf <- FALSE
if (file.exists(gtf_path) && requireNamespace('rtracklayer', quietly=TRUE)) {
  gtf        <- as.data.frame(rtracklayer::import(gtf_path))
  gtf_genes  <- gtf[gtf$type=='gene',]
  gene_annot <- data.frame(
    gene_id = gtf_genes$gene_name,
    chr     = gsub('^chr','', as.character(gtf_genes$seqnames)),
    tss_mb  = ifelse(gtf_genes$strand=='+', gtf_genes$start, gtf_genes$end) / 1e6,
    stringsAsFactors = FALSE)
  use_gtf <- TRUE
  message('  gtf genes: ', nrow(gene_annot))
}

meta_cols  <- c('subject.name', 'batch', paste0('pc', 1:10))
all_genes  <- head(setdiff(colnames(pheno_48), meta_cols), n_genes)
chromosomes <- if (scan_chrs=='all') names(kinship_loco) else strsplit(scan_chrs, ',')[[1]]
message('[scan] chromosomes: ', paste(sort(chromosomes), collapse=', '))
message('[scan] genes: ', length(all_genes))

cis_window_mb <- 0.5
n_pcs         <- 10

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — collapse.genomecache()
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 4] cache collapse')
collapsed_path <- file.path(data_dir, 'output', 'genomecache_collapsed')
if (!dir.exists(collapsed_path)) {
  message('  collapsing genomecache...')
  # discover actual argument names at runtime to avoid version-specific errors
  fn_args <- names(formals(collapse.genomecache))
  message('  collapse.genomecache formals: ', paste(fn_args, collapse=', '))
  # build call using the first two positional args (genomecache path, output dir)
  # then add model and criterion by name only if they exist in the signature
  call_args <- list(genomecache_path, collapsed_path)
  if ('model'     %in% fn_args) call_args[['model']]     <- 'additive'
  if ('criterion' %in% fn_args) call_args[['criterion']] <- 'euclidean'
  do.call(collapse.genomecache, call_args)
  message('  collapsed cache written to: ', collapsed_path)
} else {
  message('  using existing collapsed cache: ', collapsed_path)
}
# summarise loci counts per chromosome for Python comparison
chr_dirs <- list.dirs(file.path(collapsed_path, 'additive'), full.names=FALSE, recursive=FALSE)
collapse_summary <- data.frame(
  chr         = sub('^chr','', chr_dirs),
  n_loci_orig = sapply(chr_dirs, function(c) length(list.files(file.path(genomecache_path, 'additive', c, 'data')))),
  n_loci_coll = sapply(chr_dirs, function(c) length(list.files(file.path(collapsed_path,  'additive', c, 'data'))))
)
write.csv(collapse_summary, file.path(output_dir, 'collapsed_cache_summary.csv'), row.names=FALSE)
message('  saved: collapsed_cache_summary.csv')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — permutation thresholds (generate.sample.outcomes.matrix + get.gev.thresholds)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 5] permutation thresholds')
# run a quick LMM scan on one gene to get the scan object for permutation
thresh_gene <- all_genes[1]
thresh_chr  <- chromosomes[1]
k_thresh    <- kinship_loco[[thresh_chr]]
thresh_loci <- pmap_flat[pmap_flat$chr == thresh_chr, 'locus']
thresh_form <- as.formula(paste0('`', thresh_gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
thresh_scan <- tryCatch(
  scan.h2lmm(genomecache=genomecache_path, data=pheno_48,
              pheno.id='subject.name', geno.id='subject.name',
              formula=thresh_form, K=k_thresh, use.multi.impute=FALSE,
              model='additive', just.these.loci=thresh_loci,
              return.allele.effects=FALSE),
  error=function(e) { message('  scan failed: ', e$message); NULL })

if (!is.null(thresh_scan)) {
  n_perms <- 100   # small for speed; use 1000 for publication
  set.seed(42)
  perm_outcomes <- generate.sample.outcomes.matrix(
    scan.object = thresh_scan, method='permutation', num.samples=n_perms)
  perm_scans <- run.threshold.scans(
    sim.threshold.object = perm_outcomes,
    genomecache = genomecache_path,
    data = pheno_48,
    use.multi.impute = FALSE)
  thresholds <- get.gev.thresholds(threshold.scans=perm_scans, percentile=0.95)
  thresh_df  <- data.frame(percentile=0.95, lod_threshold=thresholds,
                            gene=thresh_gene, chr=thresh_chr, n_perms=n_perms)
  write.csv(thresh_df, file.path(output_dir, 'thresholds_r.csv'), row.names=FALSE)
  message('  saved: thresholds_r.csv  threshold=', round(thresholds, 3))
} else {
  message('  WARN: threshold scan failed — thresholds_r.csv not written')
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — multiple imputation scan
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 6] multiple imputation')
mi_results <- list()
for (scan_chr in chromosomes) {
  k_chr    <- kinship_loco[[scan_chr]]
  chr_loci <- pmap_flat[pmap_flat$chr==scan_chr, ]
  if (use_gtf) {
    chr_ga   <- gene_annot[gene_annot$chr==scan_chr, ]
    gnames   <- intersect(all_genes, chr_ga$gene_id)
  } else { gnames <- all_genes }
  for (gene in gnames) {
    tss <- if (use_gtf) chr_ga$tss_mb[chr_ga$gene_id==gene][1] else mean(chr_loci$pos_mb)
    if (is.na(tss)) next
    cis <- chr_loci[abs(chr_loci$pos_mb - tss) <= cis_window_mb, 'locus']
    if (!length(cis)) next
    form <- as.formula(paste0('`', gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
    res  <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_48,
                 pheno.id='subject.name', geno.id='subject.name',
                 formula=form, K=k_chr, use.multi.impute=TRUE, num.imp=11,
                 model='additive', just.these.loci=cis),
      error=function(e) NULL)
    if (is.null(res)) next
    mi_results[[paste0(gene,'_',scan_chr)]] <- data.frame(
      gene_id=gene, locus=res$loci, chr=scan_chr,
      pos_mb=pmap_flat$pos_mb[match(res$loci, pmap_flat$locus)],
      lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
  }
}
if (length(mi_results)) {
  mi_df <- do.call(rbind, mi_results); rownames(mi_df) <- NULL
  write.csv(mi_df, file.path(output_dir, 'mi_lod_r.csv'), row.names=FALSE)
  message('  saved: mi_lod_r.csv  (', nrow(mi_df), ' rows)')
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SNP scan
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 7] snp scan')
if (file.exists(snp_db_path)) {
  snp_results <- list()
  for (scan_chr in chromosomes) {
    k_chr    <- kinship_loco[[scan_chr]]
    chr_loci <- pmap_flat[pmap_flat$chr==scan_chr, ]
    if (use_gtf) {
      chr_ga  <- gene_annot[gene_annot$chr==scan_chr, ]
      gnames  <- intersect(all_genes, chr_ga$gene_id)
    } else { gnames <- all_genes }
    for (gene in gnames) {
      tss <- if (use_gtf) chr_ga$tss_mb[chr_ga$gene_id==gene][1] else mean(chr_loci$pos_mb)
      if (is.na(tss)) next
      cis <- chr_loci[abs(chr_loci$pos_mb - tss) <= cis_window_mb, 'locus']
      if (!length(cis)) next
      form <- as.formula(paste0('`', gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
      res  <- tryCatch(
        imputed.snp.scan.h2lmm(genomecache=genomecache_path,
                                snp.db=snp_db_path, data=pheno_48,
                                pheno.id='subject.name', geno.id='subject.name',
                                formula=form, K=k_chr,
                                just.these.loci=cis, chr=scan_chr),
        error=function(e) NULL)
      if (is.null(res)) next
      snp_results[[paste0(gene,'_',scan_chr)]] <- data.frame(
        gene_id=gene, snp=names(res$LOD), chr=scan_chr,
        lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
    }
  }
  if (length(snp_results)) {
    snp_df <- do.call(rbind, snp_results); rownames(snp_df) <- NULL
    write.csv(snp_df, file.path(output_dir, 'snp_lod_r.csv'), row.names=FALSE)
    message('  saved: snp_lod_r.csv  (', nrow(snp_df), ' rows)')
  }
} else {
  message('  WARN: snp_db_path not found — skipping SNP scan')
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — mediation
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 8] mediation')
# requires a pre-computed QR decomposition and a mediator expression matrix
# placeholder: scan one gene pair as a test
med_gene     <- all_genes[1]
med_mediator <- if (length(all_genes) > 1) all_genes[2] else NULL
if (!is.null(med_mediator)) {
  med_chr  <- chromosomes[1]
  k_chr    <- kinship_loco[[med_chr]]
  chr_loci <- pmap_flat[pmap_flat$chr==med_chr, ]
  form     <- as.formula(paste0('`', med_gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
  base_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_48,
               pheno.id='subject.name', geno.id='subject.name',
               formula=form, K=k_chr, use.multi.impute=FALSE,
               model='additive', just.these.loci=chr_loci$locus),
    error=function(e) NULL)
  if (!is.null(base_scan)) {
    peak_locus <- grab.locus.from.scan(base_scan)
    med_form   <- as.formula(paste0('`', med_gene, '` ~ `', med_mediator, '` + batch + ',
                                    paste0('pc', 1:n_pcs, collapse=' + ')))
    med_scan <- tryCatch(
      mediation.scan.qr(genomecache=genomecache_path, data=pheno_48,
                        pheno.id='subject.name', geno.id='subject.name',
                        formula=med_form, K=k_chr,
                        just.these.loci=chr_loci$locus),
      error=function(e) NULL)
    if (!is.null(med_scan)) {
      med_df <- data.frame(gene_id=med_gene, mediator=med_mediator, chr=med_chr,
                           locus=med_scan$loci, lod=med_scan$LOD,
                           p_value=med_scan$p.value, stringsAsFactors=FALSE)
      write.csv(med_df, file.path(output_dir, 'mediation_r.csv'), row.names=FALSE)
      message('  saved: mediation_r.csv  (', nrow(med_df), ' rows)')
    } else { message('  WARN: mediation scan failed') }
  }
} else { message('  WARN: need ≥2 genes for mediation — skipping') }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — positional CI (parametric bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 9] positional CI')
ci_gene <- all_genes[1]; ci_chr <- chromosomes[1]
k_chr   <- kinship_loco[[ci_chr]]
chr_loci <- pmap_flat[pmap_flat$chr==ci_chr, ]
ci_form  <- as.formula(paste0('`', ci_gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
ci_scan  <- tryCatch(
  scan.h2lmm(genomecache=genomecache_path, data=pheno_48,
             pheno.id='subject.name', geno.id='subject.name',
             formula=ci_form, K=k_chr, use.multi.impute=FALSE,
             model='additive', just.these.loci=chr_loci$locus),
  error=function(e) NULL)
if (!is.null(ci_scan)) {
  peak_locus <- grab.locus.from.scan(ci_scan)
  peak_fit   <- scan.h2lmm(genomecache=genomecache_path, data=pheno_48,
                            pheno.id='subject.name', geno.id='subject.name',
                            formula=ci_form, K=k_chr, use.multi.impute=FALSE,
                            model='additive', just.these.loci=peak_locus)
  set.seed(1)
  boot_outcomes <- generate.sample.outcomes.matrix(
    scan.object=peak_fit, model.type='alt', method='bootstrap',
    num.samples=100, use.REML=TRUE, use.BLUP=TRUE)
  boot_scans <- run.positional.scans(
    sim.object=boot_outcomes, genomecache=genomecache_path,
    data=pheno_48, use.multi.impute=FALSE)
  # extract CI boundaries (2.5th / 97.5th percentile of bootstrap peak positions)
  if (!is.null(boot_scans)) {
    boot_peaks <- sapply(boot_scans, function(s) s$loci[which.max(s$LOD)])
    boot_pos   <- pmap_flat$pos_mb[match(boot_peaks, pmap_flat$locus)]
    ci_df <- data.frame(gene_id=ci_gene, chr=ci_chr,
                        peak_locus=peak_locus,
                        ci_lo_mb=quantile(boot_pos, 0.025, na.rm=TRUE),
                        ci_hi_mb=quantile(boot_pos, 0.975, na.rm=TRUE),
                        n_boot=length(boot_peaks))
    write.csv(ci_df, file.path(output_dir, 'ci_r.csv'), row.names=FALSE)
    message('  saved: ci_r.csv')
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — fixed-effect scans
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 10] fixed-effect scans')
fixef_results <- list()
for (scan_chr in chromosomes) {
  chr_loci <- pmap_flat[pmap_flat$chr==scan_chr, ]
  if (use_gtf) {
    chr_ga  <- gene_annot[gene_annot$chr==scan_chr, ]
    gnames  <- intersect(all_genes, chr_ga$gene_id)
  } else { gnames <- all_genes }
  for (gene in gnames) {
    tss <- if (use_gtf) chr_ga$tss_mb[chr_ga$gene_id==gene][1] else mean(chr_loci$pos_mb)
    if (is.na(tss)) next
    cis <- chr_loci[abs(chr_loci$pos_mb - tss) <= cis_window_mb, 'locus']
    if (!length(cis)) next
    form <- as.formula(paste0('`', gene, '` ~ batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
    # instability.lm.scan (bootstrap fixed-effect stability)
    inst_res <- tryCatch(
      instability.lm.scan(genomecache=genomecache_path, data=pheno_48,
                          pheno.id='subject.name', geno.id='subject.name',
                          formula=form, model='additive',
                          just.these.loci=cis, num.samples=100, seed=1),
      error=function(e) NULL)
    if (!is.null(inst_res))
      fixef_results[[paste0(gene,'_inst_',scan_chr)]] <- data.frame(
        gene_id=gene, type='instability', chr=scan_chr,
        locus=inst_res$loci, lod=inst_res$LOD, stringsAsFactors=FALSE)
  }
}
if (length(fixef_results)) {
  fixef_df <- do.call(rbind, fixef_results); rownames(fixef_df) <- NULL
  write.csv(fixef_df, file.path(output_dir, 'fixef_r.csv'), row.names=FALSE)
  message('  saved: fixef_r.csv  (', nrow(fixef_df), ' rows)')
}

cat('\n[done]  outputs in', output_dir, '\n')
