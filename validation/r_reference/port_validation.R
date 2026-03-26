# port_validation.R — complete miQTL reference for mosaiQTL validation
#
# every section produces a CSV that validate_port.py reads and compares
# against the Python mosaiQTL implementation. run all sections together
# or one chromosome at a time.
#
# sections:
#   1.  cache info          → cache_info_r.csv
#   2.  haplotype probs     → haplotype_r.csv
#   3.  kinship loco        → kinship_r.csv
#   4.  lmm scan (rop)      → lod_r.csv + allele_effects_r.csv
#   5.  cache collapse      → collapsed_cache_summary.csv
#   6.  permutation         → thresholds_r.csv
#   7.  multiple imputation → mi_lod_r.csv
#   8.  snp scan            → snp_lod_r.csv  (gated: needs allele_dir)
#   9.  mediation           → mediation_r.csv
#  10.  positional ci       → ci_r.csv
#  11.  instability scan    → fixef_r.csv
#
# usage:
#   Rscript validation/r_reference/port_validation.R [chr] [n_genes]
#   Rscript validation/r_reference/port_validation.R 8 50
# ─────────────────────────────────────────────────────────────────────────────
lib_path <- '/data/homezvol2/gfilimba/R/x86_64-pc-linux-gnu-library/4.4'
if (dir.exists(lib_path)) {
  .libPaths(c(lib_path, .libPaths()))
  for (pkg in c('miqtl','dplyr')) library(pkg, lib.loc=lib_path, character.only=TRUE)
  if (requireNamespace('rtracklayer', quietly=TRUE))
    library(rtracklayer, lib.loc=lib_path)
} else {
  library(miqtl); library(dplyr)
  if (requireNamespace('rtracklayer', quietly=TRUE)) library(rtracklayer)
}

args        <- commandArgs(trailingOnly=TRUE)
scan_chrs   <- if (length(args) >= 1) args[1] else 'all'
n_genes     <- if (length(args) >= 2) as.integer(args[2]) else Inf
# optional 3rd arg: 'slurm' — write per-chr outputs to r_data/chr{N}/
# used by run_validation_slurm.sh to parallelise across chromosomes
slurm_mode  <- length(args) >= 3 && args[3] == 'slurm'
message('[init] chr=', scan_chrs, '  n_genes=',
        if (is.infinite(n_genes)) 'all' else n_genes,
        if (slurm_mode) '  mode=slurm' else '')

# ── paths ─────────────────────────────────────────────────────────────────────
data_dir <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir  <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

genomecache_path <- file.path(data_dir, 'output', 'genomecache')
kinship_path     <- file.path(data_dir, 'output', 'kinship_loco.rds')
pheno_strain_path<- file.path(data_dir, 'output', 'results', 'pheno_48.rds')
pheno_df_path    <- file.path(data_dir, 'output', 'pheno_df.rds')
pmap_path        <- file.path(data_dir, 'input',  'pmap.csv')
gtf_path         <- file.path(data_dir, 'input',  'gencode.vM32.gtf.gz')
allele_dir_path  <- file.path(data_dir, 'output', 'allele_dir')
collapsed_path   <- file.path(data_dir, 'output', 'genomecache_collapsed')
base_output_dir  <- file.path(pkg_dir, 'validation', 'port_validation', 'r_data')
# in slurm mode each chromosome writes to its own subdir to avoid collisions
output_dir <- if (slurm_mode && scan_chrs != 'all') {
  file.path(base_output_dir, paste0('chr', scan_chrs))
} else {
  base_output_dir
}
dir.create(output_dir, showWarnings=FALSE, recursive=TRUE)
message('[paths] output: ', output_dir)

# ── shared inputs ─────────────────────────────────────────────────────────────
message('[load]')
if (file.exists(pheno_strain_path)) {
  pheno_strain <- readRDS(pheno_strain_path)
  if (!identical(rownames(pheno_strain), pheno_strain$subject.name))
    rownames(pheno_strain) <- pheno_strain$subject.name
} else {
  pheno_df  <- readRDS(pheno_df_path)
  meta_cols <- c('sample_id','subject.name','sex','batch', paste0('pc',1:10))
  gcols     <- setdiff(colnames(pheno_df), meta_cols)
  pheno_strain <- aggregate(pheno_df[, c(gcols, paste0('pc',1:10))],
    by=list(subject.name=pheno_df$subject.name), FUN=mean, na.rm=TRUE)
  pheno_strain$batch <- tapply(pheno_df$batch, pheno_df$subject.name,
    `[`, 1)[pheno_strain$subject.name]
  rownames(pheno_strain) <- pheno_strain$subject.name
}
ph_sn  <- rownames(pheno_strain)
ph_ord <- c(ph_sn[!grepl('^CC', ph_sn)],
             ph_sn[ grepl('^CC', ph_sn)])
pheno_strain <- pheno_strain[ph_ord, ]
message('  pheno_strain: ', nrow(pheno_strain), ' × ', ncol(pheno_strain))

kinship_loco <- readRDS(kinship_path)
kinship_loco <- lapply(kinship_loco, function(k) {
  sn     <- rownames(k)
  sn_ord <- c(sn[!grepl('^CC', sn)],
               sn[ grepl('^CC', sn)])
  k[sn_ord, sn_ord]
})
pmap_flat    <- read.csv(pmap_path, stringsAsFactors=FALSE)
colnames(pmap_flat)[colnames(pmap_flat)=='marker'] <- 'locus'
colnames(pmap_flat)[colnames(pmap_flat)=='pos']    <- 'pos_mb'
pmap_flat$locus    <- sub('\\.RData$', '', pmap_flat$locus)
rownames(pmap_flat) <- pmap_flat$locus

use_gtf <- FALSE
if (file.exists(gtf_path) && requireNamespace('rtracklayer', quietly=TRUE)) {
  gtf       <- as.data.frame(rtracklayer::import(gtf_path))
  gtf_g     <- gtf[gtf$type=='gene',]
  gene_annot <- data.frame(
    gene_id = gtf_g$gene_name,
    chr     = gsub('^chr','', as.character(gtf_g$seqnames)),
    tss_mb  = ifelse(gtf_g$strand=='+', gtf_g$start, gtf_g$end)/1e6,
    stringsAsFactors=FALSE)
  use_gtf <- TRUE
  message('  gtf: ', nrow(gene_annot), ' genes')
}

meta_cols   <- c('subject.name','batch', paste0('pc',1:10))
all_genes   <- head(setdiff(colnames(pheno_strain), meta_cols), n_genes)
chromosomes <- if (scan_chrs=='all') names(kinship_loco) else
               strsplit(scan_chrs, ',')[[1]]
cis_window_mb <- 0.5; n_pcs <- 10
message('[scan] chr: ', paste(sort(chromosomes), collapse=', '),
        '  genes: ', length(all_genes))

# helpers
get_cis_loci <- function(gene, chr) {
  chr_loci <- pmap_flat[pmap_flat$chr == chr, ]
  tss <- if (use_gtf) gene_annot$tss_mb[gene_annot$gene_id==gene &
    gene_annot$chr==chr][1] else mean(chr_loci$pos_mb)
  if (is.na(tss) || !length(tss)) return(NULL)
  cis <- chr_loci$locus[abs(chr_loci$pos_mb - tss) <= cis_window_mb]
  if (!length(cis)) NULL else cis
}
make_form <- function(gene)
  as.formula(paste0('`', gene, '` ~ batch + ',
                    paste0('pc', 1:n_pcs, collapse=' + ')))
first_gene_on <- function(chr, n=1) {
  found <- c()
  for (g in all_genes) {
    if (!is.null(get_cis_loci(g, chr))) {
      found <- c(found, g)
      if (length(found) == n) break
    }
  }
  found
}

# ─────────────────────────────────────────────────────────────────────────────
# S1 — CACHE INFO
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s1] cache info')
h <- DiploprobReader$new(genomecache_path)
chr_info <- do.call(rbind, lapply(chromosomes, function(c) {
  load(file.path(genomecache_path,'additive',paste0('chr',c),'markers.RData'))
  load(file.path(genomecache_path,'additive',paste0('chr',c),'map.RData'))
  load(file.path(genomecache_path,'additive',paste0('chr',c),'bp.RData'))
  data.frame(chr=c, n_loci=length(markers),
             min_mb=min(bp/1e6, na.rm=TRUE), max_mb=max(bp/1e6, na.rm=TRUE),
             stringsAsFactors=FALSE)
}))
founders  <- h$getFounders()
sample_ids <- h$getSubjects()
sample_ids <- c(sample_ids[!grepl('^CC', sample_ids)],
                sample_ids[ grepl('^CC', sample_ids)])  
cache_info <- list(
  chr_summary = chr_info,
  founders    = founders,
  n_samples   = length(sample_ids),
  sample_ids  = sample_ids)
write.csv(chr_info, file.path(output_dir,'cache_info_r.csv'), row.names=FALSE)
write.csv(data.frame(founder=founders), file.path(output_dir,'founders_r.csv'),
          row.names=FALSE)
write.csv(data.frame(sample_id=sample_ids),
          file.path(output_dir,'sample_ids_r.csv'), row.names=FALSE)
message('  saved: cache_info_r.csv  founders_r.csv  sample_ids_r.csv')

# ─────────────────────────────────────────────────────────────────────────────
# S2 — HAPLOTYPE PROBS SPOT-CHECK
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s2] haplotype probs')
hap_rows <- list()
for (chr in chromosomes) {
  load(file.path(genomecache_path,'additive',paste0('chr',chr),'markers.RData'))
  load(file.path(genomecache_path,'additive',paste0('chr',chr),'bp.RData'))
  n_loci <- length(markers)
  idx    <- round(seq(1, n_loci, length.out=5))
  for (i in idx) {
    mat <- h$getLocusMatrix(locus=markers[i], model='additive', subjects=NULL)
    mat <- mat[!is.na(rownames(mat)), , drop=FALSE]   # strip NA rows
    row_sums <- rowSums(mat)
    hap_rows[[length(hap_rows)+1]] <- data.frame(
      chr=chr, locus=markers[i], pos_mb=bp[i]/1e6,
      mean_rowsum=mean(row_sums, na.rm=TRUE),
      min_rowsum=unname(min(row_sums, na.rm=TRUE)),
      max_rowsum=unname(max(row_sums, na.rm=TRUE)),
      mean_max_prob=mean(apply(mat, 1, max), na.rm=TRUE),
      stringsAsFactors=FALSE,
      row.names=NULL)
  }
}
hap_df <- do.call(rbind, hap_rows)
write.csv(hap_df, file.path(output_dir,'haplotype_r.csv'), row.names=FALSE)
message('  saved: haplotype_r.csv (', nrow(hap_df), ' loci)')

# ─────────────────────────────────────────────────────────────────────────────
# S3 — KINSHIP LOCO
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s3] kinship loco')
kin_rows <- list()
for (chr in chromosomes) {
  k <- kinship_loco[[chr]]
  kin_rows[[chr]] <- data.frame(
    chr=chr,
    n=nrow(k), mean_diag=mean(diag(k)),
    mean_offdiag=mean(k[lower.tri(k)]),
    min_val=min(k), max_val=max(k),
    stringsAsFactors=FALSE)
}
kin_df <- do.call(rbind, kin_rows)
write.csv(kin_df, file.path(output_dir,'kinship_r.csv'), row.names=FALSE)
# save upper triangle of each chromosome kinship for direct comparison
for (chr in chromosomes) {
  k  <- kinship_loco[[chr]]
  sn <- rownames(k)
  kt <- as.data.frame(k); kt$sample_id <- sn
  write.csv(kt, file.path(output_dir, paste0('kinship_mat_r_chr',chr,'.csv')),
            row.names=FALSE)
}
message('  saved: kinship_r.csv + per-chr kinship matrices')

# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM SCAN (ROP) — all genes on each chromosome
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s4] lmm scan')
results_list <- list(); effects_list <- list()
for (chr in chromosomes) {
  k_chr    <- kinship_loco[[chr]]
  chr_loci <- pmap_flat[pmap_flat$chr==chr, ]
  gnames   <- if (use_gtf) intersect(all_genes,
    gene_annot$gene_id[gene_annot$chr==chr]) else all_genes
  message('  chr', chr, ': ', length(gnames), ' genes  ', nrow(chr_loci), ' loci')
  for (gene in gnames) {
    if (which(gnames == gene) %% 100 == 0)
      message('    gene ', which(gnames == gene),  
              '/', length(gnames))                 
    cis <- get_cis_loci(gene, chr)
    if (is.null(cis)) next
    res <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                  pheno.id='subject.name', geno.id='subject.name',
                  formula=make_form(gene), K=k_chr,
                  use.multi.impute=FALSE, model='additive',
                  just.these.loci=cis, return.allele.effects=TRUE),
      error=function(e) NULL)
    if (is.null(res)) next
    results_list[[paste0(gene,'_',chr)]] <- data.frame(
      gene_id=gene, locus=res$loci, chr=chr,
      pos_mb=pmap_flat$pos_mb[match(res$loci, pmap_flat$locus)],
      lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
    ae <- res$allele.effects
    if (!is.null(ae) && length(res$LOD) > 0) {
      pi <- which.max(res$LOD)
      ef <- if (is.matrix(ae)) ae[pi,] else ae
      if (!is.null(ef) && !is.null(names(ef)))
        effects_list[[paste0(gene,'_',chr)]] <- data.frame(
          gene_id=gene, chr=chr,
          locus=res$loci[min(pi, length(res$loci))],
          founder=names(ef), effect=as.numeric(ef),
          stringsAsFactors=FALSE)
    }
  }
}
if (length(results_list)) {
  lod_df <- do.call(rbind, results_list); rownames(lod_df) <- NULL
  write.csv(lod_df, file.path(output_dir,'lod_r.csv'), row.names=FALSE)
  message('  saved: lod_r.csv (', nrow(lod_df), ' rows, ',
          length(unique(lod_df$gene_id)), ' genes)')
  write.csv(pmap_flat[,c('locus','chr','pos_mb')],
            file.path(output_dir,'pmap_r.csv'), row.names=FALSE)
  write.csv(pheno_strain, file.path(output_dir,'pheno_48_used.csv'), row.names=TRUE)
}
if (length(effects_list)) {
  ae_df <- do.call(rbind, effects_list); rownames(ae_df) <- NULL
  write.csv(ae_df, file.path(output_dir,'allele_effects_r.csv'), row.names=FALSE)
  message('  saved: allele_effects_r.csv (', nrow(ae_df), ' rows)')
}

# ─────────────────────────────────────────────────────────────────────────────
# S5 — CACHE COLLAPSE
# confirmed miqtl:::check.l2.norm: max_i(||h_i,l1-h_i,l2||/sqrt(J)) <= tol
# bugs patched: :::check.l2.norm (not exported) + test <- FALSE init
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s5] cache collapse')
unlockBinding('collapse.genomecache', as.environment('package:miqtl'))
fn_lines <- deparse(body(collapse.genomecache))
fn_lines <- gsub('check\\.l2\\.norm\\(',      'miqtl:::check.l2.norm(',      fn_lines)
fn_lines <- gsub('check\\.max\\.category\\(', 'miqtl:::check.max.category(', fn_lines)
target   <- which(grepl('if \\(criterion == "l2.norm"\\)', fn_lines))
fn_lines <- append(fn_lines, '            test <- FALSE', after=target-1)
body(collapse.genomecache) <- str2lang(paste(fn_lines, collapse='\n'))
lockBinding('collapse.genomecache', as.environment('package:miqtl'))
message('  collapse patched')

needs_collapse <- !dir.exists(collapsed_path) ||
  length(list.files(collapsed_path, recursive=TRUE, pattern='\\.RData$')) == 0
if (needs_collapse) {
  if (dir.exists(collapsed_path)) unlink(collapsed_path, recursive=TRUE)
  message('  collapsing (l2.norm, tol=0.02)...')
  collapse.genomecache(original.cache=genomecache_path, new.cache=collapsed_path,
    subjects=NULL, criterion='l2.norm', model='additive', proportion.tol=0.02)
} else {
  message('  using existing: ', collapsed_path)
}
chr_dirs <- list.dirs(file.path(genomecache_path,'additive'),
                      full.names=FALSE, recursive=FALSE)
first_coll <- file.path(collapsed_path,'additive',chr_dirs[1])
has_data   <- dir.exists(file.path(first_coll,'data')) &&
  length(list.files(file.path(first_coll,'data'), pattern='\\.RData$')) > 0
collapse_summary <- do.call(rbind, lapply(chr_dirs, function(c) {
  n_orig    <- length(list.files(
    file.path(genomecache_path,'additive',c,'data'), pattern='\\.RData$'))
  coll_base <- file.path(collapsed_path,'additive',c)
  n_coll    <- if (has_data) length(list.files(
    file.path(coll_base,'data'), pattern='\\.RData$'))
  else length(list.files(coll_base, pattern='\\.RData$', recursive=FALSE))
  data.frame(chr=sub('^chr','',c), n_loci_orig=n_orig, n_loci_coll=n_coll)
}))
write.csv(collapse_summary, file.path(output_dir,'collapsed_cache_summary.csv'),
          row.names=FALSE)
n_o <- sum(collapse_summary$n_loci_orig); n_c <- sum(collapse_summary$n_loci_coll)
message('  saved: collapsed_cache_summary.csv  ',
        n_o,' → ',n_c,' (',round(100*(n_o-n_c)/n_o,1),'% reduced)')

# ─────────────────────────────────────────────────────────────────────────────
# S6 — PERMUTATION THRESHOLDS
# bypasses generate.sample.outcomes.matrix (chol crash on small windows)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s6] permutation thresholds')
thresh_chr  <- chromosomes[1]
thresh_gene <- first_gene_on(thresh_chr, 1)
if (!length(thresh_gene)) {
  message('  SKIP: no gene with cis loci on chr', thresh_chr)
} else {
  thresh_gene <- thresh_gene[1]
  cis_t   <- get_cis_loci(thresh_gene, thresh_chr)
  k_thresh <- kinship_loco[[thresh_chr]]
  message('  gene=', thresh_gene, '  chr', thresh_chr, '  cis_loci=', length(cis_t))
  n_perms <- 100; set.seed(42); perm_lods <- numeric(n_perms)
  for (i in seq_len(n_perms)) {
    pd <- pheno_strain; pd[[thresh_gene]] <- sample(pheno_strain[[thresh_gene]])
    res <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pd,
                  pheno.id='subject.name', geno.id='subject.name',
                  formula=make_form(thresh_gene), K=k_thresh,
                  use.multi.impute=FALSE, model='additive',
                  just.these.loci=cis_t, return.allele.effects=FALSE),
      error=function(e) NULL)
    perm_lods[i] <- if (!is.null(res)) max(res$LOD, na.rm=TRUE) else NA_real_
    if (i %% 25 == 0) message('  ', i, '/', n_perms)
  }
  perm_lods <- perm_lods[!is.na(perm_lods)]
  # get.gev.thresholds can fail when perm_lods have low variance (few cis loci)
  # fall back to empirical quantile if GEV optim fails
  thresh_95 <- tryCatch(
    get.gev.thresholds(
      list(max.statistics=list(LOD=perm_lods)), percentile=0.95, use.lod=TRUE),
    error=function(e) {
      message('  GEV fit failed (', e$message, ') — using empirical quantile')
      quantile(perm_lods, 0.95, na.rm=TRUE)
    })
  write.csv(data.frame(percentile=0.95, lod_threshold=thresh_95,
                        gene=thresh_gene, chr=thresh_chr,
                        n_perms=length(perm_lods),
                        perm_mean=mean(perm_lods), perm_sd=sd(perm_lods)),
            file.path(output_dir,'thresholds_r.csv'), row.names=FALSE)
  write.csv(data.frame(perm_lod=perm_lods),
            file.path(output_dir,'perm_lods_r.csv'), row.names=FALSE)
  message('  saved: thresholds_r.csv  threshold=', round(thresh_95,3))
}

# ─────────────────────────────────────────────────────────────────────────────
# S7 — MULTIPLE IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s7] multiple imputation')
mi_results <- list()
for (chr in chromosomes) {
  k_chr  <- kinship_loco[[chr]]
  gnames <- if (use_gtf) intersect(all_genes,
    gene_annot$gene_id[gene_annot$chr==chr]) else all_genes
  for (gene in gnames) {
    cis <- get_cis_loci(gene, chr)
    if (is.null(cis)) next
    res <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                  pheno.id='subject.name', geno.id='subject.name',
                  formula=make_form(gene), K=k_chr,
                  use.multi.impute=TRUE, num.imp=11,
                  model='additive', just.these.loci=cis),
      error=function(e) NULL)
    if (is.null(res)) next
    mi_results[[paste0(gene,'_',chr)]] <- data.frame(
      gene_id=gene, locus=res$loci, chr=chr,
      pos_mb=pmap_flat$pos_mb[match(res$loci, pmap_flat$locus)],
      lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
  }
}
if (length(mi_results)) {
  mi_df <- do.call(rbind, mi_results); rownames(mi_df) <- NULL
  write.csv(mi_df, file.path(output_dir,'mi_lod_r.csv'), row.names=FALSE)
  message('  saved: mi_lod_r.csv (',nrow(mi_df),' rows, ',
          length(unique(mi_df$gene_id)),' genes)')
}

# ─────────────────────────────────────────────────────────────────────────────
# S8 — SNP SCAN (gated on allele_dir)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s8] snp scan')
if (!dir.exists(allele_dir_path)) {
  message('  SKIP: allele_dir not found at ', allele_dir_path)
  message('  build with: mosaiqtl.snp_scan.build_allele_dir(snp_genos, allele_dir_path)')
} else {
  snp_results <- list()
  for (chr in chromosomes) {
    k_chr  <- kinship_loco[[chr]]
    gnames <- if (use_gtf) intersect(all_genes,
      gene_annot$gene_id[gene_annot$chr==chr]) else all_genes
    for (gene in gnames) {
      cis <- get_cis_loci(gene, chr)
      if (is.null(cis)) next
      res <- tryCatch(
        imputed.snp.scan.h2lmm(
          data=pheno_strain, formula=make_form(gene), K=k_chr,
          allele.dir=allele_dir_path, genomecache=genomecache_path,
          model='additive', chr=chr, just.these.loci=cis),
        error=function(e) NULL)
      if (is.null(res)) next
      snp_results[[paste0(gene,'_',chr)]] <- data.frame(
        gene_id=gene, snp=names(res$LOD), chr=chr,
        lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
    }
  }
  if (length(snp_results)) {
    snp_df <- do.call(rbind, snp_results); rownames(snp_df) <- NULL
    write.csv(snp_df, file.path(output_dir,'snp_lod_r.csv'), row.names=FALSE)
    message('  saved: snp_lod_r.csv (',nrow(snp_df),' rows)')
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S9 — MEDIATION
# workflow: scan → peak locus → extract.mediation.qr(locus) → mediation.scan.qr
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s9] mediation')
med_chr   <- chromosomes[1]
med_genes <- first_gene_on(med_chr, 2)
if (length(med_genes) < 2) {
  message('  SKIP: need ≥ 2 genes with cis loci on chr', med_chr)
} else {
  med_gene <- med_genes[1]; med_med <- med_genes[2]
  k_chr    <- kinship_loco[[med_chr]]
  cis_med  <- get_cis_loci(med_gene, med_chr)
  base_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                pheno.id='subject.name', geno.id='subject.name',
                formula=make_form(med_gene), K=k_chr,
                use.multi.impute=FALSE, model='additive',
                just.these.loci=cis_med),
    error=function(e) NULL)
  med_peak <- if (!is.null(base_scan)) grab.locus.from.scan(base_scan) else NULL
  if (!is.null(med_peak)) {
    med_form <- as.formula(paste0('`', med_gene, '` ~ `', med_med, '` + batch + ',
      paste0('pc', 1:n_pcs, collapse=' + ')))
    qr_obj <- tryCatch(
      extract.mediation.qr(genomecache=genomecache_path, data=pheno_strain,
        formula=med_form, locus=med_peak, chr=med_chr, id='subject.name'),
      error=function(e) { message('  extract.mediation.qr: ', e$message); NULL })
    if (!is.null(qr_obj)) {
      med_res <- tryCatch(
        mediation.scan.qr(mediation.qr.object=qr_obj, data=pheno_strain,
          phenotype=med_gene, chr=med_chr, id='subject.name'),
        error=function(e) { message('  mediation.scan.qr: ', e$message); NULL })
      if (!is.null(med_res)) {
        write.csv(data.frame(
          gene_id=med_gene, mediator=med_med, chr=med_chr,
          locus=med_res$loci,
          pos_mb=pmap_flat$pos_mb[match(med_res$loci, pmap_flat$locus)],
          lod=med_res$LOD, p_value=med_res$p.value, peak_locus=med_peak,
          stringsAsFactors=FALSE),
          file.path(output_dir,'mediation_r.csv'), row.names=FALSE)
        message('  saved: mediation_r.csv')
      }
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S10 — POSITIONAL CI (parametric bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s10] positional CI')
ci_chr  <- chromosomes[1]
ci_gene <- first_gene_on(ci_chr, 1)
if (!length(ci_gene)) {
  message('  SKIP: no gene with cis loci on chr', ci_chr)
} else {
  ci_gene <- ci_gene[1]; k_chr <- kinship_loco[[ci_chr]]
  cis_ci  <- get_cis_loci(ci_gene, ci_chr)
  ci_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                pheno.id='subject.name', geno.id='subject.name',
                formula=make_form(ci_gene), K=k_chr,
                use.multi.impute=FALSE, model='additive',
                just.these.loci=cis_ci),
    error=function(e) NULL)
  if (!is.null(ci_scan)) {
    peak_locus <- grab.locus.from.scan(ci_scan)
    peak_fit   <- scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
      pheno.id='subject.name', geno.id='subject.name',
      formula=make_form(ci_gene), K=k_chr,
      use.multi.impute=FALSE, model='additive', just.these.loci=peak_locus)
    set.seed(1)
    boot_obj <- tryCatch(
      generate.sample.outcomes.matrix(scan.object=peak_fit, model.type='alt',
        method='bootstrap', num.samples=100, seed=1,
        use.REML=TRUE, use.BLUP=FALSE),
      error=function(e) { message('  bootstrap failed: ', e$message); NULL })
    if (!is.null(boot_obj)) {
      boot_scans <- run.positional.scans(sim.object=boot_obj,
        genomecache=genomecache_path, data=pheno_strain,
        use.multi.impute=FALSE, scan.seed=1)
      boot_peaks <- sapply(boot_scans, function(s) s$loci[which.max(s$LOD)])
      boot_pos   <- pmap_flat$pos_mb[match(boot_peaks, pmap_flat$locus)]
      write.csv(data.frame(
        gene_id=ci_gene, chr=ci_chr, peak_locus=peak_locus,
        ci_lo_mb=quantile(boot_pos, 0.025, na.rm=TRUE),
        ci_hi_mb=quantile(boot_pos, 0.975, na.rm=TRUE),
        peak_lod=max(ci_scan$LOD, na.rm=TRUE),
        n_boot=length(boot_peaks)),
        file.path(output_dir,'ci_r.csv'), row.names=FALSE)
      write.csv(data.frame(boot_pos=boot_pos),
                file.path(output_dir,'boot_pos_r.csv'), row.names=FALSE)
      message('  saved: ci_r.csv  boot_pos_r.csv')
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S11 — FIXED-EFFECT / INSTABILITY SCAN
# workflow: scan → generate.simple.sample.outcomes.matrix → instability.lm.scan
# ─────────────────────────────────────────────────────────────────────────────
message('\n[s11] instability scan')
fx_chr  <- chromosomes[1]
fx_gene <- first_gene_on(fx_chr, 1)
if (!length(fx_gene)) {
  message('  SKIP: no gene with cis loci on chr', fx_chr)
} else {
  fx_gene <- fx_gene[1]; k_chr <- kinship_loco[[fx_chr]]
  cis_fx  <- get_cis_loci(fx_gene, fx_chr)
  fx_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                pheno.id='subject.name', geno.id='subject.name',
                formula=make_form(fx_gene), K=k_chr,
                use.multi.impute=FALSE, model='additive',
                just.these.loci=cis_fx),
    error=function(e) NULL)
  if (!is.null(fx_scan)) {
    set.seed(1)
    simple_obj <- tryCatch(
      generate.simple.sample.outcomes.matrix(
        scan.object=fx_scan, subsample.prop=0.63, num.samples=100, seed=1),
      error=function(e) { message('  simple sample failed: ', e$message); NULL })
    if (!is.null(simple_obj)) {
      inst_res <- tryCatch(
        instability.lm.scan(simple.sample.object=simple_obj,
          genomecache=genomecache_path, model='additive',
          seed=1, just.these.loci=cis_fx),
        error=function(e) { message('  instability failed: ', e$message); NULL })
      if (!is.null(inst_res)) {
        write.csv(data.frame(
          gene_id=fx_gene, chr=fx_chr, locus=inst_res$loci,
          pos_mb=pmap_flat$pos_mb[match(inst_res$loci, pmap_flat$locus)],
          lod=inst_res$LOD, stringsAsFactors=FALSE),
          file.path(output_dir,'fixef_r.csv'), row.names=FALSE)
        message('  saved: fixef_r.csv (', length(inst_res$loci), ' loci)')
      }
    }
  }
}

cat('\n[done] all outputs in', output_dir, '\n')
