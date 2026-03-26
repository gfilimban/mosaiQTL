# port_validation.R
# miQTL R reference outputs for mosaiQTL Phase 2 validation (sections 4-10).
# sections 1-3 (haplotype, kinship, LMM) are handled by run_validation.R.
#
# section map:
#   4.  collapse.genomecache()         → collapsed_cache_summary.csv
#   5.  permutation thresholds         → thresholds_r.csv
#   6.  scan.h2lmm(use.multi.impute)   → mi_lod_r.csv
#   7.  imputed.snp.scan.h2lmm()       → snp_lod_r.csv   (gated on allele.dir)
#   8.  mediation.scan.qr()            → mediation_r.csv
#   9.  run.positional.scans()         → ci_r.csv
#  10.  instability.lm.scan()          → fixef_r.csv
#
# confirmed miQTL signatures (via writeLines(deparse(formals(...)))):
#   collapse.genomecache: original.cache, new.cache, subjects,
#                         criterion, model, proportion.tol
#   collapse.genomecache internal bug: 'test' used before assignment —
#     patched at runtime by inserting test <- FALSE before criterion block
#   instability.lm.scan: simple.sample.object, genomecache, model, seed,
#                         use.ROP, num.imp, chr, just.these.loci, ...
#   mediation.scan.qr:   mediation.qr.object, data, phenotype, chr, id, ...
#   generate.sample.outcomes.matrix: scan.object, model.type, method,
#     subsample.prop, subsample.chr, use.REML, use.BLUP, num.samples, seed
#   generate.simple.sample.outcomes.matrix: similar but for instability
#
# usage:
#   Rscript validation/r_reference/port_validation.R [chr] [n_genes]
#   Rscript validation/r_reference/port_validation.R 8 20
# ─────────────────────────────────────────────────────────────────────────────

lib_path <- '/data/homezvol2/gfilimba/R/x86_64-pc-linux-gnu-library/4.4'
if (dir.exists(lib_path)) {
  .libPaths(c(lib_path, .libPaths()))
  library(miqtl, lib.loc=lib_path); library(dplyr, lib.loc=lib_path)
  if (requireNamespace('rtracklayer', quietly=TRUE))
    library(rtracklayer, lib.loc=lib_path)
} else {
  library(miqtl); library(dplyr)
  if (requireNamespace('rtracklayer', quietly=TRUE)) library(rtracklayer)
}

args      <- commandArgs(trailingOnly=TRUE)
scan_chrs <- if (length(args) >= 1) args[1] else 'all'
n_genes   <- if (length(args) >= 2) as.integer(args[2]) else Inf
message('[init] chr=', scan_chrs, '  n_genes=',
        if (is.infinite(n_genes)) 'all' else n_genes)

# ── paths ─────────────────────────────────────────────────────────────────────
data_dir <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
pkg_dir  <- '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl'

genomecache_path <- file.path(data_dir, 'output', 'genomecache')
kinship_path     <- file.path(data_dir, 'output', 'kinship_loco.rds')
pheno_strain_path    <- file.path(data_dir, 'output', 'results', 'pheno_strain.rds')
pheno_df_path    <- file.path(data_dir, 'output', 'pheno_df.rds')
pmap_path        <- file.path(data_dir, 'input',  'pmap.csv')
gtf_path         <- file.path(data_dir, 'input',  'gencode.vM32.gtf.gz')
allele_dir_path  <- file.path(data_dir, 'output', 'allele_dir')
output_dir       <- file.path(pkg_dir, 'validation', 'port_validation', 'r_data')

dir.create(output_dir, showWarnings=FALSE, recursive=TRUE)
message('[paths] output: ', output_dir)

# ── load shared inputs ────────────────────────────────────────────────────────
message('[load]')
if (file.exists(pheno_strain_path)) {
  pheno_strain <- readRDS(pheno_strain_path)
  if (!identical(rownames(pheno_strain), pheno_strain$subject.name))
    rownames(pheno_strain) <- pheno_strain$subject.name
} else {
  pheno_df  <- readRDS(pheno_df_path)
  meta_cols <- c('sample_id','subject.name','sex','batch', paste0('pc',1:10))
  gcols     <- setdiff(colnames(pheno_df), meta_cols)
  pheno_strain  <- aggregate(pheno_df[, c(gcols, paste0('pc',1:10))],
                         by=list(subject.name=pheno_df$subject.name),
                         FUN=mean, na.rm=TRUE)
  pheno_strain$batch     <- tapply(pheno_df$batch, pheno_df$subject.name,
                                `[`, 1)[pheno_strain$subject.name]
  rownames(pheno_strain) <- pheno_strain$subject.name
}
message('  pheno_strain: ', nrow(pheno_strain), ' strains × ', ncol(pheno_strain), ' cols')

kinship_loco <- readRDS(kinship_path)
pmap_flat    <- read.csv(pmap_path, stringsAsFactors=FALSE)
colnames(pmap_flat)[colnames(pmap_flat)=='marker'] <- 'locus'
colnames(pmap_flat)[colnames(pmap_flat)=='pos']    <- 'pos_mb'
pmap_flat$locus     <- sub('\\.RData$', '', pmap_flat$locus)
rownames(pmap_flat)  <- pmap_flat$locus

use_gtf <- FALSE
if (file.exists(gtf_path) && requireNamespace('rtracklayer', quietly=TRUE)) {
  gtf      <- as.data.frame(rtracklayer::import(gtf_path))
  gtf_g    <- gtf[gtf$type=='gene',]
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
cis_window_mb <- 0.5
n_pcs         <- 10
message('[scan] chromosomes: ', paste(sort(chromosomes), collapse=', '))
message('[scan] genes: ', length(all_genes))

# helper: get cis loci for a gene on a chromosome
get_cis_loci <- function(gene, chr) {
  chr_loci <- pmap_flat[pmap_flat$chr == chr, ]
  tss <- if (use_gtf) {
    gene_annot$tss_mb[gene_annot$gene_id==gene & gene_annot$chr==chr][1]
  } else mean(chr_loci$pos_mb)
  if (is.na(tss) || length(tss)==0) return(NULL)
  cis <- chr_loci$locus[abs(chr_loci$pos_mb - tss) <= cis_window_mb]
  if (!length(cis)) return(NULL)
  cis
}

# helper: build formula
make_form <- function(gene) {
  as.formula(paste0('`', gene, '` ~ batch + ',
                    paste0('pc', 1:n_pcs, collapse=' + ')))
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — collapse.genomecache()
#
# bug in miqtl: variable 'test' used before assignment when check.l2.norm()
# throws an error. fixed by inserting test <- FALSE before the criterion block
# via unlockBinding + str2lang.
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 4] cache collapse')
collapsed_path <- file.path(data_dir, 'output', 'genomecache_collapsed')

unlockBinding('collapse.genomecache', as.environment('package:miqtl'))
fn_lines <- deparse(body(collapse.genomecache))
target   <- which(grepl('if \\(criterion == "l2.norm"\\)', fn_lines))
fn_lines <- append(fn_lines, '            test <- FALSE', after=target-1)
body(collapse.genomecache) <- str2lang(paste(fn_lines, collapse='\n'))
lockBinding('collapse.genomecache', as.environment('package:miqtl'))
message('  miqtl collapse bug patched')

if (!dir.exists(collapsed_path)) {
  message('  collapsing (l2.norm, proportion.tol=0.02)...')
  collapse.genomecache(
    original.cache = genomecache_path,
    new.cache      = collapsed_path,
    subjects       = NULL,
    criterion      = 'l2.norm',
    model          = 'additive',
    proportion.tol = 0.02
  )
}

chr_dirs <- list.dirs(file.path(genomecache_path, 'additive'),
                      full.names=FALSE, recursive=FALSE)
# detect whether collapsed cache uses data/ subdir or flat layout per chr
first_coll <- file.path(collapsed_path, 'additive', chr_dirs[1])
has_data   <- dir.exists(file.path(first_coll, 'data')) &&
  length(list.files(file.path(first_coll, 'data'), pattern='\\.RData$')) > 0
message('  collapsed layout: ',
        if (has_data) 'additive/chrN/data/' else 'additive/chrN/ (flat)')
collapse_summary <- do.call(rbind, lapply(chr_dirs, function(c) {
  n_orig    <- length(list.files(
    file.path(genomecache_path, 'additive', c, 'data'), pattern='\\.RData$'))
  coll_base <- file.path(collapsed_path, 'additive', c)
  n_coll    <- if (has_data) {
    length(list.files(file.path(coll_base, 'data'), pattern='\\.RData$'))
  } else {
    length(list.files(coll_base, pattern='\\.RData$', recursive=FALSE))
  }
  data.frame(chr=sub('^chr','',c), n_loci_orig=n_orig, n_loci_coll=n_coll)
}))
write.csv(collapse_summary, file.path(output_dir,'collapsed_cache_summary.csv'),
          row.names=FALSE)
message('  saved: collapsed_cache_summary.csv')
print(collapse_summary)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — permutation thresholds
#
# generate.sample.outcomes.matrix calls chol() even for method='permutation'
# when computing BLUP weights — crashes on near-singular kinship from small
# cis-window scans. bypass: permute phenotype labels directly in a loop.
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 5] permutation thresholds')
thresh_gene <- all_genes[1]
thresh_chr  <- chromosomes[1]
k_thresh    <- kinship_loco[[thresh_chr]]
thresh_form <- make_form(thresh_gene)

tss_t <- if (use_gtf) gene_annot$tss_mb[gene_annot$gene_id==thresh_gene &
             gene_annot$chr==thresh_chr][1] else
          median(pmap_flat$pos_mb[pmap_flat$chr==thresh_chr], na.rm=TRUE)
cis_t <- get_cis_loci(thresh_gene, thresh_chr)
if (is.null(cis_t)) cis_t <- pmap_flat$locus[pmap_flat$chr==thresh_chr][1:50]
message('  gene=', thresh_gene, '  chr', thresh_chr,
        '  cis_loci=', length(cis_t))

n_perms <- 100; set.seed(42); perm_lods <- numeric(n_perms)
for (i in seq_len(n_perms)) {
  pd <- pheno_strain; pd[[thresh_gene]] <- sample(pheno_strain[[thresh_gene]])
  res <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pd,
                pheno.id='subject.name', geno.id='subject.name',
                formula=thresh_form, K=k_thresh,
                use.multi.impute=FALSE, model='additive',
                just.these.loci=cis_t, return.allele.effects=FALSE),
    error=function(e) NULL)
  perm_lods[i] <- if (!is.null(res)) max(res$LOD, na.rm=TRUE) else NA_real_
  if (i %% 25 == 0) message('  ', i, '/', n_perms)
}
perm_lods <- perm_lods[!is.na(perm_lods)]
# get.gev.thresholds expects list(max.statistics=...) not a plain numeric vector
thresh_95 <- get.gev.thresholds(list(max.statistics=perm_lods), percentile=0.95)
write.csv(data.frame(percentile=0.95, lod_threshold=thresh_95,
                      gene=thresh_gene, chr=thresh_chr,
                      n_perms=length(perm_lods)),
          file.path(output_dir,'thresholds_r.csv'), row.names=FALSE)
message('  saved: thresholds_r.csv  threshold=', round(thresh_95,3))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — multiple imputation scan
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 6] multiple imputation')
mi_results <- list()
for (scan_chr in chromosomes) {
  k_chr  <- kinship_loco[[scan_chr]]
  gnames <- if (use_gtf) intersect(all_genes,
    gene_annot$gene_id[gene_annot$chr==scan_chr]) else all_genes
  for (gene in gnames) {
    cis <- get_cis_loci(gene, scan_chr)
    if (is.null(cis)) next
    res <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                  pheno.id='subject.name', geno.id='subject.name',
                  formula=make_form(gene), K=k_chr,
                  use.multi.impute=TRUE, num.imp=11,
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
  write.csv(mi_df, file.path(output_dir,'mi_lod_r.csv'), row.names=FALSE)
  message('  saved: mi_lod_r.csv (', nrow(mi_df), ' rows)')
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SNP scan (gated on allele.dir existence)
#
# miQTL's imputed.snp.scan.h2lmm() reads founder allele calls from a
# directory of per-allele .RData files (allele.dir), not a DataFrame.
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 7] snp scan')
if (!dir.exists(allele_dir_path)) {
  message('  SKIP: allele_dir not found at ', allele_dir_path)
  message('  to enable: create allele_dir with per-allele .RData files')
} else {
  snp_results <- list()
  for (scan_chr in chromosomes) {
    k_chr  <- kinship_loco[[scan_chr]]
    gnames <- if (use_gtf) intersect(all_genes,
      gene_annot$gene_id[gene_annot$chr==scan_chr]) else all_genes
    for (gene in gnames) {
      cis <- get_cis_loci(gene, scan_chr)
      if (is.null(cis)) next
      res <- tryCatch(
        imputed.snp.scan.h2lmm(
          data=pheno_strain, formula=make_form(gene), K=k_chr,
          allele.dir=allele_dir_path, genomecache=genomecache_path,
          model='additive', chr=scan_chr, just.these.loci=cis),
        error=function(e) NULL)
      if (is.null(res)) next
      snp_results[[paste0(gene,'_',scan_chr)]] <- data.frame(
        gene_id=gene, snp=names(res$LOD), chr=scan_chr,
        lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
    }
  }
  if (length(snp_results)) {
    snp_df <- do.call(rbind, snp_results); rownames(snp_df) <- NULL
    write.csv(snp_df, file.path(output_dir,'snp_lod_r.csv'), row.names=FALSE)
    message('  saved: snp_lod_r.csv (', nrow(snp_df), ' rows)')
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — mediation scan
#
# miQTL interface:
#   extract.mediation.qr(genomecache, data, ...)  → qr object
#   mediation.scan.qr(mediation.qr.object, data, phenotype=..., chr=..., ...)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 8] mediation')
med_gene     <- all_genes[1]
med_mediator <- if (length(all_genes) > 1) all_genes[2] else NULL
med_chr      <- chromosomes[1]
if (!is.null(med_mediator)) {
  k_chr <- kinship_loco[[med_chr]]
  # step 1: precompute QR decompositions for all loci on this chromosome
  # formula includes mediator as covariate — null model for mediation test
  med_null_form <- as.formula(paste0('`', med_gene, '` ~ `', med_mediator,
    '` + batch + ', paste0('pc', 1:n_pcs, collapse=' + ')))
  qr_obj <- tryCatch(
    extract.mediation.qr(
      genomecache = genomecache_path,
      data        = pheno_strain,
      formula     = med_null_form,
      K           = k_chr,
      chr         = med_chr,
      id          = 'subject.name'),
    error=function(e) { message('  extract.mediation.qr failed: ', e$message); NULL })

  if (!is.null(qr_obj)) {
    # step 2: run mediation scan using precomputed QR
    med_res <- tryCatch(
      mediation.scan.qr(
        mediation.qr.object = qr_obj,
        data                = pheno_strain,
        phenotype           = med_gene,
        chr                 = med_chr,
        id                  = 'subject.name'),
      error=function(e) { message('  mediation.scan.qr failed: ', e$message); NULL })

    if (!is.null(med_res)) {
      med_df <- data.frame(
        gene_id=med_gene, mediator=med_mediator, chr=med_chr,
        locus=med_res$loci,
        pos_mb=pmap_flat$pos_mb[match(med_res$loci, pmap_flat$locus)],
        lod=med_res$LOD, p_value=med_res$p.value, stringsAsFactors=FALSE)
      write.csv(med_df, file.path(output_dir,'mediation_r.csv'), row.names=FALSE)
      message('  saved: mediation_r.csv (', nrow(med_df), ' rows)')
    }
  }
} else { message('  SKIP: need ≥ 2 genes for mediation') }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — positional CI (parametric bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 9] positional CI')
ci_gene <- all_genes[1]; ci_chr <- chromosomes[1]
k_chr   <- kinship_loco[[ci_chr]]
cis_ci  <- get_cis_loci(ci_gene, ci_chr)
if (!is.null(cis_ci)) {
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
                              use.multi.impute=FALSE, model='additive',
                              just.these.loci=peak_locus)
    set.seed(1)
    # generate.sample.outcomes.matrix with model.type='alt' and method='bootstrap'
    # samples from the fitted alternative model (with QTL effect) — uses BLUP
    boot_obj <- tryCatch(
      generate.sample.outcomes.matrix(
        scan.object = peak_fit, model.type='alt', method='bootstrap',
        num.samples=100, seed=1, use.REML=TRUE, use.BLUP=TRUE),
      error=function(e) { message('  bootstrap sampling failed: ', e$message); NULL })

    if (!is.null(boot_obj)) {
      boot_scans <- run.positional.scans(
        sim.object=boot_obj, genomecache=genomecache_path,
        data=pheno_strain, use.multi.impute=FALSE, scan.seed=1)
      boot_peaks <- sapply(boot_scans, function(s) s$loci[which.max(s$LOD)])
      boot_pos   <- pmap_flat$pos_mb[match(boot_peaks, pmap_flat$locus)]
      ci_df <- data.frame(
        gene_id=ci_gene, chr=ci_chr, peak_locus=peak_locus,
        ci_lo_mb=quantile(boot_pos, 0.025, na.rm=TRUE),
        ci_hi_mb=quantile(boot_pos, 0.975, na.rm=TRUE),
        n_boot=length(boot_peaks))
      write.csv(ci_df, file.path(output_dir,'ci_r.csv'), row.names=FALSE)
      message('  saved: ci_r.csv')
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — fixed-effect / instability scan
#
# miQTL interface:
#   generate.simple.sample.outcomes.matrix(...) → simple.sample.object
#   instability.lm.scan(simple.sample.object, genomecache, ...)
# ─────────────────────────────────────────────────────────────────────────────
message('\n[section 10] fixed-effect instability scan')
fx_gene <- all_genes[1]; fx_chr <- chromosomes[1]
k_chr   <- kinship_loco[[fx_chr]]
cis_fx  <- get_cis_loci(fx_gene, fx_chr)

if (!is.null(cis_fx)) {
  fx_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                pheno.id='subject.name', geno.id='subject.name',
                formula=make_form(fx_gene), K=k_chr,
                use.multi.impute=FALSE, model='additive',
                just.these.loci=cis_fx),
    error=function(e) NULL)

  if (!is.null(fx_scan)) {
    set.seed(1)
    # generate.simple.sample.outcomes.matrix: for instability.lm.scan
    # uses subsample (0.63 proportion) not bootstrap
    simple_obj <- tryCatch(
      generate.simple.sample.outcomes.matrix(
        scan.object=fx_scan, method='subsample',
        subsample.prop=0.63, num.samples=100, seed=1),
      error=function(e) { message('  simple sample failed: ', e$message); NULL })

    if (!is.null(simple_obj)) {
      inst_res <- tryCatch(
        instability.lm.scan(
          simple.sample.object = simple_obj,
          genomecache          = genomecache_path,
          model                = 'additive',
          seed                 = 1,
          just.these.loci      = cis_fx),
        error=function(e) { message('  instability scan failed: ', e$message); NULL })

      if (!is.null(inst_res)) {
        fx_df <- data.frame(
          gene_id=fx_gene, type='instability', chr=fx_chr,
          locus=inst_res$loci,
          pos_mb=pmap_flat$pos_mb[match(inst_res$loci, pmap_flat$locus)],
          lod=inst_res$LOD, stringsAsFactors=FALSE)
        write.csv(fx_df, file.path(output_dir,'fixef_r.csv'), row.names=FALSE)
        message('  saved: fixef_r.csv (', nrow(fx_df), ' rows)')
      }
    }
  }
}

cat('\n[done] outputs in', output_dir, '\n')
