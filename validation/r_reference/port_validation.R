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
#   5.  permutation         → thresholds_r.csv
#   6.  cache collapse      → collapsed_cache_summary.csv
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
# lib_path: set R_LIB_PATH env var on HPC, or falls back to .libPaths()
lib_path <- Sys.getenv('R_LIB_PATH', unset='/Users/ghassan/miniconda3/envs/bioinfo/lib/R/library')
if (dir.exists(lib_path)) .libPaths(c(lib_path, .libPaths()))
for (pkg in c('miqtl','dplyr')) library(pkg, character.only=TRUE)
# no lib.loc here — .libPaths() already includes lib_path above
if (requireNamespace('rtracklayer', quietly=TRUE)) library(rtracklayer)

# ─────────────────────────────────────────────────────────────────────────────

# ── crossprod diagnostic and fix ──────────────────────────────────────────────
# root cause: Matrix::crossprod (a closure) masks base::crossprod (a builtin)
# when Matrix loads. miqtl contains an internal check: typeof(crossprod)=="builtin"
# that fails because crossprod resolves to the Matrix version in the search path.
# fix: restore base::crossprod in the global env so the check passes.
message('[fix] crossprod typeof: ', typeof(crossprod),
        '  is.primitive: ', is.primitive(crossprod))
if (!is.primitive(crossprod) || typeof(crossprod) != 'builtin') {
  message('[fix] crossprod masked by ', environmentName(environment(crossprod)),
          ' — restoring base version')
  assign('crossprod',  base::crossprod,  envir=globalenv())
  assign('tcrossprod', base::tcrossprod, envir=globalenv())
  message('[fix] typeof after restore (globalenv): ', typeof(get('crossprod', envir=globalenv())))
}
# deep scan: try to patch within miqtl namespace as well
.patch_miqtl_crossprod <- function() {
  ns <- asNamespace('miqtl')
  fns <- ls(ns, all.names=TRUE)
  patched <- 0L
  for (fn in fns) {
    obj <- get(fn, envir=ns)
    if (!is.function(obj)) next
    env <- environment(obj)
    if (is.null(env)) next
    if (exists('crossprod', envir=env, inherits=FALSE)) {
      if (!is.primitive(get('crossprod', envir=env, inherits=FALSE))) {
        try({
          assign('crossprod',  base::crossprod,  envir=env)
          assign('tcrossprod', base::tcrossprod, envir=env)
          patched <- patched + 1L
        }, silent=TRUE)
      }
    }
  }
  if (patched > 0) message('[fix] patched crossprod in ', patched, ' miqtl function environments')
}
.patch_miqtl_crossprod()

# enable traceback on first error for diagnosis — remove after fix confirmed
# options(error=function() { traceback(3); cat('\n') })

args        <- commandArgs(trailingOnly=TRUE)
scan_chrs   <- if (length(args) >= 1) args[1] else 'all'
n_genes     <- if (length(args) >= 2 && !args[2] %in% c('','NA','all','Inf')) {
                 v <- suppressWarnings(as.integer(args[2]))
                 if (is.na(v)) Inf else v
               } else Inf
# optional extra args (order-independent, position 3+):
#   'slurm'         — write per-chr outputs to r_data/chr{N}/
#   '11' or '8,11'  — comma-separated section numbers to run (default: all)
# usage examples:
#   Rscript port_validation.R 8 50 11        # only s11
#   Rscript port_validation.R 8 50 8,11      # s8 and s11
#   Rscript port_validation.R 8 50 slurm 11  # s11 in slurm mode
slurm_mode   <- any(args == 'slurm')
run_sections <- {
  # only inspect args[3+] — args[1]=chr and args[2]=n_genes are positional
  extra <- if (length(args) >= 3) args[3:length(args)] else character(0)
  sec_arg <- extra[grepl('^[0-9,]+$', extra) & nchar(extra) > 0]
  if (length(sec_arg)) as.integer(strsplit(sec_arg[1], ',')[[1]]) else 1:11
}
message('[init] chr=', scan_chrs, '  n_genes=',
        if (is.infinite(n_genes)) 'all' else n_genes,
        '  sections=', paste(run_sections, collapse=','),
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

message('  gtf_path exists: ', file.exists(gtf_path))
message('  rtracklayer available: ', requireNamespace('rtracklayer', quietly=TRUE))
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
  # deduplicate: keep one entry per gene_id (first occurrence = primary annotation)
  gene_annot <- gene_annot[!duplicated(gene_annot$gene_id), ]
  message('  gtf after dedup: ', nrow(gene_annot), ' unique genes')
}

meta_cols   <- c('subject.name','batch', paste0('pc',1:10))
all_genes   <- if (is.infinite(n_genes)) setdiff(colnames(pheno_strain), meta_cols) else head(setdiff(colnames(pheno_strain), meta_cols), n_genes)
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

# ── shared genomecache handle — required by multiple sections ────────────────
# initialised here (outside section guards) so per-chr array jobs that skip
# S1 still have h, founders, and sample_ids available
h              <- DiploprobReader$new(genomecache_path)
founders       <- h$getFounders()
founder_map    <- c(A='AJ',B='B6J',C='129S1J',D='NODJ',E='NZOJ',F='CASTJ',G='PWKJ',H='WSBJ')
founders_full  <- ifelse(founders %in% names(founder_map), founder_map[founders], founders)
sample_ids     <- h$getSubjects()
sample_ids     <- c(sample_ids[!grepl('^CC', sample_ids)],
                    sample_ids[ grepl('^CC', sample_ids)])
message('[init] h loaded: ', length(h$getFounders()), ' founders  ',
        length(h$getSubjects()), ' subjects')

# ─────────────────────────────────────────────────────────────────────────────
# S1 — CACHE INFO
# ─────────────────────────────────────────────────────────────────────────────
if (1 %in% run_sections) {
message('\n[s1] cache info')
chr_info <- do.call(rbind, lapply(chromosomes, function(c) {
  load(file.path(genomecache_path,'additive',paste0('chr',c),'markers.RData'))
  load(file.path(genomecache_path,'additive',paste0('chr',c),'map.RData'))
  load(file.path(genomecache_path,'additive',paste0('chr',c),'bp.RData'))
  data.frame(chr=c, n_loci=length(markers),
             min_mb=min(bp/1e6, na.rm=TRUE), max_mb=max(bp/1e6, na.rm=TRUE),
             stringsAsFactors=FALSE)
}))
# h, founders, founder_map, founders_full, sample_ids — defined in shared setup above
cache_info <- list(
  chr_summary = chr_info,
  founders    = founders,
  n_samples   = length(sample_ids),
  sample_ids  = sample_ids)
write.csv(chr_info, file.path(output_dir,'cache_info_r.csv'), row.names=FALSE)
write.csv(data.frame(founder_code=founders, founder=founders_full),
          file.path(output_dir,'founders_r.csv'), row.names=FALSE)
write.csv(data.frame(sample_id=sample_ids),
          file.path(output_dir,'sample_ids_r.csv'), row.names=FALSE)
message('  saved: cache_info_r.csv  founders_r.csv  sample_ids_r.csv')

# ─────────────────────────────────────────────────────────────────────────────
# S2 — HAPLOTYPE PROBS SPOT-CHECK
# ─────────────────────────────────────────────────────────────────────────────
} # end s1

if (2 %in% run_sections) {
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
} # end s2

if (3 %in% run_sections) {
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
  kt <- data.frame(sample_id = sn, as.data.frame(k), check.names = FALSE)
  write.csv(kt, file.path(output_dir, paste0('kinship_mat_r_chr',chr,'.csv')),
          row.names=FALSE)
}
message('  saved: kinship_r.csv + per-chr kinship matrices')

# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM SCAN (ROP) — all genes on each chromosome
# ─────────────────────────────────────────────────────────────────────────────
} # end s3

if (4 %in% run_sections) {
message('\n[s4] lmm scan')
results_list <- list(); effects_list <- list()
for (chr in chromosomes) {
  k_chr    <- kinship_loco[[chr]]
  chr_loci <- pmap_flat[pmap_flat$chr==chr, ]
  gnames   <- if (use_gtf) head(intersect(
    gene_annot$gene_id[gene_annot$chr==chr], all_genes),
      if (is.infinite(n_genes)) .Machine$integer.max else n_genes) else all_genes
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
      # ae is [founders × loci] — need columns, not rows
      if (is.matrix(ae)) {
        ci <- min(pi, ncol(ae))                  # ← ncol, not nrow
        ef <- ae[, ci, drop=TRUE]                 # ← ae[, ci] not ae[ci, ]
      } else {
        ef <- ae
      }
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

# ── deep patch: find and remove crossprod BUILTIN check in all miqtl fns ──
miqtl_ns <- asNamespace('miqtl')
all_fns   <- ls(miqtl_ns, all.names=TRUE)
patched   <- c()

for (fn_name in all_fns) {
  fn_obj <- get(fn_name, envir=miqtl_ns)
  if (!is.function(fn_obj)) next
  fn_lines <- tryCatch(deparse(body(fn_obj)), error=function(e) NULL)
  if (is.null(fn_lines)) next
  if (!any(grepl('BUILTIN|is\\.primitive|is\\.builtin', fn_lines,
                  ignore.case=TRUE))) next
  fn_lines_new <- fn_lines[!grepl('BUILTIN|is\\.primitive|is\\.builtin',
                                   fn_lines, ignore.case=TRUE)]
  new_fn <- fn_obj
  tryCatch({
    body(new_fn) <- str2lang(paste(fn_lines_new, collapse='\n'))
    environment(new_fn) <- miqtl_ns
    utils::assignInNamespace(fn_name, new_fn, ns='miqtl')
    patched <- c(patched, fn_name)
  }, error=function(e) NULL)
}

if (length(patched)) {
  message('  crossprod BUILTIN patch applied to: ', paste(patched, collapse=', '))
} else {
  message('  crossprod BUILTIN check not found at R level — may be in C code')
}

# ─────────────────────────────────────────────────────────────────────────────
# S5 — PERMUTATION THRESHOLDS
# uses generate.sample.outcomes.matrix (miqtl native permutation)
# ─────────────────────────────────────────────────────────────────────────────
} # end s4

if (5 %in% run_sections) {
message('\n[s5] permutation thresholds')
thresh_chr  <- chromosomes[1]
thresh_candidates <- first_gene_on(thresh_chr, 10)
thresh_gene <- NULL
for (cand in thresh_candidates) {
  test_res <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_strain,
                pheno.id='subject.name', geno.id='subject.name',
                formula=make_form(cand), K=kinship_loco[[thresh_chr]],
                use.multi.impute=FALSE, model='additive',
                just.these.loci=get_cis_loci(cand, thresh_chr),
                return.allele.effects=FALSE),
    error=function(e) { message('  skipping ', cand, ' (', e$message, ')'); NULL })
  if (!is.null(test_res) && any(is.finite(test_res$LOD))) {
    thresh_gene <- cand
    message('  threshold gene: ', thresh_gene)
    break
  }
}
if (!length(thresh_gene)) {
  thresh_95 <- 6.0
  message('  SKIP: no viable threshold gene on chr', thresh_chr,
          ' — using standard fallback threshold=', thresh_95)
  write.csv(data.frame(percentile=0.95, lod_threshold=thresh_95,
                        gene_id=NA, chr=thresh_chr,
                        n_perms=0, perm_mean=NA, perm_sd=NA),
            file.path(output_dir,'thresholds_r.csv'), row.names=FALSE)
} else {
  cis_t   <- get_cis_loci(thresh_gene, thresh_chr)
  # jitter kinship to ensure positive definiteness for Cholesky in generate.sample.outcomes.matrix
  k_thresh <- kinship_loco[[thresh_chr]]
  complete <- complete.cases(pheno_strain[, c(thresh_gene, 'batch', paste0('pc', 1:n_pcs))])
  pheno_thresh <- pheno_strain[complete, ]
  # intersect with genomecache subjects so outcome vector length matches pheno exactly
  pheno_thresh <- pheno_thresh[rownames(pheno_thresh) %in% h$getSubjects(), , drop=FALSE]
  k_thresh <- k_thresh[rownames(pheno_thresh), rownames(pheno_thresh)]
  k_thresh <- k_thresh + diag(1e-4, nrow(k_thresh))
  message('  gene=', thresh_gene, '  chr', thresh_chr, '  cis_loci=', length(cis_t))

  cache_subs <- h$getSubjects()
  pheno_thresh <- pheno_thresh[rownames(pheno_thresh) %in% cache_subs, , drop=FALSE]
  k_thresh     <- k_thresh[rownames(pheno_thresh), rownames(pheno_thresh)]

  base_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_thresh,
                pheno.id='subject.name', geno.id='subject.name',
                formula=make_form(thresh_gene), K=k_thresh,
                use.multi.impute=FALSE, model='additive',
                just.these.loci=cis_t, return.allele.effects=FALSE),
    error=function(e) { message('  base scan failed: ', e$message); NULL })

  if (is.null(base_scan)) {
    thresh_95 <- 6.0
    message('  base scan failed — using fallback threshold=', thresh_95)
    write.csv(data.frame(percentile=0.95, lod_threshold=thresh_95,
                          gene_id=thresh_gene, chr=thresh_chr,
                          n_perms=0, perm_mean=NA, perm_sd=NA),
              file.path(output_dir,'thresholds_r.csv'), row.names=FALSE)
  } else {
    set.seed(42)
    perm_obj <- tryCatch(
      generate.sample.outcomes.matrix(
        scan.object=base_scan, method='permutation',
        num.samples=100, seed=42),
      error=function(e) { message('  generate.sample.outcomes.matrix failed: ', e$message); NULL })

    if (is.null(perm_obj)) {
      thresh_95 <- 6.0
      message('  perm object failed — using fallback threshold=', thresh_95)
      write.csv(data.frame(percentile=0.95, lod_threshold=thresh_95,
                            gene_id=thresh_gene, chr=thresh_chr,
                            n_perms=0, perm_mean=NA, perm_sd=NA),
                file.path(output_dir,'thresholds_r.csv'), row.names=FALSE)
    } else {
    # manual permutation loop — bypasses run.threshold.scans entirely.
    # generate.sample.outcomes.matrix(method='permutation') stores RINT float values,
    # but run.threshold.scans internally uses y.matrix values as integer indices to
    # reorder the phenotype column. float RINT values near zero truncate to 0/1, producing
    # wrong-length or zero-length vectors → replacement length error.
    # fix: directly assign each permuted column into a copy of pheno_thresh and call
    # scan.h2lmm ourselves — identical math, full control over dimensions.
    perm_subs <- rownames(perm_obj$y.matrix)
    pheno_perm_base <- pheno_thresh[perm_subs, , drop=FALSE]
    k_perm <- k_thresh[perm_subs, perm_subs]
    n_perms <- ncol(perm_obj$y.matrix)
    # ── diagnostic: verify perm_obj structure matches miQTL intent ──────────
    message('  diagnostic — perm_obj$y.matrix:')
    message('    dim: ', nrow(perm_obj$y.matrix), ' × ', ncol(perm_obj$y.matrix))
    message('    col1 range: ', round(min(perm_obj$y.matrix[,1], na.rm=TRUE), 3),
            ' to ', round(max(perm_obj$y.matrix[,1], na.rm=TRUE), 3))
    message('    values are floats (not int indices): ',
            !all(perm_obj$y.matrix[,1] == round(perm_obj$y.matrix[,1])))
    message('    rownames match pheno_perm_base: ',
            identical(sort(rownames(perm_obj$y.matrix)), sort(rownames(pheno_perm_base))))
    # ── end diagnostic ───────────────────────────────────────────────────────
    # use.fix.par defaults to TRUE in scan.h2lmm — this matches miQTL run.threshold.scans
    # behavior (fixes h2 via internal grid search per permutation). do not override.
    max_lods <- numeric(n_perms)
    message('  running ', n_perms, ' permutation scans manually...')
    for (p_i in seq_len(n_perms)) {
      p_data <- pheno_perm_base
      p_data[[thresh_gene]] <- perm_obj$y.matrix[, p_i]   # insert permuted outcome
      p_scan <- tryCatch(
        scan.h2lmm(genomecache=genomecache_path, data=p_data,
                    pheno.id='subject.name', geno.id='subject.name',
                    formula=make_form(thresh_gene), K=k_perm,
                    use.multi.impute=FALSE, model='additive',
                    just.these.loci=cis_t),
        error=function(e) NULL)
      max_lods[p_i] <- if (!is.null(p_scan) && length(p_scan$LOD) > 0)
        max(p_scan$LOD, na.rm=TRUE) else NA_real_
    }
    perm_lods <- max_lods[is.finite(max_lods)]
    thresh_95 <- tryCatch(
      get.gev.thresholds(
        list(max.statistics=list(LOD=perm_lods)),
        percentile=0.95, use.lod=TRUE),
      error=function(e) {
        message('  GEV fit failed (', e$message, ') — using empirical quantile')
        quantile(perm_lods, 0.95, na.rm=TRUE)
      })
    thresh_peak_pos <- pmap_flat$pos_mb[match(
      grab.locus.from.scan(base_scan), pmap_flat$locus)]
    write.csv(data.frame(percentile=0.95, lod_threshold=thresh_95,
                          gene_id=thresh_gene, chr=thresh_chr,
                          n_perms=length(perm_lods),
                          perm_mean=mean(perm_lods), perm_sd=sd(perm_lods)),
              file.path(output_dir,'thresholds_r.csv'), row.names=FALSE)
    write.csv(data.frame(gene_id=thresh_gene, chr=thresh_chr,
                          pos_mb=thresh_peak_pos, perm_lod=perm_lods),
              file.path(output_dir,'perm_lods_r.csv'), row.names=FALSE)
    message('  saved: thresholds_r.csv  threshold=', round(thresh_95, 3))
  }
}
}



# ─────────────────────────────────────────────────────────────────────────────
# S6 — CACHE COLLAPSE
# confirmed miqtl:::check.l2.norm: max_i(||h_i,l1-h_i,l2||/sqrt(J)) <= tol
# bugs patched: :::check.l2.norm (not exported) + test <- FALSE init
# ─────────────────────────────────────────────────────────────────────────────
} # end s5

if (6 %in% run_sections) {
message('\n[s6] cache collapse')
fn_lines <- deparse(body(miqtl:::collapse.genomecache))
fn_lines <- gsub('check\\.l2\\.norm\\(',      'miqtl:::check.l2.norm(',      fn_lines)
fn_lines <- gsub('check\\.max\\.category\\(', 'miqtl:::check.max.category(', fn_lines)
target   <- which(grepl('if \\(criterion == "l2.norm"\\)', fn_lines))
fn_lines <- append(fn_lines, '            test <- FALSE', after=target-1)
new_fn   <- miqtl:::collapse.genomecache                    # copy original
body(new_fn) <- str2lang(paste(fn_lines, collapse='\n'))    # patch body only
environment(new_fn) <- asNamespace('miqtl')                 # ← keep miqtl namespace
utils::assignInNamespace('collapse.genomecache', new_fn, ns='miqtl')  # ← safe assign
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
# S7 — MULTIPLE IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
} # end s6

if (7 %in% run_sections) {
message('\n[s7] multiple imputation')
mi_results <- list()
for (chr in chromosomes) {
  k_chr  <- kinship_loco[[chr]]
  gnames <- if (use_gtf) head(intersect(
    gene_annot$gene_id[gene_annot$chr==chr], all_genes),
      if (is.infinite(n_genes)) .Machine$integer.max else n_genes) else all_genes
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
} # end s7

if (8 %in% run_sections) {
message('\n[s8] snp scan')
pheno_snp <- pheno_strain
colnames(pheno_snp)[colnames(pheno_snp) == 'subject.name'] <- 'SUBJECT.NAME'
if (!dir.exists(allele_dir_path)) {
  message('  SKIP: allele_dir not found at ', allele_dir_path)
} else {
  # patch extract.imputed.design.matrix.from.doqtl.genotype:
  # bug — original does matrix(...)[,-1] and fat_raw[fat_raw[,1]!='NA', -1] in one step,
  # so after label-col drop, [,1] is f1 numeric and apply(fat[,-1],...) covers only f2..f8 (7 cols)
  # → ref_count length 7 → non-conformable with mapping.matrix (36×8)
  fixed_extract <- function(probs, allele.dir, snp, snp.chr, model, founders, mapping.matrix) {
    raw <- system(paste0('grep -w -A 3 \'', snp, '\' ',
                         file.path(allele.dir, paste0('chr', snp.chr, '.alleles'))),
                  intern=TRUE)
    raw <- trimws(raw)                                    # macOS grep -A prepends spaces to context lines
    fat_raw <- matrix(unlist(strsplit(raw[-1], '\t', fixed=TRUE)),
                      nrow=3, byrow=TRUE)                 # 3×9: col1=allele label, cols2-9=f1..f8
    fat_filt <- fat_raw[fat_raw[, 1] != 'NA', , drop=FALSE]  # filter NA row BEFORE dropping label
    labels   <- fat_filt[, 1]                             # save allele labels (len 2)
    fat      <- fat_filt[, -1, drop=FALSE]                # drop label col → 2×8 numeric
    f_alleles <- labels[apply(fat, 2, function(x) which.max(as.numeric(x)))]  # len 8
    ref_allele <- fat_raw[1, 1]
    ref_count  <- as.numeric(f_alleles == ref_allele)     # len 8 — conformable with mm (36×8)
    full_ref   <- as.vector(mapping.matrix %*% matrix(ref_count, ncol=1))  # 36×1
    full_geno  <- cbind(as.numeric(full_ref == 2),
                        as.numeric(full_ref == 1),
                        as.numeric(full_ref == 0))
    colnames(full_geno) <- c('ref.hom', 'het', 'alt.hom')
    gp <- probs %*% full_geno
    X  <- if ((2*sum(gp[,1]) + sum(gp[,2])) / (2*nrow(gp)) > 0.5)
            cbind(gp[,3], gp[,2]) else cbind(gp[,1], gp[,2])
    colnames(X) <- c('SNP_aa', 'SNP_Aa')
    if (model == 'additive') { X <- matrix(2*X[,1] + X[,2], ncol=1); colnames(X) <- 'SNP' }
    rownames(X) <- rownames(probs)
    X
  }
  utils::assignInNamespace('extract.imputed.design.matrix.from.doqtl.genotype',
                           fixed_extract, ns='miqtl')
  message('  patch: extract.imputed.design.matrix.from.doqtl.genotype applied')

  snp_results <- list()
  for (chr in chromosomes) {
    allele_file <- file.path(allele_dir_path, paste0('chr', chr, '.alleles'))
    if (!file.exists(allele_file)) {
      message('  SKIP chr', chr, ': no allele file'); next
    }
    lines   <- readLines(allele_file)
    snp_ids <- lines[seq(1, length(lines), by=4)]
    message('  chr', chr, ': ', length(snp_ids), ' SNPs in allele file')
    gnames <- if (use_gtf) head(intersect(
      gene_annot$gene_id[gene_annot$chr==chr], all_genes),
      if (is.infinite(n_genes)) .Machine$integer.max else n_genes) else all_genes

    for (gene in gnames) {
      cis <- get_cis_loci(gene, chr)
      if (is.null(cis)) next
      cis_snp <- intersect(cis, snp_ids)
      if (length(cis_snp) == 0) next
      res <- tryCatch(
        imputed.snp.scan.h2lmm(
          data=pheno_snp, formula=make_form(gene), K=k_chr,
          allele.dir=allele_dir_path, genomecache=genomecache_path,
          model='additive', chr=chr, just.these.loci=cis_snp),
        error=function(e) { message('  snp scan failed for ', gene, ': ', e$message); NULL })
      if (is.null(res)) next
      snp_results[[paste0(gene,'_',chr)]] <- data.frame(
        gene_id=gene, chr=chr, snp=res$loci,   # $loci holds SNP IDs; names(res$LOD) is NULL
        lod=res$LOD, p_value=res$p.value, stringsAsFactors=FALSE)
    }
  }
  if (length(snp_results)) {
    snp_df <- do.call(rbind, snp_results); rownames(snp_df) <- NULL
    write.csv(snp_df, file.path(output_dir,'snp_lod_r.csv'), row.names=FALSE)
    message('  saved: snp_lod_r.csv (', nrow(snp_df), ' rows)')
  } else {
    message('  no SNP results — cis loci may not overlap allele file SNPs')
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S9 — MEDIATION
# workflow: scan → peak locus → extract.mediation.qr(locus) → mediation.scan.qr
# ─────────────────────────────────────────────────────────────────────────────
} # end s8

if (9 %in% run_sections) {
message('\n[s9] mediation')
med_chr   <- chromosomes[1]
med_genes <- first_gene_on(med_chr, 2)
if (length(med_genes) < 2) {
  message('  SKIP: need >= 2 genes with cis loci on chr', med_chr)
} else {
  med_gene <- med_genes[1]; med_med <- med_genes[2]
  cis_med  <- get_cis_loci(med_gene, med_chr)
  med_cols <- c(med_gene, med_med, 'batch', paste0('pc', 1:n_pcs))
  complete_med <- complete.cases(pheno_strain[, med_cols])
  pheno_med <- pheno_strain[complete_med, ]

  if (var(pheno_med[[med_med]], na.rm=TRUE) < 1e-10) {
    message('  SKIP: mediator ', med_med, ' has near-zero variance')
  } else {
    k_med <- kinship_loco[[med_chr]]
    k_med <- k_med[rownames(pheno_med), rownames(pheno_med)]


    # WORKAROUND: copy mediator to a valid R column name (no leading digit)
    pheno_med$mediator_expr <- pheno_med[[med_med]]

    # scan WITHOUT mediator in model
    base_form <- make_form(med_gene)
    base_scan <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_med,
        pheno.id='subject.name', geno.id='subject.name',
        formula=base_form, K=k_med,
        use.multi.impute=FALSE, model='additive',
        just.these.loci=cis_med),
      error=function(e) { message('  base scan failed: ', e$message); NULL })

    # scan WITH mediator as covariate
    # use mediator_expr (safe name) instead of the raw gene name
    med_form <- as.formula(paste0('`', med_gene, '` ~ mediator_expr + batch + ',
        paste0('pc', 1:n_pcs, collapse=' + ')))
    med_scan <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_med,
        pheno.id='subject.name', geno.id='subject.name',
        formula=med_form, K=k_med,
        use.multi.impute=FALSE, model='additive',
        just.these.loci=cis_med),
      error=function(e) { message('  mediated scan failed: ', e$message); NULL })

    if (!is.null(base_scan) && !is.null(med_scan)) {
      lod_drop <- base_scan$LOD - med_scan$LOD
      write.csv(data.frame(
        gene_id=med_gene, chr=med_chr, locus=base_scan$loci,
        pos_mb=pmap_flat$pos_mb[match(base_scan$loci, pmap_flat$locus)],
        mediator=med_med,
        lod_base=base_scan$LOD, lod_mediated=med_scan$LOD, lod_drop=lod_drop,
        stringsAsFactors=FALSE),
        file.path(output_dir,'mediation_r.csv'), row.names=FALSE)
      message('  saved: mediation_r.csv')
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S10 — POSITIONAL CI (parametric bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
} # end s9

if (10 %in% run_sections) {
message('\n[s10] positional CI')
ci_chr  <- chromosomes[1]
ci_gene <- first_gene_on(ci_chr, 1)
if (!length(ci_gene)) {
  message('  SKIP: no gene with cis loci on chr', ci_chr)
} else {
  ci_gene <- ci_gene[1]
  ci_cols <- c(ci_gene, 'batch', paste0('pc', 1:n_pcs))
  complete_ci <- complete.cases(pheno_strain[, ci_cols])
  pheno_ci <- pheno_strain[complete_ci, ]
  k_chr <- kinship_loco[[ci_chr]]
  k_chr <- k_chr[rownames(pheno_ci), rownames(pheno_ci)]
  k_chr <- k_chr + diag(1e-4, nrow(k_chr))
  cis_ci <- get_cis_loci(ci_gene, ci_chr)

  ci_scan <- tryCatch(
    scan.h2lmm(genomecache=genomecache_path, data=pheno_ci,
      pheno.id='subject.name', geno.id='subject.name',
      formula=make_form(ci_gene), K=k_chr,
      use.multi.impute=FALSE, model='additive',
      just.these.loci=cis_ci),
    error=function(e) NULL)

  if (!is.null(ci_scan)) {
    peak_locus <- grab.locus.from.scan(ci_scan)

    peak_fit <- tryCatch(
      scan.h2lmm(genomecache=genomecache_path, data=pheno_ci,
        pheno.id='subject.name', geno.id='subject.name',
        formula=make_form(ci_gene), K=k_chr,
        use.multi.impute=FALSE, model='additive',
        just.these.loci=peak_locus),
      error=function(e) { message('  peak_fit failed: ', e$message); NULL })

    if (!is.null(peak_fit)) {
      set.seed(1)
      boot_obj <- tryCatch(
        generate.sample.outcomes.matrix(scan.object=peak_fit,
          model.type='alt', method='bootstrap',
          num.samples=100, seed=1, use.REML=TRUE, use.BLUP=FALSE),
        error=function(e) { message('  bootstrap failed: ', e$message); NULL })

      if (!is.null(boot_obj)) {
        boot_scans <- tryCatch(
          run.positional.scans(sim.object=boot_obj,
            genomecache=genomecache_path, data=pheno_ci,
            model='additive', use.multi.impute=FALSE, scan.seed=1),
          error=function(e) { message('  positional scans failed: ', e$message); NULL })

        if (!is.null(boot_scans)) {
          # $full.results = list(p.values, chr, pos)
          # $pos = list(Mb=numeric[n_boot], cM=numeric[n_boot]) — one position per bootstrap
          fr       <- boot_scans$full.results
          message('  full.results names: ', paste(names(fr), collapse=', '))
          boot_pos <- as.numeric(fr$pos$Mb)   # bootstrap peak positions in Mb
          message('  n_boot: ', length(boot_pos), '  non-NA: ', sum(!is.na(boot_pos)))
          if (is.null(peak_locus) || length(peak_locus) == 0 || is.na(peak_locus)) {
            message('  SKIP: peak_locus is empty'); next
          }
          write.csv(data.frame(
            gene_id=ci_gene, chr=ci_chr, locus=peak_locus,
            ci_lo_mb=quantile(boot_pos, 0.025, na.rm=TRUE),
            ci_hi_mb=quantile(boot_pos, 0.975, na.rm=TRUE),
            peak_lod=max(ci_scan$LOD, na.rm=TRUE),
            n_boot=sum(!is.na(boot_pos))),
            file.path(output_dir,'ci_r.csv'), row.names=FALSE)
          write.csv(data.frame(
            gene_id=ci_gene, chr=ci_chr, locus=peak_locus,
            boot_pos=boot_pos),
            file.path(output_dir,'boot_pos_r.csv'), row.names=FALSE)
          message('  saved: ci_r.csv  boot_pos_r.csv')
        }
      }
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# S11 — FIXED-EFFECT / INSTABILITY SCAN
# workflow: scan → generate.simple.sample.outcomes.matrix → instability.lm.scan
# ─────────────────────────────────────────────────────────────────────────────
} # end s10

if (11 %in% run_sections) {
message('\n[s11] instability scan')
fx_chr  <- chromosomes[1]
# select highest-LOD gene from S4 scan — guarantees stable instability signal.
# first_gene_on() picks by position, may return a null-QTL gene whose bootstrap
# scans never find a peak, causing instability.lm.scan to return length($loci)==0.
fx_gene <- NULL
lod_file_s4 <- file.path(output_dir, 'lod_r.csv')
if (file.exists(lod_file_s4)) {
  lod_s4    <- read.csv(lod_file_s4, check.names=FALSE)
  lod_s4    <- lod_s4[as.character(lod_s4$chr) == as.character(fx_chr), ]
  peak_s4   <- tapply(lod_s4$lod, lod_s4$gene_id, max, na.rm=TRUE)
  top_genes <- names(sort(peak_s4, decreasing=TRUE))
  for (cand in top_genes) {
    if (cand %in% all_genes && !is.null(get_cis_loci(cand, fx_chr))) {
      fx_gene <- cand; break
    }
  }
  if (!is.null(fx_gene))
    message('  instability gene (top LOD from S4): ', fx_gene,
            '  peak_lod=', round(peak_s4[fx_gene], 2))
}
if (is.null(fx_gene)) {
  fx_gene <- first_gene_on(fx_chr, 1)
  message('  instability gene (fallback first_gene_on): ',
          if (length(fx_gene)) fx_gene[1] else 'none')
}
if (!length(fx_gene)) {
  message('  SKIP: no gene with cis loci on chr', fx_chr)
} else {
  # loop over top candidates until instability.lm.scan returns non-empty $loci.
  # use.ROP=FALSE forces plain OLS bootstrap (function's documented intent);
  # use.ROP=TRUE (default) runs a full ROP LMM per bootstrap replicate — when
  # signal is weak the internal peak-finder finds no stable locus and returns
  # length($loci)==0 regardless of gene quality.
  lod_file_s4 <- file.path(output_dir, 'lod_r.csv')
  if (file.exists(lod_file_s4)) {
    lod_s4  <- read.csv(lod_file_s4, check.names=FALSE)
    lod_s4  <- lod_s4[as.character(lod_s4$chr) == as.character(fx_chr), ]
    peak_s4 <- tapply(lod_s4$lod, lod_s4$gene_id, max, na.rm=TRUE)
    fx_candidates <- names(sort(peak_s4, decreasing=TRUE))
    fx_candidates <- fx_candidates[fx_candidates %in% all_genes]
  } else {
    fx_candidates <- fx_gene   # already set above
  }
  fx_candidates <- head(fx_candidates[
    sapply(fx_candidates, function(g) !is.null(get_cis_loci(g, fx_chr)))], 10)

  # bypass instability.lm.scan — it calls run.locus.bs.fits which is a closure
  # defined inside instability.lm.scan itself and cannot be hoisted.
  # manual OLS bootstrap: subsample 63% of strains without replacement, run
  # scan.h2lmm(K=NULL) for OLS at each cis locus, average LOD across bootstraps.
  # mirrors Python instability_lm_scan (subsample_prop=0.63, n_samples=100).
  inst_res <- NULL; fx_used <- NULL
  set.seed(1)
  n_inst    <- nrow(pheno_strain)
  k_inst    <- max(floor(0.63 * n_inst), 2)
  n_boot    <- 100

  for (cand in fx_candidates) {
    cis_fx  <- get_cis_loci(cand, fx_chr)
    message('  trying gene: ', cand, '  cis_loci=', length(cis_fx))
    inst_form <- as.formula(paste0('`', cand, '` ~ batch + ',
                                   paste0('pc', 1:n_pcs, collapse=' + ')))
    lod_accum <- NULL

    for (b in seq_len(n_boot)) {
      idx_b    <- sample(n_inst, size=k_inst, replace=FALSE)
      pheno_b  <- pheno_strain[idx_b, , drop=FALSE]
      k_b      <- kinship_loco[[fx_chr]][rownames(pheno_b), rownames(pheno_b)]
      scan_b   <- tryCatch(
        scan.h2lmm(genomecache=genomecache_path, data=pheno_b,
                   pheno.id='subject.name', geno.id='subject.name',
                   formula=inst_form, K=NULL,           # K=NULL → OLS (no kinship)
                   use.multi.impute=FALSE, model='additive',
                   just.these.loci=cis_fx),
        error=function(e) NULL)
      if (is.null(scan_b) || length(scan_b$LOD) == 0) next
      if (is.null(lod_accum)) lod_accum <- numeric(length(scan_b$LOD))
      lod_accum <- lod_accum + scan_b$LOD
    }

    if (!is.null(lod_accum) && any(is.finite(lod_accum))) {
      inst_res <- list(loci=cis_fx, LOD=lod_accum / n_boot)
      fx_used  <- cand
      message('  instability gene accepted: ', cand, '  loci=', length(inst_res$loci))
      break
    }
    message('  no valid scans for ', cand, ' — trying next candidate')
  }
  if (!is.null(inst_res) && length(inst_res$loci) > 0) {
    cis_fx <- get_cis_loci(fx_used, fx_chr)
    write.csv(data.frame(
      gene_id=fx_used, chr=fx_chr, locus=inst_res$loci,
      pos_mb=pmap_flat$pos_mb[match(inst_res$loci, pmap_flat$locus)],
      lod=inst_res$LOD, stringsAsFactors=FALSE),
      file.path(output_dir,'fixef_r.csv'), row.names=FALSE)
    message('  saved: fixef_r.csv (', length(inst_res$loci), ' loci)')
  } else {
    message('  SKIP: all candidates returned empty instability result')
  }
}
} # end s11

cat('\n[done] all outputs in', output_dir, '\n')
