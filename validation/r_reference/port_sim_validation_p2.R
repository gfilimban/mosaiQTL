# port_sim_validation_p2.R — Phase 2 simulated validation: all 11 miQTL functions
#
# Self-contained: generates its own sim.CC.data phenotypes (independent of Phase 1).
# Exports phenotypes + haplotype prob CSVs so Python can build its own cache.
# Uses LOCO kinship throughout (not the identity/OLS shortcut used in Phase 1).
#
# Sections:
#   s1.  cache info          → cache_info_r.csv, founders_r.csv, sample_ids_r.csv
#   s2.  haplotype probs     → haplotype_vals_r.csv
#   s3.  kinship loco        → kinship_mat_r_chr{N}.csv, kinship_r.csv
#   s4.  lmm scan (rop)      → lod_r.csv, allele_effects_r.csv
#   s5.  cache collapse      → collapsed_cache_summary.csv, lod_collapsed_r.csv
#   s6.  permutation         → perm_max_lods_r.csv, thresholds_r.csv
#   s7.  multiple imputation → mi_lod_r.csv
#   s8.  snp scan (gated)    → snp_lod_r.csv
#   s9.  mediation           → mediation_r.csv
#  s10.  positional ci       → ci_r.csv, ci_summary_r.csv
#  s11.  instability         → instability_r.csv
#
# Usage:
#   Rscript validation/r_reference/port_sim_validation_p2.R
# ─────────────────────────────────────────────────────────────────────────────

lib_path <- Sys.getenv('R_LIB_PATH',
    unset = '/Users/ghassan/miniconda3/envs/bioinfo/lib/R/library')
if (dir.exists(lib_path)) .libPaths(c(lib_path, .libPaths()))
for (pkg in c('sparcc', 'miqtl')) library(pkg, character.only = TRUE)

# crossprod mask fix (same as port_validation.R)
if (!is.primitive(crossprod) || typeof(crossprod) != 'builtin') {
    assign('crossprod',  base::crossprod,  envir = globalenv())
    assign('tcrossprod', base::tcrossprod, envir = globalenv())
}

# ── paths ──────────────────────────────────────────────────────────────────
genomecache <- 'validation/sparcc_cache'
r_data      <- 'validation/phase2_sim/r_data'
r_plots     <- 'validation/phase2_sim/r_plots'
allele_dir  <- 'validation/phase2_sim/allele_dir'
dir.create(r_data,   recursive = TRUE, showWarnings = FALSE)
dir.create(r_plots,  recursive = TRUE, showWarnings = FALSE)

if (!dir.exists(genomecache)) {
    download.file('https://github.com/gkeele/sparcc_cache/archive/master.zip',
                  destfile = 'validation/sparcc_cache.zip')
    unzip('validation/sparcc_cache.zip', exdir = 'validation/')
    file.rename('validation/sparcc_cache-master', genomecache)
}

# ── simulated data ─────────────────────────────────────────────────────────
# Phase 2 generates its own independent phenotypes (seed=42 for reproducibility).
# Phase 1 used an unseeded call; Phase 2 uses a fixed seed so reruns are stable.
set.seed(42)
phenotypes <- sim.CC.data(
    genomecache    = genomecache,
    num.lines      = 40,
    num.sim        = 10,
    num.replicates = 2,
    qtl.effect.size = 0.8
)

cat('Simulated', length(phenotypes$locus), 'planted QTL loci across 10 phenotypes\n')

# Export phenotypes (all 10 sims) and true loci
sim_cols <- c('SUBJECT.NAME.1', paste0('sim.y.', 1:10))
write.csv(phenotypes$data[, sim_cols],
          file.path(r_data, 'phenotypes.csv'), row.names = FALSE)
write.csv(data.frame(sim   = seq_along(phenotypes$locus),
                     locus = phenotypes$locus),
          file.path(r_data, 'true_loci.csv'), row.names = FALSE)

# ── ROP scan on sim.y.1 — shared reference used by S3/S6/S9/S10/S11 ────────
# No explicit K: scan.h2lmm computes LOCO kinship from genomecache internally.
cat('\nRunning ROP scan (sim.y.1) for shared reference...\n')
rop_scan <- scan.h2lmm(
    genomecache          = genomecache,
    data                 = phenotypes$data,
    pheno.id             = 'SUBJECT.NAME.1',
    geno.id              = 'SUBJECT.NAME.1',
    formula              = sim.y.1 ~ 1,
    use.multi.impute     = FALSE,
    return.allele.effects = TRUE
)
qtl_idx   <- which.max(rop_scan$LOD)
qtl_locus <- names(rop_scan$LOD)[qtl_idx]
qtl_chr   <- rop_scan$chr[qtl_idx]
cat('  peak:', qtl_locus, '| chr', qtl_chr,
    '| LOD', round(as.numeric(rop_scan$LOD)[qtl_idx], 3), '\n')

# ─────────────────────────────────────────────────────────────────────────────
# Export haplotype probabilities (scan loci, all chromosomes)
# Same format as port_sim_validation.R — Python uses these to build its cache.
# ─────────────────────────────────────────────────────────────────────────────
cat('\nExporting haplotype probs per chromosome...\n')
probs_dir  <- file.path(r_data, 'probs')
dir.create(probs_dir, recursive = TRUE, showWarnings = FALSE)
reader     <- DiploprobReader$new(genomecache)
scan_loci  <- names(rop_scan$LOD)
scan_chr   <- setNames(rop_scan$chr,    scan_loci)
scan_mb    <- setNames(rop_scan$pos$Mb, scan_loci)

for (chr in unique(scan_chr)) {
    chr_loci <- scan_loci[scan_chr == chr]
    rows <- lapply(chr_loci, function(loc) {
        m        <- reader$getLocusMatrix(loc, model = 'additive')
        df       <- as.data.frame(m)
        df$strain <- rownames(m)
        df$locus  <- loc
        df$pos_mb <- scan_mb[loc]
        df
    })
    probs_df <- do.call(rbind, rows)
    write.csv(probs_df,
              file.path(probs_dir, paste0('probs_', chr, '.csv')),
              row.names = FALSE)
    cat('  probs exported: chr', chr, '|', length(chr_loci), 'loci x',
        nrow(rows[[1]]), 'strains\n')
}

# ─────────────────────────────────────────────────────────────────────────────
# S1 — cache info
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS1: cache info...\n')
loci_by_chr <- split(scan_loci, scan_chr)
cache_info <- do.call(rbind, lapply(names(loci_by_chr), function(chr) {
    loci  <- loci_by_chr[[chr]]
    mb    <- scan_mb[loci]
    data.frame(chr    = chr,
               n_loci = length(loci),
               min_mb = min(mb),
               max_mb = max(mb),
               stringsAsFactors = FALSE)
}))
write.csv(cache_info, file.path(r_data, 'cache_info_r.csv'), row.names = FALSE)
write.csv(data.frame(founder = colnames(reader$getLocusMatrix(scan_loci[1], model = 'additive'))),
          file.path(r_data, 'founders_r.csv'), row.names = FALSE)
write.csv(data.frame(sample_id = phenotypes$data$SUBJECT.NAME.1),
          file.path(r_data, 'sample_ids_r.csv'), row.names = FALSE)
cat('  exported cache_info_r.csv:', nrow(cache_info), 'chromosomes\n')

# ─────────────────────────────────────────────────────────────────────────────
# S2 — haplotype probability spot-check (5 random loci per chromosome)
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS2: haplotype probs spot-check...\n')
set.seed(7)
spot_loci <- unlist(lapply(names(loci_by_chr), function(chr) {
    loci <- loci_by_chr[[chr]]
    sample(loci, min(5, length(loci)))
}))
hap_rows <- lapply(spot_loci, function(loc) {
    m       <- reader$getLocusMatrix(loc, model = 'additive')
    df      <- as.data.frame(m)
    df$strain    <- rownames(m)
    df$locus     <- loc
    df$chr       <- scan_chr[loc]
    df$pos_mb    <- scan_mb[loc]
    df
})
hap_vals <- do.call(rbind, hap_rows)
write.csv(hap_vals, file.path(r_data, 'haplotype_vals_r.csv'), row.names = FALSE)
cat('  exported haplotype_vals_r.csv:', nrow(hap_vals), 'rows\n')

# ─────────────────────────────────────────────────────────────────────────────
# S3 — LOCO kinship matrices
# K_loco_c = (L_all * K_all - L_c * K_c) / (L_all - L_c)
# calc.kinship.from.genomecache.with.DOQTL has no locus/chr filter; pass it a
# temporary single-chromosome cache (symlink) to get K_c.
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS3: LOCO kinship matrices...\n')
K_all <- calc.kinship.from.genomecache.with.DOQTL(genomecache)
# locus counts from cache directory (consistent with what K_all was computed over)
cache_norm <- normalizePath(genomecache)
cache_add  <- file.path(cache_norm, 'additive')
L_by_chr   <- sapply(names(loci_by_chr), function(chr)
    length(list.files(file.path(cache_add, paste0('chr', chr)))))
L_all <- sum(L_by_chr)
cat('  K_all computed; total loci =', L_all, '\n')

cache_norm <- normalizePath(genomecache)
kinship_summary <- list()
for (chr_c in names(loci_by_chr)) {
    # single-chromosome cache: symlink all three model dirs required by
    # happy.load.genome (genotype/, additive/, full/) for chr_c only
    tmp_cache <- file.path(tempdir(), paste0('kd_chr', chr_c))
    for (model_dir in c('additive', 'full', 'genotype')) {
        src <- file.path(cache_norm, model_dir, paste0('chr', chr_c))
        dst_parent <- file.path(tmp_cache, model_dir)
        if (dir.exists(src)) {
            dir.create(dst_parent, recursive = TRUE, showWarnings = FALSE)
            file.symlink(from = src,
                         to   = file.path(dst_parent, paste0('chr', chr_c)))
        }
    }
    K_c  <- calc.kinship.from.genomecache.with.DOQTL(tmp_cache)
    unlink(tmp_cache, recursive = TRUE)

    L_c    <- L_by_chr[[chr_c]]
    K_loco <- (L_all * K_all - L_c * K_c) / (L_all - L_c)
    K_loco <- (K_loco + t(K_loco)) / 2

    km_df        <- as.data.frame(K_loco)
    km_df$strain <- rownames(K_loco)
    km_df        <- km_df[, c('strain', setdiff(names(km_df), 'strain'))]
    write.csv(km_df,
              file.path(r_data, paste0('kinship_mat_r_chr', chr_c, '.csv')),
              row.names = FALSE)

    kinship_summary[[chr_c]] <- data.frame(
        chr          = chr_c,
        mean_diag    = mean(diag(K_loco)),
        mean_offdiag = mean(K_loco[lower.tri(K_loco)]),
        stringsAsFactors = FALSE)
    cat('  chr', chr_c, '  diag mean =',
        round(mean(diag(K_loco)), 4), '\n')
}
write.csv(do.call(rbind, kinship_summary),
          file.path(r_data, 'kinship_r.csv'), row.names = FALSE)

# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM scan (ROP) for all 10 phenotypes
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS4: LMM scan (10 phenotypes)...\n')
lod_rows    <- list()
effect_rows <- list()

for (sim_i in 1:10) {
    pheno_col <- paste0('sim.y.', sim_i)
    cat('  sim.y.', sim_i, '... ', sep = '')
    sc <- scan.h2lmm(
        genomecache          = genomecache,
        data                 = phenotypes$data,
        pheno.id             = 'SUBJECT.NAME.1',
        geno.id              = 'SUBJECT.NAME.1',
        formula              = as.formula(paste(pheno_col, '~ 1')),
        use.multi.impute     = FALSE,
        return.allele.effects = TRUE
    )
    peak_i <- which.max(sc$LOD)
    cat('peak LOD =', round(as.numeric(sc$LOD)[peak_i], 2),
        'at', names(sc$LOD)[peak_i], '\n')

    lod_rows[[sim_i]] <- data.frame(
        sim_idx  = sim_i,
        locus    = names(sc$LOD),
        chr      = sc$chr,
        pos_mb   = sc$pos$Mb,
        lod      = as.numeric(sc$LOD),
        stringsAsFactors = FALSE)

    if (!is.null(sc$allele.effects)) {
        ae <- sc$allele.effects
        peak_loc <- names(sc$LOD)[peak_i]
        if (peak_loc %in% colnames(ae)) {
            eff <- ae[, peak_loc]
            effect_rows[[sim_i]] <- data.frame(
                sim_idx  = sim_i,
                locus    = peak_loc,
                founder  = names(eff),
                beta     = as.numeric(eff),
                stringsAsFactors = FALSE)
        }
    }
}
write.csv(do.call(rbind, lod_rows),
          file.path(r_data, 'lod_r.csv'), row.names = FALSE)
write.csv(do.call(rbind, effect_rows),
          file.path(r_data, 'allele_effects_r.csv'), row.names = FALSE)
cat('  exported lod_r.csv:', nrow(do.call(rbind, lod_rows)), 'rows\n')

# ─────────────────────────────────────────────────────────────────────────────
# S5 — cache collapse
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS5: cache collapse...\n')
coll_summary <- tryCatch({
    genomecache_coll <- file.path(r_data, 'sparcc_cache_collapsed')
    collapse.genomecache(
        original.cache  = genomecache,
        new.cache       = genomecache_coll,
        criterion       = 'l2.norm',
        proportion.tol  = 0.02
    )
    w <- warnings(); if (length(w) > 0) { cat('  collapse warnings:\n'); print(w) }
    # count loci per chr by reading markers.RData directly — same source
    # collapse.genomecache uses internally (these.loci / reduced.loci)
    count_markers <- function(cache_path, chr) {
        f <- file.path(cache_path, 'additive', paste0('chr', chr), 'markers.RData')
        e <- new.env(parent = emptyenv())
        tryCatch({ load(f, envir = e); length(e$markers) }, error = function(err) NA_integer_)
    }
    all_chrs <- names(loci_by_chr)
    loci_full_by_chr <- sapply(all_chrs, function(chr) count_markers(genomecache,      chr))
    loci_coll_by_chr <- sapply(all_chrs, function(chr) count_markers(genomecache_coll, chr))
    coll_df <- data.frame(
        chr           = names(loci_by_chr),
        n_loci_full   = as.integer(loci_full_by_chr),
        n_loci_coll   = as.integer(loci_coll_by_chr),
        pct_removed   = round(100 * (1 - loci_coll_by_chr / loci_full_by_chr), 2),
        stringsAsFactors = FALSE)

    # rescan sim.y.1 with collapsed cache to test LOD preservation
    sc_coll <- scan.h2lmm(
        genomecache      = genomecache_coll,
        data             = phenotypes$data,
        pheno.id         = 'SUBJECT.NAME.1',
        geno.id          = 'SUBJECT.NAME.1',
        formula          = sim.y.1 ~ 1,
        use.multi.impute = FALSE
    )
    lod_coll_df <- data.frame(
        locus  = names(sc_coll$LOD),
        chr    = sc_coll$chr,
        pos_mb = sc_coll$pos$Mb,
        lod    = as.numeric(sc_coll$LOD),
        stringsAsFactors = FALSE)
    write.csv(lod_coll_df, file.path(r_data, 'lod_collapsed_r.csv'), row.names = FALSE)
    cat('  cache collapsed; rescan done\n')
    coll_df
}, error = function(e) {
    cat('  WARNING: collapse.genomecache failed:', conditionMessage(e), '\n')
    cat('  S5 will be skipped in Python validation\n')
    data.frame(chr = names(loci_by_chr), n_loci_full = NA, n_loci_coll = NA,
               pct_removed = NA, stringsAsFactors = FALSE)
})
write.csv(coll_summary, file.path(r_data, 'collapsed_cache_summary.csv'),
          row.names = FALSE)

# ─────────────────────────────────────────────────────────────────────────────
# S6 — permutation thresholds (n=250, peak chromosome only)
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS6: permutation thresholds (n=250, chr', qtl_chr, ')...\n')
w <- rep(1, length(rop_scan$fit0$y))
names(w) <- names(rop_scan$fit0$y)
rop_scan$fit0$weights <- w

perm_out <- generate.sample.outcomes.matrix(
    scan.object = rop_scan,
    method      = 'permutation',
    num.samples = 250)

perm_scans <- run.threshold.scans(
    sim.threshold.object = perm_out,
    keep.full.scans      = FALSE,
    genomecache          = genomecache,
    data                 = phenotypes$data,
    use.multi.impute     = FALSE,
    scan.seed            = 1)

perm_threshold <- get.gev.thresholds(
    threshold.scans = perm_scans,
    percentile      = 0.95)

perm_max_lods <- if (is.numeric(perm_scans)) {
    as.numeric(perm_scans)
} else {
    sapply(perm_scans, function(s) max(as.numeric(s$LOD), na.rm = TRUE))
}
write.csv(data.frame(perm_index = seq_along(perm_max_lods) - 1L,
                     max_lod    = perm_max_lods),
          file.path(r_data, 'perm_max_lods_r.csv'), row.names = FALSE)
write.csv(data.frame(threshold           = unname(perm_threshold),
                     empirical_quantile  = quantile(perm_max_lods, 0.95,
                                                    na.rm = TRUE)),
          file.path(r_data, 'thresholds_r.csv'), row.names = FALSE)
cat('  GEV threshold:', round(unname(perm_threshold), 3),
    '  empirical:', round(quantile(perm_max_lods, 0.95, na.rm = TRUE), 3), '\n')

# ─────────────────────────────────────────────────────────────────────────────
# S7 — multiple imputation (num.imp=11)
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS7: MI scan (num.imp=11)...\n')
mi_scan <- scan.h2lmm(
    genomecache      = genomecache,
    data             = phenotypes$data,
    pheno.id         = 'SUBJECT.NAME.1',
    geno.id          = 'SUBJECT.NAME.1',
    formula          = sim.y.1 ~ 1,
    use.multi.impute = TRUE,
    num.imp          = 11)
write.csv(data.frame(locus  = names(mi_scan$LOD),
                     chr    = mi_scan$chr,
                     pos_mb = mi_scan$pos$Mb,
                     lod    = as.numeric(mi_scan$LOD)),
          file.path(r_data, 'mi_lod_r.csv'), row.names = FALSE)
cat('  MI peak LOD:', round(max(as.numeric(mi_scan$LOD)), 3), '\n')

# ─────────────────────────────────────────────────────────────────────────────
# S8 — SNP scan (gated: requires allele_dir)
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS8: SNP scan (gated)...\n')
if (dir.exists(allele_dir)) {
    tryCatch({
        snp_scan <- imputed.snp.scan.h2lmm(
            genomecache = genomecache,
            data        = phenotypes$data,
            pheno.id    = 'SUBJECT.NAME.1',
            geno.id     = 'SUBJECT.NAME.1',
            formula     = sim.y.1 ~ 1,
            allele.dir  = allele_dir,
            chr         = qtl_chr)
        write.csv(data.frame(
            snp    = snp_scan$SNP.ID,
            chr    = snp_scan$chr,
            pos_mb = snp_scan$pos$Mb,
            lod    = as.numeric(snp_scan$LOD)),
            file.path(r_data, 'snp_lod_r.csv'), row.names = FALSE)
        cat('  SNP scan done; peak LOD:',
            round(max(as.numeric(snp_scan$LOD)), 3), '\n')
    }, error = function(e) {
        cat('  WARNING: SNP scan failed:', conditionMessage(e), '\n')
    })
} else {
    cat('  SKIPPED: allele_dir not found at', allele_dir, '\n')
    cat('  To enable: build allele_dir with miQTL build.allele.dir()\n')
}

# ─────────────────────────────────────────────────────────────────────────────
# S9 — mediation (outcome=sim.y.1, mediator=sim.y.2, peak chromosome)
# Two-scan approach: base scan vs. scan with sim.y.2 as covariate.
# lod_drop = lod_base − lod_mediated (matches Python mediation_scan_qr logic).
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS9: mediation (outcome=sim.y.1, mediator=sim.y.2, chr', qtl_chr, ')...\n')
# base scan restricted to peak chromosome (rop_scan already covers whole genome)
chr_mask    <- rop_scan$chr == qtl_chr
lod_base_c  <- as.numeric(rop_scan$LOD)[chr_mask]
loci_c      <- names(rop_scan$LOD)[chr_mask]
pos_mb_c    <- rop_scan$pos$Mb[chr_mask]

# mediated scan: include sim.y.2 as fixed covariate
med_scan <- scan.h2lmm(
    genomecache      = genomecache,
    data             = phenotypes$data,
    pheno.id         = 'SUBJECT.NAME.1',
    geno.id          = 'SUBJECT.NAME.1',
    formula          = sim.y.1 ~ sim.y.2,
    use.multi.impute = FALSE,
    chr              = qtl_chr)
lod_med_c <- as.numeric(med_scan$LOD)

write.csv(data.frame(
    locus        = loci_c,
    chr          = qtl_chr,
    pos_mb       = pos_mb_c,
    lod_base     = lod_base_c,
    lod_mediated = lod_med_c,
    lod_drop     = pmax(lod_base_c - lod_med_c, 0)),
    file.path(r_data, 'mediation_r.csv'), row.names = FALSE)
cat('  median lod_drop:',
    round(median(pmax(lod_base_c - lod_med_c, 0), na.rm = TRUE), 3), '\n')

# ─────────────────────────────────────────────────────────────────────────────
# S10 — bootstrap positional CI (100 bootstraps at peak locus)
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS10: bootstrap CI (n=100)...\n')
qtl_fit <- scan.h2lmm(
    genomecache      = genomecache,
    data             = phenotypes$data,
    pheno.id         = 'SUBJECT.NAME.1',
    geno.id          = 'SUBJECT.NAME.1',
    formula          = sim.y.1 ~ 1,
    use.multi.impute = FALSE,
    just.these.loci  = qtl_locus)

w1 <- rep(1, length(qtl_fit$fit1$y))
names(w1) <- names(qtl_fit$fit1$y)
qtl_fit$fit1$weights <- w1

boot_out <- generate.sample.outcomes.matrix(
    scan.object  = qtl_fit,
    model.type   = 'alt',
    method       = 'bootstrap',
    num.sample   = 100,
    seed         = 1,
    use.REML     = TRUE,
    use.BLUP     = TRUE)

boot_scans <- run.positional.scans(
    sim.object       = boot_out,
    keep.full.scans  = TRUE,
    genomecache      = genomecache,
    data             = phenotypes$data,
    use.multi.impute = FALSE,
    scan.seed        = 1)

boot_peaks <- sapply(boot_scans, function(bs) {
    if (is.null(bs) || !is.list(bs)) return(NA_real_)
    tryCatch({
        loci_chr <- names(bs$LOD)[bs$chr == qtl_chr]
        if (length(loci_chr) == 0) return(NA_real_)
        peak_loc <- loci_chr[which.max(as.numeric(bs$LOD[loci_chr]))]
        bs$pos$Mb[names(bs$LOD) == peak_loc]
    }, error = function(e) NA_real_)
})
boot_peaks <- as.numeric(boot_peaks)

write.csv(data.frame(boot_index = seq_along(boot_peaks) - 1L,
                     peak_mb    = boot_peaks),
          file.path(r_data, 'ci_r.csv'), row.names = FALSE)

planted_mb <- rop_scan$pos$Mb[names(rop_scan$LOD) == qtl_locus]
ci_lo <- quantile(boot_peaks, 0.025, na.rm = TRUE)
ci_hi <- quantile(boot_peaks, 0.975, na.rm = TRUE)
write.csv(data.frame(ci_lo            = ci_lo,
                     ci_hi            = ci_hi,
                     peak_mb          = planted_mb,
                     planted_locus    = qtl_locus,
                     planted_locus_mb = planted_mb),
          file.path(r_data, 'ci_summary_r.csv'), row.names = FALSE)
cat('  95% CI: [', round(ci_lo, 2), ',', round(ci_hi, 2), '] Mb\n')
cat('  peak locus:', qtl_locus, '@', round(planted_mb, 2), 'Mb\n')

# ─────────────────────────────────────────────────────────────────────────────
# S11 — instability scan
# ─────────────────────────────────────────────────────────────────────────────
cat('\nS11: instability scan (n=100, prop=0.63)...\n')
instab_scan <- instability.lm.scan(
    genomecache    = genomecache,
    data           = phenotypes$data,
    pheno.id       = 'SUBJECT.NAME.1',
    geno.id        = 'SUBJECT.NAME.1',
    formula        = sim.y.1 ~ 1,
    n.samples      = 100,
    subsample.prop = 0.63,
    seed           = 1,
    chr            = qtl_chr)
write.csv(data.frame(
    locus            = names(instab_scan$LOD),
    chr              = instab_scan$chr,
    pos_mb           = instab_scan$pos$Mb,
    lod_instability  = as.numeric(instab_scan$LOD)),
    file.path(r_data, 'instability_r.csv'), row.names = FALSE)
cat('  peak instability LOD:',
    round(max(as.numeric(instab_scan$LOD), na.rm=TRUE), 3), '\n')

# ── reference plots ────────────────────────────────────────────────────────
png(file.path(r_plots, 'rop_scan.png'), width=3000, height=1500, res=300)
genome.plotter.whole(scan.list = list(ROP = rop_scan))
dev.off()

png(file.path(r_plots, 'rop_with_threshold.png'), width=3000, height=1500, res=300)
genome.plotter.whole(scan.list = list(ROP = rop_scan),
                     hard.thresholds = perm_threshold)
dev.off()

png(file.path(r_plots, 'allele_effects_region.png'), width=2700, height=1500, res=300)
allele.plotter.region(scan.object = rop_scan, chr = qtl_chr)
dev.off()

cat('\n── Phase 2 sim R reference complete ────────────────────────────────────\n')
cat('  outputs in:', r_data, '\n')
cat('  run next: python validation/validate_sim_p2.py\n')
