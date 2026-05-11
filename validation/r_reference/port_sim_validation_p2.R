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
#  s11.  instability         → instability_r.csv  (locus, chr, pos_mb, scan_1..scan_N p-values)
#
# Usage:
#   Rscript validation/r_reference/port_sim_validation_p2.R                 # all sections
#   Rscript validation/r_reference/port_sim_validation_p2.R --sections 6,9,11
#   Rscript validation/r_reference/port_sim_validation_p2.R --sections 11 --force
#
# Setup (sim.CC.data + full rop_scan) is cached to r_data/setup_cache.RData
# on first run and reloaded on subsequent runs unless --force is passed.
# ─────────────────────────────────────────────────────────────────────────────

# ── argument parsing ──────────────────────────────────────────────────────────
args         <- commandArgs(trailingOnly = TRUE)
sections_pos <- which(args == '--sections')
if (length(sections_pos) > 0 && sections_pos[1] < length(args)) {
    run_sections <- as.integer(strsplit(args[sections_pos[1] + 1], ',')[[1]])
} else {
    run_sections <- 1:11
}
force_rerun  <- '--force' %in% args
run_section  <- function(n) n %in% run_sections
cat('Sections to run:', paste(run_sections, collapse = ', '),
    if (force_rerun) '[--force]' else '', '\n')

lib_path <- Sys.getenv('R_LIB_PATH',
    unset = '/Users/ghassan/miniconda3/envs/bioinfo/lib/R/library')
if (dir.exists(lib_path)) .libPaths(c(lib_path, .libPaths()))
for (pkg in c('sparcc', 'miqtl')) suppressWarnings(library(pkg, character.only = TRUE))

t_start    <- proc.time()
elapsed_s  <- function(t0, t1) as.numeric(t1['elapsed'] - t0['elapsed'])

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

# ── setup: sim data + rop_scan (cached to avoid full rescan on reruns) ────────
setup_cache <- file.path(r_data, 'setup_cache.RData')

if (!force_rerun && file.exists(setup_cache)) {
    cat('\nLoading cached setup from', setup_cache, '...\n')
    load(setup_cache)   # restores: phenotypes, rop_scan, reader, scan_loci,
                        #           scan_chr, scan_mb, loci_by_chr,
                        #           qtl_idx, qtl_locus, qtl_chr
    cat('  peak:', qtl_locus, '| chr', qtl_chr,
        '| LOD', round(as.numeric(rop_scan$LOD)[qtl_idx], 3), '\n')
} else {
    # Phase 2 generates its own independent phenotypes (seed=42 for reproducibility).
    set.seed(42)
    phenotypes <- sim.CC.data(
        genomecache     = genomecache,
        num.lines       = 40,
        num.sim         = 10,
        num.replicates  = 2,
        qtl.effect.size = 0.8
    )
    cat('\nSimulated', length(phenotypes$locus),
        'planted QTL loci across 10 phenotypes\n')

    # Export phenotypes and true loci (only written on fresh setup)
    sim_cols <- c('SUBJECT.NAME.1', paste0('sim.y.', 1:10))
    write.csv(phenotypes$data[, sim_cols],
              file.path(r_data, 'phenotypes.csv'), row.names = FALSE)
    write.csv(data.frame(sim   = seq_along(phenotypes$locus),
                         locus = phenotypes$locus),
              file.path(r_data, 'true_loci.csv'), row.names = FALSE)

    # ROP scan on sim.y.1 — shared reference used by S3/S6/S9/S10/S11.
    # scan.h2lmm computes LOCO kinship from genomecache internally.
    cat('\nRunning ROP scan (sim.y.1) for shared reference...\n')
    rop_scan <- scan.h2lmm(
        genomecache           = genomecache,
        data                  = phenotypes$data,
        pheno.id              = 'SUBJECT.NAME.1',
        geno.id               = 'SUBJECT.NAME.1',
        formula               = sim.y.1 ~ 1,
        use.multi.impute      = FALSE,
        return.allele.effects = TRUE
    )
    qtl_idx   <- which.max(rop_scan$LOD)
    qtl_locus <- names(rop_scan$LOD)[qtl_idx]
    qtl_chr   <- rop_scan$chr[qtl_idx]
    cat('  peak:', qtl_locus, '| chr', qtl_chr,
        '| LOD', round(as.numeric(rop_scan$LOD)[qtl_idx], 3), '\n')

    reader    <- DiploprobReader$new(genomecache)
    scan_loci <- names(rop_scan$LOD)
    scan_chr  <- setNames(rop_scan$chr,    scan_loci)
    scan_mb   <- setNames(rop_scan$pos$Mb, scan_loci)
    loci_by_chr <- split(scan_loci, scan_chr)

    save(phenotypes, rop_scan, reader, scan_loci, scan_chr, scan_mb,
         loci_by_chr, qtl_idx, qtl_locus, qtl_chr,
         file = setup_cache)
    cat('  setup cached to', setup_cache, '\n')
}
t_after_setup <- proc.time()

# ─────────────────────────────────────────────────────────────────────────────
# Export haplotype probabilities (scan loci, all chromosomes)
# Skipped if probs/ already populated and --force not set.
# ─────────────────────────────────────────────────────────────────────────────
probs_dir <- file.path(r_data, 'probs')
dir.create(probs_dir, recursive = TRUE, showWarnings = FALSE)
if (!force_rerun && length(list.files(probs_dir, '*.csv')) > 0) {
    cat('\nProbs already exported; skipping (use --force to regenerate)\n')
} else {
    cat('\nExporting haplotype probs per chromosome...\n')
    for (chr in unique(scan_chr)) {
        chr_loci <- scan_loci[scan_chr == chr]
        rows <- lapply(chr_loci, function(loc) {
            m         <- reader$getLocusMatrix(loc, model = 'additive')
            df        <- as.data.frame(m)
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
}
t_after_probs <- proc.time()

# ─────────────────────────────────────────────────────────────────────────────
# S1 — cache info
# ─────────────────────────────────────────────────────────────────────────────
t_s1 <- proc.time()
if (run_section(1)) {
cat('\nS1: cache info...\n')
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
}  # end S1

t_s2 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S2 — haplotype probability spot-check (5 random loci per chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(2)) {
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
}  # end S2

t_s3 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S3 — LOCO kinship matrices
# K_loco_c = (L_all * K_all - L_c * K_c) / (L_all - L_c)
# calc.kinship.from.genomecache.with.DOQTL has no locus/chr filter; pass it a
# temporary single-chromosome cache (symlink) to get K_c.
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(3)) {
cat('\nS3: LOCO kinship matrices...\n')
K_all <- suppressWarnings(calc.kinship.from.genomecache.with.DOQTL(genomecache))
# locus counts from cache directory (consistent with what K_all was computed over)
cache_norm <- normalizePath(genomecache)
cache_add  <- file.path(cache_norm, 'additive')
L_by_chr   <- sapply(names(loci_by_chr), function(chr)
    length(list.files(file.path(cache_add, paste0('chr', chr)))))
L_all <- sum(L_by_chr)
cat('  K_all computed; total loci =', L_all, '\n')

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
    K_c  <- suppressWarnings(calc.kinship.from.genomecache.with.DOQTL(tmp_cache))
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
}  # end S3

t_s4 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM scan (ROP) for all 10 phenotypes
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(4)) {
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
}  # end S4

t_s5 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S5 — cache collapse
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(5)) {
cat('\nS5: cache collapse...\n')
genomecache_coll <- file.path(r_data, 'sparcc_cache_collapsed')
collapse.genomecache(
    original.cache  = genomecache,
    new.cache       = genomecache_coll,
    criterion       = 'l2.norm',
    proportion.tol  = 0.02
)

# count loci per chr via markers.RData (same source collapse.genomecache uses)
count_markers <- function(cache_path, chr) {
    f <- file.path(cache_path, 'additive', paste0('chr', chr), 'markers.RData')
    e <- new.env(parent = emptyenv())
    load(f, envir = e)
    length(e$markers)
}
all_chrs         <- names(loci_by_chr)
loci_full_by_chr <- sapply(all_chrs, function(chr) count_markers(genomecache,      chr))
loci_coll_by_chr <- sapply(all_chrs, function(chr) count_markers(genomecache_coll, chr))
coll_summary <- data.frame(
    chr         = names(loci_by_chr),
    n_loci_full = as.integer(loci_full_by_chr),
    n_loci_coll = as.integer(loci_coll_by_chr),
    pct_removed = round(100 * (1 - loci_coll_by_chr / loci_full_by_chr), 2),
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
write.csv(lod_coll_df,   file.path(r_data, 'lod_collapsed_r.csv'),        row.names = FALSE)
write.csv(coll_summary,  file.path(r_data, 'collapsed_cache_summary.csv'), row.names = FALSE)
cat('  cache collapsed; rescan done\n')
}  # end S5

t_s6 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S6 — permutation thresholds (n=250, peak chromosome only)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(6)) {
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
    chr                  = qtl_chr,
    scan.seed            = 1)

perm_threshold <- get.gev.thresholds(
    threshold.scans = perm_scans,
    percentile      = 0.95)

# export raw max-LODs (same pattern as Phase 1 port_sim_validation.R)
perm_max_lods_r <- if (is.numeric(perm_scans)) {
    as.numeric(perm_scans)
} else {
    sapply(perm_scans, function(s) max(as.numeric(s$LOD), na.rm = TRUE))
}
perm_max_lods_r <- perm_max_lods_r[is.finite(perm_max_lods_r)]
write.csv(data.frame(perm_index = seq_along(perm_max_lods_r),
                     max_lod    = perm_max_lods_r),
          file.path(r_data, 'perm_max_lods_r.csv'), row.names = FALSE)
write.csv(data.frame(threshold = unname(perm_threshold)),
          file.path(r_data, 'thresholds_r.csv'), row.names = FALSE)
cat('  GEV threshold:', round(unname(perm_threshold), 3), '\n')
png(file.path(r_plots, 'rop_with_threshold.png'), width=3000, height=1500, res=300)
genome.plotter.whole(scan.list = list(ROP = rop_scan),
                     hard.thresholds = unname(perm_threshold))
dev.off()
cat('  plot: rop_with_threshold.png\n')
}  # end S6

t_s7 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S7 — multiple imputation (num.imp=11)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(7)) {
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
}  # end S7

t_s8 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S8 — SNP scan (gated: requires allele_dir)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(8)) {
cat('\nS8: SNP scan (gated)...\n')
if (dir.exists(allele_dir)) {
    K_s8 <- calc.kinship.from.genomecache.with.DOQTL(genomecache)
    snp_scan <- imputed.snp.scan.h2lmm(
        data        = phenotypes$data,
        formula     = sim.y.1 ~ 1,
        K           = K_s8,
        allele.dir  = allele_dir,
        genomecache = genomecache,
        chr         = qtl_chr)
    write.csv(data.frame(
        snp    = snp_scan$SNP.ID,
        chr    = snp_scan$chr,
        pos_mb = snp_scan$pos$Mb,
        lod    = as.numeric(snp_scan$LOD)),
        file.path(r_data, 'snp_lod_r.csv'), row.names = FALSE)
    cat('  SNP scan done; peak LOD:',
        round(max(as.numeric(snp_scan$LOD)), 3), '\n')
} else {
    cat('  SKIPPED: allele_dir not found at', allele_dir, '\n')
    cat('  To enable: build allele_dir with miQTL build.allele.dir()\n')
}
}  # end S8

t_s9 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S9 — mediation (outcome=sim.y.1, mediator=sim.y.2, peak chromosome)
# Two-scan approach: base scan vs. scan with sim.y.2 as covariate.
# lod_drop = lod_base − lod_mediated (matches Python mediation_scan_qr logic).
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(9)) {
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

# lod_drop is raw (not clipped): negative values mean the "mediator" adds noise.
# sim.y.1 and sim.y.2 are independent so null (lod_drop ≈ 0) is expected.
# Python comparison validates the null case: both sides should produce ~0 with same noise pattern.
lod_drop_raw <- lod_base_c - lod_med_c
write.csv(data.frame(
    locus        = loci_c,
    chr          = qtl_chr,
    pos_mb       = pos_mb_c,
    lod_base     = lod_base_c,
    lod_mediated = lod_med_c,
    lod_drop     = lod_drop_raw),
    file.path(r_data, 'mediation_r.csv'), row.names = FALSE)
cat('  median lod_drop:', round(median(lod_drop_raw, na.rm = TRUE), 3),
    '  (null expected: ~0, independent phenotypes)\n')
}  # end S9

t_s10 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S10 — bootstrap positional CI (100 bootstraps at peak locus)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(10)) {
cat('\nS10: bootstrap CI (n=250)...\n')
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
    num.samples  = 100,
    seed         = 1,
    use.REML     = TRUE,
    use.BLUP     = TRUE)

boot_scans <- run.positional.scans(
    sim.object       = boot_out,
    keep.full.scans  = FALSE,
    genomecache      = genomecache,
    data             = phenotypes$data,
    use.multi.impute = FALSE,
    scan.seed        = 1)

boot_peaks <- as.numeric(unlist(boot_scans))

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
}  # end S10

t_s11 <- proc.time()
# ─────────────────────────────────────────────────────────────────────────────
# S11 — instability scan
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(11)) {
cat('\nS11: instability scan (n=250 bootstrap)...\n')
simple_out <- generate.simple.sample.outcomes.matrix(
    formula     = sim.y.1 ~ 1,
    data        = phenotypes$data,
    pheno.id    = 'SUBJECT.NAME.1',
    method      = 'permutation',
    num.samples = 250,
    seed        = 1)
instab_scan <- instability.lm.scan(
    simple.sample.object = simple_out,
    genomecache          = genomecache,
    model                = 'additive',
    seed                 = 1,
    chr                  = qtl_chr)
# p.values is (n_scans x n_loci); reduce to median -log10(p) per locus
pval_mat  <- instab_scan$full.results$p.values
loci_s11  <- colnames(pval_mat)
instab_lod <- apply(-log10(pval_mat), 2, median, na.rm = TRUE)
write.csv(data.frame(
    locus  = loci_s11,
    chr    = instab_scan$full.results$chr,
    pos_mb = instab_scan$full.results$pos$Mb,
    lod    = as.numeric(instab_lod)),
    file.path(r_data, 'instability_r.csv'), row.names = FALSE)
cat('  instability scan done;', length(loci_s11), 'loci x',
    nrow(pval_mat), 'subsamples\n')
}  # end S11

t_plots <- proc.time()
# ── reference plots (only if S1/S6 ran and objects exist) ─────────────────
if (exists('rop_scan')) {
    png(file.path(r_plots, 'rop_scan.png'), width=3000, height=1500, res=300)
    genome.plotter.whole(scan.list = list(ROP = rop_scan))
    dev.off()

    png(file.path(r_plots, 'allele_effects_region.png'), width=2700, height=1500, res=300)
    allele.plotter.region(scan.object = rop_scan, chr = qtl_chr)
    dev.off()
}
t_end <- proc.time()

# ── timing summary ─────────────────────────────────────────────────────────
ts <- format(Sys.time(), '%Y-%m-%dT%H:%M:%S')
timing_rows <- list(
    data.frame(section='setup',        elapsed_s=elapsed_s(t_start,       t_after_setup), timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='probs_export', elapsed_s=elapsed_s(t_after_setup, t_after_probs), timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S1_cache_info',       elapsed_s=elapsed_s(t_s1,  t_s2),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S2_haplotype_probs',  elapsed_s=elapsed_s(t_s2,  t_s3),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S3_kinship_loco',     elapsed_s=elapsed_s(t_s3,  t_s4),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S4_lmm_scan',         elapsed_s=elapsed_s(t_s4,  t_s5),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S5_cache_collapse',   elapsed_s=elapsed_s(t_s5,  t_s6),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S6_permutation',      elapsed_s=elapsed_s(t_s6,  t_s7),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S7_mi_scan',          elapsed_s=elapsed_s(t_s7,  t_s8),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S8_snp_scan',         elapsed_s=elapsed_s(t_s8,  t_s9),  timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S9_mediation',        elapsed_s=elapsed_s(t_s9,  t_s10), timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S10_bootstrap_ci',    elapsed_s=elapsed_s(t_s10, t_s11), timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='S11_instability',     elapsed_s=elapsed_s(t_s11, t_plots), timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='reference_plots',     elapsed_s=elapsed_s(t_plots, t_end), timestamp=ts, stringsAsFactors=FALSE),
    data.frame(section='total',               elapsed_s=elapsed_s(t_start, t_end), timestamp=ts, stringsAsFactors=FALSE)
)
timing_df <- do.call(rbind, timing_rows)
write.csv(timing_df, file.path(r_data, 'timing_sections.csv'), row.names = FALSE)

cat('\n── Timing summary (R) ───────────────────────────────────────────────────\n')
for (i in seq_len(nrow(timing_df))) {
    cat(sprintf('  %-24s  %8.1f s  (%5.2f min)\n',
        timing_df$section[i],
        timing_df$elapsed_s[i],
        timing_df$elapsed_s[i] / 60))
}
cat('  timing written to:', file.path(r_data, 'timing_sections.csv'), '\n')

cat('\n── Phase 2 sim R reference complete ────────────────────────────────────\n')
cat('  outputs in:', r_data, '\n')
cat('  run next: python validation/validate_sim_p2.py\n')
