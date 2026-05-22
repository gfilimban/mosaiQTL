# port_sim_validation_p3.R — Phase 3: speed + memory benchmarking (miQTL side)
#
# Uses the same sparcc_cache genomecache but independent Phase 3 phenotypes:
#   num.lines=40  num.sim=50  num.replicates=3  seed=123
# giving 120 samples × 50 genes for a throughput-oriented benchmark.
#
# Builds everything from scratch: genomecache metadata derived directly,
# LOCO kinship computed in S3 and used for S4+.
# S5 (cache collapse) removed.  S8 gated on allele_dir.
#
# Outputs: validation/phase3/r_data/
#   phenotypes.csv, true_loci.csv   ← Phase 3 Python loads these
#   timing_sections.csv             ← wall-clock + peak heap per section
#
# Usage:
#   Rscript validation/r_reference/port_sim_validation_p3.R
#   Rscript validation/r_reference/port_sim_validation_p3.R --sections 3,4
#   Rscript validation/r_reference/port_sim_validation_p3.R --force

args         <- commandArgs(trailingOnly = TRUE)
sections_pos <- which(args == '--sections')
run_sections <- if (length(sections_pos) > 0 && sections_pos[1] < length(args)) {
    as.integer(strsplit(args[sections_pos[1] + 1], ',')[[1]])
} else 1:11
force_rerun  <- '--force' %in% args
run_section  <- function(n) n %in% run_sections
cat('Sections to run:', paste(run_sections, collapse = ', '),
    if (force_rerun) '[--force]' else '', '\n')

lib_path <- Sys.getenv('R_LIB_PATH',
    unset = '/Users/ghassan/miniconda3/envs/bioinfo/lib/R/library')
if (dir.exists(lib_path)) .libPaths(c(lib_path, .libPaths()))
for (pkg in c('sparcc', 'miqtl', 'evir')) suppressWarnings(library(pkg, character.only = TRUE))

if (!is.primitive(crossprod) || typeof(crossprod) != 'builtin') {
    assign('crossprod',  base::crossprod,  envir = globalenv())
    assign('tcrossprod', base::tcrossprod, envir = globalenv())
}

# ── paths ─────────────────────────────────────────────────────────────────────
genomecache <- 'validation/sparcc_cache'
r_data      <- 'validation/phase3/r_data'
allele_dir  <- 'validation/phase2/allele_dir'
dir.create(r_data, recursive = TRUE, showWarnings = FALSE)

# ── helpers ───────────────────────────────────────────────────────────────────
elapsed_s <- function(t0, t1) as.numeric(t1['elapsed'] - t0['elapsed'])

# Peak heap during a section: gc(reset=TRUE) resets the 'max used' counter;
# gc() after returns the max allocation reached.  Column 3 = 'max used (Mb)'.
peak_heap_mb <- function() {
    g  <- gc(verbose = FALSE)
    cn <- colnames(g)
    # 'max used' is stable across R versions; the '(Mb)' column always follows it
    sum(g[, which(cn == 'max used') + 1L])
}

timing_rows <- list()
ts          <- format(Sys.time(), '%Y-%m-%dT%H:%M:%S')

record <- function(section, elapsed, peak_mb) {
    timing_rows[[length(timing_rows) + 1]] <<- data.frame(
        section      = section,
        elapsed_s    = elapsed,
        peak_heap_mb = peak_mb,
        timestamp    = ts,
        stringsAsFactors = FALSE)
    cat(sprintf('  %-24s  %7.2f s  (%5.1f min)  peak heap %.0f MB\n',
        section, elapsed, elapsed / 60, peak_mb))
}

# ── Phase 3 setup: generate bench phenotypes + derive metadata from OLS scan ────
# No Phase 2 dependency — everything built from scratch here.
# reader3 is used by S2 for getLocusMatrix.
# scan_loci/scan_chr/scan_mb/loci_by_chr/chromosomes are derived via OLS scan so
# locus IDs match the canonical IDs used by scan.h2lmm and sim.CC.data.
reader3 <- DiploprobReader$new(genomecache)

# bench phenotypes (cached to avoid regenerating on --sections reruns)
bench_cache <- file.path(r_data, 'bench_cache_p3.RData')
if (!force_rerun && file.exists(bench_cache)) {
    cat('\nLoading Phase 3 bench phenotypes from', bench_cache, '...\n')
    load(bench_cache)   # bench_phenos, qtl_chr3, scan_loci, scan_chr, scan_mb,
                        #   loci_by_chr, chromosomes
} else {
    cat('\nGenerating Phase 3 bench phenotypes',
        '(num.sim=50, num.replicates=3, seed=123)...\n')
    set.seed(123)
    bench_phenos <- sim.CC.data(
        genomecache     = genomecache,
        num.lines       = 40,
        num.sim         = 50,
        num.replicates  = 3,
        qtl.effect.size = 0.8
    )
    cat('  simulated', length(bench_phenos$locus), 'planted QTL loci across 50 phenotypes\n')
    cat('  n samples:', nrow(bench_phenos$data), '\n')

    # OLS scan for metadata: K=NULL → fix.par=0 (pure OLS).
    # names(ols_meta$LOD) use the same locus IDs sim.CC.data uses for planted loci.
    cat('  Deriving genomecache metadata via OLS scan...\n')
    ols_meta    <- scan.h2lmm(
        genomecache           = genomecache,
        data                  = bench_phenos$data,
        pheno.id              = 'SUBJECT.NAME.1',
        geno.id               = 'SUBJECT.NAME.1',
        formula               = sim.y.1 ~ 1,
        use.multi.impute      = FALSE,
        return.allele.effects = FALSE)
    scan_loci   <- names(ols_meta$LOD)
    scan_chr    <- setNames(ols_meta$chr,    scan_loci)
    scan_mb     <- setNames(ols_meta$pos$Mb, scan_loci)
    loci_by_chr <- split(scan_loci, scan_chr)
    chromosomes <- names(loci_by_chr)
    rm(ols_meta); gc()
    cat(sprintf('  %d chromosomes, %d total loci\n', length(chromosomes), length(scan_loci)))

    qtl_chr3 <- unname(scan_chr[bench_phenos$locus[[1]]])
    cat('  planted locus chr (sim.y.1):', qtl_chr3,
        '  locus:', bench_phenos$locus[[1]], '\n')

    sim_cols <- c('SUBJECT.NAME.1', paste0('sim.y.', 1:50))
    sim_cols <- sim_cols[sim_cols %in% names(bench_phenos$data)]
    write.csv(bench_phenos$data[, sim_cols],
              file.path(r_data, 'phenotypes.csv'), row.names = FALSE)
    write.csv(data.frame(sim = seq_along(bench_phenos$locus),
                         locus = bench_phenos$locus),
              file.path(r_data, 'true_loci.csv'), row.names = FALSE)
    cat('  exported phenotypes.csv + true_loci.csv\n')

    # Export haplotype probs for Phase 3 strains so Python can build its own cache.
    # sim.CC.data(seed=123) selects different strains than Phase 2 (seed=42), so
    # Phase 3 Python cannot reuse the Phase 2 cache.h5.
    bench_strains <- bench_phenos$data$SUBJECT.NAME.1
    probs_dir_p3  <- file.path(r_data, 'probs')
    dir.create(probs_dir_p3, recursive = TRUE, showWarnings = FALSE)
    cat('  Exporting Phase 3 haplotype probs (', length(bench_strains), 'strains)...\n')
    for (chr in chromosomes) {
        chr_loci <- loci_by_chr[[chr]]
        rows <- lapply(chr_loci, function(loc) {
            m     <- reader3$getLocusMatrix(loc, model = 'additive')
            m     <- m / rowSums(m)
            avail <- bench_strains[bench_strains %in% rownames(m)]
            m_sub <- m[avail, , drop = FALSE]
            df        <- as.data.frame(m_sub)
            df$strain <- rownames(m_sub)
            df$locus  <- loc
            df$pos_mb <- scan_mb[loc]
            df
        })
        probs_df <- do.call(rbind, rows)
        write.csv(probs_df, file.path(probs_dir_p3, paste0('probs_', chr, '.csv')),
                  row.names = FALSE)
    }
    cat(sprintf('  probs exported: %d chromosomes\n', length(chromosomes)))

    save(bench_phenos, qtl_chr3, scan_loci, scan_chr, scan_mb, loci_by_chr, chromosomes,
         file = bench_cache)
    cat('  bench setup cached to', bench_cache, '\n')
}

# K_loco_bench: computed fresh in S3 and kept for S4+.
# If S3 is skipped via --sections, load from kinship cache or compute untimed.
K_loco_bench <- NULL
kinship_cache <- file.path(r_data, 'kinship_p3.RData')
if (!run_section(3)) {
    if (!force_rerun && file.exists(kinship_cache)) {
        load(kinship_cache)   # K_loco_bench
        cat('  K_loco_bench loaded from', kinship_cache, '\n')
    } else {
        cat('\nS3 not selected — computing K_loco_bench (untimed) for S4+...\n')
        cache_norm  <- normalizePath(genomecache)
        L_by_chr_b  <- sapply(chromosomes, function(chr)
            length(list.files(file.path(cache_norm, 'additive', paste0('chr', chr)))))
        L_all_b     <- sum(L_by_chr_b)
        K_all_b     <- suppressWarnings(calc.kinship.from.genomecache.with.DOQTL(genomecache))
        K_loco_bench <- list()
        for (chr_c in chromosomes) {
            tmp_c <- file.path(tempdir(), paste0('p3_kfb_', chr_c))
            for (md in c('additive', 'full', 'genotype')) {
                src <- file.path(cache_norm, md, paste0('chr', chr_c))
                if (dir.exists(src)) {
                    dir.create(file.path(tmp_c, md), recursive = TRUE, showWarnings = FALSE)
                    file.symlink(src, file.path(tmp_c, md, paste0('chr', chr_c)))
                }
            }
            K_c_b <- suppressWarnings(calc.kinship.from.genomecache.with.DOQTL(tmp_c))
            unlink(tmp_c, recursive = TRUE)
            K_b <- (L_all_b * K_all_b - L_by_chr_b[[chr_c]] * K_c_b) /
                   (L_all_b - L_by_chr_b[[chr_c]])
            K_loco_bench[[chr_c]] <- (K_b + t(K_b)) / 2
            rm(K_c_b, K_b)
        }
        rm(K_all_b); gc()
        save(K_loco_bench, file = kinship_cache)
        cat('  K_loco_bench cached to', kinship_cache, '\n')
    }
}

# rop_scan3 for S6/S9/S10/S11 — built once K_loco_bench is available
rop3_cache <- file.path(r_data, 'rop3_scan.RData')
if (!force_rerun && file.exists(rop3_cache)) {
    load(rop3_cache)   # rop_scan3
} else if (!is.null(K_loco_bench)) {
    cat('\nBuilding rop_scan3 (LOCO, chr', qtl_chr3, ')...\n')
    rop_scan3 <- scan.h2lmm(
        genomecache           = genomecache,
        data                  = bench_phenos$data,
        pheno.id              = 'SUBJECT.NAME.1',
        geno.id               = 'SUBJECT.NAME.1',
        formula               = sim.y.1 ~ 1,
        K                     = K_loco_bench[[qtl_chr3]],
        chr                   = qtl_chr3,
        use.multi.impute      = FALSE,
        return.allele.effects = FALSE)
    w1 <- rep(1, length(rop_scan3$fit1$y))
    names(w1) <- names(rop_scan3$fit1$y)
    rop_scan3$fit1$weights <- w1
    save(rop_scan3, file = rop3_cache)
    cat('  rop_scan3 cached\n')
}

# ─────────────────────────────────────────────────────────────────────────────
# S1 — cache I/O: DiploprobReader construction + spot-check locus reads
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(1)) {
cat('\nS1: cache I/O (DiploprobReader + locus matrix reads)...\n')
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
r_bench <- DiploprobReader$new(genomecache)
set.seed(1); bench_loci_s1 <- sample(scan_loci, min(50, length(scan_loci)))
invisible(lapply(bench_loci_s1, function(l)
    r_bench$getLocusMatrix(l, model = 'additive')))
rm(r_bench)
record('S1_cache_info', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ─────────────────────────────────────────────────────────────────────────────
# S2 — haplotype prob access (5 random loci per chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(2)) {
cat('\nS2: haplotype prob access (spot-check)...\n')
set.seed(7)
spot_loci3 <- unlist(lapply(names(loci_by_chr), function(chr) {
    sample(loci_by_chr[[chr]], min(5, length(loci_by_chr[[chr]])))
}))
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
invisible(lapply(spot_loci3, function(l) {
    m <- reader3$getLocusMatrix(l, model = 'additive')
    m / rowSums(m)
}))
record('S2_haplotype_probs', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ─────────────────────────────────────────────────────────────────────────────
# S3 — LOCO kinship: full recompute from genomecache (benchmark only)
# Result stored as K_loco_bench and used for S4+.
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(3)) {
cat('\nS3: LOCO kinship recompute (all chromosomes, benchmark)...\n')
cache_norm  <- normalizePath(genomecache)
cache_add   <- file.path(cache_norm, 'additive')
L_by_chr_b  <- sapply(names(loci_by_chr), function(chr)
    length(list.files(file.path(cache_add, paste0('chr', chr)))))
L_all_b <- sum(L_by_chr_b)

gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()

K_all_b      <- suppressWarnings(calc.kinship.from.genomecache.with.DOQTL(genomecache))
K_loco_bench <- list()
for (chr_c in chromosomes) {
    tmp_c <- file.path(tempdir(), paste0('p3_s3_', chr_c))
    for (md in c('additive', 'full', 'genotype')) {
        src <- file.path(cache_norm, md, paste0('chr', chr_c))
        if (dir.exists(src)) {
            dir.create(file.path(tmp_c, md), recursive = TRUE, showWarnings = FALSE)
            file.symlink(src, file.path(tmp_c, md, paste0('chr', chr_c)))
        }
    }
    K_c_b  <- suppressWarnings(calc.kinship.from.genomecache.with.DOQTL(tmp_c))
    unlink(tmp_c, recursive = TRUE)
    L_c_b  <- L_by_chr_b[[chr_c]]
    K_b    <- (L_all_b * K_all_b - L_c_b * K_c_b) / (L_all_b - L_c_b)
    K_loco_bench[[chr_c]] <- (K_b + t(K_b)) / 2
    rm(K_c_b, K_b)
}
rm(K_all_b); gc()
# cache for reruns where S3 is skipped
save(K_loco_bench, file = kinship_cache)

# build rop_scan3 now that K_loco_bench is ready (if not already cached)
if (!exists('rop_scan3')) {
    rop_scan3 <- scan.h2lmm(
        genomecache           = genomecache,
        data                  = bench_phenos$data,
        pheno.id              = 'SUBJECT.NAME.1',
        geno.id               = 'SUBJECT.NAME.1',
        formula               = sim.y.1 ~ 1,
        K                     = K_loco_bench[[qtl_chr3]],
        chr                   = qtl_chr3,
        use.multi.impute      = FALSE,
        return.allele.effects = FALSE)
    w1 <- rep(1, length(rop_scan3$fit1$y))
    names(w1) <- names(rop_scan3$fit1$y)
    rop_scan3$fit1$weights <- w1
    save(rop_scan3, file = rop3_cache)
}

record('S3_kinship_loco', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ─────────────────────────────────────────────────────────────────────────────
# S4 — LMM genome-wide scan × 50 phenotypes (LOCO K from Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(4)) {
cat('\nS4: LMM scan (50 phenotypes, genome-wide LOCO)...\n')
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()

lod_r1_df <- NULL
for (sim_i in 1:50) {
    pheno_col <- paste0('sim.y.', sim_i)
    if (!pheno_col %in% names(bench_phenos$data)) next
    lod_all <- c()
    for (chr_c in names(loci_by_chr)) {
        sc_c <- scan.h2lmm(
            genomecache           = genomecache,
            data                  = bench_phenos$data,
            pheno.id              = 'SUBJECT.NAME.1',
            geno.id               = 'SUBJECT.NAME.1',
            formula               = as.formula(paste(pheno_col, '~ 1')),
            K                     = K_loco_bench[[chr_c]],
            chr                   = chr_c,
            use.multi.impute      = FALSE,
            return.allele.effects = FALSE)
        lod_all[names(sc_c$LOD)] <- as.numeric(sc_c$LOD)
    }
    if (sim_i == 1) {
        lod_r1_df <- data.frame(
            locus  = names(lod_all),
            chr    = as.character(scan_chr[names(lod_all)]),
            pos_mb = scan_mb[names(lod_all)],
            lod    = as.numeric(lod_all),
            stringsAsFactors = FALSE)
    }
    if (sim_i %% 10 == 0)
        cat(sprintf('    sim.y.%d done  peak LOD=%.2f\n', sim_i, max(lod_all)))
}
if (!is.null(lod_r1_df)) {
    write.csv(lod_r1_df, file.path(r_data, 'lod_r.csv'), row.names = FALSE)
    cat('  exported lod_r.csv (sim.y.1)\n')
}

t1 <- proc.time()
peak_mb <- peak_heap_mb()
elapsed <- elapsed_s(t0, t1)
genes_min <- 50 / (elapsed / 60)
cat(sprintf('  throughput: %.1f genes/min  (%.2f s per gene)\n',
    genes_min, elapsed / 50))
record('S4_lmm_scan', elapsed, peak_mb)
}

# ─────────────────────────────────────────────────────────────────────────────
# S6 — permutation thresholds (n=250, peak chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(6)) {
cat('\nS6: permutation (n=250, chr', qtl_chr3, ')...\n')
K_s6    <- K_loco_bench[[qtl_chr3]]
K_eigs  <- eigen(K_s6, symmetric = TRUE, only.values = TRUE)$values
psd_tol <- max(abs(K_eigs)) * nrow(K_s6)^2 * .Machine$double.eps
if (min(K_eigs) < -psd_tol) {
    cat('  [SKIP] kinship not PSD for chr', qtl_chr3, '\n')
    timing_rows[[length(timing_rows) + 1]] <- data.frame(
        section='S6_permutation', elapsed_s=NA, peak_heap_mb=NA,
        timestamp=ts, stringsAsFactors=FALSE)
} else {
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
perm_out <- generate.sample.outcomes.matrix(
    scan.object = rop_scan3, method = 'permutation', num.samples = 250)
perm_scans <- run.threshold.scans(
    sim.threshold.object = perm_out, keep.full.scans = FALSE,
    genomecache = genomecache, data = bench_phenos$data,
    use.multi.impute = FALSE, scan.seed = 1)
max_lods <- as.numeric(perm_scans$max.statistics$LOD)
max_lods <- max_lods[is.finite(max_lods)]
gev_fit  <- evir::gev(max_lods)
thresh   <- evir::qgev(0.95, xi=gev_fit$par.est['xi'],
                       sigma=gev_fit$par.est['sigma'], mu=gev_fit$par.est['mu'])
cat('  GEV threshold:', round(unname(thresh), 3), '\n')
write.csv(data.frame(perm_index = seq_along(max_lods), max_lod = max_lods),
          file.path(r_data, 'perm_max_lods_r.csv'), row.names = FALSE)
write.csv(data.frame(threshold          = unname(thresh),
                     empirical_quantile = quantile(max_lods, 0.95, names = FALSE)),
          file.path(r_data, 'thresholds_r.csv'), row.names = FALSE)
cat('  exported perm_max_lods_r.csv + thresholds_r.csv\n')
record('S6_permutation', elapsed_s(t0, proc.time()), peak_heap_mb())
}  # end psd ok
}

# ─────────────────────────────────────────────────────────────────────────────
# S7 — MI scan (num.imp=11, peak chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(7)) {
cat('\nS7: MI scan (num.imp=11, chr', qtl_chr3, ')...\n')
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
mi_scan3 <- scan.h2lmm(
    genomecache = genomecache, data = bench_phenos$data,
    pheno.id = 'SUBJECT.NAME.1', geno.id = 'SUBJECT.NAME.1',
    formula = sim.y.1 ~ 1, K = K_loco_bench[[qtl_chr3]],
    chr = qtl_chr3, use.multi.impute = TRUE, num.imp = 11)
cat('  MI peak LOD:', round(max(as.numeric(mi_scan3$LOD)), 3), '\n')
mi_loci3 <- names(mi_scan3$LOD)
write.csv(data.frame(
    locus  = mi_loci3,
    chr    = as.character(scan_chr[mi_loci3]),
    pos_mb = scan_mb[mi_loci3],
    lod    = as.numeric(mi_scan3$LOD),
    stringsAsFactors = FALSE),
    file.path(r_data, 'mi_lod_r.csv'), row.names = FALSE)
cat('  exported mi_lod_r.csv\n')
record('S7_mi_scan', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ─────────────────────────────────────────────────────────────────────────────
# S8 — SNP scan: build chrN.alleles from GM_fo_mm39.csv, then imputed.snp.scan.h2lmm
# Same chrN.alleles format as Phase 2 S8. Timed for benchmarking.
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(8)) {
cat('\nS8: SNP scan (GM_fo_mm39.csv → chrN.alleles → imputed.snp.scan.h2lmm)...\n')
gm_csv_s8 <- '../input/GM_fo_mm39.csv'
allele_dir_p3 <- file.path(r_data, 'allele_dir')
if (!file.exists(gm_csv_s8)) {
    cat('  SKIPPED: GM_fo_mm39.csv not found at', gm_csv_s8, '\n')
    timing_rows[[length(timing_rows) + 1]] <- data.frame(
        section='S8_snp_scan', elapsed_s=NA, peak_heap_mb=NA,
        timestamp=ts, stringsAsFactors=FALSE)
} else {

gm_s8       <- read.csv(gm_csv_s8, stringsAsFactors = FALSE, check.names = FALSE)
h_s8        <- DiploprobReader$new(genomecache)
founders_s8 <- h_s8$getFounders()
loci_s8     <- loci_by_chr[[qtl_chr3]]

gm_cols_ordered <- c(AJ='A', B6J='B', '129S1J'='C', NODJ='D',
                     NZOJ='E', CASTJ='F', PWKJ='G', WSBJ='H')
cache_to_gm     <- setNames(names(gm_cols_ordered), gm_cols_ordered)
gm_founder_cols <- cache_to_gm[founders_s8]

gm_chr_s8 <- gm_s8[gm_s8$chr == qtl_chr3 & gm_s8$marker %in% loci_s8,
                    c('marker', gm_founder_cols), drop = FALSE]
rownames(gm_chr_s8) <- gm_chr_s8$marker
cat('  markers matched:', nrow(gm_chr_s8), '/', length(loci_s8), '\n')

dir.create(allele_dir_p3, recursive = TRUE, showWarnings = FALSE)
alleles_file <- file.path(allele_dir_p3, paste0('chr', qtl_chr3, '.alleles'))
na_row_tail  <- paste(rep('0', length(founders_s8)), collapse = '\t')
lines_out    <- character(4L * nrow(gm_chr_s8))
line_idx     <- 0L
for (snp_s8 in gm_chr_s8$marker) {
    alleles_s8 <- as.character(unlist(gm_chr_s8[snp_s8, gm_founder_cols]))
    alleles_s8[alleles_s8 %in% c('N', '', 'NA', NA)] <- NA
    distinct_s8 <- sort(na.omit(unique(alleles_s8)))
    a1   <- if (length(distinct_s8) >= 1L) distinct_s8[1L] else 'A'
    a2   <- if (length(distinct_s8) >= 2L) distinct_s8[2L] else 'T'
    cnt1 <- ifelse(is.na(alleles_s8), '0', ifelse(alleles_s8 == a1, '1', '0'))
    cnt2 <- ifelse(is.na(alleles_s8), '0', ifelse(alleles_s8 == a2, '1', '0'))
    # Format: line 1 = bare SNP_ID; lines 2-4 = leading-TAB + allele + counts.
    # Leading TAB: grep -w -A 3 only matches line 1 (returns 4 lines); the empty
    # first split field is removed by [,-1] inside extract.imputed.design.matrix,
    # leaving allele names in col 1 and 8 founder counts in cols 2-9.
    line_idx <- line_idx + 1L; lines_out[line_idx] <- snp_s8
    line_idx <- line_idx + 1L
    lines_out[line_idx] <- paste0('\t', a1, '\t', paste(cnt1, collapse='\t'))
    line_idx <- line_idx + 1L
    lines_out[line_idx] <- paste0('\t', a2, '\t', paste(cnt2, collapse='\t'))
    line_idx <- line_idx + 1L
    lines_out[line_idx] <- paste0('\tNA\t', na_row_tail)
}
writeLines(lines_out[seq_len(line_idx)], alleles_file)
cat('  alleles written:', nrow(gm_chr_s8), 'SNP blocks\n')

gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
# imputed.snp.scan.h2lmm hardcodes SUBJECT.NAME internally; alias column.
bench_phenos_s8 <- bench_phenos$data
bench_phenos_s8$SUBJECT.NAME <- bench_phenos_s8$SUBJECT.NAME.1
snp3 <- imputed.snp.scan.h2lmm(
    data             = bench_phenos_s8,
    formula          = sim.y.1 ~ 1,
    K                = K_loco_bench[[qtl_chr3]],
    allele.dir       = allele_dir_p3,
    genomecache      = genomecache,
    chr              = qtl_chr3,
    just.these.loci  = gm_chr_s8$marker)
cat('  SNP peak LOD:', round(max(as.numeric(snp3$LOD)), 3), '\n')

snp3_df <- data.frame(
    snp    = snp3$loci,
    chr    = snp3$chr,
    pos_mb = snp3$pos$Mb,
    lod    = as.numeric(snp3$LOD),
    stringsAsFactors = FALSE)
write.csv(snp3_df, file.path(r_data, 'snp_lod_r.csv'), row.names = FALSE)
cat('  exported snp_lod_r.csv:', nrow(snp3_df), 'SNPs\n')

record('S8_snp_scan', elapsed_s(t0, proc.time()), peak_heap_mb())
}  # end gm_csv exists
}

# ─────────────────────────────────────────────────────────────────────────────
# S9 — mediation (outcome=sim.y.1, mediator=sim.y.2, peak chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(9)) {
cat('\nS9: mediation (chr', qtl_chr3, ')...\n')
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
med3 <- scan.h2lmm(
    genomecache = genomecache, data = bench_phenos$data,
    pheno.id = 'SUBJECT.NAME.1', geno.id = 'SUBJECT.NAME.1',
    formula = sim.y.1 ~ sim.y.2, K = K_loco_bench[[qtl_chr3]],
    use.multi.impute = FALSE, chr = qtl_chr3)
cat('  mediation done;', length(med3$LOD), 'loci\n')
med_loci3 <- names(med3$LOD)
write.csv(data.frame(
    locus  = med_loci3,
    chr    = as.character(scan_chr[med_loci3]),
    pos_mb = scan_mb[med_loci3],
    lod    = as.numeric(med3$LOD),
    stringsAsFactors = FALSE),
    file.path(r_data, 'mediation_r.csv'), row.names = FALSE)
cat('  exported mediation_r.csv\n')
record('S9_mediation', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ─────────────────────────────────────────────────────────────────────────────
# S10 — bootstrap CI (n=100, peak locus)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(10)) {
cat('\nS10: bootstrap CI (n=100, chr', qtl_chr3, ')...\n')
qtl_fit3 <- scan.h2lmm(
    genomecache = genomecache, data = bench_phenos$data,
    pheno.id = 'SUBJECT.NAME.1', geno.id = 'SUBJECT.NAME.1',
    formula = sim.y.1 ~ 1, K = K_loco_bench[[qtl_chr3]],
    chr = qtl_chr3, use.multi.impute = FALSE,
    just.these.loci = names(rop_scan3$LOD)[which.max(as.numeric(rop_scan3$LOD))])
w3 <- rep(1, length(qtl_fit3$fit1$y))
names(w3) <- names(qtl_fit3$fit1$y)
qtl_fit3$fit1$weights <- w3

gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
boot3_out <- generate.sample.outcomes.matrix(
    scan.object = qtl_fit3, model.type = 'alt', method = 'bootstrap',
    num.samples = 100, seed = 1, use.REML = TRUE, use.BLUP = TRUE)
boot3_scans <- run.positional.scans(
    sim.object = boot3_out, keep.full.scans = FALSE,
    genomecache = genomecache, data = bench_phenos$data,
    use.multi.impute = FALSE, scan.seed = 1)
cat('  bootstrap CI done; n peaks =', length(unlist(boot3_scans)), '\n')
record('S10_bootstrap_ci', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ─────────────────────────────────────────────────────────────────────────────
# S11 — instability scan (n=250, peak chromosome)
# ─────────────────────────────────────────────────────────────────────────────
if (run_section(11)) {
cat('\nS11: instability (n=250, chr', qtl_chr3, ')...\n')
gc(reset = TRUE, verbose = FALSE)
t0 <- proc.time()
simple3 <- generate.simple.sample.outcomes.matrix(
    formula = sim.y.1 ~ 1, data = bench_phenos$data,
    pheno.id = 'SUBJECT.NAME.1', method = 'permutation',
    num.samples = 250, seed = 1)
instab3 <- instability.lm.scan(
    simple.sample.object = simple3, genomecache = genomecache,
    model = 'additive', seed = 1, chr = qtl_chr3)
cat('  instability done;', ncol(instab3$full.results$p.values), 'loci\n')
p_mat3     <- instab3$full.results$p.values   # (n_subsamples × n_loci)
instab_loci3 <- colnames(p_mat3)
instab_lod3  <- colMeans(-log10(p_mat3 + 1e-300), na.rm = TRUE)
write.csv(data.frame(
    locus  = instab_loci3,
    chr    = rep(qtl_chr3, length(instab_loci3)),
    pos_mb = scan_mb[instab_loci3],
    lod    = instab_lod3,
    stringsAsFactors = FALSE),
    file.path(r_data, 'instability_r.csv'), row.names = FALSE)
cat('  exported instability_r.csv\n')
record('S11_instability', elapsed_s(t0, proc.time()), peak_heap_mb())
}

# ── total + pipeline_from_scratch + export ────────────────────────────────
get_s <- function(df, sec) {
    r <- df[df$section == sec, 'elapsed_s']
    if (length(r) == 1 && !is.na(r)) r else NA_real_
}

total_s <- sum(sapply(timing_rows, function(r) r$elapsed_s), na.rm = TRUE)
timing_rows[[length(timing_rows) + 1]] <- data.frame(
    section = 'total', elapsed_s = total_s, peak_heap_mb = NA,
    timestamp = ts, stringsAsFactors = FALSE)

timing_df <- do.call(rbind, timing_rows)

# pipeline_from_scratch = S1 (cache I/O) + S3 (kinship build) + S4 (scan)
# represents the full cost a user pays on a fresh run
s1_t <- get_s(timing_df, 'S1_cache_info')
s3_t <- get_s(timing_df, 'S3_kinship_loco')
s4_t <- get_s(timing_df, 'S4_lmm_scan')
pipe_s <- sum(c(s1_t, s3_t, s4_t), na.rm = TRUE)
if (pipe_s > 0) {
    timing_rows[[length(timing_rows) + 1]] <- data.frame(
        section = 'pipeline_from_scratch', elapsed_s = pipe_s, peak_heap_mb = NA,
        timestamp = ts, stringsAsFactors = FALSE)
    timing_df <- do.call(rbind, timing_rows)
}

write.csv(timing_df, file.path(r_data, 'timing_sections.csv'), row.names = FALSE)

# S4 throughput
s4_row <- timing_df[timing_df$section == 'S4_lmm_scan', ]
if (nrow(s4_row) > 0 && !is.na(s4_row$elapsed_s)) {
    g_min <- 50 / (s4_row$elapsed_s / 60)
    cat(sprintf('\n  S4 throughput (R/miQTL): %.1f genes/min  (%.2f s/gene, 50 genes genome-wide)\n',
        g_min, s4_row$elapsed_s / 50))
}
pipe_row <- timing_df[timing_df$section == 'pipeline_from_scratch', ]
if (nrow(pipe_row) > 0)
    cat(sprintf('  pipeline_from_scratch (S1+S3+S4): %.1f s  (%.1f min)\n',
        pipe_row$elapsed_s, pipe_row$elapsed_s / 60))

cat('\n── Phase 3 R timing summary ─────────────────────────────────────────────\n')
for (i in seq_len(nrow(timing_df))) {
    s <- timing_df$elapsed_s[i]
    if (!is.na(s)) {
        cat(sprintf('  %-24s  %7.2f s  (%5.1f min)  peak %.0f MB\n',
            timing_df$section[i], s, s / 60,
            ifelse(is.na(timing_df$peak_heap_mb[i]), 0, timing_df$peak_heap_mb[i])))
    } else {
        cat(sprintf('  %-24s  SKIPPED\n', timing_df$section[i]))
    }
}
cat('\n  timing written to:', file.path(r_data, 'timing_sections.csv'), '\n')
cat('  run next: python validation/validate_phase3.py\n')
