# sim_validation.R — complete miQTL reference for mosaiQTL validation

# simulated dataset input

# devtools::install_github("gkeele/sparcc")

library(sparcc)
library(miqtl)

## sparcc genome cache — download once if not already present
genomecache <- 'validation/sparcc_cache'
if (!dir.exists(genomecache)) {
  download.file(url = 'https://github.com/gkeele/sparcc_cache/archive/master.zip',
                destfile = 'validation/sparcc_cache.zip')
  unzip('validation/sparcc_cache.zip', exdir = 'validation/')
  file.rename('validation/sparcc_cache-master', genomecache)
}

phenotypes <- sim.CC.data(genomecache = genomecache,
                          num.lines = 40,
                          num.sim = 10,
                          num.replicates = 2,
                          qtl.effect.size = 0.8)


# QTL scan using ROP which handle haplotype uncertainty using regression on probabilities (ROP)

miqtl.rop.scan <- scan.h2lmm(genomecache = genomecache,
                             data = phenotypes$data,
                             pheno.id = "SUBJECT.NAME.1",
                             geno.id =  "SUBJECT.NAME.1",
                             formula = sim.y.1 ~ 1,
                             use.multi.impute = FALSE,
                             return.allele.effects = TRUE)

# QTL scan using multiple imputation

miqtl.mi11.scan <- scan.h2lmm(genomecache = genomecache,
                              data = phenotypes$data,
                              pheno.id = "SUBJECT.NAME.1",
                              geno.id = "SUBJECT.NAME.1",
                              formula=sim.y.1 ~ 1,
                              use.multi.impute = TRUE,
                              num.imp = 11)

# permutation threshold
# fit0$weights is NULL and fit0$K is NULL — generate.sample.outcomes.matrix does
# rep(1, nrow(K)) to initialise weights, which fails when K is NULL.
# Named weights let K stay NULL (no-kinship path, correct for tau2.mle=0) and
# allow scan.h2lmm to index weights by subject name without producing NAs.
w <- rep(1, length(miqtl.rop.scan$fit0$y))
names(w) <- names(miqtl.rop.scan$fit0$y)
miqtl.rop.scan$fit0$weights <- w

permuted_phenotype <- generate.sample.outcomes.matrix(scan.object = miqtl.rop.scan,
                                                      method = "permutation", num.samples = 500)

permuted_scans <- run.threshold.scans(sim.threshold.object = permuted_phenotype,
                                      keep.full.scans=FALSE,
                                      genomecache  = genomecache,
                                      data = phenotypes$data,
                                      use.multi.impute = FALSE,
                                      scan.seed = 1)

permute_threshold <- get.gev.thresholds(threshold.scans = permuted_scans,
                                        percentile = 0.95)

# export raw max-LODs for empirical comparison in Python (avoids GEV instability at n=100)
perm_max_lods_r <- if (is.numeric(permuted_scans)) {
  as.numeric(permuted_scans)
} else {
  sapply(permuted_scans, function(s) max(as.numeric(s$LOD), na.rm = TRUE))
}
write.csv(data.frame(perm_index = seq_along(perm_max_lods_r) - 1L,
                     max_lod    = perm_max_lods_r),
          'validation/phase1/r_data/perm_max_lods.csv', row.names = FALSE)

# peak QTL locus

qtl_locus <- grab.locus.from.scan(miqtl.rop.scan)

# QTL positional confidence interval using parametric bootstraps

qtl.fit <- scan.h2lmm(genomecache = genomecache,
                      data = phenotypes$data,
                      pheno.id = "SUBJECT.NAME.1",
                      geno.id =  "SUBJECT.NAME.1",
                      formula = sim.y.1 ~ 1,
                      use.multi.impute=FALSE,
                      just.these.loci = qtl_locus)

w1 <- rep(1, length(qtl.fit$fit1$y))
names(w1) <- names(qtl.fit$fit1$y)
qtl.fit$fit1$weights <- w1

qtl.locus.bootstraps <- generate.sample.outcomes.matrix(scan.object = qtl.fit,
                                                        model.type = "alt",
                                                        method = "bootstrap",
                                                        num.sample = 100,
                                                        seed = 1,
                                                        use.REML = TRUE,
                                                        use.BLUP = TRUE)

qtl.bootstrap.scans <- run.positional.scans(sim.object = qtl.locus.bootstraps,
                                            keep.full.scans=TRUE,
                                            genomecache = genomecache,
                                            data = phenotypes$data,
                                            use.multi.impute = FALSE,
                                            scan.seed = 1)


# peak locus index — used by plots and exports below
qtl_idx_r <- which.max(miqtl.rop.scan$LOD)
qtl_chr   <- miqtl.rop.scan$chr[qtl_idx_r]

# ── save R plots ──────────────────────────────────────────────────────────────
dir.create('validation/phase1/r_plots', recursive=TRUE, showWarnings=FALSE)

# genome-wide ROP LOD profile — reference for P1-S1 (LOD r² ≥ 0.999 vs Python)
png('validation/phase1/r_plots/rop_scan.png', width=3000, height=1500, res=300)
genome.plotter.whole(scan.list=list(ROP = miqtl.rop.scan))
dev.off()

# genome-wide MI LOD profile (11 imputations) — reference for P1-S2 (LOD r ≥ 0.75 vs Python)
png('validation/phase1/r_plots/mi_scan.png', width=3000, height=1500, res=300)
genome.plotter.whole(scan.list = list(miqtl.mi11.scan))
dev.off()

# ROP LOD profile with permutation significance threshold overlaid — reference for P1-S3
png('validation/phase1/r_plots/rop_with_threshold.png', width=3000, height=1500, res=300)
genome.plotter.whole(scan.list=list(ROP = miqtl.rop.scan),
                     hard.thresholds = permute_threshold)
dev.off()

# founder haplotype probability heatmap at peak locus — confirms which founders drive signal
png('validation/phase1/r_plots/prob_heatmap.png', width=2400, height=2100, res=300)
prob.heatmap(marker = qtl_locus,
             p.value = min(miqtl.rop.scan$p.value),
             genomecache = genomecache,
             model = "additive",
             phenotype = "sim.y.1",
             phenotype.data = dplyr::rename(phenotypes$data, SUBJECT.NAME = SUBJECT.NAME.1))
dev.off()

# single-chromosome LOD with 100-bootstrap positional CI — validates CI width and coverage
png('validation/phase1/r_plots/chr_lod_with_ci.png', width=2700, height=1500, res=300)
single.chr.plotter.w.ci(scan.object = miqtl.rop.scan,
                        qtl.ci.object = qtl.bootstrap.scans,
                        ci.type = "bootstrap",
                        scan.type = "ROP",
                        scale = "Mb",
                        these.col = rep("#7BAFD4", 100))
dev.off()

# per-founder LOD contribution across QTL chromosome — shows haplotype association profile
png('validation/phase1/r_plots/founder_effects_region.png', width=2700, height=1500, res=300)
genome.plotter.region(haplotype.association = list(miqtl.rop.scan),
                      chr = qtl_chr)
dev.off()

# per-founder allele effect sizes (beta) across QTL chromosome — reference for allele effects comparison
# ylim clipped to ±3 — rare founders (H/WSB, G/PWK) near-zero probability → blown-up coefs at some loci
png('validation/phase1/r_plots/allele_effects_region.png', width=2700, height=1500, res=300)
allele.plotter.region(scan.object = miqtl.rop.scan, chr = qtl_chr)
dev.off()


# ── export for Python Phase 1 validation ─────────────────────────────────────
# minimal additions — workflow logic above is unchanged
dir.create('validation/phase1/r_data', recursive=TRUE, showWarnings=FALSE)

write.csv(phenotypes$data[, c('SUBJECT.NAME.1', 'sim.y.1')],
          'validation/phase1/r_data/phenotypes.csv', row.names=FALSE)

write.csv(data.frame(sim=seq_along(phenotypes$locus), locus=phenotypes$locus),
          'validation/phase1/r_data/true_loci.csv', row.names=FALSE)

write.csv(data.frame(locus=names(miqtl.rop.scan$LOD),
                     chr=miqtl.rop.scan$chr,
                     pos_mb=miqtl.rop.scan$pos$Mb,
                     lod=as.numeric(miqtl.rop.scan$LOD)),
          'validation/phase1/r_data/rop_lod.csv', row.names=FALSE)

write.csv(data.frame(locus=names(miqtl.mi11.scan$LOD),
                     chr=miqtl.mi11.scan$chr,
                     pos_mb=miqtl.mi11.scan$pos$Mb,
                     lod=as.numeric(miqtl.mi11.scan$LOD)),
          'validation/phase1/r_data/mi_lod.csv', row.names=FALSE)

write.csv(data.frame(threshold=unname(permute_threshold)),
          'validation/phase1/r_data/perm_threshold.csv', row.names=FALSE)

write.csv(data.frame(
  locus       = names(miqtl.rop.scan$LOD)[qtl_idx_r],
  chromosome  = miqtl.rop.scan$chr[qtl_idx_r],
  position_mb = miqtl.rop.scan$pos$Mb[qtl_idx_r],
  lod         = as.numeric(miqtl.rop.scan$LOD)[qtl_idx_r]
), 'validation/phase1/r_data/qtl_locus.csv', row.names=FALSE)

effects_at_peak <- miqtl.rop.scan$allele.effects[, names(miqtl.rop.scan$LOD)[qtl_idx_r]]
write.csv(data.frame(
  founder = names(effects_at_peak),
  beta    = as.numeric(effects_at_peak)
), 'validation/phase1/r_data/allele_effects_peak.csv', row.names=FALSE)

# ── export haplotype probabilities (scan loci only, pseudomarkers excluded) ──
# Python reads these CSVs directly — no rpy2 / R object crossing the boundary
dir.create('validation/phase1/r_data/probs', recursive=TRUE, showWarnings=FALSE)
reader     <- DiploprobReader$new(genomecache)
scan_loci  <- names(miqtl.rop.scan$LOD)
scan_chr   <- setNames(miqtl.rop.scan$chr,    scan_loci)
scan_mb    <- setNames(miqtl.rop.scan$pos$Mb, scan_loci)

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
            file.path('validation/phase1/r_data/probs',
                      paste0('probs_', chr, '.csv')),
            row.names = FALSE)
  cat('probs exported:', chr, '|', length(chr_loci), 'loci x',
      nrow(rows[[1]]), 'strains\n')
}
