# build_genomecache.R
# Builds the mosaiQTL HAPPY genome cache from the GigaMUGA consensus matrix.
#
# Design rationale (qtl2-informed):
#   The canonical qtl2convert workflow (Broman lab, kbroman.org/qtl2/pages/prep_do_data.html)
#   uses find_unique_geno() on FOUNDER genotypes to identify the two alleles per marker,
#   then encode_geno() to recode both founders and study samples (CC strains) into A/H/B/-
#   using those founder-derived codes. Markers where founders are monomorphic or fully
#   missing across all 8 lines are dropped — they carry zero haplotype-discrimination
#   power for the HAPPY HMM, which assigns founder haplotypes.
#
#   The prior version applied find_unique_geno to CC strains — incorrect because:
#   (a) it conflates missingness in the 33 CC panel with true founder monomorphism;
#   (b) it admits markers biallelic only by chance among CC strains but uninformative
#       for founder-haplotype assignment;
#   (c) it contradicts the qtl2 reference pipeline, which uses founder genotypes as
#       the allele-code source ("find_unique_geno() to determine the SNP alleles [from
#       founders]", kbroman.org/qtl2/pages/prep_do_data.html).
#
#   CC cross_info: qtl2's DO HMM uses only cross_info[0] as n_gen. For CC mice we
#   set n_gen=1 because the CC consensus genetic map already encodes the cumulative
#   recombination from the full breeding programme. The per-strain funnel data
#   (cc_crossinfo.csv) is not used here — it is reserved for a future mosaiQTL
#   Python extension (cc_funnel_init) that uses funnel-aware per-strain HMM init.
#
#   Pseudomarkers (insert_pseudomarkers) are NOT used in HAPPY cache. They are added
#   to probs_cc.rds/kinship for scan use, then filtered before convert.qtl2.to.HAPPY()
#   so HAPPY loci match gmap.csv exactly (Python from_gigamuga reads only gmap markers).
#
# Steps:
#   1. Load GigaMUGA consensus matrix + genetic/physical maps
#   2. Intersect markers: gm ∩ gmap ∩ pmap
#   3. find_unique_geno() on FOUNDERS → drop monomorphic/missing markers
#   4. encode_geno() on founders (allele codes) + encode_geno() on CC with same codes
#   5. Write per-chr geno CSVs + flat gmap.csv + pmap.csv + control.yaml
#   6. read_cross2() → calc_genoprob() (genotyped markers only, no pseudomarkers)
#   7. Build deterministic founder + F1 prob arrays
#   8. Merge CC + founders + F1s → genoprob_to_alleleprob() → calc_kinship(loco)
#   9. convert.qtl2.to.HAPPY() → HAPPY cache
#
# Usage: Rscript build_genomecache.R [base_dir] [n_cores]

suppressPackageStartupMessages({
  library(qtl2)        # read_cross2, calc_genoprob, genoprob_to_alleleprob, calc_kinship
  library(qtl2convert) # find_unique_geno + encode_geno — applied to founders (allele-code ground truth)
  library(abind)       # abind — merge sample arrays along dim 1
  library(miqtl)       # convert.qtl2.to.HAPPY
  library(data.table)  # fread — fast GigaMUGA loading
})

# ── args ───────────────────────────────────────────────────────────────────────
# Usage: Rscript build_genomecache.R [base_dir] [n_cores] [step_cM]
#   step_cM : pseudomarker spacing in cM (default 0.2, matching qtl2 insert_pseudomarkers
#             default). pass 0 to disable. pseudomarkers are included in probs_*.rds and
#             kinship_loco.rds but filtered out before convert.qtl2.to.HAPPY() so the
#             HAPPY cache contains only genotyped markers — Python from_gigamuga reads
#             only gmap.csv markers, so HAPPY loci must match gmap.csv exactly.
args     <- commandArgs(trailingOnly = TRUE)
base_dir <- if (length(args) >= 1) args[1] else
  '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl'
n_cores  <- if (length(args) >= 2) as.integer(args[2]) else 4L
step_cM  <- if (length(args) >= 3) as.numeric(args[3]) else 0.2

# ── paths ──────────────────────────────────────────────────────────────────────
gigamuga_path    <- file.path(base_dir, 'input', 'GM_ccfo_mm39.csv')
gmap_dir         <- file.path(base_dir, 'input', 'GM', 'gmap')
pmap_dir         <- file.path(base_dir, 'input', 'GM', 'pmap')
geno_dir         <- file.path(base_dir, 'input', 'geno')
founder_dir      <- file.path(base_dir, 'input', 'founder_geno')
genomecache_path <- file.path(base_dir, 'output', 'genomecache')
kinship_out      <- file.path(base_dir, 'output', 'kinship_loco.rds')
probs_cc_out     <- file.path(base_dir, 'output', 'probs_cc.rds')
probs_f_out      <- file.path(base_dir, 'output', 'probs_founder.rds')
probs_f1_out     <- file.path(base_dir, 'output', 'probs_f1.rds')

for (d in c(geno_dir, founder_dir, genomecache_path, dirname(kinship_out)))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ── strain vectors ─────────────────────────────────────────────────────────────
cc_cols <- c(
  'CC001','CC002','CC003','CC004','CC005','CC006','CC007','CC008',
  'CC009','CC010','CC011','CC012','CC013','CC015','CC017','CC018',
  'CC024','CC025','CC028','CC029','CC030','CC032','CC036','CC037',
  'CC038','CC041','CC043','CC057','CC060','CC062','CC065','CC071','CC074')

# canonical qtl2/CC founder order and single-letter codes (A-H)
founder_cols  <- c('AJ','B6J','129S1J','NODJ','NZOJ','CASTJ','PWKJ','WSBJ')
founder_codes <- c('A','B',  'C',     'D',   'E',   'F',    'G',   'H')

# B6J (code B) is dam for all 7 F1 crosses
f1_crosses <- data.frame(
  sample_id    = c('B6AF1J','B6129S1F1J','B6NODF1J','B6NZOF1J',
                   'B6CASTF1J','B6PWKF1J','B6WSBF1J'),
  parent_code1 = rep('B', 7),
  parent_code2 = c('A','C','D','E','F','G','H'),
  stringsAsFactors = FALSE)

message('\n[1] loading GigaMUGA consensus matrix...')
gm <- fread(gigamuga_path, data.table = FALSE)
rownames(gm) <- gm$marker
message('  loaded ', nrow(gm), ' markers')

# ── load genetic + physical maps ───────────────────────────────────────────────
read_map <- function(dir) {
  do.call(rbind, lapply(
    list.files(dir, pattern = '^GM_(g|p)map[0-9XYM]+\\.csv$', full.names = TRUE),
    read.csv, stringsAsFactors = FALSE, comment.char = '#'))
}
gmap_raw <- read_map(gmap_dir); rownames(gmap_raw) <- gmap_raw$marker
pmap_raw <- read_map(pmap_dir); rownames(pmap_raw) <- pmap_raw$marker

# ── step 2: intersect gm ∩ gmap ∩ pmap ────────────────────────────────────────
shared_markers <- Reduce(intersect, list(rownames(gm), rownames(gmap_raw), rownames(pmap_raw)))
message('[2] shared markers (gm ∩ gmap ∩ pmap): ', length(shared_markers))

gm       <- gm[shared_markers, ]
gmap_raw <- gmap_raw[shared_markers, ]
pmap_raw <- pmap_raw[shared_markers, ]

# replace N (GigaMUGA missing code) with - (qtl2 convention)
geno_founders <- as.matrix(gm[, founder_cols])
geno_cc       <- as.matrix(gm[, cc_cols])
geno_founders_clean <- replace(geno_founders, geno_founders == 'N', '-')
geno_cc_clean       <- replace(geno_cc,       geno_cc == 'N',       '-')

# ── step 3: find_unique_geno on FOUNDERS (qtl2 allele-code ground truth) ───────
# Applied to the 8 founder columns — exactly as in the qtl2 reference pipeline.
# Returns marker x 2 matrix; NA row = monomorphic or fully-missing in founders.
# These markers have no power to distinguish founder haplotypes → drop them.
allele_codes  <- find_unique_geno(geno = geno_founders_clean)
valid_markers <- rownames(allele_codes)[!apply(is.na(allele_codes), 1, any)]
message('[3] dropping ', nrow(allele_codes) - length(valid_markers),
        ' markers monomorphic/missing across 8 founders (find_unique_geno on founders)')
message('    valid markers: ', length(valid_markers))

allele_codes        <- allele_codes[valid_markers, , drop = FALSE]
geno_founders_clean <- geno_founders_clean[valid_markers, , drop = FALSE]
geno_cc_clean       <- geno_cc_clean[valid_markers, , drop = FALSE]
gmap_raw            <- gmap_raw[valid_markers, ]
pmap_raw            <- pmap_raw[valid_markers, ]

# ── step 4: encode_geno on founders then CC using founder-derived allele codes ─
# Founder-derived allele_codes define the A/B frame; both founders and CC strains
# are encoded with the same codes so haplotype assignments are founder-reference
# consistent throughout the pipeline.
geno_founders_encoded <- encode_geno(geno = geno_founders_clean, allele_codes = allele_codes)
geno_cc_encoded       <- encode_geno(geno = geno_cc_clean,       allele_codes = allele_codes)
message('[4] encoding complete: ', nrow(geno_founders_encoded), ' markers')

# ── step 5: final sync — drop NA-chr rows, reconcile ──────────────────────────
gmap_raw <- gmap_raw[!is.na(gmap_raw$chr), ]
pmap_raw <- pmap_raw[!is.na(pmap_raw$chr), ]

final_markers <- Reduce(intersect, list(
  rownames(geno_founders_encoded), rownames(geno_cc_encoded),
  rownames(gmap_raw),              rownames(pmap_raw)))

geno_founders_encoded <- geno_founders_encoded[final_markers, , drop = FALSE]
geno_cc_encoded       <- geno_cc_encoded[final_markers, , drop = FALSE]
gmap_raw              <- gmap_raw[final_markers, ]
pmap_raw              <- pmap_raw[final_markers, ]

# exclude Y, M — no valid genetic map positions
autosomes   <- sort(as.integer(unique(gmap_raw$chr[!gmap_raw$chr %in% c('X','Y','M')])))
chromosomes <- c(as.character(autosomes), intersect('X', unique(gmap_raw$chr)))

gmap_list <- lapply(setNames(chromosomes, chromosomes), function(chr) {
  df <- gmap_raw[gmap_raw$chr == chr, ]
  sort(setNames(df$pos, df$marker))
})
pmap_list <- lapply(setNames(chromosomes, chromosomes), function(chr) {
  df <- pmap_raw[pmap_raw$chr == chr, ]
  sort(setNames(df$pos, df$marker))
})
message('[5] final markers: ', length(final_markers),
        '  chromosomes: ', paste(chromosomes, collapse = ', '))

# ── step 6: write per-chr CSVs + flat map files + control.yaml ────────────────
for (chr in chromosomes) {
  m <- names(gmap_list[[chr]])
  write.csv(t(geno_cc_encoded[m, , drop = FALSE]),
            file = file.path(geno_dir, paste0('cc_chr', chr, '.csv')), quote = FALSE)
  write.csv(t(geno_founders_encoded[m, , drop = FALSE]),
            file = file.path(founder_dir, paste0('founder_chr', chr, '.csv')), quote = FALSE)
}

# flat gmap/pmap — written from final_markers; Python from_gigamuga reads these
# to reproduce the exact same marker set → loci_max_delta = 0 in S1
gmap_flat <- do.call(rbind, lapply(names(gmap_list), function(chr)
  data.frame(marker = names(gmap_list[[chr]]), chr = chr,
             pos = unname(gmap_list[[chr]]), stringsAsFactors = FALSE)))
pmap_flat <- do.call(rbind, lapply(names(pmap_list), function(chr)
  data.frame(marker = names(pmap_list[[chr]]), chr = chr,
             pos = unname(pmap_list[[chr]]), stringsAsFactors = FALSE)))

write.csv(gmap_flat, file.path(base_dir, 'input', 'gmap.csv'), quote = FALSE, row.names = FALSE)
write.csv(pmap_flat, file.path(base_dir, 'input', 'pmap.csv'), quote = FALSE, row.names = FALSE)

yaml_path <- file.path(base_dir, 'input', 'control.yaml')
# crosstype: do is correct for CC — CC and DO share the same 8-founder HMM
# structure in qtl2. the distinction is in cross_info/ngen below, not crosstype.
writeLines(c('crosstype: do',
             'geno:',         paste0('  - geno/cc_chr',             chromosomes, '.csv'),
             'founder_geno:', paste0('  - founder_geno/founder_chr', chromosomes, '.csv'),
             'gmap: gmap.csv', 'pmap: pmap.csv', 'na.strings: "-"',
             'alleles: [A, B, C, D, E, F, G, H]', 'x_chr: X',
             'genotypes:', '  A: 1', '  H: 2', '  B: 3'),
           yaml_path)

cross <- read_cross2(yaml_path)
message('[6] cross loaded: ', nrow(cross$geno[[1]]), ' CC strains')

# per-strain missing rate QC (informational)
miss <- apply(geno_cc_encoded == '-', 2, mean)
message('[QC] per-strain missing rate — min: ', round(min(miss), 3),
        '  median: ', round(median(miss), 3), '  max: ', round(max(miss), 3))
high_miss <- names(miss)[miss > 0.10]
if (length(high_miss))
  message('  WARNING >10% missing: ', paste(high_miss, collapse = ', '))

# ── step 7: HMM — CC genoprobs ─────────────────────────────────────────────────
# pseudomarkers (step_cM > 0): insert_pseudomarkers() adds dense positions for scan
# use — probs_cc.rds, kinship, and Python probs_cc.h5 all include pseudomarkers.
# HAPPY cache must NOT include them: convert.qtl2.to.HAPPY() writes whatever loci
# are in probs, and Python from_gigamuga reads only gmap.csv markers → loci mismatch.
# fix: filter probs back to real markers only before HAPPY write (see step 11).
if (step_cM > 0) {
  aug_map <- insert_pseudomarkers(cross$gmap, step = step_cM, stepwidth = 'fixed')
  message('\n[7] running HMM (calc_genoprob) — ', length(cc_cols),
          ' CC strains, pseudomarkers step=', step_cM, ' cM, cores = ', n_cores)
} else {
  aug_map <- cross$gmap
  message('\n[7] running HMM (calc_genoprob) — ', length(cc_cols),
          ' CC strains, genotyped markers only, cores = ', n_cores)
}
# CC cross_info for qtl2 DO HMM: DO::step uses cross_info[0] as n_gen (number of
# outbreeding generations). For CC mice we use n_gen=1 because the CC consensus
# genetic map already encodes the cumulative recombination from the full breeding
# programme (~26 generations). Passing n_gen>1 would double-count those generations
# in the transition probabilities. The funnel (cc_crossinfo.csv) is NOT passed to
# calc_genoprob — the qtl2 DO C++ HMM has no funnel-aware logic; cross_info[0] is
# the only element used, and it must be n_gen, not a founder index.
all_cc <- rownames(cross$geno[[1]])
cross$cross_info <- matrix(1L, nrow = length(all_cc), ncol = 1L,
                           dimnames = list(all_cc, 'ngen'))
message('[crossinfo] set ngen=1 for all ', length(all_cc), ' CC strains (consensus map encodes recombination history)')
probs <- calc_genoprob(cross = cross, map = aug_map,
                       error_prob = 0.002, cores = n_cores)
saveRDS(probs, probs_cc_out)
message('  saved: ', probs_cc_out,
        ' (', if (step_cM > 0) 'with pseudomarkers' else 'real markers only', ')')

# ── step 8: deterministic founder prob arrays ──────────────────────────────────
message('\n[8] building deterministic founder prob arrays...')
make_founder_probs <- function(probs_ref, strain_id, founder_code) {
  state <- paste0(founder_code, founder_code)  # homozygous diplotype e.g. 'AA'
  lapply(probs_ref, function(chr_arr) {
    arr <- array(0, dim = c(1L, dim(chr_arr)[2], dim(chr_arr)[3]),
                 dimnames = list(strain_id, dimnames(chr_arr)[[2]], dimnames(chr_arr)[[3]]))
    arr[1, state, ] <- 1.0
    arr
  })
}
founder_probs_list <- mapply(make_founder_probs,
                              strain_id = founder_cols, founder_code = founder_codes,
                              MoreArgs = list(probs_ref = probs), SIMPLIFY = FALSE)
names(founder_probs_list) <- founder_cols
saveRDS(founder_probs_list, probs_f_out)
message('  saved: ', probs_f_out)

# ── step 9: deterministic F1 prob arrays ──────────────────────────────────────
message('\n[9] building deterministic F1 prob arrays...')
make_f1_probs <- function(probs_ref, strain_id, code1, code2) {
  state <- paste0(sort(c(code1, code2)), collapse = '')  # alphabetical diplotype
  lapply(probs_ref, function(chr_arr) {
    arr <- array(0, dim = c(1L, dim(chr_arr)[2], dim(chr_arr)[3]),
                 dimnames = list(strain_id, dimnames(chr_arr)[[2]], dimnames(chr_arr)[[3]]))
    arr[1, state, ] <- 1.0
    arr
  })
}
f1_probs_list <- mapply(make_f1_probs,
                         strain_id = f1_crosses$sample_id,
                         code1 = f1_crosses$parent_code1, code2 = f1_crosses$parent_code2,
                         MoreArgs = list(probs_ref = probs), SIMPLIFY = FALSE)
names(f1_probs_list) <- f1_crosses$sample_id
saveRDS(f1_probs_list, probs_f1_out)
message('  saved: ', probs_f1_out)

# ── step 10: merge CC + founders + F1s → allele probs → LOCO kinship ──────────
message('\n[10] merging 48 genotypes and computing LOCO kinship...')
probs_combined <- lapply(names(probs), function(chr) {
  abind(probs[[chr]],
        do.call(abind, c(lapply(founder_probs_list, `[[`, chr), list(along = 1))),
        do.call(abind, c(lapply(f1_probs_list,      `[[`, chr), list(along = 1))),
        along = 1)
})
names(probs_combined) <- names(probs)
attributes(probs_combined) <- attributes(probs)
names(probs_combined)      <- names(probs)

probs_allele <- genoprob_to_alleleprob(probs = probs_combined)
kinship_loco <- calc_kinship(probs = probs_allele, type = 'loco')
saveRDS(kinship_loco, kinship_out)
message('  kinship saved: ', kinship_out)
message('  kinship dim (chr1): ', paste(dim(kinship_loco[['1']]), collapse = ' x '))
diag_v <- diag(kinship_loco[['1']])
message('  diagonal — min: ', round(min(diag_v), 3), '  max: ', round(max(diag_v), 3))

# ── step 11: convert to HAPPY cache ───────────────────────────────────────────
# HAPPY cache must contain only genotyped markers. if pseudomarkers were added,
# filter probs_combined back to real markers before writing.
message('\n[11] writing HAPPY genome cache...')
if (step_cM > 0) {
  real_marker_names <- lapply(cross$gmap, names)  # per-chr real marker names from original map
  probs_for_happy <- mapply(function(arr, real_loci) {
    arr[, , intersect(dimnames(arr)[[3]], real_loci), drop = FALSE]
  }, probs_combined, real_marker_names, SIMPLIFY = FALSE)
  names(probs_for_happy) <- names(probs_combined)
  attributes(probs_for_happy) <- attributes(probs_combined)
  message('  filtered to ', sum(lengths(real_marker_names)), ' real markers for HAPPY cache')
} else {
  probs_for_happy <- probs_combined
}
convert.qtl2.to.HAPPY(qtl2.object = probs_for_happy, cross.object = cross,
                       HAPPY.output.path = genomecache_path)
message('  HAPPY cache written: ', genomecache_path)

message('\n✓ build_genomecache.R complete')
