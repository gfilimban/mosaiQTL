# docs/math/12_haplotype_compressed_abf.md
# Haplotype-Compressed Approximate Bayes Factors for CC/DO Fine-Mapping

## Background

The Collaborative Cross (CC) and Diversity Outbred (DO) populations derive from eight
inbred founder strains.  At any genomic position a CC animal carries exactly two of the
eight founder haplotypes, so the genotype of each animal at any SNP is fully determined
by its 8-dimensional haplotype probability vector and the SNP allele pattern across the
eight founders.  This gives rise to an important simplification: at most 2^8 = 256
distinct biallelic allele patterns can exist across all SNPs in any genomic window.  In
practice, within a 2 Mb credible-interval window, 20–100 distinct patterns are typical.

## Standard Single-Causal-Variant ABF (no LD)

Given n_snps association statistics (z-scores z_j with standard errors se_j), the
Wakefield approximate Bayes factor for variant j under the single-causal-variant model is

    log ABF_j = 0.5 × log(V_j / (V_j + W)) + z_j² × W / (2(V_j + W))

where V_j = se_j², W is the prior effect variance (default 0.04 for cis-eQTL on
standardized phenotypes), and z_j = sign(β̂_j) × √(2 ln10 × LOD_j).

The posterior inclusion probability (PIP) under a uniform prior over n_snps variants is

    PIP_j = ABF_j / Σ_{k=1..n} ABF_k

With n_snps in the thousands (a 2 Mb window at ~70M Sanger SNPs genome-wide gives ~2000
SNPs per window), PIPs become small and uniformly distributed even when a true causal
signal is present — the signal is diluted by the dense LD block it sits in.

## LD Structure in CC/DO Populations

In CC/DO data, linkage disequilibrium between SNPs arises entirely from shared founder
haplotypes, not from human-like population LD.  Two SNPs are in perfect LD if and only
if they have identical founder allele patterns (same 8-bit vector of {0,1} values across
the eight founders).  More generally, LD between two SNPs i and j equals the squared
Pearson correlation between their imputed dosage vectors:

    r²(i,j) = Corr(d_i, d_j)²

where d_k[s] = Σ_f H[s,f] × geno_k[f], H[s,f] is the haplotype probability of strain s
for founder f at the locus, and geno_k[f] ∈ {0,1} is the REF/ALT allele for founder f
at SNP k.

The key structural observation: two SNPs with the same 8-bit founder allele pattern will
have identical dosage vectors (up to floating-point precision at the exact locus), giving
r² = 1.  Two SNPs with different patterns may still be highly correlated if their
patterns differ only for founders that are rare in the sample.

## Haplotype LD Groups

A haplotype LD group is a set of SNPs whose founder allele patterns are all mutually
correlated at r² ≥ threshold.  We form groups by:

1. Collecting the set of unique 8-bit founder allele patterns P = {p_1, …, p_K},
   K ≤ 256.  For each pattern p_k, select the highest-LOD SNP carrying that pattern as
   the representative r_k.

2. Computing the K×K pairwise r² matrix between representative dosage vectors.  This
   matrix has size at most 256×256, regardless of how many SNPs are in the window.

3. Applying a union-find merge: representatives i and j are merged into the same group
   when r²(r_i, r_j) ≥ threshold (default 0.8).  The canonical group id is the union-
   find root.

This produces G ≤ K ≤ 256 groups, each containing one or more patterns (and all SNPs
carrying those patterns).

## LD-Aware ABF

Instead of placing a uniform prior over n_snps variants, place it over G LD groups.
The group-level ABF is the maximum ABF among group members:

    ABF_g = max_{j ∈ group g} ABF_j

Group-level PIP:

    PIP_g = ABF_g / Σ_{g'=1..G} ABF_{g'}

Individual SNP PIP: the group's posterior mass is distributed equally among its members:

    PIP_j = PIP_g(j) / |group g(j)|

where g(j) is the group containing j and |group g(j)| is the number of members.

## Credible Set Construction

Grow the credible set group-by-group from highest to lowest group PIP until cumulative
mass ≥ coverage (default 0.95).  All members of each included group enter the credible
set.  Because all members of a group share a correlated haplotype block, the credible set
is guaranteed to have high internal r²: minimum pairwise r² ≥ threshold by construction.

## Why This Matters for CC/DO

1. **Restores interpretable PIPs.**  Dense LD blocks no longer dilute individual PIPs to
   near zero.  A true causal haplotype block receives PIP ≈ 1/G where G is the number of
   independent haplotype signals — typically 1–5 in a 2 Mb cis window.

2. **Compute-frugal.**  The LD matrix is K×K (≤ 256×256), not n_snps×n_snps (≤ 2000×2000
   for a 2 Mb window).  The dominant cost is one haplotype interpolation + dosage
   computation per SNP, which the imputed scan already performs.

3. **Biologically grounded.**  In 8-founder populations, SNPs sharing a founder allele
   pattern are perfectly tagged by that pattern.  The LD group is a natural unit of
   evidence: it represents the set of SNPs that are indistinguishable by the haplotype
   scan.

## Parameters

| Parameter     | Default | Description |
|---|---|---|
| `prior_var`   | 0.04    | Prior effect variance W; corresponds to prior SD ≈ 0.2 for standardized cis-eQTL effect sizes |
| `coverage`    | 0.95    | Credible set coverage level |
| `r2_threshold`| 0.8     | Minimum r² to merge two pattern representatives into the same LD group |

## Implementation

- `qtl/snp_scan.py`: per-SNP scan accumulates `_pat_best_dosage` dict (one representative
  dosage vector per unique founder allele pattern; O(K × n) storage where K ≤ 256).
  Returns `df.attrs["_ld"]` carrying `rep_patterns` and `r2_matrix`.

- `qtl/imputation.py::merge_ld_groups`: union-find merge of representatives; returns
  SNP-index → group-id mapping.

- `qtl/imputation.py::fine_map_region`: dispatches to LD-aware path when `ld_groups` is
  provided.

- `pipeline.py::impute_egenes`: extracts `df.attrs["_ld"]` before `pd.concat` (which
  drops attrs); accumulates per-gene LD data in `ld_by_gene` dict.

- `pipeline.py::finemap_imputed`: calls `merge_ld_groups` and passes `ld_groups` to
  `fine_map_region`.

## Relationship to Multi-Causal Models

The model above assumes at most one causal variant per region (single-causal-variant
assumption).  This is appropriate for the 2 Mb cis window standard in eQTL analysis,
where most cis-eQTL signals are driven by a single regulatory variant or a few in tight
LD.  Multi-causal models (L > 1 components) require additional machinery; the current
implementation does not attempt this.  See `docs/REFERENCES.md` (SNP FINE-MAPPING section)
for references to multi-causal fine-mapping approaches.
