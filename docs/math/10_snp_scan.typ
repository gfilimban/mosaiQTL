#import "_setup.typ": *
#show: mathdoc

= Haplotype-Imputed SNP Scan
<haplotype-imputed-snp-scan>
#strong[Module:] `qtl/snp_scan.py::imputed_snp_scan_h2lmm` \ #strong[miQTL parallel:] `imputed.snp.scan.h2lmm()`

#divider()

== 1. Purpose
<1-purpose>
The haplotype scan identifies which genomic #emph[regions] affect a phenotype and
which #emph[founder haplotypes] carry the effect. The SNP scan goes further: given
known founder genotypes at individual SNPs, it imputes per-sample SNP dosages
from the haplotype probabilities and tests each SNP for association. This connects
haplotype effects to specific sequence variants and enables fine-mapping within
a QTL interval.

#divider()

== 2. Inputs
<2-inputs>
`imputed_snp_scan_h2lmm` takes:

- `phenotype` --- continuous phenotype #mi(`(n,)`);, RINT-transformed internally (`use_rint=True`).
  When `sample_ids` is given, RINT is applied at sample level and phenotype and
  covariates are then averaged to strain means in cache order; otherwise
  #mi(`n`); must equal the number of cache strains.
- `genomecache` --- `GenomeCache` with haplotype probs at strain level.
- `snp_genos` --- `pd.DataFrame` of shape `(n_snps, n_founders)`, values #mi(`\in \{0,1\}`); or
  #mi(`\in \{0,1,2\}`); (alt allele count per founder; NaN = no call), with `chr` and `pos_mb` columns.
  The columns must match `genomecache.founders` by name.

The `snp_genos` DataFrame is the Python-native interface. Within the package it is
built from the founder allele HDF5 written by `genotype/snp_db.py`
(`build_founder_allele_gm` from a founder genotype-array CSV,
`build_founder_allele_mgp` from a Sanger Mouse Genomes Project VCF). The
production path (`pipeline.py::impute_egene`) codes alleles with
`snp_db.make_alt_mask` (1 = call differs from the REF founder `B6J`; missing calls
coded 0); `imputed_snp_scan_from_db` instead takes the alphabetically later
nucleotide as alt and codes missing calls as NaN. The flat-file helper
`read_allele_dir` (HAPPY-format `.RData` files) still exists, but it sets
`pos_mb` to NaN, so the scan skips every SNP it returns.

#divider()

== 3. SNP dosage imputation
<3-snp-dosage-imputation>
For SNP #mi(`s`); at position #mi(`p_s`); on chromosome #mi(`c`);, the imputed dosage for sample #mi(`i`); is:

#mitex(`d_{is} = \sum_{f=1}^{J} H_{if\ell(s)} \cdot g_{fs}`)

where:

- #mi(`H_{if}(p_s)`); --- haplotype probability for strain #mi(`i`);, founder #mi(`f`);, at the
  SNP position, interpolated linearly in physical position (Mb) between the two typed loci that
  bracket #mi(`p_s`); and clamped at the ends of the chromosome:
  #mi(`H_{if}(p_s) = (1-t)\,H_{if\ell} + t\,H_{if,\ell+1}`);, #mi(`t = (p_s - p_\ell)/(p_{\ell+1} - p_\ell)`);.
  A convex combination of probability vectors is again a probability vector.
- #mi(`g_{fs} \in \{0,1,2\}`); --- founder #mi(`f`);\'s alt allele count at SNP #mi(`s`);.

Genotype is a property of the strain, so the dosage is formed once per strain and copied to
every sample of that strain; the likelihood below is fit over samples. For a fully inbred CC
strain where #mi(`H_i \approx e_{f^*}`);, the dosage recovers #mi(`g_{f^*s}`); directly. For
strains with uncertain ancestry it is a probability-weighted interpolation across founders.
Interpolation is in physical position, not genetic position; its effect against a
genetic-map interpolation has not been measured.

If some founders have no call (NaN), the SNP is skipped when any sample carries
haplotype mass #mi(`> 0.05`); (`MAX_MISSING_HAP_MASS`) on those founders; otherwise
the dosage is renormalised over the called founders #mi(`F_s`);:

#mitex(`d_{is} = \frac{\sum_{f \in F_s} H_{if\ell(s)}\, g_{fs}}{\sum_{f \in F_s} H_{if\ell(s)}}`)

Monomorphic SNPs (#mi(`\text{sd}(d) < 10^{-8}`);) are skipped.

#strong[Minimum minor-allele evidence.] Let #mi(`S`); be the number of strains present among
the scanned samples and #mi(`T = \sum_{s} d_s`); the summed strain dosage. The quantity

#mitex(`m = \min(T,\; S - T)`)

is the expected number of independent genomes carrying the rarer allele (`minor_strains`).
It counts strains because testability is set by the kinship and genetic architecture:
replicate samples of one strain share a genotype and, through #mi(`Z K Z^\top`);, a fully
correlated genetic effect, so they add information about the residual variance but not about
the allele. A SNP carried by one strain with many replicates has a large sample count and one
independent genotype, and its effect is largely absorbed by the random effect. A SNP is tested
only when #mi(`m \ge 3`); (`MIN_MINOR_STRAINS`).

The reported `af`, `maf`, `mac` and `ma_samples` are descriptive and are computed over the
scanned samples. Everything inside the likelihood is at sample level; only this filter counts
strains.

The F-statistic at a SNP uses the gene\'s null #mi(`h^2`); and is exact only asymptotically, and
with very few carriers a single strain can determine the fit. The value 3 is a conventional
minimum-count rule, not a threshold derived for this design. In a null simulation on the real
chromosome-19 kinship and the real microglial sample structure (300 replicates, SNPs
stratified by #mi(`m`);), the type I error showed no trend with #mi(`m`); and a mild inflation
(#mi(`\lambda_{GC} \approx 1.1`);) in every bin, whose cause is not identified. The cache holds
no SNP with #mi(`m < 1.5`);, so the simulation does not test the threshold itself; with up to 48
strains the filter removes only the rarest SNPs present.

#divider()

== 4. Scan model
<4-scan-model>
The standardized dosage #mi(`\tilde{d}_s = (d_s - \bar{d}_s)/\text{sd}(d_s)`); is
used as a scalar predictor in the LMM:

#mitex(`y_i = \beta_d \tilde{d}_{is} + x_i^\top\alpha + u_i + \varepsilon_i,
\qquad u \sim \mathcal{N}(0,\sigma^2_g K^{(-c)})`)

This is `fit_lmm_at_locus` with a #mi(`1`);-column haplotype matrix (#mi(`J=1`);), giving
1 effective df. The model is fit at sample level with kinship #mi(`Z K^{(-c)} Z^\top`); over the
samples that have a phenotype, so the SNP and haplotype likelihoods are nested: same samples,
same rank-inverse-normal transform, same kinship and the same ML #mi(`h^2`); for the gene.
#mi(`h^2`); is estimated once per chromosome from the null model by ML
(`lmm.estimate_h2_null`) and held fixed across SNPs. SNPs are processed in chunks: the
standardised dosages of a chunk are rotated by the eigenvectors once, and each SNP is then
fit against the shared rotated phenotype and covariates.

#strong[Power comparison to haplotype scan:] the haplotype scan has #mi(`J-1 = 7`); df and
detects any founder-haplotype association regardless of SNP allele coding. The
SNP scan has 1 df and is more powerful when the causal variant is biallelic and
the founder partition aligns with a single alt allele. Multi-allelic loci with
complex founder allele patterns are better detected by the haplotype scan.

#divider()

== 5. The r = 0.55 discrepancy with miQTL
<5-the-r--055-discrepancy-with-miqtl>
mosaiQTL uses #strong[marginal haplotype probabilities]:

#mitex(`H_{if\ell} = \sum_{f'} P(f \times f' \mid i, \ell)`)

miQTL uses the full #strong[diplotype probabilities] #mi(`P(f \times f' \mid i, \ell)`);
to model heterozygosity in dosage imputation:

#mitex(`d_i^{\text{miQTL}} = \sum_{f \leq f'} \bigl[P(f \times f) \cdot g_f + P(f \times f') \cdot \tfrac{g_f + g_{f'}}{2}\bigr]`)

#strong[For inbred CC samples] the difference is negligible: heterozygous diplotype
probabilities #mi(`P(f \neq f') \approx 0`);, so the marginal approximation
#mi(`H_{if\ell} \approx P(f \times f \mid i, \ell)`); is exact.

#strong[For outbred DO samples] the difference is material: each animal carries two
distinct founder haplotypes per locus. The marginal conflates homozygous and
heterozygous diplotypes, compressing dosage variation and reducing
correlation with the true SNP genotype.

The documented #mi(`r = 0.55`); between mosaiQTL and miQTL SNP LOD profiles is
explained by this marginal-vs-diplotype model difference on CC data where
residual heterozygosity is present in some strains. The marginal approximation
is adequate for fully inbred CC; diplotype-aware dosage imputation is needed
for DO. This is a known, documented limitation --- not a bug.

#divider()

== 6. Scope notes
<6-scope-notes>
- This scan is #strong[LMM-only] in the current implementation. An NB-GLMM version
  would replace `fit_lmm_at_locus` with the NB score test at #mi(`J=1`); (scalar dosage
  predictor), using raw counts and the fixed dispersion #mi(`\hat\phi_g`); from Stage 1.
- The scan requires founder allele calls (formerly the gated roadmap item P2-S8);
  they now come from the HDF5 built by `genotype/snp_db.py`. In production the
  Nextflow `FINEMAP_CHR` process runs `workflow/scripts/run_finemap.py` →
  `pipeline.py::finemap_egenes`, scanning the SNPs inside each eGene's 1.5-LOD
  support interval, which in L mode is computed inside the scan window of #mi(`\pm`);`window_mb` around the
  strand-aware TSS and so is never wider than it (`06_permutation.typ` §6). With
  `params.sanger_alleles` unset (the default, `null`), header-only outputs are written.
