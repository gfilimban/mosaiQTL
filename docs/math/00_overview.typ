#import "_setup.typ": *
#show: mathdoc

= mosaiQTL --- Mathematical Overview
<mosaiqtl--mathematical-overview>
This directory contains the derivations for every statistical method in mosaiQTL,
ordered by data flow. Each file maps directly to one or more source modules.

#divider()

== Reading order
<reading-order>
#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([File], [Covers], [Module(s)],),
    table.hline(),
    [`01_haplotypes.typ`], [36-state diplotype HMM (forward-backward); additive haplotype tensor; HAPPY vs HDF5; pseudomarkers], [`genotype/hmm.py`, `genotype/genomecache.py`],
    [`02_kinship.typ`], [Haplotype dot-product K (not van Raden), mean-diagonal scaled; LOCO; Z-matrix expansion; eigendecomp caching], [`genotype/kinship.py`, `utils.py`],
    [`03_lmm_scan.typ`], [ML #mi(`h^2`); estimation (grid + bounded Brent); LMM scan; LOD formula; FWL vectorization], [`qtl/lmm.py`],
    [`04_scan_modes.typ`], [Additive (ROP) vs full diplotype model; MI scan; NB-GLMM (experimental)], [`qtl/scan.py`, `qtl/lmm.py`, `qtl/mi.py`, `qtl/nb.py`],
    [`05_allele_effects.typ`], [fixef (pseudoinverse) vs ranef (ridge-BLUP) at peak locus], [`qtl/effects.py`],
    [`06_permutation.typ`], [Permutation / GEV thresholds (QC only); fix\_h2; 1.5-LOD support interval], [`qtl/permutation.py`, `qtl/thresholds.py`, `qtl/ci.py`],
    [`07_instability.typ`], [Bootstrap instability scan; mean-LOD interpretation], [`qtl/instability.py`],
    [`08_nb_glmm.typ`], [NB Stage 1 (NEBULA-LN); EB dispersion shrinkage; PQL Stage 2; Woodbury (experimental; not wired into Nextflow or the CLI)], [`qtl/nb.py`],
    [`09_extensions.typ`], [Mediation LOD-drop; Khatri-Rao epistasis; BLUP conditioning; vQTL], [`qtl/mediation.py`, `qtl/epistasis.py`, `qtl/vqtl.py`],
    [`10_snp_scan.typ`], [Haplotype-imputed SNP dosage; marginal model; #mi(`r = 0.55`); gap], [`qtl/snp_scan.py`],
    [`11_storey_qvalue.typ`], [Per-gene beta-calibrated parametric-null p-values; Storey q-values over all tested genes], [`qtl/fdr.py`],
    [`12_haplotype_compressed_abf.typ`], [Haplotype-compressed Wakefield ABFs; PIPs; credible sets for fine-mapping], [`qtl/imputation.py`],
  )]
  , kind: table
  )

#divider()

== Notation used throughout
<notation-used-throughout>
#figure(
  align(center)[#table(
    columns: 2,
    align: (auto,auto,),
    table.header([Symbol], [Meaning],),
    table.hline(),
    [#mi(`n`);], [number of phenotype observations (samples / replicates)],
    [#mi(`m`);], [number of unique genotyped strains (#mi(`m \leq n`);)],
    [#mi(`J`);], [number of founders (#mi(`J = 8`); for CC/DO)],
    [#mi(`L`);], [number of loci on a chromosome],
    [#mi(`p`);], [number of fixed-effect covariates (including intercept)],
    [#mi(`H_\ell \in \mathbb{R}^{m \times J}`);], [haplotype probability matrix at locus #mi(`\ell`);],
    [#mi(`K \in \mathbb{R}^{m \times m}`);], [realized kinship matrix],
    [#mi(`K^{(-c)}`);], [LOCO kinship (chromosome #mi(`c`); excluded)],
    [#mi(`Z \in \{0,1\}^{n \times m}`);], [replicate-to-strain mapping matrix],
    [#mi(`h^2`);], [narrow-sense heritability: #mi(`\sigma_g^2 / (\sigma_g^2 + \sigma_e^2)`);],
    [#mi(`\text{LOD}`);], [#mi(`\log_{10}`); likelihood ratio; Gaussian LMM (ML): #mi(`\frac{n}{2}\log_{10}(\text{RSS}_0/\text{RSS}_1)`); (#mi(`= Q / (2\ln 10)`); for score tests)],
  )]
  , kind: table
  )

Two design spaces coexist in mosaiQTL:

- #strong[Strain space] (#mi(`m \times m`);): genome cache, kinship and its stored LOCO eigendecomp;
  MI and SNP scans (phenotype reduced to strain means via #mi(`Z`);)
- #strong[Sample space] (#mi(`n \times n`);): raw phenotype and covariates, Z-matrix, expanded
  kinship #mi(`Z K^{(-c)} Z^\top`); and its eigendecomp, #mi(`h^2`); estimation and the haplotype LMM
  scans (including the production `scan_multi_gene_lmm`)

When phenotype sample ids differ from the cache strain ids, the haplotype LMM scans expand
the kinship to sample space and re-eigendecompose it once per chromosome
(`qtl/scan.py`, see `02_kinship.typ` Section 3); when they match, #mi(`Z`); is the identity and
both spaces coincide.
