# docs/REFERENCES.md — mosaiQTL Annotated Bibliography
# Canonical citation source for all references in CLAUDE.md, GOALS.md, and ARCHITECTURE.md.
# Format: citation | DOI/URL | why it matters to mosaiQTL
# Never auto-loaded by Claude Code. @-import only when writing derivations.

---

## DATA SOURCES — LAYER-0 PUBLIC INPUTS

| Resource | URL | Contents |
|---|---|---|
| JAX Diversity Outbred Reference Data | https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data | **Primary** source for all GigaMUGA reference files: `gm_uwisc_v4.csv` (marker positions on GRCm39 + Cox genetic map), `GigaMUGA_genotypes.csv.gz` (280 reference samples, raw per-sample ACGT/H/N calls — founder and F1 consensus is computed from this file + `GigaMUGA_founder_metadata.csv`, not from a separate pre-consensused file), `GigaMUGA_founder_metadata.csv` (sample → strain/letter mapping), `CC_consensus_genotypes.csv` (76 CC strain consensus calls), `cc_sample_metadata.csv` (funnel codes + Mt/Y origin). |
| Sanger Mouse Genomes Project REL2021 | https://www.sanger.ac.uk/data/mouse-genomes-project/ | **Canonical Sanger MGP SNP reference**: `mgp_REL2021_snps_founders.vcf.gz` — built directly against GRCm39 (`bcftools mpileup -f Mus_musculus.GRCm39.dna.toplevel.fa.gz`, VEP-annotated Ensembl 104, high-confidence `FI=1` sites only, subset to the 7 non-reference CC/DO founders; C57BL/6J is absent because it *is* the GRCm39 reference and its genotype is implicitly REF/REF everywhere). No liftover is needed — the file is mm39-native from build. 33,511,156 SNPs genome-wide. |

See `docs/DATA_SOURCES.md` for the complete derivation chain from these files to `GM_ccf1fo_mm39.csv`.

---

## QTL MAPPING — MULTIPARENT POPULATIONS

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Keele GR et al. Determinants of QTL mapping power in the realized Collaborative Cross. *G3* 2019 | 10.1534/genetics.116.197434 | miQTL — the R package mosaiQTL ports and validates against |
| Broman KW et al. R/qtl2: Software for Mapping Quantitative Trait Loci with High-Dimensional Data. *Genetics* 2019 | 10.1534/genetics.118.301595 | Primary R validation reference; qtl2 LOD formula defines our LOD denominator=n rule; qtl2's RIL, X-chromosome and Diversity Outbred transition kernels are the implementation reference for the HMM |
| Gatti DM et al. Quantitative Trait Locus Mapping Methods for Diversity Outbred Mice. *G3* 2014 | 10.1534/g3.114.013748 | miQTL conceptual basis; haplotype probability scan design |
| Churchill GA & Doerge RW. Empirical threshold values for quantitative trait mapping. *Genetics* 1994 | 10.1093/genetics/138.3.963 | Permutation threshold theory — defines permute-expression-not-genotype rule |
| Joo JWJ et al. Multiple testing correction in linear mixed models. *Genome Biol* 2016 | 10.1186/s13059-016-0903-6 | Permutation is the reference standard for LMM thresholds; kinship/heritability dependence of the null max-LOD — context for the kinship-structured parametric null |
| Ongen H et al. Fast and efficient QTL mapper for thousands of molecular phenotypes. *Bioinformatics* 2016 | 10.1093/bioinformatics/btv722 | Models the per-feature permutation null with a Beta distribution fit to a few permutations and reads adjusted p-values from it — the per-gene Beta calibration |
| Dudbridge F, Gusnanto A. Estimation of significance thresholds for genomewide association scans. *Genet Epidemiol* 2008 | 10.1002/gepi.20297 | Fits a Beta distribution to the minimum p-value across permutation replicates |
| Collaborative Cross Consortium. The genome architecture of the Collaborative Cross mouse genetic reference population. *Genetics* 2012 **190**:389–401 | 10.1534/genetics.111.132639 | CC population structure and breeding scheme; real-data reference for founder-haplotype inheritance and linkage-disequilibrium decay in the CC |
| Srivastava A et al. Genomes of the Mouse Collaborative Cross. *Genetics* 2017 **206**:537–556 | 10.1534/genetics.116.198838 | Haplotype reconstructions of the CC strains from sequence; real-data check on breakpoint density |
| Broman KW. The genomes of recombinant inbred lines. *Genetics* 2005 **169**:1133–1146 | 10.1534/genetics.104.035212 | Two-locus haplotype structure and map expansion of eight-way sibling-mated RILs — the fixed-line transition kernel in `genotype/hmm.py` |
| Teuscher F, Broman KW. Haplotype probabilities for multiple-strain recombinant inbred lines. *Genetics* 2007 **175**:1267–1274 | 10.1534/genetics.106.064063 | Simpler derivation of the two- and three-point haplotype probabilities of multiple-strain RILs; extension to additional generations of intermating |
| Broman KW. Genotype probabilities at intermediate generations in the construction of recombinant inbred lines. *Genetics* 2012 **190**:403–412 | 10.1534/genetics.111.132647 | Two-locus probabilities, fixation probability and map expansion at each generation on the path to inbreeding (numerical prescription); basis for pre-CC and partly inbred extensions |
| Morgan AP et al. The Mouse Universal Genotyping Array. *G3* 2016 | 10.1534/g3.116.037622 | GigaMUGA array — our genotyping platform |
| Listgarten J et al. Improved linear mixed models for genome-wide association studies. *Nat Methods* 2012 | 10.1038/nmeth.2037 | LOCO kinship — proximal contamination rationale; mandatory LOCO rule |

---

## NB-GLMM MODEL

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| He Y et al. NEBULA: a fast negative binomial mixed model for differential or co-expression analysis of large-scale scRNA-seq data. *Commun Biol* 2021 | 10.1038/s42003-021-02146-6 | Stage 1 NEBULA-LN design; NB-gamma mixed model with per-subject RE |
| Breslow NE & Clayton DG. Approximate inference in generalized linear mixed models. *JASA* 1993 | 10.1080/01621459.1993.10476387 | Foundational PQL paper — working response z and working weights W construction; justification that PQL is necessary because the NB-Gaussian integral has no closed form |
| Chen H et al. Control for population structure and relatedness for binary traits in genetic association studies via logistic mixed models. *AJHG* 2016 | 10.1016/j.ajhg.2016.02.012 | PQL score test for GLMM — Stage 2 score stat derivation and approximation validity |
| Robinson MD & Smyth GK. Small-sample estimation of negative binomial dispersion, with applications to SAGE data. *Biostatistics* 2008 | 10.1093/biostatistics/kxm030 | Empirical Bayes dispersion shrinkage (WLEB) — Stage 1 shrink_sc_disp |
| Robinson MD et al. edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics* 2010 | 10.1093/bioinformatics/btp616 | TMM normalisation; tagwise dispersion — Stage 1 edgepython path |
| Pachter L et al. edgePython: Python NB GLM engine. *GitHub* 2026 | github.com/pachterlab/edgePython (GPL-3.0) | Primary design reference for Stage 1 NB fitting: NB log-likelihood, IRLS weights, dispersion estimation, and score test construction. glm_sc_fit, shrink_sc_disp, glm_fit map directly to our Stage 1 path. edgeR (Robinson 2010) is the conceptual ancestor; edgePython is the direct implementation we are informed by. |
| Ahlmann-Eltze C & Huber W. glmGamPoi: fitting gamma-Poisson GLMs. *Bioinformatics* 2021 | 10.1093/bioinformatics/btaa1009 | Fast NB fitting at single-cell scale; score test reference |
| Yazar S et al. Single-cell eQTL mapping identifies cell type–specific genetic control. *Science* 2022 | 10.1126/science.abf3041 | Large-scale cell-type eQTL with NB pseudobulk — scope justification |
| Sarkar A & Stephens M. Separating measurement and expression models. *Nat Genet* 2021 | 10.1038/s41588-021-00873-4 | NB measurement model theory — why raw counts not normalised values |

---

## LINEAR ALGEBRA & COMPUTATION

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Kang HM et al. Efficient control of population structure in model organism association mapping. *Genetics* 2008 | 10.1534/genetics.107.080101 | Eigendecomp LMM (lmm.py validated port); EMMA algorithm. Its Methods searches the variance ratio $\delta$ from $10^{-5}$ to $10^{5}$ in 100 log-spaced regions and takes the best of the stationary points and endpoints (the reference R code's defaults are `ngrids=100, llim=-10, ulim=10`, a different range from the paper's text); supports a search symmetric in $\log\delta$ that reaches both $h^2\approx 0$ and $h^2\approx 1$ |
| Lippert C et al. FaST linear mixed models for genome-wide association studies. *Nat Methods* 2011 | 10.1038/nmeth.1681 | Spectral decomp scan — lmm.py design; explains why ops stay at n×n for Gaussian |
| Taylor DL et al. Factored spectrally normalized phosphorylation (tensorQTL). *Genome Biol* 2019 | 10.1186/s13059-019-1836-7 | GPU-accelerated QTL scan — SPEED-03 comparison baseline |
| Halko N et al. Finding structure with randomness: probabilistic algorithms for matrix decomposition. *SIAM Rev* 2011 | 10.1137/090771806 | Randomised SVD — kinship PC computation strategy |

---

## SINGLE-CELL & PSEUDOBULK

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Wolf FA et al. SCANPY: large-scale single-cell gene expression data analysis. *Genome Biol* 2018 | 10.1186/s13059-017-1382-0 | Upstream QC and clustering (out of scope but dependency) |
| Squair JW et al. Confronting false discoveries in single-cell differential expression. *Nat Commun* 2021 | 10.1038/s41467-021-25960-2 | Pseudobulk rationale — why sample not cell is the unit |

---

## BIOINFORMATICS SOFTWARE ENGINEERING

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Wilson G et al. Good enough practices in scientific computing. *PLOS Comput Biol* 2017 | 10.1371/journal.pcbi.1005510 | SWE goals thresholds (SWE-01–05); 90% coverage standard |
| Taschuk M & Wilson G. Ten simple rules for making research software more robust. *PLOS Comput Biol* 2017 | 10.1371/journal.pcbi.1005412 | Robustness rules; fail-fast, test coverage, provenance |
| Peng RD. Reproducible research in computational science. *Science* 2011 | 10.1126/science.1213847 | DESIGN-07 deterministic results; reproducibility standard |
| Wilkinson MD et al. The FAIR Guiding Principles. *Sci Data* 2016 | 10.1038/sdata.2016.18 | FAIR4RS — output format and provenance requirements |
| Grüning B et al. Bioconda: sustainable and comprehensive software distribution. *Nat Methods* 2018 | 10.1038/s41592-018-0046-7 | Environment reproducibility; Bioconda for R validation deps |

---

## FDR / MULTIPLE TESTING

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Storey JD. A direct approach to false discovery rates. *JRSS-B* 2002 | 10.1111/1467-9868.00346 | Storey q-value: π₀ estimation via spline smoother; q(i)=π₀×m×p(i)/rank(p(i)); replaces BH placeholder (SWE-09) |
| Storey JD & Tibshirani R. Statistical significance for genome-wide studies. *PNAS* 2003 | 10.1073/pnas.1530509100 | R `qvalue` package — reference implementation used as ground-truth fixture in test_fdr.py |
| Bourgon R, Gentleman R, Huber W. Independent filtering increases detection power for high-throughput experiments. *PNAS* 2010 | 10.1073/pnas.0914005107 | Pre-filtering before FDR is valid only when the filter is independent of the test statistic under the null; a LOD-threshold gate is not, so every tested gene enters calibration and Storey, and any pre-filter uses a genotype-independent criterion |
| Lam A et al. Rapid and robust association mapping of expression quantitative trait loci. *BMC Proc* 2007 | 10.1186/1753-6561-1-s1-s144 | Removing non-expressed genes by an expression-variability filter roughly halves the number of tests in eQTL mapping — the phenotype-only (genotype-independent) pre-filter class |
| Benjamini Y & Hochberg Y. Controlling the false discovery rate. *JRSS-B* 1995 | 10.1111/j.2517-6161.1995.tb02031.x | BH-FDR (π₀=1 case of Storey; never less strict than the Storey q-value); `qval_bh`, selectable as the call rule via `call_rule='bh'` |

---

## VARIANCE-COMPONENT BOUNDARY, FDR NULL CALIBRATION AND MULTI-FAMILY TESTING

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Self SG & Liang KY. Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *JASA* 1987;82:605–610 | 10.1080/01621459.1987.10478472 | The ML estimate of a variance component whose true value is on the boundary $h^2=0$ sits at the boundary with probability about 1/2 (a $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ limit), so a mass of genes at the $h^2$ floor is expected and is not an estimator fault; what is a fault is a search range that cannot represent an interior optimum below the grid start (docs/math/03 §2) |
| Stram DO & Lee JW. Variance components testing in the longitudinal mixed effects model. *Biometrics* 1994;50:1171–1177 (correction: *Biometrics* 1995;51:1196) | DOI not verified | Applies Self & Liang to linear mixed models: variance-component hypotheses on the boundary give chi-bar-square mixtures, so the $h^2$ estimate needs a search that includes the boundary |
| Zhou X & Stephens M. Genome-wide efficient mixed-model analysis for association studies. *Nat Genet* 2012;44:821–824 | 10.1038/ng.2310 | States that FaST-LMM maximises the likelihood over the variance ratio with Brent's algorithm (super-linear) and that GEMMA uses Newton–Raphson (quadratic) on the same spectral-decomposition likelihood; supports a one-dimensional bracketed search on the rotated likelihood. The paper's own grid and range are in its Online Methods, which were not checked |
| Storey JD, Taylor JE & Siegmund D. Strong control, conservative point estimation and simultaneous conservative consistency of false discovery rates: a unified approach. *JRSS-B* 2004;66:187–205 | 10.1111/j.1467-9868.2004.00439.x | Conditions (weak dependence among tests) under which the $\hat\pi_0$-scaled q-value controls FDR conservatively; $\hat\pi_0$ is conservative only when null p-values are uniform |
| Efron B. Large-scale simultaneous hypothesis testing: the choice of a null hypothesis. *JASA* 2004;99:96–104 | 10.1198/016214504000000089 | A mis-specified theoretical null (here: an over-dispersed or over-corrected calibration null) distorts the null fraction; the null distribution must be checked against the data before $\hat\pi_0$ is interpreted |
| Benjamini Y & Yekutieli D. The control of the false discovery rate in multiple testing under dependency. *Ann Stat* 2001;29:1165–1188 | 10.1214/aos/1013699998 | BH controls FDR under positive regression dependence (PRDS); gives the dependence guarantee for the BH call rule across correlated genes |
| Phipson B & Smyth GK. Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Stat Appl Genet Mol Biol* 2010;9:Article 39 | 10.2202/1544-6115.1585 | A permutation p-value must be at least $1/(B+1)$ (observed statistic counted among the permutations); a floor on `pval_perm` follows from the number of permutations |
| Benjamini Y & Bogomolov M. Selective inference on multiple families of hypotheses. *JRSS-B* 2014;76:297–318 | 10.1111/rssb.12028 | When each cell type or tissue is a family and families are selected on the data, FDR must be controlled on the selected families; the basis for a two-level (cell type, then gene) error rate |
| Peterson CB, Bogomolov M, Benjamini Y & Sabatti C. TreeQTL: hierarchical error control for eQTL findings. *Bioinformatics* 2016;32:2556–2558 | 10.1093/bioinformatics/btw198 | Hierarchical testing of eQTL hypotheses grouped by gene (and, in later versions, by tissue); the first TreeQTL release is single-tissue, so a tissue level is an extension rather than a given |
| Urbut SM, Wang G, Carbonetto P & Stephens M. Flexible statistical methods for estimating and testing effects in genomic studies with multiple conditions. *Nat Genet* 2019;51:187–195 | 10.1038/s41588-018-0268-8 | `mash`: shares information on effect sizes across conditions (44 GTEx tissues) with arbitrary correlation; an estimation-and-sharing approach to cross-cell-type effects, separate from error-rate control |

---

## SNP FINE-MAPPING

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Keele GR et al. Determinants of QTL mapping power in the realized Collaborative Cross. *G3* 2019 | 10.1534/g3.119.400194 | `imputed.snp.scan.h2lmm()` — miQTL reference for PORT-08 haplotype-imputed SNP scan |
| Broman KW et al. R/qtl2: Software for Mapping QTL with High-Dimensional Data. *Genetics* 2019 | 10.1534/genetics.118.301595 | `scan1snps()` + `index_snps()` — direct-join SNP↔locus design used in snp_db.py; `find_peaks()` API design |
| Churchill GA et al. The Collaborative Cross, a community resource for genetic analysis. *Nat Genet* 2004 | 10.1038/ng1104-1133 | Founder probability weighting formula: d[strain]=Σ_f H[strain,f]×geno[f]; theoretical basis of PORT-08 imputation |
| Wakefield J. Bayes factors for genome-wide association studies: comparison with P-values. *Genet Epidemiol* 2009 **33**:79–86 | 10.1002/gepi.20359 | `qtl/imputation.py::compute_log_abf` — approximate Bayes factor from per-SNP (β̂, SE); the prior variance W is a user-chosen input. Direction used here is association over null (see Giambartolomei 2014), reported as the natural log |
| Wen X. Bayesian model comparison in genetic association analysis: linear mixed modeling and SNP set testing. *Biostatistics* 2015 **16**:701–712 | 10.1093/biostatistics/kxv009 | Approximate Bayes factors for linear-mixed-model association from Wald-type summary statistics; basis for feeding per-SNP LMM (β̂, SE) into `compute_log_abf` |
| Giambartolomei C et al. Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. *PLoS Genet* 2014 **10**:e1004383 | 10.1371/journal.pgen.1004383 | States the reciprocal (association-over-null) direction as used in fine-mapping: support for the model in which the variant is associated, relative to the null of no association, with shrinkage factor $r = W/(V+W)$; `log_abf` follows this convention |
| Guan Y & Stephens M. Practical issues in imputation-based association mapping. *PLoS Genet* 2008 **4**:e1000279 | 10.1371/journal.pgen.1000279 | Bayesian association with imputed genotypes; posterior-mean dosage as the regressor for imputed SNPs, the setting of the Sanger dosage scan |
| Wang G et al. A simple new approach to variable selection in regression, with application to genetic fine-mapping. *JRSS-B* 2020 | 10.1111/rssb.12388 | SuSiE: iterative Bayesian stepwise selection for multi-causal fine-mapping; credible set purity concept (min pairwise r² ≥ 0.5); motivates haplotype-compressed LD-group design in `qtl/imputation.py` |
| Zou Y et al. Fine-mapping from summary data with the "Sum of Single Effects" model. *PLOS Genet* 2022 | 10.1371/journal.pgen.1010299 | SuSiE-RSS: extends SuSiE to summary statistics via z-scores; LD matrix input; long-term target for `qtl/imputation.py` when optional `rss` dependency is available |
| Maller JB et al. (Wellcome Trust Case Control Consortium). Bayesian refinement of association signals for 14 loci in 3 common diseases. *Nat Genet* 2012 **44**:1294–1301 | 10.1038/ng.2435 | Credible sets as the smallest set of SNPs with cumulative posterior probability ≥ 95% *within a pre-specified region*; the PIPs and the coverage are conditional on that region, so the fine-mapping window must be fixed in advance and must not exceed the region the gene was tested in |
| Schaid DJ, Chen W & Larson NB. From genome-wide associations to candidate causal variants by statistical fine-mapping. *Nat Rev Genet* 2018 **19**:491–504 | 10.1038/s41576-018-0016-z | Review of fine-mapping approaches; the dependence of posterior inclusion probabilities and credible sets on the region analysed and on linkage disequilibrium within it |
| Dupuis J & Siegmund D. Statistical methods for mapping quantitative trait loci from a dense set of markers. *Genetics* 1999 **151**:373–386 | 10.1093/genetics/151.1.373 | A LOD support region is approximately a confidence region for QTL location; its width scales with the sample size, not its square root. Basis of the 1.5-LOD interval in `qtl/ci.py` |
| Manichaikul A, Dupuis J, Sen Ś & Broman KW. Poor performance of bootstrap confidence intervals for the location of a quantitative trait locus. *Genetics* 2006 **174**:481–489 | 10.1534/genetics.106.061549 | Likelihood support intervals and approximate Bayes credible intervals have appropriate coverage; the nonparametric bootstrap does not. Why `qtl/ci.py` uses the support interval |
| Lander ES & Botstein D. Mapping mendelian factors underlying quantitative traits using RFLP linkage maps. *Genetics* 1989 **121**:185–199 | 10.1093/genetics/121.1.185 | Interval mapping and the LOD score as the QTL test statistic; origin of the LOD profile from which the support interval is read |
| Veyrieras JB et al. High-resolution mapping of expression-QTLs yields insight into human gene regulation. *PLoS Genet* 2008 **4**:e1000214 | 10.1371/journal.pgen.1000214 | cis-eQTL positions are distributed relative to the transcription start and end sites (enrichment near the TSS; about 5% of eQTLs more than 20 kb upstream of the TSS). Grounds anchoring one window on the strand-aware TSS |
| GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* 2020 **369**:1318–1330 | 10.1126/science.aaz1776 | Reference cis-eQTL design in which a gene's variants are tested in a fixed window around its TSS. The ±1 Mb window is stated in the paper's Methods, not in the abstract; confirm there before quoting it |
| Wen X, Lee Y, Luca F & Pique-Regi R. Efficient integrative multi-SNP association analysis via deterministic approximation of posteriors. *Am J Hum Genet* 2016 **98**:1114–1129 | 10.1016/j.ajhg.2016.03.029 | Multi-SNP fine-mapping for molecular QTL data (cis-eQTL in GEUVADIS and GTEx), performed within each gene's cis region |
| Abdel-Azim G, Patel P, Li S, Guo S & Black MH. Fast multiple-trait genome-wide association analysis for correlated longitudinal measurements. *Sci Rep* 2023 **13**:20603 | 10.1038/s41598-023-47555-1 | Simulation: with a minimum minor allele count of 3 within genotype-by-time categories, observed type I error was roughly equal to the theoretical genome-wide level. Human longitudinal multi-trait mixed model, not a multiparent panel; supports a count-3 convention for `MIN_MINOR_STRAINS`, indirectly |
| Sitlani CM, Baldassari AR, Highland HM, Hodonsky CJ, McKnight B & Avery CL. Comparison of adaptive multiple phenotype association tests using summary statistics in genome-wide association studies. *Hum Mol Genet* 2021 **30**:1371–1383 | 10.1093/hmg/ddab126 | Asymptotic multi-phenotype tests lose type I error control when minor allele count is low and phenotype covariance is near singular; motivates a minimum-count filter. Multi-phenotype summary-statistic tests in humans, so indirect support only |
| Holmes JB et al. Revealing the secrets of sequence variation with SPDI. *Bioinformatics* 2020 | 10.1093/bioinformatics/btz856 | SPDI format for unambiguous variant representation: `{RefSeq_accession}:{pos_0based}:{ref}:{alt}`; used as the `variant` column in EQTL_FINEMAPPING_COLS; GRCm39 RefSeq accessions from NCBI assembly GCF_000001635.27 |

---

## DESIGN PRINCIPLES

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Bray NL et al. Near-optimal probabilistic RNA-seq quantification (kallisto). *Nat Biotechnol* 2016 | 10.1038/nbt.3519 | Algorithmic innovation over hardware; math-first design |
| Melsted P et al. Modular, efficient and constant-memory preprocessing (bustools). *Nat Biotechnol* 2021 | 10.1038/s41587-021-00870-2 | Constant-memory streaming; minimal deps; DESIGN-01 basis |
| Research Data Alliance FAIR4RS Working Group. FAIR Principles for Research Software. *RDA* 2022 | 10.15497/RDA00068 | Reproducibility, provenance, and open-science standards for mosaiQTL outputs |
| Pimentel H et al. Differential analysis of RNA-seq with sleuth. *Nat Methods* 2017 | 10.1038/nmeth.4324 | LRT on count models; math-first documentation |

---

## HPC & WORKFLOW

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Di Tommaso P et al. Nextflow enables reproducible computational workflows. *Nat Biotechnol* 2017 | 10.1038/nbt.3820 | Primary Nextflow citation; HPC pipeline design |
| Ewels PA et al. The nf-core framework. *Nat Biotechnol* 2020 | 10.1038/s41587-020-0439-x | DSL2 patterns; module structure; stub-run CI |
| Kurtzer GM et al. Singularity: scientific containers for mobility of compute. *PLOS ONE* 2017 | 10.1371/journal.pone.0177459 | Singularity/Apptainer — HPC container runtime |
| Amdahl GM. Validity of the single-processor approach. *AFIPS* 1967 | 10.1145/1465482.1465560 | Amdahl's law — ~5.5× ceiling with serial fraction 0.15 |
| Gustafson JL. Reevaluating Amdahl's law. *Commun ACM* 1988 | 10.1145/42411.42415 | Weak scaling — $R^2 > 0.95$ target (1→8 cores) |

---

## PYTHON PACKAGING & QUALITY

| Reference | URL | Role in mosaiQTL |
|---|---|---|
| PEP 484 — Type hints | peps.python.org/pep-0484/ | type-annotate ALL public signatures rule |
| PEP 517 — Build system | peps.python.org/pep-0517/ | pyproject.toml editable install correctness |
| PEP 518 — Build deps | peps.python.org/pep-0518/ | build-time dependency isolation |
| PEP 526 — Variable annotations | peps.python.org/pep-0526/ | variable-level type safety |
| PEP 257 — Docstrings | peps.python.org/pep-0257/ | interrogate enforces compliance; Google style |
| PEP 621 — pyproject metadata | peps.python.org/pep-0621/ | pyproject.toml project metadata format |
| PyPA Packaging User Guide | packaging.python.org | src/ layout, extras, entry points |

---

## PROMPT INJECTION SECURITY

| Reference | DOI/URL | Role in mosaiQTL |
|---|---|---|
| Greshake K et al. Not what you signed up for: indirect prompt injection. *arXiv* 2023 | 10.48550/arXiv.2302.12173 | .claudeignore rationale; untrusted file risk |
| Perez F & Ribeiro I. Ignore previous prompt. *arXiv* 2022 | 10.48550/arXiv.2211.09527 | Direct injection taxonomy |
| OWASP Top 10 for LLM Applications (LLM01: Prompt Injection) | owasp.org/www-project-top-10-for-large-language-model-applications | --allowedTools; tool surface minimisation |
| NIST AI Risk Management Framework 1.0 | nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf | Federal AI security standard |
