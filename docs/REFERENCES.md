# docs/REFERENCES.md — mosaiQTL Annotated Bibliography
# Canonical citation source for all references in CLAUDE.md, GOALS.md, and ARCHITECTURE.md.
# Format: citation | DOI | why it matters to mosaiQTL
# Never auto-loaded by Claude Code. @-import only when writing derivations.

---

## QTL MAPPING — MULTIPARENT POPULATIONS

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Broman KW et al. R/qtl2: Software for Mapping Quantitative Trait Loci with High-Dimensional Data. *Genetics* 2019 | 10.1534/genetics.119.302286 | Primary R validation reference; qtl2 LOD formula defines our LOD denominator=n rule |
| Gatti DM et al. Quantitative Trait Locus Mapping Methods for Diversity Outbred Mice. *G3* 2014 | 10.1534/g3.114.013748 | miQTL conceptual basis; haplotype probability scan design |
| Churchill GA & Doerge RW. Empirical threshold values for quantitative trait mapping. *Genetics* 1994 | 10.1093/genetics/138.3.963 | Permutation threshold theory — defines permute-expression-not-genotype rule |
| Collaborative Cross Consortium. The genome architecture of the Collaborative Cross mouse genetic reference population. *Genetics* 2012 | 10.1534/genetics.113.153452 | CC population structure — why n_gen=1 is correct for CC |
| Morgan AP et al. The Mouse Universal Genotyping Array. *G3* 2016 | 10.1534/g3.116.037622 | GigaMUGA array — our genotyping platform |
| Listgarten J et al. Improved linear mixed models for genome-wide association studies. *Nat Methods* 2012 | 10.1038/nmeth.2037 | LOCO kinship — proximal contamination rationale; mandatory LOCO rule |

---

## NB-GLMM MODEL

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| He Y et al. NEBULA: a fast negative binomial mixed model for differential or co-expression analysis of large-scale scRNA-seq data. *Commun Biol* 2021 | 10.1038/s42003-021-02146-6 | Stage 1 NEBULA-LN design; NB-gamma mixed model with per-subject RE |
| Chen H et al. Control for population structure and relatedness for binary traits in genetic association studies via logistic mixed models. *AJHG* 2016 | 10.1016/j.ajhg.2016.02.012 | PQL score test for GLMM — Stage 2 score stat derivation and approximation validity |
| Robinson MD & Smyth GK. Small-sample estimation of negative binomial dispersion, with applications to SAGE data. *Biostatistics* 2008 | 10.1093/biostatistics/kxm030 | Empirical Bayes dispersion shrinkage (WLEB) — Stage 1 shrink_sc_disp |
| Robinson MD et al. edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics* 2010 | 10.1093/bioinformatics/btp616 | TMM normalisation; tagwise dispersion — Stage 1 edgepython path |
| Pachter L et al. edgePython: Python NB GLM engine. *GitHub* 2026 | github.com/pachterlab/edgePython (GPL-3.0) | Legacy Stage 1 path (use_native=False); glm_sc_fit, shrink_sc_disp, glm_fit. Native NEBULA-LN implementation (use_native=True) follows its design. |
| Ahlmann-Eltze C & Huber W. glmGamPoi: fitting gamma-Poisson GLMs. *Bioinformatics* 2021 | 10.1093/bioinformatics/btaa1009 | Fast NB fitting at single-cell scale; score test reference |
| Yazar S et al. Single-cell eQTL mapping identifies cell type–specific genetic control. *Science* 2022 | 10.1126/science.abf3041 | Large-scale cell-type eQTL with NB pseudobulk — scope justification |
| Sarkar A & Stephens M. Separating measurement and expression models. *Nat Genet* 2021 | 10.1038/s41588-021-00873-4 | NB measurement model theory — why raw counts not normalised values |

---

## LINEAR ALGEBRA & COMPUTATION

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Kang HM et al. Efficient control of population structure in model organism association mapping. *Genetics* 2008 | 10.1534/genetics.107.080101 | Eigendecomp LMM (lmm.py validated port); EMMA algorithm |
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

## DESIGN PRINCIPLES

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Bray NL et al. Near-optimal probabilistic RNA-seq quantification (kallisto). *Nat Biotechnol* 2016 | 10.1038/nbt.3519 | Algorithmic innovation over hardware; math-first design |
| Melsted P et al. Modular, efficient and constant-memory preprocessing (bustools). *Nat Biotechnol* 2021 | 10.1038/s41587-021-00870-2 | Constant-memory streaming; minimal deps; DESIGN-01 basis |
| Pimentel H et al. Differential analysis of RNA-seq with sleuth. *Nat Methods* 2017 | 10.1038/nmeth.4324 | LRT on count models; math-first documentation |

---

## HPC & WORKFLOW

| Reference | DOI | Role in mosaiQTL |
|---|---|---|
| Di Tommaso P et al. Nextflow enables reproducible computational workflows. *Nat Biotechnol* 2017 | 10.1038/nbt.3820 | Primary Nextflow citation; HPC pipeline design |
| Ewels PA et al. The nf-core framework. *Nat Biotechnol* 2020 | 10.1038/s41587-020-0439-x | DSL2 patterns; module structure; stub-run CI |
| Kurtzer GM et al. Singularity: scientific containers for mobility of compute. *PLOS ONE* 2017 | 10.1371/journal.pone.0177459 | Singularity/Apptainer — HPC container runtime |
| Amdahl GM. Validity of the single-processor approach. *AFIPS* 1967 | 10.1145/1465482.1465560 | Amdahl's law — ~5.5× ceiling with serial fraction 0.15 |
| Gustafson JL. Reevaluating Amdahl's law. *Commun ACM* 1988 | 10.1145/42411.42415 | Weak scaling — R²>0.95 target (1→8 cores) |

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
