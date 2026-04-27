# GOALS.md — mosaiQTL
# Single source of truth for all project goals.
# Auto-updated by: python goals/update_goals.py --benchmark benchmarks/results/latest.json
# Repo: github.com/gfilimban/mosaiqtl
# Regressions auto-open GitHub Issue (label: regression). Achievements auto-close linked Issue.

## STATUS KEY
🔴 not started · 🟡 in progress · 🟢 achieved · ⚠️ needs rerun · ❌ failing

---

## PRIORITY
The priority is replicating the minimal miQTL example workflow to validate that the
coding logic base is being imported into Python accurately. Expansions stem their
ground truth from here.

Validation is structured in three phases:
  Phase 1 — minimal miQTL workflow (port_sim_validation.R mirror)
  Phase 2 — full miQTL function port + replicate-awareness (minimal structure like Phase 1)
  Phase 3 — NB vs LMM comparison (nb.py correctness vs lmm.py validated port)

---

## VALIDATION — Phase 1 (minimal miQTL workflow mirror)
Mirrors port_sim_validation.R exactly. Ground truth for all subsequent validation.
Script: validation/validate_sim.py
Data: sparcc_cache + phenotypes.csv (from port_sim_validation.R export)

| ID | Section | R function | Threshold | Current | Status |
|---|---|---|---|---|---|
| P1-S1 | ROP scan LOD r² | scan.h2lmm (ROP) | ≥ 0.990 | 0.9975 | 🟢 |
| P1-S2 | MI scan LOD r | scan.h2lmm (MI, num.imp=11) | ≥ 0.75 | 0.9756 | 🟢 |
| P1-S3 | GEV threshold delta | get.gev.thresholds (95th pct) | < 0.5 LOD | 0.1543 | 🟢 |

Run: `python validation/validate_sim.py`
Pass = all three gates green simultaneously.

---

## VALIDATION — Phase 2 (R vs Python, S1–S11)
## VALIDATION — Phase 2 simulated (S1–S11, sparc data, independent of gastroc)
Scripts: validation/r_reference/port_sim_validation_p2.R → validation/validate_sim_p2.py
Data: sparcc_cache + sim.CC.data(seed=42) — independent from Phase 1 sim
Uses real LOCO kinship (Phase 1 used identity/OLS shortcut).

| ID | Section | Threshold | Current | Status |
|---|---|---|---|---|
| P2-S1 | Cache structure | loci delta=0, founders match, samples match | - | 🔴 not yet run |
| P2-S2 | Haplotype argmax concordance | ≥ 0.95 | - | 🔴 not yet run |
| P2-S3 | LOCO kinship r (worst chr) | > 0.999 | - | 🔴 not yet run |
| P2-S4 | LMM scan pass rate (10 sims) | ≥ 50% at r² ≥ 0.990 | - | 🔴 not yet run |
| P2-S5 | Cache collapse | ≥5% removed AND lod_r² ≥ 0.999 | - | 🔴 not yet run |
| P2-S6 | GEV threshold delta | < 0.5 LOD | - | 🔴 not yet run |
| P2-S7 | MI scan LOD r | ≥ 0.75 | - | 🔴 not yet run |
| P2-S8 | SNP scan | gated: allele_dir required | ⚠️ GATED | ⚠️ gated |
| P2-S9 | Mediation LOD r | ≥ 0.90 | - | 🔴 not yet run |
| P2-S10 | Bootstrap CI contains planted locus | True | - | 🔴 not yet run |
| P2-S11 | Instability LOD r | ≥ 0.90 | - | 🔴 not yet run |

```bash
Rscript validation/r_reference/port_sim_validation_p2.R
python validation/validate_sim_p2.py
python validation/validate_sim_p2.py --sections 3,4,6   # selective
python validation/validate_sim_p2.py --force             # force rerun
```

---

## VALIDATION — Phase 3 (NB vs LMM, biological consistency)
| ID | Gate | Threshold | Current | Status |
|---|---|---|---|---|
| BIO-P1 | Piezo1 chr8 LOD | > 3 | - | 🔴 |
| BIO-P2 | scan_nb LOD distribution skew | > 0.5 | - | 🔴 |
| BIO-P3 | scan_vqtl returns result | >= 1 | - | 🔴 |
| BIO-P4 | scan_epistasis lod_additive | > 0 | - | 🔴 |
| BIO-P5 | scan_conditional signal1_lod | > 0 | - | 🔴 |
| BIO-P6 | Ccdc171 chr4 in top results | present | - | 🔴 |
| BIO-P7 | Pmis2 chr7 in top results | present | - | 🔴 |

---

- this portion needs revising 

## BIOLOGICAL ACCURACY (simulation-based)
Requires sim_cc48 fixture with planted eQTL signal. Run: `python -m mosaiqtl.utils.simulate`.
| ID | Description | Metric | Threshold | Current | Dataset |
|---|---|---|---|---|---|
| BIO-A1 | eQTL recall (local, known QTL) | recall | >= 0.80 | — | sim_cc48_n200qtl |
| BIO-A2 | Type I error at permutation threshold | FPR | <= 0.05 | — | sim_cc48_null |
| BIO-A3 | Founder effect accuracy | r vs true effects | >= 0.90 | — | sim_cc48_n200qtl |
| BIO-A4 | LOD concordance vs R/qtl2 | r² | >= 0.999 | — | chr8_50genes_subset |
| BIO-A5 | FDR control at q < 0.05 | observed FDR | <= 0.05 | — | sim_cc48_n200qtl |

---

## SPEED
Baseline must be committed to benchmarks/baseline/ before marking any goal current.
| ID | Description | Baseline | Target | Current | Delta | Last tested |
|---|---|---|---|---|---|---|
| SPEED-01 | Genome-wide scan (n=48, 500 genes, all chr) | miQTL R: TBD | < 4h | — | — | — |
| SPEED-02 | Speedup vs R/miQTL (CC48, 1 cell type) | miQTL: ~4h est | >= 5× | — | — | — |
| SPEED-03 | Speedup vs tensorQTL (CC48, 1 cell type) | tensorQTL: TBD | >= 2× | — | — | — |
| SPEED-04 | Speedup vs limix (CC48, 1 cell type) | limix: TBD | >= 3× | — | — | — |
| SPEED-05 | Permutation throughput (1000 perms, 1 CT) | miQTL: TBD | <= 4h | — | — | — |
| SPEED-06 | scan_chromosome genes/min | TBD | > 100/min | — | — | — |

Amdahl ceiling: ~5.5× (serial fraction f≈0.15, n_chr=20). Gustafson weak-scaling target: R² > 0.95 (1→8 cores).
Run comparison: `python -m mosaiqtl.bench.compare_tools --tools mosaiqtl miqtl tensorqtl limix`

---

## MEMORY
HDF5 B-tree genome cache (Python) vs R HAPPY format (per-locus .RData files, sequential I/O):
- R: loads one .RData per locus on demand — filesystem stat + deserialise per call, O(1) but high overhead
- Python: HDF5 B-tree with chunked reads + np.memmap — O(log L) seek, random access, stays out of RAM
- This difference is the primary memory advantage of the Python implementation

| ID | Description | Target | Current | Assessment |
|---|---|---|---|---|
| MEM-01 | Full run peak RAM (n=48, 500 genes, all chr) | < 32 GB | — | memory_profiler full run |
| MEM-02 | scan_chromosome peak RAM per worker | < 4 GB | — | tracemalloc per-gene loop |
| MEM-03 | Genome cache HDF5 RSS (74k loci, n=418) | < 500 MB RSS (memmap) | — | tracemalloc on cache load |
| MEM-04 | Kinship LOCO set in RAM (20 chr × 418 × 418) | < 600 MB | — | sys.getsizeof after build |
| MEM-05 | Peak LOD array (n_genes × n_loci per chr) | never materialised | architecture constraint | code review |

MEM-05 is a hard architectural constraint: the full LOD matrix must never be materialised.
Write per-chr, stream results, merge on completion. Violation = design failure.

---

## SOFTWARE ENGINEERING
All CI checks run on every PR. Thresholds from Wilson et al. 2017 (10.1371/journal.pcbi.1005510)
and Taschuk & Wilson 2017 (10.1371/journal.pcbi.1005412).
| ID | Description | Threshold | Current | CI check |
|---|---|---|---|---|
| SWE-01 | Test coverage | >= 90% | — | pytest-cov --cov-fail-under=90 + codecov upload |
| SWE-02 | mypy --strict errors | 0 | — | mypy --strict src/mosaiqtl/ |
| SWE-03 | flake8 violations | 0 | 0 🟢 | flake8 src/ tests/ |
| SWE-04 | Public API docstring coverage | 100% | 100% 🟢 | interrogate --fail-under 100 |
| SWE-05 | Bandit critical security issues | 0 | — | bandit -r src/ -ll |
| SWE-06 | src/ layout migration complete | done | 🟢 | manual |
| SWE-07 | CI pipeline active on main | green | 🔴 | .github/workflows/ci.yml (GitHub Actions docs: docs.github.com/actions) |

---

## LMM EXTENSIONS (post-validation)
| ID | Description | Prerequisite | Status |
|---|---|---|---|
| LMM-EXT-01 | Replicate-aware LMM: scan.py reduces replicate samples to strain means via Z; h² and scan operate at strain level (m×m). Implemented in scan_h2lmm and scan_h2lmm_mi. Replicate covariates (sex/batch per replicate) averaged into strain-mean covariates automatically. | — | 🟢 |
| LMM-EXT-02 | Update test_lmm.py to use sample-level fixtures (tiny_phenotype, tiny_covariates, kinship_samples, z_matrix) | LMM-EXT-01 | 🔴 |

---

## DESIGN GOALS (pachterlab-inspired)
From: Melsted et al. 2021 (10.1038/s41587-021-00870-2), Bray et al. 2016 (10.1038/nbt.3519).
Reproducibility standard: Peng RD (2011) Reproducible research in computational science. Science. 10.1126/science.1213847
| ID | Description | Verification |
|---|---|---|
| DESIGN-01 | Core install requires only numpy+pandas+scipy+anndata+h5py | pip install mosaiqtl on clean env; no optional extras |
| DESIGN-02 | Each CLI subcommand independently runnable and testable | pytest per subcommand in isolation |
| DESIGN-08 | Composable CLI: mosaiqtl fit-null/scan/permute/effects/report | no hidden state between stages; each reads/writes files |
| DESIGN-09 | Numba JIT for IRLS hot path (if profiling shows need) | profile naive numpy first; JIT only if bottleneck confirmed |
| DESIGN-03 | Every algorithm has math derivation in docs/wiki/03_Algorithms/ before code | file exists check in CI |
| DESIGN-04 | Benchmark results committed to repo with timestamps | benchmarks/baseline/ in git |
| DESIGN-05 | Derivation → pseudocode → implementation order enforced in PRs | PR template checklist item |
| DESIGN-06 | ProvenanceRecord (git hash + seed + params) on every scan output | test_provenance_header in io tests |
| DESIGN-07 | All results deterministic given same seed | pytest -p no:randomly; fixed seed in all scans |

---

## HPC / NEXTFLOW
| ID | Goal | Status |
|---|---|---|
| HPC-01 | Nextflow stub-run passes (DAG validation) | 🔴 |
| HPC-02 | SLURM profile tested on UCI CRSP | 🔴 |
| HPC-03 | Singularity/Apptainer container builds | 🔴 |
| HPC-04 | Linear scaling R² (1→8 SLURM nodes) > 0.95 | 🔴 |
| HPC-05 | Peak scratch disk per job < 50 GB | 🔴 |

---

## BENCHMARKING VS TOOLS
| Tool | Model | Dataset | LOD r² | Speedup | Memory | Status |
|---|---|---|---|---|---|---|
| miQTL (R) | haplotype LMM | chr8, 50 genes | — | not yet timed | not measured | 🔴 |
| R/qtl2 | haplotype LMM | chr8, 50 genes | — | — | — | 🔴 |
| tensorQTL | linear QTL (GPU) | sim_cc48 | n/a (no haplotype) | — | — | 🔴 |
| limix | LMM flexible | sim_cc48 | n/a (no haplotype) | — | — | 🔴 |

Note: tensorQTL and limix do not use haplotype probabilities — LOD r² comparison is not applicable.
Speed and memory comparison is the meaningful metric for those tools.

---

## VISUALIZATION ROADMAP
| ID | Plot | Module | Priority | Status |
|---|---|---|---|---|
| VIZ-01 | LOD profile, Manhattan, lod/bayes interval, region zoom, SNP Manhattan | viz/genome.py | HIGH | 🟢 implemented |
| VIZ-02 | Founder coef profile, effect bars, caterpillar+CI at locus | viz/effects.py | HIGH | 🟢 implemented |
| VIZ-03 | Prob heatmap, pxg, kinship heatmap | viz/haplotypes.py | HIGH | 🟢 implemented |
| VIZ-04 | NB: φ distribution, library size QC, NB vs LMM LOD scatter | viz/genome.py + effects.py | HIGH | 🟢 implemented |
| VIZ-05 | Scan summary: LOD histogram, gene-level Manhattan, cis-distance histogram | viz/genome.py | HIGH | 🟢 implemented |
| VIZ-06 | Cell-type LOD heatmap (genes × cell types) | viz/celltypes.py | MED | 🔴 |
| VIZ-07 | Cell-type specificity: one gene, all cell types LOD profiles overlaid | viz/celltypes.py | MED | 🔴 |
| VIZ-08 | Epistasis J×J founder interaction heatmap | viz/haplotypes.py | LOW | 🟢 implemented |
| VIZ-09 | Mediation proportion bar per candidate mediator | viz/effects.py | LOW | 🔴 |

---

## MULTIOMICS ROADMAP
| Modality | Module | Priority | Status |
|---|---|---|---|
| snRNA-seq eQTL | qtl/nb.py | — | 🟢 implemented |
| Permutation correction | qtl/permutation.py | HIGH | 🟡 S6 GEV unstable |
| SNP fine-mapping | qtl/snp_scan.py | HIGH | 🟡 r=0.55 (model diff documented) |
| ATAC-seq caQTL | qtl/atac.py | MED | 🔴 |
| sQTL | qtl/sqtl.py | MED | 🔴 |
| RNA-mod QTL (m6A) | qtl/rnamod.py | MED | 🔴 |
| Sex GxE | qtl/gxe.py | MED | 🔴 |
| Trans-eQTL | qtl/trans_scan.py | MED | 🔴 |
| ASE | qtl/ase.py | LOW | 🟡 scaffold |

## MIQTL PORTING GAPS
Functions in the miQTL R codebase not yet ported to Python. Assessed against
`port_sim_validation.R` (README example workflow) and full R/ inventory.

### Already ported (confirmed)
| miQTL R function | Python location | Notes |
|---|---|---|
| `scan.h2lmm()` ROP additive | `qtl/scan.py::scan_h2lmm` | |
| `scan.h2lmm(use.multi.impute=TRUE)` | `qtl/scan.py::scan_h2lmm_mi` | uses `multi.imput.lmmbygls` internally |
| `imputed.snp.scan.h2lmm()` | `qtl/snp_scan.py::imputed_snp_scan_h2lmm` | marginal model only |
| `generate.sample.outcomes.matrix()` | `qtl/permutation.py::generate_null_outcomes` | |
| `run.threshold.scans()` | `qtl/permutation.py::run_threshold_scans` | |
| `get.gev.thresholds()` | `qtl/permutation.py::get_gev_thresholds` | |
| `run.positional.scans()` | `qtl/permutation.py::run_positional_scans` | |
| `grab.locus.from.scan()` | `qtl/scan.py::grab_locus_from_scan` | |
| `calc.LOD.drop()` | `qtl/scan.py::lod_drop_ci` | |
| `instability.lm.scan()` | `qtl/scan.py::instability_lm_scan` | |
| `scan.qr()` / fixef eQTL | `qtl/scan.py::fixef_eqtl_scan` | |
| `mediation.scan.qr()` | `qtl/mediation.py` | |
| `calc.kinship.from.genomecache.with.DOQTL()` | `genotype/kinship.py::calc_kinship` | |
| `DiploprobReader` / `happy.*()` | `genotype/genomecache.py::GenomeCache` | |
| `get.allele.effects.from.fixef()` + `get.allele.effects.from.ranef()` | `qtl/effects.py::get_allele_effects` | fixef (pinv) + ranef (ridge/BLUP) methods |
| `genome.plotter.whole()` | `viz/genome.py::lod_profile` | |
| `allele.plotter.whole()` | `viz/effects.py::coef_profile` | |
| `prob.heatmap.*()` | `viz/haplotypes.py::prob_heatmap` | |
| `rint()` | `utils.py::rint` | |

### Already ported — HIGH priority (added)
| ID | miQTL R function(s) | Python location | Notes |
|---|---|---|---|
| PORT-01 | `collapse.genomecache()` | `genotype/genomecache.py::collapse_genomecache` | 🟢 ported — greedy L2 forward scan |
| PORT-02 | `scan.h2lmm(model='full')` | `qtl/scan.py::scan_h2lmm` + `qtl/lmm.py::scan_chromosome` | 🟢 ported — outer-product diplotype probs; valid for CC inbred data |
| PORT-03 | `get.allele.effects.from.ranef()` + `get.allele.effects.from.fixef()` | `qtl/effects.py::get_allele_effects` | 🟢 ported — FWL gram + pseudoinverse (fixef) / ridge BLUP (ranef) |

### Not yet ported — HIGH priority
| ID | miQTL R function(s) | Target Python location | Why needed |

### Not yet ported — MEDIUM priority
| ID | miQTL R function(s) | Target Python location | Why needed |
|---|---|---|---|
| PORT-04 | `snp.null.par.bs.threshold.scan()` `snp.par.perm.threshold.scan()` `snp.perm.threshold.scan()` | `qtl/snp_scan.py` or `qtl/permutation.py` | SNP scan needs own thresholds; haplotype thresholds are anti-conservative for SNP scan |
| PORT-05 | `pairwise.cor.snp.scan()` `extract.r2.interval()` | `qtl/snp_scan.py` | LD/r² between imputed SNPs; needed for credible-set fine-mapping |
| PORT-06 | `straineff.mapping.matrix()` | `utils.py::straineff_mapping_matrix` | T×M diplotype→allele mapping used by MI and SNP scan; must verify Python uses correct formula |
| PORT-07 | `lmmbygls.replicates()` | `qtl/lmm.py` | replicate-aware LMM wrapper → LMM-EXT-01 path |

### Not porting (low value or superseded)
| miQTL R function | Reason |
|---|---|
| `lmmbylmer.R` | eigendecomp adequate for CC/DO n≤418; lmer adds sparse-RE overhead |
| `scan.EM()` | EM algorithm scan — different numerical approach, not in scope |
| `scan.h2lmm.deBLUP()` | deBLUP adjustment — specialized, not in roadmap |
| `calc.SIM.cache()` / `get.SIM()` | shared IBD — useful but not blocking any validation |
| `null.pvalue.ci.plot()` | covered by `viz/genome.py::pvalue_qq` |

---

## VQTL DESIGN FIX (VQTL-01)
Current `vqtl.py` Stage 2 incorrectly calls `scan_h2lmm` (Gaussian LMM) on log(φ̂).
Correct design: Stage 2 is PQL NB-GLMM score test for dispersion effects, mirroring
`scan_nb_gene` but testing H₀: φ constant across haplotypes vs H₁: φ varies by founder.
| ID | Task | Priority | Status |
|---|---|---|---|
| VQTL-01 | Replace scan_h2lmm call in scan_vqtl with NB-GLMM dispersion score test | HIGH | 🔴 |
| VQTL-02 | Fix subject granularity: estimate_dispersion_per_subject uses sample-level IDs (individual mice via PARSE barcodes), not strain-level — aggregation to strain before scan needs explicit logic | MED | 🔴 |
| VQTL-03 | Fix precompute_sufficient_stats: currently missing from nb.py (added as stub); validate against NB sufficient stats derivation. Blocks estimate_dispersion_per_subject — TestEstimateDispersion skipped until resolved. | MED | 🔴 |

---

## REGRESSION LOG
| Date | Goal | Was | Now | Commit | Resolution |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

---

## MILESTONE MAP
| Milestone | Goals | Notes |
|---|---|---|
| v0.1 Foundation | SWE-01–07, DESIGN-01–07, BIO-A4 | structure, tests, CI, derivations |
| v0.2 Validation | VAL-S1–S11, BIO-P1–P7, BIO-A1–A5 | all validation passing |
| v0.3 Speed+Mem | SPEED-01–06, MEM-01–05, HPC-01–05 | HPC pipeline, benchmarking |
| v0.4 Multiomics | first new modality (caQTL or sQTL) | ATAC or splicing extension |
