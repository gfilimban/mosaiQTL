# GOALS.md — mosaiQTL
# Single source of truth for all project goals.
# Auto-updated by: python goals/update_goals.py --benchmark benchmarks/results/latest.json
# Repo: github.com/gfilimban/mosaiqtl
# Regressions auto-open GitHub Issue (label: regression). Achievements auto-close linked Issue.

## STATUS KEY
🔴 not started · 🟡 in progress · 🟢 achieved · ⚠️ needs rerun · ❌ failing

---

## VALIDATION — Phase 2 (R vs Python, S1–S11)
### For reference: from original validation plan using gastroc data
| ID | Section | Threshold | Current | Status |
|---|---|---|---|---|
| VAL-S1 | Cache info | loci_max_delta expected | - | 🔴 |
| VAL-S2 | Haplotype probs r | > 0.90 | - | 🔴 |
| VAL-S3 | Kinship LOCO r | > 0.999 | - | 🔴 |
| VAL-S4 | LMM LOD pass rate | >= 50% | - | 🔴 |
| VAL-S5 | LOD preservation | > 0.9990 | - | 🔴 |
| VAL-S6 | Permutation mean delta | < 0.50 | - | 🔴 |
| VAL-S7 | MI scan LOD r | > 0.75 | - | 🔴 |
| VAL-S9 | Mediation LOD r | > 0.90 | - | 🔴 |
| VAL-S10 | CI overlap | True | - | 🔴 |
| VAL-S11 | Instability r | > 0.90 | - | 🔴 |

- Threshold column values needs rechecking


## VALIDATION — Phase 2 (R vs Python, S1–S11)
### For reference: from original validation plan using gastroc data
| ID | Section | Threshold | Current | Status |
|---|---|---|---|---|
| EXVAL-S1 | Cache info | loci_max_delta expected | 1237 (correct) | 🟢 |
| EXVAL-S2 | Haplotype probs r | > 0.90 | pending R rerun | ⚠️ |
| EXVAL-S3 | Kinship LOCO r | > 0.999 | 0.9998 | 🟢 |
| EXVAL-S4 | LMM LOD pass rate | >= 50% | 60.2% | 🟢 |
| EXVAL-S5 | LOD preservation | > 0.9990 | 0.9999 | 🟢 |
| EXVAL-S6 | Permutation mean delta | < 0.50 | 0.619 | ❌ |
| EXVAL-S7 | MI scan LOD r | > 0.75 | 0.8191 | 🟢 |
| EXVAL-S9 | Mediation LOD r | > 0.90 | 0.9983 | 🟢 |
| EXVAL-S10 | CI overlap | True | True | 🟢 |
| EXVAL-S11 | Instability r | > 0.90 | pending rerun | ⚠️ |

Genuine passes: 7/11. Open: S6 GEV instability (undiagnosed) | S2 needs R rerun | S11 needs rerun.
```bash
Rscript validation/r_reference/port_validation.R all Inf 2
python validation/validate_port.py --sections 2,11
```


---

## VALIDATION — Phase 3 (biological consistency, P1–P4)
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
| SWE-03 | flake8 violations | 0 | — | flake8 src/ tests/ |
| SWE-04 | Public API docstring coverage | 100% | — | interrogate --fail-under 100 |
| SWE-05 | Bandit critical security issues | 0 | — | bandit -r src/ -ll |
| SWE-06 | src/ layout migration complete | done | 🟢 | manual |
| SWE-07 | CI pipeline active on main | green | 🔴 | .github/workflows/ci.yml (GitHub Actions docs: docs.github.com/actions) |

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
| miQTL (R) | haplotype LMM | chr8, 50 genes | 0.973 median (measured) | not yet timed | not measured | 🟡 |
| R/qtl2 | haplotype LMM | chr8, 50 genes | — | — | — | 🔴 |
| tensorQTL | linear QTL (GPU) | sim_cc48 | n/a (no haplotype) | — | — | 🔴 |
| limix | LMM flexible | sim_cc48 | n/a (no haplotype) | — | — | 🔴 |

Note: tensorQTL and limix do not use haplotype probabilities — LOD r² comparison is not applicable.
Speed and memory comparison is the meaningful metric for those tools.

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

---

## REGRESSION LOG
| Date | Goal | Was | Now | Commit | Resolution |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

---

## SESSION PROGRESS (2026-04-19)
| Task | Status | Commit |
|---|---|---|
| TASK 0a — sparc NB fixtures | 🟢 done | 517096aa |
| TASK 1 — genotype/founders.py + viz/validation.py | 🟢 done | 055fc49b |
| TASK 2 — models/records.py (ScanResult, VQTLResult, FounderEffect, QTLResult, ProvenanceRecord) | 🟢 done | 0586acd8 |
| TASK 3 — io/covariates.py | 🟡 in progress | — |

---

## MILESTONE MAP
| Milestone | Goals | Notes |
|---|---|---|
| v0.1 Foundation | SWE-01–07, DESIGN-01–07, BIO-A4 | structure, tests, CI, derivations |
| v0.2 Validation | VAL-S1–S11, BIO-P1–P7, BIO-A1–A5 | all validation passing |
| v0.3 Speed+Mem | SPEED-01–06, MEM-01–05, HPC-01–05 | HPC pipeline, benchmarking |
| v0.4 Multiomics | first new modality (caQTL or sQTL) | ATAC or splicing extension |
