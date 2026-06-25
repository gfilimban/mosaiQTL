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
  Phase 4 — NB vs LMM comparison (nb.py correctness vs lmm.py validated port)

---

## VALIDATION — Phase 1 (minimal miQTL workflow mirror)
Mirrors port_sim_validation.R exactly. Ground truth for all subsequent validation.
Script: validation/validate_sim.py
Data: sparcc_cache + phenotypes.csv (from port_sim_validation.R export)

| ID | Section | R function | Threshold | Current | Status |
|---|---|---|---|---|---|
| P1-S1 | ROP scan LOD r² | scan.h2lmm (ROP) | ≥ 0.990 | 0.9975 | 🟢 |
| P1-S2 | MI scan LOD r | scan.h2lmm (MI, num.imp=11) | ≥ 0.75 | 0.9718 | 🟢 |
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
| P2-S1 | Cache structure | loci delta=0, founders match, samples match | loci delta=0 | 🟢 |
| P2-S2 | Haplotype argmax concordance | ≥ 0.95 | ≥ 0.95 | 🟢 |
| P2-S3 | LOCO kinship r (worst chr) | > 0.999 | 0.9992 (chrX) | 🟢 |
| P2-S4 | LMM scan pass rate (10 sims) | ≥ 50% at r² ≥ 0.975 | 8/10 | 🟢 |
| P2-S5 | Cache collapse | ≥5% removed AND lod_r² ≥ 0.999 | - | ⚠️ N/A — sparcc_cache is pre-collapsed; gate requires an uncollapsed cache |
| P2-S6 | GEV threshold delta | < 0.5 LOD | 0.203 | 🟢 |
| P2-S7 | MI scan LOD r | ≥ 0.75 | PASS | 🟢 |
| P2-S8 | SNP scan | gated: founder_alleles.h5 required | PASS (PORT-08 ported; h5 built in phase2/phase3) | 🟢 |
| P2-S9 | Mediation LOD r | ≥ 0.90 | PASS | 🟢 |
| P2-S10 | Bootstrap CI contains planted locus | True | True (187.46–188.20 Mb ∋ 188.03) | 🟢 |
| P2-S11 | Instability peak_freq | ≥ 0.70 | PASS | 🟢 |

```bash
Rscript validation/r_reference/port_sim_validation_p2.R
python validation/validate_sim_p2.py
python validation/validate_sim_p2.py --sections 3,4,6   # selective
python validation/validate_sim_p2.py --force             # force rerun
```

---

## VALIDATION — Phase 4 (NB vs LMM, biological consistency)
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
| BIO-A5 | FDR control at q < 0.05 using Storey q-value (SWE-09) — BH-FDR placeholder must be replaced before this gate is meaningful | observed FDR | <= 0.05 | — | sim_cc48_n200qtl |

---

## SPEED
Baseline must be committed to benchmarks/baseline/ before marking any goal current.
| ID | Description | Baseline | Target | Current | Delta | Last tested |
|---|---|---|---|---|---|---|
| SPEED-01 | Genome-wide scan (n=415, m=48, 43k genes, 20 chr) | miQTL (HPC, chr8 only): 3.25h / 2152 genes | < 4h | mosaiQTL: 2.79h / 43,376 genes / 20 chr (M3 Mac) | 🟢 | 2026-06-25 |
| SPEED-02 | Speedup vs miQTL (n=415, m=48, chr8 genes/min) | miQTL (HPC, chr8): 11 genes/min | >= 5× | mosaiQTL: 20.2× vs miQTL (same hw M3 Mac, simulated; 50 genes genome-wide) | ⚠️ | 2026-06-25 |
| SPEED-03 | Speedup vs tensorQTL (n=415, m=48, 1 CT) | tensorQTL: TBD | >= 2× | — | — | — |
| SPEED-04 | Speedup vs limix (n=415, m=48, 1 CT) | limix: TBD | >= 3× | — | — | — |
| SPEED-05 | Permutation throughput (1000 perms, genome-wide, 1 CT) | miQTL: TBD | <= 4h | mosaiQTL: 678× vs miQTL (same hw M3 Mac; n=250 genome-wide) | 🟢 | 2026-06-25 |
| SPEED-06 | genes/min (all chr, single CT) | miQTL (HPC, chr8): 11 genes/min | > 100/min | mosaiQTL: 259 genes/min (43,376 genes, 20 chr, M3 Mac) | 🟢 | 2026-06-25 |
| SPEED-07 | **PyTorch migration — LMM scan paths.** Prerequisite: SPEED-01–06 baselines committed AND SPEED-09 (float32) merged AND S4 chr19 concordance gate still ≥ 0.999 r² at float32. Three ordered sub-steps: **(a) scan_chromosome_batch** (permutation path, 1–2 days): `np.einsum` → `torch.einsum` (contraction strings unchanged); `np.linalg.pinv` → `torch.linalg.pinv`; `np.where`/`np.errstate` → `torch.where`; boundary: `torch.from_numpy(H_perp).to(device)` + `torch.from_numpy(G_pinv).to(device)`. Toggle: `ScanConfig.device='cuda'\|'cpu'`; full numpy fallback untouched. Expected speedup: 10–50× on permutation batch (1000 perms from ~1–3 hr → <15 min on 8 GB GPU). **(b) scan_chromosome_mi** (MI path, 1–2 days): same pattern; `torch.Generator` with fixed seed replaces `np.random` for imputation draws; `n_imp=11` batch dimension maps directly to GPU einsum axis. **(c) gene-batched scan_chromosome** (main scan, 2–3 weeks): restructure per-gene loop into `scan_chromosome(Y_block, ...)` where `Y_block` is `(n, gene_batch_size=50)`; per-gene h² estimated via `torch.vmap` over Brent's optimizer (Option B — see SPEED-08 note); outputs per-gene peak LOD only (MEM-05 constraint preserved — no full genes×loci matrix). Expected net speedup including HDF5 I/O: 3–8×. Sufficient GPU: 8 GB (RTX 3070/3080 or V100-16 GB for HPC). | CPU baseline | (a) ≥ 10× permutation; (b) ≥ 5× MI; (c) ≥ 3× main scan net | — | — | 🔴 |
| SPEED-08 | **PyTorch-native NB-GLMM (nb.py) — GPU-first design.** nb.py has not run yet, so it is designed with PyTorch as the native linear algebra backend from the start; numpy is the CPU fallback. Two stages: **(Stage 1)** NEBULA-LN cell-level null per gene: batch over genes within each cell type — `W (n_gene_batch, n_cells)` update is element-wise GPU op; PQL iterations are sequential but element-wise across the batch, no sequential gene dependency. Gene batch of 1000 × 10k cells (per cell type) = 40 MB at float32 — fits 8 GB GPU. **(Stage 2)** PQL NB-GLMM score test: operates at sample level (n=416), identical tensor shapes to LMM Stage 2 (H_t_raw: 854×416×8 = 11 MB); Woodbury solve `(m×m, m=48)` supports `torch.linalg.solve` batching over `(n_gene_batch, m, m)` natively. Memory: 8 GB sufficient for both stages under per-chromosome, per-cell-type streaming. Per-chromosome max: H_t_raw 11 MB + gene-batch W matrices ~40 MB + working response ~40 MB < 100 MB. 16–40 GB only needed if processing all cell types × all genes simultaneously (never done — violates streaming architecture). Prerequisite: SPEED-07a merged; nb.py algorithm derivation complete in docs/math/. | — | same as SPEED-07 targets for Stage 2 | — | — | 🔴 |
| SPEED-09 | **Float32 scan intermediates — prerequisite for SPEED-07/08.** Cast scan hot-path arrays to float32 at `precompute_chromosome` entry: `H = probs.astype(np.float32, copy=False)` + `eigvecs.astype(np.float32)` + `z_mat.astype(np.float32)`. All downstream intermediates (H_t_raw, H_perp, G_pinv, c, rss_red, LOD output, `ScanResult.lod`) inherit float32. Keep float64: `estimate_h2_null` internals (Brent's optimizer on flat LL surface, `slogdet` call). Validation gate: re-run S4 chr19 comparison after merge; LOD r² must remain ≥ 0.999. Rationale: LOD in 0–20 range with n=416; float32 gives ~7 significant digits → resolution ~0.000001 LOD at LOD=10; LOD < 0.01 is noise below any meaningful significance threshold. Doubles GPU throughput via tensor core utilisation; halves memory footprint of all scan tensors. | — | S4 chr19 r² ≥ 0.999 at float32 | — | — | 🔴 |

Amdahl ceiling: ~5.5× CPU-only (serial fraction f≈0.15, n_chr=20). GPU breaks Amdahl: permutation and MI batch are embarrassingly parallel over the batch dimension — GPU ceiling is I/O-bound (HDF5 chromosome load), not compute-bound.
PyTorch over JAX: mosaiQTL's imperative scan loop, checkpoint I/O, and Brent's scalar optimizer are incompatible with JAX's pure-function + XLA-compile requirement. PyTorch `torch.from_numpy` is zero-copy on CPU; `.to(device)` is the single host→device transfer per chromosome (~11 MB H_t_raw).
`estimate_h2_null` vectorisation (SPEED-07c, Option B): `torch.vmap(brent_step, in_dims=0)(h2_batch)` vectorises the Brent iteration over the gene batch; each gene still runs its own convergence loop but all genes advance one Brent step simultaneously. Preserves per-gene convergence semantics. Alternative (Option A, median h2) is simpler but introduces ~2–5% LOD error for genes far from the median; Option B is exact.
`ScanConfig.device` accepts `'mps'` (Apple Silicon M-series), `'cuda'` (HPC/discrete GPU), `'cpu'` (numpy fallback). Dev/validation on M3 MacBook 16 GB unified memory (MPS backend, ~3.6 TFLOPS float32 → expect 3–10× on permutation batch); production speedups on HPC V100/A100 (10–50×). MPS→CUDA transition is a one-line device string change provided all ops have CPU fallbacks.
**MPS op validation gate** (run before any MPS result is trusted; all comparisons vs numpy within 1e-4 on a small probe input at the shapes used in production):
  - `torch.einsum` contraction strings used in `scan_chromosome_batch` and `scan_chromosome_mi`
  - `torch.linalg.pinv` on shape `(J, J)` — partial MPS support; may silently fall back to CPU
  - `torch.vmap` over Brent step (SPEED-07c Option B) — partial MPS support; needs CPU fallback branch on `NotImplementedError`
  - `torch.linalg.solve` batched on shape `(n_gene_batch, m, m)` — used in SPEED-08 Woodbury solve
  Any failing op wraps as: `try: result = op_mps(...) except RuntimeError: result = op_cpu(tensor.cpu()).to(device)`
Run comparison: `python -m mosaiqtl.bench.compare_tools --tools mosaiqtl miqtl tensorqtl limix`

---

## MEMORY
HDF5 B-tree genome cache (Python) vs R HAPPY format (per-locus .RData files, sequential I/O):
- R: loads one .RData per locus on demand — filesystem stat + deserialise per call, O(1) but high overhead
- Python: HDF5 B-tree with chunked reads + np.memmap — O(log L) seek, random access, stays out of RAM
- This difference is the primary memory advantage of the Python implementation

| ID | Description | Target | Current | Assessment |
|---|---|---|---|---|
| MEM-01 | Full run peak RAM (n=415, m=48, 43k genes, 20 chr) | < 32 GB | mosaiQTL: 897 MB RSS (43,376 genes, 20 chr, M3 Mac) | gastroc scan_stats.csv 2026-06-25 |
| MEM-02 | scan_chromosome peak RAM per worker | < 4 GB | mosaiQTL: 18 MB (M3 Mac; 50 genes genome-wide) | tracemalloc per-gene loop |
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
| SWE-07 | CI pipeline active on main | green | 🟢 | .github/workflows/ci.yml — lint+test+security+mypy(warn) |
| SWE-08 | io/base.py BioFileParser ABC: shared read/validate/stream contract for all format parsers | implement when second I/O format parser is added (post-Phase-2); premature before ≥2 concrete parsers exist | 🔴 |
| SWE-09 | Replace BH-FDR placeholder in pipeline.py with Storey q-value (Storey 2002, 10.1111/1467-9868.00346). BH assumes π₀=1 (all genes null); Storey estimates π₀ from the p-value distribution shape and calibrates FDR against the true null proportion — recovering power lost by BH when real signal is present. This directly determines which genes are called eGenes. The BH line in pipeline.py with `# Storey q-value planned` is a placeholder that has been present since initial scaffolding and should not reach production. Validation: BIO-A5 (observed FDR ≤ 0.05 on sim_cc48_n200qtl) must use Storey q-value, not BH | `pipeline.py` | Phase 2 passes | 🔴 |
| SWE-10 | Extended gastroc covariates: (1) add `body_weight_g`, `tissue_weight_mg` to `gastroc_data/metadata.csv` and incorporate into `_build_covariates`; (2) add expression PCs (top k, default k=5, computed from log-normalised count matrix before scan) as optional scan covariates. Both scripts currently use only `~ sex + dataset` — neither includes expression PCs or physical measurements. R formula must be updated in lockstep with Python (`~ sex + dataset + body_weight_g + tissue_weight_mg + expr_pc1 + ...`). Expression PCs are the highest-impact addition. Prerequisite: gastroc validation run completes (Phase 2 gastroc gates pass). | `validate_gastroc.py` + `gastroc_miqtl_scan.R` + `metadata.csv` | gastroc validation passes | 🔴 |

---

## LMM EXTENSIONS (post-validation)
| ID | Description | Prerequisite | Status |
|---|---|---|---|
| LMM-EXT-01 | Replicate-aware LMM: scan.py reduces replicate samples to strain means via Z; h² and scan operate at strain level (m×m). Implemented in scan_h2lmm and scan_h2lmm_mi. Replicate covariates (sex/batch per replicate) averaged into strain-mean covariates automatically. | — | 🟢 |
| LMM-EXT-02 | Update test_lmm.py to use sample-level fixtures (tiny_phenotype, tiny_covariates, kinship_samples, z_matrix) | LMM-EXT-01 | 🔴 |

---

## MPP GENERALIZATION (post-Phase-2)
| ID | Description | Prerequisite | Status |
|---|---|---|---|
| MPP-01 | founders.py structural refactor: migrate FOUNDER_ORDER, FOUNDER_COLORS, F1_COLORS, STRAIN_ORDER from top-level constants to CCPopulation fields; viz/effects.py, viz/haplotypes.py, and viz/genome.py pull founder order and colors from `cache.population` instead of direct imports from founders.py | Phase 2 validation passes | 🔴 |
| MPP-02 | GenomeCache carries a `population: Population` field populated on load; scan functions pass `cache.population` downstream instead of accepting separate `founders` args | MPP-01 | 🔴 |
| MPP-03 | Add MAGIC-19 example Population instance (19 founders, Arabidopsis MAGIC or wheat MAGIC as reference) | MPP-01 | 🔴 |
| MPP-04 | Add `Population.cross_type: str` field (`'cc'`, `'do'`, `'magic19'`, `'f2'`, etc.) enabling factory-style dispatch (mirrors qtl2 `crosstype` → `QTLCross` factory). Required for `geno2allele_matrix` virtual dispatch in PORT-06 and for HMM `init()`/`step()` parameterisation across designs | MPP-01 | 🔴 |
| MPP-05 | Store `founder_geno` per chromosome in GenomeCache (founders × markers integer matrix, same format as qtl2 `cross2$founder_geno`). Required for SNP scan correctness (PORT-04/05) and for `geno2allele_matrix` computation in PORT-06. Currently absent — SNP scan silently missing founder allele context | MPP-02 | 🔴 |
| MPP-06 | Add `cross_info: np.ndarray \| None` to GenomeCache — per-individual integer metadata encoding generation number (DO), cross direction (F2), or Y-chromosome / mitochondrial haplotype origin (CC). In CC, chrY and mtDNA are maternally and paternally inherited intact (no recombination), so `cross_info` encodes which founder contributed the intact Y and mt haplotype per strain; this unlocks correct sex-chr and organelle QTL scanning without conflating autosomal founder-averaging with non-recombining segments | MPP-02 | 🔴 |

---

## QTL2-DERIVED SCAN API IMPROVEMENTS (post-Phase-2)
Patterns present in R/qtl2 and absent from mosaiQTL. Scan entry points already match qtl2 (full LOCO dict accepted; kinship_decomp dispatched per-chr internally). Items below are genuinely absent.

| ID | Description | Location | Prerequisite | Status |
|---|---|---|---|---|
| QTL2-01 | `find_peaks(scan_result, map, threshold, drop=1.5, prob=0.95)` — unified peak-finding returning a DataFrame (chr, pos_mb, lod, ci_lo_mb, ci_hi_mb). Currently absent; peak-finding is done ad-hoc in pipeline.py and validation scripts. Should support both LOD-drop (via `lod_drop_ci`) and Bayes CI (QTL2-02) as the interval method | `qtl/scan.py` | — | 🔴 |
| QTL2-02 | `bayes_int(scan_result, map, prob=0.95)` — Bayesian credible interval derived by treating the LOD curve as proportional to a log-posterior (flat prior) and finding the shortest interval containing `prob` mass. Complements `lod_drop_ci` (LOD-support threshold); the two methods make different assumptions about profile shape and are reported together in fine-mapping | `qtl/scan.py` | — | 🔴 |
| QTL2-03 | Stratified permutation + X-chr split thresholds: (a) `perm_strata: np.ndarray \| None` on `run_threshold_scans` permutes within strata to preserve sex structure; (b) `get_x_covar(population)` returns a sex × genotype interaction matrix for the X-chr null model, matching qtl2's `get_x_covar(cross)`; (c) `perm_Xsp=True` produces separate autosome and X permutation thresholds. Validation: compare X-chr thresholds against qtl2 `scan1perm(perm_Xsp=TRUE)` on sparcc_cache | `qtl/permutation.py` + `utils.py` | MPP-06 | 🔴 |
| QTL2-04 | `clean_probs(value_tol=1e-6, col_tol=0.01)` on `GenomeCache` — zero probabilities below `value_tol`, zero entire founder columns where max across all individuals is below `col_tol`, then renormalize. Mirrors qtl2 `clean_genoprob`. Prevents silent rank-deficiency at loci where a founder haplotype is locally absent. Validation: unit test with synthetic probe matrix with one near-zero founder column; verify scan_chromosome does not degenerate (check betas shape and LOD is finite) | `genotype/genomecache.py` | — | 🔴 |
| QTL2-05 | Expose `hsq: np.ndarray \| None` on `ScanResult` — the per-chromosome array of LOCO null h² already computed during scan but currently discarded. Add as an optional field (None when kinship not used). Validation: compare per-chromosome h² against Phase 2 R `scan.h2lmm` internal h² estimates exported from port_sim_validation_p2.R | `models/records.py` + `qtl/scan.py` | Phase 2 passes | 🔴 |
| QTL2-06 | `scan_gxe(phenotype, genomecache, intcovar, ...)` — returns additive LOD and interaction LOD columns (LOD_interaction = LOD_full − LOD_additive), matching qtl2 `scan1(intcovar=...)` semantics. Distinct from pairwise GxG (epistasis.py): intcovar is a fixed covariate (sex, treatment), not a second genotype locus. ⚠️ NEEDS VALIDATION PATH: miQTL has no GxE equivalent so no R reference exists; requires either qtl2 `scan1(intcovar=sex)` comparison on sparcc_cache or a planted GxE simulation — neither is in the current pipeline. Do not implement until validation path is defined | `qtl/gxe.py` | validation path required | 🔴 |
| QTL2-07 | `run_threshold_scans(fix_h2=False)` — statistically rigorous permutation threshold that re-estimates h² from each permuted phenotype rather than fixing it from the real phenotype. `fix_h2=False` re-estimates h² per permutation (one REML/ML solve per permutation per chromosome). The slow path in permutation.py already implements the loop; what is missing is: (a) vectorised `scan_chromosome_batch` support when h² differs per permutation (requires passing an h²-per-column array rather than a scalar), and (b) a validation comparing thresholds against the `fix_h2=True` fast path across a range of effect sizes | `qtl/permutation.py` | Phase 2 passes | 🔴 |

---

## DESIGN GOALS
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
| DESIGN-10 | Convert all CLAUDE.md files to user-facing documentation: each module's CLAUDE.md (root, src/mosaiqtl/, genotype/, qtl/, io/, viz/) maps to a vignette or user guide section covering API, design decisions, and worked examples. CLAUDE.md files are Claude Code context only and invisible to users — a scientist evaluating mosaiQTL has nothing to read without this. Target: docs/vignettes/ with at minimum one end-to-end CC eQTL workflow vignette equivalent to miQTL's README example and qtl2's user guide | docs/vignettes/ | Phase 4 passes | 🔴 |

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
| Sex GxE / intcovar scan | qtl/gxe.py | MED | 🔴 — see QTL2-06 (validation path required) |
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

### Already ported — HIGH priority (added)
| ID | miQTL R function(s) | Python location | Notes |
|---|---|---|---|
| PORT-08 | `build.allele.dir()` | `genotype/snp_db.py::build_founder_allele_db` + `load_founder_alleles` | 🟢 ported — single HDF5 from GM_ccfo_mm39.csv; `founder_alleles.h5` built in phase2/phase3 py_data/; gastroc S8 falls back to phase2 copy |

### Not yet ported — MEDIUM priority
| ID | miQTL R function(s) | Target Python location | Why needed |
|---|---|---|---|
| PORT-04 | `snp.null.par.bs.threshold.scan()` `snp.par.perm.threshold.scan()` `snp.perm.threshold.scan()` | `qtl/snp_scan.py` or `qtl/permutation.py` | SNP scan needs own thresholds; haplotype thresholds are anti-conservative for SNP scan |
| PORT-05 | `pairwise.cor.snp.scan()` `extract.r2.interval()` | `qtl/snp_scan.py` | LD/r² between imputed SNPs; needed for credible-set fine-mapping |
| PORT-06 | `straineff.mapping.matrix()` | `utils.py::straineff_mapping_matrix` | T×M diplotype→allele mapping used by MI and SNP scan; qtl2 equivalent is `geno2allele_matrix()` virtual — CC inbreds use outer-product, DO heterozygotes differ; method should live on Population (MPP-04) for cross-type dispatch |
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
| v0.5 GPU | SPEED-07 | PyTorch GPU path for permutation + MI batch scans |
| v0.4 Multiomics | first new modality (caQTL or sQTL) | ATAC or splicing extension |
