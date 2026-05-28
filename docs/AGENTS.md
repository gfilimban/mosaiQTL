# mosaiQTL — Claude Code Agents Design Guide

## Project-specific vs general scope

| Location | Applies to | When to use |
|---|---|---|
| `.claude/agents/<name>.md` | this repo only | agents that know mosaiQTL conventions, CLAUDE.md rules, module layout — checked into git |
| `~/.claude/agents/<name>.md` | all your projects | reusable reasoning patterns that have no mosaiQTL-specific knowledge |

**For this project**: `scan-writer`, `validation-writer`, `test-writer`, `code-reviewer`, `domain-checker` → `.claude/agents/`

**General / user-level**: `architect` → `~/.claude/agents/` (MPP generalization decisions apply beyond this repo)

---

## How subagents interact with the CLAUDE.md hierarchy

Claude Code loads CLAUDE.md files in order when reading the repo:

```
CLAUDE.md                        ← root rules (inviolable)
src/mosaiqtl/CLAUDE.md           ← module inventory, dataclass fields, chr-type rule
src/mosaiqtl/genotype/CLAUDE.md  ← HMM, kinship, founder palette rules
src/mosaiqtl/qtl/CLAUDE.md       ← scan engine, LMM rules
src/mosaiqtl/io/CLAUDE.md        ← I/O, covariate handling
src/mosaiqtl/viz/CLAUDE.md       ← plot rules, founder color rules
```

Subagents spawned via the `Agent` tool inherit the full CLAUDE.md hierarchy automatically —
they see all project rules. The agent `.md` file adds role-specific constraints on top.

**Memory**: agents also load `memory/MEMORY.md` (feedback, project state) — they benefit
from accumulated session decisions without needing re-briefing.

---

## Agent set

| File | Location | Model | Purpose |
|---|---|---|---|
| `scan-writer.md` | `.claude/agents/` | sonnet | qtl/ and genotype/ engine implementation |
| `validation-writer.md` | `.claude/agents/` | sonnet | Phase 1–4 validation scripts |
| `test-writer.md` | `.claude/agents/` | sonnet | unit + integration tests |
| `code-reviewer.md` | `.claude/agents/` | opus | CLAUDE.md compliance + correctness audit |
| `domain-checker.md` | `.claude/agents/` | haiku | biological + numerical fact-checking (read-only) |
| `architect.md` | `~/.claude/agents/` | opus | MPP generalization + new modality design |

---

## Agent definitions

### scan-writer

**File**: `.claude/agents/scan-writer.md`

Implements qtl/, genotype/, and models/ code. Reads derivations in docs/math/ first.
Follows all CLAUDE.md inviolable rules: FWL vectorized (no locus loop), Z-matrix native
n=418, LOD denominator=n, chr always .astype(str), no inline hex.

Typical invocations:
- "implement PORT-06 straineff_mapping_matrix in utils.py"
- "add fix_h2=False slow path to run_threshold_scans"
- "vectorize scan_chromosome_mi over the imputation axis"

**Never**: modify validation scripts, write R code, introduce hardware-specific paths.

---

### validation-writer

**File**: `.claude/agents/validation-writer.md`

Writes and maintains `validation/validate_*.py` and `validation/r_reference/*.R`.

R scripts: pure miQTL API only — no tryCatch patches, no defensive branching.
Python scripts: logging not print; pathlib.Path not str; conda run -n bioinfo prefix;
section gates follow the phase definitions in GOALS.md; checkpoint CSVs saved per
section so any section can be resumed independently.

Typical invocations:
- "add S12 (vQTL comparison) to validate_gastroc.py"
- "fix S5 checkpoint to reload all per-chr null CSVs"
- "add throughput_by_chr.png to _make_descriptive_plots"

**Never**: modify package source code (src/), write unit tests, run scripts itself.

---

### test-writer

**File**: `.claude/agents/test-writer.md`

Writes tests in `tests/unit/` and `tests/integration/`. Appends only — never removes
or replaces existing tests. Uses fixtures from `tests/conftest.py`; adds new fixtures
there if needed but checks for existing ones first.

LMM tests use `tiny_strain_*` fixtures (strain-level, 10×10 kinship).
Replicate-aware tests use `tiny_phenotype` / `tiny_covariates` / `kinship_samples`.
NB tests use `counts_cells` / `counts_pb` / `library_sizes_cells`.

Typical invocations:
- "add unit tests for straineff_mapping_matrix in test_utils.py"
- "add integration test for S8 SNP scan (founder_alleles.h5 fixture required)"
- "parametrize test_estimate_h2_null over degenerate phenotype cases"

**Never**: modify source code to make tests pass — raise the issue to the user instead.

---

### code-reviewer

**File**: `.claude/agents/code-reviewer.md`

Audits code for compliance with CLAUDE.md inviolable rules and correctness.
Returns a structured report: PASS / WARN / FAIL per rule category.

**Checks performed**:

*Memory safety*
- [ ] No `grp['probs'][:]` eager load in genomecache.py (lazy per-chr)
- [ ] No full (genes × loci) LOD matrix materialised anywhere
- [ ] `del arr; gc.collect()` after large ephemeral arrays

*Correctness*
- [ ] chr always `.astype(str)` before merge or groupby
- [ ] LOD denominator = n (not n-p)
- [ ] FWL vectorized — no per-locus Python loop
- [ ] Z-matrix n=418 native (not n_samples)
- [ ] allele encoding: CC strain calls only
- [ ] loci_max_delta=1237 expected (pseudomarkers excluded)
- [ ] LOCO kinship used (never whole-genome kinship to scan_chromosome)
- [ ] `adata.raw.X` for counts (never `adata.X`)

*Style and safety*
- [ ] No inline hex — all colors imported from `genotype/founders.py`
- [ ] No `shell=True`, no bare `except:`, no hardcoded paths, no global mutable state
- [ ] No `np.load` for large arrays (use memmap); no `pd.read_csv` for large matrices (use HDF5)
- [ ] `logging` not `print`; `pathlib.Path` not `str`
- [ ] All public signatures type-annotated (PEP 484/526)
- [ ] Google-style docstrings on public API only
- [ ] Single quotes everywhere

*API boundaries*
- [ ] `lmm.py` only accessed through `scan.py` (not imported directly in scripts)
- [ ] `founders.py` is sole definition of hex colors and strain constants
- [ ] `FOUNDER_CODE` / `_display_founders()` used before any viz call (never raw A–H letters)
- [ ] `Population.n_founders` used in engine code (not hardcoded `8`)
- [ ] `scan_window='cis'` always paired with `gene_chr`

*ScanResult fields*
- [ ] `res.lod` — ndarray (n_loci,)
- [ ] `res.chr` — str, never int
- [ ] `res.betas` — ndarray (n_loci, J), sum ≈ 0 per locus, J founders (not J-1)
- [ ] `VQTLResult.scan_result` (not `.vqtl_scan`); `.mean_scan_result` (not `.mean_scan`)

Typical invocations:
- "review src/mosaiqtl/qtl/mediation.py for CLAUDE.md compliance"
- "audit the gastroc validation script for chr-type and founder-label rules"
- "check new snp_scan changes for memory safety"

**Output format**:
```
PASS  chr .astype(str) before merge
WARN  lmm.py imported directly in validate_sim_p2.py line 42 (use scan.py)
FAIL  inline hex '#F4C245' at effects.py:87 — import from founders.py
```

**Never**: modify files. Report only.

---

### domain-checker

**File**: `.claude/agents/domain-checker.md`

Read-only biological and numerical fact-checker. Never modifies files.
Uses only Read, Bash (grep/find), and WebSearch tools.

Checks:
- Founder strain names and hex colors match `genotype/founders.py` constants
- N_FOUNDERS=8, N_STRAINS=48, N_GENO=418, GENOME='mm39'
- loci_max_delta=1237 (correct: pseudomarkers excluded, CC strain calls)
- LOCO eigendecomp algebra: V = diag(1/W) + τ·Z·K_LOCO·Z^T
- NB working response: z = (log μ − offset) + (S − μ)/μ; W = μ/(1 + φμ)
- PQL convergence conditions
- GEV threshold: 95th percentile of max-LOD null (not mean)
- Storey π₀ estimator: natural cubic spline on λ grid 0.05–0.95

**Never invents** domain constants, founder biology, or scientific values — flags
"UNVERIFIED" and cites docs/math/ or REFERENCES.md instead.

Typical invocations:
- "verify the NB working response formula in nb.py matches NEBULA derivation"
- "check that FOUNDER_COLORS in effects.py match founders.py"
- "confirm the GEV threshold percentile used in permutation.py"

---

### architect (user-level: ~/.claude/agents/architect.md)

**File**: `~/.claude/agents/architect.md`

Senior software architect for multiparent population genomics packages.
Works at design level: API contracts, module boundaries, data-flow diagrams,
tradeoff analysis. Does not write implementation code.

Delivers: interface sketches, data flow tables, tradeoff comparisons, milestone splits.
Returns a concise design brief (< 400 words) with recommended approach and main risks.

Relevant context for mosaiQTL work:
- Two-stage NB-GLMM (nb.py Stage 1+2); lmm.py is validated port only
- Population (frozen Pydantic) is the API boundary for MPP generalization
- GenomeCache lazy per-chr I/O; full LOD matrix must never be materialised
- LOCO kinship mandatory; proximal contamination risk if violated
- Constant-memory streaming: per-chr, merge on completion

Typical invocations:
- "design the Stage 1 interface for caQTL (ATAC-seq) following nb.py conventions"
- "how should scan_gxe integrate with the existing scan_h2lmm API?"
- "design Population.cross_type dispatch for DO vs CC outer-product difference"
- "what is the minimal interface for io/base.py BioFileParser ABC (SWE-08)?"

**Never**: writes implementation code, modifies files, makes domain biology claims.

---

## Usage patterns

### Parallel independent tasks

```python
# Two agents in one message — parallel execution
Agent(description="implement PORT-06", subagent_type="scan-writer",
      prompt="Implement straineff_mapping_matrix in utils.py per PORT-06 spec in GOALS.md...")

Agent(description="write PORT-06 tests", subagent_type="test-writer",
      prompt="Add unit tests for straineff_mapping_matrix in tests/unit/test_utils.py...")
```

### Sequential with review gate

```python
# 1. implement
Agent(subagent_type="scan-writer", prompt="Add bayes_int to qtl/scan.py...")
# 2. review result
Agent(subagent_type="code-reviewer", prompt="Review the bayes_int addition in qtl/scan.py...")
# 3. domain check
Agent(subagent_type="domain-checker", prompt="Verify bayes_int credible interval formula...")
```

### Design before implement

```python
# 1. architect designs interface
Agent(subagent_type="architect", prompt="Design the scan_gxe API...")
# 2. scan-writer implements approved design
Agent(subagent_type="scan-writer", prompt="Implement scan_gxe per design: [paste brief]...")
```

---

## File layout

```
.claude/
└── agents/
    ├── scan-writer.md
    ├── validation-writer.md
    ├── test-writer.md
    ├── code-reviewer.md
    └── domain-checker.md

~/.claude/
└── agents/
    └── architect.md
```

Agent files are checked into git (`.claude/agents/`) except `architect.md` which lives at
`~/.claude/agents/` and is not repo-specific. CLAUDE.md files remain gitignored.
