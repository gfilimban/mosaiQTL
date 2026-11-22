## Summary
<!-- What changed and why; goal or fix IDs -->

## Type
- [ ] fix
- [ ] feat
- [ ] perf
- [ ] refactor
- [ ] docs / validation only

## Checks
- [ ] `pytest tests/unit tests/integration -x -q` passes; no test removed or weakened
- [ ] `make lint` clean (black, flake8, interrogate, mypy)
- [ ] qtl/ or genotype/ changed → `python validation/validate_sim.py` passes (P1-S1 peak-LOD r² ≥ 0.990, P1-S2 MI r ≥ 0.75, P1-S3 GEV Δ < 0.5 LOD)
- [ ] lmm.py, scan.py or permutation.py changed → `make bench` output pasted and new benchmarks/history.csv rows committed; ratios > 1.10 explained
- [ ] Inputs loaded lazily (only the chromosomes, genes and columns used); no genes × loci array; gene-invariant work and file opens outside per-gene loops; no gc.collect() in loops
- [ ] `chr` stored as 'chr{N}' str before merges; no inline hex, `shell=True`, bare `except:` or `print()`

## Documentation
- [ ] CHANGELOG.md `[Unreleased]` entry (required by CI when src/ or workflow/ change)
- [ ] README updated if the CLI, Nextflow params, inputs or outputs changed
- [ ] docs/ARCHITECTURE.md updated if modules or data flow changed
- [ ] docs/math/*.typ and docs/METHODS.typ (LaTeX math via Typst) updated if a formula, h² parameterisation or FWL projection changed; R/Python comparison linked

## Review
<!-- Automated review posts PASS / WARN / FAIL. Fix every FAIL; justify accepted WARNs here. -->
