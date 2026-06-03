## Summary
<!-- What changed and why -->

## Type
- [ ] Bug fix
- [ ] Feature / new functionality
- [ ] Performance improvement
- [ ] Refactor
- [ ] Docs / validation only

## Validation checklist
<!-- All boxes must be checked before merging dev → main -->

### Tests
- [ ] `pytest tests/unit/ -x -q` passes
- [ ] `pytest tests/integration/ -x -q` passes
- [ ] No existing tests removed or weakened

### Phase 1 gate (required for any change to qtl/ or genotype/)
- [ ] `python validation/validate_sim.py` passes all three gates
  - P1-S1: peak-LOD r² ≥ 0.990
  - P1-S2: MI r ≥ 0.75
  - P1-S3: GEV threshold Δ < 0.5 LOD

### Code quality
- [ ] `make lint` clean (flake8 + interrogate)
- [ ] No new `shell=True`, bare `except:`, hardcoded hex colors, or `print()`
- [ ] `chr` columns cast with `.astype(str)` before any merge
- [ ] No full genes × loci LOD matrix materialised
- [ ] No per-gene HDF5 open inside a scan loop

### Performance (required for any change to lmm.py / scan.py / permutation.py)
- [ ] `pytest tests/benchmarks/ --benchmark-compare=benchmarks/baseline/latest.json` shows < 10% regression
  - or: explain why the regression is acceptable

### Math changes (required if LOD formula, h² parameterisation, or FWL projection changed)
- [ ] Derivation updated in `docs/math/`
- [ ] R/Python comparison output attached or linked

## Claude Code review
<!-- Claude Code review runs automatically on PR open. Address all FAIL findings before merge. WARN findings require a written justification below. -->

### WARN justifications (if any)
<!-- List any WARN findings you are accepting and why -->
