# benchmarks/
Performance is tracked in two committed, append-only histories written by `benchmarks/record.py`:

| File | One row per | Written by |
|---|---|---|
| `history.csv` | benchmark per run: date, commit, dirty, branch, CPU, OS, Python, median/IQR/mean/sd/min (s), rounds, session peak RSS (MB) | `make bench` |
| `pipeline_history.csv` | Nextflow process per production run: label, commit, tasks, failures, total and max wall time, CPU-hours, max peak RSS (GB) | `make bench-trace` |
| `baseline/latest.json` | the reference pytest-benchmark run (Apple M3) | copied by hand at a release |

Micro-benchmarks (tests/benchmarks): test_scan_speed.py — LOCO kinship, kinship eigendecomposition, null $h^2$
fit, one-chromosome scan; test_memory.py — peak-memory limits (pass/fail). Simulated sparc_cc40 data only.
CI does not run benchmarks: timings are only comparable on one machine.

## Procedure
```bash
pip install pytest-benchmark                              # not in [dev]
make bench                                                # all benchmarks; ARGS='-k scan' for a subset
make bench-trace TRACE=<output_dir>/logs/trace.txt LABEL=gastroc_ct_Lw1   # after a Nextflow run
```
- `make bench` prints each median as a ratio to the baseline (same CPU only) and flags ratios > 1.10. A PR
  touching lmm.py, scan.py or permutation.py includes that output; commit the new history.csv rows with it.
- nextflow.config writes `<output_dir>/logs/trace.txt` (raw units) for every run; record each production run.
- Release: if a run is the intended new reference, save it with
  `pytest tests/benchmarks --benchmark-json=benchmarks/baseline/latest.json` on the reference machine.
- `python goals/update_goals.py [--dry-run]` refreshes goals/GOALS.md from validation/*.csv and names any
  missing input.
- Profile with `scalene --cpu --memory <script>` before optimising.

Comparators: miQTL is the ground truth (LOD $r^2$). tensorQTL, jaxQTL and SAIGE-QTL need SNP dosages; haplotype
→ dosage conversion does not exist yet. Earlier hand-recorded full gastroc scan (2026-06-25; 43,376 genes, 20 chr,
n = 416): 2.79 h, 897 MB peak RSS, 259 genes/min — its scan_stats.csv source no longer exists.
