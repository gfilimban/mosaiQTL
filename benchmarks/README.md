# benchmarks/

Speed and memory baseline results. Committed to repo so CI can detect regressions.

## Layout

| Path | Content |
|---|---|
| `baseline/` | Committed baseline JSON from `pytest-benchmark` (`latest.json` symlink) |
| `results/` | Per-run JSON outputs (gitignored) |

## Running

```bash
# Capture a new baseline
pytest tests/benchmarks/ --benchmark-autosave \
    --benchmark-json=benchmarks/results/$(date +%Y%m%d).json

# Compare against committed baseline (CI gate: ≤10% regression)
pytest tests/benchmarks/ \
    --benchmark-compare=benchmarks/baseline/latest.json \
    --benchmark-fail-max-delta=10%

# Update GOALS.md speed metrics from latest run
python goals/update_goals.py --benchmark benchmarks/results/latest.json
```

## Before any performance PR

Run the comparison command above, confirm no regression, then commit the new JSON to
`benchmarks/baseline/` if this is intended as the new baseline.
