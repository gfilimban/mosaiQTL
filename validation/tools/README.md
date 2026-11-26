# validation/tools/

Utility scripts that support validation but are not gated validation entry points.
Not imported by the package; add `validation/tools/` to `sys.path` before importing.

| Script | Purpose |
|---|---|
| `convert_h5ad.py` | Convert raw `.h5ad` → `pseudobulk.h5` + `metadata.csv` + `exp_gene_info_{tissue}.csv` |
| `build_allele_dir.py` | Build GigaMUGA `allele_dir/` from `GM_ccf1fo_mm39.csv` (Phase 2 S8 input) |
| `sim_sc.py` | Simulate single-cell NB counts from sparcc strain-level phenotypes (Phase 4 input) |
| `plot_phase2.py` | Shared plot helpers imported by `validate_port.py` (Phase 2 gastroc figures) |
| `report.py` | Generate `docs/VALIDATION_STATUS.md` from phase output CSVs |
| `check_mediation_h2_approx.py` | Empirical check: fixed-h2 vs per-mediator-h2 in mediation scan |
| `scan_weight_qtl.py` | Weight QTL experiment: test whether cis-locus explains body/tissue weight |
| `csv_to_tsv.py` | Convert pipeline output `.csv` / `.csv.gz` tables to `.tsv` / `.tsv.gz` beside the originals; verified, atomic, sources untouched |

## Run patterns

```bash
# Convert h5ad (prerequisite for gastroc/ctxhipp scans)
conda run -n bioinfo python validation/tools/convert_h5ad.py \
    --h5ad data/gastroc.h5ad --tissue gastroc \
    --out validation/gastroc_data/next

# Build allele dir (Phase 2 S8 prerequisite)
conda run -n bioinfo python validation/tools/build_allele_dir.py

# Generate validation status report
conda run -n bioinfo python validation/tools/report.py

# Convert pipeline outputs CSV -> TSV (writes X.tsv next to X.csv; never touches the CSV)
conda run -n bioinfo python validation/tools/csv_to_tsv.py -r --dry-run <output_dir>   # preview
conda run -n bioinfo python validation/tools/csv_to_tsv.py -r <output_dir>             # convert
```
