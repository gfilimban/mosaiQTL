#!/usr/bin/env bash
# Run validate_tissue.py on the chr8 synthetic dataset
conda run -n bioinfo python validation/validate_tissue.py \
    --tissue gastroc \
    --data tests/workflow/syn_data \
    --out tests/workflow/vt_out \
    --scan \
    --scan_type L \
    --chr 8
