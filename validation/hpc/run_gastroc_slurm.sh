#!/bin/bash
# run_gastroc_slurm.sh
# full miQTL S1–S11 gastroc benchmark on SLURM.
#
# prerequisite: run convert_gastroc_h5ad.py manually before submitting.
#
# workflow:
#   step 1  setup    — S1 cache, S2 haplotype probs, S3 kinship
#   step 2  array    — S4 LMM scan, one node per chromosome
#   step 3  merge    — merge_chr_outputs.R → lod_genome.csv + timing summary
#   step 4  post     — S5 permutation, S6 collapse, S7 MI, S8 SNP,
#                       S9 mediation, S10 bootstrap CI, S11 instability
#
# usage:
#   bash validation/run_gastroc_slurm.sh            # submit full pipeline
#   bash validation/run_gastroc_slurm.sh --dry-run  # print jobs only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── paths — update for HPC before submitting ──────────────────────────────────
PKG_DIR="/share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl"
# NOTE: gastroc_miqtl_scan.R line ~59 has miqtl_dir hardcoded to a local path.
# On HPC, edit that block to:
#   miqtl_dir <- '/share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc'

R_SCRIPT="${PKG_DIR}/validation/r_reference/gastroc_miqtl_scan.R"
MERGE_SCRIPT="${PKG_DIR}/validation/r_reference/merge_chr_outputs.R"
GASTROC_DATA="${PKG_DIR}/validation/gastroc_data"
LOGS="${GASTROC_DATA}/logs"

TS=$(date +%Y%m%d_%H%M%S)

CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 X)

# ── SLURM resources ───────────────────────────────────────────────────────────
PARTITION="standard"
ACCOUNT="seyedam_lab"
CONDA_ENV="bioinfo"
R_LIB_PATH="/data/homezvol2/gfilimba/miniconda3/envs/bioinfo/lib/R/library"
CONDA_BASE="/data/homezvol2/gfilimba/miniconda3"   # derived from R_LIB_PATH root

TIME_SETUP="01:00:00"   # S1–S3
TIME_CHR="144:00:00"     # S4 per-chr (conservative — ~43k genes × ~900 loci)
TIME_MERGE="06:00:00"   # merge_chr_outputs.R
TIME_POST="48:00:00"     # S5: 1000 perms × 20 chr × 1 gene scan; 48 h safety margin
MEM="128G"
CPUS=8

SCAN_TYPE="C"    # C=cis  G=genome-wide  L=local (e.g. L:2 for ±2 Mb)
SECTIONS="4"     # S4 per-chr; setup handles 1,2,3 and post handles 5,6,7,8,9,10,11

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── guard: require counts_lognorm.h5 before submitting ───────────────────────
if [[ $DRY_RUN -eq 0 && ! -f "${GASTROC_DATA}/counts_lognorm.h5" ]]; then
    echo "ERROR: ${GASTROC_DATA}/counts_lognorm.h5 not found."
    echo "  Run first: conda run -n bioinfo python ${PKG_DIR}/validation/convert_gastroc_h5ad.py"
    exit 1
fi

# ── create output directories ─────────────────────────────────────────────────
mkdir -p "${LOGS}"
mkdir -p "${GASTROC_DATA}/r_data"
mkdir -p "${GASTROC_DATA}/plots"
for CHR in "${CHRS[@]}"; do
    mkdir -p "${GASTROC_DATA}/r_data/chr${CHR}"
done
echo "[${TS}] directories created"

# helper: activate conda consistently across nodes
CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && source ${CONDA_BASE}/etc/profile.d/mamba.sh && mamba activate ${CONDA_ENV}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SETUP: S1 cache info, S2 haplotype probs, S3 kinship
# ─────────────────────────────────────────────────────────────────────────────
SETUP_LOG="${LOGS}/setup_${TS}.log"
SETUP_SCRIPT_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_setup
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_SETUP}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${SETUP_LOG}
#SBATCH --error=${SETUP_LOG}

echo "[setup] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"
cd "${PKG_DIR}"

Rscript "${R_SCRIPT}" 1,2,3 2>&1
echo "[setup] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== STEP 1: SETUP ===" && echo "$SETUP_SCRIPT_CONTENT"
else
    SETUP_JID=$(echo "$SETUP_SCRIPT_CONTENT" | sbatch --parsable)
    echo "[${TS}] setup job submitted: ${SETUP_JID}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ARRAY: S4 LMM scan, one job per chromosome
# ─────────────────────────────────────────────────────────────────────────────
CHR_JIDS=()
for CHR in "${CHRS[@]}"; do
    CHR_LOG="${LOGS}/chr${CHR}_${TS}.log"
    CHR_SCRIPT_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_s4_chr${CHR}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_CHR}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${CHR_LOG}
#SBATCH --error=${CHR_LOG}
#SBATCH --dependency=afterok:${SETUP_JID:-1}

echo "[chr${CHR}] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"
cd "${PKG_DIR}"

# HPC mode: sections=${SECTIONS}, chr=${CHR}, scan_type=${SCAN_TYPE}
Rscript "${R_SCRIPT}" ${SECTIONS} ${CHR} ${SCAN_TYPE} 2>&1
echo "[chr${CHR}] finished \$(date)"
SLURM
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "=== STEP 2: CHR${CHR} ===" && echo "$CHR_SCRIPT_CONTENT"
    else
        JID=$(echo "$CHR_SCRIPT_CONTENT" | sbatch --parsable)
        CHR_JIDS+=("${JID}")
        echo "[${TS}] chr${CHR} job submitted: ${JID}"
    fi
done

# build afterok dependency string from all chr job IDs
if [[ $DRY_RUN -eq 0 ]]; then
    DEP_STR=$(printf ":%s" "${CHR_JIDS[@]}")
    DEP_STR="afterok${DEP_STR}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MERGE: combine per-chr lod.csv + compute timing summary
# ─────────────────────────────────────────────────────────────────────────────
MERGE_LOG="${LOGS}/merge_${TS}.log"
MERGE_SCRIPT_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_merge
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_MERGE}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=${MERGE_LOG}
#SBATCH --error=${MERGE_LOG}
#SBATCH --dependency=${DEP_STR:-afterok:1}

echo "[merge] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"
cd "${PKG_DIR}"

export MOSAIQTL_DIR="${PKG_DIR}"
Rscript "${MERGE_SCRIPT}" 2>&1
echo "[merge] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== STEP 3: MERGE ===" && echo "$MERGE_SCRIPT_CONTENT"
else
    MERGE_JID=$(echo "$MERGE_SCRIPT_CONTENT" | sbatch --parsable)
    echo "[${TS}] merge job submitted: ${MERGE_JID}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — POST: S5 permutation, S6 collapse, S7 MI, S8 SNP,
#                S9 mediation, S10 bootstrap CI, S11 instability
# ─────────────────────────────────────────────────────────────────────────────
POST_LOG="${LOGS}/post_${TS}.log"
POST_SCRIPT_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_post
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_POST}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${POST_LOG}
#SBATCH --error=${POST_LOG}
#SBATCH --dependency=afterok:${MERGE_JID:-1}

echo "[post] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"
cd "${PKG_DIR}"

Rscript "${R_SCRIPT}" 5,6,7,8,9,10,11 2>&1
echo "[post] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== STEP 4: POST ===" && echo "$POST_SCRIPT_CONTENT"
else
    POST_JID=$(echo "$POST_SCRIPT_CONTENT" | sbatch --parsable)
    echo "[${TS}] post job submitted: ${POST_JID}"
    echo ""
    echo "pipeline submitted:"
    echo "  setup   : ${SETUP_JID}"
    printf "  chrs    :"; printf " %s" "${CHR_JIDS[@]}"; echo ""
    echo "  merge   : ${MERGE_JID}"
    echo "  post    : ${POST_JID}"
    echo "  logs    : ${LOGS}/"
    echo ""
    echo "monitor:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${LOGS}/post_${TS}.log"
    echo ""
    echo "if chrs fail, resubmit with:"
    echo "  bash validation/rerun_failed_chrs_gastroc.sh"
fi
