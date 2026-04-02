#!/bin/bash
# run_validation_slurm.sh
# parallelise port_validation.R across chromosomes using SLURM array jobs.
#
# workflow:
#   step 0  setup    — dirs, genome-level sections (s1 s2 s3 s5)
#   step 1  array    — per-chr sections (s4 s6 s7 s8) one node per chr
#   step 2  merge    — combine per-chr CSVs → unified r_data/ files
#   step 3  python   — run validate_port.py against merged outputs
#
# usage:
#   bash validation/run_validation_slurm.sh            # all chromosomes
#   bash validation/run_validation_slurm.sh --dry-run  # print jobs only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
PKG_DIR="/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl"
DATA_DIR="/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl"
R_SCRIPT="${PKG_DIR}/validation/r_reference/port_validation.R"
MERGE_SCRIPT="${PKG_DIR}/validation/merge_chr_outputs.R"
PY_SCRIPT="${PKG_DIR}/validation/validate_port.py"

# output directories
R_DATA="${PKG_DIR}/validation/port_validation/r_data"
LOGS="${PKG_DIR}/validation/port_validation/logs"

# timestamp for this run — used in all log filenames
TS=$(date +%Y%m%d_%H%M%S)

# chromosomes to scan
CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 X)

# SLURM resources
PARTITION="standard"    # update for your HPC partition
ACCOUNT="seyedam_lab"   # update for your HPC account
TIME_SETUP="01:00:00"     # setup step (s1-s3, s5)
TIME_CHR="08:00:00"       # per-chr (s4 s6 s7 s8) — conservative
TIME_MERGE="02:00:00"     # merge + python validation
MEM="32G"
CPUS=4
CONDA_ENV="bioinfo"
R_LIB_PATH="/share/crsp/lab/seyedam/gfilimba/miniconda3/envs/bioinfo/lib/R/library"  # update if conda path differs

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── create directories ────────────────────────────────────────────────────────
mkdir -p "${LOGS}"
for CHR in "${CHRS[@]}"; do
    mkdir -p "${R_DATA}/chr${CHR}"
done
mkdir -p "${R_DATA}/merged"
echo "[${TS}] directories created"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — SETUP: genome-level sections (s1 s2 s3 s5)
# runs once on one node before the array
# ─────────────────────────────────────────────────────────────────────────────
SETUP_LOG="${LOGS}/setup_${TS}.log"
SETUP_SCRIPT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=mqtl_setup
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_SETUP}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${SETUP_LOG}
#SBATCH --error=${SETUP_LOG}

echo "[setup] started \$(date)"
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export R_LIB_PATH="${R_LIB_PATH}"

# genome-level sections only: s1 cache, s2 haplotype, s3 kinship, s5 collapse
# per-chromosome sections (s4 s6 s7 s8 s9 s10 s11) run in parallel array jobs
Rscript "${R_SCRIPT}" all Inf 1,2,3,5 2>&1
echo "[setup] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== SETUP JOB ==="
    echo "$SETUP_SCRIPT"
else
    SETUP_JID=$(echo "$SETUP_SCRIPT" | sbatch --parsable)
    echo "[${TS}] setup job submitted: ${SETUP_JID}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ARRAY: one job per chromosome (s4 s6 s7 s8)
# ─────────────────────────────────────────────────────────────────────────────
CHR_JIDS=()
for CHR in "${CHRS[@]}"; do
    CHR_LOG="${LOGS}/chr${CHR}_${TS}.log"
    CHR_SCRIPT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=mqtl_chr${CHR}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_CHR}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${CHR_LOG}
#SBATCH --error=${CHR_LOG}
#SBATCH --dependency=afterok:${SETUP_JID:-1}

echo "[chr${CHR}] started \$(date)"
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export R_LIB_PATH="${R_LIB_PATH}"

# per-chromosome sections: s4 lmm scan, s6 permutation, s7 MI,
# s8 snp scan, s9 mediation, s10 positional CI, s11 instability
# s1 s2 s3 s5 are genome-level and handled by setup job
# slurm mode: writes to r_data/chr${CHR}/
Rscript "${R_SCRIPT}" ${CHR} Inf slurm 4,6,7,8,9,10,11 2>&1
echo "[chr${CHR}] finished \$(date)"
SLURM
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "=== CHR${CHR} JOB ==="
        echo "$CHR_SCRIPT"
    else
        JID=$(echo "$CHR_SCRIPT" | sbatch --parsable)
        CHR_JIDS+=("$JID")
        echo "[${TS}] chr${CHR} job submitted: ${JID}"
    fi
done

# build dependency string: afterok:jid1:jid2:...
if [[ $DRY_RUN -eq 0 ]]; then
    DEP_STR=$(printf ":%s" "${CHR_JIDS[@]}")
    DEP_STR="afterok${DEP_STR}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MERGE: combine per-chr CSVs + run Python validation
# ─────────────────────────────────────────────────────────────────────────────
MERGE_LOG="${LOGS}/merge_${TS}.log"
MERGE_SCRIPT_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=mqtl_merge
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_MERGE}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${MERGE_LOG}
#SBATCH --error=${MERGE_LOG}
#SBATCH --dependency=${DEP_STR:-afterok:1}

echo "[merge] started \$(date)"
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export R_LIB_PATH="${R_LIB_PATH}"

# merge per-chr R outputs
Rscript "${MERGE_SCRIPT}" 2>&1

# run Python validation against merged outputs
python "${PY_SCRIPT}" 2>&1

echo "[merge] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== MERGE JOB ==="
    echo "$MERGE_SCRIPT_CONTENT"
else
    MERGE_JID=$(echo "$MERGE_SCRIPT_CONTENT" | sbatch --parsable)
    echo "[${TS}] merge job submitted: ${MERGE_JID}"
    echo ""
    echo "pipeline submitted:"
    echo "  setup:  ${SETUP_JID}"
    echo "  chrs:   ${CHR_JIDS[*]}"
    echo "  merge:  ${MERGE_JID}"
    echo "  logs:   ${LOGS}/"
    echo ""
    echo "monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${LOGS}/merge_${TS}.log"
fi
