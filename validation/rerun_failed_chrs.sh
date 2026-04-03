#!/bin/bash
# rerun_failed_chrs.sh
#
# detects chromosomes whose per-chr array job hit the time limit (or otherwise
# failed to produce lod_r.csv) and resubmits only those jobs in slurm mode.
# existing per-chr output dirs for completed chromosomes are untouched.
#
# after all resubmitted jobs finish, submits a single merge + validate job.
#
# usage:
#   bash validation/rerun_failed_chrs.sh              # auto-detect failed chrs
#   bash validation/rerun_failed_chrs.sh 4,7,15       # force-rerun specific chrs
#   bash validation/rerun_failed_chrs.sh --dry-run    # print jobs, don't submit
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── paths (update for HPC before submitting) ──────────────────────────────────
PKG_DIR="/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/mosaiqtl"
DATA_DIR="/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/miqtl"
R_SCRIPT="${PKG_DIR}/validation/r_reference/port_validation.R"
MERGE_SCRIPT="${PKG_DIR}/validation/merge_chr_outputs.R"
PY_SCRIPT="${PKG_DIR}/validation/validate_port.py"
R_DATA="${PKG_DIR}/validation/port_validation/r_data"
LOGS="${PKG_DIR}/validation/port_validation/logs"

# ── SLURM resources (must match original job) ─────────────────────────────────
PARTITION="standard"
ACCOUNT="seyedam_lab"
MEM="32G"
CPUS=4
CONDA_ENV="bioinfo"
R_LIB_PATH="/share/crsp/lab/seyedam/gfilimba/miniconda3/envs/bioinfo/lib/R/library"

# extra time for timed-out chrs — 4 h more than original
TIME_CHR="16:00:00"
TIME_MERGE="02:00:00"

# sentinel file that proves a chr job completed successfully.
# ci_r.csv is written by S10 (positional CI) — the last section that commonly
# times out. if it exists and is non-empty the job ran to completion.
SENTINEL="ci_r.csv"

ALL_CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 X)

TS=$(date +%Y%m%d_%H%M%S)
DRY_RUN=0

# ── parse args ────────────────────────────────────────────────────────────────
FORCE_CHRS=""
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1 && continue
    # anything that looks like a chr list (digits and commas/X)
    [[ "${arg}" =~ ^[0-9X,]+$ ]] && FORCE_CHRS="${arg}" && continue
done

# ── detect failed chromosomes ─────────────────────────────────────────────────
FAILED_CHRS=()

if [[ -n "${FORCE_CHRS}" ]]; then
    # user explicitly listed chrs to rerun
    IFS=',' read -ra FAILED_CHRS <<< "${FORCE_CHRS}"
    echo "[${TS}] force-rerunning chrs: ${FAILED_CHRS[*]}"
else
    echo "[${TS}] scanning for failed/missing chr outputs in ${R_DATA}/"
    for CHR in "${ALL_CHRS[@]}"; do
        sentinel_path="${R_DATA}/chr${CHR}/${SENTINEL}"
        if [[ ! -f "${sentinel_path}" || ! -s "${sentinel_path}" ]]; then
            FAILED_CHRS+=("${CHR}")
            echo "  chr${CHR}: MISSING or EMPTY — will resubmit"
        else
            echo "  chr${CHR}: ok ($(wc -l < "${sentinel_path}") lines)"
        fi
    done
fi

if [[ ${#FAILED_CHRS[@]} -eq 0 ]]; then
    echo "[${TS}] all chromosomes have output — nothing to resubmit."
    echo "         run merge+validate manually or use --dry-run to inspect."
    exit 0
fi

echo ""
echo "[${TS}] ${#FAILED_CHRS[@]} chromosome(s) to resubmit: ${FAILED_CHRS[*]}"
echo ""

mkdir -p "${LOGS}"

# ── submit one array job per failed chromosome ────────────────────────────────
RETRY_JIDS=()

for CHR in "${FAILED_CHRS[@]}"; do
    CHR_LOG="${LOGS}/retry_chr${CHR}_${TS}.log"
    # ensure output subdir exists — do NOT delete existing files in other subdirs
    mkdir -p "${R_DATA}/chr${CHR}"

    CHR_SCRIPT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=mqtl_retry_chr${CHR}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_CHR}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${CHR_LOG}
#SBATCH --error=${CHR_LOG}

echo "[retry chr${CHR}] started \$(date)"
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export R_LIB_PATH="${R_LIB_PATH}"

# slurm mode: writes to r_data/chr${CHR}/ — does not touch other chr dirs
# sections: s4 lmm, s5 perm, s7 MI, s8 snp, s9 mediation, s10 CI, s11 instability
Rscript "${R_SCRIPT}" ${CHR} Inf slurm 4,5,7,8,9,10,11 2>&1
echo "[retry chr${CHR}] finished \$(date)"
SLURM
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "=== CHR${CHR} RETRY JOB ==="
        echo "${CHR_SCRIPT}"
        echo ""
    else
        JID=$(echo "${CHR_SCRIPT}" | sbatch --parsable)
        RETRY_JIDS+=("${JID}")
        echo "[${TS}] submitted chr${CHR} retry: job ${JID} → ${CHR_LOG}"
    fi
done

# ── submit merge + validate after all retries finish ─────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
    DEP_STR=$(printf ":%s" "${RETRY_JIDS[@]}")
    DEP_STR="afterok${DEP_STR}"
else
    DEP_STR="afterok:DRYRUN"
fi

MERGE_LOG="${LOGS}/merge_after_retry_${TS}.log"
MERGE_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=mqtl_merge_retry
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_MERGE}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${MERGE_LOG}
#SBATCH --error=${MERGE_LOG}
#SBATCH --dependency=${DEP_STR}

echo "[merge] started \$(date)"
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export R_LIB_PATH="${R_LIB_PATH}"

# merge all per-chr subdirs (completed + just-retried) into r_data/
Rscript "${MERGE_SCRIPT}" 2>&1

# validate Python against merged R outputs — all sections
python "${PY_SCRIPT}" 2>&1

echo "[merge] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== MERGE + VALIDATE JOB ==="
    echo "${MERGE_CONTENT}"
else
    MERGE_JID=$(echo "${MERGE_CONTENT}" | sbatch --parsable)
    echo ""
    echo "[${TS}] merge+validate job submitted: ${MERGE_JID}"
    echo "         dependency: ${DEP_STR}"
    echo ""
    echo "pipeline:"
    echo "  retry chrs : ${RETRY_JIDS[*]}"
    echo "  merge+val  : ${MERGE_JID}"
    echo "  logs       : ${LOGS}/"
    echo ""
    echo "monitor:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${MERGE_LOG}"
fi
