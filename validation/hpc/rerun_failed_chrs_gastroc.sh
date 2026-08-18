#!/bin/bash
# rerun_failed_chrs_gastroc.sh
#
# detects chromosomes whose S4 job failed to produce lod.csv and resubmits
# only those jobs. completed chromosome dirs are untouched.
# after retries finish, submits merge + post (S5–S11) jobs.
#
# usage:
#   bash validation/rerun_failed_chrs_gastroc.sh              # auto-detect
#   bash validation/rerun_failed_chrs_gastroc.sh 4,7,X        # force specific
#   bash validation/rerun_failed_chrs_gastroc.sh --dry-run    # print only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PKG_DIR="/share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl"
R_SCRIPT="${PKG_DIR}/validation/r_reference/gastroc_miqtl_scan.R"
MERGE_SCRIPT="${PKG_DIR}/validation/r_reference/merge_chr_outputs.R"
GASTROC_DATA="${PKG_DIR}/validation/gastroc_data"
LOGS="${GASTROC_DATA}/logs"

PARTITION="standard"
ACCOUNT="seyedam_lab"
CONDA_ENV="bioinfo"
R_LIB_PATH="/data/homezvol2/gfilimba/miniconda3/envs/bioinfo/lib/R/library"
CONDA_BASE="/data/homezvol2/gfilimba/miniconda3"
TIME_CHR="16:00:00"    # extra time for timed-out chrs
TIME_MERGE="00:30:00"
TIME_POST="06:00:00"
MEM="64G"
CPUS=4

SENTINEL="lod.csv"    # S4 writes this last; missing = job did not complete
ALL_CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 X)

TS=$(date +%Y%m%d_%H%M%S)
DRY_RUN=0
FORCE_CHRS=""

for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1 && continue
    [[ "${arg}" =~ ^[0-9X,]+$ ]] && FORCE_CHRS="${arg}" && continue
done

CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && source ${CONDA_BASE}/etc/profile.d/mamba.sh && mamba activate ${CONDA_ENV}"

# ── detect failed chromosomes ─────────────────────────────────────────────────
FAILED_CHRS=()
if [[ -n "${FORCE_CHRS}" ]]; then
    IFS=',' read -ra FAILED_CHRS <<< "${FORCE_CHRS}"
    echo "[${TS}] force-rerunning chrs: ${FAILED_CHRS[*]}"
else
    echo "[${TS}] scanning ${GASTROC_DATA}/ for missing ${SENTINEL}"
    for CHR in "${ALL_CHRS[@]}"; do
        p="${GASTROC_DATA}/chr${CHR}/${SENTINEL}"
        if [[ ! -f "${p}" || ! -s "${p}" ]]; then
            FAILED_CHRS+=("${CHR}")
            echo "  chr${CHR}: MISSING — will resubmit"
        else
            echo "  chr${CHR}: ok"
        fi
    done
fi

if [[ ${#FAILED_CHRS[@]} -eq 0 ]]; then
    echo "[${TS}] all chromosomes complete — run merge manually or use --dry-run."
    exit 0
fi
echo ""
echo "[${TS}] ${#FAILED_CHRS[@]} chromosome(s) to resubmit: ${FAILED_CHRS[*]}"
echo ""
mkdir -p "${LOGS}"

# ── resubmit failed chromosomes ───────────────────────────────────────────────
RETRY_JIDS=()
for CHR in "${FAILED_CHRS[@]}"; do
    mkdir -p "${GASTROC_DATA}/chr${CHR}"
    CHR_LOG="${LOGS}/retry_chr${CHR}_${TS}.log"
    CHR_SCRIPT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_retry_chr${CHR}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_CHR}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${CHR_LOG}
#SBATCH --error=${CHR_LOG}

echo "[retry chr${CHR}] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"

Rscript "${R_SCRIPT}" 4 ${CHR} 2>&1
echo "[retry chr${CHR}] finished \$(date)"
SLURM
    )

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "=== CHR${CHR} RETRY ===" && echo "${CHR_SCRIPT}"
    else
        JID=$(echo "${CHR_SCRIPT}" | sbatch --parsable)
        RETRY_JIDS+=("${JID}")
        echo "[${TS}] chr${CHR} retry submitted: ${JID} → ${CHR_LOG}"
    fi
done

if [[ $DRY_RUN -eq 0 ]]; then
    DEP_STR=$(printf ":%s" "${RETRY_JIDS[@]}")
    DEP_STR="afterok${DEP_STR}"
else
    DEP_STR="afterok:DRYRUN"
fi

# ── merge after retries ───────────────────────────────────────────────────────
MERGE_LOG="${LOGS}/merge_retry_${TS}.log"
MERGE_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_merge_retry
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_MERGE}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=${MERGE_LOG}
#SBATCH --error=${MERGE_LOG}
#SBATCH --dependency=${DEP_STR}

echo "[merge] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"

Rscript "${MERGE_SCRIPT}" 2>&1
echo "[merge] finished \$(date)"
SLURM
)

# ── post (S5–S11) after merge ─────────────────────────────────────────────────
POST_LOG="${LOGS}/post_retry_${TS}.log"
POST_CONTENT=$(cat << SLURM
#!/bin/bash
#SBATCH --job-name=gastroc_post_retry
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_POST}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${POST_LOG}
#SBATCH --error=${POST_LOG}
#SBATCH --dependency=afterok:MERGE_JID_PLACEHOLDER

echo "[post] started \$(date)"
${CONDA_INIT}
export R_LIB_PATH="${R_LIB_PATH}"

Rscript "${R_SCRIPT}" 5,6,7,8,9,10,11 2>&1
echo "[post] finished \$(date)"
SLURM
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== MERGE ===" && echo "${MERGE_CONTENT}"
    echo "=== POST ===" && echo "${POST_CONTENT}"
else
    MERGE_JID=$(echo "${MERGE_CONTENT}" | sbatch --parsable)
    # inject real merge JID into post dependency
    POST_CONTENT_REAL="${POST_CONTENT/MERGE_JID_PLACEHOLDER/${MERGE_JID}}"
    POST_JID=$(echo "${POST_CONTENT_REAL}" | sbatch --parsable)
    echo ""
    echo "[${TS}] merge submitted: ${MERGE_JID}"
    echo "[${TS}] post  submitted: ${POST_JID}"
    echo "  retry chrs : ${RETRY_JIDS[*]}"
    echo "  logs       : ${LOGS}/"
    echo ""
    echo "monitor:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${POST_LOG}"
fi
