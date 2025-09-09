export BASE_DIR=/workspace/3D-Generation-baseline
export WORKING_DIR=${BASE_DIR}
export VENV_DIR=${BASE_DIR}/venv

# export PATH=${VENV_DIR}/bin:$PATH
# source ${VENV_DIR}/bin/activate

# --------------------------------
# Run Parameters
# --------------------------------
SEED=42
PROMPT="a photo of a hamburger"
ITERS=500

REPULSION_TYPE="wo"


TASK_NAME="${PROMPT// /_}_ours_${REPULSION_TYPE}_modifyvisualizer"
# OUTPUT_DIR="${BASE_DIR}/outputs/${SLURM_JOB_ID}/${TASK_NAME}"
OUTPUT_DIR="${BASE_DIR}/outputs/${TASK_NAME}"
mkdir -p ${OUTPUT_DIR}

echo "[INFO] Task Name: ${TASK_NAME}"
echo "[INFO] Output Directory: ${OUTPUT_DIR}"

# # --------------------------------
# Run Main Script
# --------------------------------
CMD="python ${WORKING_DIR}/main_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    prompt=\"${PROMPT}\" \
    save_path=${PROMPT// /_} \
    outdir=${OUTPUT_DIR} \
    seed=${SEED} \
    iters=${ITERS}"

if [ "$REPULSION_TYPE" != "wo" ]; then
    CMD="${CMD} repulsion_type=${REPULSION_TYPE} kernel_type=${KERNEL_TYPE}"
fi

echo "[RUNNING COMMAND] $CMD"
eval $CMD

echo "[INFO] Job completed successfully."
