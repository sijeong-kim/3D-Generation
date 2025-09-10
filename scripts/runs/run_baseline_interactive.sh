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
ITER=500

TASK_NAME="${PROMPT// /_}_modified_baseline"
# OUTPUT_DIR="${BASE_DIR}/outputs/${SLURM_JOB_ID}/${TASK_NAME}"
OUTPUT_DIR="${BASE_DIR}/outputs/${TASK_NAME}"
mkdir -p ${OUTPUT_DIR}

echo "[INFO] Task Name: ${TASK_NAME}"
echo "[INFO] Output Directory: ${OUTPUT_DIR}"

# # --------------------------------
# Run Main Script
# --------------------------------
CMD="python ${WORKING_DIR}/main_baseline.py \
    --config ${WORKING_DIR}/configs/text_baseline.yaml \
    prompt=\"${PROMPT}\" \
    save_path=${PROMPT// /_} \
    outdir=${OUTPUT_DIR} \
    seed=${SEED} \
    iter=${ITER}"

echo "[RUNNING COMMAND] $CMD"
eval $CMD

echo "[INFO] Job completed successfully."
