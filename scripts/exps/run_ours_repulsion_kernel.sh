export BASE_DIR=/workspace/3D-Generation-fixed
export WORKING_DIR=${BASE_DIR}
export VENV_DIR=${BASE_DIR}/venv

# export PATH=${VENV_DIR}/bin:$PATH
# source ${VENV_DIR}/bin/activate

# --------------------------------
# Run Parameters
# --------------------------------
SEED=42
PROMPT="a photo of a hamburger"
# PROMPT="a small saguaro cactus planted in a clay pot"
# PROMPT="a photo of a tulip"
# PROMPT="a photo of an ice cream"

# PROMPT_KEY="hamburger"

ITERS=1500


REPULSION_TYPES=("rlsd" "svgd")
KERNEL_TYPES=("rbf" "cosine")

LAMBDA_REPULSION=1000

for REPULSION_TYPE in "${REPULSION_TYPES[@]}"; do
    for KERNEL_TYPE in "${KERNEL_TYPES[@]}"; do
        TASK_NAME="${PROMPT// /_}__${REPULSION_TYPE}__${KERNEL_TYPE}__${LAMBDA_REPULSION}"
        OUTPUT_DIR="${BASE_DIR}/outputs/${TASK_NAME}"
        mkdir -p ${OUTPUT_DIR}

        if [[ -f "${OUTPUT_DIR}/.done" ]]; then
            echo "[WARNING] Experiment already completed, skipping..."
            continue
        fi


        echo "[INFO] Task Name: ${TASK_NAME}"
        echo "[INFO] Output Directory: ${OUTPUT_DIR}"

        # # --------------------------------
        # Run Main Script
        # --------------------------------
        CMD="python ${WORKING_DIR}/main_ours.py \
            --config ${WORKING_DIR}/configs/text_ours_${REPULSION_TYPE}_${KERNEL_TYPE}.yaml \
            prompt=\"${PROMPT}\" \
            save_path=${PROMPT// /_} \
            outdir=${OUTPUT_DIR} \
            seed=${SEED} \
            iters=${ITERS} \
            > ${OUTPUT_DIR}/output.log 2>&1"


        if [ "$REPULSION_TYPE" != "wo" ]; then
            CMD="${CMD} lambda_repulsion=${LAMBDA_REPULSION}"
        fi

        echo "[RUNNING COMMAND] $CMD"
        eval $CMD

        echo "[INFO] Job completed successfully."

        # make .done file
        echo "Task completed: ${TASK_NAME}" > "${OUTPUT_DIR}/.done"
        echo "Completed at: $(date)" >> "${OUTPUT_DIR}/.done"

        sleep 60

    done
done
