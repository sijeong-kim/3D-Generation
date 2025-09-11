export BASE_DIR=/workspace/3D-Generation-fixed
export WORKING_DIR=${BASE_DIR}
export VENV_DIR=${BASE_DIR}/venv

# export PATH=${VENV_DIR}/bin:$PATH
# source ${VENV_DIR}/bin/activate

# --------------------------------
# Run Parameters
# --------------------------------
SEED=42

PROMPTS=(
    "a photo of a hamburger"
    "a small saguaro cactus planted in a clay pot"
    "a photo of a tulip"
    "a photo of an ice cream"
)

EVAL_RADIUS=(
    3.0
    4.0
    4.0
    4.0
)

PROMPT_IDX=1

PROMPT=${PROMPTS[${PROMPT_IDX}]}
EVAL_RADIUS=${EVAL_RADIUS[${PROMPT_IDX}]}

ITERS=800   


REPULSION_TYPES=("rlsd")
KERNEL_TYPES=("rbf")

LAMBDA_REPULSION=(600 800 1000 1200 1400)

for REPULSION_TYPE in "${REPULSION_TYPES[@]}"; do
    for KERNEL_TYPE in "${KERNEL_TYPES[@]}"; do
        for LAMBDA_REPULSION in "${LAMBDA_REPULSION[@]}"; do
            TASK_NAME="${PROMPT// /_}__${REPULSION_TYPE}__${KERNEL_TYPE}__${LAMBDA_REPULSION}__${ITERS}"
            OUTPUT_DIR="${BASE_DIR}/exp/exp2_lambda_repulsion_${REPULSION_TYPE}_${KERNEL_TYPE}/${TASK_NAME}"
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
                --config ${WORKING_DIR}/configs/text_ours.yaml \
                prompt=\"${PROMPT}\" \
                save_path=${PROMPT// /_} \
                outdir=${OUTPUT_DIR} \
                seed=${SEED} \
                iters=${ITERS} \
                eval_radius=${EVAL_RADIUS}"



            if [ "$REPULSION_TYPE" != "wo" ]; then
                CMD="${CMD} repulsion_type=${REPULSION_TYPE}"
                CMD="${CMD} lambda_repulsion=${LAMBDA_REPULSION}"
                CMD="${CMD} kernel_type=${KERNEL_TYPE}"
            fi

            echo "[RUNNING COMMAND] $CMD"
            eval $CMD

            # Run command and save output/error to files while showing in terminal
            eval $CMD > >(tee "${OUTPUT_DIR}/stdout.log") 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)
            local exit_code=$?

            if [ $exit_code -ne 0 ]; then
                echo "[ERROR] Command failed with exit code: $exit_code"
                exit 1
            else
                echo "[INFO] Job completed successfully."
                # make .done file
                echo "Task completed: ${TASK_NAME}" > "${OUTPUT_DIR}/.done"
                echo "Completed at: $(date)" >> "${OUTPUT_DIR}/.done"
            fi

            sleep 10

        done
    done
done