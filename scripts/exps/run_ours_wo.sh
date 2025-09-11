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

# PROMPT="a photo of a hamburger"
# PROMPT="a small saguaro cactus planted in a clay pot"
# PROMPT="a photo of a tulip"
# PROMPT="a photo of an ice cream"

# PROMPT_KEY="hamburger"

# ITERS=500
ITERS=800
SCHEDULE_ITERS=1000

REPULSION_TYPE="wo" # "rlsd" "svgd"
KERNEL_TYPE="none" # "rbf" "cosine"


for PROMPT_IDX in "${!PROMPTS[@]}"; do
    PROMPT=${PROMPTS[${PROMPT_IDX}]}
    EVAL_RADIUS=${EVAL_RADIUS[${PROMPT_IDX}]}
    
    TASK_NAME="${PROMPT// /_}__${REPULSION_TYPE}__${SCHEDULE_ITERS}"
    # OUTPUT_DIR="${BASE_DIR}/outputs/${SLURM_JOB_ID}/${TASK_NAME}"
    OUTPUT_DIR="${BASE_DIR}/exp/exp0_baseline/${TASK_NAME}"
    mkdir -p ${OUTPUT_DIR}

    echo "[INFO] Task Name: ${TASK_NAME}"
    echo "[INFO] Output Directory: ${OUTPUT_DIR}"

    if [[ -f "${OUTPUT_DIR}/.done" ]]; then
        echo "[WARNING] Experiment already completed, skipping..."
        continue
    fi

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
        total_schedule_iters=${SCHEDULE_ITERS} \
        eval_radius=${EVAL_RADIUS}"
        # stderr, stdout to file

    echo "[RUNNING COMMAND] $CMD"

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