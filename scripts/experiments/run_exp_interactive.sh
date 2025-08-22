#!/bin/bash
set -euo pipefail

# --------------------------------
# Environment & Paths (match setup_interactive.sh)
# --------------------------------
BASE_DIR=/workspace/3D-Generation
WORKING_DIR=${BASE_DIR}
VENV_DIR=${BASE_DIR}/venv

# Activate virtual environment
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="${VENV_DIR}/bin/python"
else
    echo "[ERROR] Missing virtual environment at ${VENV_DIR}. Please run scripts/envs/setup_interactive.sh" >&2
    exit 1
fi

# CUDA configuration (from setup_interactive.sh)
CUDA_VERSION="12.1"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export CUDA_HOME
if [ -f "${CUDA_HOME}/setup.sh" ]; then
    source ${CUDA_HOME}/setup.sh
fi
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# --------------------------------
# Paths to scripts/configs
# --------------------------------
HP_SCRIPT=${WORKING_DIR}/hp_ours.py
BASE_CONFIG=${WORKING_DIR}/configs/text_ours.yaml
SWEEP_CONFIG=${WORKING_DIR}/configs/text_ours_exp.yaml

# Sanity checks
if [ ! -f "$HP_SCRIPT" ]; then
    echo "[ERROR] Missing hp_ours.py at $HP_SCRIPT" >&2
    exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
    echo "[ERROR] Missing base config at $BASE_CONFIG" >&2
    exit 1
fi
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "[ERROR] Missing sweep config at $SWEEP_CONFIG" >&2
    exit 1
fi

# --------------------------------
# Default sweep names; can be overridden by CLI args
# --------------------------------
SWEEP_NAMES=("exp0_baseline" "exp1_1_lambda")

# Parse CLI args: allow --dry-run/--dry_run and optional sweep names
DRY_RUN_ARG=""
CUSTOM_SWEEPS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|--dry_run)
            DRY_RUN_ARG="--dry_run"
            ;;
        *)
            CUSTOM_SWEEPS+=("$arg")
            ;;
    esac
done

if [ "${#CUSTOM_SWEEPS[@]}" -gt 0 ]; then
    SWEEP_NAMES=("${CUSTOM_SWEEPS[@]}")
fi

# --------------------------------
# Run experiments
# --------------------------------
for SWEEP_NAME in "${SWEEP_NAMES[@]}"; do

    # Create output directory
    mkdir -p "${BASE_DIR}/exp/${SWEEP_NAME}"

    echo "========== INTERACTIVE RUN INFO =========="
    echo "Experiment    : ${SWEEP_NAME}"
    echo "User          : ${USER:-unknown}"
    echo "Run Host      : $(hostname)"
    echo "Working Dir   : ${WORKING_DIR}"
    echo "CUDA Path     : ${CUDA_HOME}"
    echo "Python        : $(${PYTHON_BIN} --version 2>&1)"
    echo "Date & Time   : $(date)"
    echo "=========================================="

    # --------------------------------
    # Save environment info
    # --------------------------------
    LOG_DIR=${BASE_DIR}/exp/${SWEEP_NAME}/logs
    mkdir -p "$LOG_DIR"

    # ---- Save environment info ----
    echo "[INFO] Dumping environment info..."
    echo "Date: $(date)" > "${LOG_DIR}/env_info.txt"
    echo "User: ${USER:-unknown}" >> "${LOG_DIR}/env_info.txt"
    echo "Host: $(hostname)" >> "${LOG_DIR}/env_info.txt"
    echo "" >> "${LOG_DIR}/env_info.txt"

    # System / GPU
    uname -a >> "${LOG_DIR}/env_info.txt"
    nvidia-smi >> "${LOG_DIR}/nvidia-smi.txt" 2>/dev/null || true

    # Python / CUDA / Torch versions
    ${PYTHON_BIN} -c "import torch, sys; print('Python', sys.version); print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())" \
        > "${LOG_DIR}/torch_info.txt"

    # Package versions
    pip freeze > "${LOG_DIR}/requirements.txt"

    # Git state (if repo)
    if [ -d "${BASE_DIR}/.git" ]; then
        git -C "${BASE_DIR}" rev-parse HEAD > "${LOG_DIR}/git_commit.txt"
        git -C "${BASE_DIR}" diff > "${LOG_DIR}/git_diff.patch" || true
    fi

    # Copy configs used
    cp "${BASE_CONFIG}" "${LOG_DIR}/"
    cp "${SWEEP_CONFIG}" "${LOG_DIR}/"

    # --------------------------------
    # Run experiments
    # --------------------------------
    cd ${WORKING_DIR}

    echo ""
    echo "Starting EXPERIMENT..."
    echo "Experiment: ${SWEEP_NAME}"
    echo "This will run full parameter sweep experiments"
    echo ""

    CMD="${PYTHON_BIN} ${HP_SCRIPT} \
        --config ${BASE_CONFIG} \
        --sweep_config ${SWEEP_CONFIG} \
        --sweep_name ${SWEEP_NAME} \
        --outdir=${BASE_DIR}/exp"

    # Append dry-run flag if requested
    if [ -n "$DRY_RUN_ARG" ]; then
        CMD+=" $DRY_RUN_ARG"
    fi

    echo "[RUNNING COMMAND] $CMD"
    echo ""

    # Run the command and save output to file in real time
    eval $CMD 2>&1 | tee "${BASE_DIR}/exp/${SWEEP_NAME}/output.out"

    echo ""
    echo "====================================="
    echo "EXPERIMENT completed!"
    echo "Results saved in: ${BASE_DIR}/exp"
    echo "Check exp/${SWEEP_NAME}/ for experiment runs"
    echo "====================================="
done
