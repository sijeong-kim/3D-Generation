#!/bin/bash
set -euo pipefail

# --------------------------------
# Usage Information
# --------------------------------
show_usage() {
    echo "Usage: $0 [OPTIONS] [SWEEP_NAMES...]"
    echo ""
    echo "Available Experiments (from text_ours_exp.yaml):"
    echo "  exp0_baseline              # Baseline experiment (wo repulsion)"
    echo "  exp1_lambda_coarse_svgd    # Coarse lambda sweep with SVGD"
    echo "  exp1_lambda_coarse_rlsd    # Coarse lambda sweep with RLSd"
    echo "  exp2_lambda_fine_rbf       # Fine lambda sweep with RBF kernel"
    echo "  exp2_lambda_fine_cosine    # Fine lambda sweep with cosine kernel"
    echo ""
    echo "Options:"
    echo "  --dry-run, --dry_run          Print commands without running experiments"
    echo "  --no-resume, --no_resume      Disable resume functionality and start fresh"
    echo "  --retry-failed-only, --retry_failed_only  Only retry failed experiments"
    echo "  --help, -h                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 exp0_baseline                    # Run specific experiment (resumes automatically)"
    echo "  $0 --dry-run exp1_lambda_coarse     # Dry run to see what would be executed"
    echo "  $0 --retry-failed-only exp2_lambda_fine # Only retry failed experiments"
    echo "  $0 --no-resume exp3_beta            # Start fresh, ignore existing results"
    echo ""
    echo "Default behavior: Resume from previous run if available"
    echo "Default sweep names: exp0_baseline exp1_lambda_coarse"
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

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

# Set environment variables for local model loading
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --------------------------------
# Default sweep names; can be overridden by CLI args
# --------------------------------
SWEEP_NAMES=("exp0_baseline" "exp1_lambda_coarse_svgd" "exp1_lambda_coarse_rlsd" "exp2_lambda_fine_rbf" "exp2_lambda_fine_cosine")

# Parse CLI args: allow various flags and optional sweep names
DRY_RUN_ARG=""
NO_RESUME_ARG=""
RETRY_FAILED_ONLY_ARG=""
CUSTOM_SWEEPS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|--dry_run)
            DRY_RUN_ARG="--dry_run"
            ;;
        --no-resume|--no_resume)
            NO_RESUME_ARG="--no_resume"
            ;;
        --retry-failed-only|--retry_failed_only)
            RETRY_FAILED_ONLY_ARG="--retry_failed_only"
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
# Validate experiment names
# --------------------------------
VALID_EXPERIMENTS=("exp0_baseline" "exp1_lambda_coarse_svgd" "exp1_lambda_coarse_rlsd" "exp2_lambda_fine_rbf" "exp2_lambda_fine_cosine")

for SWEEP_NAME in "${SWEEP_NAMES[@]}"; do
    if [[ ! " ${VALID_EXPERIMENTS[@]} " =~ " ${SWEEP_NAME} " ]]; then
        echo "[ERROR] Invalid experiment: $SWEEP_NAME"
        echo "Valid experiments: ${VALID_EXPERIMENTS[*]}"
        echo ""
        echo "Available experiments from text_ours_exp.yaml:"
        echo "  exp0_baseline              # Baseline experiment (wo repulsion)"
        echo "  exp1_lambda_coarse_svgd    # Coarse lambda sweep with SVGD"
        echo "  exp1_lambda_coarse_rlsd    # Coarse lambda sweep with RLSd"
        echo "  exp2_lambda_fine_rbf       # Fine lambda sweep with RBF kernel"
        echo "  exp2_lambda_fine_cosine    # Fine lambda sweep with cosine kernel"
        exit 1
    fi
done

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
    if [ -n "$DRY_RUN_ARG" ]; then
        echo "Mode          : DRY RUN (no experiments will be executed)"
    fi
    if [ -n "$NO_RESUME_ARG" ]; then
        echo "Mode          : FRESH START (ignoring existing results)"
    else
        echo "Mode          : RESUME ENABLED (default)"
    fi
    if [ -n "$RETRY_FAILED_ONLY_ARG" ]; then
        echo "Mode          : RETRY FAILED ONLY"
    fi
    echo "=========================================="

    # Show experiment info for dry-run
    if [ -n "$DRY_RUN_ARG" ]; then
        echo ""
        echo "DRY RUN - Experiment Information:"
        echo "  Experiment: $SWEEP_NAME"
        echo "  Config file: $SWEEP_CONFIG"
        echo "  Base config: $BASE_CONFIG"
        echo "  Output dir: ${BASE_DIR}/exp/${SWEEP_NAME}"
        echo ""
        echo "  To see parameter combinations, run:"
        echo "  ${PYTHON_BIN} ${HP_SCRIPT} --config ${BASE_CONFIG} --sweep_config ${SWEEP_CONFIG} --sweep_name ${SWEEP_NAME} --dry_run"
        echo ""
        echo "  DRY RUN MODE - Skipping actual execution"
        echo ""
        continue
    fi

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

    # Append flags if requested
    if [ -n "$DRY_RUN_ARG" ]; then
        CMD+=" $DRY_RUN_ARG"
    fi
    if [ -n "$NO_RESUME_ARG" ]; then
        CMD+=" $NO_RESUME_ARG"
    fi
    if [ -n "$RETRY_FAILED_ONLY_ARG" ]; then
        CMD+=" $RETRY_FAILED_ONLY_ARG"
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
