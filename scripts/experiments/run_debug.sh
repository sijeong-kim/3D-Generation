#!/bin/bash
set -euo pipefail

# --------------------------------
# Debug Experiment Runner
# --------------------------------
# This script is optimized for quick debugging with short iterations

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

# CUDA configuration
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
DEBUG_CONFIG=${WORKING_DIR}/configs/text_ours_debug.yaml

# Sanity checks
if [ ! -f "$HP_SCRIPT" ]; then
    echo "[ERROR] Missing hp_ours.py at $HP_SCRIPT" >&2
    exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
    echo "[ERROR] Missing base config at $BASE_CONFIG" >&2
    exit 1
fi
if [ ! -f "$DEBUG_CONFIG" ]; then
    echo "[ERROR] Missing debug config at $DEBUG_CONFIG" >&2
    exit 1
fi

# --------------------------------
# Usage Information
# --------------------------------
show_usage() {
    echo "Usage: $0 [OPTIONS] [DEBUG_EXPERIMENT_NAME]"
    echo ""
    echo "Debug Experiments Available:"
    echo "  debug_quick_test          # Quick test with minimal parameters (100 iterations)"
    echo "  debug_kernel_comparison   # Compare different kernels (200 iterations)"
    echo ""
    echo "Options:"
    echo "  --dry-run, --dry_run          Print commands without running experiments"
    echo "  --no-resume, --no_resume      Disable resume functionality and start fresh"
    echo "  --retry-failed-only, --retry_failed_only  Only retry failed experiments"
    echo "  --help, -h                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 debug_quick_test                    # Run quick debug test"
    echo "  $0 --dry-run debug_kernel_comparison   # Preview kernel comparison"
    echo ""
    echo "Default behavior: Resume from previous run if available"
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# --------------------------------
# Parse CLI args
# --------------------------------
DRY_RUN_ARG=""
NO_RESUME_ARG=""
RETRY_FAILED_ONLY_ARG=""
DEBUG_EXPERIMENT=""

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
        debug_*)
            DEBUG_EXPERIMENT="$arg"
            ;;
        *)
            echo "[WARNING] Unknown argument: $arg"
            ;;
    esac
done

# Set default experiment if none specified
if [ -z "$DEBUG_EXPERIMENT" ]; then
    DEBUG_EXPERIMENT="debug_quick_test"
    echo "[INFO] No experiment specified, using default: $DEBUG_EXPERIMENT"
fi

# Validate experiment name
VALID_EXPERIMENTS=("debug_quick_test" "debug_kernel_comparison")
if [[ ! " ${VALID_EXPERIMENTS[@]} " =~ " ${DEBUG_EXPERIMENT} " ]]; then
    echo "[ERROR] Invalid debug experiment: $DEBUG_EXPERIMENT"
    echo "Valid experiments: ${VALID_EXPERIMENTS[*]}"
    exit 1
fi

# --------------------------------
# Run debug experiment
# --------------------------------
echo "========== DEBUG EXPERIMENT RUN =========="
echo "Experiment    : ${DEBUG_EXPERIMENT}"
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

# Create output directory
mkdir -p "${BASE_DIR}/debug/${DEBUG_EXPERIMENT}"

# Save environment info
LOG_DIR=${BASE_DIR}/debug/${DEBUG_EXPERIMENT}/logs
mkdir -p "$LOG_DIR"

echo "[INFO] Dumping environment info..."
echo "Date: $(date)" > "${LOG_DIR}/env_info.txt"
echo "User: ${USER:-unknown}" >> "${LOG_DIR}/env_info.txt"
echo "Host: $(hostname)" >> "${LOG_DIR}/env_info.txt"
echo "Experiment: ${DEBUG_EXPERIMENT}" >> "${LOG_DIR}/env_info.txt"
echo "" >> "${LOG_DIR}/env_info.txt"

# System / GPU info
uname -a >> "${LOG_DIR}/env_info.txt"
nvidia-smi >> "${LOG_DIR}/nvidia-smi.txt" 2>/dev/null || true

# Python / CUDA / Torch versions
${PYTHON_BIN} -c "import torch, sys; print('Python', sys.version); print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())" \
    > "${LOG_DIR}/torch_info.txt"

# Copy configs used
cp "${BASE_CONFIG}" "${LOG_DIR}/"
cp "${DEBUG_CONFIG}" "${LOG_DIR}/"

# --------------------------------
# Run experiment
# --------------------------------
cd ${WORKING_DIR}

echo ""
echo "Starting DEBUG EXPERIMENT..."
echo "Experiment: ${DEBUG_EXPERIMENT}"
echo "This is a debug run with short iterations for quick testing"
echo ""

CMD="${PYTHON_BIN} ${HP_SCRIPT} \
    --config ${BASE_CONFIG} \
    --sweep_config ${DEBUG_CONFIG} \
    --sweep_name ${DEBUG_EXPERIMENT} \
    --outdir=${BASE_DIR}/debug"

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
eval $CMD 2>&1 | tee "${BASE_DIR}/debug/${DEBUG_EXPERIMENT}/output.out"

echo ""
echo "====================================="
echo "DEBUG EXPERIMENT completed!"
echo "Results saved in: ${BASE_DIR}/debug/${DEBUG_EXPERIMENT}"
echo "Check debug/${DEBUG_EXPERIMENT}/ for experiment results"
echo "====================================="
