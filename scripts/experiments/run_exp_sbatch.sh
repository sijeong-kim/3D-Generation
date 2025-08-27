#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --partition=AMD7-A100-T
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sk2324@ic.ac.uk
#SBATCH --output=outputs/%j/output.out
#SBATCH --error=outputs/%j/error.err

# SLURM batch script for EXPERIMENT testing - Full parameter sweep experiments
# Usage: sbatch scripts/experiments/run_exp0_1_sbatch.sh <experiment_name>
#
# Examples:
#   sbatch scripts/experiments/run_exp0_1_sbatch.sh exp0_baseline
#   sbatch scripts/experiments/run_exp0_1_sbatch.sh exp1_1_lambda
#   sbatch scripts/experiments/run_exp0_1_sbatch.sh exp1_2_all_prompts_with_win_lambda
#
# Purpose: Run full experiments with parameter sweeps and multiple prompts

# --------------------------------
# Parse command line arguments
# --------------------------------
set -euo pipefail

# --------------------------------
# Environment & Paths
# --------------------------------
if [ -f ~/.bashrc ]; then
    # Temporarily relax nounset to avoid PS1 errors in non-interactive shells
    set +u
    source ~/.bashrc
    set -u
fi

export BASE_DIR=/vol/bitbucket/${USER}/3D-Generation
export WORKING_DIR=${BASE_DIR}
export VENV_DIR=${BASE_DIR}/venv

export PATH=${VENV_DIR}/bin:$PATH
source ${VENV_DIR}/bin/activate
PYTHON_BIN=${VENV_DIR}/bin/python

# Add .so Library Paths
export LD_LIBRARY_PATH=${VENV_DIR}/lib/python3.12/site-packages/pymeshlab/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=${VENV_DIR}/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

# --------------------------------
# CUDA Configuration
# --------------------------------
export CUDA_HOME=/vol/cuda/12.4.0
source ${CUDA_HOME}/setup.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# Memory optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Cache Directories
export HF_HOME=${BASE_DIR}/.cache/huggingface
export TORCH_HOME=${BASE_DIR}/.cache/torch
export MPLCONFIGDIR=${BASE_DIR}/.cache/matplotlib
mkdir -p $MPLCONFIGDIR

# --------------------------------
# Create output directory
# --------------------------------
# Default sweep names; can be overridden by CLI args
SWEEP_NAMES=("exp0_baseline" "exp1_lambda_coarse" "exp2_lambda_fine" "exp3_beta_rbf" "exp3_beta_cosine" "exp1_lambda_coarse_svgd" "exp1_lambda_coarse_rlsd")

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

# # --------------------------------
# # Validate experiment names
# # --------------------------------
# VALID_EXPERIMENTS=("exp0_baseline" "exp1_lambda_coarse" "exp2_lambda_fine" "exp3_beta" "exp1_lambda_coarse_svgd" "exp1_lambda_coarse_rlsd")

# for SWEEP_NAME in "${SWEEP_NAMES[@]}"; do
#     if [[ ! " ${VALID_EXPERIMENTS[@]} " =~ " ${SWEEP_NAME} " ]]; then
#         echo "[ERROR] Invalid experiment: $SWEEP_NAME"
#         echo "Valid experiments: ${VALID_EXPERIMENTS[*]}"
#         echo ""
#         echo "Available experiments from text_ours_exp.yaml:"
#         echo "  exp0_baseline              # Baseline experiment (wo repulsion)"
#         echo "  exp1_lambda_coarse         # Coarse lambda sweep"
#         echo "  exp1_lambda_coarse_svgd    # Coarse lambda sweep with SVDG repulsion"
#         echo "  exp1_lambda_coarse_rlsd    # Coarse lambda sweep with RLSD repulsion"
#         echo "  exp2_lambda_fine           # Fine lambda sweep"
#         echo "  exp3_beta                  # Beta sweep for both RBF and cosine kernels"
#         echo "  exp3_beta_rbf              # Beta sweep for RBF kernel"
#         echo "  exp3_beta_cosine           # Beta sweep for cosine kernel"        
#         echo "  exp4_num_pts               # Number of points sweep"
#         echo "  exp5_num_particles         # Number of particles sweep"
#         echo "  exp6_feature_extractor     # Feature extractor sweep"
#         echo "  exp7_feature_layer         # Feature layer sweep"
#         echo "  exp8_rbf_beta              # RBF beta sweep"
#         echo "  exp9_cosine_beta           # Cosine beta sweep"
#         echo "  exp10_cosine_eps_shift     # Cosine eps shift sweep"
#         exit 1
#     fi
# done

# Paths to scripts/configs
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
# Experiment Name
# --------------------------------
for SWEEP_NAME in "${SWEEP_NAMES[@]}"; do

    # Create output directory
    mkdir -p "${BASE_DIR}/exp/${SWEEP_NAME}"

    echo "========== SLURM JOB INFO =========="
    echo "Job ID        : ${SLURM_JOB_ID}"
    echo "Job Name      : ${SLURM_JOB_NAME}"
    echo "Experiment    : ${SWEEP_NAME}"
    echo "User          : ${USER}"
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
    echo "====================================="

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
    echo "User: ${USER}" >> "${LOG_DIR}/env_info.txt"
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

    # Run the command (output already captured by SLURM --output directive)
    eval $CMD

    echo ""
    echo "====================================="
    echo "EXPERIMENT completed!"
    echo "Results saved in: ${BASE_DIR}/exp"
    echo "Check exp/${SWEEP_NAME}/ for experiment runs"
    echo "====================================="
done