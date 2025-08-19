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
# Default sweep names; can be overridden by passing names as CLI args
export SWEEP_NAMES=("exp0_baseline" "exp1_1_lambda")

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
    # --------------------------------
    # Create output directory
    # --------------------------------
    mkdir -p ${BASE_DIR}/exp/${SWEEP_NAME}

    # --------------------------------
    # Job Information
    # --------------------------------
    echo "========== SLURM JOB INFO =========="
    echo "Job ID        : ${SLURM_JOB_ID}"
    echo "Job Name      : ${SLURM_JOB_NAME}"
    echo "Experiment    : ${SWEEP_NAME}"
    echo "User          : ${USER}"
    echo "Run Host      : $(hostname)"
    echo "Working Dir   : $(pwd)"
    echo "CUDA Path     : ${CUDA_HOME}"
    echo "Date & Time   : $(date)"
    echo "====================================="

    # Create job-specific output directory
    mkdir -p "${BASE_DIR}/exp/${SWEEP_NAME}"

    # --------------------------------
    # Run hyperparameter tuning
    # --------------------------------
    cd ${WORKING_DIR}

    echo ""
    echo "Starting EXPERIMENT..."
    echo "Experiment: ${SWEEP_NAME}"
    echo "This will run full parameter sweep experiments"
    echo "Expected time: 2-24 hours depending on experiment size"
    echo ""

    CMD="$PYTHON_BIN $HP_SCRIPT \
        --config $BASE_CONFIG \
        --sweep_config $SWEEP_CONFIG \
        --sweep_name ${SWEEP_NAME} \
        --outdir=${BASE_DIR}/exp"

    # Append dry-run flag if requested
    if [ -n "$DRY_RUN_ARG" ]; then
        CMD+=" $DRY_RUN_ARG"
    fi

    echo "[RUNNING COMMAND] $CMD"
    echo ""

    eval $CMD

    echo ""
    echo "====================================="
    echo "EXPERIMENT completed!"
    echo "Results saved in: ${BASE_DIR}/exp"
    echo "Check exp/${SWEEP_NAME}/ for experiment runs"
    echo "====================================="
done