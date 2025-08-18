#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sk2324@ic.ac.uk
#SBATCH --output=outputs/%j/output.out
#SBATCH --error=outputs/%j/error.err
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# SLURM batch script for hyperparameter tuning experiments
# Usage: sbatch scripts/run_ours_exp_slurm.sh <experiment_name>
#
# Examples:
#   sbatch scripts/run_ours_exp_slurm.sh exp1_repulsion_lambda_sweep
#   sbatch scripts/run_ours_exp_slurm.sh exp1_wo_method

# Parse command line arguments
EXPERIMENT_NAME="$1"

# Validate experiment name
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "[ERROR] Experiment name is required as first argument"
    echo "Usage: sbatch $0 <experiment_name>"
    echo ""
    echo "Available experiments:"
    echo "  - exp1_repulsion_lambda_sweep"
    echo "  - exp1_wo_method"
    exit 1
fi

# --------------------------------
# Environment & Paths
# --------------------------------
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

export BASE_DIR=/vol/bitbucket/${USER}/3D-Generation
export WORKING_DIR=${BASE_DIR}
export VENV_DIR=${BASE_DIR}/venv

export PATH=${VENV_DIR}/bin:$PATH
source ${VENV_DIR}/bin/activate

# Add .so Library Paths
export LD_LIBRARY_PATH=${VENV_DIR}/lib/python3.12/site-packages/pymeshlab/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${VENV_DIR}/lib/python3.12/site-packages:$PYTHONPATH

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
# Job Information
# --------------------------------
echo "========== SLURM JOB INFO =========="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job Name      : ${SLURM_JOB_NAME}"
echo "Experiment    : ${EXPERIMENT_NAME}"
echo "User          : ${USER}"
echo "Run Host      : $(hostname)"
echo "Working Dir   : $(pwd)"
echo "CUDA Path     : ${CUDA_HOME}"
echo "Date & Time   : $(date)"
echo "====================================="

# Create job-specific output directory
mkdir -p "${BASE_DIR}/outputs/${SLURM_JOB_ID}"

# --------------------------------
# Run hyperparameter tuning
# --------------------------------
cd ${WORKING_DIR}

echo ""
echo "Starting hyperparameter tuning experiment..."
echo "Experiment: ${EXPERIMENT_NAME}"
echo ""

CMD="python ${WORKING_DIR}/hp_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    --sweep_config ${WORKING_DIR}/configs/text_ours_exp.yaml \
    --sweep_name ${EXPERIMENT_NAME} \
    --outdir=${BASE_DIR}/logs"

echo "[RUNNING COMMAND] $CMD"
echo ""

eval $CMD

echo ""
echo "====================================="
echo "Experiment completed!"
echo "Results saved in: ${BASE_DIR}/logs/${EXPERIMENT_NAME}"
echo "====================================="
