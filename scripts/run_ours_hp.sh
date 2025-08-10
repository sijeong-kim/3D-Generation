#!/bin/bash
#SBATCH --job-name=hp
#SBATCH --partition=AMD7-A100-T 
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk2324@ic.ac.uk
#SBATCH --output=outputs/%j/output.out
#SBATCH --error=outputs/%j/error.err

# SLURM batch script for hyperparameter tuning
# Usage: sbatch scripts/run_hp_slurm.sh <sweep_name> [prompts...]
#
# Examples:
#   sbatch scripts/run_hp_slurm.sh lamdda_repulsion_coarse
#   sbatch scripts/run_hp_slurm.sh lamdda_repulsion_coarse "a hamburger" "a cactus"
#
# Note: Seeds are now defined in the YAML configuration file

# --------------------------------
# Parse command line arguments
# --------------------------------
SWEEP_NAME="$1"
shift  # Remove first argument

# Parse remaining arguments (just prompts now)
PROMPTS=()

for arg in "$@"; do
    PROMPTS+=("$arg")
done

# Set defaults if not provided
if [ ${#PROMPTS[@]} -eq 0 ]; then
    PROMPTS=("a photo of a hamburger")
fi

# Validate sweep name
if [ -z "$SWEEP_NAME" ]; then
    echo "[ERROR] Sweep name is required as first argument"
    echo "Usage: sbatch $0 <sweep_name> [prompts...]"
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

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
echo "Sweep Name    : ${SWEEP_NAME}"
echo "Prompts       : ${PROMPTS[*]}"
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
echo "Starting hyperparameter tuning..."
echo "Sweep: ${SWEEP_NAME}"
echo ""

CMD="python ${WORKING_DIR}/hp_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    --sweep_config ${WORKING_DIR}/configs/text_ours_hp.yaml \
    --sweep_name ${SWEEP_NAME} \
    --prompts ${PROMPTS[*]} \
    --outdir=${BASE_DIR}/logs"

echo "[RUNNING COMMAND] $CMD"
echo ""

eval $CMD

echo ""
echo "====================================="
echo "Hyperparameter tuning completed!"
echo "Results saved in: ${BASE_DIR}/logs"
echo "====================================="
