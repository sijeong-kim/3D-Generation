#!/bin/bash
#SBATCH --job-name=ours_hp_layers
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk2324@ic.ac.uk
#SBATCH --output=outputs/%j/output.out
#SBATCH --error=outputs/%j/error.err

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
# export LD_LIBRARY_PATH=${VENV_DIR}/lib/python3.12/site-packages:$LD_LIBRARY_PATH

# Ensure Python uses correct site-packages
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
# Diagnostic Info
# --------------------------------
echo "========== SLURM JOB INFO =========="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job Name      : ${SLURM_JOB_NAME}"
echo "User          : ${USER}"
echo "Run Host      : $(hostname)"
echo "Working Dir   : $(pwd)"
echo "CUDA Path     : ${CUDA_HOME}"
echo "Date & Time   : $(date)"
echo "PyTorch Ver   : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (PyTorch): $(python -c 'import torch; print(torch.version.cuda)')"
echo "nvcc Version  : $(nvcc --version | grep release)"
nvidia-smi
echo "====================================="
echo "Output will be saved to: outputs/${SLURM_JOB_ID}/"
echo "====================================="

# --------------------------------
# Run Hyperparameter Tuning Script
# --------------------------------

# Set output directory for the hyperparameter tuning
OUTPUT_DIR="${BASE_DIR}/outputs/${SLURM_JOB_ID}"
mkdir -p ${OUTPUT_DIR}

# Run hp.py with the config file and custom output directory
CMD="python ${WORKING_DIR}/hp_layers.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    outdir=${OUTPUT_DIR}"

echo "[RUNNING COMMAND] $CMD"
eval $CMD

echo "[INFO] Hyperparameter tuning completed successfully."