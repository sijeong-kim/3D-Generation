#!/bin/bash
#SBATCH --job-name=text_to_3d_baseline
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sk2324@ic.ac.uk
#SBATCH --output=outputs/%j/output.out
#SBATCH --error=outputs/%j/error.err

# Load user shell environment
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Define paths
export USER_PATH=/vol/bitbucket/${USER}
export PROJ_HOME=${USER_PATH}/3D-Generation
export WORKING_DIR=${PROJ_HOME}/dreamgaussian
export PATH=${PROJ_HOME}/venv/bin:$PATH

# Activate virtual environment
source ${PROJ_HOME}/venv/bin/activate

# Load CUDA
export CUDA_HOME=/vol/cuda/12.4.0
source ${CUDA_HOME}/setup.sh

# PyTorch CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Cache directories
export HF_HOME=${PROJ_HOME}/.cache/huggingface
export TORCH_HOME=${PROJ_HOME}/.cache/torch
export MPLCONFIGDIR=${PROJ_HOME}/.cache/matplotlib
mkdir -p $MPLCONFIGDIR

# # Setup wandb environment
# if [ -z "${WANDB_API_KEY_GAUSSIAN}" ]; then
#     echo "[WARNING] WANDB_API_KEY_GAUSSIAN not set. Wandb will run in offline mode."
#     export WANDB_MODE="offline"
# else
#     export WANDB_API_KEY="${WANDB_API_KEY_GAUSSIAN}"
#     echo "[INFO] WANDB API key loaded from environment."
# fi

# export WANDB_PROJECT="gaussian-splatting-metrics"
# export WANDB_CONSOLE="off"




# Go to project directory
cd ${WORKING_DIR}

SEED=42
# PROMPT="a small saguaro cactus planted in a clay pot" # "A photo of a hamburger" "a campfire" "a bunny"
PROMPT="a photo of a hamburger"

TASK_NAME="${PROMPT// /_}_pure_baseline_seed_${SEED}"

# Diagnostic Info
sleep 5
echo "========== SYSTEM INFO =========="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job Name      : ${SLURM_JOB_NAME}"
echo "Task Name     : ${TASK_NAME}"
echo "Prompt        : ${PROMPT}"
echo "Run Host      : $(hostname)"
echo "Working Dir   : $(pwd)"
echo "Date & Time   : $(date)"
echo "Uptime        : $(uptime)"
echo "User          : ${USER}"
echo "CUDA Path     : ${CUDA_HOME}"
echo "PyTorch Ver   : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (PyTorch): $(python -c 'import torch; print(torch.version.cuda)')"
nvcc --version
nvidia-smi
echo "=================================="

OUTPUT_DIR="${PROJ_HOME}/outputs/${SLURM_JOB_ID}/${TASK_NAME}"
mkdir -p ${OUTPUT_DIR}
echo "Output will be saved to: ${OUTPUT_DIR}"

CMD="python ${WORKING_DIR}/main.py --config ${WORKING_DIR}/configs/text.yaml prompt=\"${PROMPT}\" save_path=${PROMPT// /_} outdir=${OUTPUT_DIR} seed=${SEED}"
# python main.py --config configs/text.yaml prompt="a photo of a hamburger" save_path=hamburger 
echo "[RUNNING COMMAND] $CMD"
eval $CMD

echo "[INFO] Job completed."