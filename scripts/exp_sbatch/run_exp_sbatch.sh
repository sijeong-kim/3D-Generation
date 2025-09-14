#!/bin/bash
#SBATCH --job-name=exp_interactive
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sk2324@ic.ac.uk
#SBATCH --output=outputs/%j/exp_output.out
#SBATCH --error=outputs/%j/exp_error.err

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
# Experiment Parameters
# --------------------------------
# Default parameters - can be overridden by command line arguments
SWEEP_NAME="default_sweep"
OUTDIR="exp"
BASE_CONFIG="configs/text_ours.yaml"
SWEEP_CONFIG="configs/text_ours_exp.yaml"
GPUS="0"
CPU_CORES="0-7"
THREADS="8"
MEM_SOFT_MB="32768"
TIMEOUT="14400"
SLEEP_BETWEEN="60"

# Parse command line arguments if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep_name) SWEEP_NAME="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --base_config) BASE_CONFIG="$2"; shift 2 ;;
    --sweep_config) SWEEP_CONFIG="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpu_cores) CPU_CORES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --mem_soft_mb) MEM_SOFT_MB="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --sleep_between) SLEEP_BETWEEN="$2"; shift 2 ;;
    --help)
      echo "Usage: sbatch run_exp_sbatch.sh [OPTIONS]"
      echo "Options:"
      echo "  --sweep_name NAME     Experiment sweep name (default: default_sweep)"
      echo "  --outdir DIR          Output directory (default: exp)"
      echo "  --base_config PATH    Base config file (default: configs/text_ours.yaml)"
      echo "  --sweep_config PATH   Sweep config file (default: configs/text_ours_exp.yaml)"
      echo "  --gpus CSV            GPU list (default: 0)"
      echo "  --cpu_cores STR       CPU cores (default: 0-7)"
      echo "  --threads N           Number of threads (default: 8)"
      echo "  --mem_soft_mb MB      Memory limit in MB (default: 32768)"
      echo "  --timeout SEC         Timeout in seconds (default: 14400)"
      echo "  --sleep_between SEC   Sleep between experiments (default: 60)"
      echo "  --help                Show this help"
      exit 0
      ;;
    *)
      if [[ -z "$SWEEP_NAME" || "$SWEEP_NAME" == "default_sweep" ]]; then
        SWEEP_NAME="$1"
      else
        echo "Unknown option: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

# --------------------------------
# Diagnostic Info
# --------------------------------
echo "========== SLURM JOB INFO =========="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Job Name      : ${SLURM_JOB_NAME}"
echo "Sweep Name    : ${SWEEP_NAME}"
echo "Output Dir    : ${OUTDIR}"
echo "Base Config   : ${BASE_CONFIG}"
echo "Sweep Config  : ${SWEEP_CONFIG}"
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

# --------------------------------
# Change to working directory
# --------------------------------
cd ${WORKING_DIR}

# --------------------------------
# Run Experiment Script
# --------------------------------
echo "Starting experiment runner..."
echo "Command: ./scripts/experiments/run_exp_interactive.sh ${SWEEP_NAME} --outdir ${OUTDIR} --base_config ${BASE_CONFIG} --sweep_config ${SWEEP_CONFIG} --gpus ${GPUS} --cpu_cores ${CPU_CORES} --threads ${THREADS} --mem_soft_mb ${MEM_SOFT_MB} --timeout ${TIMEOUT} --sleep_between ${SLEEP_BETWEEN}"

./scripts/experiments/run_exp_interactive.sh \
  "${SWEEP_NAME}" \
  --outdir "${OUTDIR}" \
  --base_config "${BASE_CONFIG}" \
  --sweep_config "${SWEEP_CONFIG}" \
  --gpus "${GPUS}" \
  --cpu_cores "${CPU_CORES}" \
  --threads "${THREADS}" \
  --mem_soft_mb "${MEM_SOFT_MB}" \
  --timeout "${TIMEOUT}" \
  --sleep_between "${SLEEP_BETWEEN}"

echo "[INFO] Experiment job completed."
