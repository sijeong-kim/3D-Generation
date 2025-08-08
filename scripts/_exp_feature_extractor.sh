#!/bin/bash
#SBATCH --job-name=feature_extractor_test
#SBATCH --partition=gpgpuB
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
# Run Parameters
# --------------------------------
SEED=42
MODEL_NAME="facebook/dinov2-base"  # Options: dinov2-small, dinov2-base, dinov2-large, dinov2-giant
TEST_IMAGES_PATH="${BASE_DIR}/experiments/feature_from_diff_layers/images1"  # Optional: path to test images
SAVE_ATTENTION_MAPS=true
SAVE_FEATURES=true

TASK_NAME="feature_extractor_${MODEL_NAME//\//_}_seed_${SEED}"
OUTPUT_DIR="${BASE_DIR}/outputs/${SLURM_JOB_ID}/${TASK_NAME}"

mkdir -p ${OUTPUT_DIR}
# mkdir -p ${OUTPUT_DIR}/extracted_data

# --------------------------------
# Diagnostic Info
# --------------------------------
echo "========== SLURM JOB INFO =========="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Job Name       : ${SLURM_JOB_NAME}"
echo "Model Name     : ${MODEL_NAME}"
echo "Task Name      : ${TASK_NAME}"
echo "Seed           : ${SEED}"
echo "Save Features  : ${SAVE_FEATURES}"
echo "Save Attn Maps : ${SAVE_ATTENTION_MAPS}"
echo "Test Images    : ${TEST_IMAGES_PATH}"
echo "User           : ${USER}"
echo "Run Host       : $(hostname)"
echo "Working Dir    : $(pwd)"
echo "CUDA Path      : ${CUDA_HOME}"
echo "Date & Time    : $(date)"
echo "PyTorch Ver    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (PyTorch) : $(python -c 'import torch; print(torch.version.cuda)')"
echo "nvcc Version   : $(nvcc --version | grep release)"
nvidia-smi
echo "====================================="
echo "Output will be saved to: ${OUTPUT_DIR}"
echo "====================================="

# --------------------------------
# Run Main Scripts
# --------------------------------

# Step 1: Extract features and attention maps using feature_extractor.py
echo "========== STEP 1: Feature Extraction =========="
CMD1="python ${WORKING_DIR}/feature_extractor.py \
    --image_dir ${TEST_IMAGES_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_name ${MODEL_NAME} \
    --device cuda"

# Add optional flags based on settings
if [ "$SAVE_FEATURES" = true ]; then
    CMD1="$CMD1 --save_features"
fi

if [ "$SAVE_ATTENTION_MAPS" = true ]; then
    CMD1="$CMD1 --save_attention_maps"
fi

echo "[RUNNING COMMAND] $CMD1"
eval $CMD1

if [ $? -ne 0 ]; then
    echo "[ERROR] Feature extraction failed!"
    exit 1
fi

echo "[INFO] Feature extraction completed successfully."
echo ""

# Step 2: Create visualizations using plot_features_attention.py
echo "========== STEP 2: Visualization =========="
CMD2="python ${WORKING_DIR}/plot_features_attention.py \
    --image_dir ${TEST_IMAGES_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_name ${MODEL_NAME} \
    --device cuda"

echo "[RUNNING COMMAND] $CMD2"
eval $CMD2

if [ $? -ne 0 ]; then
    echo "[ERROR] Visualization failed!"
    exit 1
fi

echo "[INFO] Visualization completed successfully."
echo ""

# --------------------------------
# Summary
# --------------------------------
echo "========== JOB SUMMARY =========="
echo "✅ Feature extraction completed"
echo "✅ Visualizations generated"
echo ""
echo "📁 Output files saved to:"
echo "   ${OUTPUT_DIR}/"
if [ "$SAVE_FEATURES" = true ]; then
    echo "   ├── early_features.npy"
    echo "   ├── mid_features.npy"
    echo "   └── last_features.npy"
fi
if [ "$SAVE_ATTENTION_MAPS" = true ]; then
    echo "   ├── early_attention.npy"
    echo "   ├── mid_attention.npy"
    echo "   └── last_attention.npy"
fi
echo "   ├── tsne_features_multilayer.png"
echo "   └── attention_maps_multilayer.png"
echo ""
echo "[INFO] Job completed successfully."
