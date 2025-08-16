#!/bin/bash
set -e  # Stop on error

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR=/workspace/3D-Generation   
VENV_DIR="${BASE_DIR}/venv"
CUDA_VERSION="12.1"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# -----------------------------
# 1. Virtual Environment Setup
# -----------------------------
echo "[Step 1] Creating Virtual Environment..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# -----------------------------
# 2. CUDA Environment Setup
# -----------------------------
echo "[Step 2] Loading CUDA ${CUDA_VERSION}..."
export CUDA_HOME=${CUDA_HOME}
source ${CUDA_HOME}/setup.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# -----------------------------
# 3. Python Dependencies
# -----------------------------
echo "[Step 3] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Jupyter Kernel (for local development)
python -m ipykernel install --user --name=venv

# -----------------------------
# 4. Build C++/CUDA Extensions (setup.py install)
# -----------------------------
echo "[Step 4] Building diff-gaussian-rasterization (setup.py install)..."
if [ ! -d "diff-gaussian-rasterization" ]; then
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
fi
cd diff-gaussian-rasterization; python setup.py install; cd ..

echo "[Step 4] Building simple-knn (setup.py install)..."
if [ ! -d "simple-knn" ]; then
    git clone https://github.com/camenduru/simple-knn
fi
cd simple-knn; python setup.py install; cd ..

# -----------------------------
# 5. Install External Libraries (via GitHub)
# -----------------------------
echo "[Step 5] Installing external libraries (git)..."
pip install git+https://github.com/NVlabs/nvdiffrast/ git+https://github.com/ashawkey/kiuikit git+https://github.com/bytedance/MVDream git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream

# -----------------------------
# Setup Complete
# -----------------------------
echo "====================================="
echo " ✅ Environment setup is COMPLETE!"
echo " To activate the virtual environment:"
echo "   source ${VENV_DIR}/bin/activate"
echo "====================================="
