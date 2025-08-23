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
cd "${BASE_DIR}"

# -----------------------------
# 2. CUDA Environment Setup
# -----------------------------
echo "[Step 2] Locating CUDA toolkit..."

# If a default CUDA_HOME is set and valid, use it; otherwise, try to auto-detect
if [ -x "${CUDA_HOME}/bin/nvcc" ]; then
	echo "Using CUDA at ${CUDA_HOME}"
else
	# Try to find nvcc in PATH first
	if command -v nvcc >/dev/null 2>&1; then
		NVCC_PATH="$(command -v nvcc)"
		CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
		export CUDA_HOME
		echo "Detected CUDA at ${CUDA_HOME}"
	else
		# Probe common installation locations (prefer highest version)
		CANDIDATES=$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V -r; ls -d /usr/local/cuda/ 2>/dev/null || true)
		for dir in $CANDIDATES; do
			if [ -x "${dir%/}/bin/nvcc" ]; then
				CUDA_HOME="${dir%/}"
				export CUDA_HOME
				echo "Detected CUDA at ${CUDA_HOME}"
				break
			fi
		done
	fi
fi

# If still not found, stop with guidance
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
	echo "ERROR: Could not find nvcc (CUDA compiler)." >&2
	echo "- Tried '${CUDA_HOME}/bin/nvcc' and common locations under /usr/local." >&2
	echo "- Ensure the CUDA Toolkit is installed and either set CUDA_HOME or add nvcc to PATH." >&2
	echo "- Example: export CUDA_HOME=/usr/local/cuda-12.4; export PATH=\"$CUDA_HOME/bin:$PATH\"" >&2
	exit 1
fi

# Ensure CUDA tools and libs are available
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

# Try to infer CUDA_VERSION from nvcc if possible (non-fatal)
CUDA_VERSION="$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
if [ -n "${CUDA_VERSION}" ]; then
	echo "Using CUDA version ${CUDA_VERSION}"
fi

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

git config --global user.name "Sijeong Kim"
git config --global user.email "ssonge413@gmail.com"


# download data
python download_data.py

# # set environment variables
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1