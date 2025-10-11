# 3D-Generation

<div align="center">

# Diversifying 3D Gaussian Splatting with Repulsion Mechanisms

[![Python](https://img.shields.io/badge/Python-3.10.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*MSc Individual Project - Imperial College London*

</div>

## 📋 Project Overview

This repository presents research on diversifying 3D Gaussian Splatting using repulsion mechanisms. The project extends DreamGaussian by implementing Stein Variational Gradient Descent (SVGD) and Repulsive Latent Score Distillation for Solving Inverse Problems (RLSD) to improve the diversity and quality of generated 3D objects.

### 🎯 Key Features

- **Multiple Repulsion Types**: SVGD, RLSD, and baseline (without repulsion)
- **Kernel Functions**: RBF and Cosine similarity kernels
- **Hyperparameter Optimization**: Automated parameter sweeping and analysis
- **Comprehensive Logging**: Detailed experiment tracking and metadata collection
- **Results Analysis**: Automated generation of Pareto plots, box plots, and comparative analyses

### 🔬 Research Contributions

- Novel application of particle-based optimization to 3D Gaussian Splatting
- Comparative analysis of different repulsion mechanisms for 3D generation
- Automated hyperparameter optimization framework for 3D generation tasks

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Development Environment](#development-environment)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Output Structure](#-output-structure)
- [Directory Structure](#directory-structure)
- [References](#-references)

## Development Environment

### ⚙️ System Environment
| Component | Specification |
|-----------|---------------|
| **Operating System** | Ubuntu 22.04 LTS |
| **Python Version** | 3.10.x |
| **CUDA Toolkit** | 12.8 |
| **PyTorch** | 2.8.0+cu128 |
| **cuDNN** | 9.10.2 |

### 🖥️ Hardware Configuration
| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA A100 80GB PCIe |
| **GPU Memory** | 81,920 MiB (80GB) |
| **Architecture** | x86_64 |

### 📦 Dependencies
| Package | Version |
|---------|---------|
| **PyTorch** | 2.8.0+cu128 |
| **CUDA** | 12.8 |
| **cuDNN** | 9.10.2 |
| **Extensions** | diff-gaussian-rasterization, simple-knn |

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sijeong-kim/3D-Generation.git
cd 3D-Generation

# Setup environment (interactive mode)
bash scripts/envs/setup_interactive.sh

# Or setup environment for SLURM/cluster
bash scripts/envs/setup_sbatch.sh
```

### ⚙️ Default Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Initial 3D Gaussian Particles** | 1000 | Number of initial Gaussian splats |
| **Densification Interval** | Every 50 steps | Frequency of particle densification |
| **Default Prompt** | "a photo of a hamburger" | Text prompt for 3D generation |
| **Feature Layer** | 11 | DINOv2 feature layer index (11 = last) |
| **Repulsion Type** | 'rlsd' | Repulsive Mechanism |
| **Kernel Type** | 'rbf' | Similarity Kernel |
| **Opacity LR** | 0.05 | Opacity learning rate |



## 🚀 Quick Start

### Interactive Mode (Local)

```bash
# Single run (baseline, writes to logs/ by default)
python main_ours.py --config configs/text_ours.yaml prompt="a photo of a hamburger"

# Single run (ours RLSD–RBF, writes to exp/)
python main_ours.py --config configs/text_ours.yaml \
  prompt="a photo of a hamburger" \
  repulsion_type=rlsd kernel_type=rbf guidance_scale=50 \
  lambda_repulsion=1000 rbf_beta=0.5 feature_layer=last \
  num_particles=8 outdir=exp/single_run

# Experiment sweeps (configs in configs/text_ours_exp.yaml)
bash scripts/experiments/run_exp_interactive.sh exp0_baseline
bash scripts/experiments/run_exp_interactive.sh exp6_ours_best

# Preview without running
bash scripts/experiments/run_exp_interactive.sh exp6_ours_best --dry_run
```

### SLURM/Cluster Mode

```bash
# Submit an experiment sweep
sbatch scripts/exp_sbatch/run_exp_sbatch.sh exp6_ours_best
```

### 📊 Results

- Interactive runs write per-run outputs under `exp/<sweep_name>/<config_name>/` (preferred) or `logs/` for direct single runs.
- See `analysis/` for scripts to generate figures and CSV summaries.

## 📁 Output Structure

### Experiment Results
- **Experiments (default)**: `exp/<sweep_name>/<config_name>/`
- **Direct single runs**: `logs/`

### Log Files
- **Interactive sweeps**: `exp/<sweep_name>/<config_name>/out`, `err`, `run_metadata.yaml`, `config.yaml`, `figures/`

## 🔗 References
- **DreamGaussian**: [Paper](https://arxiv.org/abs/2309.16553) | [Code](https://github.com/ashawkey/dreamgaussian)
- **3D Gaussian Splatting**: [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Stein Variational Gradient Descent**: [Paper](https://arxiv.org/abs/1608.04471)
- **Repulsive Latent Score Distillation for Solving Inverse Problems**: [Paper](https://arxiv.org/abs/2406.16683)



### Directory Structure

- Project Root Structure
    ```bash
    3D-Generation/
    ├── exp/                      # 👈 Experiment outputs (preferred)
    ├── logs/                     # 👈 Direct single-run outputs
    ├── configs/                  # 👈 Base + sweep YAMLs
    ├── scripts/
    │   ├── experiments/         # 👈 Local interactive runner
    │   └── exp_sbatch/          # 👈 SLURM submit helpers
    ├── analysis/                # 👈 Analysis/plotting scripts
    ├── results/                 # 👈 Precomputed figures/CSVs
    ├── main_ours.py             # 👈 Training entrypoint
    ├── hp_ours.py               # 👈 Legacy HP utilities
    └── [other project files]
    ```

- **Experiment Outputs (`exp/`)**
    ```bash
    exp/<sweep_name>/<config_name>/
    ├── config.yaml
    ├── out
    ├── err
    ├── run_metadata.yaml
    └── figures/
    ```

- Configuration Files (`configs/`)
    ```bash
    configs/
    ├── text_ours.yaml          # Base config (single run)
    ├── text_ours_exp.yaml      # Sweep definitions (Exp0–Exp6, appendix)
    ├── text_baseline.yaml
    └── text_pure_baseline.yaml
    ```


## Acknowledgements

This project made limited use of AI-assisted coding tools:

- GitHub Copilot (for boilerplate and utility code suggestions)  
- Cursor Agent (for refactoring support and inline documentation)

All AI outputs were critically reviewed and validated by the author.  
The final implementation and experiments were conducted independently by Sijeong Kim.
