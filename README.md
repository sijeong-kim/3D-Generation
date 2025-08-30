# 3D-Generation

<div align="center">

# Diversifying 3D Gaussian Splatting with Repulsion Mechanisms

[![Python](https://img.shields.io/badge/Python-3.12.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*MSc Individual Project - Imperial College London*

</div>

## 📋 Project Overview

This repository presents research on diversifying 3D Gaussian Splatting using repulsion mechanisms. The project extends DreamGaussian by implementing Stein Variational Gradient Descent (SVGD) and Regularized Least-Squares Descent (RLSD) to improve the diversity and quality of generated 3D objects.

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
| **Operating System** | Ubuntu 22.04.1 LTS (Linux 6.2.0-36-generic) |
| **Python Version** | 3.12.11 (conda-forge) |
| **CUDA Version** | 13.0 (Driver: 580.65.06) |
| **PyTorch Version** | 2.8.0+cu128 |
| **cuDNN Version** | 91002 |

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
| **cuDNN** | 91002 |
| **Extensions** | diff-gaussian-rasterization, simple-knn |

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sijeong-kim/3D-Generation.git
cd 3D-Generation

# Setup environment (interactive mode)
bash scripts/envs/setup_interactive.sh

# Or setup environment for SLURM cluster
bash scripts/envs/setup_slurm.sh
```

### ⚙️ Default Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Initial 3D Gaussian Particles** | 1000 | Number of initial Gaussian splats |
| **Densification Interval** | Every 50 steps | Frequency of particle densification |
| **Default Prompt** | "a photo of a hamburger" | Text prompt for 3D generation |
| **Feature Layer** | 11 | CLIP feature extraction layer |
| **Repulsion Type** | 'wo' (without) | Baseline: no repulsion mechanism |



## 🚀 Quick Start

### Interactive Mode (Local/Development)

```bash
# Run a single experiment
python main_ours.py --config configs/text_ours.yaml --prompt "a photo of a hamburger"

# Run hyperparameter experiments
bash scripts/experiments/run_exp_interactive.sh exp0_baseline

# Run with dry-run to preview experiments
bash scripts/experiments/run_exp_interactive.sh exp0_baseline --dry-run
```

### SLURM Cluster Mode (Production)

```bash
# Submit experiment job to SLURM
sbatch scripts/experiments/run_exp_sbatch.sh exp0_baseline

# Submit multiple experiments
sbatch scripts/experiments/run_exp_sbatch.sh exp0_baseline exp1_1_lambda_coarse
```

### 📊 Results Analysis

```bash
# Aggregate and analyze experiment results
python gather_results.py --exp_dir exp/ --output_dir analysis_results/

# Generate specific plot types
python gather_results.py --exp_dir exp/ --output_dir analysis_results/ --plot_type pareto
python gather_results.py --exp_dir exp/ --output_dir analysis_results/ --plot_type boxplot

# Filter experiments by parameters (using key=value directory naming)
ls exp/*/kernel_type=cosine*                    # All cosine kernel experiments
ls exp/*/repulsion_type=svgd*                   # All SVGD experiments
ls exp/*/prompt=hamburger*                      # All hamburger experiments
find exp/ -name "*lambda_repulsion=1000*"       # All experiments with lambda=1000
find exp/ -name "*kernel_type=cosine*" -name "*repulsion_type=rlsd*"  # RLSD + cosine
```

## 📁 Output Structure

### Experiment Results
- **New experiments**: `exp/<experiment_name>/`
- **Legacy experiments**: `logs/<experiment_name>/`
- **SLURM outputs**: `outputs/<SLURM_JOB_ID>/`

### Log Files
- **Job stdout**: `outputs/<SLURM_JOB_ID>/output.out`
- **Job stderr**: `outputs/<SLURM_JOB_ID>/error.err`
- **Experiment logs**: `exp/<experiment_name>/*/stdout.log`, `stderr.log`

## 🔗 References
- **DreamGaussian**: [Paper](https://arxiv.org/abs/2309.16553) | [Code](https://github.com/ashawkey/dreamgaussian)
- **3D Gaussian Splatting**: [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Stein Variational Gradient Descent**: [Paper](https://arxiv.org/abs/1608.04471)



### Directory Structure

- Project Root Structure
    ```bash
    /vol/bitbucket/sk2324/3D-Generation/
    ├── exp/               # 👈 New experiment results (hp_ours.py)
    ├── logs/              # 👈 Legacy experiment results
    ├── outputs/           # 👈 SLURM job outputs  
    ├── configs/           # 👈 Configuration files
    ├── scripts/           # 👈 SLURM batch scripts
    ├── metrics/           # 👈 Empty (legacy)
    ├── hp_ours.py         # 👈 Main hyperparameter script
    ├── gather_results.py  # 👈 Results aggregation script
    └── [other project files]
    ```

- **New Experiment Results (`exp/`)** - Enhanced hyperparameter experiments
    ```bash
    exp/experiment_name/                    # e.g., exp0_baseline, exp1_1_lambda_coarse
    ├── prompt=hamburger_repulsion_type=svgd_kernel_type=rbf_lambda_repulsion=600/
    │   ├── figures/                       # 👈 Auto-created analysis plots directory
    │   │   ├── pareto_plot.png
    │   │   ├── boxplot_comparison.png
    │   │   └── lambda_analysis.png
    │   ├── config.yaml                    # 👈 Experiment configuration
    │   ├── stdout.log                     # 👈 Standard output log
    │   ├── stderr.log                     # 👈 Error log (with auto-tail on failure)
    │   ├── run_metadata.yaml              # 👈 System/environment metadata
    │   ├── .done                          # 👈 Completion marker (prevents re-runs)
    │   └── ... (other experiment files)
    ├── prompt=icecream_repulsion_type=rlsd_kernel_type=cosine_lambda_repulsion=800/
    │   └── ... (similar structure)
    └── experiment_summary.yaml            # 👈 Aggregated results summary
    ```

- **Legacy Experiments Results (`logs/`)** - Original experiment structure
    - Debug Experiments (`run_ours_debug.sh`)
        ```bash
        logs/debug_001/                    # 3 runs total
        ├── run_001_[hash]_s42/           # SVGD method
        │   ├── metrics/                  # 👈 CSV files here
        │   │   ├── quantitative_metrics.csv
        │   │   ├── losses.csv
        │   │   └── efficiency.csv
        │   └── experiment_config.json
        ├── run_002_[hash]_s42/           # RLSD method  
        │   └── metrics/
        └── run_003_[hash]_s42/           # Baseline (wo)
            └── metrics/
        ```
    - Hyperparameter Tuning (`run_ours_hp.sh`)
        ```bash
        logs/exp_001/                     # Coarse search: 81 runs
        ├── run_001_[hash]_s42/           # svgd, layer=2, lambda=1, seed=42
        │   └── metrics/
        ├── run_002_[hash]_s42/           # svgd, layer=2, lambda=100, seed=42
        │   └── metrics/
        ├── ...                           # 79 more combinations
        └── run_081_[hash]_s456/          # wo, layer=10, lambda=10000, seed=456
            └── metrics/

        logs/exp_002/                     # Fine search: 35 runs
        logs/exp_003/                     # Generalization: 15 runs  
        logs/exp_004/                     # Efficiency: 36 runs
        logs/exp_005/                     # Multi-view: 36 runs
        ```
- SLURM Job Outputs (`outputs/`)
    ```bash
    outputs/[job_id]/
    ├── output.out                    # 👈 Job stdout
    └── error.err                     # 👈 Job stderr/errors
    ```

- Configuration Files (`configs/`)
    ```bash
    configs/
    ├── text_ours.yaml               # 👈 Base configuration
    ├── text_ours_debug.yaml         # 👈 Debug: 3 quick tests
    └── text_ours_hp.yaml            # 👈 HP: 203 full experiments
    ```
- execution scripts (`scripts/`)
    ```bash
    scripts/
    ├── run_ours_debug.sh            # 👈 Debug: --mem=16G, 30min
    └── run_ours_hp.sh               # 👈 HP: --mem=32G, 48hours
    ```


## Acknowledgements

This project made limited use of AI-assisted coding tools:

- GitHub Copilot (for boilerplate and utility code suggestions)  
- Cursor Agent (for refactoring support and inline documentation)

All AI outputs were critically reviewed and validated by the author.  
The final implementation and experiments were conducted independently by Sijeong Kim.
