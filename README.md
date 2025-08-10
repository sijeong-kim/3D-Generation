# 3D-Generation

This repository presents my MSc Individual Project aiming to diversify 3D Gaussian Splatting using repulsion mechanism.

## Development Environment
SLURM-based HPC cluster with CUDA 12.4

### ⚙️ Environment
| Setting                         | Value                                                          |
| ------------------------------- | -------------------------------------------------------------- |
| GPU Resource node                | Tesla A40 (48GB), A100 (80GB) |
| CUDA Version                     | 12.4.0                                                         |
| Virtual Environment Python       | 3.9                                                            |
| Extensions Built | diff-gaussian-rasterization, simple-knn |

### 📦 Installation

```bash
git clone https://github.com/sijeong-kim/3D-Generation.git
cd 3D-Generation
bash scripts/setup_venv.sh
```

### 📌 Configuration Notes

| Hyper-parameters | Value |
| ----------------------- | ----------------------- |
| Initial 3D Gaussian Particles    | 1000                                                           |
| Densification Interval           | Every 50 steps                                                 |
| Prompt                   | "a photo of a hamburger"                                       |



## 🚀 How to Run

### Running scripts (SLURM Job)

- Run scripts
    ```bash
    sbatch scripts/run.sh
    ```

### 📊 Output Directory Structure

- All generated outputs will be saved in:
    ```bash
    outputs/<SLURM_JOB_ID>/<TASK_NAME>/
    ```
- Logs will be saved in:
    ```bash
    outputs/<SLURM_JOB_ID>/output.out
    outputs/<SLURM_JOB_ID>/error.err
    ```
    
### 🔗 Reference
- DreamGaussian Paper



### Directory Structure

- Project Root Structure
```bash

/vol/bitbucket/sk2324/3D-Generation/
├── logs/              # 👈 Main experiment results
├── outputs/           # 👈 SLURM job outputs  
├── configs/           # 👈 Configuration files
├── scripts/           # 👈 SLURM batch scripts
├── metrics/           # 👈 Empty (legacy)
├── hp_ours.py         # 👈 Main hyperparameter script
└── [other project files]
```
- Experiments Results (`logs/`)
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

