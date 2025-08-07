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

