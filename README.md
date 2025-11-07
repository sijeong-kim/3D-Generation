# Diversifying Text-to-3D Generation with Repulsive 3D Gaussian Splatting

<div align="center">

📌 **MSc Individual Research Project — Imperial College London**  
Author: **Sijeong Kim**  
[📄 Thesis (Full PDF)](https://drive.google.com/file/d/1bXC_UATHPmgX-QN7wO7KhChED2zLHzpK/view?usp=drive_link)

</div>

## Overview

This repository explores how **repulsion-based optimization** can make text-to-3D generation **more diverse, more consistent, and more stable** using **3D Gaussian Splatting**.

🚩 Problem  
Text-to-3D pipelines often suffer from **mode collapse**, producing nearly identical shapes or weak geometry.

✨ Core Idea  
Introduce **feature-space repulsion** (using DINOv2) into DreamGaussian training to spread particles apart in semantic space while preserving fidelity.

## Key Contributions

✔  **Repulsion mechanisms** implemented for 3DGS  
- **SVGD**
- **RLSD-style repulsion**
- Baseline (no repulsion)

✔  **Feature-space guidance**
- CLIP/DINOv2 feature similarity → kernel-based repulsion
- Supports RBF + Cosine kernels

✔  **Large-scale evaluation**
- **98%↑ semantic diversity** while **preserving CLIP fidelity**
- **Multi-view consistency C > 0.83**
- **Human perceptual study (n = 41)**

✔  **Fully modular pipeline**
- Automatic sweeps
- Parallel training (N scenes simultaneously)
- Reproducible experiment logs, CSVs, and figures

---

## Demo Results

> ✅ (GIFs / rendered scenes will be inserted here after upload)  
> Example comparisons: **Baseline vs SVGD vs RLSD-Feature**

---

## Installation

```bash
git clone https://github.com/sijeong-kim/3D-Generation.git
cd 3D-Generation
bash scripts/envs/setup_interactive.sh
```

For SLURM/cluster users:
```bash
bash scripts/envs/setup_sbatch.sh
```
## Quick Start

```bash
bash scripts/experiments/run_exp_interactive.sh exp6_ours_best
```

## Output Structure

```lua
exp/
  ├── <sweep_name>/<config_name>/
  │    ├── config.yaml
  │    ├── run_metadata.yaml
  │    ├── out / err
  │    └── figures/ (PSNR, SSIM, CLIP, diversity stats, Pareto plots)
```

## Repository Structure

```bash
3D-Generation/
├── configs/               # YAML configs & sweep definitions
├── scripts/               # interactive + SLURM runners
├── analysis/              # result parsing & plot generation
├── results/               # sample outputs and CSVs
├── main_ours.py           # training pipeline
└── hp_ours.py             # legacy HP utils
```

---

## References

DreamGaussian — [https://github.com/ashawkey/dreamgaussian](https://github.com/ashawkey/dreamgaussian)

3D Gaussian Splatting — [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

SVGD — [Liu & Wang (NeurIPS 2016)](https://arxiv.org/abs/1608.04471)

RLSD — [https://arxiv.org/abs/2406.16683](https://arxiv.org/abs/2406.16683)

---

## Acknowledgements

This research was conducted as part of the MSc programme at Imperial College London.
Limited AI-assisted tools (GitHub Copilot, Cursor) were used only for boilerplate support;
all core implementation and experiments were authored by Sijeong Kim.
