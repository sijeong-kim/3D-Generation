# Diversifying Text-to-3D Generation with Repulsive 3D Gaussian Splatting

<div align="center">

📌 **MSc Individual Research Project — Imperial College London**  
Author: **Sijeong Kim**  
[📄 Thesis (Full PDF)](https://drive.google.com/file/d/1bXC_UATHPmgX-QN7wO7KhChED2zLHzpK/view?usp=drive_link)

</div>

---

## 📌 Overview

This repository investigates how **repulsion-based optimization** can improve diversity and stability in text-to-3D generation using **3D Gaussian Splatting (3DGS)**.

### 🚩 Problem
Standard SDS-based text-to-3D pipelines often produce:
- nearly identical shapes across runs,
- mode collapse,
- unstable geometry or over-smoothing.

### ✨ Core Idea
Introduce **feature-space repulsion** (DINOv2 / CLIP features) into DreamGaussian training so that Gaussian particles spread apart in semantic space while maintaining fidelity.


## ✅ Key Contributions

✔ **Repulsion variants implemented**
- SVGD repulsion
- RLSD-style feature repulsion
- Baseline (no repulsion)

✔ **Feature-space guidance**
- DINOv2 / CLIP embeddings
- RBF & cosine kernels

✔ **Large-scale evaluation**
- **↑ 98% semantic diversity**  
- **CLIP fidelity preserved** (ΔCLIP ≈ −0.006)
- **Multi-view consistency C > 0.83**
- **Human perceptual study (n = 41)**

✔ **Reproducible research pipeline**
- Automatic sweeps
- Multi-scene parallel training
- Run metadata, configs, CSVs, and plots auto-generated


---

## 🎬 Demo Results

### Comparison of Our Best Model with Baseline (seed=42)

| Prompt | Baseline | Ours (Best) |
|--------|----------|-------------|
| "a small saguaro cactus plated in a clay pot" | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/baseline.CACT__S42.gif" width="260"> | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/ours_best.CACT__S42.gif" width="260"> |
| "a photo of an ice cream" | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/baseline.ICE__S42.gif" width="260"> | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/ours_best.ICE__S42.gif" width="260"> |
| "an ice cream sundae" | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/baseline.SUND__S42.gif" width="260"> | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/ours_best.SUND__S42.gif" width="260"> |
| "a photo of a hamburger" | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/baseline.HAMB__S42.gif" width="260"> | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/ours_best.HAMB__S42.gif" width="260"> |
| "a photo of a tulip" | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/baseline.TUL__S42.gif" width="260"> | <img src="https://github.com/sijeong-kim/3D-Generation/releases/download/v1.0.0/ours_best.TUL__S42.gif" width="260"> |

---

## 🚀 Installation

```bash
git clone https://github.com/sijeong-kim/3D-Generation.git
cd 3D-Generation

# Local interactive environment
bash scripts/envs/setup_interactive.sh

# Or cluster environment (SLURM)
bash scripts/envs/setup_sbatch.sh
```


## ⚡️ Quick Start

### ✅ Single run (baseline)

```bash
python main_ours.py --config configs/text_baseline.yaml \
    prompt="a photo of a hamburger"
```

### ✅ Repulsion-enabled run (ours)
```bash
python main_ours.py --config configs/text_ours.yaml \
    prompt="a photo of a hamburger" \
    repulsion_type=rlsd \
    kernel_type=rbf \
    lambda_repulsion=1000 \
    num_particles=8 \
    outdir=exp/demo
```
### ✅ Automatic experiment sweeps
```bash
bash scripts/experiments/run_exp_interactive.sh exp6_ours_best
```
### ✅ SLURM (cluster)
```bash
sbatch scripts/exp_sbatch/run_exp_sbatch.sh exp6_ours_best
```

## 📁 Output Structure

```bash
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
├── scripts/
│   ├── experiments/       # Local interactive runs
│   └── exp_sbatch/        # SLURM submit helpers
├── analysis/              # Result parsing & plotting
├── guidance/              # Feature extraction (CLIP/DINOv2) + RNG hooks
├── results/               # Example outputs & CSVs
├── main_ours.py           # Main training pipeline (ours)
├── main_pure_baseline.py  # DreamGaussian baseline
├── kernels.py             # RBF & cosine kernels
├── feature_extractor.py   # Feature-space similarity backend
├── gs_renderer.py         # Gaussian Splatting renderer utilities
├── metrics.py             # CLIP, consistency, and diversity metrics
└── visualizer.py          # Particle visualization
```

---

## References

- DreamGaussian — [https://github.com/ashawkey/dreamgaussian](https://github.com/ashawkey/dreamgaussian)
- 3D Gaussian Splatting — [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- SVGD — [Liu & Wang (NeurIPS 2016)](https://arxiv.org/abs/1608.04471)
- RLSD — [https://arxiv.org/abs/2406.16683](https://arxiv.org/abs/2406.16683)

---

## Acknowledgements

This work was conducted as part of the MSc programme at Imperial College London.
GitHub Copilot and Cursor were used only for boilerplate refactoring;
all design, implementation, experiments, and the report were completed by **Sijeong Kim**.
