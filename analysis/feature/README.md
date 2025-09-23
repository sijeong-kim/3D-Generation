## Feature analysis: how to run and interpret results

This folder contains scripts to visualize and compare feature distributions produced during training/evolution of particles. The workflow assumes experiment data is under `exp/` (your preferred layout) and writes figures and CSVs under `results/features`.

### What the scripts do

- `analyse_particles_per_step_pca.py`: For a single run, builds a global PCA basis over all steps and projects each step to 2D using fixed axes. Exports per-step frames, a multi-step panel, an animated GIF, and metadata.
- `compare_runs_joint_embedding.py`: For two or more runs at a given step, jointly embeds their features into the same 2D space (PCA or t‑SNE), draws scatter plots with optional centroids/ellipses/convex hulls, and computes a simple diversity metric.
- `plot_compare_baseline_ours_csv.py`: Plots time series of the diversity metric from a CSV produced by the comparison script (improvement over baseline across steps).

Underlying inputs are torch files `exp/<experiment>/<run>/**/features/step_XXXXXX.pt` with keys: `particle_feats` [N,D] or `view_feats` [V,N,D]. Rows are L2‑normalized before embedding to operate in cosine geometry.

### Outputs layout

- `results/features/baseline/<RUN>/analysis_particles_pca/` and `results/features/ours/<RUN>/analysis_particles_pca/`
  - `frames/particles_step_XXXXXX.png` — per step projections (fixed global axes)
  - `panel_all_steps.png` — grid of selected steps
  - `evolution.gif` — animation over selected steps
  - `meta.json` — config, global limits, and step list
- `results/features/comparison/`
  - `compare_runs_pca_panel.{png,pdf,svg}` — multi-step joint embeddings (panel mode)
  - `compare_runs_<method>_<step>.{png,pdf,svg}` — single‑step joint embedding
  - `compare_runs_<method>_<step>.meta.json` — plot config and diversity metrics
  - `compare_runs_diversity_<step>.csv` — single‑step diversity numbers
  - `compare_baseline_vs_ours.csv` — diversity time series (metrics mode)
  - `compare_baseline_vs_ours__plots__*.{png,pdf,svg}` — timeseries plots

### 1) Single‑run evolution: global PCA with fixed axes

Run from repo root (select your run and category):

```bash
python analysis/feature/analyse_particles_per_step_pca.py \
  --base exp \
  --exp  exp6_ours_best_feature \
  --run  WO__ICE__S42 \
  --category baseline \
  --view-mode mean \
  --panel-steps 1 200 400 600 800 1000 \
  --panel-rows 2 --panel-cols 3
```

Notes:
- `--category` decides whether results go under `results/features/baseline` or `results/features/ours`. If omitted, it is inferred: names containing `RLSD` are treated as ours.
- `--view-mode {mean,first,index}` controls aggregation when features are `[V,N,D]`.
- `--panel-steps` explicitly selects steps; alternatively use `--step-interval 200` to subsample.

Expected outputs (example):
- `results/features/baseline/WO__ICE__S42/analysis_particles_pca/{frames/,panel_all_steps.png,evolution.gif,meta.json}`

Interpretation:
- Axes are shared (global PCA), enabling consistent comparison across steps.
- Point colors index particles consistently across frames.
- Larger spread suggests higher feature diversity; cluster movement shows evolution.

### 2) Compare runs at a fixed step (Baseline vs Ours)

Single step, joint PCA embedding:

```bash
python analysis/feature/compare_runs_joint_embedding.py \
  --step 001000 \
  --runs exp6_ours_best_feature/WO__ICE__S42 \
         exp6_ours_best_feature/RLSD__RBF__ICE__S42 \
  --labels "Baseline" "Ours" \
  --method pca --style paper \
  --centroid --ellipse --ellipse-std 2.0 --hull \
  --point-size 48 --alpha 0.85 \
  --legend-loc upper right \
  --dpi 300 --figsize 6 6 \
  --transparent
```

Panel across multiple steps (consistent palette and limits across subplots):

```bash
python analysis/feature/compare_runs_joint_embedding.py \
  --panel-steps 1 200 400 600 800 1000 \
  --runs exp6_ours_best_feature/WO__ICE__S42 \
         exp6_ours_best_feature/RLSD__RBF__ICE__S42 \
  --labels "Baseline" "Ours" \
  --method pca --style paper --dpi 300 --figsize 6 6 \
  --centroid --ellipse --hull --transparent
```

Outputs go to `results/features/comparison/`.

Interpretation aids:
- Centroid markers (X) show the mean location per run.
- Ellipses approximate covariance at n‑σ (default 2σ).
- Optional convex hulls outline support (requires SciPy).
- PCA is fast and stable; t‑SNE can reveal non‑linear separations (`--method tsne --perplexity 10`), but axes are not comparable across runs/steps.

### 3) Diversity metrics over steps (CSV + plots)

Produce CSV of diversity for both runs across all shared steps:

```bash
python analysis/feature/compare_runs_joint_embedding.py \
  --metrics \
  --runs exp6_ours_best_feature/WO__ICE__S42 \
         exp6_ours_best_feature/RLSD__RBF__ICE__S42
```

This writes `results/features/comparison/compare_baseline_vs_ours.csv` with columns:
- `step`: training step (6‑digit string)
- `baseline_diversity_trace`, `ours_diversity_trace`: trace(covariance) of the 2D embedding for each run
- `delta`: ours − baseline
- `improvement_pct`: 100 × (ours − baseline) / baseline

Plot the time series and improvement charts:

```bash
python analysis/feature/plot_compare_baseline_ours_csv.py \
  --csv results/features/comparison/compare_baseline_vs_ours.csv \
  --outdir results/features/comparison --style paper
```

This produces `...__diversity_traces.*` and `...__improvement.*` figures.

Metric definition and caveats:
- The “diversity” is the trace of the covariance of the 2D embedding (`diversity_trace`). It grows with spread but depends on the embedding method and scaling.
- For consistency across steps, the CSV uses PCA with shared fitting per step pair; values are comparable within a given run‑pair setup but are not absolute.
- When using t‑SNE for visualization, prefer PCA for metric computation.

### Key arguments (cheat‑sheet)

- Common:
  - `--view-mode {mean,first,index}` and `--view-index <i>` when inputs are `[V,N,D]`.
  - `--figsize W H`, `--dpi 300`, `--style paper`, `--transparent` for publication‑ready figures.
- `analyse_particles_per_step_pca.py`:
  - `--panel-steps s1 s2 ...` or `--step-interval K`; `--panel-rows R --panel-cols C` to control the grid.
  - `--center per-step|global` for building the global PCA basis.
  - Outputs under `results/features/{baseline|ours}/<RUN>/analysis_particles_pca`.
- `compare_runs_joint_embedding.py`:
  - `--method pca|tsne`, `--perplexity 8.0`, `--seed 0`.
  - `--centroid/--no-centroid`, `--ellipse/--no-ellipse --ellipse-std`, `--hull/--no-hull`.
  - `--metrics` to emit `compare_baseline_vs_ours.csv` without plots.
- `plot_compare_baseline_ours_csv.py`:
  - `--csv <path>` and `--outdir <dir>` control input and output location.

### Dependencies

- NumPy, PyTorch (CPU ok), Matplotlib, scikit‑learn. Optional: SciPy (convex hull), `imageio` (GIF), `umap` (not used by default).

### Tips

- Keep runs aligned: ensure both runs have the same set of `step_*.pt` files for clean time‑series metrics.
- For reproducible panels, use PCA and keep seeds fixed; t‑SNE panels can vary with `--perplexity` and `--seed`.
- If you use multi‑view features, start with `--view-mode mean`; then inspect specific views with `--view-mode index --view-index 0`.

### Principles behind `compare_runs_joint_embedding.py`

The script is designed to compare multiple runs fairly at a given step by embedding all runs into the same 2D space and then quantifying dispersion.

- Data preparation
  - Load features from each run at the same step: `exp/<...>/<RUN>/features/step_XXXXXX.pt`.
  - Accepts either `particle_feats` [N,D] or `view_feats` [V,N,D]. If `[V,N,D]`, aggregate to `[N,D]` by `--view-mode` (default `mean`).
  - Sanitize and L2‑normalize each row (feature vector) so all points lie on the unit sphere. This approximates cosine geometry and removes scale effects across particles/runs.

- Joint embedding (single shared space)
  - Concatenate all runs’ features: X_all = concat(X_run1, X_run2, ...).
  - PCA: center X_all by its global mean, fit 2D PCA once, then slice back per run to get Y_runᵢ. Axes are shared and interpretable across runs at that step; explained variance is reported.
  - t‑SNE: fit t‑SNE on X_all (metric=cosine, PCA init). This can reveal non‑linear separations but axis orientation/scale is not comparable across steps; use fixed `--seed` for repeatability.

- Diversity metric (dispersion)
  - For each run’s 2D embedding Y, compute `diversity_trace = trace(cov(Y_centered))`, where Y is centered by its mean. This is rotation‑invariant under orthonormal transforms, increases with spread, and is simple/robust.
  - Report `delta = ours − baseline` and `improvement_pct = 100 × (ours − baseline) / baseline`.
  - For time‑series metrics (`--metrics`), the script always uses PCA per step to keep scale consistent within each pairwise comparison.

- Visualization choices
  - Square, symmetric limits: pick a single magnitude m from combined points so x/y limits are `[-m, m]`, keeping aspect 1:1 and preventing misleading aspect stretch.
  - Panel mode: compute global limits across all selected steps to ensure consistent scale across subplots.
  - Aids: centroids (X marker), covariance ellipses (n‑σ), and optional convex hulls (SciPy) to summarize location, spread, and support.

- Design trade‑offs and caveats
  - Using 2D dispersion loses information versus full‑D; it is chosen to match the visualization and to remain simple/stable. If needed, a full‑D covariance trace could be added as an extension.
  - t‑SNE is non‑linear and non‑metric‑preserving; prefer PCA for metrics and use t‑SNE for qualitative inspection.
  - L2 normalization assumes cosine similarity downstream; if your pipeline uses raw Euclidean norms meaningfully, disable or adjust normalization before embedding.


