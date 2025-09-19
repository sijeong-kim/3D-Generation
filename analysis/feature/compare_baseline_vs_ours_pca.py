#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline vs ours at final step using global PCA.
- Loads:
    exp/exp6_ours_best/features/step_001000.pt
    exp/exp6_baseline/features/step_001000.pt
- Aggregates [V,N,D] → [N,D] by mean across views
- Projects both onto same PCA(2D) basis
- Saves scatter plot + meta.json to results/features
"""

import os, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- paths ---
ours_fp = Path("exp/exp6_ours_best/features/step_001000.pt")
base_fp = Path("exp/exp6_baseline/features/step_001000.pt")
out_dir = Path("results/features")
out_dir.mkdir(parents=True, exist_ok=True)

# --- load helper ---
def load_features(fp, view_mode="mean", view_index=0):
    d = torch.load(fp, map_location="cpu")
    if "particle_feats" in d:
        X = d["particle_feats"].float().cpu().numpy()
    elif "view_feats" in d:
        VF = d["view_feats"].float().cpu().numpy()  # [V,N,D]
        if view_mode == "mean":
            X = VF.mean(0)
        elif view_mode == "first":
            X = VF[0]
        else:
            X = VF[int(view_index) % VF.shape[0]]
    elif "particle_feats_fp16" in d:
        X = d["particle_feats_fp16"].float().cpu().numpy()
    elif "view_feats_fp16" in d:
        VF = d["view_feats_fp16"].float().cpu().numpy()
        X = VF.mean(0)
    else:
        raise KeyError("No features found in file")

    # normalize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n

# --- load both ---
X_ours = load_features(ours_fp, view_mode="mean")
X_base = load_features(base_fp, view_mode="mean")

# --- global PCA ---
X_all = np.concatenate([X_ours, X_base], 0)
X_all = X_all - X_all.mean(0, keepdims=True)
pca = PCA(n_components=2, random_state=0)
Y_all = pca.fit_transform(X_all)

Y_ours = Y_all[:X_ours.shape[0]]
Y_base = Y_all[X_ours.shape[0]:]

# --- global limits (square) ---
xmin, xmax = Y_all[:,0].min(), Y_all[:,0].max()
ymin, ymax = Y_all[:,1].min(), Y_all[:,1].max()
m = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * 1.05
xlim, ylim = (-m, m), (-m, m)

# --- plot ---
fig, ax = plt.subplots(figsize=(5,5), dpi=160)
ax.set_aspect("equal", adjustable="box")
ax.scatter(Y_base[:,0], Y_base[:,1], c="tab:blue", label="baseline", s=60, alpha=0.8)
ax.scatter(Y_ours[:,0], Y_ours[:,1], c="tab:orange", label="ours", s=60, alpha=0.8)
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title("Baseline vs Ours — PCA (step 1000)")
fig.tight_layout()
out_path = out_dir / "baseline_vs_ours_pca.png"
fig.savefig(out_path)
plt.close(fig)

# --- meta.json ---
meta = {
    "baseline_file": str(base_fp),
    "ours_file": str(ours_fp),
    "num_particles": int(X_ours.shape[0]),
    "feature_dim": int(X_ours.shape[1]),
    "pca_explained_var": pca.explained_variance_ratio_.tolist(),
    "xlim": list(xlim),
    "ylim": list(ylim),
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[INFO] saved plot → {out_path}")
print(f"[INFO] saved meta → {out_dir/'meta.json'}")

"""
python compare_baseline_vs_ours_pca.py
"""