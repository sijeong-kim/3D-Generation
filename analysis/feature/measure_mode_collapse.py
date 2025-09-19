#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure mode-collapse signals over training steps in the ORIGINAL feature space.

Outputs under exp/<exp>/<run>/analysis_collapse:
  - metrics.json (+ metrics.csv): per-step MPD, MinDist, CV, ParticipationRatio, CovTrace
  - timeseries.png : metrics over steps
  - distmat_step_XXXXXX.png : pairwise cosine distance heatmap per step
  - pca_panel.png : fixed-reference PCA (step0) panel (optional, for visual aid only)

Run:
python analysis/feature/measure_mode_collapse.py \
  --base exp --exp exp_feature_layer --run BULL__feature_layer_last \
  --view-mode mean
"""
import argparse, glob, os, re, json, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def find_feature_files(base_dir: Path, exp_name: str, run_name: str):
    pats = [
        base_dir / exp_name / run_name / "**" / "features" / "step_*.pt",
        base_dir / exp_name / "**" / run_name / "**" / "features" / "step_*.pt",
    ]
    files = []
    for p in pats: files.extend(glob.glob(str(p), recursive=True))
    if not files:
        raise FileNotFoundError(f"No step_*.pt under {base_dir}/{exp_name}/{run_name}")
    files = list(set(files))
    def stepnum(s):
        m = re.search(r"step_(\d+)\.pt$", s); return int(m.group(1)) if m else -1
    return sorted(files, key=stepnum)

def tload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def to_np(x):
    if isinstance(x, torch.Tensor): return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def load_step_feats(fp, view_mode="mean", view_index=0):
    d = tload(fp)
    step = int(d.get("step", -1))
    if "particle_feats" in d:
        X = to_np(d["particle_feats"])            # [N,D]
        pids = to_np(d.get("particle_ids", np.arange(X.shape[0]))).astype(int).ravel()
    elif "view_feats" in d:
        VF = to_np(d["view_feats"])               # [V,N,D]
        if view_mode == "mean":
            X = VF.mean(0)
        elif view_mode == "first":
            X = VF[0]
        else:
            X = VF[int(view_index) % VF.shape[0]]
        pids = to_np(d.get("particle_ids", np.arange(X.shape[0]))).astype(int).ravel()
    elif "particle_feats_fp16" in d:
        X = to_np(d["particle_feats_fp16"])
        pids = np.arange(X.shape[0], dtype=int)
    elif "view_feats_fp16" in d:
        VF = to_np(d["view_feats_fp16"])
        X = VF.mean(0)
        pids = np.arange(X.shape[0], dtype=int)
    else:
        raise KeyError("No features found.")

    # L2 normalize, sanitize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    X = X / n
    m = np.isfinite(X).all(axis=1)
    return step, pids[m], X[m], int(d.get("layer_idx", -1)), d.get("model_name", None)

# ---------------- metrics ----------------
def pairwise_cosine_distance(X):
    # X is row-normalized. cosine distance = 1 - cosine similarity
    S = X @ X.T
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, np.nan)  # ignore self
    return D

def participation_ratio(X):
    # zero-mean for covariance
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    eig = np.linalg.eigvalsh(C)  # stable for symmetric
    eig = np.clip(eig, 0.0, None)
    num = (eig.sum())**2
    den = (eig**2).sum() + 1e-12
    return float(num / den), float(eig.sum())  # (PR, trace)

# ---------------- plotting helpers ----------------
def save_dist_heatmap(out_path, D, title):
    fig, ax = plt.subplots(figsize=(4,4), dpi=160)
    im = ax.imshow(D, cmap="viridis", vmin=0, vmax=np.nanmax(D))
    ax.set_title(title); ax.set_xlabel("particle id"); ax.set_ylabel("particle id")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def save_timeseries(out_path, steps, mpd, mind, cvd, pr, tr):
    fig, axes = plt.subplots(3, 2, figsize=(10,8), dpi=160)
    axes = axes.ravel()
    axes[0].plot(steps, mpd, marker="o");  axes[0].set_title("MPD (mean pairwise cosine distance)")
    axes[1].plot(steps, mind, marker="o"); axes[1].set_title("Min pairwise distance")
    axes[2].plot(steps, cvd, marker="o");  axes[2].set_title("CV of distances")
    axes[3].plot(steps, pr, marker="o");   axes[3].set_title("Participation Ratio (effective dim.)")
    axes[4].plot(steps, tr, marker="o");   axes[4].set_title("Covariance Trace (total variance)")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.set_xlabel("step")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def pca_panel_fixed_ref(out_path, X_list, steps, cols=5):
    # Fixed PCA basis from first step
    X0 = X_list[0]
    X0c = X0 - X0.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X0c, full_matrices=False)
    W = Vt[:2].T  # [D,2]
    Ys = []
    for X in X_list:
        Xc = X - X.mean(0, keepdims=True)
        Y = Xc @ W
        Ys.append(Y.astype(np.float32))
    # global square limits
    xs = np.concatenate([Y[:,0] for Y in Ys]); ys = np.concatenate([Y[:,1] for Y in Ys])
    m = max(abs(xs).max(), abs(ys).max()) * 1.05
    rows = int(np.ceil(len(steps)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2), dpi=160)
    axes = np.atleast_2d(axes)
    for i, (s, Y) in enumerate(zip(steps, Ys)):
        r,c = divmod(i, cols); ax = axes[r,c]
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(Y[:,0], Y[:,1], s=25, alpha=0.9)
        ax.set_xlim(-m,m); ax.set_ylim(-m,m)
        ax.set_title(f"Step {s}")
        ax.grid(True, alpha=0.3)
    for j in range(len(steps), rows*cols):
        r,c = divmod(j, cols); axes[r,c].axis("off")
    fig.suptitle("Fixed-reference PCA projection (visual aid)", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out_path); plt.close(fig)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="exp")
    ap.add_argument("--exp",  type=str, required=True)
    ap.add_argument("--run",  type=str, required=True)
    ap.add_argument("--view-mode", type=str, default="mean", choices=["mean","first","index"])
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    base = Path(args.base)
    out_root = base / args.exp / args.run
    out_dir = out_root / "analysis_collapse"
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    pt_files = find_feature_files(base, args.exp, args.run)
    print(f"[INFO] found {len(pt_files)} step files")

    steps, X_list, layer_idx, model_name = [], [], None, None
    for fp in pt_files[: (args.max_steps or len(pt_files))]:
        step, pids, X, layer_idx, model_name = load_step_feats(fp, args.view_mode, args.view_index)
        # 공통 id 강제 정렬 필요 없고, 각 step의 N은 동일하다고 가정(아니면 X rows로 판단)
        steps.append(step); X_list.append(X)

    # compute metrics per step
    mpd, mind, cvd, pr, tr = [], [], [], [], []
    for s, X in zip(steps, X_list):
        D = pairwise_cosine_distance(X)  # [N,N], NaN on diag
        vals = D[~np.isnan(D)]
        mpd.append(float(vals.mean()) if vals.size else float("nan"))
        mind.append(float(vals.min()) if vals.size else float("nan"))
        cvd.append(float(vals.std()/max(vals.mean(),1e-12)) if vals.size else float("nan"))
        pr_s, tr_s = participation_ratio(X)
        pr.append(pr_s); tr.append(tr_s)

        # dist heatmap
        save_dist_heatmap(out_dir / f"frames/distmat_step_{s:06d}.png", D, f"Pairwise cosine distance (step {s})")

    # time-series plot
    save_timeseries(out_dir / "timeseries.png", steps, mpd, mind, cvd, pr, tr)

    # optional PCA panel (시각 보조용)
    pca_panel_fixed_ref(out_dir / "pca_panel.png", X_list, steps, cols=5)

    # dump metrics
    meta = {
        "exp": args.exp, "run": args.run,
        "layer_idx": layer_idx, "model_name": model_name,
        "view_mode": args.view_mode, "view_index": args.view_index,
        "num_steps": len(steps), "steps": steps,
        "metrics": {
            "mpd": mpd, "min_dist": mind, "cv_dist": cvd,
            "participation_ratio": pr, "cov_trace": tr
        }
    }
    with open(out_dir / "metrics.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step","MPD","MinDist","CV_Dist","ParticipationRatio","CovTrace"])
        for row in zip(steps, mpd, mind, cvd, pr, tr): w.writerow(row)

    print(f"[INFO] saved → {out_dir}")

if __name__ == "__main__":
    main()
