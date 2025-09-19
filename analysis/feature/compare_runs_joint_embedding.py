#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple runs (e.g., early/mid/last) at a given step on one 2D plane.
- Loads each: exp/<RUN_PATH>/features/step_<STEP>.pt
- Aggregates [V,N,D] -> [N,D] by view mean (or select index)
- Jointly fits PCA (default) or t-SNE across all runs
- Saves scatter plot + meta.json under results/features
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# optional UMAP (미사용 기본)
try:
    import umap   # noqa
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ---------- I/O ----------

def find_step_file(run_dir: Path, step_str: str):
    """
    run_dir: e.g. exp/exp_feature_layer/CACT__feature_layer_early
    step_str: '001000' 같은 6자리 문자열 또는 정수
    """
    if isinstance(step_str, int):
        step_str = f"{step_str:06d}"
    fp = run_dir / "features" / f"step_{step_str}.pt"
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {fp}")
    return fp

def load_features(fp: Path, view_mode="mean", view_index=0):
    """
    Returns X [N,D] float32 L2-normalized, and meta dict.
    Accepts keys: particle_feats / view_feats / *_fp16
    """
    d = torch.load(fp, map_location="cpu")
    if "particle_feats" in d:
        X = d["particle_feats"].float().cpu().numpy()
    elif "view_feats" in d:
        VF = d["view_feats"].float().cpu().numpy() # [V,N,D]
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
        raise KeyError(f"No features in {fp}")

    # L2 normalize (cosine geometry)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    X = X / n

    meta = {
        "file": str(fp),
        "step": int(re.search(r"step_(\d+)\.pt$", str(fp)).group(1)),
        "V": int(d.get("V", -1)),
        "layer_idx": int(d.get("layer_idx", -1)),
        "model_name": d.get("model_name", None),
        "dtype": d.get("dtype", "float32"),
    }
    return X, meta

# ---------- embedding ----------

def joint_embed(X_list, method="pca", seed=0, perplexity=8.0):
    """
    X_list: list of [N_i, D], returns Y_list: list of [N_i, 2], and info dict.
    """
    sizes = [x.shape[0] for x in X_list]
    X_all = np.concatenate(X_list, 0)

    if method.lower() == "pca":
        Xc = X_all - X_all.mean(0, keepdims=True)
        pca = PCA(n_components=2, random_state=seed)
        Y_all = pca.fit_transform(Xc).astype(np.float32)
        info = {"method": "PCA", "explained_var": pca.explained_variance_ratio_.tolist()}
    elif method.lower() == "tsne":
        perp = float(max(2, min(perplexity, X_all.shape[0] - 1)))
        try:
            tsne = TSNE(
                n_components=2, init="pca",
                learning_rate="auto", perplexity=perp,
                n_iter=1000, metric="cosine",
                random_state=seed, verbose=0, square_distances=True,
            )
        except TypeError:
            tsne = TSNE(
                n_components=2, init="pca",
                learning_rate=200.0, perplexity=perp,
                metric="cosine", random_state=seed, verbose=0,
            )
        Y_all = tsne.fit_transform(X_all.astype(np.float32, copy=False)).astype(np.float32)
        info = {"method": "t-SNE", "perplexity": perp}
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    # split back
    Y_list, s = [], 0
    for n in sizes:
        Y_list.append(Y_all[s:s+n])
        s += n
    return Y_list, info

# ---------- plotting ----------

def color_marker_cycles():
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
    markers = ["o","^","s","D","P","X","v","<",">","*"]
    return colors, markers

def square_limits(Y_list, pad=0.05):
    Y_all = np.concatenate(Y_list, 0)
    xmin, xmax = float(Y_all[:,0].min()), float(Y_all[:,0].max())
    ymin, ymax = float(Y_all[:,1].min()), float(Y_all[:,1].max())
    m = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    m = (1.0 + pad) * (m if m > 0 else 1.0)
    return (-m, m), (-m, m)

def plot_runs(Y_list, labels, out_png: Path, title, annotate=False):
    xlim, ylim = square_limits(Y_list, pad=0.05)
    colors, markers = color_marker_cycles()

    fig, ax = plt.subplots(figsize=(6,6), dpi=160)
    ax.set_aspect("equal", adjustable="box")
    for i, (Y, lab) in enumerate(zip(Y_list, labels)):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.scatter(Y[:,0], Y[:,1], s=60, alpha=0.9, c=c, marker=m, label=lab, edgecolors="none")
        if annotate:
            for j, (x,y) in enumerate(Y):
                ax.text(x, y, f"{j}", fontsize=7, ha="center", va="center", alpha=0.8, color=c)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return xlim, ylim

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=str, required=True, help="e.g. 001000 (6 digits) or int")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="list of run directories under exp/, e.g. exp_feature_layer/CACT__feature_layer_early ...")
    ap.add_argument("--method", type=str, default="pca", choices=["pca","tsne"])
    ap.add_argument("--perplexity", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view-mode", type=str, default="mean", choices=["mean","first","index"])
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results/features")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_list, metas, labels = [], [], []
    for r in args.runs:
        run_dir = Path("exp") / r
        fp = find_step_file(run_dir, args.step)
        X, meta = load_features(fp, view_mode=args.view_mode, view_index=args.view_index)
        X_list.append(X); metas.append(meta); labels.append(run_dir.name)

    # joint embedding
    Y_list, info = joint_embed(X_list, method=args.method, seed=args.seed, perplexity=args.perplexity)

    # title & plot
    title = f"Joint {info['method']} @ step {int(args.step):06d} (view={args.view_mode})"
    if info["method"] == "PCA":
        title += f" — EVR: {np.round(info['explained_var'], 3).tolist()}"
    out_png = out_dir / f"compare_runs_{info['method'].lower()}_{int(args.step):06d}.png"
    xlim, ylim = plot_runs(Y_list, labels, out_png, title, annotate=False)

    # meta
    meta_out = {
        "step": int(args.step),
        "runs": [{"label": lab, **m} for lab, m in zip(labels, metas)],
        "method": info["method"],
        "perplexity": info.get("perplexity"),
        "explained_variance_ratio": info.get("explained_var"),
        "view_mode": args.view_mode,
        "view_index": int(args.view_index),
        "xlim": list(xlim), "ylim": list(ylim),
        "num_particles_each": [int(x.shape[0]) for x in X_list],
        "feature_dim": int(X_list[0].shape[1]) if X_list else None,
        "output_png": str(out_png),
    }
    with open(out_dir / f"compare_runs_{info['method'].lower()}_{int(args.step):06d}.meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[INFO] saved → {out_png}")
    print(f"[INFO] saved meta → {out_dir}")

if __name__ == "__main__":
    main()


"""
python compare_runs_joint_embedding.py \
  --step 001000 \
  --runs exp_feature_layer/CACT__feature_layer_early \
         exp_feature_layer/CACT__feature_layer_mid \
         exp_feature_layer/CACT__feature_layer_last \
  --method pca \
  --view-mode mean
  
  
python compare_runs_joint_embedding.py \
  --step 001000 \
  --runs exp_feature_layer/CACT__feature_layer_early \
         exp_feature_layer/CACT__feature_layer_mid \
         exp_feature_layer/CACT__feature_layer_last \
  --method tsne --perplexity 10

"""

