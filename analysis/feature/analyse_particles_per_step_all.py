#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob, os, re, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Keep BLAS single-threaded for stability on tiny inputs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- I/O helpers ----------
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

def _to_np(x):
    if isinstance(x, torch.Tensor): return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def load_per_step(file_paths, view_mode="mean", view_index=0, max_steps=None):
    """
    Returns:
      steps: [S]
      ids_list: list of arrays [N_s] original particle_ids per step
      X_list:   list of arrays [N_s, D] (L2-normalized)
      meta: dict
    """
    steps, ids_list, X_list = [], [], []
    meta = {}
    fps = file_paths[: (max_steps or len(file_paths))]
    for fp in fps:
        d = tload(fp)
        step = int(d.get("step", -1))

        # particle ids (keep them!)
        if "particle_ids" in d:
            ids = _to_np(d["particle_ids"]).astype(int).ravel()
        else:
            # fallback to range(N) if missing; will intersect later
            if "particle_feats" in d:
                ids = np.arange(d["particle_feats"].shape[0], dtype=int)
            elif "particle_feats_fp16" in d:
                ids = np.arange(d["particle_feats_fp16"].shape[0], dtype=int)
            elif "view_feats" in d:
                ids = np.arange(d["view_feats"].shape[1], dtype=int)
            elif "view_feats_fp16" in d:
                ids = np.arange(d["view_feats_fp16"].shape[1], dtype=int)
            else:
                raise KeyError("Could not infer particle_ids")

        # features
        if "particle_feats" in d:
            X = _to_np(d["particle_feats"])
        elif "view_feats" in d:
            VF = _to_np(d["view_feats"])  # [V, N, D]
            if view_mode == "mean":
                X = VF.mean(axis=0)
            elif view_mode == "first":
                X = VF[0]
            elif view_mode == "index":
                X = VF[int(view_index) % VF.shape[0]]
            else:
                raise ValueError(f"Unknown view_mode '{view_mode}'")
        elif "particle_feats_fp16" in d:
            X = _to_np(d["particle_feats_fp16"])
        elif "view_feats_fp16" in d:
            VF = _to_np(d["view_feats_fp16"])
            X = VF.mean(axis=0)
        else:
            raise KeyError("No particle/view features found in file.")

        # sanitize & L2-normalize per row
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        X = X / n

        # drop non-finite rows but keep ids aligned
        m = np.isfinite(X).all(axis=1)
        ids, X = ids[m], X[m]

        steps.append(step); ids_list.append(ids); X_list.append(X)

        if not meta:
            meta = {
                "model_name": d.get("model_name", None),
                "layer_idx": int(d.get("layer_idx", -1)),
                "V": int(d.get("V", -1)) if "V" in d else None,
            }
    return steps, ids_list, X_list, meta

# ---------- embedding ----------
def embed_2d(X, method="tsne", seed=0, perplexity=8):
    N = X.shape[0]
    if N < 2 or np.allclose(X.var(axis=0).sum(), 0.0, atol=1e-12):
        theta = np.linspace(0, 2*np.pi, num=max(N,1), endpoint=False)
        Y = np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32) * 1e-3
        return Y[:N], "degenerate"

    if method == "pca":
        Xc = X - X.mean(0, keepdims=True)
        Y = PCA(n_components=2, random_state=seed).fit_transform(Xc)
        return Y.astype(np.float32), "PCA"

    if method == "umap" and HAS_UMAP:
        Y = umap.UMAP(n_components=2, random_state=seed, init="random",
                      n_neighbors=max(2, min(10, N-1)), min_dist=0.1, metric="cosine").fit_transform(X)
        return Y.astype(np.float32), "UMAP"

    # robust t-SNE
    perp = float(max(2, min(perplexity, N - 1)))
    try:
        tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=perp,
                    n_iter=1000, metric="cosine", random_state=seed, verbose=0, square_distances=True)
    except TypeError:
        tsne = TSNE(n_components=2, init="random", learning_rate=200.0, perplexity=perp,
                    metric="cosine", random_state=seed, verbose=0)
    Y = tsne.fit_transform(X.astype(np.float32, copy=False))
    return Y.astype(np.float32), "t-SNE"

# ---------- Procrustes alignment (to step 0) ----------
def procrustes_align(Y, Yref):
    """
    Align Y to Yref using similarity transform (rotation + scale + translation).
    Returns aligned Y.
    """
    if Y.shape[0] < 2 or Yref.shape[0] < 2:
        return Y.copy()
    # center
    mu  = Y.mean(0, keepdims=True);  Yc  = Y  - mu
    mur = Yref.mean(0, keepdims=True); Yrc = Yref - mur
    # scale by Frobenius norms
    sY  = np.linalg.norm(Yc);   sY  = 1.0 if sY  < 1e-12 else sY
    sYr = np.linalg.norm(Yrc);  sYr = 1.0 if sYr < 1e-12 else sYr
    Yc  /= sY;  Yrc /= sYr
    # rotation via SVD (2D Kabsch)
    H = Yc.T @ Yrc
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # scale to reference
    scale = sYr / sY
    Y_aligned = (Yc @ R) * (sYr) + mur  # equivalent: scale*sY = sYr
    return Y_aligned

# ---------- plotting ----------
def color_map_for_ids(all_ids_sorted):
    # stable color per original id (tab10 cycling)
    cmap = {}
    for idx, pid in enumerate(all_ids_sorted):
        cmap[int(pid)] = idx % 10
    return cmap

def global_limits(Y_list, pad=0.05, symmetric=True):
    xs = np.concatenate([Y[:,0] for Y in Y_list], 0)
    ys = np.concatenate([Y[:,1] for Y in Y_list], 0)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if symmetric:
        m = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        m *= (1.0 + pad)
        return (-m, m), (-m, m)
    wx = xmax - xmin; wy = ymax - ymin
    return (xmin - pad*wx, xmax + pad*wx), (ymin - pad*wy, ymax + pad*wy)

def save_frames(frames_dir: Path, steps, Y_list, id_list, id2color, xlim, ylim, title_prefix, annotate=True):
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for s, Y, ids in zip(steps, Y_list, id_list):
        fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=160)
        ax.set_aspect("equal", adjustable="box")
        colors = [id2color[int(p)] for p in ids]
        ax.scatter(Y[:,0], Y[:,1], c=colors, cmap="tab10", s=60, alpha=0.95, edgecolors="none")
        if annotate:
            for (x,y,pid) in zip(Y[:,0], Y[:,1], ids):
                ax.text(x, y, f"{int(pid)}", fontsize=8, ha="center", va="center", alpha=0.9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_title(f"Step {s}")
        fig.suptitle(title_prefix, y=0.98)
        fig.tight_layout(rect=[0,0,1,0.95])
        fp = frames_dir / f"particles_step_{s:06d}.png"
        fig.savefig(fp); plt.close(fig)
        out.append(fp)
    return out

def save_panel(panel_path: Path, steps, Y_list, id_list, id2color, xlim, ylim, title_prefix):
    cols = len(steps); rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, 3.6), dpi=160)
    if cols == 1: axes = np.array([axes])
    for i, (s, Y, ids) in enumerate(zip(steps, Y_list, id_list)):
        ax = axes[i]
        ax.set_aspect("equal", adjustable="box")
        colors = [id2color[int(p)] for p in ids]
        ax.scatter(Y[:,0], Y[:,1], c=colors, cmap="tab10", s=50, alpha=0.95, edgecolors="none")
        for (x,y,pid) in zip(Y[:,0], Y[:,1], ids):
            ax.text(x, y, f"{int(pid)}", fontsize=8, ha="center", va="center", alpha=0.9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f"Step {s}", fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.3)
        if i == 0: ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
        else:      ax.set_xlabel("");       ax.set_ylabel("")
    fig.suptitle(title_prefix, y=0.99)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(panel_path); plt.close(fig)

def maybe_gif(frame_paths, out_path: Path, fps=2):
    try:
        import imageio.v2 as imageio
        ims = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(out_path, ims, duration=1.0/max(1,fps))
        return out_path
    except Exception as e:
        print(f"[WARN] could not create GIF: {e}")
        return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="exp")
    ap.add_argument("--exp",  type=str, required=True)
    ap.add_argument("--run",  type=str, required=True)

    ap.add_argument("--method", type=str, default="tsne", choices=["tsne","pca","umap"])
    ap.add_argument("--perplexity", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)

    ap.add_argument("--view-mode", type=str, default="mean", choices=["mean","first","index"])
    ap.add_argument("--view-index", type=int, default=0)

    ap.add_argument("--align", action="store_true", default=True, help="Procrustes-align each step to step-0")
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.add_argument("--symmetric", action="store_true", default=True)
    ap.add_argument("--no-symmetric", dest="symmetric", action="store_false")

    args = ap.parse_args()

    base = Path(args.base)
    out_root = base / args.exp / args.run
    out_dir  = out_root / "analysis_particles_ind"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pt_files = find_feature_files(base, args.exp, args.run)
    print(f"[INFO] found {len(pt_files)} step files")

    steps, ids_list_raw, X_list_raw, meta = load_per_step(
        pt_files, view_mode=args.view_mode, view_index=args.view_index, max_steps=args.max_steps
    )
    print(f"[INFO] steps loaded: {len(steps)}")

    # keep only particle_ids present in ALL steps (stable set & order)
    common_ids = set(ids_list_raw[0].tolist())
    for ids in ids_list_raw[1:]:
        common_ids &= set(ids.tolist())
    common_ids = np.array(sorted(common_ids), dtype=int)
    if common_ids.size == 0:
        raise RuntimeError("No common particle_ids across steps (after NaN filtering).")

    # reorder each step to this consistent id order
    ids_list, X_list = [], []
    for ids, X in zip(ids_list_raw, X_list_raw):
        # map id -> row index
        idx_map = {int(p): i for i, p in enumerate(ids.tolist())}
        keep_idx = [idx_map[int(p)] for p in common_ids]
        ids_list.append(common_ids.copy())
        X_list.append(X[np.array(keep_idx, dtype=int)])

    # embed independently per step
    Y_list = []
    meth_used = None
    for X in X_list:
        Y, mname = embed_2d(X, method=args.method, seed=args.seed, perplexity=args.perplexity)
        Y_list.append(Y); meth_used = mname

    # optional Procrustes alignment to step 0
    if args.align and len(Y_list) >= 2:
        Y0 = Y_list[0]
        Y_list = [Y0] + [procrustes_align(Y, Y0) for Y in Y_list[1:]]

    # global square limits across aligned embeddings
    xlim, ylim = global_limits(Y_list, symmetric=args.symmetric, pad=0.05)

    # stable colors by particle id
    id2color = color_map_for_ids(common_ids)

    title = f"{args.exp}/{args.run} — {meth_used} on particle features (layer={meta.get('layer_idx')}, agg={args.view_mode}, aligned={bool(args.align)})"

    # per-step frames
    frame_paths = save_frames(frames_dir, steps, Y_list, ids_list, id2color, xlim, ylim, title_prefix=title, annotate=True)
    print(f"[INFO] saved {len(frame_paths)} frames → {frames_dir}")

    # panel
    panel_path = out_dir / "panel_all_steps.png"
    save_panel(panel_path, steps, Y_list, ids_list, id2color, xlim, ylim, title_prefix=title)
    print(f"[INFO] saved panel → {panel_path}")

    # gif
    gif_path = maybe_gif(frame_paths, out_dir / "evolution.gif", fps=2)
    if gif_path:
        print(f"[INFO] saved GIF → {gif_path}")

    # meta
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "exp": args.exp, "run": args.run,
            "method": args.method, "method_used": meth_used,
            "perplexity": args.perplexity, "seed": args.seed,
            "layer_idx": meta.get("layer_idx"), "model_name": meta.get("model_name"),
            "num_steps": len(steps), "steps": steps,
            "num_particles": int(common_ids.size), "particle_ids": common_ids.tolist(),
            "view_mode": args.view_mode, "view_index": args.view_index,
            "aligned_to_step0": bool(args.align),
            "symmetric_limits": bool(args.symmetric),
            "xlim": list(xlim), "ylim": list(ylim),
        }, f, indent=2)
    print(f"[INFO] wrote meta → {out_dir/'meta.json'}")

if __name__ == "__main__":
    main()

"""

python analysis/feature/analyse_particles_per_step_all.py \
  --base exp --exp exp_feature_layer --run BULL__feature_layer_last \
  --method tsne --perplexity 5 --view-mode mean
# Or pick a specific camera view:
# --view-mode index --view-index 0
# If you really want no alignment:
# --no-align

"""