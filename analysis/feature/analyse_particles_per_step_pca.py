#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global-PCA (fixed axes) visualization of particle distributions over steps.

Inputs under exp/<exp>/<run>/**/features/step_*.pt with keys:
  - particle_feats [N,D]  or
  - view_feats     [V,N,D]   (use --view-mode to aggregate)

Outputs under exp/<exp>/<run>/analysis_particles_pca:
  - frames/particles_step_XXXXXX.png  (square, global fixed limits)
  - panel_all_steps.png               (1 x S grid, fixed limits)
  - evolution.gif                     (animated, fixed limits)
  - meta.json                         (config + limits + steps)
"""

import argparse, glob, os, re, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- I/O ----------
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
    try:    return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: return torch.load(path, map_location="cpu")

def to_np(x):
    if isinstance(x, torch.Tensor): return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def load_per_step_particles(file_paths, view_mode="mean", view_index=0, max_steps=None):
    steps, Xs, meta = [], [], {}
    fps = file_paths[: (max_steps or len(file_paths))]
    for fp in fps:
        d = tload(fp)
        step = int(d.get("step", -1))

        if "particle_feats" in d:
            X = to_np(d["particle_feats"])  # [N,D]
        elif "view_feats" in d:
            VF = to_np(d["view_feats"])    # [V,N,D]
            if view_mode == "mean":
                X = VF.mean(0)
            elif view_mode == "first":
                X = VF[0]
            elif view_mode == "index":
                X = VF[int(view_index) % VF.shape[0]]
            else:
                raise ValueError("bad view_mode")
        elif "particle_feats_fp16" in d:
            X = to_np(d["particle_feats_fp16"])
        elif "view_feats_fp16" in d:
            VF = to_np(d["view_feats_fp16"])
            X = VF.mean(0)
        else:
            raise KeyError("No features found in file")

        # L2-normalize rows (cosine geometry) + sanitize
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-12] = 1.0
        X = X / n
        m = np.isfinite(X).all(axis=1)
        if (~m).any(): X = X[m]

        steps.append(step); Xs.append(X)
        if not meta:
            meta = {
                "model_name": d.get("model_name", None),
                "layer_idx": int(d.get("layer_idx", -1)),
                "N": int(X.shape[0]),
                "V": int(d.get("V", -1)),
            }
    return steps, Xs, meta

# ---------- PCA (global, fixed axes) ----------
def global_pca_basis(X_list, center="per-step"):
    """
    center:
      - 'per-step' : 각 step을 자기 평균으로 센터링 후 모두 concat → 전역 PCA
      - 'global'   : 전체 concat 후 하나의 전역 평균으로 센터링
    """
    if center == "per-step":
        mats = [X - X.mean(0, keepdims=True) for X in X_list]
        Xcat = np.concatenate(mats, 0)
    else:
        Xcat = np.concatenate(X_list, 0)
        Xcat = Xcat - Xcat.mean(0, keepdims=True)

    # SVD for top-2 components
    U, S, Vt = np.linalg.svd(Xcat, full_matrices=False)
    W = Vt[:2].T  # [D,2]
    return W

def project_on(W, X):
    Xc = X - X.mean(0, keepdims=True)  # step-wise centering for display
    return (Xc @ W).astype(np.float32)  # [N,2]

# ---------- plotting ----------
def stable_colors(N): return [i % 10 for i in range(N)]

def global_limits(Ys, symmetric=True, pad=0.05):
    xs = np.concatenate([Y[:,0] for Y in Ys]); ys = np.concatenate([Y[:,1] for Y in Ys])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if symmetric:
        m = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) * (1.0 + pad)
        return (-m, m), (-m, m)
    wx = xmax - xmin; wy = ymax - ymin
    return (xmin - pad*wx, xmax + pad*wx), (ymin - pad*wy, ymax + pad*wy)

def save_frames(frames_dir: Path, steps, Y_list, xlim, ylim, title, annotate=True):
    frames_dir.mkdir(parents=True, exist_ok=True)
    N = Y_list[0].shape[0]; cols = stable_colors(N)
    paths = []
    for s, Y in zip(steps, Y_list):
        fig, ax = plt.subplots(figsize=(4,4), dpi=160)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(Y[:,0], Y[:,1], c=[cols[i] for i in range(Y.shape[0])],
                   cmap="tab10", s=60, alpha=0.95, edgecolors="none")
        if annotate:
            for i,(x,y) in enumerate(Y): ax.text(x,y,str(i),fontsize=8,ha="center",va="center",alpha=0.9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
        ax.grid(True, alpha=0.3); ax.set_title(f"Step {s}")
        fig.tight_layout()
        fp = frames_dir / f"particles_step_{s:06d}.png"
        fig.savefig(fp); plt.close(fig)
        paths.append(fp)
    return paths

def save_panel_pages(panel_base_path: Path, steps, Y_list, xlim, ylim, title, rows=1, cols=None):
    if cols is None:
        cols = len(steps)
    per_page = max(1, rows * cols)
    total = len(steps)
    page_paths = []
    N = Y_list[0].shape[0]; col_idx = stable_colors(N)

    for page_idx, start in enumerate(range(0, total, per_page), start=1):
        end = min(start + per_page, total)
        steps_page = steps[start:end]
        Y_page = Y_list[start:end]
        r = rows
        c = min(cols, len(steps_page)) if rows == 1 else cols
        fig, axes = plt.subplots(r, c, figsize=(c*3.6, r*3.6), dpi=160)
        axes = np.atleast_2d(axes)

        for i,(s,Y) in enumerate(zip(steps_page, Y_page)):
            rr, cc = divmod(i, c)
            ax = axes[rr, cc]
            ax.set_aspect("equal", adjustable="box")
            ax.scatter(Y[:,0], Y[:,1], c=[col_idx[j] for j in range(Y.shape[0])],
                       cmap="tab10", s=50, alpha=0.95, edgecolors="none")
            for j,(x,y) in enumerate(Y): ax.text(x,y,str(j),fontsize=8,ha="center",va="center",alpha=0.9)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(True, alpha=0.3)
            ax.set_title(f"Step {s}", fontsize=10)
            if i==0: ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
        # hide unused axes
        total_cells = r*c
        for k in range(len(steps_page), total_cells):
            rr, cc = divmod(k, c)
            axes[rr, cc].axis("off")

        fig.tight_layout()
        # first page also saved as panel_all_steps.png for backward compatibility
        if page_idx == 1:
            first_path = panel_base_path.with_name("panel_all_steps.png")
            fig.savefig(first_path)
            page_paths.append(first_path)
        page_path = panel_base_path.with_name(f"panel_p{page_idx:03d}.png")
        fig.savefig(page_path)
        page_paths.append(page_path)
        plt.close(fig)
    return page_paths


def save_panel_single(panel_path: Path, steps, Y_list, xlim, ylim, title, rows=2, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.6), dpi=160)
    axes = np.atleast_2d(axes)
    N = Y_list[0].shape[0] if Y_list else 0
    col_idx = stable_colors(N)

    for i,(s,Y) in enumerate(zip(steps, Y_list)):
        rr, cc = divmod(i, cols)
        ax = axes[rr, cc]
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(Y[:,0], Y[:,1], c=[col_idx[j] for j in range(Y.shape[0])],
                   cmap="tab10", s=50, alpha=0.95, edgecolors="none")
        for j,(x,y) in enumerate(Y):
            ax.text(x,y,str(j),fontsize=8,ha="center",va="center",alpha=0.9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(True, alpha=0.3)
        ax.set_title(f"Step {s}", fontsize=10)
        if rr == rows-1:
            ax.set_xlabel("PC-1")
        if cc == 0:
            ax.set_ylabel("PC-2")

    # hide unused axes
    total_cells = rows*cols
    for k in range(len(steps), total_cells):
        rr, cc = divmod(k, cols)
        axes[rr, cc].axis("off")

    fig.tight_layout()
    fig.savefig(panel_path)
    plt.close(fig)
    return panel_path

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
    ap.add_argument("--exp",  type=str, default="exp6_ours_best_feature")
    ap.add_argument("--run",  type=str, required=True)
    ap.add_argument("--view-mode", type=str, default="mean", choices=["mean","first","index"])
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--symmetric", action="store_true", default=True)
    ap.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    ap.add_argument("--center", type=str, default="per-step", choices=["per-step","global"],
                    help="how to center before building the global PCA basis")
    # output controls
    ap.add_argument("--output-root", type=str, default="results/features",
                    help="Root directory for results. If --category is given, saves under <output-root>/<category>/<run>/analysis_particles_pca")
    ap.add_argument("--category", type=str, default=None, choices=["ours","baseline"],
                    help="If provided, route outputs to results/features/{category}/<run>/analysis_particles_pca. If omitted, infer from run name (RLSD -> ours; otherwise baseline).")
    ap.add_argument("--step-interval", type=int, default=None,
                    help="If set (e.g., 200), subsample steps approximately every given interval (keeps steps where step % interval == 0, plus first/last if needed)")
    ap.add_argument("--panel-rows", type=int, default=1)
    ap.add_argument("--panel-cols", type=int, default=None,
                    help="Number of columns in panel; default uses all selected steps on one row. Use with --panel-rows for grids like 2x3.")
    ap.add_argument("--panel-steps", nargs='*', type=int, default=None,
                    help="Explicit list of step numbers to show in the panel (e.g., 1 200 400 600 800 1000). Overrides --step-interval selection.")
    args = ap.parse_args()

    base = Path(args.base)

    # decide output directory
    category = args.category
    if category is None:
        category = "ours" if ("RLSD" in args.run or "rlsd" in args.run.lower()) else "baseline"

    if args.output_root:
        out_root = Path(args.output_root) / category / args.run
    else:
        out_root = base / args.exp / args.run

    out_dir = out_root / "analysis_particles_pca"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pt_files = find_feature_files(base, args.exp, args.run)
    print(f"[INFO] found {len(pt_files)} step files")

    steps, Xs, meta = load_per_step_particles(pt_files, args.view_mode, args.view_index, args.max_steps)
    print(f"[INFO] steps: {len(steps)}, particles/step≈{meta.get('N')} dim={Xs[0].shape[1]}")

    # global fixed PCA axes
    W = global_pca_basis(Xs, center=args.center)  # [D,2]
    Y_list = [project_on(W, X) for X in Xs]

    # global fixed limits (square)
    xlim, ylim = global_limits(Y_list, symmetric=args.symmetric, pad=0.05)

    title = f"{args.exp}/{args.run} — Global PCA on particle features (layer={meta.get('layer_idx')}, agg={args.view_mode})"

    # select steps for panel
    steps_plot, Y_plot = steps, Y_list
    if args.panel_steps:
        # explicit selection; keep order provided
        explicit = []
        for target in args.panel_steps:
            if target in steps:
                idx = steps.index(target)
                explicit.append((target, Y_list[idx]))
        if explicit:
            steps_plot = [s for s,_ in explicit]
            Y_plot = [y for _,y in explicit]
    elif args.step_interval is not None and args.step_interval > 0:
        # interval-based selection; fill a grid if rows*cols specified
        selected = []
        # always include the first step
        if steps:
            selected.append(steps[0])
        for s in steps:
            if (s % args.step_interval) == 0 and s not in selected:
                selected.append(s)
        # ensure last included
        if steps and steps[-1] not in selected:
            selected.append(steps[-1])
        selected = sorted(set(selected))
        steps_plot = selected
        Y_plot = [Y_list[steps.index(s)] for s in steps_plot]

    # save
    frames = save_frames(frames_dir, steps_plot, Y_plot, xlim, ylim, title, annotate=True)
    # If rows*cols is provided and finite, create a single panel trimmed/padded to fit
    if args.panel_rows and args.panel_cols:
        k = int(args.panel_rows) * int(args.panel_cols)
        steps_panel = steps_plot[:k]
        Y_panel = Y_plot[:k]
        save_panel_single(out_dir / "panel_all_steps.png", steps_panel, Y_panel, xlim, ylim, title,
                          rows=int(args.panel_rows), cols=int(args.panel_cols))
    else:
        save_panel_pages(out_dir / "panel_all_steps.png", steps_plot, Y_plot, xlim, ylim, title,
                         rows=int(args.panel_rows), cols=(int(args.panel_cols) if args.panel_cols else None))
    gif_path = maybe_gif(frames, out_dir / "evolution.gif", fps=2)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "exp": args.exp, "run": args.run,
            "layer_idx": int(meta.get("layer_idx",-1)), "model_name": meta.get("model_name"),
            "view_mode": args.view_mode, "view_index": args.view_index,
            "num_steps": len(steps), "steps": steps,
            "center_for_basis": args.center,
            "symmetric_limits": bool(args.symmetric),
            "xlim": list(xlim), "ylim": list(ylim),
        }, f, indent=2)

    print(f"[INFO] saved frames→{frames_dir}")
    print(f"[INFO] saved panel pages → {out_dir}")
    if gif_path: print(f"[INFO] saved GIF   → {gif_path}")

if __name__ == "__main__":
    main()


"""

python analysis/feature/analyse_particles_per_step_pca.py \
  --base exp --exp exp_feature_layer --run BULL__feature_layer_last \
  --view-mode mean --center per-step
# 특정 뷰만 쓰려면:
# --view-mode index --view-index 0


(torch-gpu) -----------------------------------------------------------------------------------------
~/3D-Generation (main*) » python analysis/feature/analyse_particles_per_step_pca.py \
  --run WO__ICE__S42 --category baseline \       
  --view-mode mean \
  --panel-steps 1 200 400 600 800 1000 \
  --panel-rows 2 --panel-cols 3
[INFO] found 11 step files
[INFO] steps: 11, particles/step≈8 dim=768
[INFO] saved frames→results/features/baseline/WO__ICE__S42/analysis_particles_pca/frames
[INFO] saved panel pages → results/features/baseline/WO__ICE__S42/analysis_particles_pca
[INFO] saved GIF   → results/features/baseline/WO__ICE__S42/analysis_particles_pca/evolution.gif
(torch-gpu) -----------------------------------------------------------------------------------------
~/3D-Generation (main*) » python analysis/feature/analyse_particles_per_step_pca.py \
  --run RLSD__RBF__ICE__S42 --category ours \    
  --view-mode mean \
  --panel-steps 1 200 400 600 800 1000 \
  --panel-rows 2 --panel-cols 3
[INFO] found 11 step files
[INFO] steps: 11, particles/step≈8 dim=768
[INFO] saved frames→results/features/ours/RLSD__RBF__ICE__S42/analysis_particles_pca/frames
[INFO] saved panel pages → results/features/ours/RLSD__RBF__ICE__S42/analysis_particles_pca
[INFO] saved GIF   → results/features/ours/RLSD__RBF__ICE__S42/analysis_particles_pca/evolution.gif
(torch-gpu) ----------------------------------------------------------------------------------------

"""