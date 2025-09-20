#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple runs (e.g., early/mid/last) at a given step on one 2D plane.
- Loads each: exp/<RUN_PATH>/features/step_<STEP>.pt
- Aggregates [V,N,D] -> [N,D] by view mean (or select index)
- Jointly fits PCA (default) or t-SNE across all runs
- Saves scatter plot + meta.json under results/features


# Plotting 
python analysis/feature/compare_runs_joint_embedding.py \
  --panel-steps 1 200 400 600 800 1000 --dpi 300 --figsize 6 6
  
python analysis/feature/compare_runs_joint_embedding.py --step 001000


# Metrics only
python analysis/feature/compare_runs_joint_embedding.py --metrics

"""

import argparse, json, re, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
from typing import List, Tuple
from matplotlib.lines import Line2D
try:
    from analysis.feature.diversity import diversity_trace
except Exception:
    # Fallback if running as a plain script without package context
    import numpy as _np
    def diversity_trace(Y2d: _np.ndarray) -> float:
        if Y2d is None or Y2d.size == 0:
            return float("nan")
        Yc = Y2d - Y2d.mean(axis=0, keepdims=True)
        C = _np.cov(Yc, rowvar=False)
        return float(_np.trace(C))
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

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

def apply_publication_style(base_font_size=12):
    plt.rcParams.update({
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 2,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_palette_and_markers(num: int, preferred: List[str] = None) -> Tuple[List[str], List[str]]:
    if preferred is not None and len(preferred) >= num:
        colors = preferred[:num]
    else:
        colors = [
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ][:num]
    markers = ["o","^","s","D","P","X","v","<",">","*"][:num]
    return colors, markers

def square_limits(Y_list, pad=0.05):
    Y_all = np.concatenate(Y_list, 0)
    xmin, xmax = float(Y_all[:,0].min()), float(Y_all[:,0].max())
    ymin, ymax = float(Y_all[:,1].min()), float(Y_all[:,1].max())
    m = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    m = (1.0 + pad) * (m if m > 0 else 1.0)
    return (-m, m), (-m, m)


def _global_limits_over_steps(Y_steps_list: List[List[np.ndarray]], pad: float = 0.05):
    all_Y = []
    for Y_list in Y_steps_list:
        for Y in Y_list:
            if Y is not None and Y.size > 0:
                all_Y.append(Y)
    if not all_Y:
        return (-1.0, 1.0), (-1.0, 1.0)
    Y_all = np.concatenate(all_Y, 0)
    xmin, xmax = float(Y_all[:,0].min()), float(Y_all[:,0].max())
    ymin, ymax = float(Y_all[:,1].min()), float(Y_all[:,1].max())
    m = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    m = (1.0 + pad) * (m if m > 0 else 1.0)
    return (-m, m), (-m, m)

def _centroid(Y: np.ndarray) -> np.ndarray:
    return Y.mean(axis=0, keepdims=False)


def _cov_ellipse_params(Y: np.ndarray, n_std: float = 2.0) -> Tuple[float, float, float, np.ndarray]:
    C = np.cov(Y, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width = 2.0 * n_std * np.sqrt(max(vals[0], 1e-12))
    height = 2.0 * n_std * np.sqrt(max(vals[1], 1e-12))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = _centroid(Y)
    return width, height, angle, center


def _maybe_draw_convex_hull(ax, Y: np.ndarray, color: str, alpha: float = 0.08):
    if not HAS_SCIPY:
        return
    if Y.shape[0] < 3:
        return
    try:
        hull = ConvexHull(Y)
        poly = Y[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, linewidth=0)
    except Exception:
        return


def plot_runs(
    Y_list: List[np.ndarray],
    labels: List[str],
    out_base: Path,
    title: str,
    annotate: bool = False,
    point_size: float = 60.0,
    point_alpha: float = 0.9,
    draw_centroid: bool = True,
    draw_ellipse: bool = False,
    ellipse_std: float = 2.0,
    draw_hull: bool = False,
    figsize: Tuple[float, float] = (6.0, 6.0),
    legend_loc: str = "best",
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    save_pdf: bool = True,
    save_svg: bool = True,
    dpi: int = 300,
    transparent: bool = False,
    palette: List[str] = None,
):
    if xlim is None or ylim is None:
        xlim, ylim = square_limits(Y_list, pad=0.05)

    colors, markers = get_palette_and_markers(len(Y_list), preferred=palette)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", adjustable="box")

    for i, (Y, lab) in enumerate(zip(Y_list, labels)):
        c = colors[i]
        m = markers[i]
        ax.scatter(
            Y[:,0], Y[:,1], s=point_size, alpha=point_alpha, c=c, marker=m,
            label=lab, edgecolors="none",
        )
        if draw_hull:
            _maybe_draw_convex_hull(ax, Y, color=c, alpha=0.08)
        if draw_ellipse and Y.shape[0] >= 2:
            try:
                w, h, a, ctr = _cov_ellipse_params(Y, n_std=ellipse_std)
                e = Ellipse(xy=(ctr[0], ctr[1]), width=w, height=h, angle=a,
                            edgecolor=c, facecolor=c, alpha=0.10, linewidth=1.2)
                ax.add_patch(e)
            except Exception:
                pass
        if draw_centroid:
            ctr = _centroid(Y)
            ax.scatter([ctr[0]], [ctr[1]], s=point_size*1.8, c=c, marker="X", edgecolors="white", linewidths=0.8, zorder=5)

        if annotate:
            for j, (x, y) in enumerate(Y):
                ax.text(x, y, f"{j}", fontsize=7, ha="center", va="center", alpha=0.8, color=c)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc=legend_loc)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    fig.savefig(png_path, transparent=transparent)
    if save_pdf:
        fig.savefig(out_base.with_suffix(".pdf"), transparent=transparent)
    if save_svg:
        fig.savefig(out_base.with_suffix(".svg"), transparent=transparent)
    plt.close(fig)
    return xlim, ylim


    

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=str, required=False, default=None, help="e.g. 001000 (6 digits) or int; required unless --time-series is set")
    ap.add_argument("--runs", nargs="+", required=False,
                    default=[
                        "exp6_ours_best_feature/WO__ICE__S42",
                        "exp6_ours_best_feature/RLSD__RBF__ICE__S42",
                    ],
                    help="list of run directories under exp/ (default: baseline WO__ICE__S42 vs ours RLSD__RBF__ICE__S42)")
    ap.add_argument("--labels", nargs="*", default=["Baseline", "Ours"],
                    help="Labels for runs (default: Baseline Ours)")
    ap.add_argument("--method", type=str, default="pca", choices=["pca","tsne"])
    ap.add_argument("--perplexity", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view-mode", type=str, default="mean", choices=["mean","first","index"])
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results/features/comparison")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--style", type=str, default="paper", choices=["paper","default"])
    ap.add_argument("--point-size", type=float, default=48.0)
    ap.add_argument("--alpha", type=float, default=0.85)
    # toggles default to True with disable flags
    ap.add_argument("--centroid", dest="centroid", action="store_true")
    ap.add_argument("--no-centroid", dest="centroid", action="store_false")
    ap.set_defaults(centroid=True)
    ap.add_argument("--ellipse", dest="ellipse", action="store_true")
    ap.add_argument("--no-ellipse", dest="ellipse", action="store_false")
    ap.set_defaults(ellipse=True)
    ap.add_argument("--ellipse-std", type=float, default=2.0)
    ap.add_argument("--hull", dest="hull", action="store_true")
    ap.add_argument("--no-hull", dest="hull", action="store_false")
    ap.set_defaults(hull=True)
    ap.add_argument("--legend-loc", nargs="+", type=str, default=["upper", "right"])  # default to upper right
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figsize", nargs=2, type=float, default=[6.0, 6.0])
    ap.add_argument("--transparent", dest="transparent", action="store_true")
    ap.add_argument("--no-transparent", dest="transparent", action="store_false")
    ap.set_defaults(transparent=True)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--palette", nargs="*", default=None, help="Optional list of hex colors to use for runs")
    ap.add_argument("--metrics", action="store_true", help="Compute diversity across all common steps and save CSV only (no plots)")
    # optional panel across multiple steps (keeps per-step output)
    ap.add_argument("--panel-steps", nargs="*", type=int, default=None,
                    help="If provided, also generate a 2x3 panel for these steps (e.g., 1 200 400 600 800 1000)")
    ap.add_argument("--panel-rows", type=int, default=2)
    ap.add_argument("--panel-cols", type=int, default=3)
    ap.add_argument("--panel-clean", dest="panel_clean", action="store_true",
                    help="Clean panel styling: only left column shows y-labels and bottom row shows x-labels; hide ticks elsewhere")
    ap.add_argument("--no-panel-clean", dest="panel_clean", action="store_false")
    ap.set_defaults(panel_clean=True)
    args = ap.parse_args()

    if args.style == "paper":
        apply_publication_style(base_font_size=12)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote_single = False
    out_png = None
    legend_loc_value = args.legend_loc
    if isinstance(legend_loc_value, list):
        legend_loc_value = " ".join(legend_loc_value)
    legend_loc_value = legend_loc_value.replace("_", " ")

    # -------- metrics-only mode across all steps --------
    if args.metrics and args.step is None:
        def _list_steps(run_dir: Path):
            feats_dir = run_dir / "features"
            steps = []
            if feats_dir.exists():
                for p in feats_dir.glob("step_*.pt"):
                    m = re.search(r"step_(\d+)\.pt$", p.name)
                    if m:
                        steps.append(int(m.group(1)))
            return sorted(set(steps))

        run_a = Path("exp") / args.runs[0]
        run_b = Path("exp") / args.runs[1]
        steps_a = _list_steps(run_a)
        steps_b = _list_steps(run_b)
        common_steps = [s for s in steps_a if s in set(steps_b)]
        if not common_steps:
            raise SystemExit("No common steps found for metrics computation")

        csv_path = out_dir / "compare_baseline_vs_ours.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step","baseline_diversity_trace","ours_diversity_trace","delta","improvement_pct"]) 
            for s in common_steps:
                Xa, _ = load_features(find_step_file(run_a, f"{s:06d}"), view_mode=args.view_mode, view_index=args.view_index)
                Xb, _ = load_features(find_step_file(run_b, f"{s:06d}"), view_mode=args.view_mode, view_index=args.view_index)
                Y_list_s, _ = joint_embed([Xa, Xb], method="pca", seed=args.seed, perplexity=args.perplexity)
                db = diversity_trace(Y_list_s[0])
                do = diversity_trace(Y_list_s[1])
                delta = float(do - db) if np.isfinite(db) and np.isfinite(do) else float("nan")
                impr = float((do - db) / (db + 1e-12) * 100.0) if np.isfinite(db) and np.isfinite(do) else float("nan")
                w.writerow([f"{s:06d}", f"{db:.6g}", f"{do:.6g}", f"{delta:.6g}", f"{impr:.6g}"])
        print(f"[INFO] saved diversity timeseries CSV → {csv_path}")
        return

    X_list, metas, labels = [], [], []
    if args.step is not None:
        for r in args.runs:
            run_dir = Path("exp") / r
            fp = find_step_file(run_dir, args.step)
            X, meta = load_features(fp, view_mode=args.view_mode, view_index=args.view_index)
            X_list.append(X); metas.append(meta); labels.append(run_dir.name)
        if args.labels is not None and len(args.labels) == len(labels):
            labels = list(args.labels)

        # joint embedding for single step
        Y_list, info = joint_embed(X_list, method=args.method, seed=args.seed, perplexity=args.perplexity)

        # title & plot for single step
        title = args.title if args.title is not None else None
        out_base = out_dir / f"compare_runs_{info['method'].lower()}_{int(args.step):06d}"
        xlim, ylim = plot_runs(
            Y_list,
            labels,
            out_base,
            title,
            annotate=False,
            point_size=float(args.point_size),
            point_alpha=float(args.alpha),
            draw_centroid=bool(args.centroid),
            draw_ellipse=bool(args.ellipse),
            ellipse_std=float(args.ellipse_std),
            draw_hull=bool(args.hull),
            figsize=(float(args.figsize[0]), float(args.figsize[1])),
            legend_loc=str(legend_loc_value),
            save_pdf=not args.no_pdf,
            save_svg=not args.no_svg,
            dpi=int(args.dpi),
            transparent=bool(args.transparent),
            palette=list(args.palette) if args.palette else None,
        )
        out_png = out_base.with_suffix(".png")
        # diversity-only metrics
        db = diversity_trace(Y_list[0])
        do = diversity_trace(Y_list[1]) if len(Y_list) > 1 else float("nan")
        delta = float(do - db) if np.isfinite(db) and np.isfinite(do) else float("nan")
        impr = float((do - db) / (db + 1e-12) * 100.0) if np.isfinite(db) and np.isfinite(do) else float("nan")
        analysis = {
            "diversity_trace": {labels[0]: float(db), labels[1]: float(do)},
            "diversity_delta": delta,
            "diversity_improvement_pct": impr,
        }
        wrote_single = True

    # optional panel generation
    if args.panel_steps:
        steps = [int(s) for s in args.panel_steps]
        Y_steps_list: List[List[np.ndarray]] = []
        for s in steps:
            X_list_step = []
            for r in args.runs:
                run_dir = Path("exp") / r
                fp = find_step_file(run_dir, f"{int(s):06d}")
                X, _ = load_features(fp, view_mode=args.view_mode, view_index=args.view_index)
                X_list_step.append(X)
            Y_list_step, _ = joint_embed(X_list_step, method=args.method, seed=args.seed, perplexity=args.perplexity)
            Y_steps_list.append(Y_list_step)

        panel_base = out_dir / f"compare_runs_{args.method.lower()}_panel"
        # derive labels for panel mode (use provided labels or default to run dir names)
        labels_panel = None
        if args.labels is not None and len(args.labels) == len(args.runs):
            labels_panel = list(args.labels)
        else:
            labels_panel = [(Path("exp") / r).name for r in args.runs]
        colors, markers = get_palette_and_markers(len(labels_panel), preferred=(list(args.palette) if args.palette else None))
        # compute global limits across all panel steps
        xlim_p, ylim_p = _global_limits_over_steps(Y_steps_list, pad=0.05)
        # Use full per-subplot figsize for higher resolution panels
        fig_w = int(args.panel_cols) * float(args.figsize[0])
        fig_h = int(args.panel_rows) * float(args.figsize[1])
        fig, axes = plt.subplots(int(args.panel_rows), int(args.panel_cols), figsize=(fig_w, fig_h), dpi=int(args.dpi))
        axes = np.atleast_2d(axes)
        for idx, (s, Y_list_s) in enumerate(zip(steps, Y_steps_list)):
            r, c = divmod(idx, int(args.panel_cols))
            if r >= int(args.panel_rows):
                break
            ax = axes[r, c]
            ax.set_aspect("equal", adjustable="box")
            for i, (Y, lab) in enumerate(zip(Y_list_s, labels_panel)):
                col = colors[i]; mark = markers[i]
                ax.scatter(Y[:,0], Y[:,1], s=float(args.point_size), alpha=float(args.alpha), c=col, marker=mark, edgecolors="none")
                if bool(args.hull):
                    _maybe_draw_convex_hull(ax, Y, color=col, alpha=0.08)
                if bool(args.ellipse) and Y.shape[0] >= 2:
                    try:
                        w, h, a, ctr = _cov_ellipse_params(Y, n_std=float(args.ellipse_std))
                        e = Ellipse(xy=(ctr[0], ctr[1]), width=w, height=h, angle=a, edgecolor=col, facecolor=col, alpha=0.10, linewidth=1.2)
                        ax.add_patch(e)
                    except Exception:
                        pass
                if bool(args.centroid):
                    ctr = _centroid(Y)
                    ax.scatter([ctr[0]], [ctr[1]], s=float(args.point_size)*1.8, c=col, marker="X", edgecolors="white", linewidths=0.8, zorder=5)
            ax.set_xlim(*xlim_p); ax.set_ylim(*ylim_p)
            # Clean panel styling: show fewer labels/ticks
            if args.panel_clean:
                if r < int(args.panel_rows) - 1:
                    ax.set_xticklabels([])
                if c > 0:
                    ax.set_yticklabels([])
                if r == int(args.panel_rows) - 1:
                    ax.set_xlabel("dim-1")
                if c == 0:
                    ax.set_ylabel("dim-2")
            else:
                ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
            ax.set_title(f"Step {int(s)}")
            ax.grid(True, alpha=0.25)
        # hide unused axes
        total_cells = int(args.panel_rows) * int(args.panel_cols)
        for k in range(len(steps), total_cells):
            r, c = divmod(k, int(args.panel_cols))
            axes[r, c].axis("off")
        # legend with explicit handles to ensure correct colors/markers
        legend_handles = [
            Line2D([0], [0], marker=markers[i], color='none', markerfacecolor=colors[i],
                   markersize=8, linestyle='None', label=labels_panel[i])
            for i in range(len(labels_panel))
        ]
        axes[0,0].legend(handles=legend_handles, frameon=False, loc=str(legend_loc_value))
        fig.tight_layout()
        fig.savefig(panel_base.with_suffix(".png"), transparent=bool(args.transparent))
        fig.savefig(panel_base.with_suffix(".pdf"), transparent=bool(args.transparent))
        fig.savefig(panel_base.with_suffix(".svg"), transparent=bool(args.transparent))
        plt.close(fig)

    # meta
    if wrote_single:
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
            "output_pdf": None if args.no_pdf else str(out_base.with_suffix(".pdf")),
            "output_svg": None if args.no_svg else str(out_base.with_suffix(".svg")),
            "style": args.style,
            "point_size": float(args.point_size),
            "alpha": float(args.alpha),
            "centroid": bool(args.centroid),
            "ellipse": bool(args.ellipse),
            "ellipse_std": float(args.ellipse_std),
            "hull": bool(args.hull),
            "legend_loc": str(legend_loc_value),
            "dpi": int(args.dpi),
            "figsize": [float(args.figsize[0]), float(args.figsize[1])],
            "transparent": bool(args.transparent),
            "palette": list(args.palette) if args.palette else None,
            "analysis": analysis,
        }
        meta_name = f"compare_runs_{info['method'].lower()}_{int(args.step):06d}.meta.json"
        with open(out_dir / meta_name, "w") as f:
            json.dump(meta_out, f, indent=2)
        # Save diversity to CSV (append or create)
        csv_path = out_dir / f"compare_runs_diversity_{int(args.step):06d}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step","baseline_diversity_trace","ours_diversity_trace","delta","improvement_pct"])
            w.writerow([
                f"{int(args.step):06d}",
                f"{analysis['diversity_trace'][labels[0]]:.6g}",
                f"{analysis['diversity_trace'][labels[1]]:.6g}",
                f"{analysis['diversity_delta']:.6g}",
                f"{analysis['diversity_improvement_pct']:.6g}",
            ])
        print(f"[INFO] saved CSV → {csv_path}")
        print(f"[INFO] saved → {out_png}")
        print(f"[INFO] saved meta → {out_dir}")
    else:
        print("[INFO] panel-only mode completed.")

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
  
  
python analysis/feature/compare_runs_joint_embedding.py \
  --step 001000 \
  --runs exp6_ours_best_feature/WO__ICE__S42 exp6_ours_best_feature/RLSD__RBF__ICE__S42 \
  --labels "Baseline" "Ours" \
  --method pca --style paper \
  --centroid --ellipse --ellipse-std 2.0 --hull \
  --point-size 48 --alpha 0.85 \
  --legend-loc upper right \
  --dpi 300 --figsize 6 6 \
  --transparent

"""

