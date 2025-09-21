import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(paths: List[str]):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def list_experiment_dirs(base_exp_dir: Path) -> List[Tuple[str, Path]]:
    """
    Return (label, path) pairs for experiments 1-5 under exp/.
    Maps directory names to readable labels.
    """
    name_map = {
        "exp1_repulsion_kernel": "Exp 1",
        "exp2_lambda_coarse": "Exp 2",
        "exp3_lambda_fine": "Exp 3",
        "exp4_guidance_scale": "Exp 4",
        "exp5_rbf_beta": "Exp 5",
    }
    found: List[Tuple[str, Path]] = []
    for dir_name, label in name_map.items():
        p = base_exp_dir / dir_name
        if p.exists() and p.is_dir():
            found.append((label, p))
    return found


def collect_run_efficiency(run_dir: Path) -> Dict[str, float]:
    """
    Read run_dir/metrics/efficiency.csv and return mean across steps
    for time and memory columns if present.
    """
    csv_path = run_dir / "metrics" / "efficiency.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    cols = [
        "step_wall_ms",
        "step_gpu_ms",
        "render_ms",
        "sd_guidance_ms",
        "backprop_ms",
        "densify_ms",
        "memory_allocated_mb",
        "max_memory_allocated_mb",
    ]
    present = [c for c in cols if c in df.columns]
    if not present:
        return {}
    means = {c: float(df[c].mean()) for c in present}
    # Derived: total of component times if available
    comp_cols = [c for c in ["render_ms", "sd_guidance_ms", "backprop_ms", "densify_ms"] if c in df.columns]
    if comp_cols:
        means["total_components_ms"] = float(df[comp_cols].mean().sum())
    return means


def aggregate_experiment(exp_dir: Path) -> Dict[str, float]:
    """
    Aggregate means across all runs (subdirectories that aren't logs or hidden).
    """
    per_run: List[Dict[str, float]] = []
    for cfg in sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'logs']):
        stats = collect_run_efficiency(cfg)
        if stats:
            per_run.append(stats)
    if not per_run:
        return {}
    # Union of keys across runs; compute mean ignoring missing per run
    all_keys: List[str] = sorted({k for r in per_run for k in r.keys()})
    out: Dict[str, float] = {}
    for k in all_keys:
        vals = [r[k] for r in per_run if k in r]
        if vals:
            out[k] = float(pd.Series(vals).mean())
    return out


def pretty(metric: str) -> str:
    mapping = {
        "step_wall_ms": "Step time (ms) (↓)",
        "step_gpu_ms": "GPU step time (ms) (↓)",
        "render_ms": "Render (ms) (↓)",
        "sd_guidance_ms": "Guidance (ms) (↓)",
        "backprop_ms": "Backprop (ms) (↓)",
        "densify_ms": "Densify (ms) (↓)",
        "memory_allocated_mb": "Memory (MB) (↓)",
        "max_memory_allocated_mb": "Peak memory (MB) (↓)",
        "total_components_ms": "Sum components (ms) (↓)",
    }
    return mapping.get(metric, metric)


def plot_bars_by_experiment(df: pd.DataFrame, metric: str, out_png: Path):
    labels = df["experiment"].tolist()
    vals = df[metric].tolist()
    plt.figure(figsize=(max(6.5, len(labels) * 1.1), 4.0))
    plt.bar(labels, vals, color="#4C78A8", edgecolor="#2f2f2f", linewidth=0.6)
    plt.ylabel(pretty(metric))
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stacked_time_components(df: pd.DataFrame, out_png: Path):
    comps = [c for c in ["render_ms", "sd_guidance_ms", "backprop_ms", "densify_ms"] if c in df.columns]
    if not comps:
        return
    labels = df["experiment"].tolist()
    plt.figure(figsize=(max(7.5, len(labels) * 1.2), 4.2))
    bottom = [0.0] * len(labels)
    colors = {
        "render_ms": "#4C78A8",
        "sd_guidance_ms": "#72B7B2",
        "backprop_ms": "#F58518",
        "densify_ms": "#E45756",
    }
    for c in comps:
        vals = df[c].tolist()
        # Legend labels without " ms"
        label_clean = c.replace('_', ' ').replace(' ms', '')
        plt.bar(labels, vals, bottom=bottom, label=label_clean, color=colors.get(c, None), edgecolor="#2f2f2f", linewidth=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=0, ha='center')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(comps), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="/Users/sj/3D-Generation/exp")
    parser.add_argument("--out_dir", type=str, default="/Users/sj/3D-Generation/results/efficiency")
    args = parser.parse_args()

    ensure_dirs([args.out_dir])

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#d0d0d0',
        'grid.alpha': 0.25,
    })

    base = Path(args.exp_dir)
    pairs = list_experiment_dirs(base)
    rows: List[Dict[str, float]] = []
    for label, p in pairs:
        stats = aggregate_experiment(p)
        if stats:
            stats_out: Dict[str, float] = {"experiment": label}
            stats_out.update(stats)
            rows.append(stats_out)

    if not rows:
        print("No efficiency data found under experiments 1-5.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("experiment", inplace=True)

    # Save CSV of aggregated means per experiment
    df.to_csv(Path(args.out_dir) / "efficiency_aggregates_by_experiment.csv", index=False)

    # Key plots
    if "step_wall_ms" in df.columns:
        plot_bars_by_experiment(df, "step_wall_ms", Path(args.out_dir) / "time_overall_by_experiment.png")
    if "max_memory_allocated_mb" in df.columns:
        plot_bars_by_experiment(df, "max_memory_allocated_mb", Path(args.out_dir) / "memory_overall_by_experiment.png")

    # Stacked components plot (if present)
    plot_stacked_time_components(df, Path(args.out_dir) / "time_components_overall_by_experiment.png")

    print("Saved figures and CSV to", args.out_dir)


if __name__ == "__main__":
    main()


