import argparse
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(paths: List[str]):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def parse_prompt_and_seed(dir_name: str):
    m = re.match(r"^(?P<prompt>[A-Z]+)__S(?P<seed>\d+)$", dir_name)
    if not m:
        return None, None
    return m.group("prompt"), int(m.group("seed"))


def _read_prompt_from_config(run_dir: Path) -> str:
    cfg_yaml = run_dir / "config.yaml"
    if not cfg_yaml.exists():
        return ""
    try:
        with open(cfg_yaml, 'r') as f:
            for line in f:
                ls = line.strip()
                if not ls or ls.startswith('#'):
                    continue
                if ls.startswith('prompt:'):
                    val = ls.split(':', 1)[1].strip()
                    if '#' in val:
                        val = val.split('#', 1)[0].strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    return val
    except Exception:
        return ""
    return ""


def _keyword_from_prompt(prompt_text: str, folder_code: str) -> str:
    if prompt_text:
        txt = prompt_text.strip().lower()
        if 'sundae' in txt:
            return 'sundae'
        if 'ice cream' in txt or 'icecream' in txt:
            return 'icecream'
        if 'hamburger' in txt:
            return 'hamburger'
        if 'bulldozer' in txt:
            return 'bulldozer'
        if 'cactus' in txt:
            return 'cactus'
        if 'tulip' in txt:
            return 'tulip'
        if ' of ' in txt:
            txt = txt.split(' of ')[-1].strip()
        for det in ("a ", "an ", "the "):
            if txt.startswith(det):
                txt = txt[len(det):]
        txt = txt.replace('.', '').replace(',', '').replace('  ', ' ').strip()
        words = [w for w in txt.split(' ') if w]
        if len(words) >= 1:
            txt = ' '.join(words[:3])
        if txt:
            return txt.lower()
    code = folder_code.upper()
    mapping = {
        'HAMB': 'hamburger',
        'ICE': 'icecream',
        'CACT': 'cactus',
        'TUL': 'tulip',
        'BULL': 'bull',
        'SUND': 'sundae',
    }
    return mapping.get(code, folder_code.lower())


def collect_efficiency_per_run(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for cfg in sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'logs']):
        code, seed = parse_prompt_and_seed(cfg.name)
        if code is None:
            continue
        full_prompt = _read_prompt_from_config(cfg)
        prompt = _keyword_from_prompt(full_prompt, code)
        ecsv = cfg / "metrics" / "efficiency.csv"
        if not ecsv.exists():
            continue
        df = pd.read_csv(ecsv)
        cols = [
            "step_wall_ms", "step_gpu_ms", "render_ms", "sd_guidance_ms",
            "backprop_ms", "densify_ms", "px_per_s",
            "memory_allocated_mb", "max_memory_allocated_mb",
        ]
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        rec = {"prompt": prompt, "seed": seed}
        rec.update({c: float(df[c].mean()) for c in present})
        # derived
        comp_cols = [c for c in ["render_ms", "sd_guidance_ms", "backprop_ms", "densify_ms"] if c in df.columns]
        if comp_cols:
            rec["total_components_ms"] = float(df[comp_cols].mean().sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def tidy_aggregate_by_prompt(per_run: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in per_run.columns:
            continue
        g = per_run.groupby("prompt")[metric]
        for prompt, series in g:
            rows.append({
                "prompt": prompt,
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.size > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def pretty(metric: str) -> str:
    mapping = {
        # Quantitative metrics
        "inter_particle_diversity_mean": "Diversity (↑)",
        "fidelity_mean": "Fidelity (↑)",
        "cross_view_consistency_mean": "Consistency (↑)",
        # Efficiency
        "step_wall_ms": "Step time (ms) (↓)",
        "step_gpu_ms": "GPU step time (ms) (↓)",
        "px_per_s": "Throughput (px/s) (↑)",
        "memory_allocated_mb": "Memory (MB) (↓)",
        "max_memory_allocated_mb": "Peak memory (MB) (↓)",
    }
    return mapping.get(metric, metric)


def plot_error_bars(agg_base: pd.DataFrame, agg_ours: pd.DataFrame, metric: str, out_png: Path):
    prompts = sorted(set(agg_base[agg_base.metric == metric].prompt).union(set(agg_ours[agg_ours.metric == metric].prompt)))
    x = list(range(len(prompts)))
    width = 0.38
    base_means, base_stds, ours_means, ours_stds = [], [], [], []
    for p in prompts:
        b = agg_base[(agg_base.prompt == p) & (agg_base.metric == metric)]
        o = agg_ours[(agg_ours.prompt == p) & (agg_ours.metric == metric)]
        base_means.append(float(b["mean"].values[0]) if not b.empty else float('nan'))
        base_stds.append(float(b["std"].values[0]) if not b.empty else 0.0)
        ours_means.append(float(o["mean"].values[0]) if not o.empty else float('nan'))
        ours_stds.append(float(o["std"].values[0]) if not o.empty else 0.0)

    plt.figure(figsize=(max(7.5, len(prompts) * 1.2), 4.2))
    plt.bar([xi - width/2 for xi in x], base_means, width=width, yerr=base_stds, capsize=3,
            label="Baseline", color="#4C78A8", edgecolor="#2f2f2f", linewidth=0.6)
    plt.bar([xi + width/2 for xi in x], ours_means, width=width, yerr=ours_stds, capsize=3,
            label="Ours", color="#F58518", edgecolor="#2f2f2f", linewidth=0.6)
    plt.xticks(x, [p.title() if '_' not in p else p.replace('_',' ') for p in prompts], rotation=45, ha='right')
    plt.ylabel(pretty(metric))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def _full_name_from_csv_column(col: str) -> str:
    # Convert snake_case to Title Case label from CSV column name
    # Keep simple capitalization; handle common abbreviations if appear
    parts = col.replace("_", " ").split()
    # Drop trailing 'mean' token if present
    if parts and parts[-1].lower() == "mean":
        parts = parts[:-1]
    titled = [w.upper() if w in {"lpips", "clip", "sd"} else w.title() for w in parts]
    return " ".join(titled) + " (↑)"


def collect_step1000_quant_metrics(exp_dir: Path) -> pd.DataFrame:
    """
    For each run under exp_dir, load metrics/quantitative_metrics.csv and
    take the row at step==1000 (or the last <=1000). Keep key metrics.
    Returns a DataFrame with columns: prompt, seed, and metric columns.
    """
    rows = []
    for cfg in sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'logs']):
        code, seed = parse_prompt_and_seed(cfg.name)
        if code is None:
            continue
        full_prompt = _read_prompt_from_config(cfg)
        prompt = _keyword_from_prompt(full_prompt, code)
        qcsv = cfg / "metrics" / "quantitative_metrics.csv"
        if not qcsv.exists():
            continue
        dfq = pd.read_csv(qcsv)
        if "step" not in dfq.columns:
            continue
        row = dfq[dfq["step"] == 1000]
        if row.empty:
            row = dfq[dfq["step"] <= 1000].sort_values("step").tail(1)
        if row.empty:
            continue
        r = row.iloc[0].to_dict()
        rec = {"prompt": prompt, "seed": seed}
        for k in [
            "inter_particle_diversity_mean",
            "cross_view_consistency_mean",
            "fidelity_mean",
        ]:
            if k in r:
                rec[k] = float(r[k])
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_per_prompt_simple(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.groupby("prompt")[metrics].mean().reset_index()


def plot_bars(agg_baseline: pd.DataFrame, agg_ours: pd.DataFrame, metric: str, out_png: Path):
    prompts = sorted(set(agg_baseline.get("prompt", pd.Series()).tolist()) | set(agg_ours.get("prompt", pd.Series()).tolist()))
    base_idx = agg_baseline.set_index("prompt") if not agg_baseline.empty else pd.DataFrame()
    ours_idx = agg_ours.set_index("prompt") if not agg_ours.empty else pd.DataFrame()
    x = list(range(len(prompts)))
    width = 0.38
    baseline_color = "#4C78A8"
    ours_color = "#F58518"
    edge_color = "#2f2f2f"
    plt.figure(figsize=(max(7, len(prompts) * 1.0), 4.2))
    base_vals = [float(base_idx.get(metric, pd.Series()).get(p, float('nan'))) for p in prompts]
    ours_vals = [float(ours_idx.get(metric, pd.Series()).get(p, float('nan'))) for p in prompts]
    plt.bar([xi - width/2 for xi in x], base_vals, width=width, label="Baseline",
            color=baseline_color, edgecolor=edge_color, linewidth=0.6)
    plt.bar([xi + width/2 for xi in x], ours_vals, width=width, label="Ours",
            color=ours_color, edgecolor=edge_color, linewidth=0.6)
    plt.xticks(x, [p.title() if '_' not in p else p.replace('_',' ') for p in prompts], rotation=45, ha='right')
    plt.ylabel(pretty(metric))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_dir", type=str, default="/Users/sj/3D-Generation/exp/exp6_ours_best")
    parser.add_argument("--baseline_dir", type=str, default="/Users/sj/3D-Generation/exp/exp6_baseline")
    parser.add_argument("--out_dir", type=str, default="/Users/sj/3D-Generation/results/comparison_ours_baseline")
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

    ours_runs = collect_efficiency_per_run(Path(args.ours_dir))
    base_runs = collect_efficiency_per_run(Path(args.baseline_dir))

    ours_runs.to_csv(Path(args.out_dir) / "ours_efficiency_per_run.csv", index=False)
    base_runs.to_csv(Path(args.out_dir) / "baseline_efficiency_per_run.csv", index=False)

    metrics = [
        "step_wall_ms",
        "step_gpu_ms",
        "px_per_s",
        "memory_allocated_mb",
        "max_memory_allocated_mb",
    ]

    ours_prompt = tidy_aggregate_by_prompt(ours_runs, metrics)
    base_prompt = tidy_aggregate_by_prompt(base_runs, metrics)
    ours_prompt.to_csv(Path(args.out_dir) / "ours_efficiency_per_prompt.csv", index=False)
    base_prompt.to_csv(Path(args.out_dir) / "baseline_efficiency_per_prompt.csv", index=False)

    # Overall across prompts
    def overall(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for metric in metrics:
            sub = df[df.metric == metric]
            if sub.empty:
                continue
            rows.append({
                "metric": metric,
                "mean": float(sub["mean"].mean()),
                "std": float(sub["std"].mean()),
            })
        return pd.DataFrame(rows)

    ours_overall = overall(ours_prompt)
    base_overall = overall(base_prompt)
    overall_tbl = pd.merge(
        base_overall.add_prefix("baseline_"),
        ours_overall.add_prefix("ours_"),
        left_on="baseline_metric", right_on="ours_metric",
        how="outer",
    )
    overall_tbl = overall_tbl.rename(columns={"baseline_metric": "metric"}).drop(columns=["ours_metric"])
    overall_tbl.to_csv(Path(args.out_dir) / "overall_efficiency_stats.csv", index=False)

    # Plots: per-prompt time and memory with error bars
    plot_error_bars(base_prompt, ours_prompt, "step_wall_ms", Path(args.out_dir) / "efficiency_time_per_prompt.png")
    plot_error_bars(base_prompt, ours_prompt, "max_memory_allocated_mb", Path(args.out_dir) / "efficiency_memory_per_prompt.png")

    # Overall bar plots with error bars
    for metric, fname in [
        ("step_wall_ms", "efficiency_time_overall.png"),
        ("max_memory_allocated_mb", "efficiency_memory_overall.png"),
    ]:
        b = overall_tbl[overall_tbl.metric == metric]
        if b.empty:
            continue
        x = ["Baseline", "Ours"]
        means = [float(b.baseline_mean.values[0]), float(b.ours_mean.values[0])]
        errs = [float(b.baseline_std.values[0]), float(b.ours_std.values[0])]
        plt.figure(figsize=(3.2, 4))
        plt.bar(x, means, yerr=errs, capsize=3, color=["#4C78A8", "#F58518"], edgecolor="#2f2f2f", linewidth=0.6)
        plt.ylabel(pretty(metric))
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / fname, dpi=300, bbox_inches='tight')
        plt.close()

    # ===================== Quantitative metrics (fidelity/diversity/consistency) =====================
    quant_cols = [
        "inter_particle_diversity_mean",
        "cross_view_consistency_mean",
        "fidelity_mean",
    ]

    ours_q_runs = collect_step1000_quant_metrics(Path(args.ours_dir))
    base_q_runs = collect_step1000_quant_metrics(Path(args.baseline_dir))

    # Save per-run (no per-prompt outputs requested)
    ours_q_runs.to_csv(Path(args.out_dir) / "ours_step1000_metrics_per_run.csv", index=False)
    base_q_runs.to_csv(Path(args.out_dir) / "baseline_step1000_metrics_per_run.csv", index=False)

    # Aggregate over prompts (mean of prompt means) and compute std across prompts
    ours_q_prompt = aggregate_per_prompt_simple(ours_q_runs, quant_cols)
    base_q_prompt = aggregate_per_prompt_simple(base_q_runs, quant_cols)

    # Overall aggregates across prompts
    if not ours_q_prompt.empty or not base_q_prompt.empty:
        overall_quant = pd.DataFrame({
            "metric": quant_cols,
            "baseline_mean": [base_q_prompt[m].mean() if (m in base_q_prompt.columns and not base_q_prompt.empty) else float('nan') for m in quant_cols],
            "baseline_std": [base_q_prompt[m].std(ddof=1) if (m in base_q_prompt.columns and not base_q_prompt.empty) else float('nan') for m in quant_cols],
            "ours_mean": [ours_q_prompt[m].mean() if (m in ours_q_prompt.columns and not ours_q_prompt.empty) else float('nan') for m in quant_cols],
            "ours_std": [ours_q_prompt[m].std(ddof=1) if (m in ours_q_prompt.columns and not ours_q_prompt.empty) else float('nan') for m in quant_cols],
        })
        overall_quant.to_csv(Path(args.out_dir) / "overall_step1000_metrics.csv", index=False)

        # Individual overall plots with error bars
        for metric, fname in [
            ("inter_particle_diversity_mean", "diversity_overall.png"),
            ("fidelity_mean", "fidelity_overall.png"),
            ("cross_view_consistency_mean", "consistency_overall.png"),
        ]:
            sub = overall_quant[overall_quant.metric == metric]
            if sub.empty:
                continue
            vals = [float(sub.baseline_mean.values[0]), float(sub.ours_mean.values[0])]
            errs = [float(sub.baseline_std.values[0]), float(sub.ours_std.values[0])]
            plt.figure(figsize=(3.2, 4))
            plt.bar(["Baseline", "Ours"], vals, yerr=errs, capsize=3, color=["#4C78A8", "#F58518"], edgecolor="#2f2f2f", linewidth=0.6)
            plt.ylabel(_full_name_from_csv_column(metric))
            plt.tight_layout()
            plt.savefig(Path(args.out_dir) / fname, dpi=300, bbox_inches='tight')
            plt.close()

        # Combined overall figure (3 subplots) with error bars
        try:
            fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8))
            for ax, metric in zip(axes, quant_cols):
                sub = overall_quant[overall_quant.metric == metric]
                if sub.empty:
                    continue
                vals = [float(sub.baseline_mean.values[0]), float(sub.ours_mean.values[0])]
                errs = [float(sub.baseline_std.values[0]), float(sub.ours_std.values[0])]
                ax.bar(["Baseline", "Ours"], vals, yerr=errs, capsize=3, color=["#4C78A8", "#F58518"], edgecolor="#2f2f2f", linewidth=0.6)
                ax.set_ylabel(_full_name_from_csv_column(metric))
            fig.tight_layout()
            fig.savefig(Path(args.out_dir) / "combined_overall.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    print("Saved figures and CSV to", args.out_dir)


if __name__ == "__main__":
    main()


