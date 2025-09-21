import argparse
from pathlib import Path
import re
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

"""_summary_
MPLBACKEND=Agg python3 /Users/sj/3D-Generation/analysis/aggregate_efficiency_all_experiments.py

"""

def ensure_dirs(paths: List[str]):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def parse_prompt_and_seed(name: str):
    m = re.match(r"^(?P<prompt>[A-Z]+)__S(?P<seed>\d+)$", name)
    if not m:
        return None, None
    return m.group("prompt"), int(m.group("seed"))


def collect_efficiency_per_run(exp_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for run in sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'logs']):
        code, seed = parse_prompt_and_seed(run.name)
        if code is None:
            continue
        ecsv = run / "metrics" / "efficiency.csv"
        if not ecsv.exists():
            continue
        df = pd.read_csv(ecsv)
        cols = [
            "step_wall_ms",
            "step_gpu_ms",
            "render_ms",
            "sd_guidance_ms",
            "backprop_ms",
            "densify_ms",
            "px_per_s",
            "memory_allocated_mb",
            "max_memory_allocated_mb",
        ]
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        rec: Dict[str, float] = {f"{c}_mean": float(df[c].mean()) for c in present}
        rec.update({f"{c}_std": float(df[c].std(ddof=1)) for c in present})
        rec.update({"run": run.name, "prompt_code": code, "seed": seed})
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, default="/Users/sj/3D-Generation/exp")
    parser.add_argument("--out_dir", type=str, default="/Users/sj/3D-Generation/results/efficiency")
    args = parser.parse_args()

    ensure_dirs([args.out_dir])

    exp_root = Path(args.exp_root)
    experiments = [d for d in exp_root.iterdir() if d.is_dir() and not d.name.startswith('.')]

    all_rows = []
    per_experiment: Dict[str, pd.DataFrame] = {}

    for exp in sorted(experiments):
        df = collect_efficiency_per_run(exp)
        if df.empty:
            continue
        per_experiment[exp.name] = df
        df.to_csv(Path(args.out_dir) / f"{exp.name}__efficiency_per_run.csv", index=False)
        all_rows.append(df.assign(experiment=exp.name))

    if not all_rows:
        print("No efficiency data found.")
        return

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(Path(args.out_dir) / "all_experiments__efficiency_per_run.csv", index=False)

    # Aggregate per experiment across runs
    agg_rows: List[Dict] = []
    metric_roots = sorted({c[:-5] for c in all_df.columns if c.endswith('_mean')})
    for exp_name, g in all_df.groupby("experiment"):
        for m in metric_roots:
            col = f"{m}_mean"
            if col not in g.columns:
                continue
            agg_rows.append({
                "experiment": exp_name,
                "metric": m,
                "mean": float(g[col].mean()),
                "std": float(g[col].std(ddof=1)) if len(g[col]) > 1 else 0.0,
            })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(Path(args.out_dir) / "all_experiments__efficiency_per_experiment.csv", index=False)

    # Quick overview plots for step time and peak memory per experiment
    for metric, fname in [("step_wall_ms", "all_experiments_time.png"), ("max_memory_allocated_mb", "all_experiments_memory.png")]:
        sub = agg_df[agg_df.metric == metric]
        if sub.empty:
            continue
        order = list(sub.sort_values("mean").experiment)
        means = [float(sub[sub.experiment == e]["mean"].values[0]) for e in order]
        errs = [float(sub[sub.experiment == e]["std"].values[0]) for e in order]
        plt.figure(figsize=(max(8, len(order)*0.8), 4))
        plt.bar(order, means, yerr=errs, capsize=3)
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Path(args.out_dir) / fname, dpi=300, bbox_inches='tight')
        plt.close()

    print("Done. Wrote aggregated efficiency to", args.out_dir)


if __name__ == "__main__":
    main()
