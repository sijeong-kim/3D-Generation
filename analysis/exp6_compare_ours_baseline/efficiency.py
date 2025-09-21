import argparse
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


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


def _load_prompt_reverse_map(global_yaml_path: Path) -> Dict[str, str]:
    try:
        if yaml is not None:
            with open(global_yaml_path, 'r') as f:
                data = yaml.safe_load(f)  # type: ignore
            if isinstance(data, dict) and 'prompt' in data and isinstance(data['prompt'], dict):
                return {v: k for k, v in data['prompt'].items() if isinstance(v, str)}
    except Exception:
        pass
    reverse_map: Dict[str, str] = {}
    try:
        with open(global_yaml_path, 'r') as f:
            lines = f.readlines()
        in_block = False
        for raw in lines:
            if not in_block:
                if raw.strip() == 'prompt:':
                    in_block = True
                continue
            if raw and not raw.startswith(' '):
                break
            ls = raw.strip()
            if not ls or ls.startswith('#') or ':' not in ls:
                continue
            key, rest = ls.split(':', 1)
            val = rest.strip()
            if '#' in val:
                val = val.split('#', 1)[0].strip()
            if val.startswith(('"', "'")) and val.endswith(('"', "'")) and len(val) >= 2:
                val = val[1:-1]
            key = key.strip().lstrip('#').strip()
            if key and val:
                reverse_map[val] = key
    except Exception:
        return {}
    return reverse_map


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


def collect_efficiency_stats(exp_dir: Path, prompt_reverse_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for cfg in sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'logs']):
        code, seed = parse_prompt_and_seed(cfg.name)
        if code is None:
            continue
        full_prompt = _read_prompt_from_config(cfg)
        keyword = prompt_reverse_map.get(full_prompt) if full_prompt else None
        if not keyword:
            keyword = _keyword_from_prompt(full_prompt, code)

        ecsv = cfg / "metrics" / "efficiency.csv"
        if not ecsv.exists():
            continue
        df = pd.read_csv(ecsv)
        # Known efficiency columns
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
        stats = {f"{c}_mean": float(df[c].mean()) for c in present}
        stats.update({f"{c}_std": float(df[c].std(ddof=1)) for c in present})
        stats.update({"prompt": keyword, "seed": seed})
        rows.append(stats)
    return pd.DataFrame(rows)


def tidy_aggregate_by_prompt(per_run: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in per_run.columns:
            continue
        g = per_run.groupby("prompt")[mean_col]
        for prompt, series in g:
            rows.append({
                "prompt": prompt,
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.size > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def _pretty_metric_name(metric: str) -> str:
    mapping = {
        "step_wall_ms": "Step time (ms) (↓)",
        "step_gpu_ms": "GPU step time (ms) (↓)",
        "px_per_s": "Throughput (px/s) (↑)",
        "memory_allocated_mb": "Memory (MB) (↓)",
        "max_memory_allocated_mb": "Peak memory (MB) (↓)",
    }
    return mapping.get(metric, metric)


def plot_error_bars(agg_base: pd.DataFrame, agg_ours: pd.DataFrame, metric: str, out_png: Path):
    prompts = sorted(set(agg_base[agg_base.metric == metric].prompt).union(set(agg_ours[agg_ours.metric == metric].prompt)))
    base_means = []
    base_stds = []
    ours_means = []
    ours_stds = []
    for p in prompts:
        b = agg_base[(agg_base.prompt == p) & (agg_base.metric == metric)]
        o = agg_ours[(agg_ours.prompt == p) & (agg_ours.metric == metric)]
        base_means.append(float(b["mean"].values[0]) if not b.empty else float('nan'))
        base_stds.append(float(b["std"].values[0]) if not b.empty else 0.0)
        ours_means.append(float(o["mean"].values[0]) if not o.empty else float('nan'))
        ours_stds.append(float(o["std"].values[0]) if not o.empty else 0.0)

    x = list(range(len(prompts)))
    width = 0.38
    plt.figure(figsize=(max(7.5, len(prompts)*1.2), 4.2))
    baseline_color = "#4C78A8"
    ours_color = "#F58518"
    edge_color = "#2f2f2f"
    plt.bar([xi - width/2 for xi in x], base_means, width=width, yerr=base_stds, capsize=3,
            label="Baseline", color=baseline_color, edgecolor=edge_color, linewidth=0.6)
    plt.bar([xi + width/2 for xi in x], ours_means, width=width, yerr=ours_stds, capsize=3,
            label="Ours", color=ours_color, edgecolor=edge_color, linewidth=0.6)
    plt.xticks(x, [p.title() if '_' not in p else p.replace('_',' ') for p in prompts], rotation=45, ha='right')
    plt.ylabel(_pretty_metric_name(metric))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_dir", type=str, default="/Users/sj/3D-Generation/exp/exp6_ours_best")
    parser.add_argument("--baseline_dir", type=str, default="/Users/sj/3D-Generation/exp/exp6_baseline")
    parser.add_argument("--csv_out_dir", type=str, default="/Users/sj/3D-Generation/results/efficiency/exp6_compare_ours_baseline")
    parser.add_argument("--fig_out_dir", type=str, default="/Users/sj/3D-Generation/results/efficiency/exp6_compare_ours_baseline")
    args = parser.parse_args()

    ensure_dirs([args.csv_out_dir, args.fig_out_dir])

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass

    global_cfg_path = Path("/Users/sj/3D-Generation/configs/text_ours_exp.yaml")
    prompt_reverse_map = _load_prompt_reverse_map(global_cfg_path)

    ours_runs = collect_efficiency_stats(Path(args.ours_dir), prompt_reverse_map)
    base_runs = collect_efficiency_stats(Path(args.baseline_dir), prompt_reverse_map)

    ours_runs.to_csv(Path(args.csv_out_dir) / "ours_efficiency_per_run_stats.csv", index=False)
    base_runs.to_csv(Path(args.csv_out_dir) / "baseline_efficiency_per_run_stats.csv", index=False)

    metrics = [
        "step_wall_ms",
        "step_gpu_ms",
        "px_per_s",
        "memory_allocated_mb",
        "max_memory_allocated_mb",
    ]

    ours_prompt = tidy_aggregate_by_prompt(ours_runs, metrics)
    base_prompt = tidy_aggregate_by_prompt(base_runs, metrics)
    ours_prompt.to_csv(Path(args.csv_out_dir) / "ours_efficiency_per_prompt_stats.csv", index=False)
    base_prompt.to_csv(Path(args.csv_out_dir) / "baseline_efficiency_per_prompt_stats.csv", index=False)

    # Overall aggregate across prompts (means of prompt means, std across prompt means)
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
    overall_tbl.to_csv(Path(args.csv_out_dir) / "overall_efficiency_stats.csv", index=False)

    # Plots: time and memory per prompt with error bars
    plot_error_bars(base_prompt, ours_prompt, "step_wall_ms", Path(args.fig_out_dir) / "efficiency_time_per_prompt.png")
    plot_error_bars(base_prompt, ours_prompt, "max_memory_allocated_mb", Path(args.fig_out_dir) / "efficiency_memory_per_prompt.png")

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
        plt.ylabel(_pretty_metric_name(metric))
        plt.tight_layout()
        plt.savefig(Path(args.fig_out_dir) / fname, dpi=300, bbox_inches='tight')
        plt.close()

    print("Done. CSVs saved to", args.csv_out_dir)
    print("Figures saved to", args.fig_out_dir)


if __name__ == "__main__":
    main()


