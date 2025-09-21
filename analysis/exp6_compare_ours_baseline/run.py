import argparse
import os
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

    """_summary_
    MPLBACKEND=Agg python3 /Users/sj/3D-Generation/analysis/exp6_compare_ours_baseline/run.py
    """


def parse_prompt_and_seed(dir_name: str):
    # Expect names like HAMB__S42
    m = re.match(r"^(?P<prompt>[A-Z]+)__S(?P<seed>\d+)$", dir_name)
    if not m:
        return None, None
    return m.group("prompt"), int(m.group("seed"))


def collect_step1000_metrics(exp_dir: Path, prompt_reverse_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for cfg in sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'logs']):
        prompt, seed = parse_prompt_and_seed(cfg.name)
        if prompt is None:
            continue
        full_prompt = _read_prompt_from_config(cfg)
        keyword_label = prompt_reverse_map.get(full_prompt) if full_prompt else None
        if not keyword_label:
            keyword_label = _keyword_from_prompt(full_prompt, prompt)
        # Sanitize any accidental leading comment markers
        if isinstance(keyword_label, str):
            keyword_label = re.sub(r"^[#\s]+", "", keyword_label).strip()
        qcsv = cfg / "metrics" / "quantitative_metrics.csv"
        if not qcsv.exists():
            continue
        df = pd.read_csv(qcsv)
        if "step" not in df.columns:
            continue
        row = df[df["step"] == 1000]
        if row.empty:
            # try nearest <= 1000
            row = df[df["step"] <= 1000].sort_values("step").tail(1)
            if row.empty:
                continue
        r = row.iloc[0].to_dict()
        r.update({"prompt": keyword_label, "seed": seed})
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate_per_prompt(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.groupby("prompt")[metrics].mean().reset_index()
    return agg


def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


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
                    # Strip inline comments
                    if '#' in val:
                        val = val.split('#', 1)[0].strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    return val
    except Exception:
        return ""
    return ""


def _load_prompt_reverse_map(global_yaml_path: Path) -> Dict[str, str]:
    # Build mapping: full prompt text -> keyword id (e.g., "a photo of a hamburger" -> "hamburger")
    try:
        if yaml is not None:
            with open(global_yaml_path, 'r') as f:
                data = yaml.safe_load(f)  # type: ignore
            if isinstance(data, dict) and 'prompt' in data and isinstance(data['prompt'], dict):
                return {v: k for k, v in data['prompt'].items() if isinstance(v, str)}
    except Exception:
        pass
    # Fallback simple parse
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
            # Strip inline comments
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


def _pretty_metric_name(metric: str) -> str:
    mapping = {
        "inter_particle_diversity_mean": "Diversity (↑)",
        "fidelity_mean": "Fidelity (↑)",
        "cross_view_consistency_mean": "Consistency (↑)",
    }
    return mapping.get(metric, metric)


def plot_bars(agg_baseline: pd.DataFrame, agg_ours: pd.DataFrame, metric: str, out_png: Path):
    prompts = sorted(set(agg_baseline["prompt"]).union(set(agg_ours["prompt"])) )
    base_vals = [float(agg_baseline.set_index("prompt").get(metric, pd.Series()).get(p, float('nan'))) for p in prompts]
    ours_vals = [float(agg_ours.set_index("prompt").get(metric, pd.Series()).get(p, float('nan'))) for p in prompts]

    x = range(len(prompts))
    width = 0.38
    plt.figure(figsize=(max(7, len(prompts)*1.0), 4.2))
    baseline_color = "#4C78A8"  # blue
    ours_color = "#F58518"      # orange
    edge_color = "#2f2f2f"
    plt.bar([xi - width/2 for xi in x], base_vals, width=width, label="Baseline",
            color=baseline_color, edgecolor=edge_color, linewidth=0.6)
    plt.bar([xi + width/2 for xi in x], ours_vals, width=width, label="Ours",
            color=ours_color, edgecolor=edge_color, linewidth=0.6)
    plt.xticks(list(x), [p.title() if '_' not in p else p.replace('_',' ') for p in prompts], rotation=45, ha='right')
    plt.ylabel(_pretty_metric_name(metric))
    # Place legend above to avoid overlap
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_dir", type=str, default="/Users/sj/3D-Generation/exp/exp6_ours_best")
    parser.add_argument("--baseline_dir", type=str, default="/Users/sj/3D-Generation/exp/exp6_baseline")
    parser.add_argument("--csv_out_dir", type=str, default="/Users/sj/3D-Generation/results/csv/exp6_compare_ours_baseline")
    parser.add_argument("--fig_out_dir", type=str, default="/Users/sj/3D-Generation/results/exp6_compare_ours_baseline")
    args = parser.parse_args()

    ensure_dirs([args.csv_out_dir, args.fig_out_dir])

    # Global style
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

    # Load reverse mapping from full prompt text to short keyword labels from config
    global_cfg_path = Path("/Users/sj/3D-Generation/configs/text_ours_exp.yaml")
    prompt_reverse_map = _load_prompt_reverse_map(global_cfg_path)

    ours_df = collect_step1000_metrics(Path(args.ours_dir), prompt_reverse_map)
    base_df = collect_step1000_metrics(Path(args.baseline_dir), prompt_reverse_map)

    # Save raw per-run step1000 CSVs
    ours_df.to_csv(Path(args.csv_out_dir) / "ours_step1000_per_run.csv", index=False)
    base_df.to_csv(Path(args.csv_out_dir) / "baseline_step1000_per_run.csv", index=False)

    metrics_cols = [
        "inter_particle_diversity_mean",
        "cross_view_consistency_mean",
        "fidelity_mean",
    ]

    ours_per_prompt = aggregate_per_prompt(ours_df, metrics_cols)
    base_per_prompt = aggregate_per_prompt(base_df, metrics_cols)
    ours_per_prompt.to_csv(Path(args.csv_out_dir) / "ours_step1000_per_prompt.csv", index=False)
    base_per_prompt.to_csv(Path(args.csv_out_dir) / "baseline_step1000_per_prompt.csv", index=False)

    # Overall aggregate over prompts (mean of per-prompt means)
    overall = pd.DataFrame({
        "metric": metrics_cols,
        "baseline_mean": [base_per_prompt[m].mean() if not base_per_prompt.empty else float('nan') for m in metrics_cols],
        "ours_mean": [ours_per_prompt[m].mean() if not ours_per_prompt.empty else float('nan') for m in metrics_cols],
    })
    overall.to_csv(Path(args.csv_out_dir) / "overall_step1000_aggregates.csv", index=False)

    # Plots: diversity, fidelity, consistency (per-prompt)
    plot_bars(base_per_prompt, ours_per_prompt, "inter_particle_diversity_mean", Path(args.fig_out_dir) / "diversity_per_prompt.png")
    plot_bars(base_per_prompt, ours_per_prompt, "fidelity_mean", Path(args.fig_out_dir) / "fidelity_per_prompt.png")
    plot_bars(base_per_prompt, ours_per_prompt, "cross_view_consistency_mean", Path(args.fig_out_dir) / "consistency_per_prompt.png")

    # Also plot overall bar for diversity, fidelity, consistency
    for metric, fname in [
        ("inter_particle_diversity_mean", "diversity_overall.png"),
        ("fidelity_mean", "fidelity_overall.png"),
        ("cross_view_consistency_mean", "consistency_overall.png"),
    ]:
        vals = [overall.loc[overall["metric"] == metric, "baseline_mean"].values[0],
                overall.loc[overall["metric"] == metric, "ours_mean"].values[0]]
        plt.figure(figsize=(3.2, 4))
        baseline_color = "#4C78A8"
        ours_color = "#F58518"
        edge_color = "#2f2f2f"
        plt.bar(["Baseline", "Ours"], vals, color=[baseline_color, ours_color], edgecolor=edge_color, linewidth=0.6)
        plt.ylabel(_pretty_metric_name(metric))
        plt.tight_layout()
        plt.savefig(Path(args.fig_out_dir) / fname, dpi=300, bbox_inches='tight')
        plt.close()

    # Combined figures (per-prompt and overall)
    try:
        metrics_list = [
            ("inter_particle_diversity_mean", "diversity_per_prompt"),
            ("fidelity_mean", "fidelity_per_prompt"),
            ("cross_view_consistency_mean", "consistency_per_prompt"),
        ]

        # Per-prompt combined: three subplots
        prompts = sorted(set(base_per_prompt.get("prompt", pd.Series()).tolist()) | set(ours_per_prompt.get("prompt", pd.Series()).tolist()))
        fig, axes = plt.subplots(1, 3, figsize=(max(10, len(prompts)*1.7), 3.8), sharex=False)
        width = 0.38
        x = list(range(len(prompts)))
        base_idx = base_per_prompt.set_index("prompt") if not base_per_prompt.empty else pd.DataFrame()
        ours_idx = ours_per_prompt.set_index("prompt") if not ours_per_prompt.empty else pd.DataFrame()
        baseline_color = "#4C78A8"
        ours_color = "#F58518"
        edge_color = "#2f2f2f"
        for ax, (metric, _) in zip(axes, metrics_list):
            base_vals = [float(base_idx.get(metric, pd.Series()).get(p, float('nan'))) for p in prompts]
            ours_vals = [float(ours_idx.get(metric, pd.Series()).get(p, float('nan'))) for p in prompts]
            ax.bar([xi - width/2 for xi in x], base_vals, width=width, label="Baseline",
                   color=baseline_color, edgecolor=edge_color, linewidth=0.6)
            ax.bar([xi + width/2 for xi in x], ours_vals, width=width, label="Ours",
                   color=ours_color, edgecolor=edge_color, linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([p.title() if '_' not in p else p.replace('_',' ') for p in prompts], rotation=45, ha='right')
            ax.set_ylabel(_pretty_metric_name(metric))
        # Shared legend above the subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(Path(args.fig_out_dir) / "combined_per_prompt.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Overall combined: three subplots
        fig2, axes2 = plt.subplots(1, 3, figsize=(9.6, 3.8))
        for ax, (metric, _) in zip(axes2, metrics_list):
            vals = [overall.loc[overall["metric"] == metric, "baseline_mean"].values[0],
                    overall.loc[overall["metric"] == metric, "ours_mean"].values[0]]
            ax.bar(["Baseline", "Ours"], vals, color=[baseline_color, ours_color], edgecolor=edge_color, linewidth=0.6)
            ax.set_ylabel(_pretty_metric_name(metric))
        fig2.tight_layout()
        fig2.savefig(Path(args.fig_out_dir) / "combined_overall.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
    except Exception:
        # Avoid failing the whole script if combined plots error out
        pass

    print("Done. CSVs saved to", args.csv_out_dir)
    print("Figures saved to", args.fig_out_dir)


if __name__ == "__main__":
    main()


