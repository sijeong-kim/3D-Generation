#!/usr/bin/env python3
"""
Compare lambda_repulsion results against baseline for different combinations
of kernel_type, prompt, and repulsion_type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

def parse_config_name(config_name):
    """Parse config name to extract parameters."""
    pattern = r'kernel_type=(\w+)_lambda_repulsion=([\de+-]+)_prompt=(\w+)_repulsion_type=(\w+)_seed=(\d+)'
    match = re.match(pattern, config_name)
    if match:
        return {
            'kernel_type': match.group(1),
            'lambda_repulsion': match.group(2),
            'prompt': match.group(3),
            'repulsion_type': match.group(4),
            'seed': match.group(5)
        }
    return None

def get_lambda_repulsion_order(lambda_repulsion_str):
    """Get numeric order for lambda_repulsion values."""
    # Convert scientific notation to float for sorting
    return float(lambda_repulsion_str.replace('e', 'E'))

def create_comparison_plot(baseline_data, experiment_data, config_params, output_dir):
    """Create comparison plot for a specific combination."""
    kernel_type = config_params['kernel_type']
    prompt = config_params['prompt']
    repulsion_type = config_params['repulsion_type']
    
    # Define metrics to plot
    metrics = [
        ("fidelity_mean", "CLIP Fidelity (mean)"),
        ("inter_particle_diversity_mean", "Inter-Particle Diversity (mean)"),
        ("cross_view_consistency_mean", "Cross-View Consistency (mean)"),
        ("lpips_inter_mean", "LPIPS Inter (mean)"),
        ("lpips_consistency_mean", "LPIPS Consistency (mean)"),
    ]
    
    # Filter available metrics
    available_metrics = [(col, label) for col, label in metrics if col in baseline_data.columns]
    
    if len(available_metrics) == 0:
        print(f"No metrics available for {kernel_type}_{prompt}_{repulsion_type}")
        return None
    
    # Get lambda_repulsion values and sort them
    lambda_values = list(experiment_data.keys())
    lambda_values.sort(key=get_lambda_repulsion_order)
    
    # Create subplot layout
    n_metrics = len(available_metrics)
    n_lambdas = len(lambda_values)
    
    fig, axes = plt.subplots(n_metrics, n_lambdas, figsize=(4*n_lambdas, 4*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_lambdas == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each metric
    for metric_idx, (metric_col, metric_label) in enumerate(available_metrics):
        for lambda_idx, lambda_val in enumerate(lambda_values):
            ax = axes[metric_idx, lambda_idx]
            
            # Plot baseline
            if metric_col in baseline_data.columns:
                baseline_plot = baseline_data[["step", metric_col]].dropna().sort_values("step")
                if len(baseline_plot) > 0:
                    ax.plot(baseline_plot["step"], baseline_plot[metric_col], 
                           '--', color='gray', linewidth=2, alpha=0.7, label='Baseline')
            
            # Plot experiment data
            exp_data = experiment_data[lambda_val]
            if metric_col in exp_data.columns:
                exp_plot = exp_data[["step", metric_col]].dropna().sort_values("step")
                if len(exp_plot) > 0:
                    ax.plot(exp_plot["step"], exp_plot[metric_col], 
                           '-', linewidth=2, label=f'λ={lambda_val}')
            
            # Set labels and title
            if lambda_idx == 0:  # First column
                ax.set_ylabel(metric_label, fontsize=10)
            if metric_idx == 0:  # First row
                ax.set_title(f'λ={lambda_val}', fontsize=12, fontweight='bold')
            if metric_idx == n_metrics - 1:  # Last row
                ax.set_xlabel('Step', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
            # Add legend only to first subplot
            if metric_idx == 0 and lambda_idx == 0:
                ax.legend(fontsize=8)
    
    # Add overall title
    title = f"{kernel_type.upper()} Kernel - {prompt.title()} - {repulsion_type.upper()}"
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save plot
    filename = f"comparison_{kernel_type}_{prompt}_{repulsion_type}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    return output_path

# ---------- multi-prompt comparison ----------
def create_comparison_plot_multi_prompt(
    baseline_data,
    experiment_data_by_prompt: dict[str, dict[str, pd.DataFrame]],
    config_params,
    output_dir: Path
):
    """
    Create grid (metrics × λ) like original version,
    but overlay ALL prompts in different colors inside each subplot.
    """
    kernel_type = config_params['kernel_type']
    repulsion_type = config_params['repulsion_type']

    metrics = [
        ("fidelity_mean", "CLIP Fidelity (mean)"),
        ("inter_particle_diversity_mean", "Inter-Particle Diversity (mean)"),
        ("cross_view_consistency_mean", "Cross-View Consistency (mean)"),
        ("lpips_inter_mean", "LPIPS Inter (mean)"),
        ("lpips_consistency_mean", "LPIPS Consistency (mean)"),
    ]

    # Only metrics that exist
    def has_metric(col):
        if any(col in df.columns for df in baseline_data.values()):
            return True
        for prompt_map in experiment_data_by_prompt.values():
            for exp_df in prompt_map.values():
                if col in exp_df.columns:
                    return True
        return False

    available_metrics = [(c, label) for c, label in metrics if has_metric(c)]
    if not available_metrics:
        print(f"No metrics available for {kernel_type}_{repulsion_type}")
        return None

    # Collect λ values (union across prompts)
    all_lambda_values = sorted({
        lv for prompt_map in experiment_data_by_prompt.values() for lv in prompt_map.keys()
    }, key=get_lambda_repulsion_order)

    n_metrics = len(available_metrics)
    n_lambdas = len(all_lambda_values)

    fig, axes = plt.subplots(
        n_metrics, n_lambdas,
        figsize=(4 * n_lambdas, 3.5 * n_metrics),
        squeeze=False
    )

    prompts = list(experiment_data_by_prompt.keys())
    cmap = plt.cm.get_cmap("tab10", max(1, len(prompts)))

    for m_idx, (metric_col, metric_label) in enumerate(available_metrics):
        for l_idx, lambda_val in enumerate(all_lambda_values):
            ax = axes[m_idx, l_idx]

            # plot each prompt in its own color
            for p_idx, prompt in enumerate(prompts):
                color = cmap(p_idx)
                base_key = f"prompt={prompt}_seed=42"

                # Baseline
                if base_key in baseline_data and metric_col in baseline_data[base_key].columns:
                    base_df = baseline_data[base_key][["step", metric_col]].dropna().sort_values("step")
                    if len(base_df):
                        ax.plot(
                            base_df["step"], base_df[metric_col],
                            "--", linewidth=1.8, alpha=0.7, color=color,
                            label=f"{prompt} baseline" if (m_idx == 0 and l_idx == 0) else None
                        )

                # Experiment
                exp_map = experiment_data_by_prompt.get(prompt, {})
                if lambda_val in exp_map and metric_col in exp_map[lambda_val].columns:
                    exp_df = exp_map[lambda_val][["step", metric_col]].dropna().sort_values("step")
                    if len(exp_df):
                        ax.plot(
                            exp_df["step"], exp_df[metric_col],
                            "-", linewidth=2, color=color,
                            label=f"{prompt} λ={lambda_val}" if (m_idx == 0 and l_idx == 0) else None
                        )

            # labels
            if l_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if m_idx == 0:
                ax.set_title(f"λ={lambda_val}", fontsize=12, fontweight="bold")
            if m_idx == n_metrics - 1:
                ax.set_xlabel("Step", fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    # # add legend once
    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # if handles:
    #     fig.legend(handles, labels, loc="upper center", fontsize=8, ncol=3)

    # fig.suptitle(f"{kernel_type.upper()} Kernel — {repulsion_type.upper()} (all prompts)", fontsize=14, y=0.995)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])

    # outpath = output_dir / f"comparison_{kernel_type}_{repulsion_type}_all_prompts_GRID.png"
    # fig.savefig(outpath, dpi=300, bbox_inches="tight")
    # plt.close(fig)
    # return outpath

    # add legend once (아래: 제목과 겹치지 않게 상단 바깥에 배치)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # 중복 라벨 제거(같은 라벨이 반복 수집되는 경우 대비)
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        handles, labels = list(uniq.values()), list(uniq.keys())

        n_items = len(labels)

        # 1) config에서 강제 지정 가능
        forced_ncol = config_params.get("legend_ncol", None)
        if forced_ncol is not None:
            ncol = int(forced_ncol)
        else:
            # 2) 자동: 4개면 2x2, 그 외엔 가로로 최대 4개까지
            ncol = 2 if n_items == 4 else min(n_items, 4)

        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),  # 제목 바로 아래
            fontsize=8,
            ncol=ncol,
            frameon=True,
            borderaxespad=0.2
        )

    # 제목(맨 위)
    fig.suptitle(
        f"{kernel_type.upper()} Kernel — {repulsion_type.upper()} (all prompts)",
        fontsize=14, y=0.99
    )

    # 상단 10%는 제목+legend 공간으로 비워두기
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    outpath = output_dir / f"comparison_{kernel_type}_{repulsion_type}_all_prompts_GRID.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ###### Pareto Frontier Analysis ######
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ---------- helpers ----------
# def pareto_efficient_mask(points: np.ndarray, maximize: tuple[bool, ...]) -> np.ndarray:
#     """
#     Return a boolean mask for Pareto-efficient points.
#     points: shape (N, D) array of objective values.
#     maximize: tuple of booleans, length D. True if that objective is to be maximized.
#     """
#     if points.ndim != 2:
#         raise ValueError("points must be a 2D array (N, D)")
#     if len(maximize) != points.shape[1]:
#         raise ValueError("len(maximize) must equal number of columns in points")

#     # Convert maximization to minimization by negating objectives we want to maximize
#     signed = points.copy()
#     for j, is_max in enumerate(maximize):
#         if is_max:
#             signed[:, j] = -signed[:, j]

#     # A point i is Pareto-efficient if no other point strictly dominates it
#     N = signed.shape[0]
#     efficient = np.ones(N, dtype=bool)
#     for i in range(N):
#         if not efficient[i]:
#             continue
#         # Any point that dominates i? (all <= and at least one <)
#         dominates = np.all(signed <= signed[i], axis=1) & np.any(signed < signed[i], axis=1)
#         efficient[dominates] = False  # dominated points cannot be efficient
#     return efficient

# def build_summary_table(config_groups, baseline_data, experiment_dir) -> pd.DataFrame:
#     """
#     Light refactor of your create_tradeoff_analysis_table that RETURNS a DataFrame
#     (and doesn't print/save). Keeps your plateau logic.
#     """
#     key_metrics = [
#         ("fidelity_mean", "CLIP Fidelity"),
#         ("inter_particle_diversity_mean", "Inter-Particle Diversity"),
#         ("cross_view_consistency_mean", "Cross-View Consistency"),
#     ]

#     rows = []
#     for (kernel_type, prompt, repulsion_type), lambda_configs in config_groups.items():
#         baseline_key = f"prompt={prompt}_seed=42"
#         baseline_final = {}
#         if baseline_key in baseline_data:
#             baseline_df = baseline_data[baseline_key]
#             for metric_col, _ in key_metrics:
#                 if metric_col in baseline_df.columns and len(baseline_df) > 0:
#                     baseline_final[metric_col] = baseline_df[metric_col].iloc[-1]
#                 else:
#                     baseline_final[metric_col] = np.nan
#         else:
#             # If baseline missing, skip this group
#             continue

#         for lambda_val, config_name in lambda_configs.items():
#             metrics_path = Path(experiment_dir) / config_name / "metrics" / "quantitative_metrics.csv"
#             if not metrics_path.exists():
#                 continue
#             exp_df = pd.read_csv(metrics_path)
#             if "fidelity_mean" not in exp_df.columns or len(exp_df) == 0:
#                 continue

#             plateaued_step = find_plateaued_step(exp_df, "fidelity_mean")
#             if plateaued_step < 0:
#                 continue

#             row = {
#                 "Kernel": kernel_type.upper(),
#                 "Prompt": str(prompt).title(),
#                 "Repulsion": repulsion_type.upper(),
#                 "Lambda": lambda_val,
#                 "Step": plateaued_step,
#             }

#             # Compare metrics at the plateau step vs baseline final
#             for metric_col, metric_name in key_metrics:
#                 if metric_col in exp_df.columns:
#                     exp_val = exp_df[metric_col].iloc[min(plateaued_step, len(exp_df) - 1)]
#                     base_val = baseline_final.get(metric_col, np.nan)
#                     row[f"{metric_name} (Exp)"] = exp_val
#                     row[f"{metric_name} (Base)"] = base_val
#                     if not (np.isnan(exp_val) or np.isnan(base_val)):
#                         pct = (exp_val - base_val) / (base_val + 1e-12) * 100.0
#                     else:
#                         pct = np.nan
#                     sign = "Improvement" if metric_col == "inter_particle_diversity_mean" else "Change"
#                     row[f"{metric_name} {sign} (%)"] = pct
#             rows.append(row)

#     return pd.DataFrame(rows)


# # ---------- plotting ----------
# def plot_pareto_subplots(
#     df: pd.DataFrame,
#     x_metric: str = "Inter-Particle Diversity Improvement (%)",
#     y_metric: str = "CLIP Fidelity Change (%)",
#     annotate_lambda: bool = True,
#     annotate_step: bool = False,
#     group_prompts: bool = True,
#     title_suffix: str = "",
#     output_dir: str = "analysis/pareto_analysis"
# ):
#     """
#     Make a 2x2 grid of subplots for (Kernel x Repulsion).
#     Each subplot: scatter of all points (lambdas, all prompts), highlight Pareto frontier.
#     x_metric, y_metric: columns of df to use on axes.
#     """
#     # Validate
#     needed = ["Kernel", "Repulsion", "Prompt", "Lambda", "Step", x_metric, y_metric]
#     for c in needed:
#         if c not in df.columns:
#             raise ValueError(f"Missing column: {c}")

#     # Define the 4 panels order
#     cases = [
#         ("COSINE", "RLSD"),
#         ("COSINE", "SVGD"),
#         ("RBF", "RLSD"),
#         ("RBF", "SVGD"),
#     ]
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

#     for ax, (kernel, rep) in zip(axes.ravel(), cases):
#         sub = df[(df["Kernel"] == kernel) & (df["Repulsion"] == rep)].copy()
#         ax.set_title(f"{kernel} + {rep}{(' — ' + title_suffix) if title_suffix else ''}")
#         ax.set_xlabel(x_metric)
#         ax.set_ylabel(y_metric)
#         if sub.empty:
#             ax.text(0.5, 0.5, "No data", ha="center", va="center", alpha=0.6)
#             continue

#         # Optional: different markers/colors per prompt for readability
#         if group_prompts:
#             for prompt, g in sub.groupby("Prompt"):
#                 ax.scatter(g[x_metric], g[y_metric], label=prompt, alpha=0.7)
#                 if annotate_lambda:
#                     for _, r in g.iterrows():
#                         lbl = f"λ={r['Lambda']}"
#                         if annotate_step:
#                             lbl += f"\nS{int(r['Step'])}"
#                         ax.annotate(lbl, (r[x_metric], r[y_metric]), fontsize=8, xytext=(3, 3), textcoords="offset points")
#         else:
#             ax.scatter(sub[x_metric], sub[y_metric], alpha=0.8)
#             if annotate_lambda:
#                 for _, r in sub.iterrows():
#                     lbl = f"λ={r['Lambda']}"
#                     if annotate_step:
#                         lbl += f"\nS{int(r['Step'])}"
#                     ax.annotate(lbl, (r[x_metric], r[y_metric]), fontsize=8, xytext=(3, 3), textcoords="offset points")

#         # Compute Pareto frontier (maximize both axes by default)
#         pts = sub[[x_metric, y_metric]].to_numpy()
#         mask = pareto_efficient_mask(pts, maximize=(True, True))
#         pareto_pts = sub[mask].sort_values([x_metric, y_metric])
#         # Draw frontier
#         ax.scatter(pareto_pts[x_metric], pareto_pts[y_metric], s=70, edgecolor="k", linewidth=1.0, zorder=3)
#         ax.plot(pareto_pts[x_metric], pareto_pts[y_metric], linestyle="--", zorder=2)
#         # Optional legend for prompts
#         if group_prompts:
#             ax.legend(frameon=False, fontsize=8)

#         # Light grid
#         ax.grid(True, alpha=0.25)

#     fig.suptitle("Pareto Frontier: Diversity vs Fidelity (↑ better on both axes)", fontsize=14)
    
#     # Save figure
#     if output_dir is not None:
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         fig_path = output_dir / f"pareto_diversity_vs_fidelity_change_{x_metric.replace(' ', '_')}_vs_{y_metric.replace(' ', '_')}.png"
#         fig.savefig(fig_path, dpi=300)
#         print(f"Saved Pareto plot → {fig_path}")

#         # if save_csv:
#         #     pareto_all = pd.concat(pareto_rows, ignore_index=True)
#         #     csv_path = output_dir / f"pareto_points.csv"
#         #     pareto_all.to_csv(csv_path, index=False)
#         #     print(f"Saved Pareto-optimal configs → {csv_path}")

#     plt.show()


# # ---------- high-level runner ----------
# def run_pareto_plots(
#     config_groups,
#     baseline_data,
#     experiment_dir,
#     x_metric="Inter-Particle Diversity Improvement (%)",
#     y_metric="CLIP Fidelity Change (%)",
#     output_dir: str = "analysis/pareto_analysis"
# ):
#     """
#     Convenience wrapper: builds the summary df and renders 2x2 Pareto subplots.
#     """
#     df = build_summary_table(config_groups, baseline_data, experiment_dir)

#     # Only keep rows with valid x/y
#     df = df.dropna(subset=[x_metric, y_metric]).copy()

#     # Optional: enforce your criterion (diversity must be > baseline)
#     df = df[df["Inter-Particle Diversity Improvement (%)"] > 0].copy()

    # # Plot
    # plot_pareto_subplots(
    #     df,
    #     x_metric=x_metric,
    #     y_metric=y_metric,
    #     annotate_lambda=True,
    #     annotate_step=False,
    #     group_prompts=True,
    #     title_suffix="(at fidelity plateau step)",
    #     output_dir=output_dir
    # )

    # return df  # so you can inspect/save if you want





def main():
    parser = argparse.ArgumentParser(description='Compare lambda_repulsion results against baseline')
    parser.add_argument('--baseline_exp', type=str, default="exp0_baseline",
                        help='Baseline experiment name (default: exp0_baseline)')
    parser.add_argument('--experiment_exp', type=str, default="exp1_lambda_coarse",
                        help='Experiment name with lambda_repulsion variations (default: exp1_lambda_coarse)')
    parser.add_argument('--base_dir', type=str, default="exp",
                        help='Base directory for experiments (default: exp)')
    parser.add_argument('--output_dir', type=str, default="analysis",
                        help='Output directory for comparison plots (default: comparison_plots)')
    parser.add_argument('--compare_plots_multi_prompts', action='store_true', default=False,
                        help='Whether to generate comparison plots (default: True)')
    parser.add_argument('--compare_plots_single_prompt', action='store_true', default=False,
                        help='Whether to generate comparison plots (default: True)')
    parser.add_argument('--pareto_plots', action='store_true', default=False,
                        help='Whether to generate Pareto plots (default: True)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.compare_plots_multi_prompts or args.compare_plots_single_prompt:
        comparison_plots_output_dir = Path(args.output_dir) / "comparison_plots"
        comparison_plots_output_dir.mkdir(exist_ok=True)

    if args.pareto_plots:
        pareto_analysis_output_dir = Path(args.output_dir) / "pareto_analysis"
        pareto_analysis_output_dir.mkdir(exist_ok=True)
    
    # Get baseline configs
    baseline_dir = Path(f"{args.base_dir}/{args.baseline_exp}")
    baseline_configs = []
    for item in baseline_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
            baseline_configs.append(item.name)
    
    print(f"Found baseline configs: {baseline_configs}")
    
    # Get experiment configs
    experiment_dir = Path(f"{args.base_dir}/{args.experiment_exp}")
    experiment_configs = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
            experiment_configs.append(item.name)
    
    print(f"Found experiment configs: {len(experiment_configs)}")
    
    # Parse experiment configs
    config_groups = {}
    for config_name in experiment_configs:
        params = parse_config_name(config_name)
        if params:
            key = (params['kernel_type'], params['prompt'], params['repulsion_type'])
            if key not in config_groups:
                config_groups[key] = {}
            config_groups[key][params['lambda_repulsion']] = config_name
    
    print(f"Found {len(config_groups)} unique combinations")
    
    # Load baseline data for each prompt
    baseline_data = {}
    for config_name in baseline_configs:
        metrics_path = baseline_dir / config_name / "metrics" / "quantitative_metrics.csv"
        if metrics_path.exists():
            baseline_data[config_name] = pd.read_csv(metrics_path)
            print(f"Loaded baseline data for {config_name}")
 
    if args.compare_plots_multi_prompts:
        saved_plots = []
        pairs = sorted({(k, r) for (k, _p, r) in config_groups.keys()})

        for (kernel_type, repulsion_type) in pairs:
            print(f"\nProcessing: {kernel_type}_{repulsion_type} (ALL PROMPTS in grid)")

            experiment_data_by_prompt = {}
            prompts_for_pair = sorted({p for (k, p, r) in config_groups.keys()
                                       if k == kernel_type and r == repulsion_type})

            for prompt in prompts_for_pair:
                lambda_configs = config_groups[(kernel_type, prompt, repulsion_type)]
                lambda_to_df = {}
                for lambda_val, config_name in lambda_configs.items():
                    metrics_path = experiment_dir / config_name / "metrics" / "quantitative_metrics.csv"
                    if metrics_path.exists():
                        lambda_to_df[lambda_val] = pd.read_csv(metrics_path)
                        print(f"  Loaded {prompt} | λ={lambda_val}")
                if lambda_to_df:
                    experiment_data_by_prompt[prompt] = lambda_to_df

            if not experiment_data_by_prompt:
                continue

            config_params = {"kernel_type": kernel_type, "repulsion_type": repulsion_type}
            plot_path = create_comparison_plot_multi_prompt(
                baseline_data, experiment_data_by_prompt, config_params, comparison_plots_output_dir
            )
            if plot_path:
                saved_plots.append(plot_path)
                print(f"  Generated: {plot_path.name}")

        print(f"\nGenerated {len(saved_plots)} multi-prompt grid plots in {comparison_plots_output_dir}")

    # Create comparison plots for each combination
    
    if args.compare_plots_single_prompt:
        saved_plots = []
        
        for (kernel_type, prompt, repulsion_type), lambda_configs in config_groups.items():
            print(f"\nProcessing: {kernel_type}_{prompt}_{repulsion_type}")
            
            # Find corresponding baseline
            baseline_key = f"prompt={prompt}_seed=42"
            if baseline_key not in baseline_data:
                print(f"Warning: No baseline data found for {baseline_key}")
                continue
            
            baseline_metrics = baseline_data[baseline_key]
            
            # Load experiment data for all lambda values
            experiment_data = {}
            for lambda_val, config_name in lambda_configs.items():
                metrics_path = experiment_dir / config_name / "metrics" / "quantitative_metrics.csv"
                if metrics_path.exists():
                    experiment_data[lambda_val] = pd.read_csv(metrics_path)
                    print(f"  Loaded data for λ={lambda_val}")
                else:
                    print(f"  Warning: No metrics found for {config_name}")
            
            if len(experiment_data) == 0:
                print(f"  No experiment data available, skipping...")
                continue
            
            # Create comparison plot
            config_params = {
                'kernel_type': kernel_type,
                'prompt': prompt,
                'repulsion_type': repulsion_type
            }
            
            plot_path = create_comparison_plot(baseline_metrics, experiment_data, config_params, comparison_plots_output_dir)
            if plot_path:
                saved_plots.append(plot_path)
                
                print(f"  Generated comparison plot: {plot_path.name}")
        
        print(f"\nGenerated {len(saved_plots)} comparison plots in {comparison_plots_output_dir}")
    
    # # Create summary table
    # # create_tradeoff_analysis_table(config_groups, baseline_data, experiment_dir, tradeoff_analysis_output_dir)
    
    # if args.pareto_plots:    
    #     # 1) Pareto for Diversity ↑ vs Fidelity change ↑
    #     pareto_df = run_pareto_plots(
    #         config_groups=config_groups,
    #         baseline_data=baseline_data,
    #         experiment_dir=experiment_dir,
    #         x_metric="Inter-Particle Diversity Improvement (%)",
    #         y_metric="CLIP Fidelity Change (%)",
    #         output_dir=pareto_analysis_output_dir
    #     )

    #     # 2) If you also want Diversity ↑ vs Consistency change ↑
    #     plot_pareto_subplots(
    #         pareto_df,
    #         x_metric="Inter-Particle Diversity Improvement (%)",
    #         y_metric="Cross-View Consistency Change (%)",
    #         annotate_lambda=True,
    #         annotate_step=False,
    #         group_prompts=True,
    #         title_suffix="(Consistency axis)",
    #         output_dir=pareto_analysis_output_dir
    #     )


# def detect_plateau(metric_series, window_size=5, threshold=0.01):
#     """
#     Detect if a metric has plateaued by checking if the last window_size values
#     have a standard deviation below threshold.
    
#     Args:
#         metric_series: Series of metric values
#         window_size: Number of last values to check for plateau
#         threshold: Maximum allowed std dev for plateau detection
    
#     Returns:
#         bool: True if plateaued, False otherwise
#     """
#     if len(metric_series) < window_size:
#         return False
    
#     last_values = metric_series.tail(window_size)
#     return last_values.std() < threshold

def find_plateaued_step(exp_df, metric_col, window_size=5, threshold=0.01):
    """
    Find the step where the metric has plateaued, or return the last step if no plateau detected.
    
    Args:
        exp_df: DataFrame with metric values
        metric_col: Column name for the metric
        window_size: Window size for plateau detection
        threshold: Threshold for plateau detection
    
    Returns:
        int: Step index where plateau is detected, or last step if no plateau
    """
    if metric_col not in exp_df.columns or len(exp_df) == 0:
        return -1
    
    metric_series = exp_df[metric_col]
    
    # Check from the end backwards for plateau
    for i in range(len(metric_series) - window_size + 1, 0, -1):
        window = metric_series.iloc[i-1:i-1+window_size]
        if window.std() < threshold:
            return i + window_size - 2  # Return the last step of the plateau
    
    # If no plateau found, return the last step
    return len(metric_series) - 1

# def create_tradeoff_analysis_table(config_groups, baseline_data, experiment_dir, output_dir):
#     """Create a summary table comparing final metrics with focus on diversity vs fidelity trade-offs."""
#     print("\nCreating summary table...")
    
#     # Define key metrics for analysis
#     key_metrics = [
#         ("fidelity_mean", "CLIP Fidelity"),
#         ("inter_particle_diversity_mean", "Inter-Particle Diversity"),
#         ("cross_view_consistency_mean", "Cross-View Consistency"),
#     ]
    
#     summary_data = []
#     valid_configs = []  # Store configurations that meet criteria
    
#     for (kernel_type, prompt, repulsion_type), lambda_configs in config_groups.items():
#         # Get baseline final values
#         baseline_key = f"prompt={prompt}_seed=42"
#         baseline_final = {}
#         if baseline_key in baseline_data:
#             baseline_df = baseline_data[baseline_key]
#             for metric_col, metric_name in key_metrics:
#                 if metric_col in baseline_df.columns:
#                     final_val = baseline_df[metric_col].iloc[-1] if len(baseline_df) > 0 else np.nan
#                     baseline_final[metric_col] = final_val
        
#         # Get experiment final values
#         for lambda_val, config_name in lambda_configs.items():
#             metrics_path = experiment_dir / config_name / "metrics" / "quantitative_metrics.csv"
#             if metrics_path.exists():
#                 exp_df = pd.read_csv(metrics_path)
                
#                 # Find plateaued step for fidelity
#                 plateaued_step = find_plateaued_step(exp_df, "fidelity_mean")
                
#                 if plateaued_step >= 0:
#                     row = {
#                         'Kernel': kernel_type.upper(),
#                         'Prompt': prompt.title(),
#                         'Repulsion Type': repulsion_type.upper(),
#                         'Lambda': lambda_val,
#                         'Plateaued_Step': plateaued_step,
#                     }
                    
#                     # Calculate percentage changes for key metrics at plateaued step
#                     for metric_col, metric_name in key_metrics:
#                         if metric_col in exp_df.columns and metric_col in baseline_final:
#                             exp_val = exp_df[metric_col].iloc[plateaued_step] if len(exp_df) > plateaued_step else np.nan
#                             baseline_val = baseline_final[metric_col]
                            
#                             if not np.isnan(exp_val) and not np.isnan(baseline_val):
#                                 pct_change = ((exp_val - baseline_val) / baseline_val) * 100
                                
#                                 if metric_col == "inter_particle_diversity_mean":
#                                     # For diversity, positive is improvement
#                                     row[f'{metric_name} Improvement (%)'] = pct_change
#                                 else:
#                                     # For fidelity and consistency, allow both positive and negative changes
#                                     row[f'{metric_name} Change (%)'] = pct_change
#                             else:
#                                 if metric_col == "inter_particle_diversity_mean":
#                                     row[f'{metric_name} Improvement (%)'] = np.nan
#                                 else:
#                                     row[f'{metric_name} Change (%)'] = np.nan
                    
#                     # Check if diversity exceeds baseline (main objective)
#                     diversity_improvement = row.get('Inter-Particle Diversity Improvement (%)', np.nan)
#                     if not np.isnan(diversity_improvement) and diversity_improvement > 0:
#                         valid_configs.append(row)
                    
#                     summary_data.append(row)
    
#     if summary_data:
#         summary_df = pd.DataFrame(summary_data)
        
#         # Filter for valid configurations (diversity > baseline)
#         valid_df = pd.DataFrame(valid_configs)
        
#         if len(valid_df) > 0:
#             # Select the configuration with maximum diversity improvement among valid ones
#             best_config = valid_df.loc[valid_df['Inter-Particle Diversity Improvement (%)'].idxmax()]
            
#             print("\n" + "="*80)
#             print("ROBUST DIVERSITY vs FIDELITY/CONSISTENCY ANALYSIS")
#             print("="*80)
#             print("Criteria applied:")
#             print("1. Only points where fidelity has plateaued (excludes early false peaks)")
#             print("2. Allows realistic drops in fidelity and consistency compared to baseline")
#             print("3. Requires diversity to exceed baseline (main objective)")
#             print("4. Selects λ·step with maximum Δdiversity among valid configurations")
#             print("-"*80)
            
#             # Display the best configuration
#             print("BEST CONFIGURATION (Maximum Diversity Improvement):")
#             print(f"Kernel: {best_config['Kernel']}")
#             print(f"Prompt: {best_config['Prompt']}")
#             print(f"Repulsion Type: {best_config['Repulsion Type']}")
#             print(f"Lambda: {best_config['Lambda']}")
#             print(f"Plateaued Step: {best_config['Plateaued_Step']}")
#             print(f"Diversity Improvement: {best_config['Inter-Particle Diversity Improvement (%)']:.2f}%")
#             print(f"Fidelity Change: {best_config['CLIP Fidelity Change (%)']:.2f}%")
#             print(f"Consistency Change: {best_config['Cross-View Consistency Change (%)']:.2f}%")
            
#             # Display all valid configurations sorted by diversity improvement
#             print("\n" + "="*80)
#             print("ALL VALID CONFIGURATIONS (Diversity > Baseline)")
#             print("="*80)
#             print("Sorted by diversity improvement (highest first)")
#             print("Positive values for diversity = improvement over baseline")
#             print("Positive values for fidelity/consistency = improvement, negative = degradation")
#             print("-"*80)
            
#             # Sort valid configurations by diversity improvement
#             valid_df_sorted = valid_df.sort_values('Inter-Particle Diversity Improvement (%)', ascending=False)
            
#             display_columns = [
#                 'Kernel', 'Prompt', 'Repulsion Type', 'Lambda', 'Plateaued_Step',
#                 'Inter-Particle Diversity Improvement (%)',
#                 'CLIP Fidelity Change (%)',
#                 'Cross-View Consistency Change (%)'
#             ]
            
#             print(valid_df_sorted[display_columns].to_string(index=False, float_format='%.2f'))
            
#             # Save the robust analysis
#             robust_path = output_dir / "robust_diversity_vs_fidelity_tradeoffs.csv"
#             valid_df_sorted.to_csv(robust_path, index=False)
#             print(f"\nRobust trade-off analysis saved to: {robust_path}")
            
#             # Also save the complete analysis for reference
#             summary_path = output_dir / "diversity_vs_fidelity_tradeoffs.csv"
#             summary_df.to_csv(summary_path, index=False)
#             print(f"Complete analysis saved to: {summary_path}")
            
#         else:
#             print("No configurations meet the criteria (diversity > baseline)")
#             print("Saving complete analysis without filtering...")
#             summary_path = output_dir / "diversity_vs_fidelity_tradeoffs.csv"
#             summary_df.to_csv(summary_path, index=False)
#             print(f"Analysis saved to: {summary_path}")
            
#             # Display top configurations anyway
#             print("\n" + "="*80)
#             print("TOP CONFIGURATIONS (All Data)")
#             print("="*80)
#             display_columns = [
#                 'Kernel', 'Prompt', 'Repulsion Type', 'Lambda', 'Plateaued_Step',
#                 'Inter-Particle Diversity Improvement (%)',
#                 'CLIP Fidelity Change (%)',
#                 'Cross-View Consistency Change (%)'
#             ]
            
#             # Sort by diversity improvement
#             summary_df_sorted = summary_df.sort_values('Inter-Particle Diversity Improvement (%)', ascending=False)
#             print(summary_df_sorted[display_columns].to_string(index=False, float_format='%.2f'))
    
#     else:
#         print("No data available for summary table")

if __name__ == "__main__":
    main()
