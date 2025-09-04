#!/usr/bin/env python3
"""
Compare lambda_repulsion results against baseline for different combinations
of kernel_type, prompt, and repulsion_type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from pathlib import Path
import re


# Set high-quality plot parameters for dissertation
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

def format_repulsion_type(repulsion_type):
    """Format repulsion type for display."""
    if repulsion_type.lower() == 'rlsd':
        return 'RLSD-F'
    elif repulsion_type.lower() == 'svgd':
        return 'SVGD'
    elif repulsion_type.lower() == 'wo':
        return 'Baseline'
    else:
        return repulsion_type.upper()

def get_prompt_description(prompt_key):
    """Get full description for prompt abbreviations."""
    prompt_descriptions = {
        'hamburger': 'a photo of a hamburger',
        'icecream': 'a photo of an ice cream', 
        'cactus': 'a small saguaro cactus planted in a clay pot',
        'tulip': 'a photo of a tulip'
    }
    return prompt_descriptions.get(prompt_key.lower(), prompt_key.title())

def parse_config_name(config_name):
    """Parse config name to extract parameters."""
    # New pattern: REPULSION__KERNEL__λVALUE__PROMPT__S{SEED}
    # e.g., RLSD__COS__λ100__CACT__S42
    pattern = r'(\w+)__(\w+)__λ([\dK.]+)__(\w+)__S(\d+)'
    match = re.match(pattern, config_name)
    if match:
        # Convert lambda value back to numeric format
        lambda_str = match.group(3)
        if lambda_str.endswith('K'):
            lambda_value = str(int(float(lambda_str[:-1]) * 1000))
        else:
            lambda_value = lambda_str
            
        return {
            'repulsion_type': match.group(1).lower(),  # RLSD -> rlsd
            'kernel_type': match.group(2).lower(),     # COS -> cosine, RBF -> rbf
            'lambda_repulsion': lambda_value,
            'prompt': match.group(4).lower(),          # CACT -> cactus
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
    
    fig, axes = plt.subplots(n_metrics, n_lambdas, figsize=(4.5*n_lambdas, 4.5*n_metrics))
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
                           '--', color='gray', linewidth=2, alpha=0.7, label='Baseline (no repulsion)')
            
            # Plot experiment data
            exp_data = experiment_data[lambda_val]
            if metric_col in exp_data.columns:
                exp_plot = exp_data[["step", metric_col]].dropna().sort_values("step")
                if len(exp_plot) > 0:
                    ax.plot(exp_plot["step"], exp_plot[metric_col], 
                           '-', linewidth=2, label=f'λ={lambda_val} (with repulsion)')
            
            # Set labels and title
            if lambda_idx == 0:  # First column
                ax.set_ylabel(metric_label, fontsize=10)
            if metric_idx == 0:  # First row
                ax.set_title(f'λ={lambda_val}', fontsize=12, fontweight='bold')
            if metric_idx == n_metrics - 1:  # Last row
                ax.set_xlabel('Step', fontsize=10)
            
            # Enhanced grid and styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(labelsize=9, direction='in', length=4, width=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            
            # Add legend only to first subplot
            if metric_idx == 0 and lambda_idx == 0:
                ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    
    # Add overall title
    title = f"{kernel_type.upper()} Kernel - {prompt.title()} - {format_repulsion_type(repulsion_type)}"
    fig.suptitle(title, fontsize=16, y=0.98, fontweight='bold')
    plt.tight_layout()
    
    # Save plot in multiple formats for dissertation quality
    base_name = f"comparison_{kernel_type}_{prompt}_{repulsion_type}"
    output_path_png = output_dir / f"{base_name}.png"
    output_path_pdf = output_dir / f"{base_name}.pdf"
    output_path_svg = output_dir / f"{base_name}.svg"
    
    plt.savefig(output_path_png, bbox_inches="tight", dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_path_pdf, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.savefig(output_path_svg, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path_png

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
        figsize=(4.5 * n_lambdas, 4 * n_metrics),
        squeeze=False
    )

    prompts = list(experiment_data_by_prompt.keys())
    cmap = plt.colormaps.get_cmap("tab10")

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
                            "--", linewidth=1.8, alpha=0.7, color=color
                        )

                # Experiment
                exp_map = experiment_data_by_prompt.get(prompt, {})
                if lambda_val in exp_map and metric_col in exp_map[lambda_val].columns:
                    exp_df = exp_map[lambda_val][["step", metric_col]].dropna().sort_values("step")
                    if len(exp_df):
                        ax.plot(
                            exp_df["step"], exp_df[metric_col],
                            "-", linewidth=2, color=color
                        )

            # labels
            if l_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if m_idx == 0:
                ax.set_title(f"λ={lambda_val}", fontsize=12, fontweight="bold")
            if m_idx == n_metrics - 1:
                ax.set_xlabel("Step", fontsize=10)

            # Enhanced grid and styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(labelsize=9, direction='in', length=4, width=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)


    # ---- Legend: 색(프롬프트) / 선스타일(베이스라인 vs 반발) 분리 ----
    import matplotlib.lines as mlines

    # 1) 프롬프트 컬러 전용 핸들 (마커만 써서 선스타일과 혼동 줄이기)
    prompt_handles = []
    for p_idx, prompt in enumerate(prompts):
        color = cmap(p_idx)
        prompt_handles.append(
            mlines.Line2D(
                [], [], marker='o', linestyle='none', markersize=7,
                markerfacecolor=color, markeredgecolor=color,
                label=get_prompt_description(prompt)
            )
        )

    # 2) 선스타일 설명(회색 한 벌로 충분)
    style_handles = [
        mlines.Line2D([], [], color='0.2', linestyle='--', linewidth=2,
                      label='Dashed: Baseline (no repulsion)'),
        mlines.Line2D([], [], color='0.2', linestyle='-', linewidth=2,
                      label='Solid: With repulsion'),
    ]

    # 먼저 제목을 가장 위에
    fig.suptitle(
        f"{kernel_type.upper()} Kernel — {format_repulsion_type(repulsion_type)} (All Prompts)",
        fontsize=16, y=0.995, fontweight='bold'
    )

    # 그 아래 1줄: 프롬프트(색) 레전드
    ncol_prompt = min(4, max(1, len(prompt_handles)))
    leg_prompts = fig.legend(
        handles=prompt_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=ncol_prompt,
        frameon=False,
        title="Prompts (color)"
    )

    # 그 아래 1줄: 선스타일 레전드
    leg_styles = fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
        title="Line style"
    )

    # 타이틀/레전드 영역을 위해 상단 여백을 넉넉히 확보
    plt.tight_layout(rect=[0, 0, 1, 0.86])

    # # Create custom legend with prompt colors and line styles
    # import matplotlib.lines as mlines
    
    # # Create legend handles for prompts (colors only)
    # prompt_handles = []
    # for p_idx, prompt in enumerate(prompts):
    #     color = cmap(p_idx)
    #     prompt_handles.append(
    #         mlines.Line2D([], [], color=color, linestyle='-', linewidth=2, 
    #                      label=get_prompt_description(prompt))
    #     )
    
    # # Add line style explanation
    # style_handles = [
    #     mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.8, 
    #                  label='Dashed lines: Baseline (no repulsion)'),
    #     mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2, 
    #                  label='Solid lines: With repulsion')
    # ]
    
    # # Combine all handles
    # all_handles = style_handles + prompt_handles
    # all_labels = [h.get_label() for h in all_handles]
    
    # n_items = len(all_labels)
    # ncol = 2 if n_items <= 6 else 3

    # fig.legend(
    #     all_handles, all_labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.96),  # 제목 바로 아래
    #     fontsize=9,
    #     ncol=ncol,
    #     frameon=True,
    #     borderaxespad=0.3,
    #     fancybox=True,
    #     shadow=False,
    #     framealpha=0.9
    # )

    # # 제목(맨 위)
    # fig.suptitle(
    #     f"{kernel_type.upper()} Kernel — {format_repulsion_type(repulsion_type)} (All Prompts)",
    #     fontsize=16, y=0.98, fontweight='bold'
    # )

    # # 상단 10%는 제목+legend 공간으로 비워두기
    # plt.tight_layout(rect=[0, 0, 1, 0.88])

    # Save in multiple formats for dissertation quality
    base_name = f"comparison_{kernel_type}_{repulsion_type}_all_prompts_GRID"
    outpath_png = output_dir / f"{base_name}.png"
    outpath_pdf = output_dir / f"{base_name}.pdf"
    outpath_svg = output_dir / f"{base_name}.svg"
    
    fig.savefig(outpath_png, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    fig.savefig(outpath_pdf, bbox_inches="tight", facecolor='white', edgecolor='none')
    fig.savefig(outpath_svg, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)
    return outpath_png

def create_per_prompt_metric_panel(
    baseline_df: pd.DataFrame,
    exp_lambda_to_df: dict[str, pd.DataFrame],
    kernel_type: str,
    repulsion_type: str,
    prompt: str,
    output_dir: Path
):
    """
    For a single prompt, draw three panels (Diversity/Fidelity/Consistency),
    overlaying ALL lambda curves in each panel.
    """
    metrics = [
        ("inter_particle_diversity_mean", "Inter-Particle Diversity (mean)"),
        ("fidelity_mean", "CLIP Fidelity (mean)"),
        ("cross_view_consistency_mean", "Cross-View Consistency (mean)"),
    ]
    # 사용 가능한 메트릭만 남기기
    available = [(c, l) for c, l in metrics
                 if (baseline_df is not None and c in baseline_df.columns)
                 or any(c in df.columns for df in exp_lambda_to_df.values())]
    if not available:
        return None

    # λ 정렬
    lambda_values = sorted(exp_lambda_to_df.keys(), key=get_lambda_repulsion_order)

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 4), squeeze=False)
    axes = axes[0]

    cmap = plt.colormaps.get_cmap("tab10")
    color_map = {lv: cmap(i % 10) for i, lv in enumerate(lambda_values)}

    for ax, (metric_col, metric_label) in zip(axes, available):
        # baseline (프롬프트 고정, 점선 회색)
        if baseline_df is not None and metric_col in baseline_df.columns:
            base = baseline_df[["step", metric_col]].dropna().sort_values("step")
            if len(base):
                ax.plot(base["step"], base[metric_col], "--", linewidth=2, color="0.35",
                        alpha=0.9, label="Baseline")

        # 모든 λ (실선, 색상=λ)
        for lv in lambda_values:
            df = exp_lambda_to_df[lv]
            if metric_col not in df.columns:
                continue
            cur = df[["step", metric_col]].dropna().sort_values("step")
            if len(cur):
                ax.plot(cur["step"], cur[metric_col], "-", linewidth=2,
                        color=color_map[lv], label=f"λ={lv}")

        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=9, direction='in', length=4, width=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    # 공통 Y라벨은 왼쪽 축에만
    axes[0].set_ylabel("Value")

    # 제목 + 레전드(위)
    title = f"{kernel_type.upper()} Kernel — {format_repulsion_type(repulsion_type)} — {get_prompt_description(prompt)}"
    fig.suptitle(title, fontsize=16, y=0.98, fontweight="bold")

    # λ 컬러 범례(한 줄) + 스타일 설명은 라벨 텍스트에 포함됐으므로 별도 박스 불필요
    # 중복 라벨 제거한 핸들 구성
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    ncol = min(6, max(3, len(labels)))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=ncol, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    base = f"panel_{kernel_type}_{repulsion_type}_{prompt}"
    p_png = output_dir / f"{base}.png"
    p_pdf = output_dir / f"{base}.pdf"
    p_svg = output_dir / f"{base}.svg"
    fig.savefig(p_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(p_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(p_svg, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return p_png


def create_metrics_by_prompts_grid(
    baseline_data: dict[str, pd.DataFrame],
    experiment_data_by_prompt: dict[str, dict[str, pd.DataFrame]],
    kernel_type: str,
    repulsion_type: str,
    output_dir: Path,
    unify_row_ylim: bool = True,
):
    """
    한 장의 그림: 행=3개 메트릭, 열=모든 프롬프트.
    각 서브플롯에 해당 프롬프트의 베이스라인(점선)과 모든 λ(실선, 색=λ)를 overlay.
    """
    # 3개 핵심 메트릭
    metrics = [
        ("inter_particle_diversity_mean", "Inter-Particle Diversity (mean)"),
        ("fidelity_mean", "CLIP Fidelity (mean)"),
        ("cross_view_consistency_mean", "Cross-View Consistency (mean)"),
    ]

    prompts = sorted(experiment_data_by_prompt.keys())
    n_rows, n_cols = len(metrics), len(prompts)
    if n_cols == 0:
        print("No prompts to plot.")
        return None

    # λ 값 모으기(모든 프롬프트 합집합) + 색상 지정
    all_lambda_values = sorted(
        {lv for mp in experiment_data_by_prompt.values() for lv in mp.keys()},
        key=get_lambda_repulsion_order
    )
    cmap = plt.colormaps.get_cmap("tab10")
    lambda_color = {lv: cmap(i % 10) for i, lv in enumerate(all_lambda_values)}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.3 * n_cols, 3.8 * n_rows),
        squeeze=False
    )

    # (선택) 같은 행(=같은 메트릭)끼리 y 축 범위 통일
    row_lims = {i: [np.inf, -np.inf] for i in range(n_rows)} if unify_row_ylim else None

    for c_idx, prompt in enumerate(prompts):
        # 베이스라인 키
        base_key = f"prompt={prompt}_seed=42"
        base_df = baseline_data.get(base_key, None)
        # λ→DF 맵
        lambda_map = experiment_data_by_prompt.get(prompt, {})

        for r_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[r_idx, c_idx]

            # baseline (점선 회색)
            if base_df is not None and metric_col in base_df.columns:
                base = base_df[["step", metric_col]].dropna().sort_values("step")
                if len(base):
                    ax.plot(base["step"], base[metric_col], "--", linewidth=2, color="0.35", label="Baseline")

            # 모든 λ (실선, 색=λ)
            for lv in all_lambda_values:
                df = lambda_map.get(lv, None)
                if df is None or metric_col not in df.columns:
                    continue
                cur = df[["step", metric_col]].dropna().sort_values("step")
                if len(cur):
                    ax.plot(cur["step"], cur[metric_col], "-", linewidth=2,
                            color=lambda_color[lv], label=f"λ={lv}")

            # 라벨/타이틀
            if c_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if r_idx == 0:
                ax.set_title(get_prompt_description(prompt), fontsize=12, fontweight="bold")
            if r_idx == n_rows - 1:
                ax.set_xlabel("Step", fontsize=10)

            # 스타일
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(labelsize=9, direction='in', length=4, width=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

            # (선택) y-lim 집계
            if unify_row_ylim:
                ydata = []
                if base_df is not None and metric_col in base_df.columns:
                    ydata.append(base[metric_col].values)
                for lv in all_lambda_values:
                    df = lambda_map.get(lv, None)
                    if df is not None and metric_col in df.columns:
                        ydata.append(df[metric_col].dropna().values)
                if len(ydata):
                    ymin = np.nanmin(np.concatenate(ydata))
                    ymax = np.nanmax(np.concatenate(ydata))
                    row_lims[r_idx][0] = min(row_lims[r_idx][0], ymin)
                    row_lims[r_idx][1] = max(row_lims[r_idx][1], ymax)

    # 행별 y축 통일 적용
    if unify_row_ylim:
        for r_idx in range(n_rows):
            ymin, ymax = row_lims[r_idx]
            if np.isfinite(ymin) and np.isfinite(ymax):
                pad = 0.03 * (ymax - ymin + 1e-12)
                for c_idx in range(n_cols):
                    axes[r_idx, c_idx].set_ylim(ymin - pad, ymax + pad)

    # 제목 & 레전드
    fig.suptitle(
        f"{kernel_type.upper()} Kernel — {format_repulsion_type(repulsion_type)} (Rows: Metrics, Cols: Prompts)",
        fontsize=16, y=0.995, fontweight="bold"
    )

    # 상단 레전드: 색=λ, 선스타일=베이스라인/반발
    import matplotlib.lines as mlines
    lambda_handles = [
        mlines.Line2D([], [], color=lambda_color[lv], linestyle='-', linewidth=2, label=f"λ={lv}")
        for lv in all_lambda_values
    ]
    style_handles = [
        mlines.Line2D([], [], color='0.35', linestyle='--', linewidth=2, label='Baseline (no repulsion)'),
        mlines.Line2D([], [], color='0.35', linestyle='-', linewidth=2, label='With repulsion'),
    ]
    handles = style_handles + lambda_handles
    ncol = min(6, max(3, len(handles)))
    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=ncol, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    base = f"grid_metrics_by_prompts_{kernel_type}_{repulsion_type}"
    p_png = output_dir / f"{base}.png"
    p_pdf = output_dir / f"{base}.pdf"
    p_svg = output_dir / f"{base}.svg"
    fig.savefig(p_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(p_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(p_svg, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return p_png



def main():
    parser = argparse.ArgumentParser(description='Compare lambda_repulsion results against baseline')
    parser.add_argument('--baseline_exp', type=str, default="exp0_baseline",
                        help='Baseline experiment name (default: exp0_baseline)')
    parser.add_argument('--experiment_exps', type=str, nargs='+', 
                        default=["exp1_lambda_coarse_rlsd", "exp1_lambda_coarse_svgd"],
                        help='Experiment names with lambda_repulsion variations (default: exp1_lambda_coarse_rlsd exp1_lambda_coarse_svgd)')
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
    parser.add_argument('--compare_plots_per_prompt', action='store_true', default=False,
                    help='Per prompt: 3 panels (Div/Fid/Cons) overlaying all lambdas')
    parser.add_argument('--grid_metrics_by_prompts', action='store_true', default=False,
                    help='Make a single figure: rows=3 metrics, cols=prompts; overlay all lambdas.')


    
    args = parser.parse_args()
    
    # Create output directory
    if args.compare_plots_multi_prompts or args.compare_plots_single_prompt:
        comparison_plots_output_dir = Path(args.output_dir) / "comparison_plots"
        comparison_plots_output_dir.mkdir(parents=True, exist_ok=True)

    if args.pareto_plots:
        pareto_analysis_output_dir = Path(args.output_dir) / "pareto_analysis"
        pareto_analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get baseline configs
    baseline_dir = Path(f"{args.base_dir}/{args.baseline_exp}")
    baseline_configs = []
    for item in baseline_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
            baseline_configs.append(item.name)
    
    print(f"Found baseline configs: {baseline_configs}")
    
    # Get experiment configs from all experiment directories
    experiment_configs = []
    for experiment_exp in args.experiment_exps:
        experiment_dir = Path(f"{args.base_dir}/{experiment_exp}")
        if experiment_dir.exists():
            configs = []
            for item in experiment_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
                    configs.append(item.name)
            experiment_configs.extend(configs)
            print(f"Found {len(configs)} experiment configs in {experiment_exp}")
    
    print(f"Total experiment configs: {len(experiment_configs)}")
    
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
            # New baseline naming: PROMPT__S{SEED} -> prompt=PROMPT_seed=SEED
            # e.g., CACT__S42 -> prompt=cactus_seed=42
            if '__S' in config_name:
                parts = config_name.split('__S')
                if len(parts) == 2:
                    prompt_part = parts[0].lower()  # CACT -> cactus
                    seed_part = parts[1]
                    key = f"prompt={prompt_part}_seed={seed_part}"
                else:
                    key = config_name
            else:
                key = config_name
            baseline_data[key] = pd.read_csv(metrics_path)
            print(f"Loaded baseline data for {config_name} -> {key}")
 
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
                    # Find the correct experiment directory for this config
                    experiment_dir = None
                    for exp_name in args.experiment_exps:
                        exp_dir = Path(f"{args.base_dir}/{exp_name}")
                        if exp_dir.exists() and (exp_dir / config_name).exists():
                            experiment_dir = exp_dir
                            break
                    
                    if experiment_dir is None:
                        continue
                        
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

    if args.compare_plots_per_prompt:
        saved = []
        # (kernel, prompt, repulsion) 별로 묶여있는 config_groups 재사용
        pairs = sorted({(k, p, r) for (k, p, r) in config_groups.keys()})
        for (kernel_type, prompt, repulsion_type) in pairs:
            # baseline
            base_key = f"prompt={prompt}_seed=42"
            baseline_df = baseline_data.get(base_key, None)

            # 해당 조합의 모든 λ 데이터 로드
            lambda_configs = config_groups[(kernel_type, prompt, repulsion_type)]
            exp_map = {}
            for lv, cfg in lambda_configs.items():
                # 어느 실험 폴더에 있는지 탐색
                exp_dir = None
                for exp_name in args.experiment_exps:
                    ed = Path(f"{args.base_dir}/{exp_name}")
                    if ed.exists() and (ed / cfg).exists():
                        exp_dir = ed
                        break
                if exp_dir is None:
                    continue
                mp = exp_dir / cfg / "metrics" / "quantitative_metrics.csv"
                if mp.exists():
                    exp_map[lv] = pd.read_csv(mp)

            if not exp_map:
                continue

            outdir = Path(args.output_dir) / "comparison_plots"
            outdir.mkdir(parents=True, exist_ok=True)
            p = create_per_prompt_metric_panel(
                baseline_df=baseline_df,
                exp_lambda_to_df=exp_map,
                kernel_type=kernel_type,
                repulsion_type=repulsion_type,
                prompt=prompt,
                output_dir=outdir
            )
            if p:
                saved.append(p)
                print(f"Generated per-prompt panel: {p.name}")
        print(f"\nGenerated {len(saved)} per-prompt panels in {Path(args.output_dir) / 'comparison_plots'}")

    if args.grid_metrics_by_prompts:
        saved = []
        # (kernel, repulsion) 페어별로 모든 프롬프트를 모아 전달
        pairs = sorted({(k, r) for (k, _p, r) in config_groups.keys()})
        for (kernel_type, repulsion_type) in pairs:
            # prompt -> {lambda -> df} 구성
            prompt_map: dict[str, dict[str, pd.DataFrame]] = {}
            prompts_for_pair = sorted({p for (k, p, r) in config_groups.keys()
                                       if k == kernel_type and r == repulsion_type})
            for prompt in prompts_for_pair:
                lambda_cfgs = config_groups[(kernel_type, prompt, repulsion_type)]
                lm = {}
                for lv, cfg in lambda_cfgs.items():
                    exp_dir = None
                    for exp_name in args.experiment_exps:
                        ed = Path(f"{args.base_dir}/{exp_name}")
                        if ed.exists() and (ed / cfg).exists():
                            exp_dir = ed
                            break
                    if exp_dir is None:
                        continue
                    mp = exp_dir / cfg / "metrics" / "quantitative_metrics.csv"
                    if mp.exists():
                        lm[lv] = pd.read_csv(mp)
                if lm:
                    prompt_map[prompt] = lm

            if not prompt_map:
                continue

            outdir = Path(args.output_dir) / "comparison_plots"
            outdir.mkdir(parents=True, exist_ok=True)
            p = create_metrics_by_prompts_grid(
                baseline_data=baseline_data,
                experiment_data_by_prompt=prompt_map,
                kernel_type=kernel_type,
                repulsion_type=repulsion_type,
                output_dir=outdir,
                unify_row_ylim=True,   # 필요 없으면 False
            )
            if p:
                saved.append(p)
                print(f"Generated metrics×prompts grid: {p.name}")

        print(f"\nGenerated {len(saved)} grids in {Path(args.output_dir) / 'comparison_plots'}")



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
                # Find the correct experiment directory for this config
                experiment_dir = None
                for exp_name in args.experiment_exps:
                    exp_dir = Path(f"{args.base_dir}/{exp_name}")
                    if exp_dir.exists() and (exp_dir / config_name).exists():
                        experiment_dir = exp_dir
                        break
                
                if experiment_dir is None:
                    print(f"  Warning: No experiment directory found for {config_name}")
                    continue
                    
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


if __name__ == "__main__":
    main()
