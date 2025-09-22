#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    'savefig.dpi':300,
    'figure.dpi':150,
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'font.size':14,
    'axes.titlesize':16,
    'axes.labelsize':14,
    'legend.fontsize':13,
    'xtick.labelsize':13,
    'ytick.labelsize':13,
})
sns.set_palette('colorblind')

ROOT = Path('/Users/sj/3D-Generation')
RESULTS_CSV = ROOT / 'results' / 'csv'
REP_KERNEL_DIR = ROOT / 'results' / 'repulsion_kernel'
HYPERPARAM_DIR = ROOT / 'results' / 'hyperparameters'

# New output directories with baseline overlay
REP_KERNEL_PARETO_WB = REP_KERNEL_DIR / 'pareto_w_baseline'
HYPERPARAM_PARETO_WB = HYPERPARAM_DIR / 'pareto_w_baseline'
for d in [REP_KERNEL_PARETO_WB, HYPERPARAM_PARETO_WB]:
    d.mkdir(parents=True, exist_ok=True)


def add_crosshair(ax, x, y):
    ax.axvline(x, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.axhline(y, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.scatter([x],[y], s=300, facecolors='none', edgecolors='black', linewidths=2.0, zorder=6)


def get_selection_params():
    try:
        w_fid = float(os.getenv('WEIGHT_FID', '0.4'))
    except Exception:
        w_fid = 0.4
    try:
        w_div = float(os.getenv('WEIGHT_DIV', '0.6'))
    except Exception:
        w_div = 0.6
    try:
        eps = float(os.getenv('EPSILON_CONS', '0.02'))
    except Exception:
        eps = 0.02
    s = w_fid + w_div
    if s <= 0:
        w_fid, w_div = 0.4, 0.6
    else:
        w_fid, w_div = w_fid/s, w_div/s
    return w_fid, w_div, eps


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def select_utopia_index(df: pd.DataFrame, fidelity_col: str, diversity_col: str, consistency_col: str|None=None) -> int:
    w_fid, w_div, eps = get_selection_params()
    fid = df[fidelity_col].to_numpy(dtype=float)
    div = df[diversity_col].to_numpy(dtype=float)
    cons = df[consistency_col].to_numpy(dtype=float) if (consistency_col is not None and consistency_col in df.columns) else None
    nf = _normalize(fid)
    nd = _normalize(div)
    dist = np.sqrt(w_fid * (1.0 - nf)**2 + w_div * (1.0 - nd)**2)
    if cons is not None and cons.size:
        cons_thr = np.nanmax(cons) - eps
        mask = cons >= cons_thr
        if mask.any():
            idx = int(np.nanargmin(np.where(mask, dist, np.nan)))
            return idx
    return int(np.nanargmin(dist))


def annotate_points(ax, x_vals, y_vals, labels, edge=0.88, selected_idx: int | None = None, offset_default: float = 0.05, offset_selected: float = 0.055, special_positions: dict | None = None):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    if len(x_arr) == 0:
        return

    x_rng = float(np.nanmax(x_arr) - np.nanmin(x_arr)) if np.nanmax(x_arr) != np.nanmin(x_arr) else 1.0
    y_rng = float(np.nanmax(y_arr) - np.nanmin(y_arr)) if np.nanmax(y_arr) != np.nanmin(y_arr) else 1.0

    def get_offsets(i: int):
        frac = offset_selected if (selected_idx is not None and i == selected_idx) else offset_default
        return frac * x_rng, frac * y_rng

    to_axes = ax.transAxes.inverted().transform
    to_disp = ax.transData.transform

    for i, (xi, yi, lab) in enumerate(zip(x_arr, y_arr, labels)):
        dx, dy = get_offsets(i)
        x_off = xi + dx
        y_off = yi + dy
        ha, va = 'left', 'bottom'

        ux, uy = to_axes(to_disp((xi, yi)))

        if special_positions is not None and str(lab) in special_positions:
            pos = str(special_positions[str(lab)]).lower()
            if pos in ('rb','br','right-bottom','bottom-right'):
                x_off, y_off, ha, va = xi + dx, yi - dy, 'left', 'top'
            elif pos in ('rt','tr','right-top','top-right'):
                x_off, y_off, ha, va = xi + dx, yi + dy, 'left', 'bottom'
            elif pos in ('lb','bl','left-bottom','bottom-left'):
                x_off, y_off, ha, va = xi - dx, yi - dy, 'right', 'top'
            elif pos in ('lt','tl','left-top','top-left'):
                x_off, y_off, ha, va = xi - dx, yi + dy, 'right', 'bottom'
            elif pos in ('lm','left-middle'):
                x_off, y_off, ha, va = xi - dx, yi, 'right', 'center'
            elif pos in ('rm','right-middle'):
                x_off, y_off, ha, va = xi + dx, yi, 'left', 'center'
            elif pos in ('tm','top-middle'):
                x_off, y_off, ha, va = xi, yi + dy, 'center', 'bottom'
            elif pos in ('bm','bottom-middle'):
                x_off, y_off, ha, va = xi, yi - dy, 'center', 'top'
        else:
            if uy > edge:
                y_off = yi - dy
                va = 'top'
            if ux > edge:
                x_off = xi - dx
                ha = 'right'

        ax.text(
            x_off, y_off, str(lab),
            fontsize=12, ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='none')
        )


def _set_padded_limits(ax, x_vals, y_vals, pad_frac: float = 0.06, baseline_xy: tuple[float, float] | None = None):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    if baseline_xy is not None and all(np.isfinite(baseline_xy)):
        # include baseline in limit calculation
        x_arr = np.append(x_arr, float(baseline_xy[0]))
        y_arr = np.append(y_arr, float(baseline_xy[1]))
    if x_arr.size == 0 or y_arr.size == 0:
        return
    x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
    y_min, y_max = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
    xr = x_max - x_min if x_max != x_min else 1.0
    yr = y_max - y_min if y_max != y_min else 1.0
    ax.set_xlim(x_min - pad_frac * xr, x_max + pad_frac * xr)
    ax.set_ylim(y_min - pad_frac * yr, y_max + pad_frac * yr)


def load_baseline_metrics():
    """Load baseline averaged metrics. Returns dict or None if unavailable."""
    bfile = RESULTS_CSV / 'exp0_baseline' / 'Baseline_Experiment_Parameter_Analysis_Averaged.csv'
    if not bfile.exists():
        return None
    df = pd.read_csv(bfile, comment='#')
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        'diversity': float(row['diversity_mean_mean']),
        'fidelity': float(row['fidelity_mean_mean']),
        'consistency': float(row['cross_consistency_mean_mean']),
    }


def draw_baseline(ax, x_val: float, y_val: float, label: str = 'Baseline'):
    if x_val is None or y_val is None or not (np.isfinite(x_val) and np.isfinite(y_val)):
        return
    # draw as diamond without text label overlay
    ax.scatter([x_val], [y_val], s=260, marker='D', c='tab:red', edgecolors='black', linewidths=1.0, zorder=7)


def save_single_with_baseline(base_path: Path, x, y, *, xlabel: str, ylabel: str, cvals=None, cmap=None, cbar_label: str|None=None, sel_x=None, sel_y=None, point_labels=None, selected_idx: int | None = None, special_positions: dict | None = None, baseline_xy: tuple[float,float] | None = None):
    fig, ax = plt.subplots(figsize=(6.2,4.2))
    if cvals is not None and cmap is not None:
        sc = ax.scatter(x, y, c=cvals, cmap=cmap, s=200, alpha=0.7)
        if cbar_label:
            plt.colorbar(sc, ax=ax, label=cbar_label)
    else:
        ax.scatter(x, y, s=200, alpha=0.7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _set_padded_limits(ax, x, y, baseline_xy=baseline_xy)
    if baseline_xy is not None:
        draw_baseline(ax, baseline_xy[0], baseline_xy[1])
    if sel_x is not None and sel_y is not None:
        add_crosshair(ax, sel_x, sel_y)
    if point_labels is not None:
        annotate_points(ax, x, y, point_labels, selected_idx=selected_idx, special_positions=special_positions)
    ax.grid(True, linestyle='--', alpha=0.25)
    fig.tight_layout(); fig.savefig(base_path.with_suffix('.png'), bbox_inches='tight'); fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight'); plt.close(fig)


def repulsion_kernel_pareto_with_baseline(baseline: dict | None):
    kfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv'
    mfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'

    if mfile.exists():
        df = pd.read_csv(mfile, comment='#')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        # F vs D (x=diversity, y=fidelity)
        axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], s=200, alpha=0.7)
        _set_padded_limits(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        if baseline is not None:
            draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], df['method'], selected_idx=sel_idx)
        # Fidelity vs Cross-Consistency (x=consistency, y=fidelity)
        axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], s=200, alpha=0.7)
        _set_padded_limits(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity')
        axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], df['method'], selected_idx=sel_idx)
        # Diversity vs Cross-Consistency (x=diversity, y=consistency)
        axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], s=200, alpha=0.7)
        _set_padded_limits(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency')
        axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], df['method'], selected_idx=sel_idx)
        fig.tight_layout(); fig.savefig(REP_KERNEL_PARETO_WB/'repulsion_methods_pareto.png', bbox_inches='tight'); fig.savefig(REP_KERNEL_PARETO_WB/'repulsion_methods_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        # single panels under pareto_w_baseline
        save_single_with_baseline(REP_KERNEL_PARETO_WB/'repulsion_methods_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=df['method'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(REP_KERNEL_PARETO_WB/'repulsion_methods_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=df['method'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(REP_KERNEL_PARETO_WB/'repulsion_methods_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=df['method'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)

    # Kernel type Pareto (categorical, no colorbar)
    if kfile.exists():
        dfk = pd.read_csv(kfile, comment='#')
        if {'fidelity_mean_mean','diversity_mean_mean','cross_consistency_mean_mean'}.issubset(dfk.columns):
            fig, axes = plt.subplots(1,3, figsize=(18,6))
            sel_k = select_utopia_index(dfk, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
            # F vs D
            axes[0].scatter(dfk['diversity_mean_mean'], dfk['fidelity_mean_mean'], s=220, alpha=0.85)
            _set_padded_limits(axes[0], dfk['diversity_mean_mean'], dfk['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
            axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
            if baseline is not None:
                draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
            add_crosshair(axes[0], dfk['diversity_mean_mean'].iloc[sel_k], dfk['fidelity_mean_mean'].iloc[sel_k])
            annotate_points(axes[0], dfk['diversity_mean_mean'], dfk['fidelity_mean_mean'], dfk['kernel'], selected_idx=sel_k)
            # F vs C
            axes[1].scatter(dfk['cross_consistency_mean_mean'], dfk['fidelity_mean_mean'], s=220, alpha=0.85)
            _set_padded_limits(axes[1], dfk['cross_consistency_mean_mean'], dfk['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
            axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
            if baseline is not None:
                draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
            add_crosshair(axes[1], dfk['cross_consistency_mean_mean'].iloc[sel_k], dfk['fidelity_mean_mean'].iloc[sel_k])
            annotate_points(axes[1], dfk['cross_consistency_mean_mean'], dfk['fidelity_mean_mean'], dfk['kernel'], selected_idx=sel_k)
            # D vs C
            axes[2].scatter(dfk['diversity_mean_mean'], dfk['cross_consistency_mean_mean'], s=220, alpha=0.85)
            _set_padded_limits(axes[2], dfk['diversity_mean_mean'], dfk['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
            axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
            if baseline is not None:
                draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
            add_crosshair(axes[2], dfk['diversity_mean_mean'].iloc[sel_k], dfk['cross_consistency_mean_mean'].iloc[sel_k])
            annotate_points(axes[2], dfk['diversity_mean_mean'], dfk['cross_consistency_mean_mean'], dfk['kernel'], selected_idx=sel_k)
            fig.tight_layout(); fig.savefig(REP_KERNEL_PARETO_WB/'kernel_types_pareto.png', bbox_inches='tight'); fig.savefig(REP_KERNEL_PARETO_WB/'kernel_types_pareto.pdf', bbox_inches='tight'); plt.close(fig)
            # single panels under pareto_w_baseline
            save_single_with_baseline(REP_KERNEL_PARETO_WB/'kernel_types_fidelity_vs_diversity', dfk['diversity_mean_mean'], dfk['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', sel_x=dfk['diversity_mean_mean'].iloc[sel_k], sel_y=dfk['fidelity_mean_mean'].iloc[sel_k], point_labels=dfk['kernel'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
            save_single_with_baseline(REP_KERNEL_PARETO_WB/'kernel_types_fidelity_vs_consistency', dfk['cross_consistency_mean_mean'], dfk['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', sel_x=dfk['cross_consistency_mean_mean'].iloc[sel_k], sel_y=dfk['fidelity_mean_mean'].iloc[sel_k], point_labels=dfk['kernel'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
            save_single_with_baseline(REP_KERNEL_PARETO_WB/'kernel_types_consistency_vs_diversity', dfk['diversity_mean_mean'], dfk['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', sel_x=dfk['diversity_mean_mean'].iloc[sel_k], sel_y=dfk['cross_consistency_mean_mean'].iloc[sel_k], point_labels=dfk['kernel'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)

    # Combined 4-type (method, kernel)
    cfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Complete_Analysis.csv'
    if cfile.exists():
        dfc = pd.read_csv(cfile, comment='#')
        needed = {'method','kernel','fidelity_mean_mean','diversity_mean_mean','cross_consistency_mean_mean'}
        if needed.issubset(dfc.columns):
            df4 = dfc.groupby(['method','kernel'], as_index=False)[['fidelity_mean_mean','diversity_mean_mean','cross_consistency_mean_mean']].mean()
            labels = df4.apply(lambda r: f"{r['method']}-{r['kernel']}", axis=1).values
            sel = select_utopia_index(df4, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
            # triptych
            fig, axes = plt.subplots(1,3, figsize=(18,6))
            axes[0].scatter(df4['diversity_mean_mean'], df4['fidelity_mean_mean'], s=220, alpha=0.85)
            _set_padded_limits(axes[0], df4['diversity_mean_mean'], df4['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
            axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
            if baseline is not None:
                draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
            add_crosshair(axes[0], df4['diversity_mean_mean'].iloc[sel], df4['fidelity_mean_mean'].iloc[sel])
            annotate_points(axes[0], df4['diversity_mean_mean'], df4['fidelity_mean_mean'], labels, selected_idx=sel, special_positions={"SVGD-RBF":"lt"})
            axes[1].scatter(df4['cross_consistency_mean_mean'], df4['fidelity_mean_mean'], s=220, alpha=0.85)
            _set_padded_limits(axes[1], df4['cross_consistency_mean_mean'], df4['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
            axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
            if baseline is not None:
                draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
            add_crosshair(axes[1], df4['cross_consistency_mean_mean'].iloc[sel], df4['fidelity_mean_mean'].iloc[sel])
            annotate_points(axes[1], df4['cross_consistency_mean_mean'], df4['fidelity_mean_mean'], labels, selected_idx=sel)
            axes[2].scatter(df4['diversity_mean_mean'], df4['cross_consistency_mean_mean'], s=220, alpha=0.85)
            _set_padded_limits(axes[2], df4['diversity_mean_mean'], df4['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
            axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
            if baseline is not None:
                draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
            add_crosshair(axes[2], df4['diversity_mean_mean'].iloc[sel], df4['cross_consistency_mean_mean'].iloc[sel])
            annotate_points(axes[2], df4['diversity_mean_mean'], df4['cross_consistency_mean_mean'], labels, selected_idx=sel, special_positions={"SVGD-COS":"rt","SVGD-RBF":"lt"})
            fig.tight_layout(); fig.savefig(REP_KERNEL_PARETO_WB/'method_kernel_combined_pareto.png', bbox_inches='tight'); fig.savefig(REP_KERNEL_PARETO_WB/'method_kernel_combined_pareto.pdf', bbox_inches='tight'); plt.close(fig)
            # single panels
            mk_fd_specials = {"SVGD-RBF": "lb"}
            save_single_with_baseline(REP_KERNEL_PARETO_WB/'method_kernel_fidelity_vs_diversity', df4['diversity_mean_mean'], df4['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', sel_x=df4['diversity_mean_mean'].iloc[sel], sel_y=df4['fidelity_mean_mean'].iloc[sel], point_labels=labels, selected_idx=sel, special_positions=mk_fd_specials, baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
            save_single_with_baseline(REP_KERNEL_PARETO_WB/'method_kernel_fidelity_vs_consistency', df4['cross_consistency_mean_mean'], df4['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', sel_x=df4['cross_consistency_mean_mean'].iloc[sel], sel_y=df4['fidelity_mean_mean'].iloc[sel], point_labels=labels, baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
            mk_specials = {"SVGD-COS": "rt", "SVGD-RBF": "lt"}
            save_single_with_baseline(REP_KERNEL_PARETO_WB/'method_kernel_consistency_vs_diversity', df4['diversity_mean_mean'], df4['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', sel_x=df4['diversity_mean_mean'].iloc[sel], sel_y=df4['cross_consistency_mean_mean'].iloc[sel], point_labels=labels, selected_idx=sel, special_positions=mk_specials, baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)


def lambda_cfg_beta_pareto_with_baseline(baseline: dict | None):
    # Lambda (Coarse)
    lam_coarse = RESULTS_CSV / 'exp2_lambda_coarse' / 'Lambda_Repulsion_Coarse_Search_Parameter_Analysis_Averaged.csv'
    if lam_coarse.exists():
        df = pd.read_csv(lam_coarse, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        _set_padded_limits(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        if baseline is not None:
            draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx, offset_default=0.03, offset_selected=0.033)
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        _set_padded_limits(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx, offset_default=0.03, offset_selected=0.033)
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        _set_padded_limits(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx, offset_default=0.03, offset_selected=0.033)
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='Lambda Value')
        fig.tight_layout(); fig.savefig(HYPERPARAM_PARETO_WB/'lambda_repulsion_coarse_pareto.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_PARETO_WB/'lambda_repulsion_coarse_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        # separate saves
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'lambda_repulsion_coarse_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'lambda_repulsion_coarse_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'lambda_repulsion_coarse_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)

    # Lambda (Fine)
    lam = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam.exists():
        df = pd.read_csv(lam, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        _set_padded_limits(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        if baseline is not None:
            draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx)
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        _set_padded_limits(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx)
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        _set_padded_limits(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx, special_positions={"1000":"rb"})
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='Lambda Value')
        fig.tight_layout(); fig.savefig(HYPERPARAM_PARETO_WB/'lambda_repulsion_pareto.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_PARETO_WB/'lambda_repulsion_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        # separate saves
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'lambda_repulsion_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'lambda_repulsion_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'lambda_repulsion_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']], selected_idx=sel_idx, special_positions={"1000": "rb"}, baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)

    # CFG
    cfg = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg.exists():
        df = pd.read_csv(cfg, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['guidance_scale'], cmap='plasma', s=200, alpha=0.7)
        _set_padded_limits(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        if baseline is not None:
            draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['guidance_scale']], selected_idx=sel_idx)
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['guidance_scale'], cmap='plasma', s=200, alpha=0.7)
        _set_padded_limits(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['guidance_scale']], selected_idx=sel_idx)
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['guidance_scale'], cmap='plasma', s=200, alpha=0.7)
        _set_padded_limits(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{int(v)}" for v in df['guidance_scale']], selected_idx=sel_idx)
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='CFG')
        fig.tight_layout(); fig.savefig(HYPERPARAM_PARETO_WB/'guidance_scale_pareto.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_PARETO_WB/'guidance_scale_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'guidance_scale_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['guidance_scale'], cmap='plasma', cbar_label='CFG', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['guidance_scale']], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'guidance_scale_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['guidance_scale'], cmap='plasma', cbar_label='CFG', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['guidance_scale']], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'guidance_scale_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['guidance_scale'], cmap='plasma', cbar_label='CFG', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['guidance_scale']], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)

    # RBF beta
    beta = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta.exists():
        df = pd.read_csv(beta, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['rbf_beta'], cmap='coolwarm', s=200, alpha=0.7)
        _set_padded_limits(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        if baseline is not None:
            draw_baseline(axes[0], baseline['diversity'], baseline['fidelity'])
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{v:.1f}" for v in df['rbf_beta']], selected_idx=sel_idx)
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['rbf_beta'], cmap='coolwarm', s=200, alpha=0.7)
        _set_padded_limits(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[1], baseline['consistency'], baseline['fidelity'])
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{v:.1f}" for v in df['rbf_beta']], selected_idx=sel_idx)
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['rbf_beta'], cmap='coolwarm', s=200, alpha=0.7)
        _set_padded_limits(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        if baseline is not None:
            draw_baseline(axes[2], baseline['diversity'], baseline['consistency'])
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{v:.1f}" for v in df['rbf_beta']], selected_idx=sel_idx, special_positions={"1.0":"lb"})
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='RBF Beta')
        fig.tight_layout(); fig.savefig(HYPERPARAM_PARETO_WB/'rbf_beta_pareto.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_PARETO_WB/'rbf_beta_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'rbf_beta_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['rbf_beta'], cmap='coolwarm', cbar_label='RBF Beta', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{v:.1f}" for v in df['rbf_beta']], baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'rbf_beta_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['rbf_beta'], cmap='coolwarm', cbar_label='RBF Beta', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{v:.1f}" for v in df['rbf_beta']], baseline_xy=(baseline['consistency'], baseline['fidelity']) if baseline else None)
        rb_specials = {"1.0": "lb"}
        save_single_with_baseline(HYPERPARAM_PARETO_WB/'rbf_beta_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['rbf_beta'], cmap='coolwarm', cbar_label='RBF Beta', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{v:.1f}" for v in df['rbf_beta']], selected_idx=sel_idx, special_positions=rb_specials, baseline_xy=(baseline['diversity'], baseline['consistency']) if baseline else None)


def combined_fd_1x3_with_baseline(baseline: dict | None):
    blocks = []
    mfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'
    if mfile.exists():
        df = pd.read_csv(mfile, comment='#')
        blocks.append(('Repulsion Methods', df['diversity_mean_mean'].values, df['fidelity_mean_mean'].values, df['cross_consistency_mean_mean'].values, df['method'].astype(str).values))
    kfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv'
    if kfile.exists():
        df = pd.read_csv(kfile, comment='#')
        blocks.append(('Kernel Types', df['diversity_mean_mean'].values, df['fidelity_mean_mean'].values, df['cross_consistency_mean_mean'].values, df['kernel'].astype(str).values))
    cfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Complete_Analysis.csv'
    if cfile.exists():
        dfc = pd.read_csv(cfile, comment='#')
        needed = {'method','kernel','fidelity_mean_mean','diversity_mean_mean','cross_consistency_mean_mean'}
        if needed.issubset(dfc.columns):
            df4 = dfc.groupby(['method','kernel'], as_index=False)[['fidelity_mean_mean','diversity_mean_mean','cross_consistency_mean_mean']].mean()
            labels = df4.apply(lambda r: f"{r['method']}-{r['kernel']}", axis=1).values
            blocks.append(('Method×Kernel', df4['diversity_mean_mean'].values, df4['fidelity_mean_mean'].values, df4['cross_consistency_mean_mean'].values, labels))

    if len(blocks) == 0:
        return
    blocks = blocks[:3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, x, y, cons, labels) in zip(axes, blocks):
        ax.scatter(x, y, s=180, alpha=0.85)
        _set_padded_limits(ax, x, y, baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        df_tmp = pd.DataFrame({'fidelity': y, 'diversity': x, 'consistency': cons})
        idx = select_utopia_index(df_tmp, 'fidelity', 'diversity', 'consistency') if len(y) else None
        if baseline is not None:
            draw_baseline(ax, baseline['diversity'], baseline['fidelity'])
        if idx is not None:
            add_crosshair(ax, x[idx], y[idx])
        specials = {"SVGD-COS":"rb"} if 'Method×Kernel' in title else None
        annotate_points(ax, x, y, labels, selected_idx=idx, special_positions=specials)
        ax.set_xlabel('Diversity'); ax.set_ylabel('Fidelity'); ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(REP_KERNEL_PARETO_WB/'fidelity_vs_diversity_1x3.png', bbox_inches='tight')
    fig.savefig(REP_KERNEL_PARETO_WB/'fidelity_vs_diversity_1x3.pdf', bbox_inches='tight')
    plt.close(fig)


def hyperparams_fd_1x3_with_baseline(baseline: dict | None):
    tables = []
    lam = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam.exists():
        df = pd.read_csv(lam, comment='#')
        tables.append(('Lambda', df))
    cfg = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg.exists():
        df = pd.read_csv(cfg, comment='#')
        tables.append(('CFG', df))
    beta = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta.exists():
        df = pd.read_csv(beta, comment='#')
        tables.append(('RBF Beta', df))

    if not tables:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, df) in zip(axes, tables[:3]):
        x = df['diversity_mean_mean'].values
        y = df['fidelity_mean_mean'].values
        ax.scatter(x, y, s=180, alpha=0.85)
        _set_padded_limits(ax, x, y, baseline_xy=(baseline['diversity'], baseline['fidelity']) if baseline else None)
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        if baseline is not None:
            draw_baseline(ax, baseline['diversity'], baseline['fidelity'])
        add_crosshair(ax, x[sel_idx], y[sel_idx])
        if 'lambda_repulsion' in df.columns:
            labels = [f"{int(v)}" for v in df['lambda_repulsion']]
            specials = {"1000":"rb"}
        elif 'guidance_scale' in df.columns:
            labels = [f"{int(v)}" for v in df['guidance_scale']]
            specials = None
        elif 'rbf_beta' in df.columns:
            labels = [f"{v:.1f}" for v in df['rbf_beta']]
            specials = {"1.0":"lm", "1.5":"lm"}
        else:
            labels = ['' for _ in range(len(x))]
            specials = None
        annotate_points(ax, x, y, labels, selected_idx=sel_idx, special_positions=specials)
        ax.set_xlabel('Diversity'); ax.set_ylabel('Fidelity'); ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(HYPERPARAM_PARETO_WB/'fidelity_vs_diversity_1x3.png', bbox_inches='tight')
    fig.savefig(HYPERPARAM_PARETO_WB/'fidelity_vs_diversity_1x3.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    baseline = load_baseline_metrics()
    repulsion_kernel_pareto_with_baseline(baseline)
    lambda_cfg_beta_pareto_with_baseline(baseline)
    combined_fd_1x3_with_baseline(baseline)
    hyperparams_fd_1x3_with_baseline(baseline)


if __name__ == '__main__':
    main()


