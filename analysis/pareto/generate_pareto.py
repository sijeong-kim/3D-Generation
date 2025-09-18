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
OUT_DIR = ROOT / 'results' / 'pareto'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # normalize weights
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
    # utopia at (1,1)
    dist = np.sqrt(w_fid * (1.0 - nf)**2 + w_div * (1.0 - nd)**2)
    if cons is not None and cons.size:
        cons_thr = np.nanmax(cons) - eps
        mask = cons >= cons_thr
        if mask.any():
            idx = int(np.nanargmin(np.where(mask, dist, np.nan)))
            return idx
    return int(np.nanargmin(dist))

def annotate_points(ax, x_vals, y_vals, labels):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    if len(x_arr) == 0:
        return
    x_rng = float(np.nanmax(x_arr) - np.nanmin(x_arr)) if np.nanmax(x_arr) != np.nanmin(x_arr) else 1.0
    y_rng = float(np.nanmax(y_arr) - np.nanmin(y_arr)) if np.nanmax(y_arr) != np.nanmin(y_arr) else 1.0
    for i, (xi, yi, lab) in enumerate(zip(x_arr, y_arr, labels)):
        # Consistent top-right placement, slightly closer
        x_off = xi + 0.025 * x_rng
        y_off = yi + 0.035 * y_rng
        ax.text(x_off, y_off, str(lab), fontsize=12,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

def save_single(base_path: Path, x, y, *, xlabel: str, ylabel: str, cvals=None, cmap=None, cbar_label: str|None=None, sel_x=None, sel_y=None, point_labels=None):
    fig, ax = plt.subplots(figsize=(6.2,4.2))
    if cvals is not None and cmap is not None:
        sc = ax.scatter(x, y, c=cvals, cmap=cmap, s=200, alpha=0.7)
        if cbar_label:
            plt.colorbar(sc, ax=ax, label=cbar_label)
    else:
        ax.scatter(x, y, s=200, alpha=0.7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if sel_x is not None and sel_y is not None:
        add_crosshair(ax, sel_x, sel_y)
    if point_labels is not None:
        annotate_points(ax, x, y, point_labels)
    ax.grid(True, linestyle='--', alpha=0.25)
    fig.tight_layout(); fig.savefig(base_path.with_suffix('.png'), bbox_inches='tight'); fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight'); plt.close(fig)

def repulsion_kernel_pareto():
    kfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv'
    mfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'
    if not kfile.exists() and not mfile.exists():
        return
    if mfile.exists():
        df = pd.read_csv(mfile, comment='#')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        # F vs D
        axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], s=200, alpha=0.7)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], df['method'])
        # Fidelity vs Cross-Consistency (consistency on x-axis)
        axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], s=200, alpha=0.7)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity')
        axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], df['method'])
        # Diversity vs Cross-Consistency (diversity on x-axis)
        axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], s=200, alpha=0.7)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency')
        axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], df['method'])
        fig.tight_layout(); fig.savefig(OUT_DIR/'repulsion_methods_pareto.png', bbox_inches='tight'); fig.savefig(OUT_DIR/'repulsion_methods_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        # separate saves
        save_single(OUT_DIR/'repulsion_methods_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=df['method'])
        save_single(OUT_DIR/'repulsion_methods_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=df['method'])
        save_single(OUT_DIR/'repulsion_methods_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=df['method'])

def lambda_cfg_beta_pareto():
    # Lambda
    lam = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam.exists():
        df = pd.read_csv(lam, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity');
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']])
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']])
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['lambda_repulsion'], cmap='viridis', s=200, alpha=0.7)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{int(v)}" for v in df['lambda_repulsion']])
        # One colorbar only, on the right of the last subplot
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='Lambda Value')
        fig.tight_layout(); fig.savefig(OUT_DIR/'lambda_repulsion_pareto.png', bbox_inches='tight'); fig.savefig(OUT_DIR/'lambda_repulsion_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        # separate saves
        save_single(OUT_DIR/'lambda_repulsion_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']])
        save_single(OUT_DIR/'lambda_repulsion_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']])
        save_single(OUT_DIR/'lambda_repulsion_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['lambda_repulsion'], cmap='viridis', cbar_label='Lambda Value', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['lambda_repulsion']])

    # CFG
    cfg = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg.exists():
        df = pd.read_csv(cfg, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['guidance_scale'], cmap='plasma', s=200, alpha=0.7)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['guidance_scale']])
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['guidance_scale'], cmap='plasma', s=200, alpha=0.7)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{int(v)}" for v in df['guidance_scale']])
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['guidance_scale'], cmap='plasma', s=200, alpha=0.7)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{int(v)}" for v in df['guidance_scale']])
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='CFG')
        fig.tight_layout(); fig.savefig(OUT_DIR/'guidance_scale_pareto.png', bbox_inches='tight'); fig.savefig(OUT_DIR/'guidance_scale_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        save_single(OUT_DIR/'guidance_scale_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['guidance_scale'], cmap='plasma', cbar_label='CFG', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['guidance_scale']])
        save_single(OUT_DIR/'guidance_scale_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['guidance_scale'], cmap='plasma', cbar_label='CFG', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['guidance_scale']])
        save_single(OUT_DIR/'guidance_scale_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['guidance_scale'], cmap='plasma', cbar_label='CFG', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{int(v)}" for v in df['guidance_scale']])

    # RBF beta
    beta = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta.exists():
        df = pd.read_csv(beta, comment='#')
        sel_idx = select_utopia_index(df, 'fidelity_mean_mean', 'diversity_mean_mean', 'cross_consistency_mean_mean')
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        sc = axes[0].scatter(df['diversity_mean_mean'], df['fidelity_mean_mean'], c=df['rbf_beta'], cmap='coolwarm', s=200, alpha=0.7)
        axes[0].set_xlabel('Diversity'); axes[0].set_ylabel('Fidelity')
        add_crosshair(axes[0], df['diversity_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[0], df['diversity_mean_mean'], df['fidelity_mean_mean'], [f"{v:.1f}" for v in df['rbf_beta']])
        sc1 = axes[1].scatter(df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], c=df['rbf_beta'], cmap='coolwarm', s=200, alpha=0.7)
        axes[1].set_xlabel('Cross-Consistency'); axes[1].set_ylabel('Fidelity'); axes[1].xaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[1], df['cross_consistency_mean_mean'].iloc[sel_idx], df['fidelity_mean_mean'].iloc[sel_idx])
        annotate_points(axes[1], df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], [f"{v:.1f}" for v in df['rbf_beta']])
        sc2 = axes[2].scatter(df['diversity_mean_mean'], df['cross_consistency_mean_mean'], c=df['rbf_beta'], cmap='coolwarm', s=200, alpha=0.7)
        axes[2].set_xlabel('Diversity'); axes[2].set_ylabel('Cross-Consistency'); axes[2].yaxis.set_major_locator(mticker.MaxNLocator(4))
        add_crosshair(axes[2], df['diversity_mean_mean'].iloc[sel_idx], df['cross_consistency_mean_mean'].iloc[sel_idx])
        annotate_points(axes[2], df['diversity_mean_mean'], df['cross_consistency_mean_mean'], [f"{v:.1f}" for v in df['rbf_beta']])
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3.5%", pad=0.2)
        fig.colorbar(sc, cax=cax, label='RBF Beta')
        fig.tight_layout(); fig.savefig(OUT_DIR/'rbf_beta_pareto.png', bbox_inches='tight'); fig.savefig(OUT_DIR/'rbf_beta_pareto.pdf', bbox_inches='tight'); plt.close(fig)
        save_single(OUT_DIR/'rbf_beta_fidelity_vs_diversity', df['diversity_mean_mean'], df['fidelity_mean_mean'], xlabel='Diversity', ylabel='Fidelity', cvals=df['rbf_beta'], cmap='coolwarm', cbar_label='RBF Beta', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{v:.1f}" for v in df['rbf_beta']])
        save_single(OUT_DIR/'rbf_beta_fidelity_vs_consistency', df['cross_consistency_mean_mean'], df['fidelity_mean_mean'], xlabel='Cross-Consistency', ylabel='Fidelity', cvals=df['rbf_beta'], cmap='coolwarm', cbar_label='RBF Beta', sel_x=df['cross_consistency_mean_mean'].iloc[sel_idx], sel_y=df['fidelity_mean_mean'].iloc[sel_idx], point_labels=[f"{v:.1f}" for v in df['rbf_beta']])
        save_single(OUT_DIR/'rbf_beta_consistency_vs_diversity', df['diversity_mean_mean'], df['cross_consistency_mean_mean'], xlabel='Diversity', ylabel='Cross-Consistency', cvals=df['rbf_beta'], cmap='coolwarm', cbar_label='RBF Beta', sel_x=df['diversity_mean_mean'].iloc[sel_idx], sel_y=df['cross_consistency_mean_mean'].iloc[sel_idx], point_labels=[f"{v:.1f}" for v in df['rbf_beta']])

def combined_fd_2x2():
    # Prepare dataframes if available
    blocks = []
    # Repulsion methods
    mfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'
    if mfile.exists():
        df = pd.read_csv(mfile, comment='#')
        blocks.append(('Repulsion Methods', df['diversity_mean_mean'].values, df['fidelity_mean_mean'].values, df['cross_consistency_mean_mean'].values, df['method'].astype(str).values))
    # Lambda
    lam = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam.exists():
        df = pd.read_csv(lam, comment='#')
        blocks.append(('Lambda', df['diversity_mean_mean'].values, df['fidelity_mean_mean'].values, df['cross_consistency_mean_mean'].values, df['lambda_repulsion'].astype(int).astype(str).values))
    # CFG
    cfg = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg.exists():
        df = pd.read_csv(cfg, comment='#')
        blocks.append(('CFG', df['diversity_mean_mean'].values, df['fidelity_mean_mean'].values, df['cross_consistency_mean_mean'].values, df['guidance_scale'].astype(int).astype(str).values))
    # RBF Beta
    beta = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta.exists():
        df = pd.read_csv(beta, comment='#')
        blocks.append(('RBF Beta', df['diversity_mean_mean'].values, df['fidelity_mean_mean'].values, df['cross_consistency_mean_mean'].values, df['rbf_beta'].map(lambda v: f"{v:.1f}").values))

    if not blocks:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (title, x, y, cons, labels) in zip(axes, blocks):
        ax.scatter(x, y, s=180, alpha=0.8)
        # highlight selected by utopia-distance with epsilon-constraint
        df_tmp = pd.DataFrame({'fidelity': y, 'diversity': x, 'consistency': cons})
        idx = select_utopia_index(df_tmp, 'fidelity', 'diversity', 'consistency') if len(y) else None
        if idx is not None:
            add_crosshair(ax, x[idx], y[idx])
        annotate_points(ax, x, y, labels)
        ax.set_xlabel('Diversity'); ax.set_ylabel('Fidelity')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(title)

    # If fewer than 4, hide unused
    for j in range(len(blocks), 4):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(OUT_DIR/'fidelity_vs_diversity_2x2.png', bbox_inches='tight')
    fig.savefig(OUT_DIR/'fidelity_vs_diversity_2x2.pdf', bbox_inches='tight')
    plt.close(fig)

def main():
    repulsion_kernel_pareto()
    lambda_cfg_beta_pareto()
    combined_fd_2x2()

if __name__ == '__main__':
    main()


