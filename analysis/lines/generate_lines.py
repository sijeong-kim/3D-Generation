#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'savefig.dpi':300,'figure.dpi':150,'pdf.fonttype':42,'ps.fonttype':42})
sns.set_palette('colorblind')

ROOT = Path('/Users/sj/3D-Generation')
RESULTS_CSV = ROOT / 'results' / 'csv'
HYPERPARAM_DIR = ROOT / 'results' / 'hyperparameters'
HYPERPARAM_LINES = HYPERPARAM_DIR / 'lines'
for d in [HYPERPARAM_LINES]:
    d.mkdir(parents=True, exist_ok=True)

WFID = 0.40
WDIV = 0.60
EPS_CONS = 0.02

def line_with_ring(ax, x, y, ystd, sel_x, sel_y, xlabel, ylabel):
    ax.plot(x, y, marker='o', linewidth=2)
    if ystd is not None:
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.2)
    ax.axvline(sel_x, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.axhline(sel_y, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.scatter([sel_x], [sel_y], s=200, facecolors='none', edgecolors='black', linewidths=1.8, zorder=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.3)

def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    denom = mx - mn
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(arr)
    return (arr - mn) / denom

def select_idx_utopia_with_consistency_constraint(df: pd.DataFrame,
                                                  fidelity_col: str,
                                                  diversity_col: str,
                                                  consistency_col: str,
                                                  w_fid: float = WFID,
                                                  w_div: float = WDIV,
                                                  epsilon: float = EPS_CONS) -> int:
    if df is None or len(df) == 0:
        return 0
    f = df[fidelity_col].to_numpy(dtype=float)
    d = df[diversity_col].to_numpy(dtype=float)
    c = df[consistency_col].to_numpy(dtype=float)

    f_n = _normalize_01(f)
    d_n = _normalize_01(d)

    # ε-constraint on consistency
    c_max = np.nanmax(c) if c.size > 0 else np.nan
    mask = c >= (c_max - epsilon)
    if not np.any(mask):
        mask = np.isfinite(c)
        if not np.any(mask):
            mask = np.ones_like(c, dtype=bool)

    # weighted (squared) Euclidean distance to (1,1)
    dist2 = w_fid * (1.0 - f_n) ** 2 + w_div * (1.0 - d_n) ** 2
    dist2_masked = np.where(mask, dist2, np.inf)
    sel = int(np.nanargmin(dist2_masked))
    return sel

def main():
    # Lambda fine
    lam_file = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam_file.exists():
        df = pd.read_csv(lam_file, comment='#').sort_values('lambda_repulsion')
        x = df['lambda_repulsion'].values
        sel_idx = select_idx_utopia_with_consistency_constraint(
            df,
            fidelity_col='fidelity_mean_mean',
            diversity_col='diversity_mean_mean',
            consistency_col='cross_consistency_mean_mean',
            w_fid=WFID,
            w_div=WDIV,
            epsilon=EPS_CONS,
        ) if len(x)>0 else 0
        sel_x = x[sel_idx]
        fig, axes = plt.subplots(1,3, figsize=(12,3.5))
        line_with_ring(axes[0], x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, sel_x, df['fidelity_mean_mean'].iloc[sel_idx], 'Lambda', 'Fidelity')
        line_with_ring(axes[1], x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, sel_x, df['diversity_mean_mean'].iloc[sel_idx], 'Lambda', 'Diversity')
        line_with_ring(axes[2], x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, sel_x, df['cross_consistency_mean_mean'].iloc[sel_idx], 'Lambda', 'Consistency')
        fig.tight_layout(); fig.savefig(HYPERPARAM_LINES / 'lambda_repulsion_lines.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_LINES / 'lambda_repulsion_lines.pdf', bbox_inches='tight'); plt.close(fig)

    # CFG
    cfg_file = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg_file.exists():
        df = pd.read_csv(cfg_file, comment='#').sort_values('guidance_scale')
        x = df['guidance_scale'].values
        sel_idx = select_idx_utopia_with_consistency_constraint(
            df,
            fidelity_col='fidelity_mean_mean',
            diversity_col='diversity_mean_mean',
            consistency_col='cross_consistency_mean_mean',
            w_fid=WFID,
            w_div=WDIV,
            epsilon=EPS_CONS,
        ) if len(x)>0 else 0
        sel_x = x[sel_idx]
        fig, axes = plt.subplots(1,3, figsize=(12,3.5))
        line_with_ring(axes[0], x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, sel_x, df['fidelity_mean_mean'].iloc[sel_idx], 'CFG', 'Fidelity')
        line_with_ring(axes[1], x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, sel_x, df['diversity_mean_mean'].iloc[sel_idx], 'CFG', 'Diversity')
        line_with_ring(axes[2], x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, sel_x, df['cross_consistency_mean_mean'].iloc[sel_idx], 'CFG', 'Consistency')
        fig.tight_layout(); fig.savefig(HYPERPARAM_LINES / 'guidance_scale_lines.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_LINES / 'guidance_scale_lines.pdf', bbox_inches='tight'); plt.close(fig)

    # RBF beta
    beta_file = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta_file.exists():
        df = pd.read_csv(beta_file, comment='#').sort_values('rbf_beta')
        x = df['rbf_beta'].values
        sel_idx = select_idx_utopia_with_consistency_constraint(
            df,
            fidelity_col='fidelity_mean_mean',
            diversity_col='diversity_mean_mean',
            consistency_col='cross_consistency_mean_mean',
            w_fid=WFID,
            w_div=WDIV,
            epsilon=EPS_CONS,
        ) if len(x)>0 else 0
        sel_x = x[sel_idx]
        fig, axes = plt.subplots(1,3, figsize=(12,3.5))
        line_with_ring(axes[0], x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, sel_x, df['fidelity_mean_mean'].iloc[sel_idx], 'RBF Beta', 'Fidelity')
        line_with_ring(axes[1], x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, sel_x, df['diversity_mean_mean'].iloc[sel_idx], 'RBF Beta', 'Diversity')
        line_with_ring(axes[2], x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, sel_x, df['cross_consistency_mean_mean'].iloc[sel_idx], 'RBF Beta', 'Consistency')
        fig.tight_layout(); fig.savefig(HYPERPARAM_LINES / 'rbf_beta_lines.png', bbox_inches='tight'); fig.savefig(HYPERPARAM_LINES / 'rbf_beta_lines.pdf', bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    main()


