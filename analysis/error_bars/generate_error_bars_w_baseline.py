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
REPULSION_DIR = ROOT / 'results' / 'repulsion_kernel'
OUT_H_ERR = HYPERPARAM_DIR / 'error_bars_w_baseline'
OUT_R_ERR = REPULSION_DIR / 'error_bars_w_baseline'
for d in [OUT_H_ERR, OUT_R_ERR]:
    d.mkdir(parents=True, exist_ok=True)


def load_baseline_metrics():
    bfile = RESULTS_CSV / 'exp0_baseline' / 'Baseline_Experiment_Parameter_Analysis_Averaged.csv'
    if not bfile.exists():
        return None
    df = pd.read_csv(bfile, comment='#')
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        'fidelity': float(row['fidelity_mean_mean']),
        'fidelity_std': float(row.get('fidelity_mean_std', np.nan)),
        'diversity': float(row['diversity_mean_mean']),
        'diversity_std': float(row.get('diversity_mean_std', np.nan)),
        'consistency': float(row['cross_consistency_mean_mean']),
        'consistency_std': float(row.get('cross_consistency_mean_std', np.nan)),
    }


def _add_baseline_reference(ax, mean_value: float | None, std_value: float | None, label: str):
    if mean_value is None or not np.isfinite(mean_value):
        return
    # std band if available
    if std_value is not None and np.isfinite(std_value) and std_value > 0:
        ax.axhspan(mean_value - std_value, mean_value + std_value, color='tab:red', alpha=0.12, zorder=0)
    ax.axhline(mean_value, color='tab:red', linestyle='--', linewidth=1.4, alpha=0.9)
    # small text at right edge
    try:
        xlim = ax.get_xlim()
        ax.text(xlim[1], mean_value, f"  {label}", va='center', ha='left', color='tab:red')
    except Exception:
        pass


def _save_errs(base_path: Path, x, y, yerr, xlabel: str, ylabel: str, baseline_mean: float | None, baseline_std: float | None):
    fig, ax = plt.subplots(figsize=(6.2,4))
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _add_baseline_reference(ax, baseline_mean, baseline_std, 'Baseline')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout(); fig.savefig(base_path.with_suffix('.png'), bbox_inches='tight'); fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight'); plt.close(fig)


def _save_triptych(base_path: Path, x, f, f_err, d, d_err, c, c_err, xlabel: str, baseline: dict | None):
    fig, axes = plt.subplots(1,3, figsize=(16,4))
    axes[0].errorbar(x, f, yerr=f_err, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel('Fidelity')
    _add_baseline_reference(axes[0], baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None, 'Baseline')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[1].errorbar(x, d, yerr=d_err, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel('Diversity')
    _add_baseline_reference(axes[1], baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None, 'Baseline')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[2].errorbar(x, c, yerr=c_err, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    axes[2].set_xlabel(xlabel); axes[2].set_ylabel('Cross-Consistency')
    _add_baseline_reference(axes[2], baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None, 'Baseline')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout(); fig.savefig(base_path.with_suffix('.png'), bbox_inches='tight'); fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight'); plt.close(fig)


def hyperparam_error_bars():
    baseline = load_baseline_metrics()
    # Lambda
    lam = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam.exists():
        df = pd.read_csv(lam, comment='#').sort_values('lambda_repulsion')
        x = df['lambda_repulsion'].values
        _save_errs(OUT_H_ERR/'lambda_fidelity', x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Lambda', 'Fidelity', baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None)
        _save_errs(OUT_H_ERR/'lambda_diversity', x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Lambda', 'Diversity', baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None)
        _save_errs(OUT_H_ERR/'lambda_consistency', x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Lambda', 'Cross-Consistency', baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None)
        _save_triptych(OUT_H_ERR/'lambda_1x3', x,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'Lambda', baseline)
    # CFG
    cfg = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg.exists():
        df = pd.read_csv(cfg, comment='#').sort_values('guidance_scale')
        x = df['guidance_scale'].values
        _save_errs(OUT_H_ERR/'cfg_fidelity', x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'CFG', 'Fidelity', baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None)
        _save_errs(OUT_H_ERR/'cfg_diversity', x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'CFG', 'Diversity', baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None)
        _save_errs(OUT_H_ERR/'cfg_consistency', x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'CFG', 'Cross-Consistency', baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None)
        _save_triptych(OUT_H_ERR/'cfg_1x3', x,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'CFG', baseline)
    # RBF Beta
    beta = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta.exists():
        df = pd.read_csv(beta, comment='#').sort_values('rbf_beta')
        x = df['rbf_beta'].values
        _save_errs(OUT_H_ERR/'rbf_beta_fidelity', x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'RBF Beta', 'Fidelity', baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None)
        _save_errs(OUT_H_ERR/'rbf_beta_diversity', x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'RBF Beta', 'Diversity', baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None)
        _save_errs(OUT_H_ERR/'rbf_beta_consistency', x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'RBF Beta', 'Cross-Consistency', baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None)
        _save_triptych(OUT_H_ERR/'rbf_beta_1x3', x,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'RBF Beta', baseline)


def repulsion_error_bars():
    baseline = load_baseline_metrics()
    # Method comparison
    mfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'
    if mfile.exists():
        df = pd.read_csv(mfile, comment='#')
        keys = df['method'].astype(str).values
        _save_errs(OUT_R_ERR/'methods_fidelity', keys, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Method', 'Fidelity', baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None)
        _save_errs(OUT_R_ERR/'methods_diversity', keys, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Method', 'Diversity', baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None)
        _save_errs(OUT_R_ERR/'methods_consistency', keys, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Method', 'Cross-Consistency', baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None)
    # Kernel comparison
    kfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv'
    if kfile.exists():
        df = pd.read_csv(kfile, comment='#')
        keys = df['kernel'].astype(str).values
        _save_errs(OUT_R_ERR/'kernels_fidelity', keys, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Kernel', 'Fidelity', baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None)
        _save_errs(OUT_R_ERR/'kernels_diversity', keys, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Kernel', 'Diversity', baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None)
        _save_errs(OUT_R_ERR/'kernels_consistency', keys, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Kernel', 'Cross-Consistency', baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None)
    # Method + Kernel combined comparison (SVGD-COS, SVGD-RBF, RLSD-RBF, RLSD-COS)
    mkfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Complete_Analysis.csv'
    if mkfile.exists():
        df = pd.read_csv(mkfile, comment='#')
        df['combo'] = df['method'].astype(str) + '-' + df['kernel'].astype(str)
        desired_order = ['SVGD-COS', 'SVGD-RBF', 'RLSD-RBF', 'RLSD-COS']
        df['combo'] = pd.Categorical(df['combo'], categories=desired_order, ordered=True)
        df = df.sort_values('combo')
        keys = df['combo'].astype(str).values
        _save_errs(OUT_R_ERR/'method_kernel_fidelity', keys, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Method-Kernel', 'Fidelity', baseline.get('fidelity') if baseline else None, baseline.get('fidelity_std') if baseline else None)
        _save_errs(OUT_R_ERR/'method_kernel_diversity', keys, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Method-Kernel', 'Diversity', baseline.get('diversity') if baseline else None, baseline.get('diversity_std') if baseline else None)
        _save_errs(OUT_R_ERR/'method_kernel_consistency', keys, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Method-Kernel', 'Cross-Consistency', baseline.get('consistency') if baseline else None, baseline.get('consistency_std') if baseline else None)
        _save_triptych(OUT_R_ERR/'method_kernel_1x3', keys,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'Method-Kernel', baseline)


def main():
    hyperparam_error_bars()
    repulsion_error_bars()


if __name__ == '__main__':
    main()


