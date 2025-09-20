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
OUT_H_ERR = HYPERPARAM_DIR / 'error_bars'
OUT_R_ERR = REPULSION_DIR / 'error_bars'
for d in [OUT_H_ERR, OUT_R_ERR]:
    d.mkdir(parents=True, exist_ok=True)

def _save_errs(base_path: Path, x, y, yerr, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.2,4))
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout(); fig.savefig(base_path.with_suffix('.png'), bbox_inches='tight'); fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight'); plt.close(fig)

def _save_triptych(base_path: Path, x, f, f_err, d, d_err, c, c_err, xlabel: str):
    fig, axes = plt.subplots(1,3, figsize=(16,4))
    axes[0].errorbar(x, f, yerr=f_err, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel('Fidelity')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[1].errorbar(x, d, yerr=d_err, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel('Diversity')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[2].errorbar(x, c, yerr=c_err, fmt='o-', capsize=4, elinewidth=1.2, linewidth=2)
    axes[2].set_xlabel(xlabel); axes[2].set_ylabel('Cross-Consistency')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout(); fig.savefig(base_path.with_suffix('.png'), bbox_inches='tight'); fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight'); plt.close(fig)

def hyperparam_error_bars():
    # Lambda
    lam = RESULTS_CSV / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv'
    if lam.exists():
        df = pd.read_csv(lam, comment='#').sort_values('lambda_repulsion')
        x = df['lambda_repulsion'].values
        _save_errs(OUT_H_ERR/'lambda_fidelity', x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Lambda', 'Fidelity')
        _save_errs(OUT_H_ERR/'lambda_diversity', x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Lambda', 'Diversity')
        _save_errs(OUT_H_ERR/'lambda_consistency', x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Lambda', 'Cross-Consistency')
        _save_triptych(OUT_H_ERR/'lambda_1x3', x,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'Lambda')
    # CFG
    cfg = RESULTS_CSV / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv'
    if cfg.exists():
        df = pd.read_csv(cfg, comment='#').sort_values('guidance_scale')
        x = df['guidance_scale'].values
        _save_errs(OUT_H_ERR/'cfg_fidelity', x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'CFG', 'Fidelity')
        _save_errs(OUT_H_ERR/'cfg_diversity', x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'CFG', 'Diversity')
        _save_errs(OUT_H_ERR/'cfg_consistency', x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'CFG', 'Cross-Consistency')
        _save_triptych(OUT_H_ERR/'cfg_1x3', x,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'CFG')
    # RBF Beta
    beta = RESULTS_CSV / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv'
    if beta.exists():
        df = pd.read_csv(beta, comment='#').sort_values('rbf_beta')
        x = df['rbf_beta'].values
        _save_errs(OUT_H_ERR/'rbf_beta_fidelity', x, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'RBF Beta', 'Fidelity')
        _save_errs(OUT_H_ERR/'rbf_beta_diversity', x, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'RBF Beta', 'Diversity')
        _save_errs(OUT_H_ERR/'rbf_beta_consistency', x, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'RBF Beta', 'Cross-Consistency')
        _save_triptych(OUT_H_ERR/'rbf_beta_1x3', x,
                       df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values,
                       df['diversity_mean_mean'].values, df['diversity_mean_std'].values,
                       df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values,
                       'RBF Beta')

def repulsion_error_bars():
    # Method comparison
    mfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'
    if mfile.exists():
        df = pd.read_csv(mfile, comment='#')
        keys = df['method'].astype(str).values
        _save_errs(OUT_R_ERR/'methods_fidelity', keys, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Method', 'Fidelity')
        _save_errs(OUT_R_ERR/'methods_diversity', keys, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Method', 'Diversity')
        _save_errs(OUT_R_ERR/'methods_consistency', keys, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Method', 'Cross-Consistency')
    # Kernel comparison
    kfile = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv'
    if kfile.exists():
        df = pd.read_csv(kfile, comment='#')
        keys = df['kernel'].astype(str).values
        _save_errs(OUT_R_ERR/'kernels_fidelity', keys, df['fidelity_mean_mean'].values, df['fidelity_mean_std'].values, 'Kernel', 'Fidelity')
        _save_errs(OUT_R_ERR/'kernels_diversity', keys, df['diversity_mean_mean'].values, df['diversity_mean_std'].values, 'Kernel', 'Diversity')
        _save_errs(OUT_R_ERR/'kernels_consistency', keys, df['cross_consistency_mean_mean'].values, df['cross_consistency_mean_std'].values, 'Kernel', 'Cross-Consistency')

def main():
    hyperparam_error_bars()
    repulsion_error_bars()

if __name__ == '__main__':
    main()

 
