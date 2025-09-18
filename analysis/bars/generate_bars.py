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
OUT_DIR = ROOT / 'results' / 'bars'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def bar_with_error(ax, categories, means, stds, ylabel):
    colors = sns.color_palette(n_colors=len(categories))
    ax.bar(categories, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

def main():
    # Kernel types
    kernel_file = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv'
    if kernel_file.exists():
        df = pd.read_csv(kernel_file, comment='#')
        fig, axes = plt.subplots(1,3, figsize=(12,3.5))
        bar_with_error(axes[0], df['kernel'], df['fidelity_mean_mean'], df['fidelity_mean_std'], 'Fidelity')
        bar_with_error(axes[1], df['kernel'], df['diversity_mean_mean'], df['diversity_mean_std'], 'Diversity')
        bar_with_error(axes[2], df['kernel'], df['cross_consistency_mean_mean'], df['cross_consistency_mean_std'], 'Consistency')
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'kernel_types_bars.png', bbox_inches='tight')
        fig.savefig(OUT_DIR / 'kernel_types_bars.pdf', bbox_inches='tight')
        plt.close(fig)

    # Repulsion methods
    method_file = RESULTS_CSV / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv'
    if method_file.exists():
        df = pd.read_csv(method_file, comment='#')
        fig, axes = plt.subplots(1,3, figsize=(12,3.5))
        bar_with_error(axes[0], df['method'], df['fidelity_mean_mean'], df['fidelity_mean_std'], 'Fidelity')
        bar_with_error(axes[1], df['method'], df['diversity_mean_mean'], df['diversity_mean_std'], 'Diversity')
        bar_with_error(axes[2], df['method'], df['cross_consistency_mean_mean'], df['cross_consistency_mean_std'], 'Consistency')
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'repulsion_methods_bars.png', bbox_inches='tight')
        fig.savefig(OUT_DIR / 'repulsion_methods_bars.pdf', bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()


