#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import os

RESULTS = Path('/Users/sj/3D-Generation/results/csv')

def normalize(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax - xmin == 0:
        return np.ones_like(x)
    return (x - xmin) / (x - xmin).ptp()

def pick_utopia(row_fid, row_div, fid_vals, div_vals, cons_vals=None, cons_constraint=True, epsilon=0.01, w_fid=0.5, w_div=0.5):
    fid_n = normalize(fid_vals)
    div_n = normalize(div_vals)
    if cons_vals is not None and cons_constraint:
        cons_max = np.nanmax(cons_vals)
        # epsilon-constraint: allow points within epsilon of the maximum consistency
        mask = cons_vals >= (cons_max - epsilon)
    else:
        mask = np.ones_like(fid_vals, dtype=bool)
    # weighted distance to (1,1)
    dist = np.sqrt((w_fid * (1 - fid_n))**2 + (w_div * (1 - div_n))**2)
    dist[~mask] = np.inf
    idx = int(np.nanargmin(dist))
    return idx, dist

def main():
    # Read weights/epsilon from environment (defaults: equal weights, epsilon=0.01)
    try:
        w_fid = float(os.environ.get('WEIGHT_FID', '0.5'))
        w_div = float(os.environ.get('WEIGHT_DIV', '0.5'))
        eps = float(os.environ.get('EPSILON_CONS', '0.01'))
    except Exception:
        w_fid, w_div, eps = 0.5, 0.5, 0.01
    # Normalize weights if they don't sum to 1
    s = w_fid + w_div
    if s <= 0:
        w_fid, w_div = 0.5, 0.5
    else:
        w_fid, w_div = w_fid / s, w_div / s
    out = {}

    # Repulsion method
    df = pd.read_csv(RESULTS / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Repulsion_Method_Comparison.csv', comment='#')
    idx, _ = pick_utopia(df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['cross_consistency_mean_mean'].values, epsilon=eps, w_fid=w_fid, w_div=w_div)
    out['repulsion_method'] = df.iloc[idx]['method']

    # Kernel type
    df = pd.read_csv(RESULTS / 'exp1_repulsion_kernel' / 'Repulsion_Method_and_Kernel_Analysis_Kernel_Type_Comparison.csv', comment='#')
    idx, _ = pick_utopia(df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['cross_consistency_mean_mean'].values, epsilon=eps, w_fid=w_fid, w_div=w_div)
    out['kernel_type'] = df.iloc[idx]['kernel']

    # Lambda fine
    df = pd.read_csv(RESULTS / 'exp3_lambda_fine' / 'Lambda_Repulsion_Fine_Search_Parameter_Analysis_Averaged.csv', comment='#')
    idx, dist = pick_utopia(df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                            df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                            df['cross_consistency_mean_mean'].values, epsilon=eps, w_fid=w_fid, w_div=w_div)
    out['lambda_repulsion'] = float(df.iloc[idx]['lambda_repulsion'])

    # Guidance scale
    df = pd.read_csv(RESULTS / 'exp4_guidance_scale' / 'Guidance_Scale_Analysis_Parameter_Analysis_Averaged.csv', comment='#')
    idx, _ = pick_utopia(df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['cross_consistency_mean_mean'].values, epsilon=eps, w_fid=w_fid, w_div=w_div)
    out['guidance_scale'] = int(df.iloc[idx]['guidance_scale'])

    # RBF beta
    df = pd.read_csv(RESULTS / 'exp5_rbf_beta' / 'RBF_Beta_Parameter_Analysis_Parameter_Analysis_Averaged.csv', comment='#')
    idx, _ = pick_utopia(df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['fidelity_mean_mean'].values, df['diversity_mean_mean'].values,
                         df['cross_consistency_mean_mean'].values, epsilon=eps, w_fid=w_fid, w_div=w_div)
    out['rbf_beta'] = float(df.iloc[idx]['rbf_beta'])

    for k, v in out.items():
        print(f"{k}={v}")

if __name__ == '__main__':
    main()


