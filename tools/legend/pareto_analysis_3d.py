#!/usr/bin/env python3
"""
3D Pareto Analysis for Lambda Repulsion Experiments
Generate Pareto subplots for each combination of kernel_type, prompt, and repulsion_type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import re
from itertools import product

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

def load_baseline_data(baseline_dir):
    """Load baseline data for each prompt."""
    baseline_data = {}
    for item in baseline_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
            metrics_path = item / "metrics" / "quantitative_metrics.csv"
            if metrics_path.exists():
                baseline_data[item.name] = pd.read_csv(metrics_path)
                print(f"Loaded baseline data for {item.name}")
    return baseline_data

def calculate_metric_differences(exp_df, baseline_df, metrics):
    """Calculate differences from baseline for each metric."""
    differences = {}
    
    for metric_col, metric_name in metrics.items():
        if metric_col in exp_df.columns and metric_col in baseline_df.columns:
            # Get final values
            exp_final = exp_df[metric_col].iloc[-1] if len(exp_df) > 0 else np.nan
            baseline_final = baseline_df[metric_col].iloc[-1] if len(baseline_df) > 0 else np.nan
            
            if not np.isnan(exp_final) and not np.isnan(baseline_final):
                # Calculate percentage difference
                pct_diff = ((exp_final - baseline_final) / baseline_final) * 100
                differences[metric_name] = pct_diff
            else:
                differences[metric_name] = np.nan
        else:
            differences[metric_name] = np.nan
    
    return differences

def find_pareto_optimal_3d(points_data, x_col, y_col, z_col, minimize_x=True, minimize_y=True, minimize_z=True):
    """Find Pareto optimal points in 3D space."""
    pareto_indices = []
    
    for i, row in points_data.iterrows():
        is_pareto = True
        for j, other_row in points_data.iterrows():
            if i != j:
                # Check if other point dominates this one
                x_better = (other_row[x_col] <= row[x_col] if minimize_x else other_row[x_col] >= row[x_col])
                y_better = (other_row[y_col] <= row[y_col] if minimize_y else other_row[y_col] >= row[y_col])
                z_better = (other_row[z_col] <= row[z_col] if minimize_z else other_row[z_col] >= row[z_col])
                
                x_strictly_better = (other_row[x_col] < row[x_col] if minimize_x else other_row[x_col] > row[x_col])
                y_strictly_better = (other_row[y_col] < row[y_col] if minimize_y else other_row[y_col] > row[y_col])
                z_strictly_better = (other_row[z_col] < row[z_col] if minimize_z else other_row[z_col] > row[z_col])
                
                if x_better and y_better and z_better and (x_strictly_better or y_strictly_better or z_strictly_better):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices

def find_best_pareto_optimal_3d(data, max_points=3):
    """Find the best Pareto optimal points based on overall performance."""
    # Calculate a composite score: diversity improvement - fidelity degradation - consistency degradation
    data['composite_score'] = (
        data['Diversity Improvement (%)'] - 
        np.abs(data['Fidelity Change (%)']) - 
        np.abs(data['Cross-View Consistency Change (%)'])
    )
    
    # Find Pareto optimal points
    pareto_indices = find_pareto_optimal_3d(data, 'Fidelity Change (%)', 'Diversity Improvement (%)', 'Cross-View Consistency Change (%)', 
                                           minimize_x=True, minimize_y=False, minimize_z=True)
    
    if len(pareto_indices) <= max_points:
        return pareto_indices
    
    # If too many Pareto optimal points, select the best ones based on composite score
    pareto_data = data.loc[pareto_indices]
    best_indices = pareto_data.nlargest(max_points, 'composite_score').index.tolist()
    
    return best_indices

def create_3d_pareto_subplot(ax, data, x_col, y_col, z_col, title, lambda_values):
    """Create a 3D Pareto subplot."""
    if len(data) == 0:
        ax.text(0.5, 0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Color map for lambda values
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))
    lambda_color_map = dict(zip(lambda_values, colors))
    
    # Plot all points
    for lambda_val in lambda_values:
        lambda_data = data[data['Lambda'] == lambda_val]
        if len(lambda_data) > 0:
            color = lambda_color_map[lambda_val]
            ax.scatter(lambda_data[x_col], lambda_data[y_col], lambda_data[z_col],
                      c=[color], s=50, alpha=0.7, label=f'λ={lambda_val}')
            
            # Add step labels for each point (no lambda labels needed)
            for _, row in lambda_data.iterrows():
                label = f"{row['Step']}"
                ax.text(row[x_col], row[y_col], row[z_col], label, 
                       fontsize=6, ha='center', va='bottom')
    
    # Find and highlight best Pareto optimal points (max 3)
    pareto_indices = find_best_pareto_optimal_3d(data, max_points=3)
    
    if pareto_indices:
        pareto_data = data.loc[pareto_indices]
        ax.scatter(pareto_data[x_col], pareto_data[y_col], pareto_data[z_col],
                  c='red', s=100, alpha=0.9, marker='*', label='Pareto Optimal')
        
        # Add labels for Pareto optimal points
        for _, row in pareto_data.iterrows():
            label = f"★{row['Step']}"
            ax.text(row[x_col], row[y_col], row[z_col], label, 
                   fontsize=8, ha='center', va='bottom', weight='bold', color='red')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

def main():
    # Configuration
    baseline_exp = "exp0_baseline"
    experiment_exp = "exp1_lambda_coarse"
    base_dir = Path("exp")
    output_dir = Path("analysis/pareto_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Define metrics to analyze
    metrics = {
        "inter_particle_diversity_mean": "Diversity Improvement (%)",
        "fidelity_mean": "Fidelity Change (%)",
        "cross_view_consistency_mean": "Cross-View Consistency Change (%)"
    }
    
    # Load baseline data
    baseline_dir = base_dir / baseline_exp
    baseline_data = load_baseline_data(baseline_dir)
    
    # Load experiment data
    experiment_dir = base_dir / experiment_exp
    experiment_configs = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
            experiment_configs.append(item.name)
    
    print(f"Found {len(experiment_configs)} experiment configurations")
    
    # Parse configurations
    config_data = {}
    for config_name in experiment_configs:
        params = parse_config_name(config_name)
        if params and params['seed'] == '42':
            key = (params['kernel_type'], params['prompt'], params['repulsion_type'])
            if key not in config_data:
                config_data[key] = []
            
            # Load experiment metrics
            metrics_path = experiment_dir / config_name / "metrics" / "quantitative_metrics.csv"
            if metrics_path.exists():
                exp_df = pd.read_csv(metrics_path)
                
                # Find corresponding baseline
                baseline_key = f"prompt={params['prompt']}_seed=42"
                if baseline_key in baseline_data:
                    baseline_df = baseline_data[baseline_key]
                    
                    # Calculate differences for each step
                    for step_idx in range(len(exp_df)):
                        step_data = exp_df.iloc[step_idx:step_idx+1]
                        differences = calculate_metric_differences(step_data, baseline_df, metrics)
                        
                        if not all(np.isnan(v) for v in differences.values()):
                            row_data = {
                                'Kernel': params['kernel_type'].upper(),
                                'Prompt': params['prompt'].title(),
                                'Repulsion_Type': params['repulsion_type'].upper(),
                                'Lambda': float(params['lambda_repulsion']),
                                'Step': step_idx,
                                **differences
                            }
                            config_data[key].append(row_data)
    
    # Create subplots for each combination
    combinations = list(config_data.keys())
    n_combinations = len(combinations)
    
    # Calculate subplot layout
    n_cols = 4
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 5 * n_rows))
    
    for idx, (kernel_type, prompt, repulsion_type) in enumerate(combinations):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        # Get data for this combination
        data = pd.DataFrame(config_data[(kernel_type, prompt, repulsion_type)])
        
        if len(data) > 0:
            # Get unique lambda values for coloring
            lambda_values = sorted(data['Lambda'].unique())
            
            # Create subplot
            title = f"{kernel_type} + {prompt} + {repulsion_type}"
            create_3d_pareto_subplot(
                ax, data,
                'Fidelity Change (%)', 'Diversity Improvement (%)', 'Cross-View Consistency Change (%)',
                title, lambda_values
            )
        else:
            ax.text(0.5, 0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{kernel_type} + {prompt} + {repulsion_type}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_3d_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics
    print("\n=== PARETO ANALYSIS SUMMARY ===")
    for (kernel_type, prompt, repulsion_type), data in config_data.items():
        if data:
            df = pd.DataFrame(data)
            pareto_indices = find_best_pareto_optimal_3d(df, max_points=3)
            
            print(f"\n{kernel_type} + {prompt} + {repulsion_type}:")
            print(f"  Total configurations: {len(df)}")
            print(f"  Best Pareto optimal configurations: {len(pareto_indices)}")
            
            if pareto_indices:
                pareto_df = df.loc[pareto_indices]
                print("  Best Pareto optimal configurations:")
                for _, row in pareto_df.iterrows():
                    print(f"    - Step {row['Step']}, λ={row['Lambda']}: "
                          f"Diversity={row['Diversity Improvement (%)']:.2f}%, "
                          f"Fidelity={row['Fidelity Change (%)']:.2f}%, "
                          f"Consistency={row['Cross-View Consistency Change (%)']:.2f}%")
    
    # Save detailed data
    all_data = []
    for data_list in config_data.values():
        all_data.extend(data_list)
    
    if all_data:
        all_df = pd.DataFrame(all_data)
        all_df.to_csv(output_dir / 'pareto_analysis_data.csv', index=False)
        print(f"\nDetailed data saved to: {output_dir / 'pareto_analysis_data.csv'}")

if __name__ == "__main__":
    main()
