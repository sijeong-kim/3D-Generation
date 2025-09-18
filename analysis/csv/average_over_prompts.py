#!/usr/bin/env python3
"""
Average ablation results over prompts.

This script reads ablation CSV files and creates averaged versions where
metrics are averaged across all prompts for each ablation parameter value.
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def get_descriptive_experiment_name(exp_name: str) -> str:
    """
    Convert technical experiment names to more descriptive names.
    
    Args:
        exp_name: Technical experiment name
        
    Returns:
        Descriptive experiment name
    """
    name_mapping = {
        "exp0_baseline": "Baseline_Experiment",
        "exp0_baseline_indep": "Baseline_Independent_Experiment", 
        "exp1_repulsion_kernel": "Repulsion_Method_and_Kernel_Analysis",
        "exp2_lambda_coarse": "Lambda_Repulsion_Coarse_Search",
        "exp3_lambda_fine": "Lambda_Repulsion_Fine_Search",
        "exp4_guidance_scale": "Guidance_Scale_Analysis",
        "exp5_rbf_beta": "RBF_Beta_Parameter_Analysis",
        "exp_feature_layer": "Feature_Layer_Analysis",
        "exp_num_particles": "Number_of_Particles_Analysis",
        "exp_num_pts": "Number_of_Points_Analysis",
        "exp_opacity_lr": "Opacity_Learning_Rate_Analysis",
        "exp_gaussian_reproduce": "Gaussian_Reproduction_Experiment"
    }
    return name_mapping.get(exp_name, exp_name)


def average_over_prompts(input_file: Path, output_file: Path, exp_dir: Path = None) -> None:
    """
    Average ablation results over prompts.
    
    Args:
        input_file: Path to input ablation CSV file
        output_file: Path to output averaged CSV file
    """
    print(f"Processing: {input_file.name}")
    
    # Read the ablation CSV
    df = pd.read_csv(input_file)
    
    # Check if prompt column exists
    if 'prompt' not in df.columns:
        print(f"  No 'prompt' column found, skipping {input_file.name}")
        return
    
    # Special handling for exp1_repulsion_kernel - group by both method and kernel
    if 'exp1_repulsion_kernel' in input_file.name and 'method' in df.columns and 'kernel' in df.columns:
        print(f"  Special handling: averaging over method and kernel parameters")
        print(f"  Prompts: {df['prompt'].nunique()}")
        
        # Group by method and kernel, average over prompts
        grouped = df.groupby(['method', 'kernel']).agg({
            'n_runs': 'sum',  # Sum total runs across all prompts
            **{col: 'mean' for col in df.columns if col.startswith(('fidelity_', 'diversity_', 'cross_consistency_'))}
        }).reset_index()
        
        # Add experiment column
        grouped['experiment'] = df['experiment'].iloc[0]
        
        # Reorder columns
        cols = ['experiment', 'method', 'kernel', 'n_runs'] + [col for col in grouped.columns if col not in ['experiment', 'method', 'kernel', 'n_runs']]
        grouped = grouped[cols]
        
    else:
        # Get the ablation parameter column (not experiment, prompt, or n_runs)
        ablation_param = None
        for col in df.columns:
            if col not in ['experiment', 'prompt', 'n_runs'] and not col.startswith(('fidelity_', 'diversity_', 'cross_consistency_')):
                ablation_param = col
                break
        
        if ablation_param is None:
            print(f"  No ablation parameter found, skipping {input_file.name}")
            return
        
        print(f"  Ablation parameter: {ablation_param}")
        print(f"  Prompts: {df['prompt'].nunique()}")
        
        # Group by ablation parameter and average over prompts
        grouped = df.groupby([ablation_param]).agg({
            'n_runs': 'sum',  # Sum total runs across all prompts
            **{col: 'mean' for col in df.columns if col.startswith(('fidelity_', 'diversity_', 'cross_consistency_'))}
        }).reset_index()
        
        # Add experiment column
        grouped['experiment'] = df['experiment'].iloc[0]
        
        # Reorder columns to match original structure
        cols = ['experiment', ablation_param, 'n_runs'] + [col for col in grouped.columns if col.startswith(('fidelity_', 'diversity_', 'cross_consistency_'))]
        grouped = grouped[cols]
    
    # Add averaging methodology information to the DataFrame
    grouped.attrs['averaging_methodology'] = {
        'description': 'Data averaged over prompts',
        'total_prompts': int(df['prompt'].nunique()),
        'prompts_included': sorted(df['prompt'].unique().tolist()),
        'averaging_method': 'Mean across all prompts for each parameter combination',
        'original_data_points': len(df),
        'averaged_data_points': len(grouped)
    }
    
    # Create a header comment explaining the averaging methodology
    header_comment = f"""# AVERAGED DATA - METHODOLOGY
# This file contains data averaged over prompts for easier comparison
# 
# Averaging Details:
# - Method: Mean across all prompts for each parameter combination
# - Total prompts included: {df['prompt'].nunique()}
# - Prompts: {', '.join(sorted(df['prompt'].unique()))}
# - Original data points: {len(df)}
# - Averaged data points: {len(grouped)}
# 
# Each row represents the average performance across all prompts for a specific parameter value
# Standard deviations and other statistics are also averaged accordingly
#
# Generated by: average_over_prompts.py
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
"""
    
    # Save averaged results with header comment
    with open(output_file, 'w') as f:
        f.write(header_comment)
        grouped.to_csv(f, index=False)
    
    print(f"  Saved: {output_file.name}")
    print(f"  Rows: {len(grouped)} (averaged over {df['prompt'].nunique()} prompts)")
    print(f"  Averaging: Mean across {df['prompt'].nunique()} prompts for each parameter combination")
    
    # Also save in experiment-specific directory if provided
    if exp_dir is not None:
        exp_dir.mkdir(exist_ok=True)
        
        # Create descriptive filename for experiment directory
        exp_name = input_file.stem.replace('_ablation', '').replace('_method_ablation', '').replace('_kernel_ablation', '')
        descriptive_name = get_descriptive_experiment_name(exp_name)
        
        # Determine the type of analysis based on filename
        if 'method_ablation' in input_file.stem:
            descriptive_filename = f"{descriptive_name}_Repulsion_Method_Comparison_Averaged.csv"
        elif 'kernel_ablation' in input_file.stem:
            descriptive_filename = f"{descriptive_name}_Kernel_Type_Comparison_Averaged.csv"
        elif 'ablation' in input_file.stem:
            descriptive_filename = f"{descriptive_name}_Parameter_Analysis_Averaged.csv"
        else:
            descriptive_filename = f"{descriptive_name}_Averaged_Results.csv"
        
        exp_output_path = exp_dir / descriptive_filename
        
        # Save with methodology header in experiment directory too
        with open(exp_output_path, 'w') as f:
            f.write(header_comment)
            grouped.to_csv(f, index=False)
        
        print(f"  Also saved to experiment directory: {exp_output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Average ablation results over prompts")
    parser.add_argument("--results-dir", default="./results/csv", help="Directory containing ablation CSV files")
    parser.add_argument("--output-suffix", default="_averaged", help="Suffix for output files")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1
    
    print("Averaging Ablation Results Over Prompts")
    print("=" * 50)
    
    # Find all ablation CSV files
    ablation_files = list(results_dir.glob("*_ablation.csv"))
    
    if not ablation_files:
        print("No ablation CSV files found!")
        return 1
    
    print(f"Found {len(ablation_files)} ablation files")
    print()
    
    processed_count = 0
    
    for ablation_file in ablation_files:
        # Create output filename
        output_file = results_dir / f"{ablation_file.stem}{args.output_suffix}.csv"
        
        # Determine experiment directory
        exp_name = ablation_file.stem.replace('_ablation', '').replace('_method_ablation', '').replace('_kernel_ablation', '')
        exp_dir = results_dir / exp_name
        
        try:
            average_over_prompts(ablation_file, output_file, exp_dir)
            processed_count += 1
            print()
        except Exception as e:
            print(f"  Error processing {ablation_file.name}: {e}")
            print()
    
    print("=" * 50)
    print(f"Processed {processed_count}/{len(ablation_files)} files")
    print(f"Results saved to: {results_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
