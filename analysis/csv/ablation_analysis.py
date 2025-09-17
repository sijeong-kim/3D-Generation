#!/usr/bin/env python3
"""
Script to perform ablation analysis across seeds and parameters.

This script reads consolidated experiment CSV files and generates
ablation analysis results showing mean and std across seeds for
different parameter values.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings


def load_consolidated_data(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all consolidated experiment CSV files.
    
    Args:
        results_dir: Directory containing consolidated CSV files
        
    Returns:
        Dictionary mapping experiment names to DataFrames
    """
    consolidated_files = list(results_dir.glob("*_consolidated.csv"))
    experiments = {}
    
    for file_path in consolidated_files:
        exp_name = file_path.stem.replace("_consolidated", "")
        try:
            df = pd.read_csv(file_path)
            experiments[exp_name] = df
            print(f"Loaded {exp_name}: {len(df)} runs")
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
    
    return experiments


def perform_ablation_analysis(experiments: Dict[str, pd.DataFrame], results_dir: Path) -> None:
    """
    Perform ablation analysis for each experiment.
    
    Args:
        experiments: Dictionary of experiment DataFrames
        results_dir: Directory to save results
    """
    for exp_name, df in experiments.items():
        print(f"\nPerforming ablation analysis for {exp_name}")
        
        # Identify the main ablation parameter based on experiment type
        ablation_param = identify_ablation_parameter(exp_name, df)
        
        if ablation_param is None:
            print(f"  No clear ablation parameter found for {exp_name}")
            continue
        
        print(f"  Ablation parameter: {ablation_param}")
        
        # Group by ablation parameter and prompt (if available)
        group_cols = [ablation_param]
        if 'prompt' in df.columns:
            group_cols.append('prompt')
        elif 'target' in df.columns:
            group_cols.append('target')
        
        ablation_results = []
        
        for group_key, group_df in df.groupby(group_cols):
            if isinstance(group_key, tuple):
                param_value, prompt_or_target = group_key
            else:
                param_value = group_key
                prompt_or_target = None
            
            # Calculate statistics across seeds
            result = {
                'experiment': exp_name,
                ablation_param: param_value,
                'n_runs': len(group_df)
            }
            
            if prompt_or_target is not None:
                # Use 'prompt' if available, otherwise 'target'
                if 'prompt' in df.columns:
                    result['prompt'] = prompt_or_target
                else:
                    result['target'] = prompt_or_target
            
            # Calculate mean and std for each metric across seeds
            for metric in ['fidelity', 'diversity', 'cross_consistency']:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                
                if mean_col in group_df.columns:
                    # Mean of means across seeds
                    result[f'{metric}_mean_mean'] = group_df[mean_col].mean()
                    # Std of means across seeds (simple std, not pooled)
                    result[f'{metric}_mean_std'] = group_df[mean_col].std()
                    
                    # Also calculate min/max for additional insights
                    result[f'{metric}_mean_min'] = group_df[mean_col].min()
                    result[f'{metric}_mean_max'] = group_df[mean_col].max()
                    
                    # Calculate coefficient of variation
                    if result[f'{metric}_mean_mean'] != 0:
                        result[f'{metric}_mean_cv'] = result[f'{metric}_mean_std'] / result[f'{metric}_mean_mean']
                    else:
                        result[f'{metric}_mean_cv'] = np.nan
                
                if std_col in group_df.columns:
                    # Mean of stds across seeds
                    result[f'{metric}_std_mean'] = group_df[std_col].mean()
                    # Std of stds across seeds
                    result[f'{metric}_std_std'] = group_df[std_col].std()
            
            ablation_results.append(result)
        
        if ablation_results:
            ablation_df = pd.DataFrame(ablation_results)
            
            # Sort by ablation parameter
            sort_cols = [ablation_param]
            if 'prompt' in ablation_df.columns:
                sort_cols.append('prompt')
            elif 'target' in ablation_df.columns:
                sort_cols.append('target')
            ablation_df = ablation_df.sort_values(sort_cols)
            
            # Save ablation results
            ablation_output_path = results_dir / f"{exp_name}_ablation.csv"
            ablation_df.to_csv(ablation_output_path, index=False)
            print(f"  Saved ablation analysis to {ablation_output_path}")
            
            # Print summary
            print(f"  Analyzed {len(ablation_df)} parameter combinations")
            if 'fidelity_mean_mean' in ablation_df.columns:
                best_fidelity_idx = ablation_df['fidelity_mean_mean'].idxmax()
                best_fidelity = ablation_df.loc[best_fidelity_idx]
                print(f"  Best fidelity: {best_fidelity[ablation_param]} = {best_fidelity['fidelity_mean_mean']:.4f} ± {best_fidelity['fidelity_mean_std']:.4f}")


def identify_ablation_parameter(exp_name: str, df: pd.DataFrame) -> Optional[str]:
    """
    Identify the main ablation parameter for an experiment.
    
    Args:
        exp_name: Name of the experiment
        df: DataFrame with experiment data
        
    Returns:
        Name of the ablation parameter column, or None if not found
    """
    # Define expected ablation parameters for each experiment type
    ablation_params = {
        "exp0_baseline": "repulsion_type",
        "exp1_repulsion_kernel": "kernel",  # or "method"
        "exp2_lambda_coarse": "lambda_repulsion",
        "exp3_lambda_fine": "lambda_repulsion",
        "exp4_guidance_scale": "guidance_scale",
        "exp5_rbf_beta": "rbf_beta",
        "exp_feature_layer": "feature_layer",
        "exp_num_particles": "num_particles",
        "exp_num_pts": "num_pts",
        "exp_opacity_lr": "opacity_lr"
    }
    
    # Check if the expected parameter exists
    expected_param = ablation_params.get(exp_name)
    if expected_param and expected_param in df.columns:
        return expected_param
    
    # Fallback: look for parameters with multiple unique values
    param_candidates = []
    for col in df.columns:
        if col not in ['experiment', 'run_id', 'last_step', 'n_samples', 'prompt', 'seed', 'target']:
            if not col.startswith('cfg_') and not col.endswith(('_mean', '_std', '_min', '_max', '_cv')):
                unique_vals = df[col].nunique()
                if unique_vals > 1:  # More than one unique value
                    param_candidates.append((col, unique_vals))
    
    if param_candidates:
        # Return the parameter with the most unique values
        param_candidates.sort(key=lambda x: x[1], reverse=True)
        return param_candidates[0][0]
    
    return None


def create_summary_table(experiments: Dict[str, pd.DataFrame], results_dir: Path) -> None:
    """
    Create a summary table with key results from all experiments.
    
    Args:
        experiments: Dictionary of experiment DataFrames
        results_dir: Directory to save results
    """
    summary_data = []
    
    for exp_name, df in experiments.items():
        # Calculate overall statistics for this experiment
        summary = {
            'experiment': exp_name,
            'n_runs': len(df),
            'n_prompts': df['prompt'].nunique() if 'prompt' in df.columns else (df['target'].nunique() if 'target' in df.columns else 1),
            'n_seeds': df['seed'].nunique() if 'seed' in df.columns else 1
        }
        
        # Add metric statistics
        for metric in ['fidelity', 'diversity', 'cross_consistency']:
            mean_col = f'{metric}_mean'
            if mean_col in df.columns:
                summary[f'{metric}_overall_mean'] = df[mean_col].mean()
                summary[f'{metric}_overall_std'] = df[mean_col].std()
                summary[f'{metric}_best'] = df[mean_col].max()
                summary[f'{metric}_worst'] = df[mean_col].min()
        
        summary_data.append(summary)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('experiment')
        
        # Save summary table
        summary_output_path = results_dir / "experiment_summary.csv"
        summary_df.to_csv(summary_output_path, index=False)
        print(f"\nSaved experiment summary to {summary_output_path}")
        
        # Print summary
        print("\nExperiment Summary:")
        print("=" * 80)
        for _, row in summary_df.iterrows():
            prompt_col = 'n_prompts' if 'n_prompts' in row else 'n_targets'
            print(f"{row['experiment']:20} | Runs: {row['n_runs']:3} | Prompts: {row[prompt_col]:2} | Seeds: {row['n_seeds']:2}")
            if 'fidelity_overall_mean' in row:
                print(f"{'':20} | Fidelity: {row['fidelity_overall_mean']:.4f} ± {row['fidelity_overall_std']:.4f}")
            if 'diversity_overall_mean' in row:
                print(f"{'':20} | Diversity: {row['diversity_overall_mean']:.4f} ± {row['diversity_overall_std']:.4f}")
            print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Perform ablation analysis on consolidated experiment results")
    parser.add_argument("--results-dir", default="./results", help="Directory containing consolidated CSV files")
    parser.add_argument("--exp-root", default="./exp", help="Root directory containing experiments (for reference)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return
    
    # Load consolidated data
    print("Loading consolidated experiment data...")
    experiments = load_consolidated_data(results_dir)
    
    if not experiments:
        print("No consolidated experiment files found")
        return
    
    # Perform ablation analysis
    print("\nPerforming ablation analysis...")
    perform_ablation_analysis(experiments, results_dir)
    
    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(experiments, results_dir)
    
    print("\nAblation analysis complete!")


if __name__ == "__main__":
    main()
