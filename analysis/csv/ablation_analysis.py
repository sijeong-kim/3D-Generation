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
from typing import Dict, List, Optional, Tuple, Any, Union
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


def average_over_prompts_for_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average data over prompts for ablation analysis.
    
    Args:
        df: DataFrame with raw experimental data
        
    Returns:
        DataFrame averaged over prompts
    """
    # Group by method, kernel, and seed, then average over prompts
    group_cols = ['method', 'kernel', 'seed']
    if 'prompt' in df.columns:
        group_cols.append('prompt')
    
    # Calculate mean and std for each metric across prompts
    prompt_averaged = df.groupby(['method', 'kernel', 'seed']).agg({
        'fidelity_mean': 'mean',
        'fidelity_std': 'mean', 
        'diversity_mean': 'mean',
        'diversity_std': 'mean',
        'cross_consistency_mean': 'mean',
        'cross_consistency_std': 'mean'
    }).reset_index()
    
    return prompt_averaged


def perform_method_ablation_analysis(prompt_averaged_df: pd.DataFrame, exp_name: str) -> Optional[pd.DataFrame]:
    """
    Perform ablation analysis for repulsion methods, averaging over kernels.
    
    Args:
        prompt_averaged_df: DataFrame already averaged over prompts
        exp_name: Name of the experiment
        
    Returns:
        DataFrame with method ablation results
    """
    # Group by method and seed, then average over kernels
    method_results = []
    
    for method in prompt_averaged_df['method'].unique():
        method_data = prompt_averaged_df[prompt_averaged_df['method'] == method]
        
        # Calculate statistics across seeds and kernels
        result = {
            'experiment': exp_name,
            'method': method,
            'n_runs': len(method_data)
        }
        
        # Calculate statistics for each metric
        for metric in ['fidelity', 'diversity', 'cross_consistency']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in method_data.columns:
                result[f'{metric}_mean_mean'] = method_data[mean_col].mean()
                result[f'{metric}_mean_std'] = method_data[mean_col].std()
                result[f'{metric}_mean_min'] = method_data[mean_col].min()
                result[f'{metric}_mean_max'] = method_data[mean_col].max()
                
                if result[f'{metric}_mean_mean'] != 0:
                    result[f'{metric}_mean_cv'] = result[f'{metric}_mean_std'] / result[f'{metric}_mean_mean']
                else:
                    result[f'{metric}_mean_cv'] = np.nan
            
            if std_col in method_data.columns:
                result[f'{metric}_std_mean'] = method_data[std_col].mean()
                result[f'{metric}_std_std'] = method_data[std_col].std()
        
        method_results.append(result)
    
    if not method_results:
        return None
    
    return pd.DataFrame(method_results)


def perform_kernel_ablation_analysis(prompt_averaged_df: pd.DataFrame, exp_name: str) -> Optional[pd.DataFrame]:
    """
    Perform ablation analysis for kernel types, averaging over repulsion methods.
    
    Args:
        prompt_averaged_df: DataFrame already averaged over prompts
        exp_name: Name of the experiment
        
    Returns:
        DataFrame with kernel ablation results
    """
    # Group by kernel and seed, then average over methods
    kernel_results = []
    
    for kernel in prompt_averaged_df['kernel'].unique():
        kernel_data = prompt_averaged_df[prompt_averaged_df['kernel'] == kernel]
        
        # Calculate statistics across seeds and methods
        result = {
            'experiment': exp_name,
            'kernel': kernel,
            'n_runs': len(kernel_data)
        }
        
        # Calculate statistics for each metric
        for metric in ['fidelity', 'diversity', 'cross_consistency']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in kernel_data.columns:
                result[f'{metric}_mean_mean'] = kernel_data[mean_col].mean()
                result[f'{metric}_mean_std'] = kernel_data[mean_col].std()
                result[f'{metric}_mean_min'] = kernel_data[mean_col].min()
                result[f'{metric}_mean_max'] = kernel_data[mean_col].max()
                
                if result[f'{metric}_mean_mean'] != 0:
                    result[f'{metric}_mean_cv'] = result[f'{metric}_mean_std'] / result[f'{metric}_mean_mean']
                else:
                    result[f'{metric}_mean_cv'] = np.nan
            
            if std_col in kernel_data.columns:
                result[f'{metric}_std_mean'] = kernel_data[std_col].mean()
                result[f'{metric}_std_std'] = kernel_data[std_col].std()
        
        kernel_results.append(result)
    
    if not kernel_results:
        return None
    
    return pd.DataFrame(kernel_results)


def perform_combined_ablation_analysis(prompt_averaged_df: pd.DataFrame, exp_name: str) -> Optional[pd.DataFrame]:
    """
    Perform combined ablation analysis for method+kernel combinations.
    
    Args:
        prompt_averaged_df: DataFrame already averaged over prompts
        exp_name: Name of the experiment
        
    Returns:
        DataFrame with combined ablation results
    """
    # Group by method, kernel, and seed
    combined_results = []
    
    for (method, kernel), group_data in prompt_averaged_df.groupby(['method', 'kernel']):
        # Calculate statistics across seeds
        result = {
            'experiment': exp_name,
            'method': method,
            'kernel': kernel,
            'n_runs': len(group_data)
        }
        
        # Calculate statistics for each metric
        for metric in ['fidelity', 'diversity', 'cross_consistency']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in group_data.columns:
                result[f'{metric}_mean_mean'] = group_data[mean_col].mean()
                result[f'{metric}_mean_std'] = group_data[mean_col].std()
                result[f'{metric}_mean_min'] = group_data[mean_col].min()
                result[f'{metric}_mean_max'] = group_data[mean_col].max()
                
                if result[f'{metric}_mean_mean'] != 0:
                    result[f'{metric}_mean_cv'] = result[f'{metric}_mean_std'] / result[f'{metric}_mean_mean']
                else:
                    result[f'{metric}_mean_cv'] = np.nan
            
            if std_col in group_data.columns:
                result[f'{metric}_std_mean'] = group_data[std_col].mean()
                result[f'{metric}_std_std'] = group_data[std_col].std()
        
        combined_results.append(result)
    
    if not combined_results:
        return None
    
    return pd.DataFrame(combined_results)


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


def perform_ablation_analysis(experiments: Dict[str, pd.DataFrame], results_dir: Path) -> None:
    """
    Perform ablation analysis for each experiment.
    
    Args:
        experiments: Dictionary of experiment DataFrames
        results_dir: Directory to save results
    """
    for exp_name, df in experiments.items():
        print(f"\nPerforming ablation analysis for {exp_name}")
        
        # Special handling for exp1_repulsion_kernel - analyze method and kernel separately
        if exp_name == "exp1_repulsion_kernel":
            print(f"  Special handling: analyzing method and kernel separately")
            
            # Create experiment-specific directory
            exp_dir = results_dir / exp_name
            exp_dir.mkdir(exist_ok=True)
            
            # First, average over prompts to get prompt-averaged data
            print(f"  Step 1: Averaging over prompts first")
            prompt_averaged = average_over_prompts_for_ablation(df)
            
            # Analyze by method (RLSD vs SVGD) - average over kernels
            print(f"  Step 2: Analyzing repulsion methods (averaging over kernels)")
            method_results = perform_method_ablation_analysis(prompt_averaged, exp_name)
            if method_results is not None:
                descriptive_name = get_descriptive_experiment_name(exp_name)
                method_file = exp_dir / f"{descriptive_name}_Repulsion_Method_Comparison.csv"
                
                # Add methodology header for method comparison
                method_header = f"""# REPULSION METHOD COMPARISON - METHODOLOGY
# This file compares RLSD vs SVGD repulsion methods
# 
# Analysis Details:
# - Parameter analyzed: repulsion method (RLSD vs SVGD)
# - Method: First averaged over prompts, then averaged over kernels
# - Metrics calculated: Mean, standard deviation, min, max, coefficient of variation
# - Grouping: By repulsion method type
# - Total method combinations: {len(method_results)}
# 
# Each row represents the aggregated performance for a specific repulsion method
# Statistics are calculated across all seeds, prompts, and kernel types for that method
#
# Generated by: ablation_analysis.py
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
"""
                
                with open(method_file, 'w') as f:
                    f.write(method_header)
                    method_results.to_csv(f, index=False)
                
                print(f"  Method analysis saved to: {method_file}")
            
            # Analyze by kernel (COS vs RBF) - average over repulsion methods
            print(f"  Step 3: Analyzing kernel types (averaging over repulsion methods)")
            kernel_results = perform_kernel_ablation_analysis(prompt_averaged, exp_name)
            if kernel_results is not None:
                descriptive_name = get_descriptive_experiment_name(exp_name)
                kernel_file = exp_dir / f"{descriptive_name}_Kernel_Type_Comparison.csv"
                
                # Add methodology header for kernel comparison
                kernel_header = f"""# KERNEL TYPE COMPARISON - METHODOLOGY
# This file compares COS vs RBF kernel types
# 
# Analysis Details:
# - Parameter analyzed: kernel type (COS vs RBF)
# - Method: First averaged over prompts, then averaged over repulsion methods
# - Metrics calculated: Mean, standard deviation, min, max, coefficient of variation
# - Grouping: By kernel type
# - Total kernel combinations: {len(kernel_results)}
# 
# Each row represents the aggregated performance for a specific kernel type
# Statistics are calculated across all seeds, prompts, and repulsion methods for that kernel
#
# Generated by: ablation_analysis.py
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
"""
                
                with open(kernel_file, 'w') as f:
                    f.write(kernel_header)
                    kernel_results.to_csv(f, index=False)
                
                print(f"  Kernel analysis saved to: {kernel_file}")
            
            # Analyze by both method and kernel combined (keep all combinations)
            print(f"  Step 4: Complete analysis (all method+kernel combinations)")
            combined_results = perform_combined_ablation_analysis(prompt_averaged, exp_name)
            if combined_results is not None:
                descriptive_name = get_descriptive_experiment_name(exp_name)
                combined_file = exp_dir / f"{descriptive_name}_Complete_Analysis.csv"
                
                # Add methodology header for complete analysis
                combined_header = f"""# COMPLETE REPULSION AND KERNEL ANALYSIS - METHODOLOGY
# This file contains complete analysis of all method+kernel combinations
# 
# Analysis Details:
# - Parameters analyzed: repulsion method (RLSD/SVGD) + kernel type (COS/RBF)
# - Method: First averaged over prompts, then grouped by method+kernel combination
# - Metrics calculated: Mean, standard deviation, min, max, coefficient of variation
# - Grouping: By method and kernel combination
# - Total combinations: {len(combined_results)}
# 
# Each row represents the aggregated performance for a specific method+kernel combination
# Statistics are calculated across all seeds and prompts for that combination
#
# Generated by: ablation_analysis.py
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
"""
                
                with open(combined_file, 'w') as f:
                    f.write(combined_header)
                    combined_results.to_csv(f, index=False)
                
                print(f"  Combined analysis saved to: {combined_file}")
            
            # Also save in main results directory for compatibility
            if method_results is not None:
                main_method_file = results_dir / f"{exp_name}_method_ablation.csv"
                method_results.to_csv(main_method_file, index=False)
            if kernel_results is not None:
                main_kernel_file = results_dir / f"{exp_name}_kernel_ablation.csv"
                kernel_results.to_csv(main_kernel_file, index=False)
            if combined_results is not None:
                main_combined_file = results_dir / f"{exp_name}_ablation.csv"
                combined_results.to_csv(main_combined_file, index=False)
            
            continue
        
        # Standard ablation analysis for other experiments
        ablation_param = identify_ablation_parameter(exp_name, df)
        
        if ablation_param is None:
            print(f"  No clear ablation parameter found for {exp_name}")
            continue
        
        print(f"  Ablation parameter: {ablation_param}")
        
        # Create experiment-specific directory
        exp_dir = results_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
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
            
            # Save ablation results in experiment-specific directory with descriptive name
            descriptive_name = get_descriptive_experiment_name(exp_name)
            ablation_output_path = exp_dir / f"{descriptive_name}_Parameter_Analysis.csv"
            
            # Add methodology header to ablation analysis
            methodology_header = f"""# ABLATION ANALYSIS - METHODOLOGY
# This file contains ablation analysis results showing how different parameter values affect performance
# 
# Analysis Details:
# - Parameter analyzed: {ablation_param}
# - Method: Statistical analysis across multiple seeds and prompts
# - Metrics calculated: Mean, standard deviation, min, max, coefficient of variation
# - Grouping: By {ablation_param} parameter value
# - Total parameter combinations: {len(ablation_df)}
# 
# Each row represents the aggregated performance for a specific parameter value
# Statistics are calculated across all seeds and prompts for that parameter value
#
# Generated by: ablation_analysis.py
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
"""
            
            with open(ablation_output_path, 'w') as f:
                f.write(methodology_header)
                ablation_df.to_csv(f, index=False)
            
            print(f"  Saved ablation analysis to {ablation_output_path}")
            
            # Also save in main results directory for compatibility
            main_output_path = results_dir / f"{exp_name}_ablation.csv"
            ablation_df.to_csv(main_output_path, index=False)
            
            # Print summary
            print(f"  Analyzed {len(ablation_df)} parameter combinations")
            if 'fidelity_mean_mean' in ablation_df.columns:
                best_fidelity_idx = ablation_df['fidelity_mean_mean'].idxmax()
                best_fidelity = ablation_df.loc[best_fidelity_idx]
                print(f"  Best fidelity: {best_fidelity[ablation_param]} = {best_fidelity['fidelity_mean_mean']:.4f} ± {best_fidelity['fidelity_mean_std']:.4f}")


def perform_single_ablation_analysis(df: pd.DataFrame, group_cols: Union[str, List[str]], param_name: str, exp_name: str) -> Optional[pd.DataFrame]:
    """
    Perform ablation analysis for a single parameter or combination of parameters.
    
    Args:
        df: DataFrame with experiment data
        group_cols: Column(s) to group by
        param_name: Name for the parameter column in results
        exp_name: Name of the experiment
        
    Returns:
        DataFrame with ablation results or None if no valid results
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    # Add prompt/target to grouping if available
    if 'prompt' in df.columns:
        group_cols = group_cols + ['prompt']
    elif 'target' in df.columns:
        group_cols = group_cols + ['target']
    
    # Calculate statistics for each group
    ablation_results = []
    for group_key, group_df in df.groupby(group_cols):
        if len(group_cols) == 1:
            param_value = group_key
            prompt_or_target = None
        elif len(group_cols) == 2:
            if 'prompt' in df.columns or 'target' in df.columns:
                param_value, prompt_or_target = group_key
            else:
                param_value = group_key
                prompt_or_target = None
        else:  # 3 columns: param1, param2, prompt/target
            param1, param2, prompt_or_target = group_key
            param_value = f"{param1}_{param2}"
        
        # Create result dictionary
        result = {
            'experiment': exp_name,
            'n_runs': len(group_df)
        }
        
        # Add parameter values
        if len(group_cols) == 1:
            result[param_name] = param_value
        elif len(group_cols) == 2 and ('prompt' in df.columns or 'target' in df.columns):
            result[param_name] = param_value
        else:  # Multiple parameters
            for i, col in enumerate(group_cols):
                if col not in ['prompt', 'target']:
                    result[col] = group_key[i]
        
        # Add prompt/target
        if prompt_or_target is not None:
            if 'prompt' in df.columns:
                result['prompt'] = prompt_or_target
            else:
                result['target'] = prompt_or_target
        
        # Calculate statistics for each metric
        for metric in ['fidelity', 'diversity', 'cross_consistency']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in group_df.columns:
                result[f'{metric}_mean_mean'] = group_df[mean_col].mean()
                result[f'{metric}_mean_std'] = group_df[mean_col].std()
                result[f'{metric}_mean_min'] = group_df[mean_col].min()
                result[f'{metric}_mean_max'] = group_df[mean_col].max()
                
                if result[f'{metric}_mean_mean'] != 0:
                    result[f'{metric}_mean_cv'] = result[f'{metric}_mean_std'] / result[f'{metric}_mean_mean']
                else:
                    result[f'{metric}_mean_cv'] = np.nan
            
            if std_col in group_df.columns:
                result[f'{metric}_std_mean'] = group_df[std_col].mean()
                result[f'{metric}_std_std'] = group_df[std_col].std()
        
        ablation_results.append(result)
    
    if not ablation_results:
        return None
    
    ablation_df = pd.DataFrame(ablation_results)
    
    # Sort by parameter values
    sort_cols = [col for col in group_cols if col not in ['prompt', 'target']]
    if 'prompt' in ablation_df.columns:
        sort_cols.append('prompt')
    elif 'target' in ablation_df.columns:
        sort_cols.append('target')
    ablation_df = ablation_df.sort_values(sort_cols)
    
    return ablation_df


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
        "exp0_baseline_indep": "repulsion_type",  # Both baselines use repulsion_type
        "exp1_repulsion_kernel": "method",  # Changed to method (RLSD vs SVGD) instead of kernel
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


def perform_cross_experiment_analysis(experiments: Dict[str, pd.DataFrame], results_dir: Path) -> None:
    """
    Perform cross-experiment analysis for baseline comparisons.
    
    Args:
        experiments: Dictionary of experiment DataFrames
        results_dir: Directory to save results
    """
    # Compare exp0_baseline vs exp0_baseline_indep
    baseline_exps = ["exp0_baseline", "exp0_baseline_indep"]
    available_baselines = [exp for exp in baseline_exps if exp in experiments]
    
    if len(available_baselines) >= 2:
        print(f"\nPerforming cross-experiment analysis: {available_baselines}")
        
        # Combine baseline experiments for comparison
        baseline_data = []
        for exp_name in available_baselines:
            df = experiments[exp_name].copy()
            df['experiment_type'] = exp_name
            baseline_data.append(df)
        
        if baseline_data:
            combined_baseline_df = pd.concat(baseline_data, ignore_index=True)
            
            # Group by experiment type and prompt (if available)
            group_cols = ['experiment_type']
            if 'prompt' in combined_baseline_df.columns:
                group_cols.append('prompt')
            elif 'target' in combined_baseline_df.columns:
                group_cols.append('target')
            
            comparison_results = []
            
            for group_key, group_df in combined_baseline_df.groupby(group_cols):
                if isinstance(group_key, tuple):
                    exp_type, prompt_or_target = group_key
                else:
                    exp_type = group_key
                    prompt_or_target = None
                
                # Calculate statistics across seeds
                result = {
                    'experiment_type': exp_type,
                    'n_runs': len(group_df)
                }
                
                if prompt_or_target is not None:
                    if 'prompt' in combined_baseline_df.columns:
                        result['prompt'] = prompt_or_target
                    else:
                        result['target'] = prompt_or_target
                
                # Calculate mean and std for each metric across seeds
                for metric in ['fidelity', 'diversity', 'cross_consistency']:
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    
                    if mean_col in group_df.columns:
                        result[f'{metric}_mean_mean'] = group_df[mean_col].mean()
                        result[f'{metric}_mean_std'] = group_df[mean_col].std()
                        result[f'{metric}_mean_min'] = group_df[mean_col].min()
                        result[f'{metric}_mean_max'] = group_df[mean_col].max()
                        
                        if result[f'{metric}_mean_mean'] != 0:
                            result[f'{metric}_mean_cv'] = result[f'{metric}_mean_std'] / result[f'{metric}_mean_mean']
                        else:
                            result[f'{metric}_mean_cv'] = np.nan
                    
                    if std_col in group_df.columns:
                        result[f'{metric}_std_mean'] = group_df[std_col].mean()
                        result[f'{metric}_std_std'] = group_df[std_col].std()
                
                comparison_results.append(result)
            
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                
                # Sort by experiment type and prompt/target
                sort_cols = ['experiment_type']
                if 'prompt' in comparison_df.columns:
                    sort_cols.append('prompt')
                elif 'target' in comparison_df.columns:
                    sort_cols.append('target')
                comparison_df = comparison_df.sort_values(sort_cols)
                
                # Create cross-experiment analysis directory
                cross_exp_dir = results_dir / "cross_experiment_analysis"
                cross_exp_dir.mkdir(exist_ok=True)
                
                # Save comparison results in cross-experiment directory
                cross_exp_path = cross_exp_dir / "Baseline_Experiments_Comparison.csv"
                
                # Add methodology header for cross-experiment comparison
                comparison_header = f"""# CROSS-EXPERIMENT COMPARISON - METHODOLOGY
# This file compares performance between different experiment types
# 
# Comparison Details:
# - Experiments compared: {', '.join(available_baselines)}
# - Method: Statistical analysis across multiple seeds and prompts
# - Metrics calculated: Mean, standard deviation, min, max, coefficient of variation
# - Grouping: By experiment type
# - Total experiment types: {len(comparison_df)}
# 
# Each row represents the aggregated performance for a specific experiment type
# Statistics are calculated across all seeds and prompts for that experiment type
#
# Generated by: ablation_analysis.py
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
"""
                
                with open(cross_exp_path, 'w') as f:
                    f.write(comparison_header)
                    comparison_df.to_csv(f, index=False)
                
                print(f"  Saved baseline comparison to {cross_exp_path}")
                
                # Also save in main results directory for compatibility
                main_comparison_path = results_dir / "baseline_comparison.csv"
                comparison_df.to_csv(main_comparison_path, index=False)
                
                # Print summary
                print(f"  Analyzed {len(comparison_df)} experiment type combinations")
                if 'fidelity_mean_mean' in comparison_df.columns:
                    for exp_type in comparison_df['experiment_type'].unique():
                        exp_data = comparison_df[comparison_df['experiment_type'] == exp_type]
                        if not exp_data.empty:
                            avg_fidelity = exp_data['fidelity_mean_mean'].mean()
                            avg_std = exp_data['fidelity_mean_std'].mean()
                            print(f"    {exp_type}: {avg_fidelity:.4f} ± {avg_std:.4f}")


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
        
        # Create global config directory
        global_config_dir = results_dir / "global_config"
        global_config_dir.mkdir(exist_ok=True)
        
        # Save summary table in main directory
        summary_output_path = results_dir / "experiment_summary.csv"
        summary_df.to_csv(summary_output_path, index=False)
        print(f"\nSaved experiment summary to {summary_output_path}")
        
        # Also save in global config directory with descriptive name
        global_summary_path = global_config_dir / "All_Experiments_Summary.csv"
        summary_df.to_csv(global_summary_path, index=False)
        print(f"Also saved to global config directory: {global_summary_path}")
        
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
    
    # Perform cross-experiment analysis
    print("\nPerforming cross-experiment analysis...")
    perform_cross_experiment_analysis(experiments, results_dir)
    
    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(experiments, results_dir)
    
    print("\nAblation analysis complete!")


if __name__ == "__main__":
    main()
