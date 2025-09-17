#!/usr/bin/env python3
"""
Script to consolidate experiment results into single CSV files per experiment.

This script processes experiment data from the ./exp/ directory structure,
extracts metrics from CSV files, and generates consolidated results.
"""

import argparse
import pandas as pd
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings


def flatten_config(config: Dict[str, Any], prefix: str = "cfg_") -> Dict[str, Any]:
    """
    Flatten a nested configuration dictionary with dot notation.
    
    Args:
        config: Nested configuration dictionary
        prefix: Prefix to add to all keys
        
    Returns:
        Flattened dictionary with prefixed keys
    """
    flattened = {}
    
    def _flatten(obj, parent_key="", sep="."):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                _flatten(v, new_key, sep)
        else:
            flattened[f"{prefix}{parent_key}"] = obj
    
    _flatten(config)
    return flattened


def find_last_step_from_filename(csv_path: Path) -> Optional[int]:
    """
    Extract step number from filename pattern like *step{N}*.csv.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Step number if found, None otherwise
    """
    filename = csv_path.name
    match = re.search(r'step(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def find_last_step_from_data(df: pd.DataFrame) -> Optional[int]:
    """
    Find the last step from the step column in the data.
    
    Args:
        df: DataFrame with step column
        
    Returns:
        Maximum step number if step column exists, None otherwise
    """
    if 'step' in df.columns:
        return int(df['step'].max())
    return None


def load_metrics_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load and clean a metrics CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Cleaned DataFrame or None if loading fails
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Convert metric columns to numeric, dropping non-numeric rows
        metric_columns = ['fidelity', 'diversity', 'cross_consistency', 'inter_particle_diversity', 'cross_view_consistency']
        for col in metric_columns:
            if col in df.columns:
                # Try to convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values in metric columns
        metric_cols_present = [col for col in metric_columns if col in df.columns]
        if metric_cols_present:
            initial_rows = len(df)
            df = df.dropna(subset=metric_cols_present)
            if len(df) < initial_rows:
                warnings.warn(f"Dropped {initial_rows - len(df)} non-numeric rows from {csv_path}")
        
        return df
        
    except Exception as e:
        warnings.warn(f"Failed to load {csv_path}: {e}")
        return None


def get_last_step_data(run_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    """
    Get the last step data for a run.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tuple of (DataFrame with last step data, step number)
    """
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return None, None
    
    # Look for quantitative_metrics.csv first
    csv_files = list(metrics_dir.glob("*.csv"))
    if not csv_files:
        return None, None
    
    # Prioritize quantitative_metrics.csv
    quantitative_metrics_file = metrics_dir / "quantitative_metrics.csv"
    if quantitative_metrics_file.exists():
        df = load_metrics_csv(quantitative_metrics_file)
        if df is not None:
            step_from_data = find_last_step_from_data(df)
            if step_from_data is not None:
                last_step_df = df[df['step'] == df['step'].max()].copy()
                return last_step_df, step_from_data
    
    # Find the file with the highest step number
    best_file = None
    best_step = -1
    
    for csv_file in csv_files:
        # Try to get step from filename first
        step_from_filename = find_last_step_from_filename(csv_file)
        
        # Load the file to check step column
        df = load_metrics_csv(csv_file)
        if df is None:
            continue
            
        step_from_data = find_last_step_from_data(df)
        
        # Use the higher step number
        current_step = max(
            step_from_filename if step_from_filename is not None else -1,
            step_from_data if step_from_data is not None else -1
        )
        
        if current_step > best_step:
            best_step = current_step
            best_file = csv_file
    
    if best_file is None:
        return None, None
    
    # Load the best file and get the last step data
    df = load_metrics_csv(best_file)
    if df is None:
        return None, None
    
    # If there's a step column, get the last step
    if 'step' in df.columns:
        last_step_df = df[df['step'] == df['step'].max()].copy()
    else:
        last_step_df = df.copy()
    
    return last_step_df, best_step


def extract_run_info(run_dir: Path) -> Dict[str, Any]:
    """
    Extract run information from directory name and config.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with run information
    """
    run_info = {
        'run_id': run_dir.name,
        'prompt': None,
        'seed': None
    }
    
    # Try to extract prompt and seed from config
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            run_info['prompt'] = config.get('prompt')
            run_info['seed'] = config.get('seed')
        except Exception as e:
            warnings.warn(f"Failed to load config from {config_path}: {e}")
    
    return run_info


def extract_ablation_parameters(run_id: str, experiment_name: str, run_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ablation parameters from run_id based on experiment type.
    Uses full prompts instead of abbreviated target names.
    
    Args:
        run_id: Run directory name
        experiment_name: Name of the experiment
        run_info: Dictionary with run information including prompt
        
    Returns:
        Dictionary with extracted parameters
    """
    params = {}
    
    # Use full prompt if available, otherwise fall back to target abbreviation
    prompt = run_info.get('prompt', '')
    if prompt:
        params['prompt'] = prompt
    else:
        # Fallback to extracting target abbreviation
        if experiment_name == "exp0_baseline":
            parts = run_id.split("__")
            if len(parts) >= 3:
                params['target'] = parts[1]
        elif experiment_name == "exp1_repulsion_kernel":
            parts = run_id.split("__")
            if len(parts) >= 4:
                params['target'] = parts[2]
        elif experiment_name in ["exp2_lambda_coarse", "exp3_lambda_fine"]:
            parts = run_id.split("__")
            if len(parts) >= 3:
                params['target'] = parts[1]
        elif experiment_name in ["exp4_guidance_scale", "exp5_rbf_beta"]:
            parts = run_id.split("__")
            if len(parts) >= 3:
                params['target'] = parts[0]
        elif experiment_name in ["exp_feature_layer", "exp_num_particles", "exp_num_pts", "exp_opacity_lr"]:
            parts = run_id.split("__")
            if len(parts) >= 2:
                params['target'] = parts[0]
    
    if experiment_name == "exp0_baseline":
        # Format: WO__{TARGET}__S{SEED}
        parts = run_id.split("__")
        if len(parts) >= 3:
            params['repulsion_type'] = parts[0]
            params['seed'] = parts[2].replace('S', '')
    
    elif experiment_name == "exp1_repulsion_kernel":
        # Format: {METHOD}__{KERNEL}__{TARGET}__S{SEED}
        parts = run_id.split("__")
        if len(parts) >= 4:
            params['method'] = parts[0]
            params['kernel'] = parts[1]
            params['seed'] = parts[3].replace('S', '')
    
    elif experiment_name == "exp2_lambda_coarse":
        # Format: λ{VALUE}__{TARGET}__S{SEED}
        parts = run_id.split("__")
        if len(parts) >= 3:
            lambda_part = parts[0]
            # Handle K suffix properly (K = 1000)
            lambda_value = lambda_part.replace('λ', '')
            if lambda_value.endswith('K'):
                # Convert K to 1000, handling decimals like 1.2K -> 1200
                base_value = lambda_value[:-1]  # Remove K
                params['lambda_repulsion'] = str(float(base_value) * 1000)
            else:
                params['lambda_repulsion'] = lambda_value
            params['seed'] = parts[2].replace('S', '')
    
    elif experiment_name == "exp3_lambda_fine":
        # Format: λ{VALUE}__{TARGET}__S{SEED}
        parts = run_id.split("__")
        if len(parts) >= 3:
            lambda_part = parts[0]
            # Handle K suffix properly (K = 1000)
            lambda_value = lambda_part.replace('λ', '')
            if lambda_value.endswith('K'):
                # Convert K to 1000, handling decimals like 1.2K -> 1200
                base_value = lambda_value[:-1]  # Remove K
                params['lambda_repulsion'] = str(float(base_value) * 1000)
            else:
                params['lambda_repulsion'] = lambda_value
            params['seed'] = parts[2].replace('S', '')
    
    elif experiment_name == "exp4_guidance_scale":
        # Format: {TARGET}__S{SEED}__guidance_scale_{VALUE}
        parts = run_id.split("__")
        if len(parts) >= 3:
            params['seed'] = parts[1].replace('S', '')
            guidance_part = parts[2]
            if guidance_part.startswith('guidance_scale_'):
                params['guidance_scale'] = guidance_part.replace('guidance_scale_', '')
    
    elif experiment_name == "exp5_rbf_beta":
        # Format: {TARGET}__S{SEED}__rbf_beta_{VALUE}
        parts = run_id.split("__")
        if len(parts) >= 3:
            params['seed'] = parts[1].replace('S', '')
            beta_part = parts[2]
            if beta_part.startswith('rbf_beta_'):
                params['rbf_beta'] = beta_part.replace('rbf_beta_', '')
    
    elif experiment_name == "exp_feature_layer":
        # Format: {TARGET}__feature_layer_{VALUE}
        parts = run_id.split("__")
        if len(parts) >= 2:
            layer_part = parts[1]
            if layer_part.startswith('feature_layer_'):
                params['feature_layer'] = layer_part.replace('feature_layer_', '')
    
    elif experiment_name == "exp_num_particles":
        # Format: {TARGET}__num_particles_{VALUE}
        parts = run_id.split("__")
        if len(parts) >= 2:
            particles_part = parts[1]
            if particles_part.startswith('num_particles_'):
                params['num_particles'] = particles_part.replace('num_particles_', '')
    
    elif experiment_name == "exp_num_pts":
        # Format: {TARGET}__num_pts_{VALUE}
        parts = run_id.split("__")
        if len(parts) >= 2:
            pts_part = parts[1]
            if pts_part.startswith('num_pts_'):
                params['num_pts'] = pts_part.replace('num_pts_', '')
    
    elif experiment_name == "exp_opacity_lr":
        # Format: {TARGET}__opacity_lr_{VALUE}
        parts = run_id.split("__")
        if len(parts) >= 2:
            lr_part = parts[1]
            if lr_part.startswith('opacity_lr_'):
                params['opacity_lr'] = lr_part.replace('opacity_lr_', '')
    
    return params


def process_experiment(exp_dir: Path, results_dir: Path) -> None:
    """
    Process a single experiment directory and generate consolidated CSV.
    
    Args:
        exp_dir: Path to experiment directory
        results_dir: Path to results directory
    """
    exp_name = exp_dir.name
    print(f"Processing experiment: {exp_name}")
    
    # Find all run directories (exclude files and special directories)
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not run_dirs:
        print(f"  No run directories found in {exp_dir}")
        return
    
    # Load experiment config if it exists
    config_path = exp_dir / "test_ours_exp.yaml"
    if not config_path.exists():
        # Try to find any config.yaml in a run directory
        for run_dir in run_dirs:
            run_config = run_dir / "config.yaml"
            if run_config.exists():
                config_path = run_config
                break
    
    experiment_config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                experiment_config = yaml.safe_load(f)
        except Exception as e:
            warnings.warn(f"Failed to load experiment config from {config_path}: {e}")
    
    # Flatten config
    flattened_config = flatten_config(experiment_config)
    
    # Process each run
    runs_data = []
    
    for run_dir in run_dirs:
        print(f"  Processing run: {run_dir.name}")
        
        # Get last step data
        last_step_df, last_step = get_last_step_data(run_dir)
        if last_step_df is None or last_step is None:
            print(f"    No valid metrics data found")
            continue
        
        # Extract run info
        run_info = extract_run_info(run_dir)
        
        # Extract ablation parameters
        ablation_params = extract_ablation_parameters(run_dir.name, exp_name, run_info)
        
        # Calculate metrics
        run_data = {
            'experiment': exp_name,
            'run_id': run_info['run_id'],
            'last_step': last_step,
            'n_samples': len(last_step_df)
        }
        
        # Add prompt and seed if available
        if run_info['prompt'] is not None:
            run_data['prompt'] = run_info['prompt']
        if run_info['seed'] is not None:
            run_data['seed'] = run_info['seed']
        
        # Add ablation parameters
        run_data.update(ablation_params)
        
        # Calculate metric statistics
        metric_mappings = {
            'fidelity': ['fidelity_mean', 'fidelity_std'],
            'diversity': ['inter_particle_diversity_mean', 'inter_particle_diversity_std'],
            'cross_consistency': ['cross_view_consistency_mean', 'cross_view_consistency_std']
        }
        
        for metric, (mean_col, std_col) in metric_mappings.items():
            if mean_col in last_step_df.columns and std_col in last_step_df.columns:
                run_data[f'{metric}_mean'] = last_step_df[mean_col].mean()
                run_data[f'{metric}_std'] = last_step_df[std_col].mean()
            elif metric == 'fidelity' and 'fidelity' in last_step_df.columns:
                # Handle case where we have raw fidelity values
                run_data[f'{metric}_mean'] = last_step_df['fidelity'].mean()
                run_data[f'{metric}_std'] = last_step_df['fidelity'].std()
            elif metric == 'diversity' and 'inter_particle_diversity' in last_step_df.columns:
                # Handle case where we have raw diversity values
                run_data[f'{metric}_mean'] = last_step_df['inter_particle_diversity'].mean()
                run_data[f'{metric}_std'] = last_step_df['inter_particle_diversity'].std()
            elif metric == 'cross_consistency' and 'cross_view_consistency' in last_step_df.columns:
                # Handle case where we have raw cross_consistency values
                run_data[f'{metric}_mean'] = last_step_df['cross_view_consistency'].mean()
                run_data[f'{metric}_std'] = last_step_df['cross_view_consistency'].std()
        
        # Add flattened config
        run_data.update(flattened_config)
        
        runs_data.append(run_data)
    
    if not runs_data:
        print(f"  No valid runs found for experiment {exp_name}")
        return
    
    # Create runs DataFrame
    runs_df = pd.DataFrame(runs_data)
    
    # Sort by run_id, prompt, seed
    sort_cols = ['run_id']
    if 'prompt' in runs_df.columns:
        sort_cols.append('prompt')
    if 'seed' in runs_df.columns:
        sort_cols.append('seed')
    runs_df = runs_df.sort_values(sort_cols)
    
    # Save consolidated runs data
    consolidated_output_path = results_dir / f"{exp_name}_consolidated.csv"
    runs_df.to_csv(consolidated_output_path, index=False)
    print(f"  Saved consolidated data to {consolidated_output_path}")
    
    # Print summary statistics
    print(f"  Summary: {len(runs_df)} runs processed")
    if 'fidelity_mean' in runs_df.columns:
        print(f"    Fidelity: {runs_df['fidelity_mean'].mean():.4f} ± {runs_df['fidelity_mean'].std():.4f}")
    if 'diversity_mean' in runs_df.columns:
        print(f"    Diversity: {runs_df['diversity_mean'].mean():.4f} ± {runs_df['diversity_mean'].std():.4f}")
    if 'cross_consistency_mean' in runs_df.columns:
        print(f"    Cross-consistency: {runs_df['cross_consistency_mean'].mean():.4f} ± {runs_df['cross_consistency_mean'].std():.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Consolidate experiment results into single CSV files")
    parser.add_argument("--exp-root", default="./exp", help="Root directory containing experiments")
    parser.add_argument("--results-dir", default="./results", help="Directory to save results")
    
    args = parser.parse_args()
    
    exp_root = Path(args.exp_root)
    results_dir = Path(args.results_dir)
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not exp_root.exists():
        print(f"Experiment root directory {exp_root} does not exist")
        return
    
    # Find all experiment directories
    exp_dirs = [d for d in exp_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not exp_dirs:
        print(f"No experiment directories found in {exp_root}")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Process each experiment
    for exp_dir in sorted(exp_dirs):
        try:
            process_experiment(exp_dir, results_dir)
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Consolidation complete!")


if __name__ == "__main__":
    main()
