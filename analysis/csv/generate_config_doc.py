#!/usr/bin/env python3
"""
Script to generate consolidated configuration documentation.

This script reads experiment configurations and creates a single
YAML file documenting all experiment parameters for easy reference.
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings


def load_experiment_configs(exp_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load configuration from all experiments.
    
    Args:
        exp_root: Root directory containing experiments
        
    Returns:
        Dictionary mapping experiment names to their configurations
    """
    experiments = {}
    
    for exp_dir in exp_root.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        exp_name = exp_dir.name
        print(f"Loading config for {exp_name}")
        
        # Try to find config files
        config_files = []
        
        # Look for test_ours_exp.yaml first
        test_config = exp_dir / "test_ours_exp.yaml"
        if test_config.exists():
            config_files.append(test_config)
        
        # Look for experiment_summary.yaml
        summary_config = exp_dir / "experiment_summary.yaml"
        if summary_config.exists():
            config_files.append(summary_config)
        
        # Look for config.yaml in any run directory
        for run_dir in exp_dir.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith('.'):
                run_config = run_dir / "config.yaml"
                if run_config.exists():
                    config_files.append(run_config)
                    break  # Just need one representative config
        
        # Load and merge configurations
        exp_config = {}
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        exp_config.update(config)
            except Exception as e:
                warnings.warn(f"Failed to load {config_file}: {e}")
        
        if exp_config:
            experiments[exp_name] = exp_config
        else:
            warnings.warn(f"No configuration found for {exp_name}")
    
    return experiments


def extract_experiment_parameters(exp_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key parameters for an experiment.
    
    Args:
        exp_name: Name of the experiment
        config: Configuration dictionary
        
    Returns:
        Dictionary with extracted parameters
    """
    params = {
        'experiment_name': exp_name,
        'description': get_experiment_description(exp_name),
        'parameters': {}
    }
    
    # Extract key parameters based on experiment type
    if exp_name == "exp0_baseline":
        params['parameters'] = {
            'repulsion_type': 'wo',
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'kernel_type': config.get('kernel_type', 'none'),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42, 123, 456, 789]
        }
    
    elif exp_name == "exp1_repulsion_kernel":
        params['parameters'] = {
            'methods': ['RLSD', 'SVGD'],
            'kernels': ['RBF', 'COS'],
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42, 123]
        }
    
    elif exp_name == "exp2_lambda_coarse":
        params['parameters'] = {
            'lambda_repulsion_values': [1, 10, 100, 1000, 10000],
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'kernel_type': config.get('kernel_type', 'rbf'),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42]
        }
    
    elif exp_name == "exp3_lambda_fine":
        params['parameters'] = {
            'lambda_repulsion_values': [600, 800, 1000, 1200, 1400],
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'kernel_type': config.get('kernel_type', 'rbf'),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42, 123]
        }
    
    elif exp_name == "exp4_guidance_scale":
        params['parameters'] = {
            'guidance_scale_values': [30, 50, 70, 100],
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42, 123]
        }
    
    elif exp_name == "exp5_rbf_beta":
        params['parameters'] = {
            'rbf_beta_values': [0.5, 1.0, 1.5, 2.0],
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'kernel_type': config.get('kernel_type', 'rbf'),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42, 123]
        }
    
    elif exp_name == "exp_feature_layer":
        params['parameters'] = {
            'feature_layer_values': ['early', 'mid', 'last'],
            'feature_extractor': config.get('feature_extractor_model_name', 'facebook/dinov2-base'),
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42]
        }
    
    elif exp_name == "exp_num_particles":
        params['parameters'] = {
            'num_particles_values': [2, 4, 8],
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'targets': ['BULL', 'ICE', 'CACT', 'TUL'],
            'seeds': [42]
        }
    
    elif exp_name == "exp_num_pts":
        params['parameters'] = {
            'num_pts_values': [1000, 3000, 5000],
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'targets': ['ICE', 'CACT'],
            'seeds': [42]
        }
    
    elif exp_name == "exp_opacity_lr":
        params['parameters'] = {
            'opacity_lr_values': [0.005, 0.01, 0.05],
            'lambda_repulsion': config.get('lambda_repulsion', 1000),
            'repulsion_type': config.get('repulsion_type', 'rlsd'),
            'targets': ['ICE', 'CACT'],
            'seeds': [42]
        }
    
    # Add common parameters
    common_params = {
        'iters': config.get('iters', 1000),
        'guidance_scale': config.get('guidance_scale', 50),
        'num_views': config.get('num_views', 8),
        'batch_size': config.get('batch_size', 1),
        'feature_lr': config.get('feature_lr', 0.01),
        'geom_lr': config.get('geom_lr', 0.0001),
        'rotation_lr': config.get('rotation_lr', 0.005),
        'scaling_lr': config.get('scaling_lr', 0.005),
        'texture_lr': config.get('texture_lr', 0.2),
        'opacity_lr': config.get('opacity_lr', 0.01),
        'rbf_beta': config.get('rbf_beta', 1.0),
        'radius': config.get('radius', 2.5),
        'fovy': config.get('fovy', 49.1),
        'eval_radius': config.get('eval_radius', 4.5),
        'feature_extractor_model_name': config.get('feature_extractor_model_name', 'facebook/dinov2-base'),
        'repulsion_type': config.get('repulsion_type', 'wo'),
        'kernel_type': config.get('kernel_type', 'none'),
        'lambda_repulsion': config.get('lambda_repulsion', 1000),
        'lambda_sd': config.get('lambda_sd', 1),
        'lambda_zero123': config.get('lambda_zero123', 0)
    }
    
    params['common_parameters'] = common_params
    
    return params


def get_experiment_description(exp_name: str) -> str:
    """
    Get a description for an experiment.
    
    Args:
        exp_name: Name of the experiment
        
    Returns:
        Description string
    """
    descriptions = {
        "exp0_baseline": "Baseline experiment with no repulsion (WO - Without repulsion)",
        "exp1_repulsion_kernel": "Repulsion kernel ablation: RLSD vs SVGD with RBF vs COS kernels",
        "exp2_lambda_coarse": "Coarse lambda repulsion ablation: [1, 10, 100, 1K, 10K]",
        "exp3_lambda_fine": "Fine lambda repulsion ablation: [600, 800, 1K, 1.2K, 1.4K]",
        "exp4_guidance_scale": "Guidance scale ablation: [30, 50, 70, 100]",
        "exp5_rbf_beta": "RBF kernel beta parameter ablation: [0.5, 1.0, 1.5, 2.0]",
        "exp_feature_layer": "Feature layer ablation: [early, mid, last]",
        "exp_num_particles": "Number of particles ablation: [2, 4, 8]",
        "exp_num_pts": "Number of points ablation: [1K, 3K, 5K]",
        "exp_opacity_lr": "Opacity learning rate ablation: [0.005, 0.01, 0.05]"
    }
    
    return descriptions.get(exp_name, f"Experiment: {exp_name}")


def create_consolidated_config(experiments: Dict[str, Dict[str, Any]], results_dir: Path) -> None:
    """
    Create consolidated configuration documentation.
    
    Args:
        experiments: Dictionary of experiment configurations
        results_dir: Directory to save results
    """
    consolidated_config = {
        'experiment_suite': {
            'description': '3D Generation Experiment Suite - Comprehensive ablation study',
            'total_experiments': len(experiments),
            'experiments': {}
        },
        'metrics': {
            'fidelity': 'Image fidelity metric (higher is better)',
            'diversity': 'Inter-particle diversity metric (higher is better)',
            'cross_consistency': 'Cross-view consistency metric (higher is better)'
        },
        'prompts': {
            'a bulldozer made out of toy bricks': 'Bulldozer made out of toy bricks',
            'a photo of an ice cream': 'Photo of an ice cream',
            'a small saguaro cactus planted in a clay pot': 'Small saguaro cactus planted in a clay pot',
            'a tulip flower in a vase': 'Tulip flower in a vase'
        },
        'seeds': {
            'description': 'Random seeds used for reproducibility',
            'values': [42, 123, 456, 789]
        }
    }
    
    # Add experiment details
    for exp_name, config in experiments.items():
        exp_params = extract_experiment_parameters(exp_name, config)
        consolidated_config['experiment_suite']['experiments'][exp_name] = exp_params
    
    # Save consolidated configuration
    config_output_path = results_dir / "experiment_configuration.yaml"
    with open(config_output_path, 'w') as f:
        yaml.dump(consolidated_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved consolidated configuration to {config_output_path}")
    
    # Create a simplified summary
    summary_output_path = results_dir / "experiment_summary.md"
    with open(summary_output_path, 'w') as f:
        f.write("# 3D Generation Experiment Suite\n\n")
        f.write("## Overview\n\n")
        f.write(f"Total experiments: {len(experiments)}\n\n")
        
        f.write("## Experiments\n\n")
        for exp_name, config in experiments.items():
            exp_params = extract_experiment_parameters(exp_name, config)
            f.write(f"### {exp_name}\n")
            f.write(f"**Description:** {exp_params['description']}\n\n")
            
            if 'parameters' in exp_params:
                f.write("**Key Parameters:**\n")
                for param, value in exp_params['parameters'].items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
        
        f.write("## Metrics\n\n")
        f.write("- **Fidelity**: Image fidelity metric (higher is better)\n")
        f.write("- **Diversity**: Inter-particle diversity metric (higher is better)\n")
        f.write("- **Cross-consistency**: Cross-view consistency metric (higher is better)\n\n")
        
        f.write("## Prompts\n\n")
        for prompt, desc in consolidated_config['prompts'].items():
            f.write(f"- **\"{prompt}\"**: {desc}\n")
    
    print(f"Saved experiment summary to {summary_output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate consolidated configuration documentation")
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
    
    # Load experiment configurations
    print("Loading experiment configurations...")
    experiments = load_experiment_configs(exp_root)
    
    if not experiments:
        print("No experiment configurations found")
        return
    
    print(f"Loaded configurations for {len(experiments)} experiments")
    
    # Create consolidated configuration
    print("Creating consolidated configuration documentation...")
    create_consolidated_config(experiments, results_dir)
    
    print("Configuration documentation complete!")


if __name__ == "__main__":
    main()
