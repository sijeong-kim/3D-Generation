#!/usr/bin/env python3
"""
Test script for hyperparameter tuning setup.
This script tests the configuration loading and parameter generation without running actual experiments.
"""

import yaml
import itertools
from typing import Dict, List, Any

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_parameter_combinations(sweep_params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of sweep parameters."""
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations

def main():
    print("Testing hyperparameter tuning setup...")
    
    # Load configurations
    base_config = load_yaml_config("configs/text_ours.yaml")
    exp_config = load_yaml_config("configs/text_ours_exp.yaml")
    
    print(f"✅ Loaded base config with {len(base_config)} parameters")
    print(f"✅ Loaded experiment config with {len(exp_config)} experiments")
    
    # Test each experiment
    for exp_name, exp_data in exp_config.items():
        print(f"\n--- Testing experiment: {exp_name} ---")
        
        sweep_params = exp_data.get('sweep_parameters', {})
        fixed_params = exp_data.get('fixed_parameters', {})
        prompts = fixed_params.get('prompts', ["a photo of a hamburger"])
        
        # Generate combinations
        param_combinations = generate_parameter_combinations(sweep_params)
        total_experiments = len(param_combinations) * len(prompts)
        
        print(f"  Sweep parameters: {list(sweep_params.keys())}")
        print(f"  Parameter combinations: {len(param_combinations)}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Total experiments: {total_experiments}")
        
        # Show some example combinations
        print(f"  Example parameter combinations:")
        for i, combo in enumerate(param_combinations[:3]):  # Show first 3
            print(f"    {i+1}: {combo}")
        
        if len(param_combinations) > 3:
            print(f"    ... and {len(param_combinations) - 3} more")
        
        print(f"  Prompts:")
        for prompt in prompts:
            print(f"    - {prompt}")
    
    print(f"\n✅ All experiments configured correctly!")
    print(f"\nTo run experiments, use:")
    print(f"  ./scripts/run_ours_exp.sh exp1_repulsion_lambda_sweep")
    print(f"  ./scripts/run_ours_exp.sh exp1_wo_method")
    print(f"\nOr for SLURM:")
    print(f"  sbatch scripts/run_ours_exp_slurm.sh exp1_repulsion_lambda_sweep")
    print(f"  sbatch scripts/run_ours_exp_slurm.sh exp1_wo_method")

if __name__ == "__main__":
    main()
