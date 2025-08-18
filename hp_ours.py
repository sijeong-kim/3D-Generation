#!/usr/bin/env python3
"""
Hyperparameter tuning script for 3D Generation experiments.
Reads experiment configurations from YAML files and runs grid search experiments.
"""

import argparse
import itertools
import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import datetime


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_combinations(sweep_params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of sweep parameters."""
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    prompts_dict = sweep_params['prompts_dict']
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations, prompts_dict


def merge_configs(base_config: Dict[str, Any], fixed_params: Dict[str, Any], 
                  setting_params: Dict[str, Any], sweep_params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with fixed and sweep parameters."""
    # Start with base config
    merged_config = base_config.copy()
    
    # Override with setting parameters
    merged_config.update(setting_params)
    
    # Override with fixed parameters
    merged_config.update(fixed_params)
    
    # Override with sweep parameters
    merged_config.update(sweep_params)
    
    return merged_config


def save_config_to_file(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

# run_single_experiment(merged_config, output_dir, args.config, args.sweep_name)
def run_single_experiment(
    config: Dict[str, Any], 
    output_dir: str, 
    base_config_path: str, 
    exp_name: str
    ) -> bool:
    """Run a single experiment with the given configuration."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output directory in config
    config['outdir'] = output_dir
    
    # Save configuration to output directory
    config_path = os.path.join(output_dir, "config.yaml")
    save_config_to_file(config, config_path)
    

    # Prepare command
    cmd = [
        sys.executable, "main_ours.py",
        "--config", config_path,
        "--outdir", output_dir
    ]
    
    # Add prompt if specified
    if config.get('prompt'):
        cmd.extend(["--prompt", config['prompt']])
    
    print(f"Running experiment: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # TODO✅:check cmd
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Save stdout and stderr
        with open(os.path.join(output_dir, "stdout.log"), 'w') as f:
            f.write(result.stdout)
        
        with open(os.path.join(output_dir, "stderr.log"), 'w') as f:
            f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ Experiment completed successfully: {output_dir}")
            return True
        else:
            print(f"❌ Experiment failed (return code {result.returncode}): {output_dir}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during experiment: {output_dir}")
        print(f"Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for 3D Generation")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to base configuration file")
    parser.add_argument("--sweep_config", type=str, required=True,
                       help="Path to sweep configuration file")
    parser.add_argument("--sweep_name", type=str, required=True,
                       help="Name of the experiment to run")
    parser.add_argument("--outdir", type=str, default="logs",
                       help="Base output directory")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without running experiments")
    
    args = parser.parse_args()
    
    # Load configurations
    # print(f"Loading base config: {args.config}") # 3D-Generation/configs/text_ours.yaml
    base_config = load_yaml_config(args.config)
    
    # print(f"Loading sweep config: {args.sweep_config}") # 3D-Generation/configs/text_ours_exp.yaml
    sweep_config = load_yaml_config(args.sweep_config)
    
    # Get experiment configuration
    if args.sweep_name not in sweep_config:
        print(f"❌ Experiment '{args.sweep_name}' not found in sweep config") #
        print(f"Available experiments: {list(sweep_config.keys())}")
        sys.exit(1)
    
    exp_config = sweep_config[args.sweep_name] # exp1_lambda_repulsion_sweep
    
    # Extract parameters
    sweep_params = exp_config.get('sweep_parameters', {}) # repulsion_type: ['svgd', 'rlsd']
    fixed_params = exp_config.get('fixed_parameters', {}) # num_particles: 8
    setting_params = exp_config.get('settings', {}) # visualize: True
    
    # Generate all parameter combinations
    param_combinations, prompts_dict = generate_parameter_combinations(sweep_params)
    # param_combinations[0]: {'seed': 42, 'prompts_dict': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}
    # prompts_dict: {'hamburger': ['a photo of a hamburger'], 'icecream': ['a photo of an ice cream'], 'saguaro': ['a small saguaro cactus planted in a clay pot'], 'tulip': ['a photo of a tulip']}
    
    print(f"Experiment: {args.sweep_name}")
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"Total experiments: {len(param_combinations)}")
    print(f"Sweep parameters: {list(sweep_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Setting parameters: {list(setting_params.keys())}")
    print("=" * 80)
    
    # Create base output directory
    base_output_dir = os.path.join(args.outdir, args.sweep_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save experiment summary
    summary = {
        'experiment_name': args.sweep_name,
        'parameter_combinations': len(param_combinations),
        'total_experiments': len(param_combinations),
        'sweep_parameters': list(sweep_params.keys()),
        'fixed_parameters': list(fixed_params.keys()),
        'setting_parameters': list(setting_params.keys()),
        'start_time': datetime.datetime.now().isoformat(),
        'combinations': []
    }
    
    # Run experiments
    successful_runs = 0
    failed_runs = 0
    
    # Run experiments for each prompt and parameter combination
    total_experiments = len(param_combinations)
    experiment_count = 0
    
    for i, params in enumerate(param_combinations):
        experiment_count += 1
        print(f"\n[{experiment_count}/{total_experiments}] Processing combination...")
        
        # Merge configurations
        merged_config = merge_configs(base_config, fixed_params, setting_params, params)
        
        
        # TODO✅ Update config with prompt
        prompt_key = params['prompts_dict'] # hamburger
        prompt_value = prompts_dict[prompt_key][0] # a photo of a hamburger
        merged_config['prompt'] = prompt_value
        del merged_config['prompts_dict']
        
        # Create output directory name
        output_dir_name = "_".join(str(v) for v in params.values())
        output_dir = os.path.join(base_output_dir, output_dir_name)
        
        # Record combination
        combination_info = {
            'experiment_id': experiment_count,
            'parameters': params,
            'output_dir': output_dir,
            'prompt': prompt_value,
            'status': 'pending'
        }
        summary['combinations'].append(combination_info)
        
        if args.dry_run:
            print(f"DRY RUN - Would run: {output_dir}")
            print(f"  Parameters: {params}")
            continue
        
        
        # TODO✅:check merged_config
        # Run experiment
        success = run_single_experiment(merged_config, output_dir, args.config, args.sweep_name)
        
        if success:
            successful_runs += 1
            combination_info['status'] = 'success'
        else:
            failed_runs += 1
            combination_info['status'] = 'failed'
        
        # Save updated summary
        summary['end_time'] = datetime.datetime.now().isoformat()
        summary['successful_runs'] = successful_runs
        summary['failed_runs'] = failed_runs
        
        summary_path = os.path.join(base_output_dir, "experiment_summary.yaml")
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Experiment: {args.sweep_name}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/total_experiments*100:.1f}%")
    print(f"Results saved in: {base_output_dir}")
    print(f"Summary file: {os.path.join(base_output_dir, 'experiment_summary.yaml')}")
    
    if failed_runs > 0:
        print(f"\n⚠️  {failed_runs} experiments failed. Check individual logs for details.")
        sys.exit(1)
    else:
        print(f"\n✅ All experiments completed successfully!")


if __name__ == "__main__":
    main()
