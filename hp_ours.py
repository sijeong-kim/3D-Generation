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


def generate_parameter_combinations_and_sweep_params_dict(sweep_params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of sweep parameters.

    Supports special handling for the following cases:
    - sweep_parameters.prompt provided as a mapping {abbr: full_prompt}. We sweep over the abbr keys
      and later map to the full string in merge_configs.
    - sweep_parameters.lambda_repulsion provided as a mapping {"<repulsion>_<kernel>": [values...]}. We expand
      lambda values AFTER choosing repulsion_type and kernel_type for each base combination.
    """

    sweep_params_dict: Dict[str, Any] = {}

    # Build base sweep lists (exclude lambda_repulsion when it's a dict; handle later)
    base_keys: List[str] = []
    base_values: List[List[Any]] = []

    for key, value in sweep_params.items():
        if isinstance(value, dict):
            # Store dict-valued sweep parameters for later mapping in merge (e.g., prompt)
            # but skip adding lambda_repulsion into sweep_params_dict to avoid mapping over numeric values later
            if key != 'lambda_repulsion':
                sweep_params_dict[key] = value

            if key == 'prompt':
                # Sweep over prompt abbreviations (keys)
                base_keys.append(key)
                base_values.append(list(value.keys()))
            elif key == 'lambda_repulsion':
                # Defer handling lambda_repulsion dict until after base product is formed
                continue
            else:
                # Generic behavior: if a dict is provided and it's not lambda_repulsion, sweep over its keys
                base_keys.append(key)
                base_values.append(list(value.keys()))
        else:
            # Standard list-like sweep parameter
            base_keys.append(key)
            base_values.append(value)

    # Create base combinations (without expanding lambda_repulsion dict)
    base_combinations: List[Dict[str, Any]] = []
    for combination in itertools.product(*base_values) if base_values else [()]:
        param_dict = dict(zip(base_keys, combination))
        base_combinations.append(param_dict)

    # Expand lambda_repulsion if it is provided as a dict keyed by method+kernel
    combinations: List[Dict[str, Any]] = []
    lambda_spec = sweep_params.get('lambda_repulsion', None)
    if isinstance(lambda_spec, dict):
        for param_dict in base_combinations:
            repulsion = param_dict.get('repulsion_type')
            kernel = param_dict.get('kernel_type')
            # Only expand if both keys are present
            if repulsion is not None and kernel is not None:
                method_kernel_key = f"{repulsion}_{kernel}"
                lambda_values = lambda_spec.get(method_kernel_key)
                if isinstance(lambda_values, list) and len(lambda_values) > 0:
                    for lam in lambda_values:
                        new_dict = param_dict.copy()
                        new_dict['lambda_repulsion'] = lam
                        combinations.append(new_dict)
                elif isinstance(lambda_values, (int, float)):
                    new_dict = param_dict.copy()
                    new_dict['lambda_repulsion'] = lambda_values
                    combinations.append(new_dict)
                else:
                    # No lambda values provided for this method+kernel; keep combination without setting it
                    combinations.append(param_dict)
            else:
                # Cannot infer method+kernel; keep as-is
                combinations.append(param_dict)
    else:
        # lambda_repulsion is not a dict; it was already included in base sweep (if present)
        combinations = base_combinations

    return combinations, sweep_params_dict
    # combinations: [{'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}]
    # sweep_params_dict: {'prompt': {'hamburger': 'a photo of a hamburger', 'icecream': 'a photo of an ice cream', 'saguaro': 'a small saguaro cactus planted in a clay pot', 'tulip': 'a photo of a tulip'}}

def generate_fixed_param_dict(fixed_params: Dict[str, List]) -> Dict[str, Any]:
    """Generate fixed parameters."""

    fixed_params_dict = {}
    for key, value in fixed_params.items():
        if isinstance(value, dict):
            fixed_params_dict[key] = value
            
    # remove the key from fixed_params
    for key in fixed_params_dict.keys():
        fixed_params.pop(key)
    
    return fixed_params,fixed_params_dict
    # fixed_params_dict: {'lambda_repulsion': {'rlsd_rbf': 1000, 'rlsd_cosine': 1000, 'svgd_rbf': 1000, 'svgd_cosine': 1000}}

def merge_configs(base_config: Dict[str, Any], fixed_params: Dict[str, Any], fixed_params_dict: Dict[str, Any],
                  setting_params: Dict[str, Any], sweep_params: Dict[str, Any], sweep_params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with fixed and sweep parameters."""
    # Start with base config
    merged_config = base_config.copy()
    
    # Override with setting parameters
    merged_config.update(setting_params)
    
    # Override with fixed parameters
    merged_config.update(fixed_params)
    
    # Override with sweep parameters
    merged_config.update(sweep_params)
    
    # TODO✅ Update config with prompt
    for key, value in sweep_params_dict.items():
        merged_config[key] = value[sweep_params[key]] # merged_config['prompt'] = a photo of a hamburger
    
    
    # TODO✅ set lambda_repulsion for each method-kernel combination
    if 'lambda_repulsion' in fixed_params_dict and \
        'repulsion_type' in sweep_params and \
        'kernel_type' in sweep_params:
        method_kernel_key = sweep_params['repulsion_type'] + "_" + sweep_params['kernel_type']
        method_kernel_value = fixed_params_dict['lambda_repulsion'][method_kernel_key]
        merged_config['lambda_repulsion'] = method_kernel_value
        print(f"Method kernel (key, value): ({method_kernel_key}, {method_kernel_value})")
    
    # TODO✅ set eval_radius for each prompt
    if 'eval_radius' in fixed_params_dict and isinstance(fixed_params_dict['eval_radius'], dict) and \
        'prompt' in sweep_params:
        prompt_key = sweep_params['prompt']
        if prompt_key in fixed_params_dict['eval_radius']:
            eval_radius_value = fixed_params_dict['eval_radius'][prompt_key]
            merged_config['eval_radius'] = eval_radius_value
            print(f"Prompt-specific eval_radius (prompt, radius): ({prompt_key}, {eval_radius_value})")
        else:
            print(f"Warning: No eval_radius found for prompt '{prompt_key}'. Available: {list(fixed_params_dict['eval_radius'].keys())}")

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
    sweep_params_combinations, sweep_params_dict = generate_parameter_combinations_and_sweep_params_dict(sweep_params)
    # param_combinations[0]: {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}
    # sweep_params_dict: {'prompt': {'hamburger': ['a photo of a hamburger'], 'icecream': ['a photo of an ice cream'], 'saguaro': ['a small saguaro cactus planted in a clay pot'], 'tulip': ['a photo of a tulip']}}
    
    fixed_params, fixed_params_dict = generate_fixed_param_dict(fixed_params)

    
    print(f"Experiment: {args.sweep_name}")
    print(f"Parameter combinations: {len(sweep_params_combinations)}")
    print(f"Total experiments: {len(sweep_params_combinations)}")
    print(f"Sweep parameters: {list(sweep_params.keys())}")
    print(f"Sweep parameters dict: {sweep_params_dict.keys()}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")
    print(f"Fixed parameters dict: {fixed_params_dict.keys()}")
    print(f"Setting parameters: {list(setting_params.keys())}")
    print("=" * 80)
    
    # Create base output directory
    base_output_dir = os.path.join(args.outdir, args.sweep_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save experiment summary
    summary = {
        'experiment_name': args.sweep_name,
        'parameter_combinations': len(sweep_params_combinations),
        'total_experiments': len(sweep_params_combinations),
        'sweep_parameters': list(sweep_params.keys()),
        'fixed_parameters': list(fixed_params.keys()),
        'sweep_parameters_dict': list(sweep_params_dict.keys()),
        'fixed_parameters_dict': list(fixed_params_dict.keys()),
        'setting_parameters': list(setting_params.keys()),
        'start_time': datetime.datetime.now().isoformat(),
        'combinations': []
    }
    
    # Run experiments
    successful_runs = 0
    failed_runs = 0
    
    # Run experiments for each prompt and parameter combination
    total_experiments = len(sweep_params_combinations)

    
    for i, sweep_params in enumerate(sweep_params_combinations):
        # params: {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}
        print(f"\n[{i+1}/{total_experiments}] Processing combination...")
        
        # Merge configurations
        merged_config = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params, sweep_params_dict)
        
        # Create output directory name
        # output_dir_name = "_".join(str(v) for v in sweep_params.values())
        output_dir_name = "_".join(f"{str(sweep_params[k])}" for k in sorted(sweep_params, key=lambda k: k))
        output_dir = os.path.join(base_output_dir, output_dir_name)
        

        # Record combination
        combination_info = {
            'experiment_id': i+1,
            'parameters': sweep_params,
            'output_dir': output_dir,
            'prompt': merged_config['prompt'],
            'status': 'pending'
        }
        
        summary['combinations'].append(combination_info)
        
        if args.dry_run:
            print(f"DRY RUN - Would run: {output_dir}")
            print(f"  Parameters: {sweep_params}")
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
