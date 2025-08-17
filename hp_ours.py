# hyperparameter_tuning.py

import subprocess
import os
import time
from copy import deepcopy
import itertools
from typing import Dict, List, Any
import hashlib
import json

import torch
import numpy as np
import yaml
from omegaconf import OmegaConf

from main_ours import GUI

def parse_yaml_sweeps(yaml_file_path: str) -> Dict[str, Dict]:
    """Parse YAML file containing multiple experiment configurations."""
    with open(yaml_file_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    experiments = {}
    
    if not config_data:
        print(f"[ERROR] No data found in {yaml_file_path}")
        return experiments
    
    # Parse each experiment configuration
    for exp_key, exp_config in config_data.items():
        # Skip non-experiment entries (comments, etc.)
        if not isinstance(exp_config, dict) or not (exp_key.startswith('exp') or exp_key.startswith('debug_')):
            continue
            
        if 'name' not in exp_config:
            print(f"[WARNING] Experiment {exp_key} missing 'name' field, skipping")
            continue
            
        exp_name = exp_config['name']
        experiments[exp_name] = {
            'exp_id': exp_key,
            'name': exp_name,
            'method': exp_config.get('method', 'grid'),
            'sweep_parameters': exp_config.get('sweep_parameters', {}),
            'fixed_parameters': exp_config.get('fixed_parameters', {}),
            'settings': exp_config.get('settings', {})
        }
        
        print(f"[INFO] Loaded experiment: {exp_key} -> {exp_name}")
    
    return experiments

def generate_parameter_combinations(exp_config: Dict) -> List[Dict]:
    """Generate all parameter combinations from an experiment configuration."""
    if exp_config['method'] != 'grid':
        raise ValueError(f"Only 'grid' method is supported, got {exp_config['method']}")
    
    sweep_parameters = exp_config['sweep_parameters']
    fixed_parameters = exp_config['fixed_parameters']
    
    # Combine sweep and fixed parameters for full grid search
    # This allows fixed parameters to also have multiple values (like multiple seeds)
    all_param_names = list(sweep_parameters.keys()) + list(fixed_parameters.keys())
    all_param_values = (
        [sweep_parameters[name] for name in sweep_parameters.keys()] +
        [fixed_parameters[name] for name in fixed_parameters.keys()]
    )
    
    # Generate cartesian product of all parameter combinations
    combinations = []
    for combination in itertools.product(*all_param_values):
        param_dict = dict(zip(all_param_names, combination))
        combinations.append(param_dict)
    
    return combinations

def run_hyperparameter_sweeping(base_opt, sweep_name: str = None, yaml_path: str = None, 
                                prompts: List[str] = None):
    """Run hyperparameter tuning experiments with YAML-defined parameter combinations."""
    
    opt = deepcopy(base_opt)
    
    # Default YAML path if not provided
    if yaml_path is None:
        yaml_path = "configs/text_ours_hp.yaml"
    
    # Parse YAML experiments
    print(f"[INFO] Loading experiment configurations from {yaml_path}")
    experiments = parse_yaml_sweeps(yaml_path)
    
    if not experiments:
        print(f"[ERROR] No experiments found in {yaml_path}")
        return
    
    print(f"[INFO] Available experiments: {list(experiments.keys())}")
    
    # Select experiment
    if sweep_name is None:
        print("[ERROR] Please specify an experiment name with --sweep_name")
        print(f"Available options: {list(experiments.keys())}")
        return
    
    if sweep_name not in experiments:
        print(f"[ERROR] Experiment '{sweep_name}' not found. Available: {list(experiments.keys())}")
        return
    
    selected_experiment = experiments[sweep_name]
    print(f"[INFO] Running experiment: {sweep_name} (ID: {selected_experiment['exp_id']})")
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(selected_experiment)
    print(f"[INFO] Generated {len(param_combinations)} parameter combinations")
    
    # Use provided prompts or defaults (seeds are now handled in YAML config)
    PROMPTS = prompts if prompts is not None else ["a photo of a hamburger"]
    
    # Apply settings from experiment configuration
    settings = selected_experiment['settings']
    for setting_name, setting_value in settings.items():
        if hasattr(opt, setting_name):
            setattr(opt, setting_name, setting_value)
            print(f"[INFO] Applied setting: {setting_name} = {setting_value}")
        else:
            print(f"[WARNING] Setting '{setting_name}' not found in opt, skipping")
    
    job_dir = opt.outdir
    
    # Calculate total experiments (seeds are now in parameter combinations)
    total_experiments = len(PROMPTS) * len(param_combinations)
    
    print(f"[INFO] Starting hyperparameter tuning...")
    print(f"[INFO] Will run {total_experiments} experiments total")
    print(f"[INFO] {len(PROMPTS)} prompts x {len(param_combinations)} param combinations")
    print(f"[INFO] Experiment: {sweep_name}")
    print(f"--------------------------------")
    
    experiment_count = 0
    
    # ---RUN EXPERIMENTS---
    for prompt in PROMPTS:
        print(f"\n[INFO] Running experiments for prompt: '{prompt}'")
        opt.prompt = prompt
        prompt_clean = prompt.replace(" ", "_")
        
        # Run each parameter combination
        for param_idx, params in enumerate(param_combinations):
            experiment_count += 1
            print(f"\n[INFO] Experiment {experiment_count}/{total_experiments}")
            print(f"[INFO] Parameters: {params}")
            
            # Apply parameters to opt (including seed from YAML)
            for param_name, param_value in params.items():
                if hasattr(opt, param_name):
                    setattr(opt, param_name, param_value)
                else:
                    print(f"[WARNING] Parameter '{param_name}' not found in opt, skipping")
            
            # Get seed from parameters for directory naming
            seed = params.get('seed', 42)
            
            # Get SLURM job ID from environment or use timestamp as fallback
            job_id = os.environ.get('SLURM_JOB_ID', f"local_{int(time.time())}")
            
            # Create dynamic run name based on sweep parameters
            sweep_params = {k: v for k, v in params.items() if k in selected_experiment['sweep_parameters']}
            
            # Create abbreviated parameter string from sweep parameters
            param_parts = []
            for key, value in sweep_params.items():
                # Create short abbreviations for common parameter names
                abbreviations = {
                    'repulsion_type': 'rep',
                    'lambda_repulsion': 'lr',
                    'feature_layer': 'fl',
                    'num_particles': 'np',
                    'num_views': 'nv',
                    'learning_rate': 'lr',
                    'batch_size': 'bs',
                    'kernel_type': 'kt'
                }
                abbrev = abbreviations.get(key, key[:3])  # Use abbreviation or first 3 chars
                param_parts.append(f"{abbrev}{value}")
            
            param_string = "_".join(param_parts) if param_parts else "default"
            run_name = f"run_{experiment_count:03d}_{param_string}_s{seed}"
            
            # New directory structure: sweep_name/run_sweep_parameters/
            sweep_dir = os.path.join(job_dir, sweep_name)
            run_dir = os.path.join(sweep_dir, run_name)
            
            # Set output directory and save path
            opt.outdir = run_dir
            opt.save_path = os.path.join(sweep_name, run_name)
            
            # Save parameter details to a config file for reference
            param_config = {
                'experiment_id': experiment_count,
                'job_id': job_id,
                'exp_id': selected_experiment['exp_id'],
                'sweep_name': sweep_name,
                'prompt': prompt,
                'seed': seed,
                'all_parameters': params,
                'sweep_parameters': {k: v for k, v in params.items() if k in selected_experiment['sweep_parameters']},
                'fixed_parameters': {k: v for k, v in params.items() if k in selected_experiment['fixed_parameters']},
                'settings': selected_experiment['settings'],
                'directory_structure': {
                    'sweep_dir': sweep_name,
                    'run_dir': run_name,
                    'full_path': opt.save_path
                }
            }
            
            # Create output directory
            os.makedirs(opt.outdir, exist_ok=True)
            
            # Save parameter configuration for reference
            config_file = os.path.join(opt.outdir, 'experiment_config.json')
            with open(config_file, 'w') as f:
                json.dump(param_config, f, indent=2)
            
            print(f"[INFO] Experiment directory: {sweep_name}/{run_name}")
            print(f"[INFO] Full parameters saved to: experiment_config.json")
            
            # Run experiment
            try:
                gui = GUI(opt)
                gui.train(opt.iters)
                
                # More aggressive memory cleanup
                del gui
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                
                print(f"[INFO] Completed experiment {experiment_count}/{total_experiments}")
                
            except Exception as e:
                # Print full traceback with line numbers to stderr (captured by SLURM error.err)
                import sys, traceback
                print(f"[ERROR] Experiment failed: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                # More aggressive memory cleanup on failure
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection
                import gc
                gc.collect()
                continue
    
    print(f"\n[INFO] Hyperparameter tuning completed!")
    print(f"[INFO] Ran {experiment_count} experiments")
    print(f"[INFO] Results saved in {job_dir}/{sweep_name}/ organized by run parameters")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="YAML-based hyperparameter sweeping")
    parser.add_argument("--config", default="configs/text_ours.yaml", 
                       help="path to the base yaml config file with default values")
    parser.add_argument("--sweep_config", default="configs/text_ours_hp.yaml", 
                       help="path to the sweep configuration yaml file with parameter grids")
    parser.add_argument("--sweep_name", required=True, 
                       help="name of the experiment configuration to run (from sweep_config)")
    parser.add_argument("--prompts", nargs="+", default=["a photo of a hamburger"],
                       help="list of prompts to test")
    
    args, extras = parser.parse_known_args()

    # Fix prompt parsing: if multiple words are passed, join them into a single prompt
    # This handles cases where --prompts a photo of a hamburger becomes ['a', 'photo', 'of', 'a', 'hamburger']
    if len(args.prompts) > 1 and all(isinstance(p, str) and ' ' not in p for p in args.prompts):
        # Join words into a single prompt
        args.prompts = [' '.join(args.prompts)]

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print(f"[INFO] Base config loaded from {args.config}")
    print(f"[INFO] Sweep config: {args.sweep_config}")
    print(f"[INFO] Requested sweep: {args.sweep_name}")
    print(f"[INFO] Prompts: {args.prompts}")

    # Run hyperparameter tuning with YAML configuration
    run_hyperparameter_sweeping(opt, sweep_name=args.sweep_name, yaml_path=args.sweep_config,
                                prompts=args.prompts)
