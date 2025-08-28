#!/usr/bin/env python3
"""
Experiment configuration module for 3D Generation experiments.
Handles loading, parsing, and generating parameter combinations from YAML configs.
"""

import yaml
import itertools
import copy
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_combinations_and_sweep_params_dict(sweep_params: Dict[str, List]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate all combinations of sweep parameters.
    
    Based on the logic from hp_ours.py - handles nested structures like lambda_repulsion
    and prompt mappings properly.
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
        
        # SAFEGUARD: For "wo" repulsion_type, skip kernel_type iteration
        if param_dict.get('repulsion_type') == 'wo' and 'kernel_type' in param_dict:
            # Remove kernel_type for "wo" case since it's not needed
            param_dict.pop('kernel_type')
            
        base_combinations.append(param_dict)

    # Expand lambda_repulsion if it is provided as a dict keyed by method+kernel
    combinations: List[Dict[str, Any]] = []
    lambda_spec = sweep_params.get('lambda_repulsion', None)
    if isinstance(lambda_spec, dict):
        for param_dict in base_combinations:
            repulsion = param_dict.get('repulsion_type')
            kernel = param_dict.get('kernel_type')
            
            # SAFEGUARD: Skip lambda_repulsion expansion for "wo" (without repulsion) case
            if repulsion == 'wo':
                # For "wo" case, no lambda_repulsion is needed, keep combination as-is
                combinations.append(param_dict)
                continue
                
            # Only expand if both keys are present and repulsion is not "wo"
            if repulsion is not None and kernel is not None and repulsion != 'wo':
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


def generate_fixed_param_dict(fixed_params: Dict[str, List]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate fixed parameters.
    
    Based on hp_ours.py logic - separates regular fixed params from dict-valued ones.
    """
    # DEFENSIVE COPY: Create a deep copy to prevent mutation of the original
    fixed_params_copy = copy.deepcopy(fixed_params)
    
    fixed_params_dict = {}
    for key, value in fixed_params_copy.items():
        if isinstance(value, dict):
            fixed_params_dict[key] = value
            
    # remove the key from fixed_params_copy (safe mutation)
    for key in fixed_params_dict.keys():
        fixed_params_copy.pop(key)
    
    return fixed_params_copy, fixed_params_dict


def merge_configs(base_config: Dict[str, Any], fixed_params: Dict[str, Any], fixed_params_dict: Dict[str, Any],
                  setting_params: Dict[str, Any], sweep_params: Dict[str, Any], sweep_params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with fixed and sweep parameters.
    
    Based on hp_ours.py logic - handles prompt mappings, lambda_repulsion, and eval_radius properly.
    """
    # Start with base config
    merged_config = base_config.copy()
    
    # Override with setting parameters
    merged_config.update(setting_params)
    
    # Override with fixed parameters
    merged_config.update(fixed_params)
    
    # Override with sweep parameters
    merged_config.update(sweep_params)
    
    # Update config with mapping parameters (e.g., prompt)
    # Define allowed mapping parameters to prevent accidental processing of other dict-valued parameters
    ALLOWED_MAPPING_PARAMS = {'prompt'}  # Add other mapping parameters here as needed
    
    for key, value in sweep_params_dict.items():
        if key in ALLOWED_MAPPING_PARAMS:
            # Process only allowed mapping parameters
            merged_config[key] = value[sweep_params[key]]
        else:
            # Skip other dict-valued parameters that shouldn't be treated as mappings
            print(f"Warning: Skipping dict-valued parameter '{key}' as it's not in allowed mapping parameters: {ALLOWED_MAPPING_PARAMS}")
    
    # Set lambda_repulsion for each method-kernel combination
    if 'lambda_repulsion' in fixed_params_dict and \
        'repulsion_type' in sweep_params and \
        'kernel_type' in sweep_params:
        
        # SAFEGUARD: Handle "wo" (without repulsion) case
        if sweep_params['repulsion_type'] == 'wo':
            # For "wo" case, set lambda_repulsion to 0 or remove it entirely
            merged_config['lambda_repulsion'] = 0
            # print(f"Baseline case (wo): lambda_repulsion set to 0")
        else:
            method_kernel_key = sweep_params['repulsion_type'] + "_" + sweep_params['kernel_type']
            method_kernel_value = fixed_params_dict['lambda_repulsion'][method_kernel_key]
            merged_config['lambda_repulsion'] = method_kernel_value
            # print(f"Method kernel (key, value): ({method_kernel_key}, {method_kernel_value})")
    
    # Set eval_radius for each prompt with default fallback
    if 'eval_radius' in fixed_params_dict and isinstance(fixed_params_dict['eval_radius'], dict) and \
        'prompt' in sweep_params:
        prompt_key = sweep_params['prompt']
        default_radius = fixed_params_dict['eval_radius'].get('default', 4.0)
        eval_radius_value = fixed_params_dict['eval_radius'].get(prompt_key, default_radius)
        merged_config['eval_radius'] = eval_radius_value
        
        # Don't print debug messages to stdout - they interfere with bash parsing
        # if prompt_key in fixed_params_dict['eval_radius']:
        #     print(f"Prompt-specific eval_radius (prompt, radius): ({prompt_key}, {eval_radius_value})")
        # else:
        #     print(f"Warning: No eval_radius found for prompt '{prompt_key}'. Using default: {default_radius}")
        #     print(f"Available keys: {list(fixed_params_dict['eval_radius'].keys())}")

    return merged_config


def create_output_dir_name(sweep_params: Dict[str, Any]) -> str:
    """Create a clean, distinguishable output directory name.
    
    Based on the naming logic from the bash script but implemented in Python.
    Avoids redundant parameter naming.
    """
    param_pairs = []
    
    # Always put key parameters first in consistent order with clear separators
    if 'repulsion_type' in sweep_params:
        rt = sweep_params['repulsion_type']
        if rt == 'svgd':
            param_pairs.append('SVGD')
        elif rt == 'rlsd':
            param_pairs.append('RLSD')
        elif rt == 'wo':
            param_pairs.append('WO')
        else:
            param_pairs.append(f'RT_{rt}')
    
    if 'kernel_type' in sweep_params:
        kt = sweep_params['kernel_type']
        if kt == 'rbf':
            param_pairs.append('RBF')
        elif kt == 'cosine':
            param_pairs.append('COS')
        else:
            param_pairs.append(f'KT_{kt}')
    
    # Add lambda_repulsion parameter with clear formatting (only if not already processed)
    # Check if lambda_repulsion is a sweep parameter that needs to be included in naming
    if 'lambda_repulsion' in sweep_params:
        lambda_value = sweep_params['lambda_repulsion']
        # Format lambda value with K for thousands
        if lambda_value >= 1000:
            lambda_str = f'{lambda_value//1000}K'
        else:
            lambda_str = str(lambda_value)
        param_pairs.append(f'λ{lambda_str}')
    
    # Add prompt with smart abbreviation
    for k, v in sweep_params.items():
        if k.startswith('prompt_') or k == 'prompt':
            if k == 'prompt':
                prompt_key = v
            else:
                prompt_key = k.replace('prompt_', '')
            
            # Use smart abbreviations for prompts
            if prompt_key == 'hamburger':
                param_pairs.append('HAMB')
            elif prompt_key == 'icecream':
                param_pairs.append('ICE')
            elif prompt_key == 'cactus':
                param_pairs.append('CACT')
            elif prompt_key == 'tulip':
                param_pairs.append('TUL')
            else:
                param_pairs.append(prompt_key[:4].upper())
    
    # Add seed at the end
    if 'seed' in sweep_params:
        param_pairs.append(f'S{sweep_params["seed"]}')
    
    # Add other parameters (excluding already processed ones and lambda_repulsion to avoid redundancy)
    for k, v in sweep_params.items():
        if (k not in ['repulsion_type', 'kernel_type', 'seed', 'lambda_repulsion'] and 
            not k.startswith('lambda_repulsion_') and 
            not k.startswith('prompt_') and 
            k != 'prompt'):
            param_pairs.append(f'{k}_{v}')
    
    return '__'.join(param_pairs)


def get_experiment_configs(base_config_path: str, sweep_config_path: str, experiment_name: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Get all experiment configurations for a given experiment name.
    
    Returns:
        - List of merged configs (ready to use with main_ours.py)
        - List of output directory names
    """
    # Load configurations
    base_config = load_yaml_config(base_config_path)
    sweep_config = load_yaml_config(sweep_config_path)
    
    # Get experiment configuration
    if experiment_name not in sweep_config:
        raise ValueError(f"Experiment '{experiment_name}' not found in sweep config. Available: {list(sweep_config.keys())}")
    
    exp_config = sweep_config[experiment_name]
    
    # Extract parameters
    sweep_params = exp_config.get('sweep_parameters', {})
    fixed_params = exp_config.get('fixed_parameters', {})
    setting_params = exp_config.get('settings', {})
    
    # Generate all parameter combinations
    sweep_params_combinations, sweep_params_dict = generate_parameter_combinations_and_sweep_params_dict(sweep_params)
    fixed_params, fixed_params_dict = generate_fixed_param_dict(fixed_params)
    
    # Generate merged configs and output dir names
    configs = []
    output_dirs = []
    
    for sweep_params_combo in sweep_params_combinations:
        # Merge configurations
        merged_config = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params_combo, sweep_params_dict)
        
        # Create output directory name
        output_dir_name = create_output_dir_name(sweep_params_combo)
        
        configs.append(merged_config)
        output_dirs.append(output_dir_name)
    
    return configs, output_dirs


def save_experiment_configs_to_files(base_config_path: str, sweep_config_path: str, experiment_name: str, output_base_dir: str = "exp"):
    """Save each experiment configuration as a separate YAML file in exp/{experiment_name}/{output_dir_name}/ directory.
    
    This function saves individual config files in each experiment's output directory instead of a central location.
    """
    try:
        configs, output_dirs = get_experiment_configs(base_config_path, sweep_config_path, experiment_name)
        
        # Create base experiment directory
        exp_dir = Path(output_base_dir) / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each config as a separate YAML file in its respective output directory
        config_paths = []
        for i, (config, output_dir_name) in enumerate(zip(configs, output_dirs)):
            # Create the individual experiment output directory
            output_dir = exp_dir / output_dir_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config as config.yaml in the experiment output directory
            config_path = output_dir / "config.yaml"
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            config_paths.append(str(config_path))
            print(f"Saved config {i+1}/{len(configs)}: {config_path}")
        
        # Save a summary file with all output directory names and config paths
        summary_path = exp_dir / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Total configurations: {len(configs)}\n")
            f.write(f"Generated at: {Path.cwd()}\n\n")
            f.write("Output directories and config files:\n")
            for i, (output_dir_name, config_path) in enumerate(zip(output_dirs, config_paths)):
                f.write(f"{i+1:03d}: {output_dir_name} -> {config_path}\n")
        
        print(f"\nSaved {len(configs)} configuration files in experiment directories under: {exp_dir}")
        print(f"Summary file: {summary_path}")
        
        return len(configs), output_dirs, config_paths
        
    except Exception as e:
        # Use stderr for error messages to avoid pipe issues
        import sys
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def print_experiment_configs(base_config_path: str, sweep_config_path: str, experiment_name: str):
    """Print experiment configurations in a simple format for bash parsing.
    
    Format:
    CONFIG_COUNT
    OUTPUT_DIR_1
    OUTPUT_DIR_2
    ...
    ---
    CONFIG_1_YAML
    ---
    CONFIG_2_YAML
    ...
    """
    try:
        configs, output_dirs = get_experiment_configs(base_config_path, sweep_config_path, experiment_name)
        
        # Print count
        print(len(configs))
        
        # Print output directories
        for output_dir in output_dirs:
            print(output_dir)
        
        # Print separator
        print("---")
        
        # Print each config as YAML
        for config in configs:
            print(yaml.dump(config, default_flow_style=False, indent=2))
            print("---")
            
    except Exception as e:
        # Use stderr for error messages to avoid pipe issues
        import sys
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def save_config_to_file(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    # Main execution mode for bash script integration
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("Usage: python exp_config.py <base_config> <sweep_config> <experiment_name> [--save-files]")
        print("  --save-files: Save each config as separate YAML file in exp/{experiment_name}/")
        sys.exit(1)
    
    base_config_path = sys.argv[1]
    sweep_config_path = sys.argv[2]
    experiment_name = sys.argv[3]
    
    # Check if --save-files flag is provided
    save_files = len(sys.argv) > 4 and sys.argv[4] == "--save-files"
    
    try:
        if save_files:
            # Save configs to files instead of printing to stdout
            save_experiment_configs_to_files(base_config_path, sweep_config_path, experiment_name)
        else:
            # Use the print_experiment_configs function for bash script integration
            print_experiment_configs(base_config_path, sweep_config_path, experiment_name)
        
    except Exception as e:
        # Use stderr for error messages to avoid pipe issues
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
