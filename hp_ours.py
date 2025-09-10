#!/usr/bin/env python3
"""
Hyperparameter tuning script for 3D Generation experiments.
Reads experiment configurations from YAML files and runs grid search experiments.

IMPORTANT CHANGES (GPU Resource Management):
- Replaced subprocess execution with direct GUI import from main_ours
- Added explicit GPU device initialization for each experiment
- Implemented comprehensive GPU cleanup after each experiment
- Added GPU resource monitoring and logging
- Each experiment now properly initializes its own GPU context

This ensures that GPU resources are properly managed and cleaned up between experiments,
preventing memory leaks and resource conflicts in hyperparameter tuning runs.
"""

import argparse
import itertools
import os
import sys
import yaml
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import datetime
import torch
import gc

# Import the GUI class directly instead of using subprocess
from main_ours import GUI
from omegaconf import OmegaConf


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_combinations_and_sweep_params_dict(sweep_params: Dict[str, List]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
    # combinations: [{'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}, {'seed': 42, 'prompt': 'hamburger', 'repulsion_type': 'svgd', 'kernel_type': 'rbf', 'lambda_repulsion': 600}]
    # sweep_params_dict: {'prompt': {'hamburger': 'a photo of a hamburger', 'icecream': 'a photo of an ice cream', 'saguaro': 'a small saguaro cactus planted in a clay pot', 'tulip': 'a photo of a tulip'}}

def generate_fixed_param_dict(fixed_params: Dict[str, List]) -> Dict[str, Any]:
    """Generate fixed parameters."""

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
    
    # TODO✅ Update config with mapping parameters (e.g., prompt)
    # Define allowed mapping parameters to prevent accidental processing of other dict-valued parameters
    ALLOWED_MAPPING_PARAMS = {'prompt'}  # Add other mapping parameters here as needed
    
    for key, value in sweep_params_dict.items():
        if key in ALLOWED_MAPPING_PARAMS:
            # Process only allowed mapping parameters
            merged_config[key] = value[sweep_params[key]] # merged_config['prompt'] = a photo of a hamburger
        else:
            # Skip other dict-valued parameters that shouldn't be treated as mappings
            print(f"Warning: Skipping dict-valued parameter '{key}' as it's not in allowed mapping parameters: {ALLOWED_MAPPING_PARAMS}")
    
    
    # TODO✅ set lambda_repulsion for each method-kernel combination
    if 'lambda_repulsion' in fixed_params_dict and \
        'repulsion_type' in sweep_params and \
        'kernel_type' in sweep_params:
        
        # SAFEGUARD: Handle "wo" (without repulsion) case
        if sweep_params['repulsion_type'] == 'wo':
            # For "wo" case, set lambda_repulsion to 0 or remove it entirely
            merged_config['lambda_repulsion'] = 0
            print(f"Baseline case (wo): lambda_repulsion set to 0")
        else:
            method_kernel_key = sweep_params['repulsion_type'] + "_" + sweep_params['kernel_type']
            method_kernel_value = fixed_params_dict['lambda_repulsion'][method_kernel_key]
            merged_config['lambda_repulsion'] = method_kernel_value
            print(f"Method kernel (key, value): ({method_kernel_key}, {method_kernel_value})")
    
    # TODO✅ set eval_radius for each prompt with default fallback
    if 'eval_radius' in fixed_params_dict and isinstance(fixed_params_dict['eval_radius'], dict) and \
        'prompt' in sweep_params:
        prompt_key = sweep_params['prompt']
        default_radius = fixed_params_dict['eval_radius'].get('default', 4.0)
        eval_radius_value = fixed_params_dict['eval_radius'].get(prompt_key, default_radius)
        merged_config['eval_radius'] = eval_radius_value
        
        if prompt_key in fixed_params_dict['eval_radius']:
            print(f"Prompt-specific eval_radius (prompt, radius): ({prompt_key}, {eval_radius_value})")
        else:
            print(f"Warning: No eval_radius found for prompt '{prompt_key}'. Using default: {default_radius}")
            print(f"Available keys: {list(fixed_params_dict['eval_radius'].keys())}")

    return merged_config


def save_config_to_file(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs."""
    gpu_info = {
        'available_gpus': [],
        'total_memory': {},
        'free_memory': {},
        'gpu_names': {}
    }
    
    try:
        import subprocess
        # Get GPU information using nvidia-smi
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL
        )
        
        for line in nvidia_smi_output.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    gpu_name = parts[1]
                    total_memory = int(parts[2])
                    free_memory = int(parts[3])
                    
                    gpu_info['available_gpus'].append(gpu_id)
                    gpu_info['total_memory'][gpu_id] = total_memory
                    gpu_info['free_memory'][gpu_id] = free_memory
                    gpu_info['gpu_names'][gpu_id] = gpu_name
                    
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Fallback: try to get basic GPU info
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info['available_gpus'] = list(range(gpu_count))
                for i in range(gpu_count):
                    gpu_info['gpu_names'][i] = torch.cuda.get_device_name(i)
        except ImportError:
            pass
    
    return gpu_info


def verify_gpu_availability(gpu_id: int) -> bool:
    """Verify that the specified GPU is available and accessible."""
    if not torch.cuda.is_available():
        print(f"❌ CUDA is not available on this system")
        return False
    
    if gpu_id >= torch.cuda.device_count():
        print(f"❌ GPU {gpu_id} does not exist. Available GPUs: {list(range(torch.cuda.device_count()))}")
        return False
    
    try:
        # Test if we can access the GPU
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        print(f"✅ GPU {gpu_id} is available and accessible: {torch.cuda.get_device_name(gpu_id)}")
        return True
    except Exception as e:
        print(f"❌ Failed to access GPU {gpu_id}: {e}")
        return False


def select_gpu_for_experiment(gpu_info: Dict[str, Any], experiment_id: int) -> int:
    """
    Select a GPU for the experiment.
    Currently uses round-robin assignment for sequential execution.
    """
    if not gpu_info['available_gpus']:
        return 0  # Fallback
    
    # Simple round-robin assignment
    selected_gpu = gpu_info['available_gpus'][experiment_id % len(gpu_info['available_gpus'])]
    return selected_gpu


def collect_run_metadata() -> Dict[str, Any]:
    """Collect comprehensive metadata for the current run."""
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
        'user': os.getenv('USER', 'unknown'),
    }
    
    # Git information
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                           cwd=os.getcwd(), text=True, stderr=subprocess.DEVNULL).strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                           cwd=os.getcwd(), text=True, stderr=subprocess.DEVNULL).strip()
        metadata['git'] = {
            'commit': git_commit,
            'branch': git_branch
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata['git'] = {'commit': 'unknown', 'branch': 'unknown'}
    
    # Python and package versions
    try:
        import torch
        metadata['versions'] = {
            'python': sys.version,
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'
        }
    except ImportError:
        metadata['versions'] = {'python': sys.version, 'torch': 'N/A', 'cuda': 'N/A', 'cudnn': 'N/A'}
    
    # GPU information
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.DEVNULL)
        metadata['nvidia_smi'] = nvidia_smi
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata['nvidia_smi'] = 'N/A'
    
    # Partial pip freeze (top packages)
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        # Take first 20 lines to avoid huge output
        metadata['pip_freeze'] = '\n'.join(pip_freeze.split('\n')[:20]) + '\n... (truncated)'
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata['pip_freeze'] = 'N/A'
    
    return metadata

# run_single_experiment(merged_config, output_dir, args.config, args.sweep_name)
def run_single_experiment(
    config: Dict[str, Any], 
    output_dir: str, 
    base_config_path: str, 
    exp_name: str,
    gpu_id: int = 0
    ) -> Dict[str, Any]:
    """Run a single experiment with the given configuration."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figures subdirectory for analysis plots
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Set output directory in config
    config['outdir'] = output_dir
    
    # Save configuration to output directory
    config_path = os.path.join(output_dir, "config.yaml")
    save_config_to_file(config, config_path)
    

    print(f"Running experiment: {output_dir}")
    print(f"GPU ID: {gpu_id}")
    print("-" * 80)
    
    # Collect metadata before running
    metadata = collect_run_metadata()
    metadata['start_time'] = datetime.datetime.now().isoformat()
    metadata['gpu_id'] = gpu_id
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "run_metadata.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, indent=2)
    
    # Set GPU device for this experiment
    if torch.cuda.is_available():
        print(f"🖥️  Available GPUs: {torch.cuda.device_count()}")
        print(f"🖥️  Current GPU: {torch.cuda.current_device()}")
        print(f"🖥️  Target GPU: {gpu_id}")
        
        # Verify GPU availability before starting
        if not verify_gpu_availability(gpu_id):
            print(f"⚠️  Warning: GPU {gpu_id} verification failed, but continuing...")
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
    try:
        # Create OmegaConf object from config dict
        opt = OmegaConf.create(config)
        
        # Add GPU ID to the config for the GUI to use
        opt.gpu_id = gpu_id
        
        # Initialize GUI and run training
        print(f"🚀 Starting training with {opt.iters} iterations...")
        gui = GUI(opt)
        
        # Explicitly set GPU device in GUI
        if torch.cuda.is_available():
            if not gui.set_gpu_device(gpu_id):
                print(f"⚠️  Warning: Failed to set GPU {gpu_id}, experiment may use different GPU")
        
        # Run training
        print(f"🎯 Starting training on device: {gui.device}")
        gui.train(opt.iters)
        
        # Calculate duration
        end_time = datetime.datetime.now()
        duration_sec = (end_time - datetime.datetime.fromisoformat(metadata['start_time'])).total_seconds()
        
        # Update metadata with results
        metadata['end_time'] = end_time.isoformat()
        metadata['duration_sec'] = duration_sec
        metadata['return_code'] = 0  # Success
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)
        
        print(f"✅ Experiment completed successfully: {output_dir} (duration: {duration_sec:.1f}s)")
        
        # Create .done marker to prevent duplicate runs
        done_flag = os.path.join(output_dir, ".done")
        Path(done_flag).write_text(end_time.isoformat())
        
        # Cleanup GPU resources after successful completion
        if torch.cuda.is_available():
            print(f"🧹 Cleaning up GPU {gpu_id} resources...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        return {
            'success': True,
            'return_code': 0,
            'start_time': metadata['start_time'],
            'end_time': metadata['end_time'],
            'duration_sec': duration_sec
        }
            
    except Exception as e:
        print(f"❌ Exception during experiment: {output_dir}")
        print(f"Error: {str(e)}")
        
        # Calculate duration even for failed experiments
        end_time = datetime.datetime.now()
        duration_sec = (end_time - datetime.datetime.fromisoformat(metadata['start_time'])).total_seconds()
        
        # Update metadata with error results
        metadata['end_time'] = end_time.isoformat()
        metadata['duration_sec'] = duration_sec
        metadata['return_code'] = -1
        metadata['error'] = str(e)
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)
        
        # Cleanup GPU resources even after failure
        if torch.cuda.is_available():
            print(f"🧹 Cleaning up GPU {gpu_id} resources after failure...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        return {
            'success': False,
            'return_code': -1,
            'start_time': metadata['start_time'],
            'end_time': metadata['end_time'],
            'duration_sec': duration_sec,
            'error': str(e)
        }


def load_existing_summary(base_output_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load existing experiment summary and categorize combinations by status."""
    summary_path = os.path.join(base_output_dir, "experiment_summary.yaml")
    
    if not os.path.exists(summary_path):
        return None, [], [], []
    
    try:
        with open(summary_path, 'r') as f:
            summary = yaml.safe_load(f)
        
        # Categorize combinations by status
        failed_combinations = []
        successful_combinations = []
        pending_combinations = []
        
        for combo in summary.get('combinations', []):
            status = combo.get('status', 'pending')
            if status == 'failed':
                failed_combinations.append(combo)
            elif status in ['success', 'skipped']:
                successful_combinations.append(combo)
            else:
                pending_combinations.append(combo)
        
        print(f"📊 Loaded existing summary:")
        print(f"  - Failed: {len(failed_combinations)}")
        print(f"  - Successful: {len(successful_combinations)}")
        print(f"  - Pending: {len(pending_combinations)}")
        
        return summary, failed_combinations, successful_combinations, pending_combinations
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing summary: {e}")
        return None, [], [], []


def create_combination_info(sweep_params: Dict[str, Any], merged_config: Dict[str, Any], 
                          output_dir: str, experiment_id: int) -> Dict[str, Any]:
    """Create combination info dictionary."""
    return {
        'experiment_id': experiment_id,
        'parameters': sweep_params,
        'output_dir': output_dir,
        'prompt': merged_config.get('prompt', ''),
        'status': 'pending'
    }


def categorize_combinations(sweep_params_combinations: List[Dict[str, Any]], 
                          base_config: Dict[str, Any], fixed_params: Dict[str, Any], 
                          fixed_params_dict: Dict[str, Any], setting_params: Dict[str, Any], 
                          sweep_params_dict: Dict[str, Any], base_output_dir: str,
                          existing_summary: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Categorize combinations into failed, pending, and successful based on existing results."""
    
    failed_combinations = []
    pending_combinations = []
    successful_combinations = []
    
    # Create a mapping from parameters to existing results if available
    existing_results = {}
    if existing_summary:
        for combo in existing_summary.get('combinations', []):
            param_key = "_".join(f"{k}={str(combo['parameters'][k])}" for k in sorted(combo['parameters'], key=lambda k: k))
            existing_results[param_key] = combo
    
    for i, sweep_params in enumerate(sweep_params_combinations):
        # Merge configurations
        merged_config = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params, sweep_params_dict)
        
        # Create output directory name with key=value pairs for easy filtering
        # Format: prompt=hamburger_repulsion_type=svgd_kernel_type=cosine_lambda_repulsion=1000_seed=42
        # This allows easy filtering: ls exp/*/kernel_type=cosine* or find exp/ -name "*kernel_type=cosine*"
        output_dir_name = "_".join(f"{k}={str(sweep_params[k])}" for k in sorted(sweep_params, key=lambda k: k))
        output_dir = os.path.join(base_output_dir, output_dir_name)
        
        # Create combination info
        combination_info = create_combination_info(sweep_params, merged_config, output_dir, i+1)
        
        # Check if this combination already exists
        if output_dir_name in existing_results:
            existing_combo = existing_results[output_dir_name]
            status = existing_combo.get('status', 'pending')
            
            if status == 'failed':
                # Update with current info but keep failed status
                combination_info.update(existing_combo)
                failed_combinations.append(combination_info)
            elif status in ['success', 'skipped']:
                # Update with current info but keep success status
                combination_info.update(existing_combo)
                successful_combinations.append(combination_info)
            else:
                pending_combinations.append(combination_info)
        else:
            # Check if experiment directory exists and has .done flag
            done_flag = os.path.join(output_dir, ".done")
            if os.path.exists(done_flag):
                combination_info['status'] = 'skipped'
                successful_combinations.append(combination_info)
            else:
                pending_combinations.append(combination_info)
    
    return failed_combinations, pending_combinations, successful_combinations


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
    parser.add_argument("--no_resume", action="store_true",
                       help="Disable resume functionality and start fresh (default: resume enabled)")
    parser.add_argument("--retry_failed_only", action="store_true",
                       help="Only retry failed experiments, skip pending ones")
    
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
    
    # Initialize GPU management for sequential execution
    gpu_info = get_gpu_info()
    print(f"GPU Management: {len(gpu_info['available_gpus'])} GPUs available")
    for gpu_id in gpu_info['available_gpus']:
        gpu_name = gpu_info['gpu_names'].get(gpu_id, 'Unknown')
        total_memory = gpu_info['total_memory'].get(gpu_id, 'Unknown')
        print(f"  GPU {gpu_id}: {gpu_name} ({total_memory}MB total)")
    print("Execution Strategy: Sequential (one experiment per GPU at a time)")
    print("=" * 80)
    
    # Create base output directory
    base_output_dir = os.path.join(args.outdir, args.sweep_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load existing summary and categorize combinations
    # Resume is enabled by default, only disable if --no_resume flag is used
    if args.no_resume:
        existing_summary = None
        print(f"🆕 Starting fresh (--no_resume flag used)")
    else:
        existing_summary, existing_failed, existing_successful, existing_pending = load_existing_summary(base_output_dir)
        if existing_summary:
            print(f"🔄 Resume mode enabled (default behavior)")
        else:
            print(f"🆕 No existing results found, starting fresh")
    
    # Categorize all combinations based on existing results
    failed_combinations, pending_combinations, successful_combinations = categorize_combinations(
        sweep_params_combinations, base_config, fixed_params, fixed_params_dict, 
        setting_params, sweep_params_dict, base_output_dir, existing_summary
    )
    
    # Determine which combinations to process
    if args.retry_failed_only:
        all_combinations_to_process = failed_combinations
        print(f"🔄 Retrying failed experiments only (--retry_failed_only flag used)")
        print(f"  - Failed combinations to retry: {len(failed_combinations)}")
        if len(failed_combinations) == 0:
            print(f"  ✅ No failed experiments to retry!")
            return
    else:
        # Prioritize failed combinations first, then pending
        all_combinations_to_process = failed_combinations + pending_combinations
        if not existing_summary:
            print(f"🆕 No existing summary found. Starting with all {len(sweep_params_combinations)} combinations.")
        else:
            print(f"🔄 Resuming experiment:")
            print(f"  - Failed combinations to retry: {len(failed_combinations)}")
            print(f"  - Pending combinations: {len(pending_combinations)}")
            print(f"  - Already successful: {len(successful_combinations)}")
    
    if len(all_combinations_to_process) == 0:
        print(f"✅ No experiments to process! All combinations are already successful.")
        return
    
    # Initialize or load summary
    if existing_summary:
        summary = existing_summary
        # Update start time if this is a resume
        if 'resume_time' not in summary:
            summary['resume_time'] = datetime.datetime.now().isoformat()
    else:
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
    
    # Initialize counters
    successful_runs = len(successful_combinations)
    failed_runs = 0
    
    # Process combinations in order of priority (failed, then pending)
    total_experiments_to_run = len(all_combinations_to_process)
    
    print(f"\n🚀 Starting execution of {total_experiments_to_run} experiments...")
    print(f"Priority order: Failed ({len(failed_combinations)}) → Pending ({len(pending_combinations)})")
    
    for i, combination_info in enumerate(all_combinations_to_process):
        sweep_params = combination_info['parameters']
        output_dir = combination_info['output_dir']
        status = combination_info.get('status', 'pending')
        
        print(f"\n[{i+1}/{total_experiments_to_run}] Processing combination...")
        print(f"  Status: {status}")
        print(f"  Parameters: {sweep_params}")
        print(f"  Output: {output_dir}")
        
        # Merge configurations
        merged_config = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params, sweep_params_dict)
        
        # Add combination to summary if not already present
        if combination_info not in summary['combinations']:
            summary['combinations'].append(combination_info)
        
        if args.dry_run:
            print(f"DRY RUN - Would run: {output_dir}")
            continue
        
        # Check for duplicate run prevention
        done_flag = os.path.join(output_dir, ".done")
        if os.path.exists(done_flag) and status != 'failed':
            print(f"↪ skip: already done ({output_dir})")
            combination_info['status'] = 'skipped'
            continue
        
        # Select GPU for this experiment (round-robin assignment)
        selected_gpu = select_gpu_for_experiment(gpu_info, i)
        print(f"🖥️  Using GPU {selected_gpu} for experiment {i+1}")
        
        # Run experiment
        result = run_single_experiment(merged_config, output_dir, args.config, args.sweep_name, selected_gpu)
        
        # Update combination info with detailed results
        combination_info.update({
            'status': 'success' if result['success'] else 'failed',
            'return_code': result['return_code'],
            'start_time': result['start_time'],
            'end_time': result['end_time'],
            'duration_sec': result['duration_sec']
        })
        
        if result['success']:
            successful_runs += 1
        else:
            failed_runs += 1
            if 'error' in result:
                combination_info['error'] = result['error']
        
        # Save updated summary after each experiment
        summary['end_time'] = datetime.datetime.now().isoformat()
        summary['successful_runs'] = successful_runs
        summary['failed_runs'] = failed_runs
        
        summary_path = os.path.join(base_output_dir, "experiment_summary.yaml")
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    # Final summary
    total_processed = len(all_combinations_to_process)
    total_all = len(sweep_params_combinations)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Experiment: {args.sweep_name}")
    print(f"Total experiments processed: {total_processed}")
    print(f"Total experiments overall: {total_all}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate (processed): {successful_runs/total_processed*100:.1f}%")
    print(f"Success rate (overall): {successful_runs/total_all*100:.1f}%")
    print(f"Results saved in: {base_output_dir}")
    print(f"Summary file: {os.path.join(base_output_dir, 'experiment_summary.yaml')}")
    
    if failed_runs > 0:
        print(f"\n⚠️  {failed_runs} experiments failed. You can rerun this script to retry failed experiments.")
        print(f"   The script will automatically prioritize failed combinations first.")
    else:
        print(f"\n✅ All experiments completed successfully!")


if __name__ == "__main__":
    main()
