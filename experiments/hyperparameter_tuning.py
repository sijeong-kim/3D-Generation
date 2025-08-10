# hyperparameter_tuning.py

import subprocess
import os
from copy import deepcopy

import torch
import numpy as np

from main import seed_everything, GUI

def run_hyperparameter_tuning(base_opt):
    """Run hyperparameter tuning experiments with different parameter combinations."""
    
    opt = deepcopy(base_opt)
    
    # Define hyperparameter grid
    # LAMBDAS = [1, 10, 100, 1000, 10000] # coarse grid
    LAMBDAS = [200, 300, 400, 500, 600, 700, 800] # fine grid
    # LAMBDAS=[350, 400, 450, 500, 550, 600, 650] # finer grid
    GRADIENT_TYPES = ["svgd", "rlsd"] 
    
        
    # PROMPTS = ["a ripe strawberry", "a delicious hamburger", "a photo of a hamburger", 
    #           "a photo of an ice cream", "a campfire", "a small saguaro cactus planted in a clay pot", 
    #           "a photo of a tulip", "a 3D model of a fox"]
    COMPARISON_PROMPTS = ["a small saguaro cactus planted in a clay pot"] # for baseline comparison (single viewpoint)
    MULTI_VIEW_PROMPTS = ["a 3D model of a fox", "a delicious hamburger"] # for multi-viewpoint comparison (multi-viewpoint)
    DIVERSITY_PROMPTS = ["a ripe strawberry", "a campfire"] # for diversity comparison (diversity)
    # PROMPTS = COMPARISON_PROMPTS + MULTI_VIEW_PROMPTS + DIVERSITY_PROMPTS
    
    PROMPTS = COMPARISON_PROMPTS
    
    opt.iters = 1500
    opt.visualize = False
    opt.metrics = True
    opt.quantitative_metrics_interval = 50


    # Multiple seeds for statistical significance
    SEEDS = [42, 123, 456, 789, 999]  # 5 different seeds for each configuration
    
    total_experiments = len(PROMPTS) * len(SEEDS) * (len(LAMBDAS) * len(GRADIENT_TYPES) + 1)
    print("[INFO] Starting hyperparameter tuning...")
    print(f"[INFO] Will run {total_experiments} experiments total")
    print(f"[INFO] {len(SEEDS)} seeds per configuration for statistical significance")
    
    job_dir= opt.outdir
    
    print(f"--------------------------------")
    print(f"[INFO] outdir: {opt.outdir}")
    print(f"[INFO] SEEDS: {SEEDS}")
    print(f"[INFO] PROMPTS: {PROMPTS}")
    print(f"[INFO] LAMBDAS: {LAMBDAS}")
    print(f"[INFO] GRADIENT_TYPES: {GRADIENT_TYPES}")
    print(f"--------------------------------\n")
    
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n[INFO] Running with seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
        
        # Set seed for this run
        seed_everything(seed)
            
        for prompt in PROMPTS:
            print(f"\n[INFO] Running experiments for prompt: '{prompt}'")
            opt.prompt = prompt
            prompt_clean = prompt.replace(" ", "_")
            
            # Experiments with repulsion enabled
            for repulsion_lambda in LAMBDAS:
                for gradient_type in GRADIENT_TYPES:
                    print(f"\n[INFO] Running: {gradient_type} with lambda={repulsion_lambda}, seed={seed}")

                    opt.repulsion_enabled = True
                    opt.repulsion_type = gradient_type
                    opt.lambda_repulsion = repulsion_lambda
                    
                    # Set output directory and save path
                    opt.outdir = job_dir + f"/{prompt_clean}_w_repulsion_{gradient_type}_{repulsion_lambda}_seed_{seed}"
                    opt.save_path = f"{prompt_clean}_w_repulsion_{gradient_type}_{repulsion_lambda}_seed_{seed}"
                    
                    # Create output directory
                    os.makedirs(opt.outdir, exist_ok=True)


                    # Run experiment
                    gui = GUI(opt)
                    gui.train(opt.iters)
                    
                    # Clean up GPU memory
                    del gui
                    torch.cuda.empty_cache()
                        
                    print(f"[INFO] Completed: {gradient_type} with lambda={repulsion_lambda}, seed={seed}")
                    
            
            # Experiment without repulsion (baseline) - using the same seed
            print(f"\n[INFO] Running baseline without repulsion, seed={seed}")
            
            opt.repulsion_enabled = False
            
            # Set output directory and save path for baseline
            opt.outdir = job_dir + f"/{prompt_clean}_wo_repulsion_seed_{seed}"
            opt.save_path = f"{prompt_clean}_wo_repulsion_seed_{seed}"
            
            # Create output directory
            os.makedirs(opt.outdir, exist_ok=True)
            
            # Run experiment
            gui = GUI(opt)
            gui.train(opt.iters)
            
            # Clean up GPU memory
            del gui
            torch.cuda.empty_cache()
            
            print(f"[INFO] Completed baseline without repulsion, seed={seed}")
    
    print("\n[INFO] Hyperparameter tuning completed!")
    print(f"[INFO] Results saved in outputs/hypertuning/ with seed-specific subdirectories")
    print(f"[INFO] Each configuration was run {len(SEEDS)} times for statistical analysis")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # Note: We don't set seed here as it will be set individually for each experiment
    print(f"[INFO] Base config loaded, will use multiple seeds for experiments")

    # Run hyperparameter tuning
    run_hyperparameter_tuning(opt)
