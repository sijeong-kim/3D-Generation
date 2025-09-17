#!/usr/bin/env python3
"""
Simple ablation t-SNE visualization script.

This script creates t-SNE plots for ablation studies, handling different seed patterns
and working with available data.
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def find_available_seeds(exp_dir):
    """Find available seeds in an experiment directory."""
    seeds = set()
    for run_dir in Path(exp_dir).iterdir():
        if run_dir.is_dir() and "S" in run_dir.name:
            # Extract seed from run name
            parts = run_dir.name.split("S")
            if len(parts) > 1:
                try:
                    seed = int(parts[-1])
                    seeds.add(seed)
                except ValueError:
                    continue
    return sorted(list(seeds))

def find_runs_with_seed(exp_dir, seed):
    """Find all runs with a specific seed."""
    runs = []
    for run_dir in Path(exp_dir).iterdir():
        if run_dir.is_dir() and f"S{seed}" in run_dir.name:
            runs.append(run_dir)
    return runs

def load_features_simple(run_dir, step="last"):
    """
    Load features from a run directory.
    This is a simplified version that doesn't require PyTorch.
    """
    features_dir = Path(run_dir) / "features"
    if not features_dir.exists():
        return None, None, None
    
    # Find feature files
    feature_files = sorted(glob.glob(str(features_dir / "step_*.pt")))
    if not feature_files:
        return None, None, None
    
    if step == "last":
        feature_file = feature_files[-1]
    elif step == "first":
        feature_file = feature_files[0]
    else:
        # Find specific step
        target_file = f"step_{step:06d}.pt"
        feature_file = None
        for f in feature_files:
            if f.endswith(target_file):
                feature_file = f
                break
        if not feature_file:
            return None, None, None
    
    # Try to load with PyTorch
    try:
        import torch
        data = torch.load(feature_file, map_location='cpu')
        
        # Extract features
        if "features_fp16" in data:
            features = data["features_fp16"].float().numpy()
        elif "features" in data:
            features = data["features"].float().numpy()
        else:
            return None, None, None
        
        # Extract particle IDs
        if "particle_ids" in data:
            particle_ids = data["particle_ids"].long().numpy()
        else:
            particle_ids = np.arange(features.shape[0])
        
        # Extract step
        step_num = data.get("step", 0)
        
        return features, particle_ids, step_num
        
    except ImportError:
        print("PyTorch not available - cannot load features")
        return None, None, None
    except Exception as e:
        print(f"Error loading {feature_file}: {e}")
        return None, None, None

def create_ablation_plot(exp_dir, output_dir, seed=None, max_samples=1000):
    """Create ablation comparison plot for an experiment."""
    exp_name = Path(exp_dir).name
    print(f"\nProcessing: {exp_name}")
    
    # Find available seeds
    available_seeds = find_available_seeds(exp_dir)
    print(f"  Available seeds: {available_seeds}")
    
    if not available_seeds:
        print(f"  No runs found in {exp_name}")
        return
    
    # Use specified seed or first available
    if seed is None:
        seed = available_seeds[0]
    elif seed not in available_seeds:
        print(f"  Seed {seed} not available, using {available_seeds[0]}")
        seed = available_seeds[0]
    
    # Find runs with this seed
    runs = find_runs_with_seed(exp_dir, seed)
    print(f"  Found {len(runs)} runs with seed {seed}")
    
    if len(runs) < 2:
        print(f"  Need at least 2 runs for ablation comparison, found {len(runs)}")
        return
    
    # Load features from all runs
    all_features = []
    all_particle_ids = []
    run_labels = []
    
    for run_dir in runs:
        print(f"    Loading {run_dir.name}...")
        features, particle_ids, step_num = load_features_simple(run_dir, step="last")
        
        if features is None:
            print(f"      Failed to load features")
            continue
        
        # Subsample if needed
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            particle_ids = particle_ids[indices]
        
        all_features.append(features)
        all_particle_ids.append(particle_ids)
        
        # Create label from run name
        run_name = run_dir.name
        if "__" in run_name:
            parts = run_name.split("__")
            if len(parts) >= 2:
                # Extract ablation parameter value
                ablation_value = parts[1].replace(f"S{seed}", "")
                run_labels.append(ablation_value)
            else:
                run_labels.append(run_name)
        else:
            run_labels.append(run_name)
    
    if len(all_features) < 2:
        print(f"  Not enough valid runs for comparison")
        return
    
    # Concatenate features
    X = np.vstack(all_features)
    P = np.concatenate(all_particle_ids)
    
    # Create run labels for each sample
    run_labels_expanded = []
    for i, (feat, label) in enumerate(zip(all_features, run_labels)):
        run_labels_expanded.extend([label] * len(feat))
    run_labels_expanded = np.array(run_labels_expanded)
    
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Ablation values: {set(run_labels)}")
    
    # Run t-SNE
    print(f"  Running t-SNE...")
    try:
        from cuml.manifold import TSNE
        import cupy as cp
        X_gpu = cp.asarray(X, dtype=cp.float32)
        tsne = TSNE(n_components=2, perplexity=min(30, X_gpu.shape[0]-1), 
                   random_state=42, max_iter=1000)
        Z = tsne.fit_transform(X_gpu)
        Z = cp.asnumpy(Z)
    except ImportError:
        try:
            from umap import UMAP
            Z = UMAP(n_components=2, random_state=42, n_neighbors=min(15, X.shape[0]-1)).fit_transform(X)
        except ImportError:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=min(30, X.shape[0]-1), 
                       random_state=42, max_iter=1000)
            Z = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(run_labels_expanded)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = run_labels_expanded == label
        z_label = Z[mask]
        
        plt.scatter(z_label[:, 0], z_label[:, 1], 
                   c=[colors[i]], s=30, alpha=0.7, 
                   label=label, edgecolors='black', linewidths=0.5)
    
    plt.title(f"{exp_name} - Ablation Comparison (Seed {seed})\nFinal Step Features", fontsize=14)
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{exp_name}_ablation_tsne_seed{seed}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create ablation t-SNE visualizations")
    parser.add_argument("--exp-root", default="../exp", help="Root directory containing experiments")
    parser.add_argument("--output-dir", default="../results/tsne", help="Output directory for plots")
    parser.add_argument("--seed", type=int, help="Specific seed to use (default: first available)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples per run")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to process")
    
    args = parser.parse_args()
    
    exp_root = Path(args.exp_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find experiments to process
    if args.experiments:
        exp_dirs = [exp_root / exp for exp in args.experiments]
    else:
        # Default ablation experiments
        ablation_experiments = [
            "exp_feature_layer",
            "exp_num_particles", 
            "exp2_lambda_coarse",
            "exp3_lambda_fine",
            "exp4_guidance_scale",
            "exp5_rbf_beta"
        ]
        exp_dirs = [exp_root / exp for exp in ablation_experiments if (exp_root / exp).exists()]
    
    print(f"Processing {len(exp_dirs)} experiments...")
    
    for exp_dir in exp_dirs:
        if exp_dir.exists():
            create_ablation_plot(exp_dir, output_dir, args.seed, args.max_samples)
        else:
            print(f"Experiment not found: {exp_dir}")
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
