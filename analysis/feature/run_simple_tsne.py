#!/usr/bin/env python3
"""
Run simple ablation t-SNE analysis.

This script runs t-SNE analysis on the available ablation experiments.
"""

import subprocess
import sys
from pathlib import Path

def run_simple_tsne():
    """Run simple t-SNE analysis for available experiments."""
    
    print("Simple Ablation t-SNE Analysis")
    print("=" * 40)
    
    # Check which experiments are available
    # Try both relative paths (from project root and from analysis/feature)
    exp_root = Path("../../exp")
    if not exp_root.exists():
        exp_root = Path("../exp")
    
    available_experiments = []
    
    ablation_experiments = [
        "exp_feature_layer",
        "exp_num_particles", 
        "exp2_lambda_coarse",
        "exp3_lambda_fine",
        "exp4_guidance_scale",
        "exp5_rbf_beta"
    ]
    
    for exp in ablation_experiments:
        exp_path = exp_root / exp
        if exp_path.exists():
            available_experiments.append(exp)
            print(f"✅ {exp} - available")
        else:
            print(f"❌ {exp} - not found")
    
    if not available_experiments:
        print("No ablation experiments found!")
        return False
    
    print(f"\nFound {len(available_experiments)} experiments to process")
    
    # Run the simple t-SNE script
    cmd = [
        sys.executable, "ablation_tsne_simple.py",
        "--exp-root", str(exp_root),
        "--output-dir", "../../results/feature",
        "--max-samples", "1000",
        "--experiments"] + available_experiments
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    success = run_simple_tsne()
    if success:
        print("\n✅ Simple ablation t-SNE analysis completed successfully!")
        print("Check results in: ../../results/feature/")
    else:
        print("\n❌ Simple ablation t-SNE analysis failed!")
        sys.exit(1)
