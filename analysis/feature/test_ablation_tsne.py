#!/usr/bin/env python3
"""
Test script for ablation t-SNE analysis.

This script tests the feature loading and basic functionality without requiring
all dependencies to be installed.
"""

import os
import glob
from pathlib import Path

def test_feature_loading():
    """Test loading features from a sample run."""
    print("Testing feature loading...")
    
    # Test with a sample run
    test_run = Path("../../exp/exp0_baseline/WO__BULL__S42")
    features_dir = test_run / "features"
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        return False
    
    print(f"✅ Features directory found: {features_dir}")
    
    # List feature files
    feature_files = sorted(glob.glob(str(features_dir / "step_*.pt")))
    print(f"Found {len(feature_files)} feature files:")
    for f in feature_files:
        print(f"  - {Path(f).name}")
    
    # Try to load one file (without torch dependency)
    try:
        import torch
        print("\nTesting torch loading...")
        data = torch.load(feature_files[0], map_location='cpu')
        print("✅ Successfully loaded feature file")
        print(f"Keys: {list(data.keys())}")
        
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available - cannot test loading")
        return False
    except Exception as e:
        print(f"❌ Error loading feature file: {e}")
        return False

def test_experiment_structure():
    """Test the experiment directory structure."""
    print("\nTesting experiment structure...")
    
    exp_root = Path("../../exp")
    if not exp_root.exists():
        print(f"❌ Experiment root not found: {exp_root}")
        return False
    
    print(f"✅ Experiment root found: {exp_root}")
    
    # List experiments
    experiments = [d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith("exp")]
    print(f"Found {len(experiments)} experiments:")
    
    for exp in experiments:
        print(f"  - {exp.name}")
        
        # Check for runs with seed 42
        runs_with_seed42 = [d for d in exp.iterdir() if d.is_dir() and "S42" in d.name]
        print(f"    Runs with seed 42: {len(runs_with_seed42)}")
        
        for run in runs_with_seed42[:3]:  # Show first 3
            features_dir = run / "features"
            if features_dir.exists():
                feature_files = list(features_dir.glob("step_*.pt"))
                print(f"      {run.name}: {len(feature_files)} feature files")
            else:
                print(f"      {run.name}: No features directory")
    
    return True

def main():
    """Run all tests."""
    print("Ablation t-SNE Test Script")
    print("=" * 40)
    
    # Test experiment structure
    structure_ok = test_experiment_structure()
    
    # Test feature loading
    loading_ok = test_feature_loading()
    
    print("\n" + "=" * 40)
    if structure_ok and loading_ok:
        print("✅ All tests passed! Ready to run ablation t-SNE analysis.")
        print("\nTo run the full analysis:")
        print("  python run_ablation_tsne.py")
    else:
        print("❌ Some tests failed. Check the output above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
