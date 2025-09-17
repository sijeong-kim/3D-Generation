#!/usr/bin/env python3
"""
Master script to run the complete analysis pipeline.

This script runs all analysis steps in sequence:
1. Consolidate experiments into single CSV files
2. Perform ablation analysis across seeds
3. Generate consolidated configuration documentation
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command to run as list
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run complete analysis pipeline")
    parser.add_argument("--exp-root", default="./exp", help="Root directory containing experiments")
    parser.add_argument("--results-dir", default="./results/csv", help="Directory to save results")
    parser.add_argument("--skip-consolidate", action="store_true", help="Skip consolidation step")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation analysis step")
    parser.add_argument("--skip-average", action="store_true", help="Skip averaging over prompts step")
    parser.add_argument("--skip-config", action="store_true", help="Skip configuration documentation step")
    
    args = parser.parse_args()
    
    exp_root = Path(args.exp_root)
    results_dir = Path(args.results_dir)
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not exp_root.exists():
        print(f"Experiment root directory {exp_root} does not exist")
        return 1
    
    print("3D Generation Analysis Pipeline")
    print("=" * 60)
    print(f"Experiment root: {exp_root}")
    print(f"Results directory: {results_dir}")
    print("=" * 60)
    
    success_count = 0
    total_steps = 0
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Step 1: Consolidate experiments
    if not args.skip_consolidate:
        total_steps += 1
        cmd = [
            sys.executable, str(script_dir / "consolidate_experiments.py"),
            "--exp-root", str(exp_root),
            "--results-dir", str(results_dir)
        ]
        if run_command(cmd, "Consolidating experiments into single CSV files"):
            success_count += 1
    else:
        print("\nSkipping consolidation step")
    
    # Step 2: Ablation analysis
    if not args.skip_ablation:
        total_steps += 1
        cmd = [
            sys.executable, str(script_dir / "ablation_analysis.py"),
            "--results-dir", str(results_dir),
            "--exp-root", str(exp_root)
        ]
        if run_command(cmd, "Performing ablation analysis across seeds"):
            success_count += 1
    else:
        print("\nSkipping ablation analysis step")
    
    # Step 3: Average over prompts
    if not args.skip_average:
        total_steps += 1
        cmd = [
            sys.executable, str(script_dir / "average_over_prompts.py"),
            "--results-dir", str(results_dir)
        ]
        if run_command(cmd, "Averaging ablation results over prompts"):
            success_count += 1
    else:
        print("\nSkipping averaging over prompts step")
    
    # Step 4: Configuration documentation
    if not args.skip_config:
        total_steps += 1
        cmd = [
            sys.executable, str(script_dir / "generate_config_doc.py"),
            "--exp-root", str(exp_root),
            "--results-dir", str(results_dir)
        ]
        if run_command(cmd, "Generating consolidated configuration documentation"):
            success_count += 1
    else:
        print("\nSkipping configuration documentation step")
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS PIPELINE SUMMARY")
    print('='*60)
    print(f"Steps completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("✅ All steps completed successfully!")
        print(f"\nResults saved to: {results_dir}")
        print("\nGenerated files:")
        
        # List generated files
        result_files = list(results_dir.glob("*"))
        for file_path in sorted(result_files):
            if file_path.is_file():
                print(f"  - {file_path.name}")
        
        return 0
    else:
        print("❌ Some steps failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
