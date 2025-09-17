#!/usr/bin/env python3
"""
Average ablation results over prompts.

This script reads ablation CSV files and creates averaged versions where
metrics are averaged across all prompts for each ablation parameter value.
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def average_over_prompts(input_file: Path, output_file: Path) -> None:
    """
    Average ablation results over prompts.
    
    Args:
        input_file: Path to input ablation CSV file
        output_file: Path to output averaged CSV file
    """
    print(f"Processing: {input_file.name}")
    
    # Read the ablation CSV
    df = pd.read_csv(input_file)
    
    # Check if prompt column exists
    if 'prompt' not in df.columns:
        print(f"  No 'prompt' column found, skipping {input_file.name}")
        return
    
    # Get the ablation parameter column (not experiment, prompt, or n_runs)
    ablation_param = None
    for col in df.columns:
        if col not in ['experiment', 'prompt', 'n_runs'] and not col.startswith(('fidelity_', 'diversity_', 'cross_consistency_')):
            ablation_param = col
            break
    
    if ablation_param is None:
        print(f"  No ablation parameter found, skipping {input_file.name}")
        return
    
    print(f"  Ablation parameter: {ablation_param}")
    print(f"  Prompts: {df['prompt'].nunique()}")
    
    # Group by ablation parameter and average over prompts
    grouped = df.groupby([ablation_param]).agg({
        'n_runs': 'sum',  # Sum total runs across all prompts
        **{col: 'mean' for col in df.columns if col.startswith(('fidelity_', 'diversity_', 'cross_consistency_'))}
    }).reset_index()
    
    # Add experiment column
    grouped['experiment'] = df['experiment'].iloc[0]
    
    # Reorder columns to match original structure
    cols = ['experiment', ablation_param, 'n_runs'] + [col for col in grouped.columns if col.startswith(('fidelity_', 'diversity_', 'cross_consistency_'))]
    grouped = grouped[cols]
    
    # Save averaged results
    grouped.to_csv(output_file, index=False)
    print(f"  Saved: {output_file.name}")
    print(f"  Rows: {len(grouped)} (averaged over {df['prompt'].nunique()} prompts)")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Average ablation results over prompts")
    parser.add_argument("--results-dir", default="./results/csv", help="Directory containing ablation CSV files")
    parser.add_argument("--output-suffix", default="_averaged", help="Suffix for output files")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1
    
    print("Averaging Ablation Results Over Prompts")
    print("=" * 50)
    
    # Find all ablation CSV files
    ablation_files = list(results_dir.glob("*_ablation.csv"))
    
    if not ablation_files:
        print("No ablation CSV files found!")
        return 1
    
    print(f"Found {len(ablation_files)} ablation files")
    print()
    
    processed_count = 0
    
    for ablation_file in ablation_files:
        # Create output filename
        output_file = results_dir / f"{ablation_file.stem}{args.output_suffix}.csv"
        
        try:
            average_over_prompts(ablation_file, output_file)
            processed_count += 1
            print()
        except Exception as e:
            print(f"  Error processing {ablation_file.name}: {e}")
            print()
    
    print("=" * 50)
    print(f"Processed {processed_count}/{len(ablation_files)} files")
    print(f"Results saved to: {results_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
