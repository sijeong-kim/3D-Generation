# We'll parse the two uploaded CSVs, filter to rows matching prompt='hamburger' and seed=42,
# then plot useful training curves. We'll also display the filtered tables for transparency.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot metrics for experiment configs')
parser.add_argument('--exp_name', type=str, default="exp0_baseline", 
                    help='Experiment name (default: exp0_baseline)')
parser.add_argument('--config_name', type=str, default=None,
                    help='Specific config name (default: auto-detect all configs)')
parser.add_argument('--base_dir', type=str, default="exp",
                    help='Base directory for experiments (default: exp)')
parser.add_argument('--overview_only', action='store_true',
                    help='Generate only overview plots (losses and metrics separately, faster, saves space)')
parser.add_argument('--individual_only', action='store_true',
                    help='Generate only individual plots (no overview)')

args = parser.parse_args()

base_dir = args.base_dir
exp_name = args.exp_name
config_name = args.config_name

# Function to get available configs
def get_available_configs(base_dir, exp_name):
    exp_path = Path(f"{base_dir}/{exp_name}")
    if not exp_path.exists():
        print(f"Experiment directory {exp_path} does not exist!")
        return []
    
    configs = []
    for item in exp_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'logs':
            # Check if this directory has metrics
            metrics_path = item / "metrics"
            if metrics_path.exists():
                configs.append(item.name)
    
    return sorted(configs)

# Auto-detect configs if not specified
if config_name is None:
    available_configs = get_available_configs(base_dir, exp_name)
    if not available_configs:
        print(f"No configs found in {base_dir}/{exp_name}")
        exit(1)
    print(f"Found configs: {available_configs}")
    configs_to_process = available_configs
else:
    configs_to_process = [config_name]

# Process each config
for config_name in configs_to_process:
    print(f"\nProcessing config: {config_name}")
    
    losses_path = Path(f"{base_dir}/{exp_name}/{config_name}/metrics/losses.csv")
    metrics_path = Path(f"{base_dir}/{exp_name}/{config_name}/metrics/quantitative_metrics.csv")
    
    # Check if files exist
    if not losses_path.exists():
        print(f"Warning: {losses_path} does not exist, skipping...")
        continue
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} does not exist, skipping...")
        continue
    
    # Load CSVs
    losses = pd.read_csv(losses_path)
    metrics = pd.read_csv(metrics_path)

    # Prepare step columns
    losses_step_col = "step"
    metrics_step_col = "step"

    # Define plot helper for individual plots
    def plot_line(df, x, y, title, ylabel, fname):
        dfp = df[[x, y]].dropna().sort_values(x)
        if len(dfp) == 0:
            return None
        plt.figure()
        plt.plot(dfp[x].values, dfp[y].values)
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(ylabel)
        out = Path(f"{base_dir}/{exp_name}/{config_name}/figures") / fname
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()  # Close figure to prevent memory issues
        return out

    # Define function to create losses overview plot
    def create_losses_overview_plot(losses_df, config_name):
        # Collect all available loss columns
        loss_cols = [
            ("attraction_loss", "Attraction Loss"),
            ("repulsion_loss", "Repulsion Loss"),
            ("scaled_attraction_loss", "Scaled Attraction Loss"),
            ("scaled_repulsion_loss", "Scaled Repulsion Loss"),
            ("total_loss", "Total Loss"),
            ("scaled_repulsion_loss_ratio", "Scaled Repulsion Loss Ratio (%)"),
        ]
        
        # Filter available columns
        available_loss_cols = [(col, label) for col, label in loss_cols if col in losses_df.columns]
        
        if len(available_loss_cols) == 0:
            return None
            
        # Calculate subplot layout
        cols = 3
        rows = (len(available_loss_cols) + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot loss curves
        for col, label in available_loss_cols:
            row_idx = plot_idx // cols
            col_idx = plot_idx % cols
            ax = axes[row_idx, col_idx]
            
            dfp = losses_df[["step", col]].dropna().sort_values("step")
            if len(dfp) > 0:
                ax.plot(dfp["step"].values, dfp[col].values, linewidth=2)
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.set_xlabel("Step", fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
            
            plot_idx += 1
        
        # Hide empty subplots
        while plot_idx < rows * cols:
            row_idx = plot_idx // cols
            col_idx = plot_idx % cols
            axes[row_idx, col_idx].set_visible(False)
            plot_idx += 1
        
        # Add overall title
        fig.suptitle(f"Training Losses - {config_name}", fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save losses overview plot
        out = Path(f"{base_dir}/{exp_name}/{config_name}/figures") / "losses_overview.png"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        
        return out

    # Define function to create metrics overview plot
    def create_metrics_overview_plot(metrics_df, config_name):
        # Collect all available metric columns
        metric_cols = [
            ("fidelity_mean", "CLIP Fidelity (mean)"),
            ("inter_particle_diversity_mean", "Inter-Particle Diversity (mean)"),
            ("cross_view_consistency_mean", "Cross-View Consistency (mean)"),
            ("lpips_inter_mean", "LPIPS Inter (mean)"),
            ("lpips_consistency_mean", "LPIPS Consistency (mean)"),
        ]
        
        # Filter available columns
        available_metric_cols = [(col, label) for col, label in metric_cols if col in metrics_df.columns]
        
        if len(available_metric_cols) == 0:
            return None
            
        # Calculate subplot layout
        cols = 3
        rows = (len(available_metric_cols) + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot metrics
        for col, label in available_metric_cols:
            row_idx = plot_idx // cols
            col_idx = plot_idx % cols
            ax = axes[row_idx, col_idx]
            
            dfp = metrics_df[["step", col]].dropna().sort_values("step")
            if len(dfp) > 0:
                ax.plot(dfp["step"].values, dfp[col].values, linewidth=2, color='green')
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.set_xlabel("Step", fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
            
            plot_idx += 1
        
        # Hide empty subplots
        while plot_idx < rows * cols:
            row_idx = plot_idx // cols
            col_idx = plot_idx % cols
            axes[row_idx, col_idx].set_visible(False)
            plot_idx += 1
        
        # Add overall title
        fig.suptitle(f"Quantitative Metrics - {config_name}", fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save metrics overview plot
        out = Path(f"{base_dir}/{exp_name}/{config_name}/figures") / "metrics_overview.png"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        
        return out

    saved_files = []

    # Create separate overview plots (unless individual_only is specified)
    if not args.individual_only:
        # Create losses overview plot
        losses_overview_plot = create_losses_overview_plot(losses, config_name)
        if losses_overview_plot is not None:
            saved_files.append(losses_overview_plot)
            print(f"Generated losses overview plot for {config_name}")
        
        # Create metrics overview plot
        metrics_overview_plot = create_metrics_overview_plot(metrics, config_name)
        if metrics_overview_plot is not None:
            saved_files.append(metrics_overview_plot)
            print(f"Generated metrics overview plot for {config_name}")

    # Generate individual plots (unless overview_only is specified)
    if not args.overview_only:
        # Loss curves (plot if columns exist)
        loss_cols = [
            ("attraction_loss", "Attraction Loss"),
            ("repulsion_loss", "Repulsion Loss"),
            ("scaled_attraction_loss", "Scaled Attraction Loss"),
            ("scaled_repulsion_loss", "Scaled Repulsion Loss"),
            ("total_loss", "Total Loss"),
            ("scaled_repulsion_loss_ratio", "Scaled Repulsion Loss Ratio (%)"),
        ]
        for col, label in loss_cols:
            if col in losses.columns:
                p = plot_line(losses, losses_step_col, col, f"{label} vs {losses_step_col}", label, f"{col}.png")
                if p is not None:
                    saved_files.append(p)

        # Quantitative metrics (plot common ones if present)
        metric_cols = [
            ("fidelity_mean", "CLIP Fidelity (mean)"),
            ("inter_particle_diversity_mean", "Inter-Particle Diversity (mean)"),
            ("cross_view_consistency_mean", "Cross-View Consistency (mean)"),
            ("lpips_inter_mean", "LPIPS Inter (mean)"),
            ("lpips_consistency_mean", "LPIPS Consistency (mean)"),
        ]

        for col, label in metric_cols:
            if col in metrics.columns:
                p = plot_line(metrics, metrics_step_col, col, f"{label} vs {metrics_step_col}", label, f"{col}.png")
                if p is not None:
                    saved_files.append(p)

    print(f"Generated {len(saved_files)} plots for {config_name}")

print("\nAll configs processed!")
