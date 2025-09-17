#!/usr/bin/env python3
"""
Enhanced Pareto Plots for Fidelity vs Diversity Analysis

This script generates publication-quality Pareto plots to visualize the trade-offs between
fidelity and diversity for each parameter in the main ablation experiments.

python analysis/optimization/pareto_plots.py --results-dir results/csv --output-dir results/optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
from scipy.spatial.distance import pdist, squareform

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EnhancedParetoPlotGenerator:
    """Generates publication-quality Pareto plots for fidelity vs diversity analysis."""
    
    def __init__(self, results_dir: str = "results/csv"):
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.baseline_data = {}
        
        # Main ablation experiments (exp1-exp5) with actual column names
        self.main_experiments = {
            'exp1_repulsion_kernel': 'kernel',
            'exp2_lambda_coarse': 'lambda_repulsion',
            'exp3_lambda_fine': 'lambda_repulsion',
            'exp4_guidance_scale': 'guidance_scale',
            'exp5_rbf_beta': 'rbf_beta'
        }
    
    def load_data(self):
        """Load experiment data from CSV files."""
        print("Loading experiment data...")
        
        # Look for ablation_averaged.csv files directly in the results directory
        for csv_file in self.results_dir.glob("*_ablation_averaged.csv"):
            exp_name = csv_file.stem.replace("_ablation_averaged", "")
            if exp_name.startswith("exp"):
                df = pd.read_csv(csv_file)
                self.experiments[exp_name] = df
                print(f"  Loaded {exp_name}: {len(df)} runs")
    
    def get_baseline_data(self):
        """Load baseline data for comparison."""
        baseline_file = self.results_dir / "exp0_baseline_ablation_averaged.csv"
        if baseline_file.exists():
            df = pd.read_csv(baseline_file)
            # Use the aggregated baseline data
            if len(df) > 0:
                row = df.iloc[0]  # Take the first (and only) row
                self.baseline_data['baseline'] = {
                    'fidelity': row['fidelity_mean_mean'],
                    'diversity': row['diversity_mean_mean'],
                    'cross_consistency': row['cross_consistency_mean_mean']
                }
                print(f"Baseline data loaded: fidelity={row['fidelity_mean_mean']:.4f}, diversity={row['diversity_mean_mean']:.4f}")
    
    def find_pareto_points(self, df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """Find Pareto optimal points."""
        if len(df) == 0:
            return df
        
        # For Pareto optimality, we want to maximize both x and y
        # So we need to find points where no other point dominates
        pareto_indices = []
        
        for i, row in df.iterrows():
            is_pareto = True
            for j, other_row in df.iterrows():
                if i != j:
                    # Check if other_row dominates row
                    if (other_row[x_col] >= row[x_col] and other_row[y_col] >= row[y_col] and
                        (other_row[x_col] > row[x_col] or other_row[y_col] > row[y_col])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return df.loc[pareto_indices]
    
    def plot_pareto_frontier(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           x_label: str, y_label: str, title: str, 
                           ablation_param: str, output_path: Path):
        """Plot Pareto frontier for a single experiment with publication-quality styling."""
        
        # Group by parameter value
        grouped = df.groupby(ablation_param).agg({
            x_col: 'mean',
            y_col: 'mean',
            'cross_consistency_mean_mean': 'mean'
        }).reset_index()
        
        if len(grouped) == 0:
            print(f"  No data for {title}")
            return
        
        # Find Pareto points
        pareto_df = self.find_pareto_points(grouped, x_col, y_col)
        
        # Set up publication-quality figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.patch.set_facecolor('white')
        
        # Set font sizes for publication
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # Define professional color scheme
        colors = {
            'all_points': '#2E86AB',      # Professional blue
            'pareto_points': '#A23B72',   # Professional red
            'best_point': '#F18F01',      # Professional gold
            'baseline': '#C73E1D',        # Professional orange
            'pareto_line': '#A23B72'      # Same as Pareto points
        }
        
        # Plot all points with professional styling
        scatter = ax.scatter(grouped[x_col], grouped[y_col], 
                           c=grouped['cross_consistency_mean_mean'], 
                           s=120, alpha=0.8, cmap='viridis', 
                           edgecolors='white', linewidth=1.5,
                           zorder=2)
        
        # Plot Pareto frontier with enhanced styling
        if len(pareto_df) > 0:
            # Sort Pareto points for line plotting
            pareto_sorted = pareto_df.sort_values(x_col)
            ax.plot(pareto_sorted[x_col], pareto_sorted[y_col], 
                   color=colors['pareto_line'], linewidth=3, alpha=0.9, 
                   linestyle='-', label='Pareto Frontier', zorder=1)
            
            # Highlight Pareto points
            ax.scatter(pareto_df[x_col], pareto_df[y_col], 
                      c=colors['pareto_points'], s=150, alpha=0.9, 
                      edgecolors='white', linewidth=2, 
                      label='Pareto Optimal', zorder=4)
        
        # Add labels and title with professional styling
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\nPareto Frontier: {x_label} vs {y_label}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar positioned close to plot
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.05)
        cbar.set_label('Cross-view Consistency', fontsize=11, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=9, width=1.0, length=4)
        
        # Professional grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Highlight the best parameter value using constraint-based selection
        best_value = self._get_constraint_based_best_parameter(grouped, ablation_param)
        if best_value is not None:
            best_row = grouped[grouped[ablation_param] == best_value].iloc[0]
            ax.scatter(best_row[x_col], best_row[y_col], 
                      c=colors['best_point'], s=250, marker='*', 
                      edgecolors='white', linewidth=2,
                      label='Best Parameter (Constraint-Based)', zorder=6)
        
        # Add baseline reference point (optional)
        show_baseline = True  # Set to False to hide baseline
        if show_baseline and x_col == 'fidelity_mean_mean' and y_col == 'diversity_mean_mean':
            baseline_fidelity = np.mean([data['fidelity'] for data in self.baseline_data.values()])
            baseline_diversity = np.mean([data['diversity'] for data in self.baseline_data.values()])
            baseline_consistency = np.mean([data['cross_consistency'] for data in self.baseline_data.values()])
            
            print(f"    Baseline: fidelity={baseline_fidelity:.4f}, diversity={baseline_diversity:.4f}, consistency={baseline_consistency:.4f}")
            print(f"    Data range: fidelity=[{grouped[x_col].min():.4f}, {grouped[x_col].max():.4f}], diversity=[{grouped[y_col].min():.4f}, {grouped[y_col].max():.4f}]")
            
            # Use the same colorbar as other points for consistency
            ax.scatter(baseline_fidelity, baseline_diversity, 
                      c=baseline_consistency, s=200, marker='s', 
                      edgecolors='white', linewidth=2,
                      label='Baseline', zorder=5, cmap='viridis')
        
        # Clean, beautiful legend - positioned to avoid colorbar overlap
        legend = ax.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', 
                          frameon=True, fancybox=False, shadow=False, 
                          framealpha=0.95, edgecolor='black',
                          fontsize=10, markerscale=0.9, handletextpad=0.6,
                          columnspacing=0.8, labelspacing=0.4, borderpad=0.6)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_boxstyle("round,pad=0.2")
        
        # Add clean parameter value annotations with custom positioning per experiment
        for _, row in grouped.iterrows():
            # Custom positioning based on experiment
            if 'exp5' in title.lower():
                # For exp5: place labels on top of datapoints
                xytext = (0, 15)
                ha, va = 'center', 'bottom'
            elif 'exp2' in title.lower() and str(row[ablation_param]) == '1.0':
                # For exp2: only place 1.0 label on the right of datapoints
                xytext = (15, 0)
                ha, va = 'left', 'center'
            else:
                # Default: top right positioning (including exp2 non-1.0 labels)
                xytext = (8, 8)
                ha, va = 'left', 'bottom'
            
            ax.annotate(f'{row[ablation_param]}', 
                       (row[x_col], row[y_col]),
                       xytext=xytext, textcoords='offset points',
                       fontsize=9, fontweight='bold', alpha=0.8,
                       ha=ha, va=va,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor='black', alpha=0.9, linewidth=0.8))
        
        # Set axis limits with some padding, including baseline if present
        x_min = grouped[x_col].min()
        x_max = grouped[x_col].max()
        y_min = grouped[y_col].min()
        y_max = grouped[y_col].max()
        
        # Include baseline in range if showing baseline
        if show_baseline and x_col == 'fidelity_mean_mean' and y_col == 'diversity_mean_mean':
            baseline_fidelity = np.mean([data['fidelity'] for data in self.baseline_data.values()])
            baseline_diversity = np.mean([data['diversity'] for data in self.baseline_data.values()])
            x_min = min(x_min, baseline_fidelity)
            x_max = max(x_max, baseline_fidelity)
            y_min = min(y_min, baseline_diversity)
            y_max = max(y_max, baseline_diversity)
        
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save with high quality
        plot_filename = f"pareto_{title.lower().replace(' ', '_').replace(':', '')}.png"
        plt.savefig(output_path / plot_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  Saved publication-quality Pareto plot: {plot_filename}")
    
    def _get_constraint_based_best_parameter(self, grouped: pd.DataFrame, param_name: str) -> any:
        """Get best parameter using constraint-based method for plotting."""
        if len(grouped) == 0:
            return None
        
        # Check if we have the required columns
        if 'fidelity_mean' not in grouped.columns or 'diversity_mean' not in grouped.columns:
            # Fallback to highest diversity if columns not available
            if 'diversity_mean' in grouped.columns:
                best_idx = grouped['diversity_mean'].idxmax()
                return grouped.loc[best_idx, param_name]
            return None
        
        # Calculate baseline fidelity
        baseline_fidelity = np.mean([data['fidelity'] for data in self.baseline_data.values()])
        
        # Calculate relative fidelity change
        grouped = grouped.copy()
        grouped['relative_fidelity_change'] = (grouped['fidelity_mean'] - baseline_fidelity) / baseline_fidelity
        
        # First try with δ = 0.05
        constraint_05 = grouped[grouped['relative_fidelity_change'] >= -0.05]
        if len(constraint_05) > 0:
            best_idx = constraint_05['diversity_mean'].idxmax()
            return constraint_05.loc[best_idx, param_name]
        
        # If all cases fail with δ=0.05, try with δ=0.10
        constraint_10 = grouped[grouped['relative_fidelity_change'] >= -0.10]
        if len(constraint_10) > 0:
            best_idx = constraint_10['diversity_mean'].idxmax()
            return constraint_10.loc[best_idx, param_name]
        
        return None
    
    def generate_pareto_plots(self, output_dir: str = "results/optimization"):
        """Generate all Pareto plots with publication-quality styling."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("ENHANCED PARETO PLOTS GENERATION")
        print("Focus: Publication-Quality Fidelity vs Diversity Trade-offs")
        print(f"{'='*80}")
        
        # Load data
        self.load_data()
        self.get_baseline_data()
        
        # Generate plots for main experiments
        for exp_name, param_name in self.main_experiments.items():
            if exp_name in self.experiments:
                print(f"\nGenerating enhanced Pareto plots for {exp_name}...")
                
                df = self.experiments[exp_name]
                
                # Fidelity vs Diversity
                self.plot_pareto_frontier(df, 'fidelity_mean_mean', 'diversity_mean_mean',
                                        'Fidelity', 'Diversity', f'{exp_name}: {param_name}',
                                        param_name, output_path)
                
                # Relative metrics if available
                if 'delta_fidelity' in df.columns and 'delta_diversity' in df.columns:
                    self.plot_pareto_frontier(df, 'delta_fidelity', 'delta_diversity',
                                            'Δ Fidelity (relative)', 'Δ Diversity (absolute)', 
                                            f'{exp_name}: {param_name} (Relative)',
                                            param_name, output_path)
        
        print(f"\n{'='*80}")
        print("ENHANCED PARETO PLOTS GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Publication-quality plots saved to: {output_path}")
        print("Generated files:")
        print("  - pareto_*.png (individual experiment plots)")
        print("  - All plots optimized for publication with professional styling")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate enhanced Pareto plots')
    parser.add_argument('--results-dir', type=str, default='results/csv',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='results/optimization',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Generate enhanced Pareto plots
    generator = EnhancedParetoPlotGenerator(args.results_dir)
    generator.generate_pareto_plots(args.output_dir)


if __name__ == "__main__":
    main()
