#!/usr/bin/env python3
"""
Experiment Results Gatherer

This script aggregates results from multiple experiment runs to enable quick generation
of thesis figures including Pareto plots, box plots, and comparative analyses.

Usage:
    python gather_results.py --exp_dir exp/ --output_dir analysis_results/
    python gather_results.py --exp_dir exp/ --output_dir analysis_results/ --plot_type pareto
    python gather_results.py --exp_dir exp/ --output_dir analysis_results/ --plot_type boxplot
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import glob
import json
from datetime import datetime


def load_experiment_summary(exp_path: str) -> Optional[Dict[str, Any]]:
    """Load experiment summary from a single experiment directory."""
    summary_path = os.path.join(exp_path, "experiment_summary.yaml")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load {summary_path}: {e}")
    return None


def load_metrics_from_run(run_path: str) -> Optional[Dict[str, Any]]:
    """Load metrics from a single experiment run."""
    metrics = {}
    
    # Load quantitative metrics
    metrics_path = os.path.join(run_path, "metrics", "quantitative_metrics.csv")
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            # Get the final row (last iteration)
            if not df.empty:
                final_metrics = df.iloc[-1].to_dict()
                metrics['quantitative'] = final_metrics
        except Exception as e:
            print(f"Warning: Could not load metrics from {metrics_path}: {e}")
    
    # Load losses
    losses_path = os.path.join(run_path, "losses.csv")
    if os.path.exists(losses_path):
        try:
            df = pd.read_csv(losses_path)
            if not df.empty:
                final_losses = df.iloc[-1].to_dict()
                metrics['losses'] = final_losses
        except Exception as e:
            print(f"Warning: Could not load losses from {losses_path}: {e}")
    
    # Load efficiency metrics
    efficiency_path = os.path.join(run_path, "efficiency.csv")
    if os.path.exists(efficiency_path):
        try:
            df = pd.read_csv(efficiency_path)
            if not df.empty:
                final_efficiency = df.iloc[-1].to_dict()
                metrics['efficiency'] = final_efficiency
        except Exception as e:
            print(f"Warning: Could not load efficiency from {efficiency_path}: {e}")
    
    return metrics if metrics else None


def extract_parameters_from_dirname(dirname: str) -> Dict[str, Any]:
    """Extract parameters from directory name (key=value format)."""
    params = {}
    if '=' in dirname:
        parts = dirname.split('_')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
    return params


def gather_all_results(exp_dir: str) -> pd.DataFrame:
    """Gather all results from experiment directory into a single DataFrame."""
    all_results = []
    
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    
    for exp_name in exp_dirs:
        exp_path = os.path.join(exp_dir, exp_name)
        summary = load_experiment_summary(exp_path)
        
        if not summary:
            continue
        
        print(f"Processing experiment: {exp_name}")
        
        # Process each combination in the experiment
        for combination in summary.get('combinations', []):
            run_path = combination.get('output_dir', '')
            if not run_path or not os.path.exists(run_path):
                continue
            
            # Extract parameters from directory name
            dirname = os.path.basename(run_path)
            params = extract_parameters_from_dirname(dirname)
            
            # Load metrics from the run
            metrics = load_metrics_from_run(run_path)
            
            # Create result row
            result_row = {
                'experiment_name': exp_name,
                'run_dir': dirname,
                'status': combination.get('status', 'unknown'),
                'duration_sec': combination.get('duration_sec', 0),
                'return_code': combination.get('return_code', -1),
                'start_time': combination.get('start_time', ''),
                'end_time': combination.get('end_time', ''),
                **params  # Add all extracted parameters
            }
            
            # Add metrics if available
            if metrics:
                if 'quantitative' in metrics:
                    result_row.update({f"metric_{k}": v for k, v in metrics['quantitative'].items()})
                if 'losses' in metrics:
                    result_row.update({f"loss_{k}": v for k, v in metrics['losses'].items()})
                if 'efficiency' in metrics:
                    result_row.update({f"efficiency_{k}": v for k, v in metrics['efficiency'].items()})
            
            all_results.append(result_row)
    
    return pd.DataFrame(all_results)


def create_pareto_plot(df: pd.DataFrame, output_dir: str, x_metric: str = 'metric_fidelity_mean', 
                      y_metric: str = 'metric_inter_particle_diversity_mean'):
    """Create Pareto frontier plot."""
    plt.figure(figsize=(12, 8))
    
    # Filter successful runs
    successful_df = df[df['status'] == 'success'].copy()
    
    if successful_df.empty:
        print("No successful runs found for Pareto plot")
        return
    
    # Create scatter plot
    plt.scatter(successful_df[x_metric], successful_df[y_metric], alpha=0.6, s=50)
    
    # Add labels for each point
    for idx, row in successful_df.iterrows():
        label = f"{row.get('repulsion_type', 'N/A')}_{row.get('kernel_type', 'N/A')}"
        plt.annotate(label, (row[x_metric], row[y_metric]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel(x_metric.replace('metric_', '').replace('_', ' ').title())
    plt.ylabel(y_metric.replace('metric_', '').replace('_', ' ').title())
    plt.title('Pareto Frontier: Fidelity vs Diversity')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_frontier.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pareto plot saved to {os.path.join(output_dir, 'pareto_frontier.png')}")


def create_box_plots(df: pd.DataFrame, output_dir: str):
    """Create box plots for different metrics across methods."""
    successful_df = df[df['status'] == 'success'].copy()
    
    if successful_df.empty:
        print("No successful runs found for box plots")
        return
    
    # Create method identifier
    successful_df['method'] = successful_df['repulsion_type'] + '_' + successful_df['kernel_type']
    
    # Metrics to plot
    metrics_to_plot = [
        'metric_fidelity_mean',
        'metric_inter_particle_diversity_mean', 
        'metric_cross_view_consistency_mean'
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in successful_df.columns:
            # Create box plot
            sns.boxplot(data=successful_df, x='method', y=metric, ax=axes[i])
            axes[i].set_title(metric.replace('metric_', '').replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to {os.path.join(output_dir, 'box_plots.png')}")


def create_lambda_analysis(df: pd.DataFrame, output_dir: str):
    """Create lambda analysis plots."""
    successful_df = df[df['status'] == 'success'].copy()
    
    if successful_df.empty or 'lambda_repulsion' not in successful_df.columns:
        print("No lambda_repulsion data found for analysis")
        return
    
    # Create method identifier
    successful_df['method'] = successful_df['repulsion_type'] + '_' + successful_df['kernel_type']
    
    # Plot lambda vs metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['metric_fidelity_mean', 'metric_inter_particle_diversity_mean']
    
    for i, metric in enumerate(metrics):
        if metric in successful_df.columns:
            # Plot for each method
            for method in successful_df['method'].unique():
                method_data = successful_df[successful_df['method'] == method]
                axes[i, 0].scatter(method_data['lambda_repulsion'], method_data[metric], 
                                 label=method, alpha=0.7)
                axes[i, 0].set_xlabel('Lambda Repulsion')
                axes[i, 0].set_ylabel(metric.replace('metric_', '').replace('_', ' ').title())
                axes[i, 0].set_title(f'{metric.replace("metric_", "").replace("_", " ").title()} vs Lambda')
                axes[i, 0].legend()
                axes[i, 0].set_xscale('log')
                axes[i, 0].grid(True, alpha=0.3)
    
    # Efficiency analysis
    if 'efficiency_training_time_per_iter' in successful_df.columns:
        for method in successful_df['method'].unique():
            method_data = successful_df[successful_df['method'] == method]
            axes[0, 1].scatter(method_data['lambda_repulsion'], 
                             method_data['efficiency_training_time_per_iter'], 
                             label=method, alpha=0.7)
        axes[0, 1].set_xlabel('Lambda Repulsion')
        axes[0, 1].set_ylabel('Training Time per Iter (s)')
        axes[0, 1].set_title('Training Efficiency vs Lambda')
        axes[0, 1].legend()
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Duration analysis
    for method in successful_df['method'].unique():
        method_data = successful_df[successful_df['method'] == method]
        axes[1, 1].scatter(method_data['lambda_repulsion'], method_data['duration_sec'], 
                          label=method, alpha=0.7)
    axes[1, 1].set_xlabel('Lambda Repulsion')
    axes[1, 1].set_ylabel('Total Duration (s)')
    axes[1, 1].set_title('Total Duration vs Lambda')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lambda analysis saved to {os.path.join(output_dir, 'lambda_analysis.png')}")


def create_summary_report(df: pd.DataFrame, output_dir: str):
    """Create a comprehensive summary report."""
    report = {
        'summary_generated_at': datetime.now().isoformat(),
        'total_runs': len(df),
        'successful_runs': len(df[df['status'] == 'success']),
        'failed_runs': len(df[df['status'] == 'failed']),
        'skipped_runs': len(df[df['status'] == 'skipped']),
        'experiments': df['experiment_name'].unique().tolist(),
        'methods_tested': df[df['status'] == 'success']['repulsion_type'].unique().tolist(),
        'prompts_tested': df[df['status'] == 'success']['prompt'].unique().tolist(),
    }
    
    # Add statistical summaries for successful runs
    successful_df = df[df['status'] == 'success'].copy()
    if not successful_df.empty:
        report['metrics_summary'] = {}
        metric_columns = [col for col in successful_df.columns if col.startswith('metric_')]
        for metric in metric_columns:
            report['metrics_summary'][metric] = {
                'mean': float(successful_df[metric].mean()),
                'std': float(successful_df[metric].std()),
                'min': float(successful_df[metric].min()),
                'max': float(successful_df[metric].max())
            }
    
    # Save report
    with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Summary report saved to {os.path.join(output_dir, 'summary_report.json')}")


def main():
    parser = argparse.ArgumentParser(description="Gather and analyze experiment results")
    parser.add_argument("--exp_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--plot_type", type=str, default="all",
                       choices=["all", "pareto", "boxplot", "lambda"],
                       help="Type of plots to generate")
    parser.add_argument("--save_csv", action="store_true",
                       help="Save aggregated results as CSV")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Gathering results from: {args.exp_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Gather all results
    df = gather_all_results(args.exp_dir)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Found {len(df)} total runs")
    print(f"Successful runs: {len(df[df['status'] == 'success'])}")
    print(f"Failed runs: {len(df[df['status'] == 'failed'])}")
    
    # Save aggregated CSV if requested
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, 'aggregated_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Aggregated results saved to: {csv_path}")
    
    # Generate plots based on type
    if args.plot_type in ["all", "pareto"]:
        create_pareto_plot(df, args.output_dir)
    
    if args.plot_type in ["all", "boxplot"]:
        create_box_plots(df, args.output_dir)
    
    if args.plot_type in ["all", "lambda"]:
        create_lambda_analysis(df, args.output_dir)
    
    # Create summary report
    create_summary_report(df, args.output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
