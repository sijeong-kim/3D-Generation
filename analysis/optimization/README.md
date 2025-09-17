# Optimization Analysis

This directory contains scripts for constraint-based parameter optimization in the 3D generation experiments.

## Scripts

- `pareto_plots.py` - Generates Pareto plots and implements constraint-based parameter selection

## Constraint-Based Selection Rule

The analysis uses a systematic approach to select the best parameter for each experiment:

1. **Calculate relative fidelity change**: `(fidelity - fidelity_baseline) / fidelity_baseline`
2. **Apply constraint**: `relative_fidelity_change >= -0.05` (5% fidelity loss tolerance)
3. **If all cases fail**: Relax to `relative_fidelity_change >= -0.10` (10% fidelity loss tolerance)
4. **Among successful cases**: Select the one with highest diversity

## Usage

```bash
# Generate Pareto plots and constraint-based parameter selection
python pareto_plots.py --results-dir ../results/csv --output-dir ../results/optimization
```

## Output

Results are saved to `../results/optimization/` including:
- `best_parameters_per_experiment.csv` - Best parameter values using constraint-based selection
- `pareto_*.png` - Individual experiment Pareto plots with constraint-based selection highlighted
- `pareto_summary.csv` - Pareto efficiency statistics

## Best Parameter Values

| Experiment | Parameter | Best Value | Rel Fidelity Δ | Diversity | Constraint |
|------------|-----------|------------|----------------|-----------|------------|
| exp1_repulsion_kernel | method | RLSD | -1.47% | 0.2064 | δ=0.05 |
| exp2_lambda_coarse | lambda_repulsion | 1000.0 | -6.81% | 0.2699 | δ=0.10 |
| exp3_lambda_fine | lambda_repulsion | 1400.0 | -7.93% | 0.2758 | δ=0.10 |
| exp4_guidance_scale | guidance_scale | 50.0 | -2.68% | 0.2783 | δ=0.05 |
| exp5_rbf_beta | rbf_beta | 1.0 | -8.82% | 0.2681 | δ=0.10 |

## Key Features

- **Systematic Selection**: Uses clear, repeatable constraint-based rules
- **Fidelity Preservation**: Ensures fidelity doesn't drop too much (max 10%)
- **Diversity Maximization**: Selects highest diversity among valid options
- **Visual Analysis**: Pareto plots show trade-offs and best selections
- **Robust Fallback**: Relaxes constraints when necessary