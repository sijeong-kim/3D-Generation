# Analysis Scripts

This directory contains the analysis scripts for the 3D Generation project.

## Directory Structure

- `csv/` - Contains all executable Python scripts for data analysis
  - `run_full_analysis.py` - Master script to run the complete analysis pipeline
  - `consolidate_experiments.py` - Consolidates experiment data into single CSV files
  - `ablation_analysis.py` - Performs ablation analysis across seeds
  - `average_over_prompts.py` - Averages ablation results over prompts
  - `generate_config_doc.py` - Generates consolidated configuration documentation

## Usage

Run the complete analysis pipeline:

```bash
python analysis/csv/run_full_analysis.py
```

Or run individual scripts:

```bash
python analysis/csv/consolidate_experiments.py --exp-root ./exp --results-dir ./results/csv
python analysis/csv/ablation_analysis.py --results-dir ./results/csv --exp-root ./exp
python analysis/csv/average_over_prompts.py --results-dir ./results/csv
python analysis/csv/generate_config_doc.py --exp-root ./exp --results-dir ./results/csv
```

## Results

All analysis results are saved to `results/csv/` directory, including:
- Consolidated experiment data (`*_consolidated.csv`)
- Ablation analysis results (`*_ablation.csv`)
- Averaged ablation results over prompts (`*_ablation_averaged.csv`)
- Configuration documentation (`experiment_configuration.yaml`, `experiment_summary.md`)
- Overall experiment summary (`experiment_summary.csv`)
