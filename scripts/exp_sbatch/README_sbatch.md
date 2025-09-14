# SBATCH Experiment Scripts

This directory contains scripts for running 3D Generation experiments via SLURM job scheduler.

## Files

- `run_exp_sbatch.sh` - Main SLURM batch script that runs experiments
- `submit_exp.sh` - Wrapper script for easy job submission
- `run_exp_interactive.sh` - Original interactive experiment runner
- `example_usage.sh` - Examples of how to use the scripts
- `README_sbatch.md` - This documentation file

## Quick Start

1. **Basic usage:**
   ```bash
   ./submit_exp.sh my_experiment_name
   ```

2. **With custom parameters:**
   ```bash
   ./submit_exp.sh my_experiment --outdir results --gpus 0,1 --timeout 28800
   ```

3. **Dry run to check parameters:**
   ```bash
   ./submit_exp.sh my_experiment --dry_run
   ```

## Script Details

### `run_exp_sbatch.sh`

This is the main SLURM batch script that:
- Sets up the environment (CUDA, Python, libraries)
- Configures SLURM resources (1 GPU, 8 CPUs, 32GB RAM, 24h time limit)
- Runs the `run_exp_interactive.sh` script with specified parameters
- Handles logging and error reporting

**SLURM Configuration:**
- Job name: `exp_interactive`
- Partition: `gpgpu`
- Resources: 1 GPU, 8 CPUs, 32GB RAM
- Time limit: 24 hours
- Email notifications: END,FAIL

### `submit_exp.sh`

A convenient wrapper script that:
- Parses command-line arguments
- Validates parameters
- Builds and submits the sbatch command
- Provides help and usage information

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sweep_name` | (required) | Name of the experiment sweep |
| `--outdir` | `exp` | Output directory for results |
| `--base_config` | `configs/text_ours.yaml` | Base configuration file |
| `--sweep_config` | `configs/text_ours_exp.yaml` | Sweep configuration file |
| `--gpus` | `0` | GPU list (comma-separated) |
| `--cpu_cores` | `0-7` | CPU cores to use |
| `--threads` | `8` | Number of threads for BLAS/torch |
| `--mem_soft_mb` | `32768` | Soft memory limit in MB |
| `--timeout` | `14400` | Per-experiment timeout in seconds |
| `--sleep_between` | `60` | Sleep between experiments in seconds |

## Examples

### Basic Experiment
```bash
./submit_exp.sh my_basic_experiment
```

### Custom Output Directory
```bash
./submit_exp.sh my_experiment --outdir results/my_exp
```

### Multiple GPUs with Extended Timeout
```bash
./submit_exp.sh my_experiment --gpus 0,1 --timeout 28800
```

### Custom Configuration Files
```bash
./submit_exp.sh my_experiment \
  --base_config configs/my_base.yaml \
  --sweep_config configs/my_sweep.yaml
```

### High Memory and CPU Usage
```bash
./submit_exp.sh my_experiment \
  --cpu_cores 0-15 \
  --mem_soft_mb 65536 \
  --threads 16
```

### Dry Run
```bash
./submit_exp.sh my_experiment --dry_run
```

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Job Details
```bash
scontrol show job <job_id>
```

### Cancel Job
```bash
scancel <job_id>
```

### View Output
```bash
# Real-time output
tail -f outputs/<job_id>/exp_output.out

# Error output
tail -f outputs/<job_id>/exp_error.err
```

## Output Files

SLURM creates the following output files:
- `outputs/<job_id>/exp_output.out` - Standard output
- `outputs/<job_id>/exp_error.err` - Standard error

The experiment script itself creates additional logs in the output directory specified by `--outdir`.

## Environment Setup

The sbatch script automatically sets up:
- Python virtual environment
- CUDA environment
- Library paths (PyMeshLab, etc.)
- Cache directories
- PyTorch configuration

## Troubleshooting

### Common Issues

1. **Permission denied**: Make sure scripts are executable
   ```bash
   chmod +x scripts/experiments/*.sh
   ```

2. **Job fails immediately**: Check the error output
   ```bash
   cat outputs/<job_id>/exp_error.err
   ```

3. **Out of memory**: Increase memory limit or reduce batch size
   ```bash
   ./submit_exp.sh my_exp --mem_soft_mb 65536
   ```

4. **Timeout**: Increase timeout or optimize your experiment
   ```bash
   ./submit_exp.sh my_exp --timeout 28800
   ```

### Getting Help

- View script help: `./submit_exp.sh --help`
- Check example usage: `./example_usage.sh`
- View SLURM documentation: `man sbatch`

## Integration with Existing Workflow

These scripts are designed to work with the existing experiment infrastructure:
- Uses the same `run_exp_interactive.sh` script
- Compatible with existing configuration files
- Maintains the same output structure
- Supports all existing experiment parameters

The main difference is that experiments now run in a SLURM-managed environment with proper resource allocation and job scheduling.
